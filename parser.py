"""Recursive-descent parser for Lua 5.1 - 5.4.

The parser accepts the union of the 5.1/5.2/5.3/5.4 grammars (goto/labels,
integer division, bitwise operators, `<const>`/`<close>` attributes).  Code
using 5.4-only syntax simply will not run on an older interpreter, exactly as
before obfuscation.
"""

from __future__ import annotations

from . import ast as A
from .lexer import LuaSyntaxError, Token

# Binary operator precedence, mirroring lparser.c.
BIN_OPS: dict[str, tuple[int, bool]] = {
    "or": (1, False),
    "and": (2, False),
    "<": (3, False), ">": (3, False), "<=": (3, False),
    ">=": (3, False), "~=": (3, False), "==": (3, False),
    "|": (4, False),
    "~": (5, False),
    "&": (6, False),
    "<<": (7, False), ">>": (7, False),
    "..": (8, True),
    "+": (9, False), "-": (9, False),
    "*": (10, False), "/": (10, False), "//": (10, False), "%": (10, False),
    "^": (12, True),
}
UNARY_PREC = 11

BLOCK_END = frozenset({"end", "else", "elseif", "until"})


class Parser:
    def __init__(self, tokens: list[Token], filename: str = "<input>"):
        self.toks = tokens
        self.pos = 0
        self.filename = filename

    # -- token helpers ----------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        idx = self.pos + k
        return self.toks[idx] if idx < len(self.toks) else self.toks[-1]

    def at(self, kind: str, text: str | None = None) -> bool:
        tok = self.peek()
        return tok.kind == kind and (text is None or tok.text == text)

    def at_op(self, *ops: str) -> bool:
        tok = self.peek()
        return tok.kind == "op" and tok.text in ops

    def at_kw(self, *kws: str) -> bool:
        tok = self.peek()
        return tok.kind == "keyword" and tok.text in kws

    def take(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def error(self, message: str, tok: Token | None = None):
        tok = tok or self.peek()
        raise LuaSyntaxError(message, tok.line, tok.col, self.filename)

    def expect_op(self, op: str) -> Token:
        if not self.at_op(op):
            self.error(f"'{op}' expected near {self.peek().text or '<eof>'!r}")
        return self.take()

    def expect_kw(self, kw: str) -> Token:
        if not self.at_kw(kw):
            self.error(f"'{kw}' expected near {self.peek().text or '<eof>'!r}")
        return self.take()

    def expect_name(self) -> str:
        if not self.at("name"):
            self.error(f"<name> expected near {self.peek().text or '<eof>'!r}")
        return self.take().text

    # -- entry point ------------------------------------------------------
    def parse_chunk(self) -> A.Block:
        block = self.parse_block()
        if not self.at("eof"):
            self.error(f"unexpected symbol near {self.peek().text!r}")
        return block

    def parse_block(self) -> A.Block:
        stmts: list[A.Node] = []
        while True:
            tok = self.peek()
            if tok.kind == "eof":
                break
            if tok.kind == "keyword" and tok.text in BLOCK_END:
                break
            if tok.kind == "keyword" and tok.text == "return":
                stmts.append(self.parse_return())
                break
            stmt = self.parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return A.Block(stmts)

    # -- statements -------------------------------------------------------
    def parse_statement(self) -> A.Node | None:
        tok = self.peek()

        if tok.kind == "op":
            if tok.text == ";":
                self.take()
                return None
            if tok.text == "::":
                self.take()
                name = self.expect_name()
                self.expect_op("::")
                return A.Label(name)

        if tok.kind == "keyword":
            kw = tok.text
            if kw == "break":
                self.take()
                return A.Break()
            if kw == "do":
                self.take()
                body = self.parse_block()
                self.expect_kw("end")
                return A.Do(body)
            if kw == "while":
                self.take()
                cond = self.parse_expr()
                self.expect_kw("do")
                body = self.parse_block()
                self.expect_kw("end")
                return A.While(cond, body)
            if kw == "repeat":
                self.take()
                body = self.parse_block()
                self.expect_kw("until")
                cond = self.parse_expr()
                return A.Repeat(body, cond)
            if kw == "if":
                return self.parse_if()
            if kw == "for":
                return self.parse_for()
            if kw == "function":
                return self.parse_function_decl()
            if kw == "local":
                return self.parse_local()

        if tok.kind == "name" and tok.text == "goto" and self.peek(1).kind == "name":
            self.take()
            return A.Goto(self.take().text)

        return self.parse_expr_statement()

    def parse_if(self) -> A.If:
        self.expect_kw("if")
        clauses = []
        cond = self.parse_expr()
        self.expect_kw("then")
        clauses.append((cond, self.parse_block()))
        while self.at_kw("elseif"):
            self.take()
            cond = self.parse_expr()
            self.expect_kw("then")
            clauses.append((cond, self.parse_block()))
        orelse = None
        if self.at_kw("else"):
            self.take()
            orelse = self.parse_block()
        self.expect_kw("end")
        return A.If(clauses, orelse)

    def parse_for(self) -> A.Node:
        self.expect_kw("for")
        first = self.expect_name()
        if self.at_op("="):
            self.take()
            start = self.parse_expr()
            self.expect_op(",")
            stop = self.parse_expr()
            step = None
            if self.at_op(","):
                self.take()
                step = self.parse_expr()
            self.expect_kw("do")
            body = self.parse_block()
            self.expect_kw("end")
            return A.NumFor(first, start, stop, step, body)
        names = [first]
        while self.at_op(","):
            self.take()
            names.append(self.expect_name())
        self.expect_kw("in")
        exprs = self.parse_exprlist()
        self.expect_kw("do")
        body = self.parse_block()
        self.expect_kw("end")
        return A.GenFor(names, exprs, body)

    def parse_function_decl(self) -> A.FuncDecl:
        self.expect_kw("function")
        target: A.Node = A.Name(self.expect_name())
        is_method = False
        while True:
            if self.at_op("."):
                self.take()
                field = A.FieldKey(self.expect_name().encode())
                target = A.Index(target, field)
            elif self.at_op(":"):
                self.take()
                field = A.FieldKey(self.expect_name().encode())
                target = A.Index(target, field)
                is_method = True
                break
            else:
                break
        func = self.parse_funcbody(is_method)
        return A.FuncDecl(target, func)

    def parse_local(self) -> A.Node:
        self.expect_kw("local")
        if self.at_kw("function"):
            self.take()
            name = self.expect_name()
            func = self.parse_funcbody(False)
            return A.LocalFunction(name, func)
        names: list[tuple[str, str | None]] = []
        while True:
            name = self.expect_name()
            attrib = None
            if self.at_op("<") and self.peek(1).kind == "name" \
                    and self.peek(1).text in ("const", "close") \
                    and self.peek(2).kind == "op" and self.peek(2).text == ">":
                self.take()
                attrib = self.take().text
                self.expect_op(">")
            names.append((name, attrib))
            if not self.at_op(","):
                break
            self.take()
        exprs: list[A.Node] = []
        if self.at_op("="):
            self.take()
            exprs = self.parse_exprlist()
        return A.LocalAssign(names, exprs)

    def parse_return(self) -> A.Return:
        self.expect_kw("return")
        exprs: list[A.Node] = []
        if not (self.at("eof") or (self.peek().kind == "keyword" and self.peek().text in BLOCK_END)
                or self.at_op(";")):
            exprs = self.parse_exprlist()
        if self.at_op(";"):
            self.take()
        return A.Return(exprs)

    def parse_expr_statement(self) -> A.Node:
        start = self.peek()
        first = self.parse_suffixed_expr()
        if self.at_op("=") or self.at_op(","):
            targets = [first]
            while self.at_op(","):
                self.take()
                targets.append(self.parse_suffixed_expr())
            self.expect_op("=")
            exprs = self.parse_exprlist()
            for t in targets:
                if not isinstance(t, (A.Name, A.Index)):
                    raise LuaSyntaxError("syntax error: cannot assign to this expression",
                                         start.line, start.col, self.filename)
            return A.Assign(targets, exprs)
        if not isinstance(first, (A.Call, A.MethodCall)):
            self.error("syntax error: expression is not a statement", start)
        return A.CallStmt(first)

    # -- expressions ------------------------------------------------------
    def parse_exprlist(self) -> list[A.Node]:
        exprs = [self.parse_expr()]
        while self.at_op(","):
            self.take()
            exprs.append(self.parse_expr())
        return exprs

    def parse_expr(self, min_prec: int = 1) -> A.Node:
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok.kind == "op":
                op = tok.text
            elif tok.kind == "keyword" and tok.text in ("and", "or"):
                op = tok.text
            else:
                break
            info = BIN_OPS.get(op)
            if info is None:
                break
            prec, right_assoc = info
            if prec < min_prec:
                break
            self.take()
            right = self.parse_expr(prec if right_assoc else prec + 1)
            left = A.BinOp(op, left, right)
        return left

    def parse_unary(self) -> A.Node:
        tok = self.peek()
        if tok.kind == "op" and tok.text in ("-", "#", "~"):
            self.take()
            return A.UnOp(tok.text, self.parse_expr(UNARY_PREC))
        if tok.kind == "keyword" and tok.text == "not":
            self.take()
            return A.UnOp("not", self.parse_expr(UNARY_PREC))
        return self.parse_simple_expr()

    def parse_simple_expr(self) -> A.Node:
        tok = self.peek()
        if tok.kind == "number":
            self.take()
            return A.Number(tok.text)
        if tok.kind == "string":
            self.take()
            return A.String(tok.value or b"", tok.text)
        if tok.kind == "keyword":
            kw = tok.text
            if kw == "nil":
                self.take()
                return A.Nil()
            if kw == "true":
                self.take()
                return A.TrueLit()
            if kw == "false":
                self.take()
                return A.FalseLit()
            if kw == "function":
                self.take()
                return self.parse_funcbody(False)
        if tok.kind == "op":
            if tok.text == "...":
                self.take()
                return A.Vararg()
            if tok.text == "{":
                return self.parse_table()
        if tok.kind == "name" or self.at_op("("):
            return self.parse_suffixed_expr()
        self.error(f"unexpected symbol near {tok.text or '<eof>'!r}")

    def parse_primary_expr(self) -> A.Node:
        if self.at("name"):
            return A.Name(self.take().text)
        if self.at_op("("):
            self.take()
            expr = self.parse_expr()
            self.expect_op(")")
            return A.Paren(expr)
        self.error(f"unexpected symbol near {self.peek().text or '<eof>'!r}")

    def parse_suffixed_expr(self) -> A.Node:
        expr = self.parse_primary_expr()
        while True:
            if self.at_op("."):
                self.take()
                expr = A.Index(expr, A.FieldKey(self.expect_name().encode()))
            elif self.at_op("["):
                self.take()
                key = self.parse_expr()
                self.expect_op("]")
                expr = A.Index(expr, key)
            elif self.at_op(":"):
                self.take()
                name = self.expect_name()
                args = self.parse_args()
                expr = A.MethodCall(expr, name, args)
            elif self.at_op("(", "{") or self.at("string"):
                args = self.parse_args()
                expr = A.Call(expr, args)
            else:
                return expr

    def parse_args(self) -> list[A.Node]:
        if self.at("string"):
            tok = self.take()
            return [A.String(tok.value or b"", tok.text)]
        if self.at_op("{"):
            return [self.parse_table()]
        if self.at_op("("):
            self.take()
            if self.at_op(")"):
                self.take()
                return []
            args = self.parse_exprlist()
            self.expect_op(")")
            return args
        self.error(f"function arguments expected near {self.peek().text or '<eof>'!r}")

    def parse_table(self) -> A.Table:
        self.expect_op("{")
        fields: list[A.FieldNode] = []
        while not self.at_op("}"):
            if self.at_op("["):
                self.take()
                key = self.parse_expr()
                self.expect_op("]")
                self.expect_op("=")
                fields.append(A.FieldNode(self.parse_expr(), key=key))
            elif self.at("name") and self.peek(1).kind == "op" and self.peek(1).text == "=":
                name = self.take().text
                self.take()  # '='
                fields.append(A.FieldNode(self.parse_expr(), name_key=name))
            else:
                fields.append(A.FieldNode(self.parse_expr()))
            if self.at_op(",") or self.at_op(";"):
                self.take()
            else:
                break
        self.expect_op("}")
        return A.Table(fields)

    def parse_funcbody(self, is_method: bool) -> A.Function:
        self.expect_op("(")
        params: list[str] = []
        is_vararg = False
        if not self.at_op(")"):
            while True:
                if self.at_op("..."):
                    self.take()
                    is_vararg = True
                    break
                params.append(self.expect_name())
                if not self.at_op(","):
                    break
                self.take()
        self.expect_op(")")
        body = self.parse_block()
        self.expect_kw("end")
        return A.Function(params, is_vararg, body, is_method)


def parse(src: str, filename: str = "<input>") -> A.Block:
    from .lexer import tokenize
    return Parser(tokenize(src, filename), filename).parse_chunk()
