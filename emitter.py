"""AST -> Lua source code.

Two output modes:

* ``compact=True``  - one dense line, statements separated by ``;``.
* ``compact=False`` - indented, one statement per line (handy for debugging
  an obfuscation pass).

Both modes produce code that parses back to the same AST.
"""

from __future__ import annotations

import re

from . import ast as A
from .parser import BIN_OPS, UNARY_PREC

_ATOM_PREC = 100
_IDENT_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*\Z")


def quote_bytes(data: bytes) -> str:
    """Render bytes as a Lua short-string literal that is pure ASCII."""
    out = ['"']
    for b in data:
        if b == 34:
            out.append('\\"')
        elif b == 92:
            out.append("\\\\")
        elif b == 10:
            out.append("\\n")
        elif b == 9:
            out.append("\\t")
        elif b == 13:
            out.append("\\r")
        elif 32 <= b < 127:
            out.append(chr(b))
        else:
            out.append("\\%03d" % b)
    out.append('"')
    return "".join(out)


def _prec(node: A.Node) -> int:
    if isinstance(node, A.BinOp):
        return BIN_OPS[node.op][0]
    if isinstance(node, A.UnOp):
        return UNARY_PREC
    if isinstance(node, (A.Function, A.Table)):
        return 0
    return _ATOM_PREC


def _join_bin(left: str, op: str, right: str) -> str:
    # `and` / `or` are words: they need separators on both sides.
    if op in ("and", "or"):
        return f"{left} {op} {right}"
    # `a..b` needs spaces: `1..2` would lex as two numbers.
    if op == "..":
        return f"{left} .. {right}"
    # `a--b` would start a comment.
    if op == "-" and right.startswith("-"):
        return f"{left} - {right}"
    return f"{left}{op}{right}"


class Emitter:
    def __init__(self, compact: bool = True, indent: str = "  "):
        self.compact = compact
        self.indent = indent

    # -- public -----------------------------------------------------------
    def emit(self, block: A.Block) -> str:
        text = self.block(block, 0)
        if self.compact:
            return text
        return text + ("\n" if text else "")

    # -- blocks -----------------------------------------------------------
    def block(self, block: A.Block, level: int) -> str:
        """Render a block.  In pretty mode the text is fully indented."""
        parts = [self.stmt(s, level) for s in block.stmts]
        parts = [p for p in parts if p]
        if self.compact:
            # The trailing `;` keeps the last statement from fusing with a
            # following `end`/`else`/`until` keyword (`return n end`).
            return "".join(p + ";" for p in parts)
        out = []
        for p in parts:
            if p.startswith("("):
                p = ";" + p          # keep it a separate statement
            out.append(self.indent * level + p)
        return "\n".join(out)

    def _body(self, block: A.Block, level: int) -> str:
        """Block text for a construct opened at `level` (closer added by caller)."""
        if self.compact:
            body = self.block(block, level + 1)
            return " " + body + " " if body else " "
        body = self.block(block, level + 1)
        return "\n" + body + "\n" + self.indent * level if body else "\n" + self.indent * level

    # -- statements -------------------------------------------------------
    def stmt(self, node: A.Node, level: int = 0) -> str:
        if isinstance(node, A.LocalAssign):
            names = ", ".join(f"{n} <{a}>" if a else n for n, a in node.names)
            if node.exprs:
                return f"local {names} = {self.exprlist(node.exprs)}"
            return f"local {names}"

        if isinstance(node, A.LocalFunction):
            return f"local function {node.name}{self.funcbody(node.func, level)}"

        if isinstance(node, A.Assign):
            targets = ", ".join(self.expr(t) for t in node.targets)
            return f"{targets} = {self.exprlist(node.exprs)}"

        if isinstance(node, A.CallStmt):
            return self.expr(node.call)

        if isinstance(node, A.Do):
            return f"do{self._body(node.body, level)}end"

        if isinstance(node, A.While):
            return f"while {self.expr(node.cond)} do{self._body(node.body, level)}end"

        if isinstance(node, A.Repeat):
            return f"repeat{self._body(node.body, level)}until {self.expr(node.cond)}"

        if isinstance(node, A.If):
            return self._if(node, level)

        if isinstance(node, A.NumFor):
            head = f"for {node.var} = {self.expr(node.start)}, {self.expr(node.stop)}"
            if node.step is not None:
                head += f", {self.expr(node.step)}"
            return f"{head} do{self._body(node.body, level)}end"

        if isinstance(node, A.GenFor):
            head = f"for {', '.join(node.names)} in {self.exprlist(node.exprs)}"
            return f"{head} do{self._body(node.body, level)}end"

        if isinstance(node, A.FuncDecl):
            return f"function {self.funcname(node)}{self.funcbody(node.func, level)}"

        if isinstance(node, A.Return):
            return f"return {self.exprlist(node.exprs)}" if node.exprs else "return"

        if isinstance(node, A.Break):
            return "break"

        if isinstance(node, A.Goto):
            return f"goto {node.label}"

        if isinstance(node, A.Label):
            return f"::{node.label}::"

        return self.expr(node)   # expression statement (a call)

    def _if(self, node: A.If, level: int) -> str:
        if self.compact:
            out = []
            for i, (cond, body) in enumerate(node.clauses):
                head = "if " if i == 0 else "elseif "
                out.append(f"{head}{self.expr(cond)} then{self._body(body, level)}")
            if node.orelse is not None:
                out.append("else" + self._body(node.orelse, level))
            out.append("end")
            return "".join(out)

        pad = self.indent * level
        out = []
        for i, (cond, body) in enumerate(node.clauses):
            head = "if " if i == 0 else pad + "elseif "
            out.append(f"{head}{self.expr(cond)} then")
            if body.stmts:
                out.append(self.block(body, level + 1))
        if node.orelse is not None:
            out.append(pad + "else")
            if node.orelse.stmts:
                out.append(self.block(node.orelse, level + 1))
        out.append(pad + "end")
        return "\n".join(out)

    def funcname(self, node: A.FuncDecl) -> str:
        parts: list[str] = []
        cur: A.Node = node.target
        while isinstance(cur, A.Index):
            key = cur.key
            if not (isinstance(key, A.FieldKey) and _IDENT_RE.match(key.value)):
                raise ValueError("invalid function name")
            parts.append(key.value.decode())
            cur = cur.obj
        if not isinstance(cur, A.Name):
            raise ValueError("invalid function name")
        parts.reverse()
        base = cur.name
        if node.func.is_method and parts:
            method = parts[-1]
            prefix = ".".join(parts[:-1])
            return f"{base}.{prefix}:{method}" if prefix else f"{base}:{method}"
        return base + "".join("." + p for p in parts)

    def funcbody(self, fn: A.Function, level: int) -> str:
        params = ", ".join(fn.params)
        if fn.is_vararg:
            params = params + ", ..." if params else "..."
        if self.compact:
            return f"({params}){self.block(fn.body, level + 1)}end"
        inner = self.block(fn.body, level + 1)
        if inner:
            return f"({params})\n{inner}\n{self.indent * level}end"
        return f"({params})\n{self.indent * level}end"

    # -- expressions ------------------------------------------------------
    def exprlist(self, exprs: list[A.Node]) -> str:
        return ", ".join(self.expr(e) for e in exprs)

    def expr(self, node: A.Node, min_prec: int = 0) -> str:
        text = self._expr(node)
        if _prec(node) < min_prec:
            return "(" + text + ")"
        return text

    def _expr(self, node: A.Node) -> str:
        if isinstance(node, A.Nil):
            return "nil"
        if isinstance(node, A.TrueLit):
            return "true"
        if isinstance(node, A.FalseLit):
            return "false"
        if isinstance(node, A.Vararg):
            return "..."
        if isinstance(node, A.Number):
            return node.raw
        if isinstance(node, A.String):
            return quote_bytes(node.value)
        if isinstance(node, A.FieldKey):
            return quote_bytes(node.value)
        if isinstance(node, A.Name):
            return node.name
        if isinstance(node, A.Paren):
            return "(" + self.expr(node.expr) + ")"
        if isinstance(node, A.Function):
            return "function" + self.funcbody(node, 0)
        if isinstance(node, A.Table):
            fields = []
            for f in node.fields:
                if f.key is not None:
                    fields.append(f"[{self.expr(f.key)}] = {self.expr(f.value)}")
                elif f.name_key is not None:
                    fields.append(f"{f.name_key} = {self.expr(f.value)}")
                else:
                    fields.append(self.expr(f.value))
            return "{" + ", ".join(fields) + "}"
        if isinstance(node, A.Index):
            return f"{self.prefix(node.obj)}[{self.expr(node.key)}]"
        if isinstance(node, A.Call):
            return f"{self.prefix(node.func)}({self.exprlist(node.args)})"
        if isinstance(node, A.MethodCall):
            return f"{self.prefix(node.obj)}:{node.name}({self.exprlist(node.args)})"
        if isinstance(node, A.UnOp):
            operand = self.expr(node.operand, UNARY_PREC)
            if node.op == "not":
                return "not " + operand
            if node.op == "-" and operand.startswith("-"):
                return "- " + operand
            return node.op + operand
        if isinstance(node, A.BinOp):
            prec, right_assoc = BIN_OPS[node.op]
            left = self.expr(node.left, prec)
            right = self.expr(node.right, prec if right_assoc else prec + 1)
            return _join_bin(left, node.op, right)
        raise TypeError(f"cannot emit node {type(node).__name__}")

    def prefix(self, node: A.Node) -> str:
        """Render a node in a position that requires a prefix expression."""
        if isinstance(node, (A.Name, A.Index, A.Call, A.MethodCall, A.Paren)):
            return self._expr(node)
        return "(" + self.expr(node) + ")"
