"""AST node definitions for Lua 5.1 - 5.4."""

from __future__ import annotations

from dataclasses import dataclass, field


class Node:
    """Base class for every AST node."""

    __slots__ = ()


# --------------------------------------------------------------------------
# Blocks and statements
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Block(Node):
    stmts: list


@dataclass(slots=True)
class LocalAssign(Node):
    """local a, b <const> = 1, 2"""
    names: list[tuple[str, str | None]]
    exprs: list


@dataclass(slots=True)
class LocalFunction(Node):
    name: str
    func: "Function"


@dataclass(slots=True)
class Assign(Node):
    targets: list
    exprs: list


@dataclass(slots=True)
class CallStmt(Node):
    call: Node


@dataclass(slots=True)
class Do(Node):
    body: Block


@dataclass(slots=True)
class While(Node):
    cond: Node
    body: Block


@dataclass(slots=True)
class Repeat(Node):
    body: Block
    cond: Node


@dataclass(slots=True)
class If(Node):
    clauses: list[tuple[Node, Block]]
    orelse: Block | None


@dataclass(slots=True)
class NumFor(Node):
    var: str
    start: Node
    stop: Node
    step: Node | None
    body: Block


@dataclass(slots=True)
class GenFor(Node):
    names: list[str]
    exprs: list
    body: Block


@dataclass(slots=True)
class FuncDecl(Node):
    """function a.b.c:m() ... end  -- `target` is a Name/Index chain."""
    target: Node
    func: "Function"


@dataclass(slots=True)
class Return(Node):
    exprs: list


@dataclass(slots=True)
class Break(Node):
    pass


@dataclass(slots=True)
class Goto(Node):
    label: str


@dataclass(slots=True)
class Label(Node):
    label: str


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Nil(Node):
    pass


@dataclass(slots=True)
class TrueLit(Node):
    pass


@dataclass(slots=True)
class FalseLit(Node):
    pass


@dataclass(slots=True)
class Vararg(Node):
    pass


@dataclass(slots=True)
class Number(Node):
    raw: str


@dataclass(slots=True)
class String(Node):
    value: bytes
    raw: str = ""


@dataclass(slots=True)
class FieldKey(Node):
    """A field name written as `obj.name` / `function obj:name()`.

    Kept distinct from `String` so string obfuscation never turns a field
    name into a runtime expression.
    """
    value: bytes


@dataclass(slots=True)
class Name(Node):
    name: str


@dataclass(slots=True)
class Index(Node):
    obj: Node
    key: Node


@dataclass(slots=True)
class Call(Node):
    func: Node
    args: list


@dataclass(slots=True)
class MethodCall(Node):
    obj: Node
    name: str
    args: list


@dataclass(slots=True)
class Function(Node):
    params: list[str]
    is_vararg: bool
    body: Block
    is_method: bool = False


@dataclass(slots=True)
class FieldNode(Node):
    """One entry in a table constructor.

    Exactly one of `key` / `name_key` is set for `[k]=v` / `k=v`; both are
    None for a plain positional value.
    """
    value: Node
    key: Node | None = None
    name_key: str | None = None


@dataclass(slots=True)
class Table(Node):
    fields: list[FieldNode] = field(default_factory=list)


@dataclass(slots=True)
class BinOp(Node):
    op: str
    left: Node
    right: Node


@dataclass(slots=True)
class UnOp(Node):
    op: str
    operand: Node


@dataclass(slots=True)
class Paren(Node):
    expr: Node
