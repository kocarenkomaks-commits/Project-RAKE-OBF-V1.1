"""Obfuscation passes (RAKE OBF V1.1).

Pipeline:

1. collect every identifier in the chunk (fresh names must not collide);
2. rename locals/parameters/labels (scope aware, globals and fields untouched);
3. replace string literals with calls to a generated per-string decoder;
4. replace booleans with arithmetic comparisons;
5. rewrite integer constants as larger arithmetic expressions;
6. prepend the decoder helper (added last so it is not itself renamed);
7. optionally wrap the whole result in an encrypted loader;
8. add decoy blocks so the output reaches the requested size ratio.
"""

from __future__ import annotations

import base64 as _base64
import math
import random
import string
from dataclasses import dataclass, fields, replace

from . import ast as A
from .emitter import Emitter, quote_bytes
from .lexer import KEYWORDS
from .parser import parse

BANNER = "---- RAKE OBF V1.1 ----"
DEFAULT_SIZE_RATIO = 4.0

RESERVED = set(KEYWORDS) | {"self", "_ENV", "_G", "..."}

# Names used literally by the generated helpers.  Fresh names must never
# collide with them or a helper would shadow its own library reference.
HELPER_NAMES = frozenset("""
v t n k i e s m acc out bits mult buf plain b64 map tbl load loadstring
error string table math char sub byte concat floor type select
""".split())


# --------------------------------------------------------------------------
# generic tree helpers
# --------------------------------------------------------------------------

def walk(node):
    """Yield every AST node in the tree (pre-order)."""
    if isinstance(node, list):
        for item in node:
            yield from walk(item)
        return
    if isinstance(node, tuple):
        for item in node:
            yield from walk(item)
        return
    if not isinstance(node, A.Node):
        return
    yield node
    for f in fields(node):
        yield from walk(getattr(node, f.name))


def transform(node, fn):
    """Bottom-up rebuild of the tree, applying `fn` to every node."""
    if isinstance(node, list):
        return [transform(x, fn) for x in node]
    if isinstance(node, tuple):
        return tuple(transform(x, fn) for x in node)
    if not isinstance(node, A.Node):
        return node
    changes = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, (A.Node, list, tuple)):
            new = transform(value, fn)
            if new is not value:
                changes[f.name] = new
    if changes:
        node = replace(node, **changes)
    return fn(node)


def collect_identifiers(block: A.Block) -> set[str]:
    """Every name that exists anywhere in the source, including declarations."""
    names: set[str] = set()
    for node in walk(block):
        if isinstance(node, A.Name):
            names.add(node.name)
        elif isinstance(node, A.LocalAssign):
            names.update(n for n, _ in node.names)
        elif isinstance(node, A.LocalFunction):
            names.add(node.name)
        elif isinstance(node, A.Function):
            names.update(node.params)
        elif isinstance(node, A.NumFor):
            names.add(node.var)
        elif isinstance(node, A.GenFor):
            names.update(node.names)
        elif isinstance(node, (A.Goto, A.Label)):
            names.add(node.label)
        elif isinstance(node, A.FieldNode) and node.name_key:
            names.add(node.name_key)
    return names


class NameGenerator:
    """Hands out fresh, globally unique, short identifiers.

    Candidate names are visited through a random permutation of each length
    (1 char, then 2 chars, ...).  Unlike repeated random probing, this can
    never spin forever when the source already uses every short name.
    """

    _ALPHA = string.ascii_lowercase + string.ascii_uppercase
    _ALNUM = _ALPHA + string.digits + "_"

    def __init__(self, taken: set[str], seed: int | None = None):
        self.taken = set(taken) | RESERVED | HELPER_NAMES
        self.rng = random.Random(seed)
        self.count = 0
        self._length = 0
        self._pos = 0
        self._start = 0
        self._step = 1
        self._total = 1
        self._init_length(1)

    @classmethod
    def _total_for(cls, length: int) -> int:
        return len(cls._ALPHA) * (len(cls._ALNUM) ** (length - 1))

    @classmethod
    def _name_at(cls, index: int, length: int) -> str:
        first = cls._ALPHA[index % len(cls._ALPHA)]
        index //= len(cls._ALPHA)
        rest = []
        for _ in range(length - 1):
            rest.append(cls._ALNUM[index % len(cls._ALNUM)])
            index //= len(cls._ALNUM)
        return first + "".join(rest)

    def _init_length(self, length: int) -> None:
        total = self._total_for(length)
        start = self.rng.randrange(total)
        while True:
            step = self.rng.randrange(1, total)
            if math.gcd(step, total) == 1:
                break
        self._length = length
        self._pos = 0
        self._start = start
        self._step = step
        self._total = total

    def fresh(self) -> str:
        while self._length <= 8:
            if self._pos >= self._total:
                self._init_length(self._length + 1)
                continue
            index = (self._start + self._pos * self._step) % self._total
            self._pos += 1
            name = self._name_at(index, self._length)
            if name not in self.taken:
                self.taken.add(name)
                self.count += 1
                return name
        raise RuntimeError("out of identifier names")


# --------------------------------------------------------------------------
# 1. scope-aware renaming
# --------------------------------------------------------------------------

class Scope:
    __slots__ = ("parent", "names")

    def __init__(self, parent: "Scope | None" = None):
        self.parent = parent
        self.names: dict[str, str] = {}

    def lookup(self, name: str) -> str | None:
        scope: Scope | None = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None

    def declare(self, original: str, mangled: str) -> None:
        self.names[original] = mangled


class Renamer:
    """Rename locals, parameters, loop variables and labels."""

    def __init__(self, gen: NameGenerator):
        self.gen = gen
        self.renamed = 0
        self.labels = 0

    def apply(self, block: A.Block) -> None:
        self.visit_block(block, Scope(), new_scope=False)
        self.rename_labels(block)

    def rename_labels(self, block: A.Block) -> None:
        names = {node.label for node in walk(block)
                 if isinstance(node, (A.Goto, A.Label))}
        if not names:
            return
        mapping = {name: self.gen.fresh() for name in names}
        for node in walk(block):
            if isinstance(node, (A.Goto, A.Label)):
                node.label = mapping[node.label]
        self.labels = len(mapping)

    # -- scope handling ---------------------------------------------------
    def visit_block(self, block: A.Block, scope: Scope, new_scope: bool = True) -> None:
        if new_scope:
            scope = Scope(scope)
        for stmt in block.stmts:
            self.visit_stmt(stmt, scope)

    def visit_function(self, fn: A.Function, scope: Scope) -> None:
        inner = Scope(scope)
        for i, param in enumerate(fn.params):
            mangled = self.gen.fresh()
            inner.declare(param, mangled)
            fn.params[i] = mangled
        self.visit_block(fn.body, inner)

    # -- statements -------------------------------------------------------
    def visit_stmt(self, node: A.Node, scope: Scope) -> None:
        if isinstance(node, A.LocalAssign):
            node.exprs = [self.visit_expr(e, scope) for e in node.exprs]
            for i, (name, attrib) in enumerate(node.names):
                if name == "_ENV":
                    scope.declare(name, name)
                    continue
                mangled = self.gen.fresh()
                scope.declare(name, mangled)
                node.names[i] = (mangled, attrib)
                self.renamed += 1

        elif isinstance(node, A.LocalFunction):
            mangled = self.gen.fresh()
            scope.declare(node.name, mangled)
            node.name = mangled
            self.renamed += 1
            self.visit_function(node.func, scope)

        elif isinstance(node, A.Assign):
            node.targets = [self.visit_expr(t, scope) for t in node.targets]
            node.exprs = [self.visit_expr(e, scope) for e in node.exprs]

        elif isinstance(node, A.CallStmt):
            node.call = self.visit_expr(node.call, scope)

        elif isinstance(node, A.Do):
            self.visit_block(node.body, scope)

        elif isinstance(node, A.While):
            node.cond = self.visit_expr(node.cond, scope)
            self.visit_block(node.body, scope)

        elif isinstance(node, A.Repeat):
            inner = Scope(scope)
            self.visit_block(node.body, inner, new_scope=False)
            node.cond = self.visit_expr(node.cond, inner)

        elif isinstance(node, A.If):
            node.clauses = [
                (self.visit_expr(cond, scope), self._visit_body(body, scope))
                for cond, body in node.clauses
            ]
            if node.orelse is not None:
                node.orelse = self._visit_body(node.orelse, scope)

        elif isinstance(node, A.NumFor):
            node.start = self.visit_expr(node.start, scope)
            node.stop = self.visit_expr(node.stop, scope)
            if node.step is not None:
                node.step = self.visit_expr(node.step, scope)
            inner = Scope(scope)
            mangled = self.gen.fresh()
            inner.declare(node.var, mangled)
            node.var = mangled
            self.renamed += 1
            self.visit_block(node.body, inner)

        elif isinstance(node, A.GenFor):
            node.exprs = [self.visit_expr(e, scope) for e in node.exprs]
            inner = Scope(scope)
            for i, name in enumerate(node.names):
                mangled = self.gen.fresh()
                inner.declare(name, mangled)
                node.names[i] = mangled
                self.renamed += 1
            self.visit_block(node.body, inner)

        elif isinstance(node, A.FuncDecl):
            node.target = self.visit_expr(node.target, scope)
            self.visit_function(node.func, scope)

        elif isinstance(node, A.Return):
            node.exprs = [self.visit_expr(e, scope) for e in node.exprs]

    def _visit_body(self, block: A.Block, scope: Scope) -> A.Block:
        self.visit_block(block, scope)
        return block

    # -- expressions ------------------------------------------------------
    def visit_expr(self, node: A.Node, scope: Scope) -> A.Node:
        if isinstance(node, A.Name):
            mangled = scope.lookup(node.name)
            if mangled is not None and mangled != node.name:
                return A.Name(mangled)
            return node

        if isinstance(node, A.Index):
            node.obj = self.visit_expr(node.obj, scope)
            node.key = self.visit_expr(node.key, scope)
            return node

        if isinstance(node, A.Call):
            node.func = self.visit_expr(node.func, scope)
            node.args = [self.visit_expr(a, scope) for a in node.args]
            return node

        if isinstance(node, A.MethodCall):
            node.obj = self.visit_expr(node.obj, scope)
            node.args = [self.visit_expr(a, scope) for a in node.args]
            return node

        if isinstance(node, A.Function):
            self.visit_function(node, scope)
            return node

        if isinstance(node, A.Table):
            for f in node.fields:
                if f.key is not None:
                    f.key = self.visit_expr(f.key, scope)
                f.value = self.visit_expr(f.value, scope)
            return node

        if isinstance(node, A.BinOp):
            node.left = self.visit_expr(node.left, scope)
            node.right = self.visit_expr(node.right, scope)
            return node

        if isinstance(node, A.UnOp):
            node.operand = self.visit_expr(node.operand, scope)
            return node

        if isinstance(node, A.Paren):
            node.expr = self.visit_expr(node.expr, scope)
            return node

        return node


# --------------------------------------------------------------------------
# 2. string encryption
# --------------------------------------------------------------------------

class StringEncryptor:
    """Per-string encrypted payloads with a small stream cipher.

    Every payload starts with:

        skip byte | skip random bytes | salt | seed_hi | seed_lo | cipher

    The decoder extracts the per-string key, rebuilds the stream and caches
    the result.  Long strings are split into several independently encrypted
    pieces and concatenated at runtime.
    """

    def __init__(self, gen: NameGenerator, rng: random.Random):
        self.gen = gen
        self.rng = rng
        self.decoder = gen.fresh()
        self.cache = gen.fresh()
        self.param = gen.fresh()
        self.roles = {name: gen.fresh() for name in (
            "v", "n", "skip", "salt", "state", "out", "i", "stream")}
        self._payloads: dict[bytes, bytes] = {}
        self.count = 0

    def _payload(self, data: bytes) -> bytes:
        cached = self._payloads.get(data)
        if cached is not None:
            return cached
        rng = self.rng
        salt = rng.randrange(256)
        seed = rng.randrange(65536)
        skip = rng.randrange(8)
        header = bytes([skip]) + bytes(rng.randrange(256) for _ in range(skip))
        header += bytes([salt, seed >> 8, seed & 0xFF])
        state = seed
        out = bytearray(header)
        for idx, byte in enumerate(data, 1):
            state = (state * 25173 + 13849) % 65536
            stream = ((state >> 8) + salt + idx) % 256
            out.append((byte - stream) % 256)
        payload = bytes(out)
        self._payloads[data] = payload
        return payload

    def make_decoder(self) -> list[A.Node]:
        r = self.roles
        template = """
local @CACHE@ = {}
local function @DEC@(@S@)
  local @V@ = @CACHE@[@S@]
  if @V@ then return @V@ end
  local @N@ = #@S@
  local @SKIP@ = @S@:byte(1)
  local @SALT@ = @S@:byte(2+@SKIP@)
  local @STATE@ = @S@:byte(3+@SKIP@)*256+@S@:byte(4+@SKIP@)
  local @OUT@ = {}
  for @I@ = 1, @N@-(4+@SKIP@) do
    @STATE@ = (@STATE@*25173+13849)%65536
    local @STREAM@ = (math.floor(@STATE@/256)+@SALT@+@I@)%256
    @OUT@[@I@] = string.char((@S@:byte(@I@+4+@SKIP@)+@STREAM@)%256)
  end
  @V@ = table.concat(@OUT@)
  @CACHE@[@S@] = @V@
  return @V@
end
""".replace("@CACHE@", self.cache) \
   .replace("@DEC@", self.decoder) \
   .replace("@S@", self.param) \
   .replace("@V@", r["v"]) \
   .replace("@N@", r["n"]) \
   .replace("@SKIP@", r["skip"]) \
   .replace("@SALT@", r["salt"]) \
   .replace("@STATE@", r["state"]) \
   .replace("@OUT@", r["out"]) \
   .replace("@I@", r["i"]) \
   .replace("@STREAM@", r["stream"])
        return parse(template, "<decoder>").stmts

    def _call(self, data: bytes) -> A.Node:
        return A.Call(A.Name(self.decoder),
                      [A.String(self._payload(data), "")])

    def _encode_string(self, data: bytes) -> A.Node:
        if len(data) >= 48 and self.rng.random() < 0.65:
            parts = 3 if len(data) >= 256 else 2
            cuts = sorted(self.rng.sample(range(4, len(data) - 3), parts - 1))
            chunks = [data[a:b] for a, b in
                      zip([0] + cuts, cuts + [len(data)])]
        else:
            chunks = [data]
        node = self._call(chunks[0])
        for chunk in chunks[1:]:
            node = A.BinOp("..", node, self._call(chunk))
        return node

    def apply(self, block: A.Block) -> int:
        def visit(node: A.Node) -> A.Node:
            if isinstance(node, A.String):
                self.count += 1
                return self._encode_string(node.value)
            return node

        new_block = transform(block, visit)
        block.stmts = new_block.stmts
        return self.count


# --------------------------------------------------------------------------
# 3. boolean / number obfuscation
# --------------------------------------------------------------------------

class BoolObfuscator:
    def __init__(self):
        self.count = 0

    def apply(self, block: A.Block) -> int:
        def visit(node: A.Node) -> A.Node:
            if isinstance(node, A.TrueLit):
                self.count += 1
                return A.BinOp("==", A.Number("1"), A.Number("1"))
            if isinstance(node, A.FalseLit):
                self.count += 1
                return A.BinOp("~=", A.Number("1"), A.Number("1"))
            return node

        new_block = transform(block, visit)
        block.stmts = new_block.stmts
        return self.count


class NumberObfuscator:
    """Turn integers into larger, equivalent arithmetic expressions."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.count = 0

    @staticmethod
    def _parse(raw: str) -> int | None:
        if raw[:2].lower() == "0x":
            if "." in raw or "p" in raw.lower():
                return None
            try:
                return int(raw, 16)
            except ValueError:
                return None
        if not raw.isdigit():
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _encode(self, value: int) -> A.Node:
        rng = self.rng
        style = rng.randrange(5)
        if style == 0:
            a = rng.randrange(1, 1000)
            return A.BinOp("-", A.BinOp("+", A.Number(str(value)),
                                        A.Number(str(a))), A.Number(str(a)))
        if style == 1:
            a = rng.randrange(1, 100000)
            return A.BinOp("+", A.BinOp("-", A.Number(str(value)),
                                        A.Number(str(a))), A.Number(str(a)))
        if style == 2:
            a = rng.randrange(1, 500)
            b = rng.randrange(1, 500)
            return A.BinOp(
                "-",
                A.BinOp("+", A.BinOp("+", A.Number(str(value)),
                                     A.Number(str(a))), A.Number(str(b))),
                A.BinOp("+", A.Number(str(a)), A.Number(str(b))))
        if style == 3:
            c = rng.randrange(2, 10)
            d = rng.randrange(0, 1000)
            return A.BinOp(
                "-",
                A.BinOp("+", A.BinOp("*", A.Number(str(value)),
                                     A.Number(str(c))), A.Number(str(d))),
                A.BinOp("+", A.BinOp("*", A.Number(str(value)),
                                     A.Number(str(c - 1))), A.Number(str(d))))
        c = rng.randrange(2, 12)
        return A.BinOp(
            "-",
            A.BinOp("*", A.Number(str(value)), A.Number(str(c))),
            A.BinOp("*", A.Number(str(value)), A.Number(str(c - 1))))

    def apply(self, block: A.Block) -> int:
        def visit(node: A.Node) -> A.Node:
            if not isinstance(node, A.Number):
                return node
            value = self._parse(node.raw)
            if value is None or not (0 <= value <= 0x7FFFFFFF):
                return node
            self.count += 1
            return self._encode(value)

        new_block = transform(block, visit)
        block.stmts = new_block.stmts
        return self.count


# --------------------------------------------------------------------------
# 4. packing (whole-chunk encryption + loader)
# --------------------------------------------------------------------------

B64_ALPHABET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/")


def b64encode(data: bytes) -> str:
    return _base64.b64encode(data).decode("ascii")


def pack_source(source: str, gen: NameGenerator, rng: random.Random,
                chunkname: str | None = None) -> str:
    key = bytes(rng.randrange(256) for _ in range(rng.randint(16, 32)))
    salt = rng.randrange(256)
    seed = rng.randrange(65536)
    data = source.encode("utf-8", "surrogateescape")

    state = seed
    cipher = bytearray()
    for idx, byte in enumerate(data, 1):
        state = (state * 25173 + 13849) % 65536
        stream = ((state >> 8) + key[(idx - 1) % len(key)] + salt + idx) % 256
        cipher.append((byte - stream) % 256)
    payload = b64encode(bytes(cipher))

    dec = gen.fresh()
    b64 = gen.fresh()
    table_ = gen.fresh()
    out = gen.fresh()
    acc = gen.fresh()
    bits = gen.fresh()
    val = gen.fresh()
    mult = gen.fresh()
    plain = gen.fresh()
    buf = gen.fresh()
    kvar = gen.fresh()
    saltvar = gen.fresh()
    statevar = gen.fresh()
    loader = gen.fresh()
    func = gen.fresh()
    err = gen.fresh()

    if chunkname is None:
        chunkname = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(8, 14)))
    name_lit = quote_bytes(b"=" + chunkname.encode("utf-8", "surrogateescape"))

    return f"""local {dec} = function(e)
  local {b64} = "{B64_ALPHABET}"
  local {table_} = {{}}
  for i = 1, #{b64} do {table_}[{b64}:sub(i, i)] = i - 1 end
  local {out}, {acc}, {bits} = {{}}, 0, 0
  for i = 1, #e do
    local {val} = {table_}[e:sub(i, i)]
    if {val} then
      {acc} = {acc} * 64 + {val}
      {bits} = {bits} + 6
      if {bits} >= 8 then
        {bits} = {bits} - 8
        local {mult} = 2 ^ {bits}
        {out}[#{out} + 1] = string.char(math.floor({acc} / {mult}) % 256)
        {acc} = {acc} % {mult}
      end
    end
  end
  local {plain} = table.concat({out})
  local {buf}, {kvar}, {saltvar}, {statevar} = {{}}, {quote_bytes(key)}, {salt}, {seed}
  for i = 1, #{plain} do
    {statevar} = ({statevar} * 25173 + 13849) % 65536
    {buf}[i] = string.char(({plain}:byte(i) + math.floor({statevar} / 256) + {kvar}:byte((i - 1) % #{kvar} + 1) + {saltvar} + i) % 256)
  end
  return table.concat({buf})
end
local {loader} = loadstring or load
local {func}, {err} = {loader}({dec}({quote_bytes(payload.encode("ascii"))}), {name_lit}, "t", _ENV)
if not {func} then error({err}) end
return {func}(...)
"""


# --------------------------------------------------------------------------
# 5. decoy padding
# --------------------------------------------------------------------------

PAD_ALPHABET = (string.ascii_letters + string.digits
                + "+/=-.,;:!?@#$%^&*()_ ")
PAD_OPEN = "[=======["
PAD_CLOSE = "]=======]"
_PAD_PREFIX = "do local q=1;if q*q<0 then local s=" + PAD_OPEN
_PAD_SUFFIX = PAD_CLOSE + " end end\n"
PAD_OVERHEAD = len(_PAD_PREFIX) + len(_PAD_SUFFIX)
BLOB_TARGET = 300_000


def _material(rng: random.Random, size: int) -> str:
    """Exactly `size` printable bytes, broken into readable lines."""
    if size <= 0:
        return ""
    chars = rng.choices(PAD_ALPHABET, k=size)
    for idx in range(119, size, 120):
        chars[idx] = "\n"
    return "".join(chars)


def _comment(size: int, rng: random.Random) -> str:
    """A valid Lua long comment of exactly `size` bytes (or whitespace)."""
    if size <= 0:
        return ""
    opening = "--[===["
    closing = "]===]"
    if size < len(opening) + len(closing):
        return " " * size if size > 1 else "\n"
    body = _material(rng, size - len(opening) - len(closing))
    return opening + body + closing


def make_padding(size: int, rng: random.Random) -> str:
    """Return syntactically valid padding of exactly `size` bytes.

    Most of the payload is a dead branch containing huge long strings.  It
    costs parser time and file size, but almost no bytecode and no runtime
    work, so even LuaJIT's per-function instruction limit is not an issue.
    """
    if size <= 0:
        return ""
    if size < PAD_OVERHEAD + 32:
        return _comment(size, rng)

    count = max(1, (size + BLOB_TARGET - 1) // BLOB_TARGET)
    while count > 1 and size - count * PAD_OVERHEAD < 16:
        count -= 1
    materials = size - count * PAD_OVERHEAD
    if materials < 0:
        return _comment(size, rng)

    base, extra = divmod(materials, count)
    chunks = []
    for idx in range(count):
        length = base + (1 if idx < extra else 0)
        chunks.append(_PAD_PREFIX + _material(rng, length) + _PAD_SUFFIX)
    out = "".join(chunks)
    if len(out) < size:
        out += _comment(size - len(out), rng)
    return out[:size]


def make_filler(size: int, rng: random.Random) -> str:
    """End-of-file filler that never comments out preceding code."""
    if size <= 0:
        return ""
    if size == 1:
        return "\n"
    return "\n" + _comment(size - 1, rng)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

@dataclass
class Options:
    rename: bool = True
    strings: bool = True
    numbers: bool = True
    bools: bool = True
    pack: bool = False
    pretty: bool = False
    seed: int | None = None
    size_ratio: float = DEFAULT_SIZE_RATIO


@dataclass
class Result:
    code: str
    renamed: int = 0
    labels: int = 0
    strings: int = 0
    numbers: int = 0
    bools: int = 0
    padding: int = 0


def obfuscate(source: str, filename: str = "<input>",
              options: Options | None = None) -> Result:
    opts = options or Options()
    if source.startswith("\ufeff"):
        source = source[1:]
    block = parse(source, filename)

    seed = opts.seed if opts.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    gen = NameGenerator(collect_identifiers(block), seed=seed)

    result = Result(code="")

    if opts.rename:
        renamer = Renamer(gen)
        renamer.apply(block)
        result.renamed = renamer.renamed
        result.labels = renamer.labels

    if opts.strings:
        enc = StringEncryptor(gen, rng)
        result.strings = enc.apply(block)
        if result.strings:
            block.stmts[:0] = enc.make_decoder()

    if opts.bools:
        result.bools = BoolObfuscator().apply(block)

    if opts.numbers:
        result.numbers = NumberObfuscator(rng).apply(block)

    shebang = ""
    if source.startswith("#!"):
        end = source.find("\n")
        shebang = source[:end + 1] if end >= 0 else source + "\n"

    body = Emitter(compact=not opts.pretty).emit(block)
    prefix = shebang + BANNER + "\n"
    input_size = len(source.encode("utf-8", "surrogateescape"))
    ratio = opts.size_ratio or 0
    target = int(input_size * ratio + 0.5) if ratio > 1 else 0

    def build(payload: str) -> str:
        if opts.pack:
            return prefix + pack_source(payload, gen, rng)
        return prefix + payload

    code = build(body)
    padding_used = 0

    def byte_len(text: str) -> int:
        return len(text.encode("utf-8", "surrogateescape"))

    if target and byte_len(code) < target:
        base = code[len(prefix):]
        pad_size = target - byte_len(prefix) - byte_len(base)
        if pad_size > 0:
            padding = make_padding(pad_size, rng)
            padding_used = len(padding)
            code = prefix + padding + base
        if byte_len(code) < target:
            filler = make_filler(target - byte_len(code), rng)
            code += filler
            padding_used += len(filler)

    result.code = code
    result.padding = padding_used
    return result
