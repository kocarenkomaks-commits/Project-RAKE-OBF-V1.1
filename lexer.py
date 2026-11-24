"""A Lua 5.1 - 5.4 lexer.

Produces a flat list of tokens.  Comments and whitespace are dropped, but
string tokens keep both their decoded byte value and their original source
text so literals can be re-emitted verbatim or re-encoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEYWORDS = frozenset(
    "and break do else elseif end false for function if in local nil not or "
    "repeat return then true until while".split()
)

# `goto` is a keyword only from 5.2 on and a perfectly legal identifier in 5.1,
# so it is deliberately not in KEYWORDS: the parser recognises `goto Name`.

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(
    r"""
    0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?
  | (?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?
    """,
    re.VERBOSE,
)

# Longest first so that `...` wins over `..`, `~=` over `~`, etc.
_OPERATORS = (
    "...", "..",
    "==", "~=", "<=", ">=", "//", "<<", ">>", "::",
    "+", "-", "*", "/", "%", "^", "#", "&", "~", "|", "<", ">",
    "=", "(", ")", "{", "}", "[", "]", ";", ":", ",", ".",
)

_SIMPLE_ESCAPES = {
    "a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11,
    "\\": 92, '"': 34, "'": 39, "\n": 10,
}


class LuaSyntaxError(Exception):
    """Raised for malformed Lua source."""

    def __init__(self, message: str, line: int, col: int, filename: str = "<input>"):
        self.line = line
        self.col = col
        self.filename = filename
        super().__init__(f"{filename}:{line}:{col}: {message}")


@dataclass(slots=True)
class Token:
    kind: str      # 'name' | 'number' | 'string' | 'keyword' | 'op' | 'eof'
    text: str      # source text (for strings: the quoted literal)
    line: int
    col: int
    value: bytes | None = None   # decoded bytes, strings only

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.kind!r}, {self.text!r}, {self.line}:{self.col})"


def _long_bracket_open(src: str, i: int) -> tuple[int, int] | None:
    """If src[i] starts `[==[`, return (level, index_after_opening)."""
    if src[i] != "[":
        return None
    j = i + 1
    while j < len(src) and src[j] == "=":
        j += 1
    if j < len(src) and src[j] == "[":
        return j - i - 1, j + 1
    return None


def _long_bracket_close(src: str, i: int, level: int) -> int:
    """Index of the closing bracket given its level, or -1."""
    close = "]" + "=" * level + "]"
    return src.find(close, i)


def _decode_short_string(src: str, i: int, line: int, col: int,
                         filename: str) -> tuple[bytes, int]:
    """Decode a quoted string starting at src[i]; return (bytes, next_index)."""
    quote = src[i]
    i += 1
    out = bytearray()
    n = len(src)
    while i < n:
        c = src[i]
        if c == quote:
            return bytes(out), i + 1
        if c == "\n":
            raise LuaSyntaxError("unfinished string (newline in short string)",
                                 line, col, filename)
        if c == "\\":
            i += 1
            if i >= n:
                raise LuaSyntaxError("unfinished string", line, col, filename)
            c = src[i]
            if c in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[c])
                i += 1
            elif c == "x":
                h = src[i + 1:i + 3]
                if len(h) != 2 or not re.fullmatch(r"[0-9a-fA-F]{2}", h):
                    raise LuaSyntaxError("invalid hexadecimal escape", line, col, filename)
                out.append(int(h, 16))
                i += 3
            elif c.isdigit():
                j = i
                while j < n and j < i + 3 and src[j].isdigit():
                    j += 1
                v = int(src[i:j])
                if v > 255:
                    raise LuaSyntaxError("decimal escape too large", line, col, filename)
                out.append(v)
                i = j
            elif c == "z":
                i += 1
                while i < n and src[i] in " \t\r\n\v\f":
                    if src[i] == "\n":
                        line += 1
                    i += 1
            elif c == "u":
                m = re.match(r"u\{([0-9a-fA-F]+)\}", src[i:])
                if not m:
                    raise LuaSyntaxError("invalid unicode escape", line, col, filename)
                out += chr(int(m.group(1), 16)).encode("utf-8")
                i += m.end()
            elif c in "\r\n":
                # escaped line break: the string continues on the next line
                if c == "\r" and i + 1 < n and src[i + 1] == "\n":
                    i += 1
                out.append(10)
                line += 1
                i += 1
            else:
                raise LuaSyntaxError(f"invalid escape sequence \\{c}", line, col, filename)
        else:
            out += c.encode("utf-8", "surrogateescape")
            i += 1
    raise LuaSyntaxError("unfinished string", line, col, filename)


def _decode_long_string(src: str, i: int, level: int, line: int,
                        filename: str) -> tuple[bytes, int, int]:
    """Decode a long string whose content starts at i; return (bytes, next, lines)."""
    end = _long_bracket_close(src, i, level)
    if end < 0:
        raise LuaSyntaxError("unfinished long string", line, 0, filename)
    body = src[i:end]
    # A newline immediately after the opening bracket is ignored by Lua.
    if body.startswith("\r\n") or body.startswith("\n\r"):
        body = body[2:]
    elif body.startswith("\n") or body.startswith("\r"):
        body = body[1:]
    body = body.replace("\r\n", "\n").replace("\n\r", "\n").replace("\r", "\n")
    return (body.encode("utf-8", "surrogateescape"), end + level + 2,
            body.count("\n"))


def tokenize(src: str, filename: str = "<input>") -> list[Token]:
    """Tokenise Lua source.  Raises LuaSyntaxError on malformed input."""
    if src.startswith("\ufeff"):
        src = src[1:]

    toks: list[Token] = []
    i, n = 0, len(src)
    line, col = 1, 1

    def advance(text: str) -> None:
        nonlocal line, col
        nl = text.count("\n")
        if nl:
            line += nl
            col = len(text) - text.rfind("\n")
        else:
            col += len(text)

    # A leading `#!` line is a shebang; Lua's own interpreter skips it.
    if src.startswith("#!"):
        nl = src.find("\n")
        if nl < 0:
            return [Token("eof", "", line, col)]
        advance(src[: nl + 1])
        i = nl + 1

    while i < n:
        c = src[i]

        # ---- whitespace -------------------------------------------------
        if c in " \t\r\n\v\f":
            j = i + 1
            while j < n and src[j] in " \t\r\n\v\f":
                j += 1
            advance(src[i:j])
            i = j
            continue

        # ---- comments ---------------------------------------------------
        if c == "-" and src.startswith("--", i):
            lvl = _long_bracket_open(src, i + 2) if i + 2 < n else None
            if lvl is not None:
                level, body_start = lvl
                end = _long_bracket_close(src, body_start, level)
                if end < 0:
                    raise LuaSyntaxError("unfinished long comment", line, col, filename)
                j = end + level + 2
                advance(src[i:j])
                i = j
            else:
                j = src.find("\n", i)
                if j < 0:
                    j = n
                advance(src[i:j])
                i = j
            continue

        # ---- long strings ----------------------------------------------
        if c == "[" and (lvl := _long_bracket_open(src, i)) is not None:
            level, body_start = lvl
            start_line, start_col = line, col
            data, j, extra_lines = _decode_long_string(src, body_start, level,
                                                        line, filename)
            toks.append(Token("string", src[i:j], start_line, start_col, data))
            advance(src[i:j])
            i = j
            continue

        # ---- short strings ---------------------------------------------
        if c in "\"'":
            start_line, start_col = line, col
            data, j = _decode_short_string(src, i, line, col, filename)
            toks.append(Token("string", src[i:j], start_line, start_col, data))
            advance(src[i:j])
            i = j
            continue

        # ---- numbers ----------------------------------------------------
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = _NUM_RE.match(src, i)
            if not m:  # pragma: no cover - defensive
                raise LuaSyntaxError(f"malformed number near {src[i:i + 10]!r}",
                                     line, col, filename)
            text = m.group(0)
            toks.append(Token("number", text, line, col))
            advance(text)
            i = m.end()
            continue

        # ---- names / keywords -------------------------------------------
        m = _NAME_RE.match(src, i)
        if m:
            text = m.group(0)
            kind = "keyword" if text in KEYWORDS else "name"
            toks.append(Token(kind, text, line, col))
            advance(text)
            i = m.end()
            continue

        # ---- operators ---------------------------------------------------
        for op in _OPERATORS:
            if src.startswith(op, i):
                toks.append(Token("op", op, line, col))
                advance(op)
                i += len(op)
                break
        else:
            raise LuaSyntaxError(f"unexpected character {c!r}", line, col, filename)

    toks.append(Token("eof", "", line, col))
    return toks
