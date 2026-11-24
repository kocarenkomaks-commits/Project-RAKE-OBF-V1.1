"""luaobf - a pure-Python Lua source obfuscator.

Quick start::

    from luaobf import obfuscate, Options

    result = obfuscate(open("script.lua").read(), "script.lua",
                       Options(pack=True))
    print(result.code)
"""

from .emitter import Emitter, quote_bytes
from .lexer import LuaSyntaxError, tokenize
from .parser import Parser, parse
from .transform import (
    BANNER, DEFAULT_SIZE_RATIO, Options, Result, obfuscate,
)

__all__ = [
    "obfuscate", "Options", "Result", "parse", "tokenize", "Parser",
    "Emitter", "quote_bytes", "LuaSyntaxError", "BANNER",
    "DEFAULT_SIZE_RATIO",
]

__version__ = "1.1.0"
