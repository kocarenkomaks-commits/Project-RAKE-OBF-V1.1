"""Command line interface: python -m luaobf script.lua -o script.obf.lua"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from . import __version__
from .lexer import LuaSyntaxError
from .transform import DEFAULT_SIZE_RATIO, Options, obfuscate

LUA_NAMES = ("lua", "lua5.4", "lua54", "lua5.3", "lua53",
             "luajit", "lua5.2", "lua52", "lua5.1", "lua51")


def _find_lua() -> str | None:
    for name in LUA_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _scratch_dir() -> Path:
    """First writable location for scratch files (sandboxes may lock %TEMP%).

    ``tempfile.mkdtemp`` is deliberately avoided: on Windows it creates the
    directory with a private 0o700 ACL, which some restricted tokens cannot
    write to.
    """
    for candidate in (Path(tempfile.gettempdir()), Path.cwd()):
        probe = candidate / f".luaobf-probe-{os.getpid()}"
        try:
            probe.mkdir(exist_ok=True)
            (probe / "x").write_text("x", encoding="utf-8")
            (probe / "x").unlink()
            probe.rmdir()
            return candidate
        except OSError:
            shutil.rmtree(probe, ignore_errors=True)
    raise RuntimeError("no writable temporary directory")


def _run(code: str) -> tuple[int, str, str]:
    lua = _find_lua()
    if lua is None:
        raise RuntimeError("no Lua interpreter found on PATH")
    workdir = _scratch_dir() / f"luaobf-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    workdir.mkdir()
    try:
        path = workdir / "chunk.lua"
        with open(path, "w", encoding="utf-8", errors="surrogateescape",
                  newline="\n") as fh:
            fh.write(code)
        proc = subprocess.run(
            [lua, str(path)], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="luaobf",
        description="Obfuscate Lua source code (RAKE OBF V1.1: locals and "
                    "labels renamed, strings and numbers encoded, optional "
                    "encrypted loader, decoy padding).",
    )
    p.add_argument("input", nargs="?", default="-",
                   help="input .lua file, or - for stdin (default: -)")
    p.add_argument("-o", "--output", default="-",
                   help="output file, or - for stdout (default: -)")
    p.add_argument("--pack", action="store_true",
                   help="wrap the result in an encrypted loader (needs load/loadstring)")
    p.add_argument("--no-rename", action="store_true", help="do not rename locals")
    p.add_argument("--no-strings", action="store_true", help="do not encode string literals")
    p.add_argument("--no-numbers", action="store_true", help="do not encode numbers")
    p.add_argument("--no-bools", action="store_true", help="do not encode booleans")
    p.add_argument("--ratio", type=float, default=None,
                   help="target output/input size ratio (default 4.0; use 1 to disable padding)")
    p.add_argument("--no-padding", action="store_true",
                   help="same as --ratio 1")
    p.add_argument("--pretty", action="store_true", help="indented output instead of one line")
    p.add_argument("--seed", type=int, default=None,
                   help="deterministic output for a given seed")
    p.add_argument("--verify", action="store_true",
                   help="run the original and the result with a Lua interpreter and compare")
    p.add_argument("--stats", action="store_true", help="print statistics to stderr")
    p.add_argument("--version", action="version", version=f"luaobf {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input == "-":
        source = sys.stdin.read()
        filename = "<stdin>"
    else:
        path = Path(args.input)
        try:
            with open(path, "r", encoding="utf-8", errors="surrogateescape",
                      newline="") as fh:
                source = fh.read()
        except OSError as exc:
            print(f"luaobf: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        filename = path.name

    ratio = 1.0 if args.no_padding else args.ratio
    if ratio is None:
        ratio = DEFAULT_SIZE_RATIO
    options = Options(
        rename=not args.no_rename,
        strings=not args.no_strings,
        numbers=not args.no_numbers,
        bools=not args.no_bools,
        pack=args.pack,
        pretty=args.pretty,
        seed=args.seed,
        size_ratio=ratio,
    )

    try:
        result = obfuscate(source, filename, options)
    except LuaSyntaxError as exc:
        print(f"luaobf: {exc}", file=sys.stderr)
        return 1

    if args.stats:
        input_size = len(source.encode("utf-8", "surrogateescape"))
        output_size = len(result.code.encode("utf-8", "surrogateescape"))
        ratio = output_size / max(input_size, 1)
        print(
            f"luaobf: {result.renamed} locals, {result.labels} labels, "
            f"{result.strings} strings, {result.numbers} numbers, "
            f"{result.bools} booleans, {result.padding} padding bytes",
            file=sys.stderr,
        )
        print(
            f"luaobf: {input_size} -> {output_size} bytes ({ratio:.2f}x)",
            file=sys.stderr,
        )

    if args.output == "-":
        sys.stdout.write(result.code)
        if not result.code.endswith("\n"):
            sys.stdout.write("\n")
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", errors="surrogateescape",
                  newline="\n") as fh:
            fh.write(result.code)
        print(f"luaobf: wrote {out} ({len(result.code)} bytes)", file=sys.stderr)

    if not args.verify:
        return 0

    try:
        before = _run(source)
        after = _run(result.code)
    except (RuntimeError, OSError) as exc:
        print(f"luaobf: cannot verify: {exc}", file=sys.stderr)
        return 3

    if before != after:
        print("luaobf: verification FAILED - behaviour changed", file=sys.stderr)
        print(f"  original:   exit={before[0]} stdout={before[1][:400]!r} "
              f"stderr={before[2][:400]!r}", file=sys.stderr)
        print(f"  obfuscated: exit={after[0]} stdout={after[1][:400]!r} "
              f"stderr={after[2][:400]!r}", file=sys.stderr)
        return 4

    print("luaobf: verification passed (identical output)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
