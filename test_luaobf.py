"""Smoke tests for RAKE OBF V1.1.

Run from the project root:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from luaobf import BANNER, Options, obfuscate, parse  # noqa: E402
from luaobf.ast import Goto, Label  # noqa: E402
from luaobf.transform import walk  # noqa: E402

LUA = shutil.which("lua") or shutil.which("lua5.4") or shutil.which("lua5.3")


def run_lua(code: str) -> tuple[int, str, str]:
    assert LUA is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunk.lua"
        path.write_text(code, encoding="utf-8", newline="\n")
        proc = subprocess.run([LUA, str(path)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        return proc.returncode, proc.stdout, proc.stderr


SOURCE = r'''
local function fact(n)
  if n <= 1 then return 1 end
  return n * fact(n - 1)
end
local t = {a = 1, ["b"] = 2, 3, 4}
local s = "quote \" slash \\ newline\nzero\0"
print(fact(6), t.a, t["b"], #s, s:sub(1, 8))
'''


class ObfuscationTests(unittest.TestCase):
    def test_variants_stay_valid(self):
        for opts in (Options(seed=1, size_ratio=1),
                     Options(seed=1, size_ratio=1, pack=True),
                     Options(seed=1, size_ratio=1, pretty=True)):
            result = obfuscate(SOURCE, "sample.lua", opts)
            parse(result.code, "obfuscated")
            self.assertTrue(result.code.startswith(BANNER + "\n"))

    @unittest.skipIf(LUA is None, "Lua interpreter not found")
    def test_runtime_matches(self):
        before = run_lua(SOURCE)
        for opts in (Options(seed=2, size_ratio=1),
                     Options(seed=2, size_ratio=1, pack=True),
                     Options(seed=2, pack=True)):
            after = run_lua(obfuscate(SOURCE, "sample.lua", opts).code)
            self.assertEqual(before, after)

    def test_size_ratio_is_exact_for_large_input(self):
        source = ("-- filler\n" * 8000) + "local x = 1\nprint(x)\n"
        result = obfuscate(source, "big.lua", Options(seed=3))
        target = len(source.encode("utf-8")) * 4
        self.assertEqual(len(result.code.encode("utf-8")), target)
        parse(result.code, "padded")
        self.assertTrue(result.code.startswith(BANNER + "\n"))

    def test_padding_can_be_disabled(self):
        source = "-- filler\n" * 1000 + "print(1)\n"
        result = obfuscate(source, "big.lua",
                           Options(seed=3, size_ratio=1))
        self.assertEqual(result.padding, 0)
        self.assertLess(len(result.code), len(source))

    def test_labels_are_renamed(self):
        source = """
local x = 0
::again::
x = x + 1
if x < 3 then goto again end
print(x)
"""
        result = obfuscate(source, "labels.lua",
                           Options(seed=4, size_ratio=1))
        tree = parse(result.code)
        labels = [n.label for n in walk(tree)
                  if isinstance(n, (Label, Goto))]
        self.assertEqual(len(labels), 2)
        self.assertNotIn("again", labels)
        self.assertEqual(labels[0], labels[1])


if __name__ == "__main__":
    unittest.main()
