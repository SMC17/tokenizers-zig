"""Fault tests: a differential gate must require actual successful encodings."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).resolve().parents[1] / "scripts/diff_check.py"
GOOD = {"line": 0, "ok": True, "ids": [1, 9, 2]}


class DifferentialGateTests(unittest.TestCase):
    def run_gate(self, left, right):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ["left.jsonl", "right.jsonl"]]
            for path, rows in zip(paths, [left, right]):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return subprocess.run([sys.executable, str(CHECKER), *map(str, paths), "fixture"],
                                  capture_output=True, text=True, timeout=10)

    def test_real_success_and_empty_token_sequence(self):
        for ids in [[1, 9, 2], []]:
            row = GOOD | {"ids": ids}
            self.assertEqual(self.run_gate([row], [row]).returncode, 0)

    def test_empty_dumps_refuse(self):
        self.assertNotEqual(self.run_gate([], []).returncode, 0)

    def test_equal_errors_refuse(self):
        row = {"line": 0, "ok": False, "error": "InvalidInput"}
        self.assertNotEqual(self.run_gate([row], [row]).returncode, 0)

    def test_duplicate_line_ids_refuse(self):
        self.assertNotEqual(self.run_gate([GOOD, GOOD], [GOOD]).returncode, 0)

    def test_missing_line_refuses(self):
        self.assertNotEqual(self.run_gate([GOOD, GOOD | {"line": 1}], [GOOD]).returncode, 0)

    def test_different_tokens_refuse(self):
        self.assertNotEqual(self.run_gate([GOOD], [GOOD | {"ids": [1, 8, 2]}]).returncode, 0)

    def test_invalid_row_fields_refuse(self):
        for row in [[], GOOD | {"line": -1}, GOOD | {"line": True},
                    GOOD | {"ids": None}, GOOD | {"ids": [True]},
                    GOOD | {"ids": [-1]}, GOOD | {"ok": 1}]:
            with self.subTest(row=row):
                self.assertNotEqual(self.run_gate([row], [row]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
