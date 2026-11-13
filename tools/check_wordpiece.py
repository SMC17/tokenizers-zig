#!/usr/bin/env python3
"""Check the original WordPiece fixture; --live also runs pinned HF bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=ROOT / "zig-out/bin/diff-dump")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/original-wordpiece.json"
    corpus = ROOT / "tests/corpus/original-wordpiece.jsonl"
    expected = ROOT / "tests/fixtures/original-wordpiece.expected.jsonl"
    meta = json.loads((ROOT / "tests/fixtures/original-wordpiece.meta.json").read_text())
    if meta["version"] != "0.21.4" or meta["cases"] != 48:
        raise RuntimeError("Unexpected reference identity")
    for path, key in [(fixture, "fixture_sha256"), (corpus, "corpus_sha256"), (expected, "expected_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[key]:
            raise RuntimeError(f"Reference identity mismatch: {path.name}")
    texts = [json.loads(line) for line in corpus.read_text().splitlines()]
    references = [json.loads(line) for line in expected.read_text().splitlines()]
    if len(texts) != 48 or len(references) != 48 or [r["line"] for r in references] != list(range(48)):
        raise RuntimeError("Incomplete or reordered reference")
    if args.live:
        import tokenizers
        from tokenizers import Tokenizer
        if tokenizers.__version__ != meta["version"]:
            raise RuntimeError("Use the exact version in tools/requirements-oracle.txt")
        reference = Tokenizer.from_file(str(fixture))
        for text, row in zip(texts, references):
            if reference.encode(text).ids != row["ids"]:
                raise RuntimeError(f"Frozen reference differs at line {row['line']}")
    command = [str(args.bin.resolve()), str(fixture), str(corpus)]
    first = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
    second = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
    if first != second:
        raise RuntimeError("Repeated Zig output differs")
    with tempfile.TemporaryDirectory(prefix="wordpiece-reference-") as temp:
        actual = Path(temp) / "actual.jsonl"
        actual.write_bytes(first)
        subprocess.run([sys.executable, str(ROOT / "scripts/diff_check.py"),
                        str(actual), str(expected), "original-wordpiece"], check=True)
    print(f"WORDPIECE_PARITY_OK rows=48 reference=0.21.4 live={str(args.live).lower()} repeat=identical")


if __name__ == "__main__":
    main()
