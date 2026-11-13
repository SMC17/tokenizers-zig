#!/usr/bin/env python3
"""Live, pinned-reference token-ID comparison; any mismatch returns nonzero."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tokenizers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bert", "tinyllama", "t5", "all"], default="bert")
    parser.add_argument("--bin", type=Path, default=ROOT / "zig-out/bin/diff-dump")
    args = parser.parse_args()
    if tokenizers.__version__ != "0.21.4":
        raise RuntimeError("Install the pinned oracle requirements")
    pins = json.loads((ROOT / "tests/fixtures/real-model-pins.json").read_text())
    corpus = ROOT / "tests/corpus/differential_corpus.jsonl"
    texts = [json.loads(line) for line in corpus.read_text().splitlines()]
    failed = False
    for key, pin in zip(["bert", "t5", "tinyllama"], pins):
        if args.model not in {key, "all"}:
            continue
        fixture = ROOT / pin["path"]
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != pin["sha256"]:
            raise RuntimeError("Fixture identity differs")
        reference = Tokenizer.from_file(str(fixture))
        output = subprocess.run([str(args.bin.resolve()), str(fixture), str(corpus)],
                                capture_output=True, check=True, text=True, timeout=60)
        rows = [json.loads(line) for line in output.stdout.splitlines()]
        if len(rows) != len(texts) or not rows:
            raise RuntimeError("Incomplete encoding dump")
        mismatches = [i for i, (row, text) in enumerate(zip(rows, texts))
                      if row.get("line") != i or row.get("ok") is not True
                      or row.get("ids") != reference.encode(text).ids]
        print(json.dumps({"model": key, "reference": "0.21.4", "exact": len(rows) - len(mismatches),
                          "total": len(rows), "mismatch_lines": mismatches}, sort_keys=True))
        failed |= bool(mismatches)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
