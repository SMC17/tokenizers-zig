#!/usr/bin/env python3
"""Differential-correctness checker: compares token IDs dumped by the Zig
and Rust (HuggingFace) tokenizers over the same corpus + fixture and
reports exact-match counts plus every mismatching line.

Usage: scripts/diff_check.py <zig.jsonl> <rust.jsonl> <label>
Exit code 0 iff every line matches exactly.
"""
import json
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["line"]] = d
    return out


def main():
    if len(sys.argv) != 4:
        print("usage: diff_check.py <zig.jsonl> <rust.jsonl> <label>", file=sys.stderr)
        sys.exit(2)
    zig_path, rust_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    zig = load(zig_path)
    rust = load(rust_path)

    lines = sorted(set(zig) | set(rust))
    matches = 0
    mismatches = []
    for ln in lines:
        z = zig.get(ln)
        r = rust.get(ln)
        if z is None or r is None:
            mismatches.append((ln, "missing on one side", z, r))
            continue
        if z.get("ok") != r.get("ok"):
            mismatches.append((ln, "ok/error mismatch", z, r))
            continue
        if z.get("ids") != r.get("ids"):
            mismatches.append((ln, "token-id mismatch", z, r))
            continue
        matches += 1

    total = len(lines)
    print(f"[{label}] {matches}/{total} lines exact-match")
    for ln, kind, z, r in mismatches:
        print(f"  MISMATCH line {ln} ({kind})")
        print(f"    zig:  {z}")
        print(f"    rust: {r}")

    sys.exit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
