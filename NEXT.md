# NEXT — tokenizers-zig

_Last update: 2026-06-02 (bootstrapped per C17)_

## What just shipped

- HEAD — FIXTURES voice tightening (commit `15903a6`)
- v0.26.0 — current tag at HEAD, originally v0.0.1 2026-05-21 (24 minor versions of substrate progression)
- 163 unit tests across `src/` + `tests/`
- ByteLevel + BPE with HF tokenizer.json compat; PQ-merge speedup 9× → 1.2× Rust gap from baseline

## Next concrete action

Either: (a) run the cross-tokenizer reproducibility bench documented in FIXTURES.md and file a `/verdict` against an existing open register; or (b) pre-register the next encoding-quality hypothesis (e.g., "byte-level pretokenization matches HF on UTF-8 edge cases at N=10k corpus") before the next implementation lever. STATUS.md currently authoritative — pull the next-action from its proof-vocab table.

## In-flight hypotheses

| exp-id | hypothesis | threshold | next measurement |
|--------|------------|-----------|------------------|

(11 registers in this cwd; per global ledger sweep most are already verdicted — confirm via `stax-experiment open --lane tokenizers` before assuming open.)

## Context not obvious from git

- 24 minor versions in ~2 weeks indicates very tight ship cadence. The C9 register-before-claim discipline is being applied; the C17 NEXT.md cursor was the missing piece (closed today).
- FIXTURES.md is the cross-bench calibration baseline — every encoding-throughput claim must cite fixture.
- Pre-1.0; Zig 0.16 bound; AGPL.
- Public-surface repo — claim-scan PostToolUse hook fires on edits to this tree.
