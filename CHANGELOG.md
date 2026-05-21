# Changelog

## 0.0.1 — 2026-05-21

Initial commit. ByteLevel pre-tokenizer + BPE encode/decode with
HuggingFace tokenizer.json compatibility (BPE+ByteLevel subset).

- `src/byte_level.zig` — comptime GPT-2 byte→unicode bijection
- `src/bpe.zig` — BPE Model with greedy lowest-rank merge
- `src/root.zig` — Tokenizer + `fromJson` loader
- 20 unit tests
- `tokenize` CLI demo, `bench-encode` synthetic bench
- AGPL-3.0
