# Changelog

## 0.2.0 — 2026-05-21

Priority-queue O(n log n) BPE merge replaces O(n²) greedy. **~10x
speedup** on the real TinyLlama bench (491 μs → 56 μs median).
Gap to HF Rust upstream closed from ~9x to ~1.2x.

- `src/bpe.zig::bpeEncodePQ`: doubly-linked list of pieces + min-heap
  of (rank, left_idx) candidates
- Stale-entry detection: dead-node sentinel (`prev = -2`) on absorbed
  right-merged nodes + piece-identity snapshot on left + rank re-lookup
  at pop time (catches downstream-merged neighbors)
- Encoding byte-identical to v0.1 (33 tokens, matches HF Rust exactly)
- 23 unit tests pass (no behavioural regression)

## 0.1.0 — 2026-05-21

Real Llama tokenizer support + audited head-to-head against
huggingface/tokenizers Rust crate.

- `src/normalizer.zig`: Prepend + Replace + Sequence pipeline for
  SentencePiece-style pre-BPE normalization (Llama / Mistral family).
  Auto-detects U+2581 ▁ space marker; decode reverses the mapping.
- Tokenizer.fromJson now parses `normalizer` objects including nested
  Sequence with Prepend + Replace substeps. Unknown normalizer types
  degrade gracefully to none.
- `tests/test_real_tinyllama.zig`: 3 integration tests against the real
  TinyLlama-1.1B-Chat-v1.0 tokenizer.json (32K vocab, 61249 merges).
- `bench/bench_real_tinyllama.zig` (Zig) + `bench/bench_rust/` (Cargo
  + huggingface/tokenizers 0.21 + onig): head-to-head encode bench
  on the same fixture.
- Honest result: Rust ~9x faster (55μs vs 491μs per encode of 123
  bytes / 33 tokens, ReleaseFast consumer Linux). Published as
  README headline ratio. v0.2 path = priority-queue O(n log n) merge.

## 0.0.1 — 2026-05-21

Initial commit. ByteLevel pre-tokenizer + BPE encode/decode with
HuggingFace tokenizer.json compatibility (BPE+ByteLevel subset).
20 unit tests. AGPL-3.0.
