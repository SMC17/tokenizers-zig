# STATUS

**Version:** v0.1
**Last update:** 2026-05-21

## What landed in v0.1 (on top of v0.0.1)

- `src/normalizer.zig` — Prepend + Replace + Sequence normalizer
  pipeline (~150 LOC). Detects U+2581 ▁ space marker for
  SentencePiece-style tokenizers (Llama / Mistral family).
- `src/root.zig` Tokenizer extended:
  - `normalizer` field with arbitrary NormalizerKind
  - `sp_space_marker` field; decode automatically reverses ▁ → " "
    and strips the leading space when SP marker is detected
  - `fromJson` parses `normalizer` object including nested
    `Sequence` with `Prepend` + `Replace` substeps
- `tests/test_real_tinyllama.zig` — 3 integration tests against the
  real TinyLlama-1.1B-Chat-v1.0 tokenizer.json (32K vocab, 61249
  merges):
  - Parses correctly (vocab size, merge count, SP marker detected)
  - Encodes "Hello, how are you today?" to known IDs matching HF
  - Round-trip preserves text exactly across 3 sample sentences
- `bench/bench_rust/` — Cargo project with `huggingface/tokenizers`
  v0.21 + `onig` regex feature, runs the same encode loop on the
  same fixture as the Zig bench
- `bench/bench_real_tinyllama.zig` — Zig encode bench at the real
  TinyLlama fixture
- Measured head-to-head: Rust ~9x faster (55μs vs 491μs per encode
  of a 123-byte text on consumer Linux ReleaseFast)

## What's NOT covered (v0.2+)

1. **Priority-queue O(n log n) merge loop** — current is O(n²) per
   word. The 9x Rust gap is mostly attributable to this; v0.2
   should close the gap to 2-3x.
2. **Word-boundary regex pre-tokenization** matching HF's behavior
   for multi-word inputs with mixed whitespace
3. **Special tokens / added_tokens** end-to-end (skeleton present,
   not exercised)
4. **WordPiece** model (BERT family)
5. **Unigram / SentencePiece-raw** model (T5, Llama-1)
6. **Other normalizers**: NFC / NFD / NFKC / NFKD / Lowercase /
   StripAccents / Regex Replace
7. **TemplateProcessing post-processor** (BOS/EOS template)
8. **ByteFallback decoder** (handles <0xXX> tokens in Llama outputs)

## Honest non-claims

- Rust upstream is ~9x faster on the real-tokenizer bench. Same fixture,
  same correctness. Published as the README headline.
- The v0.1 parser handles the canonical Llama / Mistral pipeline. It
  will degrade to "no normalization" or "unknown normalizer" for
  tokenizer.json files with normalizers we don't recognise.
- v0.1 has no special-token-aware encode. Prefix BOS / suffix EOS
  must be added by the caller.
- Tested only on x86_64-linux with TinyLlama-1.1B; Qwen, Mistral,
  GPT-2 tokenizer.json files not yet exercised end-to-end.
