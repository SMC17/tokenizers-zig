# STATUS

**Version:** v0.0.1
**Last update:** 2026-05-21

## What landed

- `src/byte_level.zig` — canonical GPT-2 byte→unicode bijection,
  built at comptime as a 256-entry table; encode + decode round-trip
  functions
- `src/bpe.zig` — BPE Model with vocab + merges hashmaps, greedy
  lowest-rank merge encode, decode via reverse-vocab lookup
- `src/root.zig` — high-level Tokenizer + `fromJson` HF
  tokenizer.json loader supporting BPE + ByteLevel + both v1
  string-merges and v2 array-merges formats
- 20 unit tests:
  - 7 ByteLevel (Ġ mapping, ASCII identity, bijection over 256 bytes,
    encode+decode ASCII, space-as-Ġ, non-ASCII round-trip, newline
    round-trip)
  - 8 BPE (single-char vocab, single merge, lowest-rank merge wins,
    unk error, unk fallback, encode+decode round-trip, empty input,
    chained-3-piece merges)
  - 5 tokenizer.json (minimal ByteLevel+BPE, unsupported model error,
    missing model field, simple round-trip, v2 array-merges format)
- `examples/tokenize.zig` — `tokenize <tokenizer.json> "<text>"` CLI
- `bench/bench_encode.zig` — synthetic encode throughput bench
- AGPL-3.0 LICENSE

## What is open for v0.1

1. **Real-tokenizer head-to-head bench** against HF tokenizers (Rust)
2. **Priority-queue O(n log n) merge loop** (current is O(n²) per word)
3. **Word-boundary regex pre-tokenization** matching HF's behavior
4. **Special tokens / added_tokens** end-to-end
5. **WordPiece** model (BERT family)
6. **Unigram / SentencePiece** model (T5, Llama-1, Mistral-old)

## Honest non-claims

- v0.0.1 is correctness-first against the synthetic fixtures. Real
  tokenizer.json files (Llama-3.2: ~17 MB with 130K vocab + millions
  of merges) are NOT yet in the test matrix. Correctness on real
  models is gated on v0.1.
- The encode is O(n²) per word. Pathological long words are slow.
  Production tokenization (single text → many short words) is fine.
- No perf claim against HF Rust tokenizers yet.
- Tested only on x86_64-linux.
