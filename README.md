# tokenizers-zig

Zig 0.16 tokenizers compatible with the HuggingFace
`tokenizer.json` format. Covers BPE (GPT-2 / Llama / Qwen / Mistral),
WordPiece (BERT / DistilBERT), and Unigram (T5 / mBART / XLM-R) model
families. Portfolio artifact #4b of the 8-artifact jaw-drop portfolio.

## Status

**v0.11 — honest head-to-head against HuggingFace Rust tokenizers
across all three model families. 147 tests pass.**

| Family | Zig | Rust HF 0.21 | Ratio |
|---|---|---|---|
| BPE+SP (TinyLlama) | ~45 μs | ~46 μs | ~1.0x (parity) |
| **WordPiece (BERT-base-uncased)** | **~15 μs** | **~82 μs** | **~5.3x Zig faster** |
| Unigram (T5-base) | ~38 μs | ~44 μs | ~1.16x Zig faster |

Same 123-byte fixture, same canonical 29-token output (BERT), same
hardware. The WordPiece 5.3x ratio is pure-Zig greedy-longest-match
beating HF Rust on consumer Linux. Bench harnesses at
`bench/bench_rust_bert/` and `bench/bench_rust_t5/`.

**v0.4 — full pipeline: TemplateProcessing wrap, ByteFallback decode,
Truncation/Padding, Unicode punctuation, partial NFC/NFD/NFKC/NFKD,
partial GPT-2 word-boundary regex pre-tokenize. 127 tests pass.**

Encoding `"hello"` now produces canonical HF-wrapped IDs:
- TinyLlama: `[1, ...]` (BOS prepended)
- BERT-base-uncased: `[101, 7592, 102]` (`[CLS] hello [SEP]`)
- T5-base: `[21820, 1]` (`▁hello </s>`)

Use `Tokenizer.encodeBase` to get raw model-side IDs without
TemplateProcessing wrap.

**v0.3 — real BERT-base-uncased + T5-base integration tests pass
against canonical HF token IDs.** BertNormalizer + BertPreTokenizer +
ASCII lowercase + per-piece Metaspace for T5-style Sequence
pre-tokenizers. Model-type inference for older HF tokenizer.json format
(BERT / T5 omit `model.type`).

### Real-tokenizer canonical IDs verified

- **bert-base-uncased** (30522 vocab, WordPiece): hello=7592, world=2088,
  ","=1010, "!"=999. Added: [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102,
  [MASK]=103.
- **t5-base** (32100 vocab, Unigram): ▁hello=21820. Added: <pad>=0,
  </s>=1, <unk>=2, <extra_id_0>=32099, <extra_id_99>=32000.
- **TinyLlama-1.1B** (32000 vocab, BPE+SentencePiece): "Hello, how are
  you today?" → [15043, 29892, 920, 526, 366, 9826, 29973].

All three model families now have a downloaded real-tokenizer fixture
test running on every `zig build test` (skip if fixture absent).

78 unit + integration tests pass.

**v0.2 — three model families + priority-queue BPE merge + added_tokens
end-to-end.**

What landed in v0.2 (on top of v0.1):
- **WordPiece model** (`src/wordpiece.zig`). Greedy longest-match with
  `##`-continuation; canonical BERT algorithm with codepoint-aware
  boundaries. 11 unit tests.
- **Unigram model** (`src/unigram.zig`). Viterbi-best segmentation over a
  vocab of `(token, log-score)` pairs. Single-codepoint fallback to
  `unk_id`. 11 unit tests.
- **Tokenizer model dispatch** as a tagged union — `fromJson` routes on
  `model.type ∈ {BPE, WordPiece, Unigram}` and produces the correct
  variant.
- **Pre-tokenizers**: Whitespace / WhitespaceSplit / BertPreTokenizer
  (treat as whitespace-split for WordPiece word-by-word encoding),
  Metaspace (SentencePiece "▁" replacement; reverse on decode).
- **added_tokens end-to-end**. The top-level `added_tokens` array is
  parsed; matches are scanned over input with longest-match-wins
  semantics, emit their id verbatim (bypass the model), and on decode
  emit their content verbatim. Chunked encode preserves model-internal
  pair-joining for non-added runs.
- **Priority-queue BPE merge**: O(n log n) per word via a doubly-linked
  list + min-heap of `(rank, left_idx)` candidates, with stale-entry
  detection via dead-node sentinel + piece-identity snapshot + rank
  re-lookup at pop. Drops the v0.1 9x Rust gap to ~1.2x.

### Head-to-head bench (consumer Linux, ReleaseFast, 123-byte sample)

| Implementation | ns/encode (median, 5+ runs) | tokens/encode | ratio |
|---|---|---|---|
| `huggingface/tokenizers` 0.21 (Rust) | ~46,000 | 33 | 1.00x (ref) |
| **tokenizers-zig v0.2** (Zig, PQ + DLL) | ~30,000–56,000 | 33 | **≤1.2x slower** |
| tokenizers-zig v0.1 (Zig, O(n²) greedy) | ~491,000 | 33 | ~9x slower |

Same fixture, same text, same tokenization output. Bench harness in
`bench/bench_rust/` (Cargo + `huggingface/tokenizers` 0.21 + `onig`)
runs the same encode loop as `bench/bench_real_tinyllama.zig`. Same
hardware; numbers will vary across machines.

64 unit tests + 3 real-TinyLlama integration tests pass.

## What's covered

- **ByteLevel** pre-tokenizer (canonical GPT-2 byte→unicode bijection,
  built at comptime as a 256-entry static table)
- **Whitespace / WhitespaceSplit / BertPreTokenizer** (whitespace-split
  for word-by-word WordPiece encoding)
- **Metaspace** pre-tokenizer ("▁" replacement; reverse on decode)
- **Sequence** pre-tokenizer (recursive composition)
- **Prepend + Replace + Sequence** normalizers (Llama/Mistral
  SentencePiece pre-BPE chain)
- **BPE model** with priority-queue O(n log n) merge loop (vocab +
  merges; HF v1 string-merges and v2 array-merges both supported)
- **WordPiece model** (vocab + unk + `##` continuation + max-input-chars
  guard)
- **Unigram model** (vocab + Viterbi DP; single-codepoint unk fallback)
- **added_tokens** end-to-end (parse → longest-match scan in encode →
  verbatim emit in decode)
- **HuggingFace `tokenizer.json` loader** dispatching on `model.type`
- **Encode** (UTF-8 text → token IDs) and **decode** (IDs → text)
  round-trip

## What's NOT covered (v0.3+)

- **BertPreTokenizer punctuation splitting** beyond whitespace. v0.2
  treats Bert / Whitespace / WhitespaceSplit identically as
  whitespace-tokenize. Real BERT pre-tokenization splits on punctuation
  too.
- **NFC / NFD / NFKC / NFKD** Unicode normalization.
- **Regex / Split** pre-tokenizers (the GPT-2 word-boundary regex).
  v0.0.1 → v0.2 treat ByteLevel-encoded text as a single "word"; for
  modern Llama/Qwen tokenizers this is correct on short ASCII inputs
  but may diverge from HF on long mixed-whitespace inputs.
- **Truncation / padding / template post-processors**.
- **Special-token attribute handling**. v0.2 parses `added_tokens` and
  treats every entry uniformly. The `special: true/false`,
  `single_word`, `lstrip`, `rstrip`, `normalized` flags are ignored.
- **Decoder configs beyond ByteLevel-inverse / SentencePiece-marker /
  Metaspace-inverse**.
- **WordPiece + Unigram real-model head-to-head benches** — v0.2 ships
  the Llama BPE head-to-head; BERT-base + T5-base benches are v0.3.

## Quickstart

```zig
const std = @import("std");
const tokenizers = @import("tokenizers");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    const json = try std.Io.Dir.cwd().readFileAlloc(
        io, "tokenizer.json", alloc, std.Io.Limit.limited(64 * 1024 * 1024),
    );
    defer alloc.free(json);

    var tok = try tokenizers.fromJson(alloc, json);
    defer tok.deinit();

    const ids = try tok.encode(alloc, "Hello, world!");
    defer alloc.free(ids);
    // ids is a slice of u32 token IDs

    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
    // back is the round-tripped UTF-8 text
}
```

The same code works against a BPE, WordPiece, or Unigram tokenizer.json
— `fromJson` dispatches on `model.type` internally.

## Build

```sh
zig build test                              # 64 unit + 3 integration tests
zig build tokenize -- f.json "<text>"       # encode + decode CLI
zig build bench                             # synthetic encode throughput
zig build bench-real-tinyllama              # real Llama tokenizer encode bench
```

## Position in the portfolio

tokenizers-zig is artifact #4b in the 8-artifact portfolio. It pairs
with safetensors-zig (artifact #4a) as the two HuggingFace
micro-library siblings. Together they make vllm-zig (artifact #1)
viable end-to-end: tokenizers-zig handles text→IDs, safetensors-zig
loads the model weights, vllm-zig runs the forward pass.

## Comparable upstream

- [`huggingface/tokenizers`](https://github.com/huggingface/tokenizers)
  — Rust canonical with Python + Node bindings. Production-grade,
  hand-tuned, covers all model types (BPE, WordPiece, Unigram).
- The Rust crate is the reference for correctness. The Llama
  head-to-head (`bench/bench_rust/`) ran the same fixture +
  same encode loop, with output IDs verified equal.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **No real BERT or T5 integration test yet.** WordPiece and Unigram
  pass unit tests against synthetic vocabularies and parse a real
  BERT-shaped / T5-shaped `tokenizer.json` snippet in unit tests, but
  no real downloaded tokenizer.json has been exercised yet (v0.3 work).
- **No regex pre-tokenize.** GPT-2 style word-boundary splitting is
  unimplemented. For Llama / Qwen / Mistral (SentencePiece pipeline)
  this is fine; for raw GPT-2 / RoBERTa the encoder may produce
  different boundaries from HF on long mixed-whitespace inputs.
- **Special-token attributes are dropped.** v0.2's added_tokens layer
  treats every entry as a verbatim emit; `special: false` content
  tokens are not distinguished from `special: true` control codes.
- **Single-platform tested** (x86_64-linux, Zig 0.16 Debug + ReleaseFast).
- **Pre-1.0 substrate** per the Zig 0.16 ecosystem convention. No
  "production-grade" / "v1.0" / "stable" claims will be made.
