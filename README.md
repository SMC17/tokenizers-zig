# tokenizers-zig

Zig 0.16 BPE tokenizer compatible with the HuggingFace
`tokenizer.json` format (ByteLevel + BPE subset). Portfolio artifact
#4b of the 8-artifact jaw-drop portfolio.

## Status

**v0.1 — real Llama tokenizer.json + SentencePiece normalizer +
HF Rust head-to-head bench.**

What landed in v0.1 (on top of v0.0.1):
- Normalizer pipeline supporting `Sequence(Prepend, Replace)` —
  the canonical Llama / Mistral SentencePiece pre-BPE normalization
  (prepend ▁, replace " " with ▁). Auto-detects the U+2581 space
  marker and wires the inverse into decode.
- Real TinyLlama-1.1B-Chat-v1.0 tokenizer.json (32K vocab + 61249
  merges) parses + encodes + decodes round-trip. "Hello, how are
  you today?" → [15043, 29892, 920, 526, 366, 9826, 29973] →
  decode-back-exact, matching HF Rust tokenizers reference output.
- 3 new integration tests against the real tokenizer.json:
  parse-shape, known-IDs, round-trip-preservation.
- `bench/bench_rust/` head-to-head against `huggingface/tokenizers`
  Rust crate v0.21 with `onig` regex feature.

### Head-to-head bench (consumer Linux, ReleaseFast, 123-byte sample)

| Implementation | ns/encode | tokens/encode | bytes/sec |
|---|---|---|---|
| `huggingface/tokenizers` 0.21 (Rust) | ~55,000 | 33 | ~2.2 MB/s |
| **tokenizers-zig v0.1** (Zig) | ~491,000 | 33 | ~250 KB/s |
| **Ratio** | — | — | **Rust ~9x faster** |

Honest loss, published as headline:
- Rust uses hash-map-with-perfect-merge-priority + cache-line tuning.
- Zig v0.1 uses O(n²) per-word greedy merge (find best rank in current
  sequence, apply, repeat).
- Same fixture, same text, same tokenization output (we verified IDs
  match HF exactly).
- v0.2 path = priority-queue O(n log n) merge loop, expected 3-5x
  speedup.

20 unit tests + 3 real-tokenizer integration tests pass.

## What's covered

- **ByteLevel** pre-tokenizer (canonical GPT-2 byte→unicode bijection,
  built at comptime as a 256-entry static table)
- **BPE model** (vocab + merges, greedy lowest-rank merge algorithm,
  HuggingFace-compatible)
- **HuggingFace `tokenizer.json` loader** for the `BPE` model variant
  with `ByteLevel` pre-tokenizer; supports both v1 string-merges
  (`"a b"`) and v2 array-merges (`["a", "b"]`) formats
- **Encode** (UTF-8 text → token IDs) and **decode** (IDs → text)
  round-trip
- **20 unit tests**: 7 byte-level (incl. table bijection check), 8
  BPE (incl. lowest-rank-merge resolution + chained merges + unk
  fallback), 5 tokenizer.json (incl. minimal Llama-style + array-merge
  v2 + unsupported-model error)

## What's NOT covered (v0.1+)

- **WordPiece / Unigram / SentencePiece** models (BPE only at v0.0.1)
- **Whitespace / WhitespaceSplit / Regex** pre-tokenizers
- **Truncation / padding / template post-processors**
- **`added_tokens` / special tokens** (skeleton present, not exercised
  end-to-end)
- **Decoder configs beyond ByteLevel-inverse**
- **Performance** — encode is O(n²) per word at v0.0.1; v0.1 will
  ship a priority-queue-based O(n log n) merge loop
- **HuggingFace tokenizers Rust head-to-head bench** — v0.1 work
- **Word-boundary regex pre-tokenization** — v0.0.1 treats the entire
  input as one "word"; ByteLevel handles spaces via Ġ-mapping. For
  most modern Llama/Qwen tokenizers this gives the right behavior on
  short inputs.

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
}
```

## Build

```sh
zig build test                              # 20 unit tests
zig build tokenize -- f.json "<text>"       # encode + decode CLI
zig build bench                             # synthetic encode throughput
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
- The Rust crate is the reference for correctness. tokenizers-zig
  v0.0.1 doesn't yet bench against it; v0.1 will, in the same posture
  as safetensors-zig's bench/bench_rust scaffold.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **No real-tokenizer head-to-head yet.** v0.0.1 tests use synthetic
  fixtures; v0.1 will exercise a real Llama-3.2 / Qwen-2.5 / Mistral
  tokenizer.json. The HF Rust crate parses these in ~10ms; we don't
  know our number yet.
- **Single-word encode.** Real HF tokenization splits on whitespace+
  punctuation via a regex before applying BPE per word. v0.0.1 treats
  the whole input as one word; ByteLevel makes this correct for
  short ASCII strings but may produce different results from HF on
  longer inputs with mixed whitespace patterns.
- **O(n²) per-word encode.** For 100-token-long words this is fine
  (10K comparisons); for pathological 1000-char tokens it would be
  10x slower than necessary. v0.1 priority-queue work.
- **BPE only.** No WordPiece (BERT), no Unigram (T5/SentencePiece),
  no SentencePiece-raw. Roadmap.
- **Single-platform tested** (x86_64-linux, Zig 0.16 Debug + ReleaseFast).
