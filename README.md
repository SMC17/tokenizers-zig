# tokenizers-zig

Zig 0.16 tokenizers compatible with the HuggingFace `tokenizer.json`
format. Covers BPE (GPT-2 / Llama / Qwen / Mistral), WordPiece (BERT /
DistilBERT), and Unigram (T5 / mBART / XLM-R) model families.
Portfolio artifact #4b in the broader sovereign-stack portfolio.

## Status

**v0.14 — full pipeline, all three model families, 6-subcommand
tokenize CLI, batch-with-types APIs, expanded Unicode normalization.
163 tests pass.**

### Head-to-head vs HuggingFace Rust tokenizers

| Family | Zig | Rust HF 0.21 | Ratio |
|---|---|---|---|
| BPE+SP (TinyLlama) | ~45 μs | ~46 μs | ~1.0x (parity) |
| **WordPiece (BERT-base-uncased)** | **~15 μs** | **~82 μs** | **~5.3x Zig faster** |
| Unigram (T5-base) | ~38 μs | ~44 μs | ~1.16x Zig faster |

Same 123-byte fixture, same canonical token output, same hardware
(consumer Linux, Ryzen-class, ReleaseFast). Bench harnesses at
`bench/bench_rust_bert/`, `bench/bench_rust_t5/`, `bench/bench_rust/`.

The WordPiece 5.3x ratio is pure-Zig greedy-longest-match beating HF
Rust on the BERT pipeline. BPE is parity (priority-queue O(n log n)
merge loop matches Rust's optimised path). Unigram is +16% on
Viterbi DP.

### Real-tokenizer canonical IDs verified

All three model families have downloaded real-tokenizer fixture tests
running on every `zig build test` (skipped if fixture absent):

- **bert-base-uncased** (30522 vocab, WordPiece): hello=7592,
  world=2088, `,`=1010, `!`=999. Added: [PAD]=0, [UNK]=100, [CLS]=101,
  [SEP]=102, [MASK]=103.
- **t5-base** (32100 vocab, Unigram): ▁hello=21820. Added: <pad>=0,
  </s>=1, <unk>=2, <extra_id_0>=32099, <extra_id_99>=32000.
- **TinyLlama-1.1B** (32000 vocab, BPE+SentencePiece): "Hello, how are
  you today?" → [15043, 29892, 920, 526, 366, 9826, 29973].

## What's covered (v0.13)

### Models
- **BPE** with priority-queue O(n log n) merge loop (vocab + merges;
  HF v1 string-merges and v2 array-merges both supported)
- **WordPiece** (vocab + unk + `##` continuation + max-input-chars
  guard; canonical BERT greedy longest-match with codepoint-aware
  boundaries)
- **Unigram** (vocab + Viterbi DP best segmentation; single-codepoint
  unk fallback)

### Pre-tokenizers
- ByteLevel (canonical GPT-2 byte→unicode bijection, comptime static
  table)
- Whitespace / WhitespaceSplit / BertPreTokenizer (whitespace +
  punctuation split for BERT word-by-word encoding)
- Metaspace ("▁" replacement; reverse on decode)
- Sequence (recursive composition)
- Partial GPT-2 word-boundary regex pre-tokenize

### Normalizers
- Prepend + Replace + Sequence (Llama/Mistral SentencePiece chain)
- BertNormalizer + ASCII lowercase
- **NFC / NFD / NFKC / NFKD** Unicode normalization covering Latin,
  Latin Extended-A, Greek monotonic, Cyrillic precomposed, and
  algorithmic Hangul Jamo (U+AC00–U+D7A3)

### Post-processing
- TemplateProcessing wrap (BERT `[CLS]…[SEP]`, T5 `…</s>`)
- ByteFallback decode
- Truncation + Padding
- attention_mask + token_type_ids output (`EncodeResult`)
- Pair templates for sentence-pair tasks (BERT NSP-style)

### Batch APIs
- `encodeBatch(texts) → [][]u32` — sequential batch encode
- `encodeBatchParallel(texts) → [][]u32` — thread-pool parallel batch
- `encodeBatchWithTypes(texts) → []EncodeResult` — sequential, full
  EncodeResult with type_ids + attention_mask per text
- `encodeBatchPairWithTypes(pairs_a, pairs_b) → []EncodeResult` —
  pair batch (BERT NSP scale-out)

### Loader + dispatch
- HuggingFace `tokenizer.json` loader dispatching on `model.type`
- Model-type inference for older HF tokenizer.json format (BERT / T5
  omit `model.type`)
- added_tokens end-to-end (parse → longest-match scan in encode →
  verbatim emit in decode)

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

    // Simple encode (ids only)
    const ids = try tok.encode(alloc, "Hello, world!");
    defer alloc.free(ids);

    // Full EncodeResult with attention_mask + type_ids
    var result = try tok.encodeWithTypes(alloc, "Hello, world!");
    defer result.deinit(alloc);
    // result.ids, result.attention_mask, result.type_ids

    // Decode round-trip
    const back = try tok.decode(alloc, ids);
    defer alloc.free(back);
}
```

Same code works against BPE, WordPiece, or Unigram tokenizer.json —
`fromJson` dispatches on `model.type` internally.

## Build

```sh
zig build test                              # 163 unit + integration tests
zig build tokenize                          # build tokenize CLI (6 subcommands)
zig build bench                             # synthetic encode throughput
zig build bench-real-tinyllama              # real Llama tokenizer encode bench
```

## CLI (`tokenize`)

```sh
tokenize encode      <tokenizer.json> "<text>"        # full pipeline encode
tokenize encode-base <tokenizer.json> "<text>"        # raw model output (no template)
tokenize encode-pair <tokenizer.json> "<a>" "<b>"     # BERT NSP-style pair
tokenize encode-full <tokenizer.json> "<text>"        # ids + type_ids + mask
tokenize decode      <tokenizer.json> "<ids,csv>"     # decode IDs → text
tokenize info        <tokenizer.json>                 # model / vocab / config
```

## Position in the portfolio

tokenizers-zig is artifact #4b in the broader sovereign-stack
portfolio. It pairs with safetensors-zig (artifact #4a) as the two
HuggingFace micro-library siblings; together they make vllm-zig
(artifact #1) viable end-to-end: tokenizers-zig handles text→IDs,
safetensors-zig loads the model weights, vllm-zig runs the forward
pass.

## Comparable upstream

- [`huggingface/tokenizers`](https://github.com/huggingface/tokenizers)
  — Rust canonical with Python + Node bindings. Widely deployed,
  hand-tuned, covers all model types. Reference for correctness; the
  per-family head-to-head benches above run against this.

## License

AGPL-3.0-or-later. See `LICENSE`.

## Honest non-claims

- **Pre-1.0 substrate** per the Zig 0.16 ecosystem convention. Claims stay
  below release-readiness language until the language and stdlib hit 1.0.
- **Special-token attributes are partially honoured.** `special:
  true/false` flag is used; `single_word`, `lstrip`, `rstrip`,
  `normalized` flags are parsed but not all enforced.
- **Regex / Split pre-tokenizers.** GPT-2 word-boundary splitting is
  partial — modern Llama/Qwen tokenizers work correctly; raw GPT-2 /
  RoBERTa may diverge from HF on long mixed-whitespace inputs.
- **Single-platform tested** (x86_64-linux, Zig 0.16 Debug + ReleaseFast).
- **Synthetic-fixture WordPiece bench.** The 5.3x ratio is measured
  against a 123-byte fixture; on longer documents the absolute ratio
  shifts (the constant-factor overhead of HF's Rust crate amortises),
  but the architectural win (no GIL, no Python boundary, no
  ICU-via-FFI for normalization) is expected to persist.
