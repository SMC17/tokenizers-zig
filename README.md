# tokenizers-zig

**[Status](STATUS.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Issues](https://github.com/SMC17/tokenizers-zig/issues) · [All SMC17 work](https://github.com/users/SMC17/projects/16)**

[![CI](https://github.com/SMC17/tokenizers-zig/actions/workflows/ci.yml/badge.svg)](https://github.com/SMC17/tokenizers-zig/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/SMC17/tokenizers-zig?sort=semver)](https://github.com/SMC17/tokenizers-zig/releases)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.16-orange.svg)](https://ziglang.org/)

Zig 0.16 tokenizers compatible with the HuggingFace `tokenizer.json`
format. Covers BPE (GPT-2 / Llama / Qwen / Mistral), WordPiece (BERT /
DistilBERT), and Unigram (T5 / mBART / XLM-R) model families.

## Current bounded task — 12 November 2026

The original 32-token WordPiece fixture now matches pinned Hugging Face
`tokenizers==0.21.4` on all 48 short inputs. The eight-input regression matched
only two inputs before the punctuation repair; all eight now agree. A generated
predicate observes the reference's behavior over all 1,112,064 Unicode scalar
values: 726 punctuation scalars in 156 ranges. Currency symbols and some CJK
letters no longer split as punctuation, while Arabic/Indic punctuation does.

```sh
zig build test --summary all
zig build -Doptimize=ReleaseSafe
python3 -m unittest discover -s tests -p test_diff_check.py
python3 tools/check_wordpiece.py
```

The current local gate passes 166 of 195 Zig tests; 29 real-model/fuzz tests
remain skipped because their separate fixtures are absent. Seven Python gate
fault tests pass. The offline task checks exact token IDs against frozen
reference output; optional live verification uses the pinned wheel and fetches
no models. See [the original fixture contract](docs/WORDPIECE_TASK.md).

Empty dumps, duplicate line IDs, malformed token IDs and matching errors now
fail the comparator. Earlier code reported `0/0` empty dumps as success.

This task covers token IDs for one explicit vocabulary and the pinned BERT
punctuation component. It does not certify BPE, Unigram, general normalization,
offsets, every Encoding field, or current real-model parity. Existing fixture
expectations remain unchanged; no release or visibility change is made.

### Pinned real-model follow-up

Optional `python3 tools/fetch_real_fixtures.py` downloads three immutable,
checksum-verified tokenizer JSON files (about 3.7 MB total; no weights). With
those files, all 195 tests pass. The existing 39-input corpus now matches
BERT 39/39 and TinyLlama 39/39 against the pinned reference. T5 remains 36/39;
`tools/check_real_parity.py --model all` reports the mismatches and fails.
The files stay gitignored; existing real-fixture expectations are unchanged.
See the task contract for exact commands and source identities.

## Historical status and remaining model gaps

**v0.27 — full pipeline, all three model families, 6-subcommand
tokenize CLI, batch-with-types APIs, HF `Encoding` return fields, sub-token offsets, property-fuzz-tested. The current clean-source result is above; 29 real-model tests (BERT / T5 / TinyLlama) skip
because the HF `tokenizer.json` fixtures are not distributed here. To
enable them, place the upstream `tokenizer.json` files in
`tests/fixtures/` under the names the tests expect (e.g.
`bert-base-uncased-tokenizer.json`). The 600-iter invariant fuzz also
requires those fixtures.**

**Differential correctness vs HF Rust — see [COMPATIBILITY.md](COMPATIBILITY.md).**
`parity_proof.pdf` (2026-06-13) claims universal bit-identical parity;
that claim is false — a 39-line adversarial corpus found real
mismatches on all three model families. Root-caused and fixed 6 bugs:
SentencePiece byte-fallback was unimplemented (TinyLlama → **39/39**);
Unigram never collapsed consecutive-unknown runs (T5 → **36/39**); and
BertNormalizer's lowercase was ASCII-only, its `clean_text` didn't
strip zero-width joiners, its NFD table had zero Hiragana/Katakana
decomposition entries, and NFD decomposition wasn't recursive (needed
for Vietnamese's double-diacritic characters) (BERT → **38/39**).
The historical BERT currency gap is resolved in the current pinned replay.
Three T5 gaps remain (confirmed cause: T5 uses
SentencePiece's separate `Precompiled`-charsmap normalizer, entirely
unimplemented — falls through to a silent no-op rather than erroring,
because erroring breaks loading T5 at all) remain open. Full detail,
reproduction steps, and the open gaps are in COMPATIBILITY.md — read
that before the PDF.

### Historical timing evidence

The older timing tables remain in STATUS.md. They have not been remeasured in
this acceptance run and are not a current speedup claim. Compatibility failures
must be resolved separately from timing a single successful input.

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

## What's covered

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
zig build test                              # 195 tests (29 skip without HF fixtures)
zig build test-fuzz                          # standalone fuzz (BERT/T5/TinyLlama × 200 random texts)
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

## Siblings

- [`SMC17/safetensors-zig`](https://github.com/SMC17/safetensors-zig) — HuggingFace safetensors weight loader.

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
- **Not bit-identical to HF Rust on all input.** See
  [COMPATIBILITY.md](COMPATIBILITY.md): BERT/WordPiece has 5 known
  mismatches (CJK, Cyrillic, Vietnamese, astral-plane input) and
  T5/Unigram has 3, none yet root-caused. Do not treat `parity_proof.pdf`
  as current — it predates this finding.

## Reproducible setup

```
nix develop        # Zig 0.16.0 + jq + curl + gawk + python3(websockets)
zig build test     # 195 tests; 166 pass without HF fixtures, 29 skip
```

The repo's `flake.nix` pins exact dep versions (see `STATUS.md`).
`scripts/checkpoint.sh` + `.githooks/post-commit` auto-regenerate
`NEXT.md` after every commit (opt-in: `git config core.hooksPath .githooks`).
