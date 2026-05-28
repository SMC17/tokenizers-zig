# Test fixtures — what they are, why they bound the claims

The "~5x faster than HF Rust on WordPiece" claim and the canonical-ID
verification across BPE / WordPiece / Unigram are bounded by the
specific tokenizer fixtures used to produce them. The fixtures are
part of the published contribution.

## Fixture inventory

### `tests/fixtures/bert-base-uncased/tokenizer.json`

**What:** The official HuggingFace `bert-base-uncased` tokenizer.json
(WordPiece, 30,522-token vocabulary). Used as the calibration fixture
for the WordPiece path.

**Canonical IDs verified per the `tests/test_realmodel_bert.zig` test:**
- `hello` → 7592
- `world` → 2088
- `,` → 1010
- `!` → 999
- Special tokens: `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103

These are the canonical IDs the model expects. Any reimplementation
that returns different IDs for the same inputs against this same
vocabulary is wrong (or, more politely, is a different tokenizer).

**Bench claim provenance:** the "~5x faster than HF Rust on WordPiece"
number comes from a 123-byte BERT-base-uncased input bench-fixture
("`This is a relatively long sentence used to exercise the WordPiece
greedy-longest-match path with multiple subwords.`" or close), median
of N=100, ReleaseFast. The bench harness lives at
`bench/bench_rust_bert/`.

**Taste decisions embedded:**

1. **123 bytes, not 12 or 12,000.** Short inputs (single-word) bottleneck
   on Python-boundary call overhead; very long inputs bottleneck on
   memory bandwidth. The 123-byte range is the "single sentence"
   workload that most BERT-class inference traffic actually hits. The
   speedup ratio is fixture-bound.
2. **English input.** Tokenizer hot-path behaviour differs on
   high-codepoint scripts (CJK, Arabic, Indic) because pre-tokenization
   + Unicode normalization paths trigger differently. The English
   fixture is the most common production case; non-Latin scripts are a
   separate bench.
3. **No HF `Encoding` wrap overhead in the Rust path** — the bench
   isolates the model-and-pre-tokenizer hot path, not the
   TemplateProcessing wrap. See [this filed issue][issue-2075] for the
   methodology question we raised with the HF team.

[issue-2075]: https://github.com/huggingface/tokenizers/issues/2075

### `tests/fixtures/tinyllama-real/tokenizer.json`

**What:** The official TinyLlama-1.1B-Chat-v1.0 `tokenizer.json`
(BPE+SentencePiece, 32,000-token vocabulary).

**Canonical IDs verified:**
- `"Hello, how are you today?"` → `[15043, 29892, 920, 526, 366, 9826, 29973]`

**Why this specific prompt:** it exercises the BPE-merge path, the
SentencePiece-byte-fallback path, the `▁` (Metaspace) replacement,
and the punctuation pre-tokenization. A simpler prompt (e.g.,
`"Hello"`) would not surface as many merge-decisions in the hot loop.

**Bench claim provenance:** "BPE+SP at ~45 µs" is for this exact
prompt against this exact tokenizer at this exact vocabulary state.

### `tests/fixtures/t5-base/tokenizer.json`

**What:** The official T5-base `tokenizer.json` (Unigram,
32,100-token vocabulary).

**Canonical IDs verified:**
- `▁hello` → 21820
- Added tokens: `<pad>`=0, `</s>`=1, `<unk>`=2, `<extra_id_0>`=32099, `<extra_id_99>`=32000

### `tests/test_fuzz_invariants.zig`

**What:** Property-based invariant fuzzer. For each real fixture
(BERT, TinyLlama, T5), generates 200 random ASCII texts and verifies
five invariants:

1. Non-empty input → ≥1 token
2. All offsets within `[0, text.len]`
3. Offsets monotonic non-decreasing
4. `attention_mask` / `type_ids` / `offsets` / `tokens` /
   `special_tokens_mask` all parallel to `ids`
5. Decode does not error

**What the fuzz caught (v0.26):** TinyLlama BPE+SP-marker offset remap
violated invariant 3 when the input contained adjacent spaces (the
double-`▁▁` from `Replace(" "→"▁")` produced all-synthetic-marker tokens
that the fallback wrote at `chunk_text.len` rather than at the
running cursor, breaking monotonicity for tokens *after* the synthetic
ones). Caught by the fuzz on text `iHuL?OXlo,kxi0nl7a  VfXytl` (two
adjacent spaces hit the case). Fix shipped same release.

**Why this fixture matters:** the fuzz harness IS the rigour. Anyone
reimplementing the tokenizer who skips this fuzz step will probably
also miss this offset-remap bug. The fuzz harness is curated to
exercise the exact axes where BPE+SP tokenization has historical
correctness pitfalls.

## Why the fixtures matter for reproducibility

A port to another language reproducing this work needs to:

1. Match the canonical IDs against the exact `tokenizer.json` files
   listed above. Different vocabulary files will give different IDs;
   that's not refutation, it's a different tokenizer.
2. Run the same property-fuzz harness on the same fixtures (the
   harness is committed in `tests/test_fuzz_invariants.zig`). Fuzz
   findings are seed-deterministic.
3. Use the same bench-input string (123-byte BERT prompt) when
   citing the "~5x faster than HF Rust" comparison. Different prompts
   produce different ratios.

The fixture set is the calibration baseline that bounds every claim.
Ports running different fixtures are not failing to reproduce —
they're running a different experiment.

## Fixture provenance + license

- `bert-base-uncased/tokenizer.json` — published by Google under
  Apache 2.0 via HuggingFace Hub. Gitignored; CI downloads at test time.
- `tinyllama-real/tokenizer.json` — published by TinyLlama authors
  under Apache 2.0 via HuggingFace Hub. Gitignored; CI downloads at test time.
- `t5-base/tokenizer.json` — published by Google under Apache 2.0 via
  HuggingFace Hub. Gitignored; CI downloads at test time.

The property-fuzz harness (`tests/test_fuzz_invariants.zig`) and the
bench harness (`bench/bench_rust_bert/`, `bench/bench_rust_t5/`,
`bench/bench_rust/`) are committed under AGPL-3.0-or-later.

When citing benchmark numbers, cite the fixture: `tokenizers-zig
v0.26 vs huggingface/tokenizers 0.21 on bert-base-uncased / 123-byte
sentence / ReleaseFast / x86_64-linux; bench at bench/bench_rust_bert/`.
