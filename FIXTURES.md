# Fixtures

The ~5× WordPiece speedup vs Rust upstream and the canonical-ID verification across BPE / WordPiece / Unigram are bounded by the specific tokenizer fixtures used to produce them. The fixtures are part of the published contribution.

## `tests/fixtures/bert-base-uncased/tokenizer.json`

Official HuggingFace `bert-base-uncased` tokenizer.json. WordPiece, 30,522-token vocabulary. The WordPiece calibration fixture.

Canonical IDs verified by `tests/test_realmodel_bert.zig`:

- `hello` → 7592
- `world` → 2088
- `,` → 1010
- `!` → 999
- `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103

These are the IDs the BERT model expects. A reimplementation returning different IDs for the same inputs against this vocabulary is wrong — or, more politely, is a different tokenizer.

Bench claim provenance: ~5× faster than HF Rust on WordPiece comes from a 123-byte BERT-base-uncased input ("`This is a relatively long sentence used to exercise the WordPiece greedy-longest-match path with multiple subwords.`" or close), median of N=100, ReleaseFast. Harness lives at `bench/bench_rust_bert/`.

Curation choices:

- 123 bytes, not 12 or 12,000. Short inputs bottleneck on Python-boundary cost; very long inputs bottleneck on memory bandwidth. The 123-byte range is the "single sentence" workload most BERT-class inference hits.
- English. Tokenizer hot-path behaviour differs on high-codepoint scripts (CJK, Arabic, Indic) because pre-tokenization and Unicode normalization paths trigger differently. Non-Latin scripts would be a separate bench.
- No HF `Encoding` wrap overhead in the Rust path. The bench isolates the model-and-pre-tokenizer hot path, not the TemplateProcessing wrap. The methodology question was filed upstream as [huggingface/tokenizers#2075](https://github.com/huggingface/tokenizers/issues/2075).

## `tests/fixtures/tinyllama-real/tokenizer.json`

Official TinyLlama-1.1B-Chat-v1.0 tokenizer.json. BPE + SentencePiece, 32,000-token vocabulary.

Canonical IDs verified:

- `"Hello, how are you today?"` → `[15043, 29892, 920, 526, 366, 9826, 29973]`

That prompt exercises the BPE-merge path, the SentencePiece byte-fallback path, the `▁` (Metaspace) replacement, and punctuation pre-tokenization. A shorter prompt would not surface as many merge-decisions in the hot loop.

Bench claim provenance: ~45 µs for BPE+SP encoding on this prompt against this tokenizer at this vocabulary state.

## `tests/fixtures/t5-base/tokenizer.json`

Official T5-base tokenizer.json. Unigram, 32,100-token vocabulary.

Canonical IDs verified:

- `▁hello` → 21820
- `<pad>`=0, `</s>`=1, `<unk>`=2, `<extra_id_0>`=32099, `<extra_id_99>`=32000

## `tests/test_fuzz_invariants.zig` — property fuzz

Property-based invariant fuzzer. For each real fixture (BERT, TinyLlama, T5), generates 200 random ASCII texts and verifies five invariants:

1. Non-empty input produces at least one token
2. All offsets within `[0, text.len]`
3. Offsets monotonic non-decreasing
4. `attention_mask`, `type_ids`, `offsets`, `tokens`, `special_tokens_mask` all parallel to `ids`
5. Decode does not error

What the fuzz caught at v0.26: the TinyLlama BPE+SP-marker offset remap violated invariant 3 when the input contained adjacent spaces. The double-`▁▁` from `Replace(" "→"▁")` produced all-synthetic-marker tokens that the fallback wrote at `chunk_text.len` rather than at the running cursor — breaking monotonicity for tokens *after* the synthetic ones. The fuzz hit it on text `iHuL?OXlo,kxi0nl7a  VfXytl` (two adjacent spaces). Fix shipped in the same release.

The fuzz harness IS the rigour. A reimplementation that skips it will probably also miss this remap. The harness is curated to exercise the axes where BPE+SP tokenization has historical correctness pitfalls.

## Why reproducibility binds to the fixture

A port reproducing this work needs to:

1. Match canonical IDs against the exact `tokenizer.json` files listed above. Different vocabulary files give different IDs; that is not refutation, it is a different tokenizer.
2. Run the same property-fuzz harness on the same fixtures. Fuzz findings are seed-deterministic.
3. Use the same bench-input string (123-byte BERT prompt) when citing the ~5× comparison. Different prompts produce different ratios.

The fixture set is the calibration baseline that bounds every claim. Ports running different fixtures are not failing to reproduce — they are running a different experiment.

## Provenance, licence

`bert-base-uncased/`, `tinyllama-real/`, `t5-base/`: each published under Apache 2.0 via HuggingFace Hub. Gitignored, downloaded at test time.

Property-fuzz harness (`tests/test_fuzz_invariants.zig`) and bench harnesses (`bench/bench_rust_bert/`, `bench/bench_rust_t5/`, `bench/bench_rust/`): AGPL-3.0-or-later.

Cite the fixture and the version it ran against: `tokenizers-zig v0.26 vs huggingface/tokenizers 0.21 on bert-base-uncased, 123-byte sentence, ReleaseFast, x86_64-linux, bench/bench_rust_bert/`.
