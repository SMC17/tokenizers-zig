# Compatibility: differential correctness vs HuggingFace Rust

`parity_proof.pdf` (2026-06-13) states the formal claim

    ∀s ∈ Σ*, EncodeZig(s) ≡ EncodeRust(s)

backed by a 600-iteration random-fuzz harness reporting "100% ID
parity." **That claim is false.** A 39-line adversarial corpus
(`tests/corpus/differential_corpus.jsonl` — empty/whitespace-only
strings, CJK, Arabic/Hebrew RTL, zero-width joiners, astral-plane
codepoints, emoji ZWJ sequences, Cyrillic/Vietnamese/Turkish/German
diacritics, code snippets, repeated punctuation) found real
mismatches on all three model families the first time it was run.
The 600-iteration fuzzer's random-codepoint generator evidently never
sampled the categories that broke — that's a coverage gap in the
fuzzer, not evidence the implementation was correct.

This file is the current, reproducible compatibility record. It
replaces the PDF's universal claim until the harness below passes
100% on a materially larger corpus.

## How to reproduce

```sh
zig build diff-dump -- tests/fixtures/<fixture>.json tests/corpus/differential_corpus.jsonl > /tmp/zig.jsonl
cd bench/diff_rust && cargo build --release && cd -
./bench/diff_rust/target/release/diff_dump tests/fixtures/<fixture>.json tests/corpus/differential_corpus.jsonl > /tmp/rust.jsonl
python3 scripts/diff_check.py /tmp/zig.jsonl /tmp/rust.jsonl <label>
```

`diff_check.py` compares token IDs line-by-line and exits nonzero on
any mismatch — timing is not part of the pass criterion.

## Current results (2026-07-12, commit at HEAD)

| Fixture | Family | Exact match | Before this pass |
|---|---|---|---|
| TinyLlama-1.1B | BPE + SentencePiece | **39/39 (100%)** | 27/39 |
| T5-base | Unigram | **36/39 (92%)** | 25/39 |
| BERT-base-uncased | WordPiece | 33/39 (85%) | 33/39 (unchanged) |

## Root causes found and fixed

1. **SentencePiece byte-fallback was entirely unimplemented** (both
   `bpe.zig` and `unigram.zig`). When a codepoint had no vocab entry,
   the encoder always emitted a single `unk_id` instead of decomposing
   the codepoint into its UTF-8 bytes and looking up each byte's
   `<0xXX>` vocab entry — the mechanism SentencePiece actually uses
   for any Unicode outside a model's byte-pair vocabulary, and which
   TinyLlama's fixture explicitly declares (`model.byte_fallback:
   true`, 256 `<0xXX>` entries present). This is why the emoji/CJK/
   astral-plane lines all diverged on the BPE fixture. Fixed in both
   models; TinyLlama went to 39/39.

2. **Unigram never collapsed consecutive `unk_id` runs.** HF Rust
   treats a maximal run of unsegmentable codepoints as a *single*
   `<unk>` token; this implementation emitted one `unk_id` per
   codepoint (a 47-character unsegmentable CJK run produced 47 unk
   tokens instead of 1). Fixed via a post-processing collapse pass in
   both `encode()` and `encodeWithOffsets()`. T5 (which has no
   byte-fallback tokens at all — `model.byte_fallback` is unset in the
   T5 fixture) went from 25/39 to 36/39 on this fix alone.

## Known open gaps (not yet fixed)

- **T5, 3 remaining mismatches** (lines with a Python code snippet,
  the zero-width-joiner test, and a Gothic/astral-plane + emoji-ZWJ
  test): token-level divergences not explained by unk-collapse or
  byte-fallback — likely a Viterbi scoring or normalizer interaction
  specific to zero-width/astral-plane input. Not root-caused yet.
- **BERT/WordPiece, 5 mismatches, unchanged by this pass**: Japanese
  (mixed hiragana/katakana/kanji), Cyrillic, Vietnamese (heavy
  combining/precomposed diacritics), and a Gothic/astral-plane + ZWJ
  emoji line. WordPiece has no byte-fallback concept in HF's
  implementation, so this is a separate, unexplored bug class —
  probably in `normalizer.zig` / `unicode_norm.zig`'s handling of
  combining marks or multi-codepoint grapheme clusters, not yet
  investigated.

## What this means for the README claims

- Drop "bit-identical" / "100% ID parity" language until the fixture
  scores above all read 39/39 on a corpus at least an order of
  magnitude larger than this one.
- The WordPiece/BPE/Unigram head-to-head *timing* numbers in the
  README are unaffected by this — they were never about correctness.
  But a speed comparison against an implementation that silently
  produces wrong token IDs on non-ASCII input is not a fair
  comparison until the ID-parity gaps above are closed.
