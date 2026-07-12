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

| Fixture | Family | Exact match | Before any fixes |
|---|---|---|---|
| TinyLlama-1.1B | BPE + SentencePiece | **39/39 (100%)** | 27/39 |
| T5-base | Unigram | **36/39 (92%)** | 25/39 |
| BERT-base-uncased | WordPiece | **38/39 (97%)** | 33/39 |

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

3. **`normalizer.zig`'s `.lowercase` was ASCII-only.** Diffed
   normalized text directly against HF Rust (`bench/norm_dump.zig` +
   `bench/diff_rust/src/bin/norm_dump.rs`, isolating normalization from
   subword matching) and found HF's real BertNormalizer lowercases
   Cyrillic and Greek too (`char.is_uppercase()` in Rust, not an ASCII
   table). Added codepoint-aware case folding for those two scripts
   (verified — not a general Unicode case-folding table; other scripts
   are unverified and not claimed). Caught and fixed a regression this
   introduced (broke ASCII lowercasing) via the existing unit-test
   suite before it shipped.

4. **`isControl` (clean_text) didn't strip zero-width format
   characters.** HF's real `is_control` is `char.is_other()` — the
   general Unicode category grouping Cc+Cf+Cn+Co — not a hardcoded
   control-char range. Added the Cf ranges that matter in practice
   (ZWSP/ZWNJ/ZWJ, bidi controls, BOM, soft hyphen), verified against
   the corpus's zero-width-joiner test line matching exactly. Cn/Co are
   not covered.

5. **`stripAccents` only covered U+0300–U+036F, and NFD had no
   Hiragana/Katakana decomposition entries at all.** HF's real
   `strip_accents` is `!c.is_mark_nonspacing()` (general Mn category).
   Two fixes: added the Combining Katakana-Hiragana Voiced/Semi-Voiced
   Sound Marks (U+3099/U+309A) to `isCombiningMark`, and — the actual
   root cause — added the full Hiragana+Katakana dakuten/handakuten
   decomposition table to `unicode_norm.zig`'s NFD table (it had zero
   kana entries; "で" never decomposed to "て"+U+3099 in the first
   place, so there was nothing for strip_accents to strip). Also added
   the Thai combining-mark range (U+0E31, U+0E34–U+0E3A, U+0E47–U+0E4E)
   after diffing normalized Thai text found the exact stripped
   codepoints empirically rather than guessing from Unicode charts.

6. **NFD decomposition was single-level; Vietnamese compound
   diacritics need recursion.** Real Unicode canonical decomposition
   is recursive by definition — a character's Decomposition_Mapping
   can itself be a further-decomposable character. Vietnamese "ệ"
   (U+1EC7) decomposes to "ẹ"+circumflex, and "ẹ" (U+1EB9) itself
   further decomposes to "e"+dot-below; the old `nfd()` did exactly
   one `decompose()` lookup per input codepoint, so "ẹ" was never
   re-decomposed and `stripAccents` had nothing left to strip beyond
   the outer mark. Made `nfd()` recursive (`appendDecomposedRecursive`)
   and added the specific codepoints verified against the corpus (6
   Vietnamese compound characters + 3 intermediate single-diacritic
   bases they resolve through) — not the full ~130-entry Vietnamese
   set. BERT went 37/39 → 38/39 on this fix alone.

## Known open gaps (not yet fixed)

- **BERT/WordPiece, 1 remaining mismatch**: a currency/number line
  (£15.00 tail — normalized text is byte-identical between Zig and
  Rust, so this is confirmed a genuine WordPiece greedy-match
  divergence, not a normalizer issue). Not root-caused to the exact
  code path yet — needs a WordPiece-algorithm trace, not normalizer
  diffing.
- **T5/Unigram, 3 remaining mismatches — confirmed cause: `Precompiled`
  is silently a no-op.** T5's `tokenizer.json` specifies
  `"normalizer": {"type": "Precompiled", "precompiled_charsmap": ...}`
  — SentencePiece's own binary-charsmap normalizer, entirely separate
  from BertNormalizer's named-step Sequence, and not implemented here.
  `parseNormalizer` (root.zig) falls through to `.none` for any
  unrecognized type, so T5 loads and encodes with **zero normalization
  applied** — no error, no warning. Tried making this a hard error for
  `Precompiled` specifically (loud failure instead of silent wrong
  output); reverted immediately because it breaks loading T5 entirely
  (0/7 tests failed) for a model family — T5/mBART/ALBERT/XLNet — where
  `Precompiled` is the *only* normalizer type that ever appears, not an
  edge case. A working-but-imperfect T5 path (36/39 on the differential
  corpus despite the gap) is more useful than none. The real fix is
  implementing the Precompiled charsmap format (a compact double-array
  trie), not erroring around not having it — that's a genuinely new,
  bounded feature, not a small extension of the BERT fixes above.

## What this means for the README claims

- Drop "bit-identical" / "100% ID parity" language until the fixture
  scores above all read 39/39 on a corpus at least an order of
  magnitude larger than this one.
- The WordPiece/BPE/Unigram head-to-head *timing* numbers in the
  README are unaffected by this — they were never about correctness.
  But a speed comparison against an implementation that silently
  produces wrong token IDs on non-ASCII input is not a fair
  comparison until the ID-parity gaps above are closed.
