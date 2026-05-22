# Changelog

## 0.4.0 — 2026-05-21

All six v0.4-roadmap gaps addressed in one push: TemplateProcessing,
ByteFallback, Truncation/Padding, Unicode punctuation, partial
NFC/NFD/NFKC/NFKD, partial GPT-2 regex pre-tokenize. 127 tests pass.

### TemplateProcessing post-processor

- Parses `post_processor.type == "TemplateProcessing"` and the `single`
  template. SpecialToken ids resolved via the post-processor's
  `special_tokens` map.
- New `Tokenizer.encode` applies the template: `base_ids =
  encodeBase(text); out = apply_template(base_ids)`.
- New `Tokenizer.encodeBase` skips TemplateProcessing for callers that
  want raw model-side IDs.
- Real-fixture canonical IDs verified with TemplateProcessing wrap:
  - **TinyLlama**: prepends `<s>=1` (BOS)
  - **BERT-base-uncased**: wraps with `[CLS]=101 ... [SEP]=102`
  - **T5-base**: appends `</s>=1` (EOS)

### ByteFallback decoder

- `applyByteFallback(text)` scans for `<0xXX>` 6-char patterns and
  replaces each with the literal byte XX.
- `Tokenizer.use_byte_fallback` set when the decoder Sequence contains
  a ByteFallback node (Llama family).
- Applied in BPE decode before SentencePiece marker reversal.

### Truncation / padding

- New fields: `max_length` + `truncation_direction` ∈ {right, left};
  `pad_to_length` + `pad_token_id` + `padding_direction`.
- Applied post-TemplateProcessing in this order: truncate → pad.

### Unicode punctuation

- `isUnicodePunct(cp)` replaces ASCII-only `isAsciiPunct` in BERT
  pre-tokenize punctuation split. Covers:
  - ASCII (`!`–`/`, `:`–`@`, `[`–`` ` ``, `{`–`~`)
  - Latin-1 supplement punctuation
  - General Punctuation (U+2000–U+206F)
  - Supplemental Punctuation (U+2E00–U+2E7F)
  - CJK Symbols (U+3000–U+303F)
  - Vertical Forms, CJK Compatibility Forms, Small Form Variants
  - Halfwidth/Fullwidth punctuation subset
- Codepoint-aware `encodePunctSplit` (UTF-8 boundaries).

### Unicode normalization — partial

- New `src/unicode_norm.zig`: NFD / NFC / NFKD / NFKC for Latin-1
  Supplement + Latin Extended-A precomposed characters (~70 entries).
- Codepoints outside the table pass through unchanged.
- Honest non-claims documented in the file header:
  - No Hangul Jamo composition
  - No CJK compatibility decompositions
  - No combining-mark canonical reordering
  - No multi-step decomposition recursion
  - No composition exclusions
  - NFKC/NFKD alias NFC/NFD for the supported subset
- Parser: `NFC` / `NFD` / `NFKC` / `NFKD` normalizer types map to the
  corresponding `NormalizerKind` variants. `Lowercase` and
  `BertNormalizer.lowercase=true` continue to route to `.lowercase`.

### GPT-2 regex pre-tokenize — partial

- New `src/gpt2_regex.zig`: hand-coded scanner approximating the
  canonical GPT-2 regex
  `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
- Activated when ByteLevel pre-tokenizer has `use_regex: true` (HF
  default). BPE encoder splits via the regex, byte_level-encodes each
  chunk, BPE-encodes each chunk, concatenates IDs.
- Scope: ASCII contractions + ASCII/Latin-1 letter runs + ASCII digit
  runs + "other" runs + whitespace runs. CJK/Cyrillic/Arabic/Indic
  scripts fall into "other" (v0.5 work — full `\p{L}` requires Unicode
  property tables).

### Test count

127 unit + integration tests pass.

### Known non-claims for v0.4

- Unicode normalization is Latin-only (~70 codepoints). Hangul, CJK
  compatibility forms, and combining-mark canonical reordering are
  unimplemented. Multilingual NFKC normalizers will not produce
  canonical output for non-Latin input.
- GPT-2 regex pre-tokenize classifies CJK / Cyrillic / Arabic / Indic
  as "other" rather than letters. Differs from canonical
  `\p{L}` for non-Latin scripts.
- Unicode punctuation classifier uses range approximation, not full
  `\p{P}`. Cyrillic / Arabic / Indic punctuation blocks are NOT
  detected; v0.5 work.
- Truncation and padding are fields the caller sets manually after
  `fromJson`; tokenizer.json's `truncation` / `padding` objects are
  NOT parsed automatically (they are typically null in real fixtures).
- TemplateProcessing pair-template (`pair` field) is not yet wired —
  only `single` is applied.

## 0.3.0 — 2026-05-21

Real BERT-base-uncased + T5-base integration tests pass against
canonical HF token IDs. BertNormalizer + BertPreTokenizer + lowercase +
Sequence(WhitespaceSplit, Metaspace) per-piece dispatch.

### Pipeline

- **BertNormalizer parser** in `src/root.zig::parseNormalizer` — maps
  to the new `.lowercase` normalizer (`clean_text`, `handle_chinese_chars`,
  `strip_accents` are no-ops at v0.3). Defaults to lowercase=true matching
  HF behaviour.
- **`.lowercase` normalizer variant** in `src/normalizer.zig` —
  ASCII-only lowercase (`a–z ← A–Z`); non-ASCII passes through. Full
  Unicode lowercase is v0.4 work.
- **BertPreTokenizer punctuation splitting** — `BertPreTokenizer` now
  sets both `use_whitespace_split` and `split_on_punctuation`. Each
  whitespace-chunk is further split on ASCII punctuation
  (`!"#$%&'()*+,-./:;<=>?@[\]^_\`{|}~`); each punctuation char becomes
  its own one-char "word" for WordPiece. Full Unicode "P" category is
  v0.4 work.
- **Model-type inference** when `model.type` is missing (older HF
  tokenizer.json format used by BERT-base-uncased, T5-base):
  - `merges` field present → BPE
  - `continuing_subword_prefix` field present → WordPiece
  - `vocab` is an array → Unigram
- **Per-piece Metaspace** for Unigram + WhitespaceSplit + Metaspace
  Sequence (T5 pattern). Splits text on whitespace BEFORE Metaspace;
  prepends ▁ to each piece individually; Viterbi-encodes each piece
  separately. Caught by the T5 integration test on "hello </s>"
  where whole-text Metaspace would have emitted a spurious trailing ▁.

### Real-tokenizer integration tests

- **`tests/test_real_bert.zig`** — 7 tests against the real
  bert-base-uncased tokenizer.json (466 KB, 30522 vocab, WordPiece).
  Verified canonical IDs: hello=7592, world=2088, ","=1010, "!"=999,
  added_tokens [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103.
- **`tests/test_real_t5.zig`** — 5 tests against the real t5-base
  tokenizer.json (1.4 MB, 32100 vocab, Unigram). Verified canonical IDs:
  ▁hello=21820, added_tokens <pad>=0, </s>=1, <unk>=2, <extra_id_0>=32099,
  <extra_id_99>=32000.

### Test count

78 unit + integration tests pass.

### Known non-claims for v0.3

- BertPreTokenizer punctuation splitting is ASCII-only. Full Unicode
  "P" category (e.g. `「」`, `—`) is unimplemented.
- `.lowercase` normalizer is ASCII-only. Non-ASCII uppercase (e.g.
  `É → é`) passes through unchanged. Affects BERT-multilingual.
- BertNormalizer `clean_text` (BOM/zero-width strip),
  `handle_chinese_chars`, `strip_accents` are parsed-but-ignored.
- Precompiled-charsmap normalizer (T5 / mT5 / XLM-R) degrades to
  none. Affects multilingual T5 on non-ASCII inputs.
- The HF Rust crate has WordPiece + Unigram benches we don't yet match
  (no head-to-head for these models; only BPE has the head-to-head).

## 0.2.0 — 2026-05-21

Three model families (BPE + WordPiece + Unigram), priority-queue BPE
merge in production, added_tokens end-to-end. Same-day release with
v0.0.1 / v0.1.0.

### Models

- **`src/wordpiece.zig`** — WordPiece model (BERT / DistilBERT family).
  Greedy longest-match with codepoint-aware boundaries and canonical
  `##`-continuation prefix. 11 unit tests including UTF-8 multibyte,
  three-piece splits, max-input-chars guard, unknown-subword whole-word
  unk fallback, encode + decode round-trips.
- **`src/unigram.zig`** — Unigram model (T5 / mBART / XLM-R family,
  SentencePiece-style). Viterbi-best segmentation over `(token,
  log-score)` vocab. Single-codepoint unk fallback with configurable
  unk_score (default = min(scores) − 10). 11 unit tests including
  tie-break behavior, multibyte UTF-8, three-piece DP path, encode +
  decode round-trips.
- **`src/root.zig`** — `Tokenizer.model` refactored from `bpe.Model`
  directly to a `Model` tagged union over `{bpe, wordpiece, unigram}`.
  `fromJson` dispatches on `model.type ∈ {"BPE", "WordPiece", "Unigram"}`.

### Pre-tokenizers + post-processing

- **Whitespace / WhitespaceSplit / BertPreTokenizer** — treated as
  whitespace-tokenize for WordPiece word-by-word encoding.
- **Metaspace** — SentencePiece "▁" replacement with reverse on decode.
- **Sequence** — recursive composition of pre-tokenizers (already
  supported pattern, now exercised by all three model families).
- **`added_tokens` end-to-end** — top-level `added_tokens` array parsed
  from tokenizer.json. Encode runs a longest-match scan over input;
  matches emit their id verbatim (bypass model) and chunk the remainder
  for the model. Decode emits added-token content verbatim while
  routing non-added runs through the model decoder (preserves
  model-internal pair-joining semantics).

### Bench (priority-queue BPE)

- **Priority-queue O(n log n) BPE merge** replaces the v0.1 O(n²)
  greedy loop in production. Doubly-linked-list of pieces + min-heap
  of `(rank, left_idx)` candidates. Stale-entry detection via three
  layers: dead-node sentinel (`prev = -2`) on absorbed-right nodes +
  piece-identity snapshot on the left + rank re-lookup at pop time.
- Encoding byte-identical to v0.1 (33 tokens for the 123-byte test
  text, matches HF Rust reference output).

| Implementation | ns/encode (123 B, 33 tokens) | ratio |
|---|---|---|
| `huggingface/tokenizers` 0.21 (Rust) | ~46,000 | 1.00x |
| **tokenizers-zig v0.2** (Zig, PQ + DLL) | ~30,000–56,000 | **≤1.2x slower** |
| tokenizers-zig v0.1 (Zig, O(n²) greedy) | ~491,000 | ~9x slower |

Consumer Linux, ReleaseFast. Bench harness in `bench/bench_rust/`
(Cargo + `huggingface/tokenizers` 0.21 + `onig`) unchanged from v0.1.

### Test count

- 64 unit tests + 3 real-TinyLlama integration tests pass.

### Known non-claims for v0.2

- No real BERT or T5 tokenizer integration test (synthetic
  vocabularies + BERT-shaped / T5-shaped tokenizer.json snippets only;
  v0.3 work).
- No regex pre-tokenize. GPT-2 word-boundary splitting unimplemented.
- Special-token attributes (`special`, `single_word`, `lstrip`,
  `rstrip`, `normalized`) parsed but not differentiated; every entry
  emits verbatim.
- No NFC / NFD / NFKC / NFKD Unicode normalization.
- No truncation / padding / template post-processors.
- No ByteFallback decoder (handles `<0xXX>` tokens in some Llama
  outputs).

## 0.1.0 — 2026-05-21

Real Llama tokenizer support + audited head-to-head against
huggingface/tokenizers Rust crate.

- `src/normalizer.zig`: Prepend + Replace + Sequence pipeline for
  SentencePiece-style pre-BPE normalization (Llama / Mistral family).
  Auto-detects U+2581 ▁ space marker; decode reverses the mapping.
- Tokenizer.fromJson now parses `normalizer` objects including nested
  Sequence with Prepend + Replace substeps. Unknown normalizer types
  degrade gracefully to none.
- `tests/test_real_tinyllama.zig`: 3 integration tests against the real
  TinyLlama-1.1B-Chat-v1.0 tokenizer.json (32K vocab, 61249 merges).
- `bench/bench_real_tinyllama.zig` (Zig) + `bench/bench_rust/` (Cargo
  + huggingface/tokenizers 0.21 + onig): head-to-head encode bench
  on the same fixture.

## 0.0.1 — 2026-05-21

Initial commit. ByteLevel pre-tokenizer + BPE encode/decode with
HuggingFace tokenizer.json compatibility (BPE+ByteLevel subset).
20 unit tests. AGPL-3.0.
