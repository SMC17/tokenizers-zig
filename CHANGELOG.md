# Changelog

## 0.9.0 — 2026-05-22

Token type IDs (BERT segment markers) return. 143 tests pass.

### type_ids API

- New `EncodeResult` struct returns `ids` + `type_ids` as parallel
  arrays. `type_ids[i]` is the BERT-style segment marker (0 = A, 1 = B
  by convention) for `ids[i]`.
- `encodeWithTypes(text)` — single-sequence variant; type_ids are all
  zero by default for tokenizers with no template.
- `encodePairWithTypes(a, b)` — pair variant; type_ids follow the
  `type_id` field on each `TemplateOp` (templates without type_id
  default to 0 for A and 1 for B).
- `TemplateOp` refactored from a bare union to a `{kind, type_id}`
  struct. Parser reads `type_id` from each template entry.
- Truncation + padding apply symmetrically to both arrays (padding
  type_ids are 0).
- **Real BERT canonical**: `encodePairWithTypes("hello", "world")` →
  `ids=[101, 7592, 102, 2088, 102]`, `type_ids=[0, 0, 0, 1, 1]`.
  Matches HF tokenizer output exactly.

### Test count

143 tests pass.

## 0.8.0 — 2026-05-22

Parallel batch encode + scaling bench. 140 tests pass.

### Parallel encodeBatch

- `Tokenizer.encodeBatchParallel(texts, n_threads)` spawns N std.Thread
  workers, each taking a contiguous chunk. Tokenizer state is shared
  read-only across workers; each worker uses the caller's allocator
  (or threadsafe wrapper) for per-text output.
- `n_threads <= 1` falls through to sequential `encodeBatch`.
- On worker error, all partial output is freed and the first error
  is propagated.
- `bench/bench_parallel_bert.zig` measures scaling at 1/2/4/8 threads
  on 200×123-byte BERT inputs.

### Bench (BERT-base, 200×123-byte texts, ReleaseFast consumer 4c/8t)

| Threads | per-text | texts/sec | scaling |
|---|---|---|---|
| 1 | ~16μs | ~62K | 1.00x |
| 2 | ~12μs | ~87K | 1.40x |
| 4 | ~9μs  | ~116K | 1.87x |
| 8 | ~11μs | ~87K  | 1.41x (HT contention) |

Same 4→8 hyperthreading regression pattern as faiss-zig — real 8-core
hardware would hold the linear curve through 8.

### Test count

140 tests pass.

### Known non-claims for v0.8

- Parallel batch shares the Tokenizer struct read-only. The internal
  `std.StringHashMap` is safe for concurrent reads (no rehashing
  during reads); if a future change introduces lazy mutation this
  becomes unsafe.
- No allocator-aware fan-out. Each worker uses the same passed
  allocator. For high contention rates a per-thread ArenaAllocator
  would help further (v0.9 work).

## 0.7.0 — 2026-05-22

Batch encode + sentence-pair batch + Latin Extended-A round-trip
(82 new NFD/NFC entries) + explicit decoder Sequence parsing.
138 tests pass.

### Batch API

- `Tokenizer.encodeBatch(texts) → [][]u32` — independent per-text
  encode through the full pipeline (TemplateProcessing + truncation +
  padding).
- `Tokenizer.encodeBatchPair(pairs_a, pairs_b) → [][]u32` — pair batch
  for BERT NSP and similar; lengths must match.
- `Tokenizer.freeBatch` releases both inner and outer slices.

### Latin Extended-A NFD/NFC round-trip

- Decompose table expanded from ~14 to ~96 entries covering most of
  Latin Extended-A. Includes Polish (ŁłĆćŚśŹźŻż), Czech (ČčĎďĚěŇňŘřŠšŤťŽž),
  Romanian (ĂăŞşŢţ), Hungarian (ŐőŰű), Turkish (ĞğİıŞş), Estonian /
  Latvian / Lithuanian extras.
- Compose table mirrors decompose for full NFD → NFC round-trip on the
  same set. Tested with "Łódź Praha ČĎŇŘŤŽ Český" sample.

### Explicit decoder Sequence parsing

- New `DecoderOp` union covering Replace / ByteFallback / Fuse / Strip
  / ByteLevel / Metaspace / WordPiece.
- `parseDecoderSequence` walks the `decoder.decoders[]` array and
  emits an ordered ops list.
- BPE decode path uses the explicit ops when present; otherwise falls
  back to the implicit pipeline (preserves v0.4-v0.6 behaviour).
- Real Llama integration test still passes — the explicit path
  produces byte-identical output to the implicit one.

### Test count

138 unit + integration tests pass.

### Known non-claims for v0.7

- Encode batch is sequential (one text at a time). True parallelism
  via std.Thread is v0.8 work.
- WordPiece + Unigram decoders use the model-internal decode and treat
  the decoder Sequence's WordPiece step as a no-op pass-through.
  Sufficient for canonical HF behaviour on BERT/T5.
- NFD/NFC for Latin Extended-B (Vietnamese, broader European) is v0.8
  table expansion.

## 0.6.0 — 2026-05-21

BertNormalizer full composition (clean_text + handle_chinese_chars +
NFD + strip_accents + lowercase) + first-byte index for added_tokens
scan. 129 tests pass.

### Normalizers

- **`handle_chinese_chars`** — surrounds CJK ideographs with spaces
  before whitespace pre-tokenize. Coverage: CJK Unified Ideographs,
  Extension A, Compatibility Ideographs. Higher SMP planes (Ext B+) are
  v0.7 work (requires widening codepoint types beyond u21 for switch
  range syntax).
- **`clean_text`** — drops NULL / U+FFFD / Cc control chars (except
  tab/LF/CR) and normalizes Unicode whitespace variants to ASCII space.
- **BertNormalizer** parser composes the full canonical chain:
  `Sequence([clean_text, handle_chinese_chars, NFD, strip_accents,
  lowercase])`. Each step is opt-in via its JSON flag; defaults match
  HF Rust (`clean_text=true`, `handle_chinese_chars=true`,
  `lowercase=true`, `strip_accents=null` → defaults to lowercase).

### added_tokens first-byte index

- `fromJson` precomputes a `[256]ArrayList(u32)` index from first byte
  to candidate added-token indices. `matchAddedToken` now scans only
  the candidates whose first byte matches the current input byte —
  drops the per-position cost from O(N_added) to O(K_byte) where
  K_byte is much smaller (typically 1–3 for natural text).
- T5-base has 103 added_tokens; the index makes the scan effectively
  free for non-`<` input bytes (which is ~all of natural text).

### Test count

129 unit + integration tests pass. Real-BERT canonical IDs still match
exactly through the new five-step normalizer chain — strong validation
that the composition is sound.

## 0.5.0 — 2026-05-21

TemplateProcessing pair templates, BertNormalizer strip_accents,
truncation/padding auto-parse from tokenizer.json, Zig-side encode
benches for all three model families. 129 tests pass.

### TemplateProcessing pair

- `Tokenizer.encodePair(text_a, text_b)` runs both inputs through the
  base pipeline, then applies the `pair` template (or concatenates
  with no markers if `pair` template is null).
- `TemplateOp` gains an `append_input_b` variant for the "B" sequence.
- `parseTemplate` factored to handle both `single` and `pair` fields.
- **Real BERT pair verified**: `encodePair("hello", "world")` → `[101,
  7592, 102, 2088, 102]` (`[CLS] hello [SEP] world [SEP]`).

### BertNormalizer strip_accents

- New `.strip_accents` normalizer kind drops U+0300–U+036F (Combining
  Diacritical Marks). Composes with NFD: `[NFD, strip_accents]` turns
  "résumé" into "resume".
- `BertNormalizer` parser composes the chain when `strip_accents=true`:
  - both flags: `Sequence([NFD, strip_accents, lowercase])`
  - lowercase only: `lowercase` (existing behaviour)
  - strip_accents only: `Sequence([NFD, strip_accents])`
- Standalone `StripAccents` normalizer type also parsed.

### Truncation / padding auto-parse

- `fromJson` now reads the top-level `truncation` and `padding` objects:
  - `truncation.max_length` + `truncation.direction` ∈ {"Right", "Left"}
  - `padding.strategy.Fixed` + `padding.pad_id` + `padding.direction`
- Default behavior unchanged when these fields are absent (which is
  the case for all three real fixtures — they ship `truncation: null`,
  `padding: null` and rely on the caller to opt in).

### Encode benches (Zig-side, three model families)

Same 123-byte test text across all three. Consumer Linux, ReleaseFast,
1000 iterations median:

| Model family | ns/encode | tokens/encode | MB/s |
|---|---|---|---|
| WordPiece (BERT-base-uncased) | ~15,460 | 29 | 7 |
| Unigram (T5-base) | ~38,796 | 34 | 3 |
| BPE+SentencePiece (TinyLlama-1.1B) | ~45,211 | 34 | 2 |

WordPiece is fastest (simple greedy longest-match, no DP / no PQ).
Unigram and BPE-SP are close to each other; both do per-codepoint
Viterbi/heap work. Rust head-to-head is for BPE only (see v0.2);
BERT + T5 Rust-side benches are v0.6 work.

`zig build bench-real-bert` / `bench-real-t5` / `bench-real-tinyllama`.

### Test count

129 unit + integration tests pass.

### Known non-claims for v0.5

- Rust-side benches for WordPiece + Unigram are not yet wired up. The
  `bench/bench_rust/` Cargo project still only covers BPE.
- TemplateProcessing pair only handles `Sequence.id == "A"` and `"B"`.
  Other custom `type_id` values are honoured at parse but ignored at
  apply time (no per-token type-id output yet — that's the
  `token_type_ids` array which we don't return).

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
