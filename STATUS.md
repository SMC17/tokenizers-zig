# STATUS

**Version:** v0.23
**Last update:** 2026-05-22

## What landed in v0.23 (on top of v0.22)

- **`Model.idToToken(id)`** per-variant lookup.
- **`Tokenizer.idToToken(id)`** with added_tokens check.
- **`Tokenizer.tokens(ids)`** parallel array of vocab strings.
- **`Tokenizer.specialTokensMask(ids)`** mask of added-token positions.

Canonical HF return surface now complete: ids + type_ids +
attention_mask + offsets + sequence_ids + tokens + special_tokens_mask.

184 tests pass.

## What landed in v0.22 (on top of v0.21)

- **`encodePairWithOffsets(a, b)`** — pair encode returning ids +
  offsets + sequence_ids (which source each token came from). Closes
  the BERT NSP + NER use case.

183 tests pass.

## What landed in v0.21 (on top of v0.20)

- **`encodeBatchWithOffsets(texts)`** — batch encode returning
  per-text ids + offsets.
- **`encodeBatchParallelWithOffsets(texts, n_threads)`** — std.Thread
  fan-out; falls through to sequential when `n_threads <= 1`.
- Parallel output verified byte-identical to sequential in tests.

182 tests pass.

## What landed in v0.20 (on top of v0.19)

- **Real-fixture sub-token offset tests** across BERT-base-uncased,
  T5-base, and TinyLlama-1.1B. Each test verifies that
  `encodeWithOffsets` on a real downloaded tokenizer produces byte
  ranges that map back to the original input correctly.
  - BERT: "hello, world!" → 4 tokens with exact byte spans
  - T5: "hello world" → 2 tokens with Metaspace-corrected spans
  - TinyLlama: "Hello world" → monotonic byte spans through the
    Llama SP-marker normalizer chain

181 tests pass.

## What landed in v0.19 (on top of v0.18)

- **`normalizer.applyWithOffsets`** — output + input_index parallel
  arrays, with `SYNTHETIC` sentinel for introduced bytes. Fully
  supports `Prepend / Replace / Sequence / Lowercase` (the Llama chain).
- **`Tokenizer.encodeWithOffsets` BPE+SP marker path** — Llama / Mistral
  pipelines now produce sub-token byte ranges via normalizer-offsets →
  BPE-offsets → remap.

### Sub-token offset coverage matrix (now complete across all 4 pipelines)

| Pipeline | Sub-token offsets |
|---|---|
| WordPiece + BertPreTokenizer | ✓ |
| BPE + ByteLevel (GPT-2/Qwen/RoBERTa) | ✓ |
| Unigram + Metaspace (T5/mBART) | ✓ |
| **BPE + SP marker (Llama/Mistral)** | **✓ (v0.19)** |

178 tests pass.

## What landed in v0.18 (on top of v0.17)

- **`unigram.encodeWithOffsets`** — Viterbi returns ids + per-segment
  byte ranges.
- **`Tokenizer.encodeWithOffsets` Unigram+Metaspace path** — true
  sub-token byte ranges through whitespace-split → metaspace-prepend →
  Viterbi → metaspace-prefix-subtract pipeline.
- **All three families now produce sub-token offsets** (WordPiece v0.16,
  BPE+ByteLevel v0.17, Unigram+Metaspace v0.18). BPE+SP marker (Llama)
  remains chunk-level — v0.19 work.

176 tests pass.

## What landed in v0.17 (on top of v0.16)

- **`byte_level.encodeWithOffsets`** — parallel `input_index` table
  for output-byte → input-byte mapping.
- **`bpe.encodeWithOffsets`** — PQ-based merge tracks `(start, end)`
  byte ranges through DLL nodes; merged ranges absorb the right's end.
- **`Tokenizer.encodeWithOffsets` BPE+ByteLevel path** — true sub-token
  byte ranges in the original input via the byte_level remap chain.
  GPT-2/Qwen/RoBERTa pipelines now produce per-token NER-grade offsets.

173 tests pass.

## What landed in v0.16 (on top of v0.15)

- **True sub-word WordPiece offsets**. `wordpiece.Model.encodeWithOffsets`
  emits per-piece byte ranges; `Tokenizer.encodeWithOffsets` uses them.
  `unaffable` → `un=[0,2]`, `##aff=[2,5]`, `##able=[5,9]`.
- NER-grade precision on BERT pipelines.

167 tests pass.

## What landed in v0.15 (on top of v0.14)

- **`encodeWithOffsets(text) → EncodeWithOffsetsResult`** returns
  ids + per-token byte offsets in the original text.
- WordPiece + whitespace_split: word-level offset granularity.
- Added tokens: exact match offsets.
- Other pipelines: full chunk range.
- Word-level (not sub-word) — true sub-word offsets are future work.

165 tests pass.

## What landed in v0.14 (on top of v0.13)

- **`tokenize` CLI** with six subcommands: encode / encode-base /
  encode-pair / encode-full / decode / info.
- All subcommands exercise the public API end-to-end.
- Verified on real BERT / T5 / Llama fixtures.

163 tests pass.

## What landed in v0.13 (on top of v0.12)

- **`encodeBatchWithTypes(texts)`** and **`encodeBatchPairWithTypes(a, b)`**
  return `[]EncodeResult` with ids + type_ids + attention_mask per row.
- `freeBatchWithTypes` helper.

163 tests pass.

## What landed in v0.12 (on top of v0.11)

- **Hangul Jamo algorithmic NFD/NFC** — Unicode Standard Annex #15
  algorithm; no table lookup. `한국어 각` round-trips.
- **Greek monotonic** — Ά Έ Ή Ί Ό Ύ Ώ + lowercase + diaeresis combos.
- **Cyrillic** — Ё ё Ѝ ѝ Ў ў Й й.
- **u3 overflow fix** in NFC Hangul LVT branch (widens to usize).

Round-trip verified: `"Άθήνα 한국어 résumé"` decomposes through partial
NFD and reassembles byte-identical via NFC.

161 tests pass.

## What landed in v0.11 (on top of v0.10)

- **`decodeBatch` + `decodeBatchParallel`** — sequential + parallel
  batch decode, mirrors encode-side. 147 tests pass.
- **Rust-side benches for BERT + T5** — new Cargo projects exercise
  `huggingface/tokenizers` 0.21 on the same fixtures.

Head-to-head across all three families:

| Family | Zig | Rust HF 0.21 | Ratio |
|---|---|---|---|
| BPE+SP (TinyLlama) | ~45μs | ~46μs | ~1.0x |
| **WordPiece (BERT)** | **~15μs** | **~82μs** | **~5.3x Zig faster** |
| Unigram (T5) | ~38μs | ~44μs | ~1.16x Zig faster |

WordPiece 5.3x ratio is the headline. Pure-Zig greedy-longest-match
with codepoint-aware boundaries + first-byte added-token index
outpaces HF Rust's WordPiece by ~5x on consumer hardware.

## What landed in v0.10 (on top of v0.9)

- **attention_mask return** completes the canonical HF EncodeResult
  surface. `ids + type_ids + attention_mask` parallel arrays, in
  lockstep through truncation + padding.
- Padding rows are now 0 across attention_mask (downstream models
  use this to ignore them during attention).

145 tests pass.

## What landed in v0.9 (on top of v0.8)

- **Token type IDs return** — new `EncodeResult { ids, type_ids }`
  with `encodeWithTypes` and `encodePairWithTypes` APIs.
- `TemplateOp` refactored to a `{kind, type_id}` struct; parser reads
  `type_id` from each template entry. BERT pair encode now produces
  canonical `token_type_ids = [0, 0, 0, 1, 1]` on `("hello", "world")`,
  matching HF exactly.
- Truncation + padding apply symmetrically to both arrays.

143 tests pass.

## What landed in v0.8 (on top of v0.7)

- **`encodeBatchParallel(texts, n_threads)`** — std.Thread-based
  parallel batch encode. Tokenizer state shared read-only across
  workers. Falls through to sequential when `n_threads <= 1`.
- **`bench/bench_parallel_bert.zig`** — scaling bench at 1/2/4/8
  threads on 200×123-byte BERT inputs. Real scaling observed: 1.87x
  speedup at 4 threads (4c/8t consumer CPU); HT contention regression
  past 4.

140 tests pass.

## What landed in v0.7 (on top of v0.6)

- **Batch APIs**: `encodeBatch(texts)` and `encodeBatchPair(a, b)`
  for HF-style batch processing.
- **Latin Extended-A NFD/NFC round-trip** — decompose + compose
  tables expanded from ~14 to ~96 entries. Polish, Czech, Romanian,
  Hungarian, Turkish, Baltic Latin all round-trip cleanly.
- **Explicit decoder Sequence parsing** — `DecoderOp` union covers
  Replace / ByteFallback / Fuse / Strip / ByteLevel / Metaspace /
  WordPiece. BPE decode uses the parsed sequence when present.
  Real Llama integration test still validates the explicit path
  produces canonical output.

138 unit + integration tests pass.

## What landed in v0.6 (on top of v0.5)

- **`handle_chinese_chars`** normalizer — wraps CJK ideographs in
  spaces. Coverage: CJK Unified Ideographs, Extension A, Compatibility
  Ideographs (basic plane).
- **`clean_text`** normalizer — drops control chars + replaces Unicode
  whitespace variants with ASCII space.
- **BertNormalizer** parses the full HF chain
  `Sequence([clean_text, handle_chinese_chars, NFD, strip_accents,
  lowercase])` based on individual flags.
- **First-byte index for added_tokens** — encode scan goes from
  O(N_added) per position to O(K_byte). Effectively free for natural
  text (no special-token characters). Helps T5 (103 added_tokens) and
  any multilingual tokenizer with large sentinel sets.

129 unit + integration tests pass; real-BERT canonical IDs validate
the composed normalizer chain.

## What landed in v0.5 (on top of v0.4)

- **TemplateProcessing pair templates** — `encodePair(a, b)` produces
  HF-canonical wrapped output (verified `[CLS] hello [SEP] world [SEP]`
  on real BERT).
- **BertNormalizer.strip_accents** — new `.strip_accents` normalizer
  drops Combining Diacritical Marks. Composes with NFD to turn
  "résumé" into "resume". `BertNormalizer` parser auto-composes
  `Sequence([NFD, strip_accents, lowercase])` when both flags set.
- **Truncation / padding auto-parse** from top-level `truncation` /
  `padding` objects in tokenizer.json. Defaults preserved for
  fixtures that ship null (the common case).
- **BERT + T5 Zig encode benches** — `bench-real-bert` (15μs / 29
  tokens) and `bench-real-t5` (38μs / 34 tokens) on the standard
  123-byte fixture.

129 unit + integration tests pass.

## What landed in v0.4 (on top of v0.3)

- **TemplateProcessing post-processor** with single-template support.
  TinyLlama (BOS), BERT (`[CLS]/[SEP]`), T5 (EOS) all produce canonical
  HF-wrapped IDs through `Tokenizer.encode`. Raw IDs via
  `Tokenizer.encodeBase`.
- **ByteFallback decoder** — `<0xXX>` token strings expand to literal
  bytes in BPE decode (Llama family).
- **Truncation + padding** — `max_length` / `pad_to_length` /
  direction fields, applied post-TemplateProcessing.
- **Unicode punctuation classifier** — ranges-based, covers ASCII +
  Latin-1 + General Punctuation + CJK Symbols + Halfwidth/Fullwidth
  punctuation subset.
- **Partial Unicode normalization** (`src/unicode_norm.zig`) — NFD/NFC/
  NFKD/NFKC over Latin-1 Supplement + Latin Extended-A. ~70 codepoints.
- **GPT-2 regex pre-tokenize** (`src/gpt2_regex.zig`) — hand-coded
  scanner for the canonical GPT-2 word-boundary pattern. Activated by
  ByteLevel `use_regex: true`. ASCII + Latin-1 letter coverage.

127 unit + integration tests pass.

## What landed in v0.3 (on top of v0.2)

- **BertNormalizer** parser → `.lowercase` normalizer (ASCII).
  Defaults to lowercase=true matching HF behaviour.
- **BertPreTokenizer** with ASCII punctuation splitting — each
  punctuation char becomes its own WordPiece word.
- **Model-type inference** when missing (older HF format). BPE if
  `merges`, WordPiece if `continuing_subword_prefix`, Unigram if
  `vocab` is an array.
- **Per-piece Metaspace** for Unigram + WhitespaceSplit + Metaspace
  Sequence (T5 pattern). Splits BEFORE Metaspace.
- **`tests/test_real_bert.zig`** — 7 tests against real
  bert-base-uncased (30522 vocab, WordPiece). Canonical IDs verified:
  hello=7592, [CLS]=101, [SEP]=102, etc.
- **`tests/test_real_t5.zig`** — 5 tests against real t5-base (32100
  vocab, Unigram). Canonical IDs verified: ▁hello=21820, <pad>=0,
  </s>=1, <extra_id_0>=32099.

78 unit + integration tests pass.

## What landed in v0.2 (on top of v0.1)

### Models

- **`src/wordpiece.zig`** — WordPiece model (BERT / DistilBERT family).
  Greedy longest-match with codepoint-aware boundaries and `##`
  continuation prefix. 11 unit tests.
- **`src/unigram.zig`** — Unigram model (T5 / mBART / XLM-R,
  SentencePiece-style). Viterbi DP over `(token, log-score)` vocab.
  11 unit tests.
- **`Tokenizer.model`** is now a tagged union over `{bpe, wordpiece,
  unigram}`. `fromJson` dispatches on `model.type`.

### Pipeline

- **Whitespace / WhitespaceSplit / BertPreTokenizer** pre-tokenizers —
  whitespace-tokenize for WordPiece word-by-word encoding.
- **Metaspace** pre-tokenizer — SentencePiece "▁" replacement with
  reverse on decode.
- **`added_tokens` end-to-end** — parsed from top-level
  `added_tokens` array; encode runs longest-match scan over input
  (emit id verbatim, chunk remainder through model); decode emits
  content verbatim while routing non-added runs through the model.

### Performance

- **Priority-queue BPE merge** documented as the production path
  (the `bpeEncodePQ` doubly-linked-list + min-heap was merged into
  source in v0.1 but flagged as "v0.2 path" in docs).
- Stale-entry detection: dead-node sentinel + piece-identity snapshot
  + rank re-lookup at pop time.
- ~10x faster than v0.1's O(n²) loop on the TinyLlama bench;
  gap-to-Rust closed from ~9x to ~1.2x.

## Bench (v0.2)

Same fixture and methodology as v0.1. Output IDs verified equal to HF
Rust reference.

| Implementation | ns/encode (123 B, 33 tokens) | ratio |
|---|---|---|
| `huggingface/tokenizers` 0.21 (Rust) | ~46,000 | 1.00x |
| **tokenizers-zig v0.2** (Zig, PQ + DLL) | ~30,000–56,000 | **≤1.2x slower** |
| tokenizers-zig v0.1 (Zig, O(n²) greedy) | ~491,000 | ~9x slower |

Consumer Linux, ReleaseFast. Variance across machines and runs.

## Test count

64 unit tests + 3 real-TinyLlama integration tests pass.

## What's NOT covered (v0.3+)

1. **BertPreTokenizer punctuation splitting** — v0.2 treats Bert /
   Whitespace / WhitespaceSplit identically as whitespace-only
   tokenize. Real BERT pre-tokenization also splits on punctuation and
   applies the BertNormalizer chain.
2. **NFC / NFD / NFKC / NFKD Unicode normalization** — required by
   some real Unigram tokenizers (NFKC pass before Viterbi).
3. **Regex / Split pre-tokenizers** — GPT-2 word-boundary regex is
   not implemented. Modern Llama/Qwen tokenizers don't need it
   (ByteLevel handles spaces via Ġ-mapping); raw GPT-2 / RoBERTa
   may diverge from HF on long mixed-whitespace inputs.
4. **Special-token attribute handling** — `special`, `single_word`,
   `lstrip`, `rstrip`, `normalized` flags on `added_tokens` are
   parsed-but-ignored. Every added_token entry emits verbatim.
5. **Real BERT-base / T5-base integration tests** — v0.2 ships
   synthetic + small BERT-shaped / T5-shaped tokenizer.json snippets
   only. v0.3 will exercise downloaded real fixtures.
6. **TemplateProcessing post-processor** (BOS/EOS template).
7. **ByteFallback decoder** (handles `<0xXX>` tokens in some Llama
   outputs).
8. **Truncation / padding / template post-processors**.

## Honest non-claims

- Rust upstream is ≤1.2x faster on the real-tokenizer bench. Same
  fixture, same correctness. Published as README headline.
- WordPiece and Unigram pass unit + synthetic-tokenizer.json tests but
  no real BERT-base / T5-base tokenizer.json has been exercised yet.
- Pre-1.0 substrate per Zig 0.16 ecosystem convention. No
  "production-grade" / "v1.0" / "stable" claims.
- Tested on x86_64-linux only.

## Position in portfolio

tokenizers-zig pairs with safetensors-zig (artifact #4a) as the HF
micro-library siblings. Together they make vllm-zig (artifact #1)
viable end-to-end: tokenizers-zig text→IDs, safetensors-zig weight
loading, vllm-zig forward pass.
