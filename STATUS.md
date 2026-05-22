# STATUS

**Version:** v0.3
**Last update:** 2026-05-21

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
