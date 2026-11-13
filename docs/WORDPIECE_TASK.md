# Original WordPiece compatibility exercise

The fixture contains 32 original tokens and 48 original short inputs. It is
small enough to ship, with SHA-256 identities for the vocabulary/pipeline JSON,
corpus and frozen token IDs in tests/fixtures/original-wordpiece.meta.json.
No pretrained tokenizer dump, model weights or external corpus is included.

## Offline acceptance

```sh
zig build test --summary all
zig build -Doptimize=ReleaseSafe
python3 -m unittest discover -s tests -p test_diff_check.py
python3 tools/check_wordpiece.py
```

The checker requires every frozen input, exact successful token IDs, reference
file identities and repeated Zig output bytes. The shared comparator refuses
empty evidence, duplicate line IDs, encoding errors and malformed token IDs.
Its seven fault tests distinguish legitimate empty token sequences from an
empty dump. In the old implementation both empty dumps and matching encoding
errors returned success; the new regression record requires refusal.

## Optional live reference

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r tools/requirements-oracle.txt
.venv/bin/python tools/check_wordpiece.py --live
.venv/bin/python tools/generate_bert_punctuation.py
```

The deliberately pinned reference is huggingface/tokenizers 0.21.4, not a
claim about the newest release. It is installed as Python bindings over the
Rust implementation; no Cargo bench or model download runs. The upstream BERT
pre-tokenizer uses ASCII punctuation plus its Unicode punctuation predicate:
https://github.com/huggingface/tokenizers/blob/v0.21.4/tokenizers/src/pre_tokenizers/bert.rs

The generator observes the original a<scalar>b probe at every valid Unicode
scalar. It records 726 punctuation scalars in 156 ranges and a SHA-256 of the
ordered little-endian-u32 set. The checked-in table follows that exact reference
rather than whole Unicode blocks. To intentionally regenerate it, use --write
and review the resulting diff; normal verification writes nothing.

## Evidence boundary

The original eight-input regression matched 2/8 before the repair and 8/8
afterward. The 48-input task covers one explicit vocabulary, normalizer,
pre-tokenizer and single-input post-processor. It checks token IDs, not general
BPE/Unigram behavior, offset semantics, pair/batch APIs or every Encoding field.
The exhaustive scalar probe checks the punctuation component only. It does not
certify the separate normalization tables or future upstream versions.

Clean-source Zig acceptance retains 29 missing real-model/fuzz fixture skips.
Real-model fixture expectations and historical scores are not silently changed.
The existing AGPL terms remain intact. The owner made the repository public
during this work; this change does not alter visibility or cut a release.

## Optional pinned real-model replay

```sh
python3 tools/fetch_real_fixtures.py
zig build test --summary all
.venv/bin/python tools/check_real_parity.py --model bert
.venv/bin/python tools/check_real_parity.py --model tinyllama
.venv/bin/python tools/check_real_parity.py --model all
```

The three downloads total about 3.7 MB of tokenizer JSON, with immutable model
revisions and SHA-256 identities in tests/fixtures/real-model-pins.json. They
remain gitignored and their upstream Apache-2.0 terms remain separate. No model
weights are fetched. Existing differing files are refused rather than replaced.

With these files present, the existing suite passes 195/195, including the
previously skipped fixture/fuzz tests. Live reference replay of the existing
39-input corpus now gives BERT 39/39 and TinyLlama 39/39. T5 remains 36/39, with
mismatch lines 18, 19 and 30 (zero based). Consequently --model all returns
nonzero. This is a real remaining normalization gap, not a successful parity
claim. Existing real-fixture test expectations were not changed.
