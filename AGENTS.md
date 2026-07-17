# tokenizers-zig Agent Entry Point

Zig 0.16 tokenizers compatible with HuggingFace `tokenizer.json`
(BPE, WordPiece, Unigram families).

## Before you change anything

1. Read `README.md` and `STATUS.md`.
2. Run the proof gate; fixture tests may skip if fixtures are absent — say so.
3. Parity claims require canonical token ID checks against the stated fixtures.

## Proof gate

```sh
zig build test --summary all
```

Do **not** run Cargo head-to-head benches under `bench/bench_rust*` as default
agent work (network + non-hermetic).

## Allowed work

- docs/status reconciliation
- tokenizer pipeline fixes with unit + fixture tests
- CLI / Encoding API improvements with tests
- CI / proof-gate maintenance
- triage of open GitHub issues when scoped

## Forbidden without explicit operator intent

- committing large HF tokenizer dumps that may be secret-bearing or huge
- release without fixture tests green
- parity or speed claims without re-running the named harness
- changing real-tokenizer fixture expectations silently

## Escalation

Ask before release claims or changing real-tokenizer fixture expectations.

## Hygiene

Org templates: `SMC17/stax-hygiene`. Open issue backlog should stay labeled.
