# Contributing

Pre-1.0 substrate. Interface and storage format may change between
minor versions without ceremony.

## AI codegen policy

External contributors **must not** submit AI-authored patches, issues,
commit messages, or comments. This is a trust-bound rule mirroring the
upstream Zig project's posture, scoped to this repo for the same reason:
the maintainer does not trust AI codegen from contributors whose review
discipline is unknown.

The maintainer reserves the right to use AI codegen internally on their
own substrate. That is a separate decision and is not transitive.

If your patch is AI-assisted, do not open the PR. Hand-author it first.

## Filing

- Bugs: include a minimal repro, expected vs actual output, and Zig
  version (`zig version`).
- Features: open an issue with the use case before writing code. Patches
  for unscoped features will be declined without prejudice.

## Style

- Match existing formatting (`zig fmt`).
- One logical change per commit.
- Commit messages: imperative mood, short subject line, optional body.
- Tests required for behavior changes.
