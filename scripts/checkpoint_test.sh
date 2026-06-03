#!/usr/bin/env bash
# Smoke test: checkpoint.sh runs cleanly + produces well-formed NEXT.md.
set -euo pipefail

# Save existing NEXT.md if present so we don't clobber.
BACKUP=""
if [[ -f NEXT.md ]]; then BACKUP=$(mktemp); cp NEXT.md "$BACKUP"; fi
trap '[[ -n "$BACKUP" ]] && mv "$BACKUP" NEXT.md' EXIT

# Run dry-run; expect non-empty stdout with the headline + sections.
out=$(./scripts/checkpoint.sh --dry-run)
if ! echo "$out" | grep -q '^# NEXT —'; then
  echo "FAIL: missing headline"; exit 1
fi
if ! echo "$out" | grep -q '## What just shipped'; then
  echo "FAIL: missing 'What just shipped' section"; exit 1
fi
if ! echo "$out" | grep -q '## Working-tree state'; then
  echo "FAIL: missing 'Working-tree state' section"; exit 1
fi
if ! echo "$out" | grep -q '## Open hypotheses'; then
  echo "FAIL: missing 'Open hypotheses' section"; exit 1
fi
if ! echo "$out" | grep -q '## Next concrete action'; then
  echo "FAIL: missing 'Next concrete action' section"; exit 1
fi

# Run actual (write); confirm NEXT.md exists + has > 5 lines.
./scripts/checkpoint.sh
n_lines=$(wc -l < NEXT.md)
if [[ "$n_lines" -lt 10 ]]; then
  echo "FAIL: NEXT.md too short ($n_lines lines)"; exit 1
fi

echo "PASS: checkpoint.sh produces well-formed NEXT.md ($n_lines lines)"
