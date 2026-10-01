#!/usr/bin/env bash
# Runs the real `uv run python sync_gpt_dev.py` against the real train_gpt_dev.py
# and train_gpt.py, in an isolated temp copy (so this never touches your working
# tree), then checks that every FROZEN block in the result still matches
# scripts/frozen_baseline.py -- the pinned, known-good FROZEN content. Catches a
# sync_gpt_dev.py regression, and relies on train_gpt_dev.py's existing FROZEN-block
# MPS tampering (the dataloader's device hack) to prove content is taken from
# train_gpt.py, not train_gpt_dev.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR_SYNC="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_SYNC"' EXIT

cp "$REPO_ROOT/train_gpt_dev.py" "$TMPDIR_SYNC/train_gpt_dev.py"
cp "$REPO_ROOT/train_gpt.py" "$TMPDIR_SYNC/train_gpt.py"
cp "$REPO_ROOT/sync_gpt_dev.py" "$TMPDIR_SYNC/sync_gpt_dev.py"

(cd "$TMPDIR_SYNC" && uv run --project "$REPO_ROOT" python sync_gpt_dev.py)

if diff -u "$REPO_ROOT/scripts/frozen_baseline.py" \
    <(python3 "$REPO_ROOT/scripts/extract_frozen.py" "$TMPDIR_SYNC/train_gpt.py"); then
    echo "OK: every FROZEN block matches scripts/frozen_baseline.py"
else
    echo "FAIL: a FROZEN block changed (above)." >&2
    echo "If this is an intentional, deliberate change to FROZEN content:" >&2
    echo "  python3 scripts/extract_frozen.py train_gpt.py > scripts/frozen_baseline.py" >&2
    exit 1
fi
