#!/usr/bin/env python3
"""Prints every FROZEN block (including its markers) from a file, concatenated.

Used by check_sync_fixture.sh to verify that sync_gpt_dev.py's output has FROZEN
content identical to the current train_gpt.py, without requiring the rest of the
generated file to match some full-file golden copy. Reuses sync_gpt_dev.py's own
FROZEN_RE so this never drifts from what the real tool considers "frozen".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sync_gpt_dev import FROZEN_RE

text = Path(sys.argv[1]).read_text()
for match in FROZEN_RE.finditer(text):
    print(match.group(0), end="")
