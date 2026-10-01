#!/usr/bin/env python3
"""
sync_gpt_dev.py

Prepares `train_gpt_dev.py` to be a valid `train_gpt.py`.

Two mechanisms:

1. Any `# ===== FROZEN: ... =====` / `# ===== END FROZEN =====` span is always
   taken verbatim from `train_gpt.py`.

2. Any `# >>> sync:<label>` / `# <<< sync` block must be a plain
   `if TRAIN_DEVICE == "mps": ... else: ...` statement. Everything in the
   `else:` branch is taken verbatim (dedented) as what train_gpt.py should
   contain there; the `if` branch is dev-only (e.g. gloo/mps setup) and is
   discarded. A `# >>> devonly` / `# <<< devonly` span is dropped outright
   (e.g. this file's own explanatory comments, the local self-launch helper).

Everything else outside those kinds of span is copied from `train_gpt_dev.py`.

Usage: python sync_gpt_dev.py [--check]
  --check   only print the diff / validate structure, do not write train_gpt.py
"""
import re
import sys
import difflib
from pathlib import Path

DEV_FILE = Path("train_gpt_dev.py")
PROD_FILE = Path("train_gpt.py")

FROZEN_RE = re.compile(
    r"^[ \t]*# ===== FROZEN:(?P<label>.*?)=====\n"
    r".*?"
    r"^[ \t]*# ===== END FROZEN =====\n",
    re.DOTALL | re.MULTILINE,
)

SYNC_RE = re.compile(
    r"^(?P<indent>[ \t]*)# >>> sync:(?P<label>\S+)\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)# <<< sync\n",
    re.DOTALL | re.MULTILINE,
)

DEVONLY_RE = re.compile(
    r"^[ \t]*# >>> devonly\n"
    r".*?"
    r"^[ \t]*# <<< devonly\n",
    re.DOTALL | re.MULTILINE,
)


def collapse_sync_block(match: "re.Match") -> str:
    """Replace an `if TRAIN_DEVICE == "mps": ... else: ...` sync block with
    just its else-branch body, dedented back to the if/else's own indent level."""
    label, body = match["label"], match["body"]
    lines = body.splitlines(keepends=True)
    if not lines or "if TRAIN_DEVICE" not in lines[0]:
        sys.exit(f"error: sync block {label!r} must open with an `if TRAIN_DEVICE == \"mps\":` line")
    base_indent = len(lines[0]) - len(lines[0].lstrip(" "))
    else_prefix = " " * base_indent + "else:"
    try:
        else_idx = next(i for i, line in enumerate(lines) if line.startswith(else_prefix))
    except StopIteration:
        sys.exit(f"error: sync block {label!r} has no top-level `else:` branch")
    else_body = lines[else_idx + 1:]
    dedent = 4  # one indent level: else-branch body sits one level deeper than `else:` itself
    out = []
    for line in else_body:
        if line.strip() == "":
            out.append(line)
        else:
            if not line.startswith(" " * dedent):
                sys.exit(f"error: sync block {label!r} else-branch has unexpected indentation")
            out.append(line[dedent:])
    return "".join(out)


def frozen_labels(src: str) -> list[str]:
    return [m["label"].strip() for m in FROZEN_RE.finditer(src)]


def build_prod_source(dev_src: str, prod_src: str) -> str:
    dev_labels = frozen_labels(dev_src)
    prod_labels = frozen_labels(prod_src)
    if dev_labels != prod_labels:
        sys.exit(
            "error: FROZEN block structure differs between train_gpt_dev.py and "
            f"train_gpt.py.\n  dev:  {dev_labels}\n  prod: {prod_labels}\n"
            "Did a FROZEN marker get added, removed, or reordered in the dev file?"
        )

    collapsed_dev = DEVONLY_RE.sub("", dev_src)
    collapsed_dev = SYNC_RE.sub(collapse_sync_block, collapsed_dev)
    leftover_marker = re.search(
        r"^[ \t]*# (>>> sync:|<<< sync|>>> devonly|<<< devonly)", collapsed_dev, re.MULTILINE)
    if leftover_marker:
        sys.exit("error: unresolved marker survived collapse")

    # Walk both the collapsed-dev source and the prod source in lockstep,
    # taking FROZEN spans from prod and everything else from collapsed-dev.
    prod_frozen_spans = list(FROZEN_RE.finditer(prod_src))
    dev_frozen_spans = list(FROZEN_RE.finditer(collapsed_dev))
    if len(prod_frozen_spans) != len(dev_frozen_spans):
        sys.exit("error: FROZEN block count mismatch after sync-marker collapse")

    out = []
    dev_pos = 0
    for prod_m, dev_m in zip(prod_frozen_spans, dev_frozen_spans):
        out.append(collapsed_dev[dev_pos:dev_m.start()])  # editable text, from dev
        out.append(prod_src[prod_m.start():prod_m.end()])  # frozen text, from prod
        dev_pos = dev_m.end()
    out.append(collapsed_dev[dev_pos:])
    return "".join(out)


def main():
    check_only = "--check" in sys.argv
    dev_src = DEV_FILE.read_text()
    prod_src = PROD_FILE.read_text()
    new_prod_src = build_prod_source(dev_src, prod_src)

    diff = list(difflib.unified_diff(
        prod_src.splitlines(keepends=True),
        new_prod_src.splitlines(keepends=True),
        fromfile=str(PROD_FILE), tofile=f"{PROD_FILE} (from {DEV_FILE})",
    ))
    if diff:
        sys.stdout.writelines(diff)
    else:
        print(f"no changes: {PROD_FILE} already matches {DEV_FILE}")

    if check_only:
        return
    if diff:
        PROD_FILE.write_text(new_prod_src)
        print(f"\nwrote {PROD_FILE}")


if __name__ == "__main__":
    main()
