"""Prevent accidental reuse of checkpoints trained under another protocol."""
import json
from pathlib import Path

TRAINING_REVISION = "mtf_sidehead_lr_periodic_v1_20261030"


def guard_output_root(root, protocol):
    root = Path(root)
    path = root / "protocol.json"
    canonical = json.loads(json.dumps(protocol))
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(k for k in set(previous) | set(canonical)
                         if previous.get(k) != canonical.get(k))
        if changed:
            raise RuntimeError(f"Output protocol differs in {changed}; use a new output-root: {root}")
    elif root.exists() and any(root.rglob("*.pth")):
        raise RuntimeError(f"Unverified checkpoints without protocol.json; use a new output-root: {root}")
