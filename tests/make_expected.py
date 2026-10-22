"""Freeze the current outputs of tests/cases.py in tests/expected.json.

Run only when a change is meant to move the library's numbers:
    python tests/make_expected.py
"""
import json
from pathlib import Path

from cases import CASES

out = {name: fn() for name, fn in CASES.items()}
path = Path(__file__).with_name("expected.json")
path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(f"wrote {path} ({len(out)} cases)")
