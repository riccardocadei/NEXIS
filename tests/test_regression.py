"""The library's outputs on tests/cases.py must match tests/expected.json.

Runs with pytest (`pytest tests`) or as a plain script (`python tests/test_regression.py`).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import CASES  # noqa: E402

EXPECTED = json.loads(Path(__file__).with_name("expected.json").read_text())
RTOL, ATOL = 1e-12, 1e-14


def _compare(got, want, where):
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{where}: keys differ"
        for k in want:
            _compare(got[k], want[k], f"{where}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, (list, tuple)) and len(got) == len(want), f"{where}: length differs"
        for i, (g, w) in enumerate(zip(got, want)):
            _compare(g, w, f"{where}[{i}]")
    elif isinstance(want, float) and not isinstance(want, bool):
        assert np.isclose(got, want, rtol=RTOL, atol=ATOL) or (np.isnan(got) and np.isnan(want)), \
            f"{where}: {got!r} != {want!r}"
    else:
        assert got == want, f"{where}: {got!r} != {want!r}"


def _roundtrip(x):
    return json.loads(json.dumps(x))


def _make_test(name):
    def test():
        _compare(_roundtrip(CASES[name]()), EXPECTED[name], name)
    test.__name__ = f"test_{name}"
    return test


for _name in CASES:
    globals()[f"test_{_name}"] = _make_test(_name)


if __name__ == "__main__":
    failed = 0
    for _name in CASES:
        try:
            globals()[f"test_{_name}"]()
            print(f"ok    {_name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {_name}: {e}")
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)
