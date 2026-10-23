"""The examples run and find what they should.

Runs with pytest (`pytest tests`) or as a plain script (`python tests/test_examples.py`).
"""
import contextlib
import io
import runpy
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_quickstart():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ns = runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    assert ns["result"].modifiers == ["atom_3"]
    assert ns["interpretation"].frame["covariate"].tolist() == ["atom_3"]


if __name__ == "__main__":
    test_quickstart()
    print("ok    test_quickstart")
