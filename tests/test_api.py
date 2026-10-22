"""The public API (Source, Dataset, designs, discover, Result) on synthetic data.

Runs with pytest (`pytest tests`) or as a plain script (`python tests/test_api.py`).
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import multilevel_data  # noqa: E402  (puts src/ on sys.path)

import nexis  # noqa: E402
from nexis import Dataset, Discontinuity, Randomized, Source, discover  # noqa: E402
from nexis.design import LevelAwareTest, block_fe, build_rule  # noqa: E402


def _raises(fn, *words):
    try:
        fn()
    except (ValueError, TypeError) as e:
        msg = str(e)
        assert all(w in msg for w in words), f"message {msg!r} lacks {words}"
        return
    raise AssertionError(f"no error raised (expected one mentioning {words})")


class Echo:
    """A dummy interpreter: lists the raw items it is given."""

    def describe(self, top, bottom):
        return f"top {top} vs bottom {bottom}"


def village_study(during=False, pre_extra=None):
    """tests/cases.py::multilevel_data as a Dataset: rows in villages in districts,
    T randomized by village within district."""
    d = multilevel_data()
    vil, dist = d["levels"]["village"], d["levels"]["district"]
    units = pd.DataFrame({"village": vil, "district": dist, "T": d["t"]},
                         index=pd.RangeIndex(len(vil), name="row"))
    z = d["z"]
    first = pd.Series(np.arange(len(vil))).groupby(vil).first().to_numpy()
    villages = pd.DataFrame(z[first, :3], columns=["v1", "v2", "v3"],
                            index=pd.Index(np.arange(40), name="village"))
    districts = pd.DataFrame({"d1": z[pd.Series(np.arange(len(vil))).groupby(dist).first(), 3]},
                             index=pd.Index(np.arange(8), name="district"))
    rows = pd.DataFrame(z[:, 4:], columns=["r1", "r2"], index=units.index)
    outcome = pd.DataFrame({"y": d["y"]}, index=units.index)
    sources = [Source(villages, "pre", origin="learned", raw={v: f"tile_{v}" for v in range(40)},
                      interpreter=Echo()),
               Source(districts, "pre"), Source(rows, "pre", name="survey"),
               Source(outcome, "post")]
    if during:
        sources.append(Source(pd.DataFrame({"rain": np.random.default_rng(0).normal(size=8)},
                                           index=districts.index), "during"))
    if pre_extra is not None:
        sources.append(Source(pre_extra, "pre"))
    design = Randomized(treatment="T", assigned_at="village", stratified_by="district")
    return Dataset(units, sources, design), d


# ── construction and validation ──────────────────────────────────────────────

def test_source_validation():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="village"))
    _raises(lambda: Source(df, "before"), "time")
    _raises(lambda: Source(df, "pre", origin="guessed"), "origin")
    _raises(lambda: Source(df.reset_index(drop=True), "pre"), "named index")
    _raises(lambda: Source(df, "pre", origin="learned"), "raw=", "interpreter=")
    _raises(lambda: Source(df, "pre", origin="learned", raw={}, interpreter=object()), "describe")
    _raises(lambda: Source(df.assign(b=["x", "y"]), "pre"), "numeric", "b")
    _raises(lambda: Source(df.assign(b=[np.inf, 0.0]), "pre"), "infinite")
    dup = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 0], name="village"))
    _raises(lambda: Source(dup, "pre"), "repeats ids")


def test_dataset_validation():
    data, d = village_study()
    units, srcs = data.units, data.sources
    # a unit id without a row in the source
    short = srcs[0].data.iloc[1:]
    _raises(lambda: Dataset(units, [Source(short, "pre")] + srcs[1:], data.design), "no row", "village")
    # duplicate names across sources
    clash = pd.DataFrame({"v1": 0.0}, index=pd.Index(range(8), name="district"))
    _raises(lambda: Dataset(units, srcs + [Source(clash, "pre")], data.design), "'v1'")
    # a level that is not in units
    other = pd.DataFrame({"x": 0.0}, index=pd.Index(range(3), name="county"))
    _raises(lambda: Dataset(units, srcs + [Source(other, "pre")], data.design), "county")
    # treatment varying within the assignment unit
    bad = units.copy()
    bad.loc[0, "T"] = 1 - bad.loc[0, "T"]
    _raises(lambda: Dataset(bad, srcs, data.design), "varies within", "village")
    # treatment not 0/1
    _raises(lambda: Dataset(units.assign(T=units["T"] * 2), srcs, data.design), "0/1")


def test_discontinuity_checks():
    rng = np.random.default_rng(0)
    n = 200
    score = rng.uniform(0, 10, n)
    units = pd.DataFrame({"village": np.arange(n) // 10, "score": score, "T": (score <= 4.0).astype(float)},
                         index=pd.Index(np.arange(n), name="hh"))
    src = [Source(pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)}, index=units.index), "pre")]
    des = Discontinuity("T", running_variable="score", assigned_at="hh", sampled_by="village")
    data = Dataset(units, src, des)
    lo, hi = des.cutoff_interval
    assert des.direction == "<=" and lo == score[score <= 4].max() and hi == score[score > 4].min()
    assert data.checks[0].startswith("sharp: treated iff score <= cutoff")
    # a given cutoff inside the interval passes, outside fails
    Dataset(units, src, Discontinuity("T", "score", "hh", cutoff=4.0))
    _raises(lambda: Dataset(units, src, Discontinuity("T", "score", "hh", cutoff=5.0)), "cutoff 5.0", "<=")
    # treated above the cutoff: direction >=
    d2 = Discontinuity("T", "score", "hh")
    Dataset(units.assign(T=1 - units["T"]), src, d2)
    assert d2.direction == ">="
    # not sharp: the violators are named
    fuzzy = units.copy()
    fuzzy.loc[fuzzy.index[score.argmax()], "T"] = 1.0
    _raises(lambda: Dataset(fuzzy, src, Discontinuity("T", "score", "hh")), "not a sharp", f"{score.argmax()} (score=")
    # the running variable is a design variable, never a candidate: it may live in a source
    src2 = [Source(pd.DataFrame({"x": rng.normal(size=n), "pmt": score}, index=units.index), "pre"),
            Source(pd.DataFrame({"y": rng.normal(size=n)}, index=units.index), "post")]
    data2 = Dataset(units.drop(columns="score"), src2, Discontinuity("T", "pmt", "hh"))
    assert data2.removed == ["pmt"] and data2.candidates() == ["x"] and data2.outcomes == ["y"]


def test_levels_and_roles():
    data, _ = village_study(during=True)
    assert list(data.levels) == ["district", "village", "row"]
    assert data.candidates() == ["v1", "v2", "v3", "d1", "r1", "r2"]
    assert data.candidates(include_during=True)[-1] == "rain"
    assert data.outcomes == ["y"]
    assert "T" not in data.candidates()
    # broadcast: a village column is constant within villages
    assert (data.frame.groupby(data.units["village"])["v1"].nunique() == 1).all()


def test_outcome_must_be_post():
    data, _ = village_study()
    _raises(lambda: discover(data, "v1"), "time='pre'", "post")
    _raises(lambda: discover(data, "nothing"), "unknown outcome")


# ── discover equals the direct call ──────────────────────────────────────────

def test_discover_matches_direct_call():
    data, d = village_study()
    res = discover(data, "y")
    keys, parts, info = build_rule(d)
    pool = [j for j, i in enumerate(info) if i["testable"]]
    fn = LevelAwareTest(keys, parts, fe=block_fe(d["levels"]["district"])).subset(pool)
    ref = nexis.nexis(d["y"], d["t"], d["z"][:, pool], alpha=0.05, max_rounds=20, rho=0.5,
                      adjust="FWER", backward=False, terminal_filter=True, pvalue_fn=fn)
    assert res.selection.selected == ref.selected
    assert np.array_equal(res.selection.pvalues, ref.pvalues)
    assert res.modifiers == [d["names"][pool[j]] for j in ref.selected] == ["v1"]
    tab = res.table().set_index("covariate")
    assert tab.loc["v1", "status"] == "selected"
    assert tab.loc["v3", "status"].startswith("dropped: support")
    assert tab.loc["v1", "clustered_at"] == "village" and tab.loc["v1", "n_clusters"] == 40
    assert tab.loc["d1", "clustered_at"] == "district" and tab.loc["d1", "n_clusters"] == 8
    assert tab.loc["r1", "clustered_at"] == "village"      # assigned at the village
    S = list(ref.selected)
    j = pool.index(1)
    assert np.isclose(tab.loc["v2", "p_value"], fn(d["y"], d["t"], d["z"][:, pool], S, [j])[j], rtol=1e-12)


def test_during_excluded_unless_asked():
    data, _ = village_study(during=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = discover(data, "y")
    assert any("during treatment" in str(x.message) for x in w)
    tab = res.table().set_index("covariate")
    assert tab.loc["rain", "status"] == "excluded: measured during treatment"
    assert "rain" not in res.tested
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res2 = discover(data, "y", include_during=True)
    assert "rain" in res2.tested or res2.table().set_index("covariate").loc["rain", "status"].startswith("dropped")


# ── interpret and info ───────────────────────────────────────────────────────

def test_interpret():
    data, _ = village_study()
    res = discover(data, "y")
    out = res.interpret(k=2)
    v1 = data.sources[0].data["v1"].sort_values()
    want = f"top {['tile_' + str(i) for i in v1.index[::-1][:2]]} vs bottom {['tile_' + str(i) for i in v1.index[:2]]}"
    assert out == {"v1": want}


def test_interpret_nothing_learned():
    data, d = village_study()
    # the same study with the village table declared measured: nothing to interpret
    srcs = [Source(data.sources[0].data, "pre")] + data.sources[1:]
    res = discover(Dataset(data.units, srcs, data.design), "y")
    assert res.modifiers == ["v1"]
    logging.disable(logging.WARNING)
    try:
        assert res.interpret() == {}
    finally:
        logging.disable(logging.NOTSET)


def test_info_and_repr():
    import contextlib
    import io
    data, _ = village_study(during=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        data.info()
        data.design.info()
    text = buf.getvalue()
    assert "district 8, village 40, row 320" in text and "candidates during: 1" in text
    assert "passed: treatment constant within village" in text
    assert repr(data.design) == "Randomized(treatment='T', assigned_at='village', stratified_by='district')"
    assert "Dataset(320 row" in repr(data)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
