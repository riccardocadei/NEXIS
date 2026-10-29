"""The public API (Source, Dataset, designs, discover, Result) on synthetic data.

Runs with pytest (`pytest tests`) or as a plain script (`python tests/test_api.py`).
"""
import contextlib
import io
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


def _warns(fn, *words):
    """Run fn, assert a warning mentions all `words`, return fn's result."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    msgs = [str(x.message) for x in w]
    assert any(all(word in m for word in words) for m in msgs), f"no warning with {words} in {msgs}"
    return out


def _quiet(fn):
    """Run fn and fail on any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


def _printed(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


class Echo:
    """A dummy interpreter: lists the raw items it is given."""

    def describe(self, top, bottom):
        return f"top {top} vs bottom {bottom}"


def village_study(during=False, pre_extra=None, y=None):
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
    outcome = pd.DataFrame({"y": d["y"] if y is None else y}, index=units.index)
    sources = [Source(villages, level="village", time="pre", support={"v1": "continuous", "v2": "continuous", "v3": "binary"}, origin="learned",
                      raw={v: f"tile_{v}" for v in range(40)}, interpreter=Echo()),
               Source(districts, level="district", time="pre", support="continuous"),
               Source(rows, level="row", time="pre", support="continuous", name="survey"),
               Source(outcome, level="row", time="post", support="continuous")]
    if during:
        sources.append(Source(pd.DataFrame({"rain": np.random.default_rng(0).normal(size=8)},
                                           index=districts.index), level="district", time="during",
                              support="continuous"))
    if pre_extra is not None:
        sources.append(pre_extra)
    design = Randomized(treatment="T", assigned_at="village", stratified_by="district")
    return Dataset(units, sources, design), d


def rdd_units(n=200, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 10, n)
    units = pd.DataFrame({"village": np.arange(n) // 5, "score": score, "T": (score <= 4.0).astype(float)},
                         index=pd.Index(np.arange(n), name="hh"))
    src = [Source(pd.DataFrame({"x": rng.normal(size=n)}, index=units.index), level="hh", time="pre",
                  support="continuous"),
           Source(pd.DataFrame({"y": rng.normal(size=n)}, index=units.index), level="hh", time="post",
                  support="continuous")]
    return units, src, score


# ── construction and validation ──────────────────────────────────────────────

def test_source_validation():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="village"))
    kw = dict(level="village", support="continuous")
    _raises(lambda: Source(df, time="before", **kw), "time")
    _raises(lambda: Source(df, time="pre", origin="guessed", **kw), "origin")
    _raises(lambda: Source(df, time="pre", origin="learned", **kw), "raw=", "interpreter=")
    _raises(lambda: Source(df, time="pre", origin="learned", raw={}, interpreter=object(), **kw), "describe")
    _raises(lambda: Source(df.assign(b=["x", "y"]), time="pre", **kw), "numeric", "b")
    _raises(lambda: Source(df.assign(b=[np.inf, 0.0]), time="pre", **kw), "infinite")
    dup = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 0], name="village"))
    _raises(lambda: Source(dup, time="pre", **kw), "repeats ids")
    nan_id = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0.0, np.nan], name="village"))
    _raises(lambda: Source(nan_id, time="pre", **kw), "1 missing ids")
    # level, time and support are required keywords
    _raises(lambda: Source(df, "pre"), "positional")
    _raises(lambda: Source(df, level="village", time="pre"), "support")
    _raises(lambda: Source(df, time="pre", support="continuous"), "level")


def test_source_level():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="village"))
    # an index named otherwise is an error; an unnamed one is named after the level
    _raises(lambda: Source(df, level="district", time="pre", support="continuous"),
            "indexed by 'village'", "level='district'")
    s = Source(df.reset_index(drop=True), level="district", time="pre", support="continuous")
    assert s.level == "district" and s.data.index.name == "district" and s.name == "district_pre"


def test_source_support():
    idx = pd.Index(range(6), name="v")
    ok = {"binary": [0, 1, 1, np.nan, 0, 0], "count": [0, 3, 7, 2, 0, np.nan],
          "nonnegative": [0.0, 0.5, 2.3, 9.0, 0.1, 1.0], "sparse": [0, 0, 0.7, 0, 2.5, np.nan],
          "continuous": [-1.5, 0.0, 2.0, 3.3, -7.0, np.nan]}
    for kind, vals in ok.items():                          # NaN is ignored
        Source(pd.DataFrame({"x": vals}, index=idx), level="v", time="pre", support=kind)
    bad = {"binary": ([0, 1, 2, 0.5, 0, 1], "2 values", "0.5"), "count": ([0, 1, 2.5, -1, 0, 0], "2 values", "-1.0"),
           "nonnegative": ([0, 1, -0.2, 3, 4, 5], "1 values", "-0.2"),
           "sparse": ([0, 1, 1, 1, 0, 2], "zero (2 rows)", "most frequent"),
           }
    for kind, (vals, *words) in bad.items():
        _raises(lambda: Source(pd.DataFrame({"x": vals}, index=idx), level="v", time="pre", support=kind),
                "'x'", f"declared {kind}", *words)
    _raises(lambda: Source(pd.DataFrame({"x": [-1.0] * 6}, index=idx), level="v", time="pre",
                           support="sparse"), "declared sparse", "-1.0")
    _raises(lambda: Source(pd.DataFrame({"x": [1.0] * 6}, index=idx), level="v", time="pre",
                           support="real"), "support of 'x'", "'real'")
    # one kind per column: a dict covering every column exactly
    two = pd.DataFrame({"a": [0, 1] * 3, "b": np.arange(6.0) - 2}, index=idx)
    s = Source(two, level="v", time="pre", support={"b": "continuous", "a": "binary"})
    assert s.support == {"a": "binary", "b": "continuous"}
    _raises(lambda: Source(two, level="v", time="pre", support={"a": "binary"}), "every column", "['b']")
    _raises(lambda: Source(two, level="v", time="pre", support={"a": "binary", "b": "continuous", "c": "count"}),
            "every column", "['c']")
    _raises(lambda: Source(two, level="v", time="pre", support="binary"), "'b'", "declared binary")


def test_declared_support_rule():
    """The support rule applies to columns declared binary, count or sparse even without
    a data mass point (the union with the data rule); declared continuous columns with a
    mass point get it from the data, with a warning."""
    rng = np.random.default_rng(5)
    # sparse village atom: zero is its most frequent value but holds only 3 of 40 villages
    atom = rng.exponential(size=40)
    atom[[0, 1, 2]] = 0.0
    extra = pd.DataFrame({"atom": atom}, index=pd.Index(np.arange(40), name="village"))
    sparse, _ = village_study(pre_extra=Source(extra, level="village", time="pre", support="sparse"))
    nonneg, _ = village_study(pre_extra=Source(extra, level="village", time="pre", support="nonnegative"))
    t_sp = discover(sparse, "y").table().set_index("covariate")
    t_nn = discover(nonneg, "y").table().set_index("covariate")
    assert t_sp.loc["atom", "kind"] == "sparse" and t_nn.loc["atom", "kind"] == "nonnegative"
    assert pd.isna(t_nn.loc["atom", "support"]) and not t_nn.loc["atom", "status"].startswith("dropped")
    assert t_sp.loc["atom", "support"] <= 3
    assert t_sp.loc["atom", "status"].startswith("dropped: support")
    # every other candidate is unchanged
    assert t_sp.drop(index="atom").drop(columns="p_value").equals(t_nn.drop(index="atom").drop(columns="p_value"))
    # declared continuous with a mass point: the data rule applies, with a warning
    dummy = pd.DataFrame({"dummy": (np.arange(40) % 3 == 0).astype(float)},
                         index=pd.Index(np.arange(40), name="village"))
    data = _warns(lambda: village_study(pre_extra=Source(dummy, level="village", time="pre",
                                                         support="continuous"))[0],
                  "1 columns declared without a mass point repeat a value", "dummy (continuous)")
    assert not pd.isna(discover(data, "y").table().set_index("covariate").loc["dummy", "support"])


def test_descriptive_attributes():
    data, _ = village_study()
    src = data.sources[0]
    kw = dict(level="village", time="pre", support=src.support)
    sat = Source(src.data, origin="learned", raw=src.raw, interpreter=src.interpreter, name="satellite",
                 modality="image", access="public", reference="data/README.md#satellite", **kw)
    _raises(lambda: Source(src.data, access="open", **kw), "access", "'open'")
    _raises(lambda: Source(src.data, modality="", **kw), "modality")
    data2 = Dataset(data.units, [sat] + data.sources[1:], data.design)
    text = _printed(data2.info)
    assert "[modality image, access public, ref data/README.md#satellite]" in text
    tab = discover(data2, "y").table().set_index("covariate")
    assert list(tab.columns[-3:]) == ["modality", "access", "reference"]
    assert tab.loc["v1", "modality"] == "image" and tab.loc["v1", "access"] == "public"
    assert pd.isna(tab.loc["r1", "modality"]) and pd.isna(tab.loc["r1", "access"])
    # the search ignores them
    assert np.array_equal(discover(data, "y").selection.pvalues, discover(data2, "y").selection.pvalues)


def test_declared_vs_detected_level():
    """A household column constant within villages is reported and clustered at the village."""
    data, d = village_study()
    vil = data.units["village"].to_numpy()
    rows = pd.DataFrame({"hh_v": np.random.default_rng(2).normal(size=40)[vil]}, index=data.units.index)
    data2 = Dataset(data.units, data.sources + [Source(rows, level="row", time="pre", support="continuous")],
                    data.design)
    assert data2.varies_at()["hh_v"] == "village" and data2.varies_at()["r1"] == "row"
    assert "1: hh_v (row -> village)" in _printed(data2.info)
    tab = discover(data2, "y").table().set_index("covariate")
    assert (tab.loc["hh_v", "level"], tab.loc["hh_v", "varies_at"]) == ("row", "village")
    assert tab.loc["hh_v", "clustered_at"] == "village"


def test_dataset_validation():
    data, d = village_study()
    units, srcs = data.units, data.sources
    # a unit id without a row in the source
    short = srcs[0].data.iloc[1:]
    short_src = Source(short, level="village", time="pre", support=srcs[0].support)
    _raises(lambda: Dataset(units, [short_src] + srcs[1:], data.design), "no row", "village")
    # duplicate names across sources
    clash = pd.DataFrame({"v1": 0.0}, index=pd.Index(range(8), name="district"))
    _raises(lambda: Dataset(units, srcs + [Source(clash, level="district", time="pre", support="continuous")], data.design), "'v1'")
    # a level that is not in units
    other = pd.DataFrame({"x": 0.0}, index=pd.Index(range(3), name="county"))
    _raises(lambda: Dataset(units, srcs + [Source(other, level="county", time="pre", support="continuous")], data.design), "county")
    # missing ids in units: the unit index or a level column
    _raises(lambda: Dataset(units.set_axis(pd.Index([np.nan] + list(range(1, len(units))), name="row")),
                            srcs, data.design), "missing ids")
    _raises(lambda: Dataset(units.assign(village=units["village"].where(units.index > 2)), srcs, data.design),
            "'village' id of 3 units")
    # a non-numeric treatment
    _raises(lambda: Dataset(units.assign(T=units["T"].map({0.0: "c", 1.0: "t"})), srcs, data.design),
            "must be numeric")


def test_randomized_checks():
    data, _ = village_study()
    units, srcs, des = data.units, data.sources, data.design
    assert data.checks == ["treatment 0/1: 160 treated, 160 control", "treatment constant within village",
                           "each village lies in one district stratum", "40 village clusters"]
    # treatment not 0/1, or a single arm
    _raises(lambda: Dataset(units.assign(T=units["T"] * 2), srcs, des), "0/1", "[2.0]")
    _raises(lambda: Dataset(units.assign(T=1.0), srcs, des), "no control units")
    _raises(lambda: Dataset(units.assign(T=0.0), srcs, des), "no treated units")
    # treatment varying within the assignment unit
    bad = units.copy()
    bad.loc[0, "T"] = 1 - bad.loc[0, "T"]
    _raises(lambda: Dataset(bad, srcs, des), "varies within 1 village", "0")
    # assignment units spanning strata
    span = units.copy()
    span.loc[0, "district"] = 7
    _raises(lambda: Dataset(span, srcs, des), "1 village units span several district strata")
    # a stratum with a single arm
    flat = units.copy()
    flat.loc[flat["district"] == 0, "T"] = 1.0
    _warns(lambda: Dataset(flat, srcs, des), "1 of 8 district strata hold a single arm")
    # sampled_by not nested with assigned_at
    crossed = units.assign(team=np.arange(len(units)) % 3)
    _raises(lambda: Dataset(crossed, srcs, Randomized("T", "village", sampled_by="team")), "not nested")
    # few clusters at the assignment level
    by_district = units.assign(T=(units["district"] % 2).astype(float))
    _warns(lambda: Dataset(by_district, srcs, Randomized("T", "district")), "only 8 district clusters")


def test_discontinuity_checks():
    units, src, score = rdd_units()
    des = Discontinuity("T", running_variable="score", assigned_at="hh", sampled_by="village")
    data = _quiet(lambda: Dataset(units, src, des))
    lo, hi = des.cutoff_interval
    assert des.direction == "<=" and lo == score[score <= 4].max() and hi == score[score > 4].min()
    assert data.checks[1].startswith("sharp: treated iff score <= cutoff")
    assert data.checks[-1] == "40 village clusters"
    # a given cutoff inside the interval passes, outside fails
    Dataset(units, src, Discontinuity("T", "score", "hh", cutoff=4.0))
    _raises(lambda: Dataset(units, src, Discontinuity("T", "score", "hh", cutoff=5.0)), "cutoff 5.0", "<=")
    # units exactly at the given cutoff: treated passes and is reported, control fails
    at = units.copy()
    at.loc[0, ["score", "T"]] = [4.0, 1.0]
    d_at = Discontinuity("T", "score", "hh", cutoff=4.0)
    assert "1 units exactly at the cutoff, treated" in Dataset(at, src, d_at).checks[1]
    at.loc[0, ["score", "T"]] = [4.0, 0.0]
    _raises(lambda: Dataset(at, src, Discontinuity("T", "score", "hh", cutoff=4.0)),
            "1 units lie exactly at score = 4.0, all control")
    # treated above the cutoff: direction >=
    d2 = Discontinuity("T", "score", "hh")
    Dataset(units.assign(T=1 - units["T"]), src, d2)
    assert d2.direction == ">="
    # not sharp: the violators are named and counted by arm
    fuzzy = units.copy()
    fuzzy.loc[fuzzy.index[score.argmax()], "T"] = 1.0
    _raises(lambda: Dataset(fuzzy, src, Discontinuity("T", "score", "hh")), "not a sharp",
            "misclassifies 1 units (1 treated, 0 control)", f"{score.argmax()} (score=")
    # an infinite running variable
    _raises(lambda: Dataset(units.assign(score=units["score"].replace(score[3], np.inf)), src,
                            Discontinuity("T", "score", "hh")), "infinite", "3")
    # the running variable is a design variable, never a candidate: it may live in a source
    src2 = [Source(pd.DataFrame({"x": src[0].data["x"], "pmt": score}, index=units.index), level="hh",
                   time="pre", support="continuous"), src[1]]
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
    res = _quiet(lambda: discover(data, "y", min_support=5))
    keys, parts, info = build_rule(d)                       # build_rule's own default: 5
    pool = [j for j, i in enumerate(info) if i["testable"]]
    fn = LevelAwareTest(keys, parts, fe=block_fe(d["levels"]["district"])).subset(pool)
    ref = nexis.nexis(d["y"], d["t"], d["z"][:, pool], alpha=0.05, max_rounds=20, rho=0.5,
                      adjust="FWER", backward=False, terminal_filter=True, pvalue_fn=fn)
    assert res.selection.selected == ref.selected
    assert np.array_equal(res.selection.pvalues, ref.pvalues)
    assert res.modifiers == [d["names"][pool[j]] for j in ref.selected] == ["v1"]
    assert (res.n, res.n_treated, res.n_control) == (320, 160, 160)
    tab = res.table().set_index("covariate")
    assert tab.loc["v1", "status"] == "selected"
    assert tab.loc["v3", "status"] == "dropped: support 1 < 5 clusters" and tab.loc["v3", "support"] == 1
    assert tab.loc["v1", "clustered_at"] == "village" and tab.loc["v1", "n_clusters"] == 40
    assert tab.loc["d1", "clustered_at"] == "district" and tab.loc["d1", "n_clusters"] == 8
    assert tab.loc["r1", "clustered_at"] == "village"      # assigned at the village
    S = list(ref.selected)
    j = pool.index(1)
    assert np.isclose(tab.loc["v2", "p_value"], fn(d["y"], d["t"], d["z"][:, pool], S, [j])[j], rtol=1e-12)


def test_min_support_default():
    """discover's default is 4: a sparse village dummy whose four cells span exactly 4
    villages is tested by default and dropped with min_support=5."""
    d = multilevel_data()
    t_vil = pd.Series(d["t"]).groupby(d["levels"]["village"]).first()
    on = list(t_vil.index[t_vil == 1][:4]) + list(t_vil.index[t_vil == 0][:4])
    extra = pd.DataFrame({"s4": np.isin(np.arange(40), on).astype(float)},
                         index=pd.Index(np.arange(40), name="village"))
    data, _ = village_study(pre_extra=Source(extra, level="village", time="pre", support="binary"))
    res = discover(data, "y")
    assert res.min_support == 4 and "s4" in res.tested
    assert res.table().set_index("covariate").loc["s4", "support"] == 4
    res5 = discover(data, "y", min_support=5)
    assert "s4" not in res5.tested and len(res5.tested) == len(res.tested) - 1


def test_during_excluded_unless_asked():
    data, _ = village_study(during=True)
    res = _warns(lambda: discover(data, "y"), "during treatment", "rain")
    tab = res.table().set_index("covariate")
    assert tab.loc["rain", "status"] == "excluded: measured during treatment"
    assert "rain" not in res.tested
    res2 = discover(data, "y", include_during=True)
    assert "rain" in res2.table()["covariate"].tolist()
    assert res2.table().set_index("covariate").loc["rain", "status"] != "excluded: measured during treatment"


# ── missing values ───────────────────────────────────────────────────────────

def test_missing_design_variables_dropped():
    data, _ = village_study()
    units = data.units.copy()
    units.loc[units["village"] == 0, "T"] = np.nan          # village 0 (8 units) has no treatment
    data2 = _warns(lambda: Dataset(units, data.sources, data.design), "8 of 320 units are dropped",
                   "8 miss 'T'")
    assert len(data2.units) == 312 and list(data2.dropped_units) == ["missing T"]
    assert data2.levels["village"].nunique() == 39
    assert "dropped: 8 units (missing T)" in _printed(data2.info)
    # a missing running variable: the warning counts the arms
    units, src, score = rdd_units()
    units.loc[[0, 1], "score"] = np.nan
    t01 = units.loc[[0, 1], "T"]
    data3 = _warns(lambda: Dataset(units, src, Discontinuity("T", "score", "hh")),
                   f"({int(t01.sum())} treated, {int(2 - t01.sum())} control)")
    assert len(data3.units) == 198


def test_missing_outcome_dropped():
    data, d = village_study()
    y = d["y"].copy()
    y[:5] = np.nan
    data_na, _ = village_study(y=y)
    t5 = d["t"][:5]
    res = _warns(lambda: discover(data_na, "y"), "5 of 320 units have no 'y'",
                 f"({int(t5.sum())} treated, {int(5 - t5.sum())} control)")
    assert res.n == 315 and list(res.units) == list(range(5, 320))
    # the same as building the Dataset on the units with an outcome
    keep = data.units.index[5:]
    srcs = [s if s.level != "row" else Source(s.data.loc[keep], level="row", time=s.time, support=s.support, name=s.name) for s in data_na.sources]
    ref = discover(Dataset(data.units.loc[keep], srcs, data.design), "y")
    assert res.modifiers == ref.modifiers and np.array_equal(res.selection.pvalues, ref.selection.pvalues)
    assert "outcome y missing for 5 units" in _printed(data_na.info)


def test_missing_candidate_excluded():
    data, d = village_study()
    rows = data.sources[2].data.copy()
    rows.iloc[[3, 7], 1] = np.nan                           # r2 missing for 2 units
    srcs = data.sources[:2] + [Source(rows, level="row", time="pre", support="continuous", name="survey")] + data.sources[3:]
    data_na = Dataset(data.units, srcs, data.design)
    assert data_na.missing.to_dict() == {"r2": 2}
    assert "candidates with missing values (excluded from the pool): r2 (2)" in _printed(data_na.info)
    res = _warns(lambda: discover(data_na, "y"), "1 candidates have missing values", "r2 (2)")
    tab = res.table().set_index("covariate")
    assert tab.loc["r2", "status"] == "excluded: 2 missing values" and tab.loc["r2", "missing"] == 2
    assert "r2" not in res.tested
    # the same as without the column
    srcs_wo = data.sources[:2] + [Source(rows[["r1"]], level="row", time="pre", support="continuous",
                                                    name="survey")] + data.sources[3:]
    ref = discover(Dataset(data.units, srcs_wo, data.design), "y")
    assert res.tested == ref.tested and res.modifiers == ref.modifiers
    assert np.array_equal(res.selection.pvalues, ref.selection.pvalues)
    # missing only where the outcome is missing too: the column stays in the pool
    y = d["y"].copy()
    y[[3, 7]] = np.nan
    out = Source(pd.DataFrame({"y": y}, index=data.units.index), level="row", time="post", support="continuous")
    res2 = _warns(lambda: discover(Dataset(data.units, srcs[:3] + [out], data.design), "y"), "2 of 320 units")
    assert "r2" in res2.tested


# ── interpret and info ───────────────────────────────────────────────────────

def test_interpret():
    data, _ = village_study()
    res = discover(data, "y")
    out = res.interpret(k=2)
    assert isinstance(out, nexis.Interpretation) and len(out) == 1
    v1 = data.sources[0].data["v1"].sort_values()             # continuous: no ties
    top, bottom = list(v1.index[::-1][:2]), list(v1.index[:2])
    row = out.frame.iloc[0]
    assert (row.covariate, row.source, row.top, row.bottom) == ("v1", "village_pre", top, bottom)
    assert out.labels == {"v1": f"top {['tile_' + str(i) for i in top]} vs bottom {['tile_' + str(i) for i in bottom]}"}
    assert out.items["v1"] == ([f"tile_{i}" for i in top], [f"tile_{i}" for i in bottom])
    text = repr(out)
    assert text.startswith("Interpretation of 1 learned modifier:\n  v1 (village_pre): top ['tile_")
    assert f"top ids    {top[0]}, {top[1]}" in text
    print(text)


def test_interpret_ties_at_random():
    """A sparse atom: the bottom examples are drawn among the tied zeros with `seed`."""
    rng = np.random.default_rng(3)
    atom = np.where(rng.random(40) < 0.3, rng.exponential(size=40), 0.0)
    extra = pd.DataFrame({"a": atom}, index=pd.Index(np.arange(40), name="village"))
    from nexis.interpret import extremes
    top, bottom = extremes(extra["a"], 5, seed=0)
    assert all(atom[i] == 0 for i in bottom) and bottom != sorted(bottom)[:5]
    assert list(np.argsort(-atom)[:5]) == top
    assert extremes(extra["a"], 5, seed=0) == (top, bottom)
    assert extremes(extra["a"], 5, seed=1)[1] != bottom


def test_interpret_nothing_learned():
    data, _ = village_study()
    # the same study with the village table declared measured: nothing to interpret
    srcs = [Source(data.sources[0].data, level="village", time="pre", support=data.sources[0].support)]
    srcs += data.sources[1:]
    res = discover(Dataset(data.units, srcs, data.design), "y")
    assert res.modifiers == ["v1"]
    out = res.interpret()
    assert len(out) == 0 and out.labels == {}
    assert repr(out) == ("Interpretation: no learned modifier was selected (modifiers: v1); "
                         "nothing to interpret")
    assert _printed(lambda: out.show()).startswith("Interpretation: no learned modifier")


def test_interpret_show():
    import matplotlib
    matplotlib.use("Agg")
    data, d = village_study()
    imgs = {v: np.full((4, 4), float(v)) for v in range(40)}
    srcs = [Source(data.sources[0].data, level="village", time="pre", support=data.sources[0].support,
                   origin="learned", raw=imgs, interpreter=Echo())] + data.sources[1:]
    out = discover(Dataset(data.units, srcs, data.design), "y").interpret(k=3)
    fig = out.show()
    assert len(fig.axes) == 2 * 3
    assert fig.axes[0].get_title(loc="left").startswith("v1: top")
    assert fig.axes[0].images[0].get_array()[0, 0] == out.frame.iloc[0].top[0]
    # raw items that are not images
    _raises(lambda: discover(data, "y").interpret(k=3).show(), "raw items must be image", "tile_")


def test_info_and_repr():
    data, _ = village_study(during=True)
    text = _printed(lambda: (data.info(), data.design.info()))
    assert "district 8, village 40, row 320" in text and "candidates during: 1" in text
    assert "passed: treatment constant within village" in text
    assert repr(data.design) == "Randomized(treatment='T', assigned_at='village', stratified_by='district')"
    assert "Dataset(320 row" in repr(data)
    res = discover(data, "y", include_during=True)
    assert repr(res).startswith("Result(outcome='y', n=320 (160 treated), m=")


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(f"ok    {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
