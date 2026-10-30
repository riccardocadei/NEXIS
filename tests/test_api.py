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

    def describe(self, name, top, bottom):
        return f"{name}: top {top} vs bottom {bottom}"


META = dict(origin="raw", domain="economy", access="public")


def src(data, name, level, time, support="continuous", **kw):
    """A Source with the bookkeeping attributes the test does not care about."""
    return Source(data, name=name, level=level, time=time, support=support, **{**META, **kw})


def like(s, data=None, **kw):
    """A copy of Source s (optionally on other data, with some attributes changed)."""
    data = s.data if data is None else data
    attrs = dict(name=s.name, level=s.level, time=s.time, access=s.access, modality=s.modality,
                 support={c: s.support[c] for c in data.columns}, origin={c: s.origin[c] for c in data.columns},
                 domain={c: s.domain[c] for c in data.columns}, raw=s.raw, interpreter=s.interpreter,
                 render=s.render)
    return Source(data, **{**attrs, **kw})


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
    sources = [Source(villages, name="satellite", level="village", time="pre",
                      support={"v1": "continuous", "v2": "continuous", "v3": "binary"}, origin="learned",
                      domain="environment", access="public", modality="image",
                      raw={v: f"tile_{v}" for v in range(40)}, interpreter=Echo()),
               src(districts, "district census", "district", "pre"),
               src(rows, "survey", "row", "pre", access="restricted", modality="survey"),
               src(outcome, "endline", "row", "post", access="restricted", modality="survey")]
    if during:
        sources.append(src(pd.DataFrame({"rain": np.random.default_rng(0).normal(size=8)},
                                        index=districts.index), "rainfall", "district", "during",
                           domain="environment", modality="raster"))
    if pre_extra is not None:
        sources.append(pre_extra)
    design = Randomized(treatment="T", assigned_at="village", stratified_by="district")
    return Dataset(units, sources, design), d


def rdd_units(n=200, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 10, n)
    units = pd.DataFrame({"village": np.arange(n) // 5, "score": score, "T": (score <= 4.0).astype(float)},
                         index=pd.Index(np.arange(n), name="hh"))
    sources = [src(pd.DataFrame({"x": rng.normal(size=n)}, index=units.index), "baseline", "hh", "pre"),
               src(pd.DataFrame({"y": rng.normal(size=n)}, index=units.index), "endline", "hh", "post")]
    return units, sources, score


# ── construction and validation ──────────────────────────────────────────────

def test_source_validation():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="village"))
    kw = dict(name="census", level="village", support="continuous", domain="economy", access="public")
    _raises(lambda: Source(df, time="before", origin="raw", **kw), "time", "'historic'", "'before'")
    _raises(lambda: Source(df, time="pre", origin="guessed", **kw), "origin", "'hand_crafted'", "'guessed'")
    _raises(lambda: Source(df, time="pre", origin="measured", **kw), "origin")     # the old vocabulary
    _raises(lambda: Source(df, time="pre", origin="learned", **kw), "raw=", "interpreter=")
    _raises(lambda: Source(df, time="pre", origin="learned", raw={0: 0, 1: 1}, interpreter=object(), **kw),
            "describe")
    _raises(lambda: Source(df.assign(b=["x", "y"]), time="pre", origin="raw", **kw), "numeric", "b")
    _raises(lambda: Source(df.assign(b=[np.inf, 0.0]), time="pre", origin="raw", **kw), "infinite")
    dup = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 0], name="village"))
    _raises(lambda: Source(dup, time="pre", origin="raw", **kw), "repeats ids")
    nan_id = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0.0, np.nan], name="village"))
    _raises(lambda: Source(nan_id, time="pre", origin="raw", **kw), "1 missing ids")
    # every attribute but modality, raw and interpreter is a required keyword
    _raises(lambda: Source(df, "pre"), "positional")
    full = dict(name="census", level="village", time="pre", support="continuous", origin="raw",
                domain="economy", access="public")
    Source(df, **full)
    for key in full:
        _raises(lambda: Source(df, **{k: v for k, v in full.items() if k != key}), key)
    assert Source(df, **full).modality is None
    # the source name: a non-empty str
    for bad in ("", "  ", None, 3):
        _raises(lambda: Source(df, **{**full, "name": bad}), "name", "non-empty str")


def test_source_level():
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([0, 1], name="village"))
    # an index named otherwise is an error; an unnamed one is named after the level
    _raises(lambda: src(df, "census", "district", "pre"), "source 'census'", "indexed by 'village'",
            "level='district'")
    s = src(df.reset_index(drop=True), "census", "district", "pre")
    assert s.level == "district" and s.data.index.name == "district" and s.name == "census"
    _raises(lambda: src(df, "census", "", "pre"), "level", "non-empty str")
    two = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    _raises(lambda: src(two, "census", "village", "pre"), "MultiIndex")


def test_vocabularies():
    """time, origin, access and modality take a fixed vocabulary; domain an identifier;
    origin and domain one value for all columns or one per column."""
    idx = pd.Index(range(4), name="v")
    two = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 3.0, 2.0]}, index=idx)
    for time in ("historic", "pre", "during", "post"):
        assert src(two, "s", "v", time).time == time
    for access in ("public", "restricted", "proprietary"):
        assert src(two, "s", "v", "pre", access=access).access == access
    _raises(lambda: src(two, "s", "v", "pre", access="open"), "access", "'open'")
    _raises(lambda: src(two, "s", "v", "pre", access=None), "access")
    for modality in ("survey", "image", "raster", "events", "time_series", "text"):
        assert src(two, "s", "v", "pre", modality=modality).modality == modality
    assert set(nexis.data.MODALITIES) == {"survey", "image", "raster", "events", "time_series", "text"}
    for bad in ("", "tabular", "Image"):
        _raises(lambda: src(two, "s", "v", "pre", modality=bad), "modality", repr(bad))
    _raises(lambda: src(two, "s", "v", "pre", modality=1), "modality")
    # origin and domain: one for all or per column
    s = src(two, "s", "v", "pre", origin={"b": "hand_crafted", "a": "raw"},
            domain={"a": "economy", "b": "housing_quality2"})
    assert s.origin == {"a": "raw", "b": "hand_crafted"} and s.domain == {"a": "economy", "b": "housing_quality2"}
    assert src(two, "s", "v", "pre", domain="health").domain == {"a": "health", "b": "health"}
    _raises(lambda: src(two, "s", "v", "pre", origin={"a": "raw"}), "origin", "every column", "['b']")
    _raises(lambda: src(two, "s", "v", "pre", domain={"a": "x", "b": "y", "c": "z"}), "domain", "every column",
            "['c']")
    _raises(lambda: src(two, "s", "v", "pre", origin={"a": "raw", "b": "guessed"}), "origin of 'b'", "'guessed'")
    for bad in ("Economy", "", "land use", "2nd", "eco-nomy"):
        _raises(lambda: src(two, "s", "v", "pre", domain=bad), "domain of 'a'", "lowercase identifier")
    # wrong types
    _raises(lambda: src(two, "s", "v", "pre", domain=["economy"]), "domain", "str or a dict")
    _raises(lambda: src(two, "s", "v", "pre", origin={"a": "raw", "b": 1}), "origin values must be str")
    _raises(lambda: src(two, "s", "v", "pre", support=None), "support", "str or a dict")
    _raises(lambda: src(two, "s", "v", 1), "time")


def test_learned_columns():
    """Learned columns need raw (covering every id) and an interpreter taking
    (name, top, bottom); raw and interpreter on a source without learned columns are an error."""
    idx = pd.Index(range(4), name="v")
    two = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 3.0, 2.0]}, index=idx)
    raw = {i: f"tile_{i}" for i in range(4)}
    s = src(two, "s", "v", "pre", origin={"a": "learned", "b": "hand_crafted"}, raw=raw, interpreter=Echo())
    assert s.raw_items([2, 0]) == ["tile_2", "tile_0"]
    src(two, "s", "v", "pre", origin="learned", raw=lambda i: f"tile_{i}", interpreter=Echo())
    _raises(lambda: src(two, "s", "v", "pre", origin="learned", raw={0: "x", 1: "y"}, interpreter=Echo()),
            "raw has no item for 2 of 4 v ids", "[2, 3]")
    _raises(lambda: src(two, "s", "v", "pre", origin="learned", raw=["x"] * 4, interpreter=Echo()),
            "raw must be a mapping or a callable")
    _raises(lambda: src(two, "s", "v", "pre", origin={"a": "learned", "b": "raw"}, raw=raw),
            "learned columns (a)", "interpreter=")

    class Old:
        def describe(self, top, bottom):
            return ""
    _raises(lambda: src(two, "s", "v", "pre", origin="learned", raw=raw, interpreter=Old()),
            "describe must accept (name, top, bottom)")

    class Flexible:
        def describe(self, *args, **kwargs):
            return ""
    src(two, "s", "v", "pre", origin="learned", raw=raw, interpreter=Flexible())
    _raises(lambda: src(two, "s", "v", "pre", raw=raw), "raw=, interpreter=", "no column has origin 'learned'")
    _raises(lambda: src(two, "s", "v", "pre", interpreter=Echo()), "raw=, interpreter=")


def test_source_edge_cases():
    idx = pd.Index(range(4), name="v")
    _raises(lambda: src(pd.DataFrame(index=idx), "s", "v", "pre"), "no columns")
    _raises(lambda: src(pd.DataFrame({"a": []}, index=pd.Index([], name="v")), "s", "v", "pre"), "no rows")
    _raises(lambda: src(pd.DataFrame({0: [1.0, 2, 3, 4]}, index=idx), "s", "v", "pre"), "column names", "[0]")
    _raises(lambda: src(pd.DataFrame({"a": [1 + 1j, 2, 3, 4]}, index=idx), "s", "v", "pre"), "numeric", "['a']")
    _raises(lambda: src(pd.DataFrame({"a": pd.to_datetime(["2020-01-01"] * 4)}, index=idx), "s", "v", "pre"),
            "numeric")
    # booleans and nullable dtypes: True/False -> 1/0, NA -> NaN
    df = pd.DataFrame({"b": [True, False, True, False],
                       "nb": pd.array([True, None, False, True], dtype="boolean"),
                       "ni": pd.array([1, None, 3, 0], dtype="Int64")}, index=idx)
    s = src(df, "s", "v", "pre", support={"b": "binary", "nb": "binary", "ni": "count"})
    assert s.data["nb"].isna().sum() == 1 and s.data["b"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert s.data.dtypes.eq(float).all()
    # a Series is a one-column source
    assert list(src(pd.Series([1.0, 2, 3, 4], index=idx, name="x"), "s", "v", "pre").data.columns) == ["x"]


def test_source_support():
    idx = pd.Index(range(6), name="v")
    ok = {"binary": [0, 1, 1, np.nan, 0, 0], "count": [0, 3, 7, 2, 0, np.nan],
          "nonnegative": [0.0, 0.5, 2.3, 9.0, 0.1, 1.0], "sparse": [0, 0, 0.7, 0, 2.5, np.nan],
          "continuous": [-1.5, 0.0, 2.0, 3.3, -7.0, np.nan]}
    for kind, vals in ok.items():                          # NaN is ignored
        src(pd.DataFrame({"x": vals}, index=idx), "s", "v", "pre", support=kind)
    bad = {"binary": ([0, 1, 2, 0.5, 0, 1], "2 values", "0.5"), "count": ([0, 1, 2.5, -1, 0, 0], "2 values", "-1.0"),
           "nonnegative": ([0, 1, -0.2, 3, 4, 5], "1 values", "-0.2"),
           "sparse": ([0, 1, 1, 1, 0, 2], "zero (2 rows)", "most frequent"),
           }
    for kind, (vals, *words) in bad.items():
        _raises(lambda: src(pd.DataFrame({"x": vals}, index=idx), "s", "v", "pre", support=kind),
                "'x'", f"declared {kind}", *words)
    _raises(lambda: src(pd.DataFrame({"x": [-1.0] * 6}, index=idx), "s", "v", "pre", support="sparse"),
            "declared sparse", "-1.0")
    _raises(lambda: src(pd.DataFrame({"x": [1.0] * 6}, index=idx), "s", "v", "pre", support="real"),
            "support of 'x'", "'real'")
    # one kind per column: a dict covering every column exactly
    two = pd.DataFrame({"a": [0, 1] * 3, "b": np.arange(6.0) - 2}, index=idx)
    s = src(two, "s", "v", "pre", support={"b": "continuous", "a": "binary"})
    assert s.support == {"a": "binary", "b": "continuous"}
    _raises(lambda: src(two, "s", "v", "pre", support={"a": "binary"}), "every column", "['b']")
    _raises(lambda: src(two, "s", "v", "pre", support={"a": "binary", "b": "continuous", "c": "count"}),
            "every column", "['c']")
    _raises(lambda: src(two, "s", "v", "pre", support="binary"), "'b'", "declared binary")


def test_declared_support_rule():
    """The support rule applies to columns declared binary or sparse even without a data
    mass point (the union with the data rule), and to count columns only with one;
    declared continuous columns with a mass point get it from the data, with a warning."""
    rng = np.random.default_rng(5)
    # sparse village atom: zero is its most frequent value but holds only 3 of 40 villages
    atom = rng.exponential(size=40)
    atom[[0, 1, 2]] = 0.0
    extra = pd.DataFrame({"atom": atom}, index=pd.Index(np.arange(40), name="village"))
    sparse, _ = village_study(pre_extra=src(extra, "extra", "village", "pre", support="sparse"))
    nonneg, _ = village_study(pre_extra=src(extra, "extra", "village", "pre", support="nonnegative"))
    t_sp = discover(sparse, "y").table().set_index("covariate")
    t_nn = discover(nonneg, "y").table().set_index("covariate")
    assert t_sp.loc["atom", "kind"] == "sparse" and t_nn.loc["atom", "kind"] == "nonnegative"
    assert pd.isna(t_nn.loc["atom", "support"]) and not t_nn.loc["atom", "status"].startswith("dropped")
    assert t_sp.loc["atom", "support"] <= 3
    assert t_sp.loc["atom", "status"].startswith("dropped: support")
    # every other candidate is unchanged
    assert t_sp.drop(index="atom").drop(columns="p_value").equals(t_nn.drop(index="atom").drop(columns="p_value"))
    # declared count without a data mass point: no support rule
    cnt = pd.DataFrame({"n": np.arange(40, dtype=float) * 7 % 41}, index=pd.Index(np.arange(40), name="village"))
    t_ct = discover(village_study(pre_extra=src(cnt, "extra", "village", "pre", support="count"))[0],
                    "y").table().set_index("covariate")
    assert pd.isna(t_ct.loc["n", "support"]) and not t_ct.loc["n", "status"].startswith("dropped")
    # declared continuous with a mass point: the data rule applies, with a warning
    dummy = pd.DataFrame({"dummy": (np.arange(40) % 3 == 0).astype(float)},
                         index=pd.Index(np.arange(40), name="village"))
    data = _warns(lambda: village_study(pre_extra=src(dummy, "s", "village", "pre"))[0],
                  "1 columns declared without a mass point repeat a value", "dummy (continuous)")
    assert not pd.isna(discover(data, "y").table().set_index("covariate").loc["dummy", "support"])


def test_attributes_in_tables():
    """info() and table() show every attribute; only level, time and support change the search."""
    data, _ = village_study(during=True)
    tab = discover(data, "y").table()
    assert list(tab.columns) == ["covariate", "source", "level", "varies_at", "time", "kind", "origin", "domain",
                                 "clustered_at", "n_clusters", "support", "missing", "p_value", "status",
                                 "access", "modality"]
    tab = tab.set_index("covariate")
    assert tuple(tab.loc["v1", ["source", "origin", "domain", "access", "modality"]]) == \
        ("satellite", "learned", "environment", "public", "image")
    assert tuple(tab.loc["r1", ["source", "origin", "domain", "access", "modality"]]) == \
        ("survey", "raw", "economy", "restricted", "survey")
    assert pd.isna(tab.loc["d1", "modality"])
    assert (tab.loc["rain", "time"], tab.loc["rain", "varies_at"]) == ("during", "district")
    st = data.sources_table()
    assert list(st.columns) == ["source", "level", "varies_at", "time", "columns", "support", "origin", "domain",
                                "access", "modality"]
    assert st.iloc[0].to_dict() == dict(source="satellite", level="village", varies_at="village", time="pre",
                                        columns=3, support="continuous 2, binary 1", origin="learned",
                                        domain="environment", access="public", modality="image")
    assert st.set_index("source").loc["endline", "varies_at"] == ""
    text = _printed(data.info)
    assert "district census" in text and "continuous 2, binary 1" in text and "proprietary" not in text
    # the search ignores name, origin, domain, access and modality
    s0 = data.sources[0]
    other = [like(s0, name="other", origin={"v1": "raw", "v2": "hand_crafted", "v3": "raw"}, raw=None,
                  interpreter=None, domain="housing", access="proprietary", modality=None)]
    other += [like(s, name=f"{s.name} bis", domain="health", access="proprietary") for s in data.sources[1:]]
    data2 = Dataset(data.units, other, data.design)
    assert np.array_equal(discover(data, "y").selection.pvalues, discover(data2, "y").selection.pvalues)
    assert "proprietary" in _printed(data2.info)


def test_declared_vs_detected_level():
    """A household column constant within villages is reported and clustered at the village."""
    data, d = village_study()
    vil = data.units["village"].to_numpy()
    rows = pd.DataFrame({"hh_v": np.random.default_rng(2).normal(size=40)[vil]}, index=data.units.index)
    data2 = Dataset(data.units, data.sources + [src(rows, "extra", "row", "pre", support="continuous")],
                    data.design)
    assert data2.varies_at()["hh_v"] == "village" and data2.varies_at()["r1"] == "row"
    assert "1: hh_v (row -> village)" in _printed(data2.info)
    tab = discover(data2, "y").table().set_index("covariate")
    assert (tab.loc["hh_v", "level"], tab.loc["hh_v", "varies_at"]) == ("row", "village")
    assert tab.loc["hh_v", "clustered_at"] == "village"


def test_extra_level_detection():
    """An extra level nested over the villages (pairs, one pair a single village) takes
    the columns constant within its cells, which are clustered by it (coarser, never
    finer than declared).  A level cutting across the villages is never detected for a
    village column, even when the column is constant within its cells by coincidence."""
    rng = np.random.default_rng(4)
    n, nv = 400, 40
    vil = np.arange(n) // 10
    units = pd.DataFrame({"village": vil, "T": rng.permutation(np.repeat([0.0, 1.0], n // 2))},
                         index=pd.Index(np.arange(n), name="hh"))
    vidx = pd.Index(range(nv), name="village")
    out = src(pd.DataFrame({"y": rng.normal(size=n)}, index=units.index), "endline", "hh", "post")
    hh = src(pd.DataFrame({"h": rng.normal(size=n)}, index=units.index), "survey", "hh", "pre")
    # nearly 1:1 nested: 39 groups over 40 villages (villages 0 and 1 merged)
    near = np.where(vil == 1, 0, vil)
    paired = rng.normal(size=nv)
    paired[1] = paired[0]                                    # equal on the merged pair: re-classified
    cols = pd.DataFrame({"same": paired, "differ": rng.normal(size=nv)}, index=vidx)
    gids = pd.Index(np.unique(near), name="group")            # a level is one some source is indexed by
    price = src(pd.DataFrame({"price": rng.normal(size=len(gids))}, index=gids), "prices", "group", "pre")
    data = Dataset(units.assign(group=near), [hh, src(cols, "census", "village", "pre"), price, out],
                   Randomized("T", "hh"))
    assert list(data.levels) == ["group", "village", "hh"]
    assert data.varies_at().to_dict() == {"h": "hh", "same": "group", "differ": "village", "price": "group"}
    tab = discover(data, "y").table().set_index("covariate")
    assert (tab.loc["same", "clustered_at"], tab.loc["same", "n_clusters"]) == ("group", 39)
    assert (tab.loc["differ", "clustered_at"], tab.loc["differ", "n_clusters"]) == ("village", 40)
    # cutting across: groups of village pairs, but half of village 0 sits with villages 2, 3
    cut = vil // 2
    cut[5:10] = 1
    x = rng.normal(size=20)[np.arange(nv) // 2]
    x[:4] = x[0]                                             # constant within every cut group
    cross = pd.DataFrame({"x": x}, index=vidx)
    gsrc = src(pd.DataFrame({"g": rng.normal(size=20)}, index=pd.Index(range(20), name="group")),
               "groups", "group", "pre")
    data = Dataset(units.assign(group=cut), [hh, src(cross, "census", "village", "pre"), gsrc, out],
                   Randomized("T", "hh"))
    assert list(data.levels) == ["group", "village", "hh"]
    assert data.varies_at().to_dict() == {"h": "hh", "x": "village", "g": "group"}
    from nexis.data import nested_order
    lv = {k: v.to_numpy() for k, v in data.levels.items()}
    assert nested_order(lv, "village") == ["village", "hh"] and nested_order(lv, "group") == ["group", "hh"]
    tab = discover(data, "y").table().set_index("covariate")
    assert tab.loc["x", "clustered_at"] == "village" and tab.loc["g", "clustered_at"] == "group"


def test_dataset_validation():
    data, d = village_study()
    units, srcs = data.units, data.sources
    # a unit id without a row in the source
    short = srcs[0].data.iloc[1:]
    short_src = src(short, "satellite", "village", "pre", support=srcs[0].support)
    _raises(lambda: Dataset(units, [short_src] + srcs[1:], data.design), "no row", "village")
    # duplicate names across sources
    clash = pd.DataFrame({"v1": 0.0}, index=pd.Index(range(8), name="district"))
    _raises(lambda: Dataset(units, srcs + [src(clash, "clash", "district", "pre", support="continuous")], data.design), "'v1'")
    # a level that is not in units
    other = pd.DataFrame({"x": 0.0}, index=pd.Index(range(3), name="county"))
    _raises(lambda: Dataset(units, srcs + [src(other, "county data", "county", "pre", support="continuous")], data.design), "county")
    # missing ids in units: the unit index or a level column
    _raises(lambda: Dataset(units.set_axis(pd.Index([np.nan] + list(range(1, len(units))), name="row")),
                            srcs, data.design), "missing ids")
    _raises(lambda: Dataset(units.assign(village=units["village"].where(units.index > 2)), srcs, data.design),
            "'village' id of 3 units")
    # a non-numeric treatment
    _raises(lambda: Dataset(units.assign(T=units["T"].map({0.0: "c", 1.0: "t"})), srcs, data.design),
            "must be numeric")
    # source names are unique
    rows2 = pd.DataFrame({"r9": np.arange(len(units), dtype=float)}, index=units.index)
    _raises(lambda: Dataset(units, srcs + [src(rows2, "survey", "row", "pre")], data.design),
            "source names must be unique", "survey")
    Dataset(units, srcs + [src(rows2, "survey 2", "row", "pre")], data.design)
    # a source column named like the unit index; units with a column named like their index
    _raises(lambda: Dataset(units, srcs + [src(rows2.rename(columns={"r9": "row"}), "x", "row", "pre")],
                            data.design), "'row'", "index", "units")
    _raises(lambda: Dataset(units.assign(row=1), srcs, data.design), "named like its index", "'row'")
    # ids of another dtype than the units' level column: a hint
    dis = srcs[1].data.set_axis(srcs[1].data.index.astype(str))
    _raises(lambda: Dataset(units, [srcs[0], like(srcs[1], dis)] + srcs[2:], data.design), "no row for 8 of 8",
            "district ids of units are int64, the source index is object")
    # wrong types
    _raises(lambda: Dataset(units.to_numpy(), srcs, data.design), "units must be a pandas DataFrame")
    _raises(lambda: Dataset(units, srcs[0], data.design), "list of nexis Source")
    _raises(lambda: Dataset(units, srcs + [rows2], data.design), "list of nexis Source")


def test_pool_checks():
    """Over the units: a constant or duplicated candidate is an error naming the columns
    and their sources; |corr| > 0.999 is a warning.  Outcomes and design variables are
    not candidates, and rows of ids outside the units do not count."""
    data, _ = village_study()
    units, srcs, des = data.units, data.sources, data.design
    vidx = srcs[0].data.index

    def add(*cols, time="pre"):
        frame = pd.DataFrame(dict(cols), index=vidx)
        return Dataset(units, srcs + [src(frame, "extra", "village", time)], des)

    rng = np.random.default_rng(1)
    noise = rng.normal(size=40)
    _raises(lambda: add(("c0", np.full(40, 2.0))), "1 candidate columns are constant", "c0 (extra): always 2")
    with_nan = np.where(np.arange(40) < 5, np.nan, 3.0)
    _raises(lambda: add(("c1", with_nan)), "c1 (extra): always 3, 40 missing")       # 5 villages x 8 units
    _raises(lambda: add(("c2", np.full(40, np.nan))), "c2 (extra): all missing")
    _raises(lambda: add(("c3", np.full(40, 1.0)), time="historic"), "c3 (extra)")       # historic is a candidate
    _raises(lambda: add(("c4", np.full(40, 1.0)), time="during"), "c4 (extra)")
    add(("c5", np.full(40, 1.0)), time="post")                                         # an outcome may be constant
    # identical: within and across sources, NaN patterns included, -0.0 == 0.0
    _raises(lambda: add(("a", noise), ("b", noise.copy())), "identical", "a (extra) = b (extra)")
    v1 = srcs[0].data["v1"].to_numpy()
    _raises(lambda: add(("w", v1)), "identical", "v1 (satellite) = w (extra)")
    z = np.where(noise > 0, noise, 0.0)
    _raises(lambda: add(("p", z), ("q", np.where(noise > 0, noise, -0.0))), "p (extra) = q (extra)")
    na1, na2 = noise.copy(), noise.copy()
    na1[3] = na2[3] = np.nan
    _raises(lambda: add(("n1", na1), ("n2", na2)), "n1 (extra) = n2 (extra)")
    na2[3], na2[4] = noise[3], np.nan                                                   # other NaN pattern
    _warns(lambda: add(("n1", na1), ("n2", na2)), "1 pairs", "n1 (extra) ~ n2 (extra) (r = 1.0000)")
    # near-collinear: a warning, also for a negative correlation; the Dataset is built
    d2 = _warns(lambda: add(("e1", noise), ("e2", -2 * noise + 1e-6 * rng.normal(size=40))),
                "|correlation| > 0.999", "e1 (extra) ~ e2 (extra) (r = -1.0000)")
    assert "e2" in d2.candidates()
    _quiet(lambda: add(("f1", noise), ("f2", noise + 0.5 * rng.normal(size=40))))
    # only the units count: a column varying only on ids outside the units is constant
    extra = pd.DataFrame({"g": np.r_[np.ones(40), 5.0]}, index=pd.Index(range(41), name="village"))
    _raises(lambda: Dataset(units, srcs + [src(extra, "extra", "village", "pre")], des), "g (extra): always 1")
    # nor do units dropped for a missing treatment
    u2 = units.copy()
    u2.loc[u2["village"] == 0, "T"] = np.nan
    h = pd.DataFrame({"h": np.r_[7.0, np.ones(39)]}, index=vidx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _raises(lambda: Dataset(u2, srcs + [src(h, "extra", "village", "pre")], des), "h (extra): always 1")
    # a design variable in a source is not a candidate: a copy of the running variable is
    # a candidate like any other, a copy of the treatment is an error
    units3, rdd, score = rdd_units()
    base = pd.DataFrame({"x": rdd[0].data["x"], "pmt": score, "pmt2": score}, index=units3.index)
    d3 = Dataset(units3.drop(columns="score"), [src(base, "b", "hh", "pre"), rdd[1]], Discontinuity("T", "pmt", "hh"))
    assert d3.candidates() == ["x", "pmt2"]
    base["t_copy"] = units3["T"].astype(bool)
    _raises(lambda: Dataset(units3.drop(columns="score"), [src(base, "b", "hh", "pre", support={
        "x": "continuous", "pmt": "continuous", "pmt2": "continuous", "t_copy": "binary"}), rdd[1]],
        Discontinuity("T", "pmt", "hh")), "identical to the treatment 'T'", "t_copy (b)")


def test_rank_check():
    """An exact linear dependence among candidates (with an intercept, over the units) is
    a warning naming a minimal dependent set; the columns stay in the pool, and columns
    with missing values are left out of the check."""
    data, _ = village_study()
    units, srcs, des = data.units, data.sources, data.design
    vidx = srcs[0].data.index
    rng = np.random.default_rng(2)
    b, c, e = rng.normal(size=(3, 40))

    def add(*cols, time="pre"):
        frame = pd.DataFrame(dict(cols), index=vidx)
        return Dataset(units, srcs + [src(frame, "extra", "village", time)], des)

    # a = b + c: the set is {a, b, c}, not the unrelated e; every column stays a candidate
    d = _warns(lambda: add(("a", b + c), ("b", b), ("c", c), ("e", e)), "1 candidate columns are exact linear",
               "{a (extra), b (extra), c (extra)}", "p = 1")
    assert {"a", "b", "c", "e"} <= set(d.candidates())
    # the intercept counts: 2 b - 3 = a
    _warns(lambda: add(("a", 2 * b - 3), ("b", b), ("e", e)), "{a (extra), b (extra)}")
    # a total of counts in another source (household size = children + adults + elderly)
    hh = units.index
    kids, adults, old = (rng.integers(0, 4, len(hh)).astype(float) for _ in range(3))
    cnt = pd.DataFrame({"children": kids, "adults": adults, "elderly": old, "size": kids + adults + old,
                        "noise": rng.normal(size=len(hh))}, index=hh)
    _warns(lambda: Dataset(units, srcs + [src(cnt, "roster", hh.name, "pre", support={
        "children": "count", "adults": "count", "elderly": "count", "size": "count", "noise": "continuous"})], des),
        "{children (roster), adults (roster), elderly (roster), size (roster)}")
    # dummies of every category sum to the intercept
    cat = rng.integers(0, 3, 40)
    _warns(lambda: add(*[(f"g{k}", (cat == k).astype(float)) for k in range(3)], ("e", e)),
           "{g0 (extra), g1 (extra), g2 (extra)}")
    # two independent dependencies: two sets
    _warns(lambda: add(("a", b + c), ("b", b), ("c", c), ("f", e - b), ("e", e)), "2 candidate columns")
    # a column with a missing value is out of the pool, so not part of a dependence
    a = b + c
    a[0] = np.nan
    _quiet(lambda: add(("a", a), ("b", b), ("c", c)))
    # a dependence involving a during column counts (during columns are candidates)...
    d = add(("b", b), ("c", c))
    dur = pd.DataFrame({"s": b + c}, index=vidx)
    _warns(lambda: Dataset(units, d.sources + [src(dur, "rain", "village", "during")], des), "s (rain)")
    # ...but not one through an outcome
    out = pd.DataFrame({"s": b + c}, index=vidx)
    _quiet(lambda: Dataset(units, d.sources + [src(out, "outcome", "village", "post")], des))
    # near-dependence is not exact: no warning
    _quiet(lambda: add(("a", b + c + 1e-3 * e), ("b", b), ("c", c)))
    # the minimal set is found when other columns are also involved elsewhere
    from nexis.data import dependent_sets
    x = np.column_stack([b, c, e, b + c, rng.normal(size=40)])
    assert dependent_sets(x, list("bcexy")) == [["b", "c", "x"]]
    assert dependent_sets(np.column_stack([b, c, e]), list("bce")) == []


def test_dependent_column_not_selected():
    """A total and its parts all stay in the pool.  With y = T (b - 0.6 c) b enters first;
    given it, c and the total have the same residual (total = b + c), so one of them
    enters and the third column of the set, now in the span of the selected ones, gets
    p = 1 -- also when the total is on a large scale (discover standardises the
    candidates).  Which of c and the total enters is a tie decided by round-off."""
    rng = np.random.default_rng(5)
    n = 20000
    units = pd.DataFrame({"T": rng.integers(0, 2, n).astype(float)}, index=pd.Index(np.arange(n), name="hh"))
    b, c, e = rng.normal(size=(3, n))
    t = units["T"].to_numpy()
    y = t * (b - 0.6 * c) + rng.normal(size=n)
    out = pd.DataFrame({"y": y}, index=units.index)
    for scale in (1.0, 1e4):
        hh = pd.DataFrame({"b": b, "c": c, "total": scale * (b + c) + 3.0, "e": e}, index=units.index)
        data = _warns(lambda: Dataset(units, [src(hh, "survey", "hh", "pre"), src(out, "endline", "hh", "post")],
                                      Randomized("T", "hh")), "{b (survey), c (survey), total (survey)}")
        res = discover(data, "y")
        tab = res.table().set_index("covariate")
        assert len(res.modifiers) == 2 and "b" in res.modifiers, (scale, res.modifiers)
        rest = ({"c", "total"} - set(res.modifiers)).pop()
        assert tab.loc[rest, "status"] == "not selected" and tab.loc[rest, "p_value"] == 1.0, scale
        assert tab.loc["e", "status"] == "not selected"
        assert len(res.tested) == 4                                   # m counts every column of the set


def test_pool_checks_cost():
    """The pool checks on 200 candidates x 2,000 units take well under a second."""
    import time
    rng = np.random.default_rng(0)
    n = 2000
    units = pd.DataFrame({"village": np.arange(n) // 10, "T": (np.arange(n) // 10 % 2).astype(float)},
                         index=pd.Index(np.arange(n), name="hh"))
    hh = pd.DataFrame(rng.normal(size=(n, 150)), columns=[f"h{k}" for k in range(150)], index=units.index)
    hh.iloc[::7, 3] = np.nan
    vil = pd.DataFrame(rng.exponential(size=(200, 50)) * (rng.random((200, 50)) < 0.5),
                       columns=[f"a{k}" for k in range(50)], index=pd.Index(range(200), name="village"))
    out = pd.DataFrame({"y": rng.normal(size=n)}, index=units.index)
    sources = [src(hh, "survey", "hh", "pre"), src(vil, "satellite", "village", "pre", support="sparse"),
               src(out, "endline", "hh", "post")]
    data = Dataset(units, sources, Randomized("T", "village"))
    t0 = time.perf_counter()
    data._check_pool()
    took = time.perf_counter() - t0
    print(f"pool checks, 200 candidates x 2,000 units: {took:.3f} s")
    assert took < 2.0


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
    units, rsrc, score = rdd_units()
    des = Discontinuity("T", running_variable="score", assigned_at="hh", sampled_by="village")
    data = _quiet(lambda: Dataset(units, rsrc, des))
    lo, hi = des.cutoff_interval
    assert des.direction == "<=" and lo == score[score <= 4].max() and hi == score[score > 4].min()
    assert data.checks[1].startswith("sharp: treated iff score <= cutoff")
    assert data.checks[-1] == "40 village clusters"
    # a given cutoff inside the interval passes, outside fails
    Dataset(units, rsrc, Discontinuity("T", "score", "hh", cutoff=4.0))
    _raises(lambda: Dataset(units, rsrc, Discontinuity("T", "score", "hh", cutoff=5.0)), "cutoff 5.0", "<=")
    # units exactly at the given cutoff: treated passes and is reported, control fails
    at = units.copy()
    at.loc[0, ["score", "T"]] = [4.0, 1.0]
    d_at = Discontinuity("T", "score", "hh", cutoff=4.0)
    assert "1 units exactly at the cutoff, treated" in Dataset(at, rsrc, d_at).checks[1]
    at.loc[0, ["score", "T"]] = [4.0, 0.0]
    _raises(lambda: Dataset(at, rsrc, Discontinuity("T", "score", "hh", cutoff=4.0)),
            "1 units lie exactly at score = 4.0, all control")
    # treated above the cutoff: direction >=
    d2 = Discontinuity("T", "score", "hh")
    Dataset(units.assign(T=1 - units["T"]), rsrc, d2)
    assert d2.direction == ">="
    # not sharp: the violators are named and counted by arm
    fuzzy = units.copy()
    fuzzy.loc[fuzzy.index[score.argmax()], "T"] = 1.0
    _raises(lambda: Dataset(fuzzy, rsrc, Discontinuity("T", "score", "hh")), "not a sharp",
            "misclassifies 1 units (1 treated, 0 control)", f"{score.argmax()} (score=")
    # an infinite running variable
    _raises(lambda: Dataset(units.assign(score=units["score"].replace(score[3], np.inf)), rsrc,
                            Discontinuity("T", "score", "hh")), "infinite", "3")
    # the running variable is a design variable, never a candidate: it may live in a source
    src2 = [src(pd.DataFrame({"x": rsrc[0].data["x"], "pmt": score}, index=units.index), "baseline", "hh",
                "pre"), rsrc[1]]
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
    z = d["z"][:, pool]
    z = (z - z.mean(axis=0)) / z.std(axis=0)                   # discover standardises the candidates
    ref = nexis.nexis(d["y"], d["t"], z, alpha=0.05, max_rounds=20, rho=0.5,
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
    assert np.isclose(tab.loc["v2", "p_value"], fn(d["y"], d["t"], z, S, [j])[j], rtol=1e-12)
    # on the raw columns the p-values agree up to round-off
    raw = nexis.nexis(d["y"], d["t"], d["z"][:, pool], alpha=0.05, max_rounds=20, rho=0.5,
                      adjust="FWER", backward=False, terminal_filter=True, pvalue_fn=fn)
    assert raw.selected == ref.selected
    assert np.allclose(raw.pvalues, ref.pvalues, rtol=1e-9, atol=1e-14)


def test_min_support_default():
    """discover's default is 4: a sparse village dummy whose four cells span exactly 4
    villages is tested by default and dropped with min_support=5."""
    d = multilevel_data()
    t_vil = pd.Series(d["t"]).groupby(d["levels"]["village"]).first()
    on = list(t_vil.index[t_vil == 1][:4]) + list(t_vil.index[t_vil == 0][:4])
    extra = pd.DataFrame({"s4": np.isin(np.arange(40), on).astype(float)},
                         index=pd.Index(np.arange(40), name="village"))
    data, _ = village_study(pre_extra=src(extra, "extra", "village", "pre", support="binary"))
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
    units, rsrc, score = rdd_units()
    units.loc[[0, 1], "score"] = np.nan
    t01 = units.loc[[0, 1], "T"]
    data3 = _warns(lambda: Dataset(units, rsrc, Discontinuity("T", "score", "hh")),
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
    srcs = [s if s.level != "row" else like(s, s.data.loc[keep]) for s in data_na.sources]
    ref = discover(Dataset(data.units.loc[keep], srcs, data.design), "y")
    assert res.modifiers == ref.modifiers and np.array_equal(res.selection.pvalues, ref.selection.pvalues)
    assert "outcome y missing for 5 units" in _printed(data_na.info)


def test_missing_candidate_excluded():
    data, d = village_study()
    rows = data.sources[2].data.copy()
    rows.iloc[[3, 7], 1] = np.nan                           # r2 missing for 2 units
    srcs = data.sources[:2] + [like(data.sources[2], rows)] + data.sources[3:]
    data_na = Dataset(data.units, srcs, data.design)
    assert data_na.missing.to_dict() == {"r2": 2}
    assert "candidates with missing values (excluded from the pool): r2 (2)" in _printed(data_na.info)
    res = _warns(lambda: discover(data_na, "y"), "1 candidates have missing values", "r2 (2)")
    tab = res.table().set_index("covariate")
    assert tab.loc["r2", "status"] == "excluded: 2 missing values" and tab.loc["r2", "missing"] == 2
    assert "r2" not in res.tested
    # the same as without the column
    srcs_wo = data.sources[:2] + [like(data.sources[2], rows[["r1"]])] + data.sources[3:]
    ref = discover(Dataset(data.units, srcs_wo, data.design), "y")
    assert res.tested == ref.tested and res.modifiers == ref.modifiers
    assert np.array_equal(res.selection.pvalues, ref.selection.pvalues)
    # missing only where the outcome is missing too: the column stays in the pool
    y = d["y"].copy()
    y[[3, 7]] = np.nan
    out = src(pd.DataFrame({"y": y}, index=data.units.index), "endline", "row", "post")
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
    assert (row.covariate, row.source, row.top, row.bottom) == ("v1", "satellite", top, bottom)
    assert out.labels == {"v1": f"v1: top {['tile_' + str(i) for i in top]} vs bottom {['tile_' + str(i) for i in bottom]}"}
    assert out.items["v1"] == ([f"tile_{i}" for i in top], [f"tile_{i}" for i in bottom])
    text = repr(out)
    assert text.startswith("Interpretation of 1 learned modifier:\n  v1 (satellite): v1: top ['tile_")
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
    # the same study with the village table declared hand-crafted: nothing to interpret
    srcs = [like(data.sources[0], origin="hand_crafted", raw=None, interpreter=None)]
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
    srcs = [like(data.sources[0], raw=imgs)] + data.sources[1:]
    out = discover(Dataset(data.units, srcs, data.design), "y").interpret(k=3)
    fig = out.show()
    assert len(fig.axes) == 2 * 3
    assert fig.axes[0].get_title(loc="left").startswith("v1: v1: top")
    assert fig.axes[0].images[0].get_array()[0, 0] == out.frame.iloc[0].top[0]
    # raw items that are not images
    _raises(lambda: discover(data, "y").interpret(k=3).show(), "cannot draw a raw item of 'v1' (source 'satellite')",
            "raw items must be image", "tile_", "render=")


def test_interpret_show_render():
    """render= turns raw items matplotlib cannot draw (here 6-band arrays) into images."""
    import matplotlib
    matplotlib.use("Agg")
    data, _ = village_study()
    tiles = {v: np.full((4, 4, 6), float(v) * 1000) for v in range(40)}      # 6 bands, raw reflectance
    rgb = lambda a: (a[..., [3, 2, 1]] / 40000).astype(np.float32)             # noqa: E731

    def study(render, raw=tiles):
        srcs = [like(data.sources[0], raw=raw, render=render)] + data.sources[1:]
        return discover(Dataset(data.units, srcs, data.design), "y").interpret(k=3)

    out = study(rgb)
    fig = out.show()
    v = out.frame.iloc[0].top[0]
    assert np.allclose(fig.axes[0].images[0].get_array()[0, 0], v * 1000 / 40000)
    assert study(lambda a: (a[..., 0] / 40000 * 255).astype(np.uint8)).show() is not None
    # without render: a clear error naming the source
    _raises(lambda: study(None).show(), "cannot draw a raw item of 'v1' (source 'satellite')", "(4, 4, 6)",
            "render=")
    # a render returning something that is not drawable
    _raises(lambda: study(lambda a: a).show(), "render of source 'satellite'", "shape (4, 4, 6)")
    _raises(lambda: study(lambda a: a[..., :3]).show(), "render of source 'satellite'", "floats in [0, 1]")
    _raises(lambda: study(lambda a: a[..., :3].astype(int)).show(), "render of source 'satellite'", "dtype int64")
    _raises(lambda: study(lambda a: list(a)).show(), "render of source 'satellite'", "numpy array")
    _raises(lambda: study(lambda a: a["band"]).show(), "render of source 'satellite' failed", "'v1'")
    # render: learned columns only, and callable
    _raises(lambda: like(data.sources[0], render="rgb"), "render must be a callable")
    _raises(lambda: like(data.sources[1], render=rgb), "render= apply to learned columns only")


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
