"""Regression cases for the nexis library: small synthetic data, fixed seeds.

Every case returns a JSON-serialisable dict (selected sets, p-values, t-statistics).
tests/make_expected.py freezes them in tests/expected.json; tests/test_regression.py
checks that the current code still returns the same values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexis import marginal_select, nexis  # noqa: E402
from nexis.checks import randomization_test, wild_cluster_bootstrap, worst_subset  # noqa: E402
from nexis.design import LevelAwareTest, block_fe, build_rule, codes  # noqa: E402
from nexis.effects import ate_ols  # noqa: E402


def flat_data(n=300, m=30, seed=0):
    """Row-level data: modifiers 0 and 3, a main effect of column 5, clusters of 10."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, m))
    z[:, 7] = (rng.random(n) < 0.3).astype(float)          # binary column
    t = rng.integers(0, 2, n).astype(float)
    cluster = np.repeat(np.arange(n // 10), 10)
    y = (0.5 * t + 0.8 * t * z[:, 0] + 0.6 * t * z[:, 3] + 0.7 * z[:, 5]
         + rng.normal(size=n))
    return y, t, z, cluster


def multilevel_data(seed=1):
    """2-level design: 40 villages x 8 rows in 8 districts (blocks); T at the village."""
    rng = np.random.default_rng(seed)
    n_vil, per = 40, 8
    vil = np.repeat(np.arange(n_vil), per)
    dist = vil // 5
    n = len(vil)
    t_vil = np.zeros(n_vil)
    for b in range(8):                                       # 2 or 3 treated per block
        idx = np.arange(5 * b, 5 * b + 5)
        t_vil[rng.choice(idx, size=2 + b % 2, replace=False)] = 1.0
    t = t_vil[vil]
    v1 = rng.normal(size=n_vil)[vil]                         # village level, modifier
    v2 = rng.normal(size=n_vil)[vil]                         # village level, null
    v3 = (rng.random(n_vil) < 0.15).astype(float)[vil]       # sparse village binary
    d1 = rng.normal(size=8)[dist]                            # district level
    r1, r2 = rng.normal(size=n), rng.normal(size=n)          # row level
    z = np.column_stack([v1, v2, v3, d1, r1, r2])
    y = 0.3 * t + 0.9 * t * v1 + 0.5 * r1 + rng.normal(size=n) + 0.5 * rng.normal(size=n_vil)[vil]
    levels = {"district": codes(dist), "village": codes(vil), "row": np.arange(n)}
    return dict(y=y, t=t, z=z, names=["v1", "v2", "v3", "d1", "r1", "r2"], published=["v1"],
                levels=levels, order=["district", "village", "row"], assign="village",
                block="district", community=vil)


def _plain(x):
    """Convert numpy scalars and arrays inside dicts / lists to plain Python."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def _res(r):
    return _plain(dict(selected=r.selected, pvalues=r.pvalues, method=r.method,
                       feature_names=r.feature_names, metadata=r.metadata))


def case_default():
    y, t, z, _ = flat_data()
    return _res(nexis(y, t, z))


def case_paper_algorithm():
    y, t, z, _ = flat_data()
    return _res(nexis(y, t, z, backward=False, terminal_filter=True))


def case_cluster():
    y, t, z, cl = flat_data()
    return _res(nexis(y, t, z, cluster=cl, backward=False, terminal_filter=True))


def case_hc1():
    y, t, z, _ = flat_data()
    return _res(nexis(y, t, z, hc1=True))


def case_pathwise():
    y, t, z, _ = flat_data()
    return _res(nexis(y, t, z, backward_gate="pathwise"))


def case_gcm():
    y, t, z, _ = flat_data(n=200, m=10)
    return _res(nexis(y, t, z, test="GCM: quadratic", nuisance="poly2"))


def case_pcm():
    y, t, z, _ = flat_data(n=200, m=10)
    return _res(nexis(y, t, z, test="PCM: quadratic"))


def case_marginal():
    y, t, z, cl = flat_data()
    out = {}
    for adj in (None, "FWER", "FDR"):
        out[str(adj)] = _res(marginal_select(y, t, z, adjust=adj))
    out["FWER_cluster"] = _res(marginal_select(y, t, z, adjust="FWER", cluster=cl))
    out["FWER_groups"] = _res(marginal_select(y, t, z, adjust="FWER",
                                              groups={"a": list(range(10)), "b": list(range(10, 30))}))
    return out


def case_level_aware():
    d = multilevel_data()
    keys, parts, info = build_rule(d)
    fn = LevelAwareTest(keys, parts, fe=block_fe(d["levels"][d["block"]]))
    pool = [j for j, i in enumerate(info) if i["testable"]]
    r = nexis(d["y"], d["t"], d["z"][:, pool], pvalue_fn=fn.subset(pool),
              backward=False, terminal_filter=True)
    marg = fn(d["y"], d["t"], d["z"], [], list(range(d["z"].shape[1])), return_tstats=True)
    return _plain(dict(keys=keys, info=info, pool=pool, result=_res(r),
                       marginal_p=marg[0], marginal_t=marg[1]))


def case_checks():
    d = multilevel_data()
    ri = randomization_test(d, [], 0, n_perm=199, seed=0)
    wb = wild_cluster_bootstrap(d, [], 0, d["levels"]["village"], n_boot=199, seed=0)
    keys, parts, _ = build_rule(d)
    fn = LevelAwareTest(keys, parts)
    ws = worst_subset(fn, d["y"], d["t"], d["z"], [0, 1, 4], 0, 0.05 / 6)
    est, se, n = ate_ols(d["y"], d["t"], d["z"])
    return _plain(dict(ri=ri, wild=wb, worst=ws, ate=[est, se, n]))


CASES = {name[5:]: fn for name, fn in sorted(globals().items()) if name.startswith("case_")}
