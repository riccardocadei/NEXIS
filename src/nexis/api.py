"""discover(): search a Dataset for the effect modifiers of its treatment on an outcome.

The search is the paper's algorithm: nexis(backward=False, terminal_filter=True) with
the FWER (Bonferroni) gates, rho = 0.5 and the linear T x Z_j interaction test, made
level-aware by the design (nexis.design.LevelAwareTest):

  * candidate j is clustered at the coarsest of the assignment level, the sampling level
    and the level where Z_j varies (HC1 when that is the unit itself), detected as the
    coarsest level within which Z_j is constant among the levels nested with its declared
    level (coarser or finer; nexis.data.nested_order), so a detected level is always at
    least as coarse as the declared one;
  * strata of a stratified randomization enter as fixed effects;
  * support rule (nexis.design.build_rule): a candidate declared binary or sparse
    (cells split at zero: active / inactive), or whose modal value holds >= 10% of the
    units, is dropped before the search when one of the four cells (Z_j at / off its
    mode) x (treated / control)
    spans fewer than `min_support` clusters; candidates with missing values are
    excluded too.  Bonferroni counts the rest (m).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assignment import MIN_CLUSTERS
from .data import MASS_SUPPORTS, Dataset, nested_order
from .design import LevelAwareTest, block_fe, build_rule, codes, is_coarser
from .interpret import Interpretation, extremes
from .search import SelectionResult, nexis

RHO, MAX_ROUNDS = 0.5, 20
MIN_SUPPORT = 4


@dataclass
class Result:
    """What discover() found.

    modifiers: the selected covariates.
    tested: the pool the search ran on (m = len(tested)); `selection` is the raw
        SelectionResult of the search, whose indices refer to `tested`.
    units: the units analysed (n = len(units), n_treated + n_control).
    table(): one row per candidate (see discover).
    """

    data: Dataset
    outcome: str
    alpha: float
    min_support: int
    modifiers: list[str]
    selection: SelectionResult
    tested: list[str]
    units: pd.Index
    n_treated: int
    n_control: int
    _table: pd.DataFrame = field(repr=False)

    @property
    def n(self) -> int:
        return len(self.units)

    def table(self) -> pd.DataFrame:
        """One row per candidate: covariate, source (its name), level (declared),
        varies_at (detected), time, declared support kind (`kind`), origin, domain,
        clustered_at, n_clusters, support (clusters in the smallest cell, if the support
        rule applies), missing, p_value (given the selected set, or given the other
        modifiers for a modifier), status, and the source's access and modality."""
        return self._table.copy()

    def interpret(self, k: int = 5, seed: int = 0) -> Interpretation:
        """Describe every selected learned modifier with its source's interpreter, from
        the raw items of the k ids (at the source's level, among the units analysed)
        with the highest values and the k with the lowest (ties broken at random with
        `seed`).  Empty, with a message, when no learned modifier was selected."""
        learned = [c for c in self.modifiers if self.data.column_source[c].origin[c] == "learned"]
        rows, items = [], {}
        for c in learned:
            src = self.data.column_source[c]
            ids = pd.Index(self.data.levels[src.level].loc[self.units].unique())
            top, bottom = extremes(src.data.loc[src.data.index.intersection(ids), c], k, seed)
            items[c] = (src.raw_items(top), src.raw_items(bottom))
            rows.append(dict(covariate=c, source=src.name, label=src.interpreter.describe(c, *items[c]),
                             top=top, bottom=bottom))
        frame = pd.DataFrame(rows, columns=["covariate", "source", "label", "top", "bottom"])
        message = "" if learned else (
            f"no learned modifier was selected (modifiers: {', '.join(self.modifiers) or 'none'}); "
            "nothing to interpret")
        render = {c: self.data.column_source[c].render for c in learned}
        return Interpretation(frame, items, message, render=render)

    def __repr__(self) -> str:
        return (f"Result(outcome={self.outcome!r}, n={self.n} ({self.n_treated} treated), "
                f"m={len(self.tested)}, alpha={self.alpha}, modifiers={self.modifiers})")


def discover(data: Dataset, outcome: str, alpha: float = 0.05, min_support: int = MIN_SUPPORT,
             include_during: bool = False) -> Result:
    """Select the effect modifiers of the design's treatment on `outcome` (a post column),
    with family-wise error rate `alpha` over the tested pool.

    min_support: the support rule's minimum number of clusters per (side of a sparse
        candidate) x (arm) cell (see the module docstring).
    include_during: also search the columns measured during treatment; do it only if
        treatment cannot have changed them (excluded with a warning otherwise).

    Units with a missing outcome are dropped and candidates with a missing value are
    excluded, each with a warning; Result.table() lists every candidate and its status.
    The tested candidates are standardised over the analysed units before the search
    (the support rule sees the raw values): p-values are those of the raw columns up to
    round-off, and a column in the span of the selected ones gets p = 1 whatever its
    scale.
    """
    if outcome not in data.outcomes:
        if outcome in data.frame.columns:
            raise ValueError(f"outcome {outcome!r} comes from source {data.column_source[outcome].name!r} "
                             f"(time={data.column_source[outcome].time!r}); outcomes are post columns: "
                             f"{data.outcomes}")
        raise ValueError(f"unknown outcome {outcome!r}; outcomes are the post columns: {data.outcomes}")
    des = data.design
    during = [c for c in data.candidates(True) if data.column_source[c].time == "during"]
    if during and not include_during:
        warnings.warn(f"{len(during)} columns measured during treatment are excluded from the pool "
                      f"({', '.join(during[:5])}{', ...' if len(during) > 5 else ''}); pass "
                      "include_during=True if treatment cannot have changed them", stacklevel=2)
    names = data.candidates(include_during)
    if not names:
        raise ValueError("no candidate covariates")

    # units: those with an outcome
    y_all = data.frame[outcome].to_numpy(float)
    t_all = data.design_frame[des.treatment].to_numpy(float)
    rows = np.isfinite(y_all)
    if not rows.all():
        gone = ~rows
        warnings.warn(f"{int(gone.sum())} of {len(rows)} units have no {outcome!r} and are dropped "
                      f"({int((t_all[gone] == 1).sum())} treated, {int((t_all[gone] == 0).sum())} control)",
                      stacklevel=2)
    n1 = int((t_all[rows] == 1).sum())
    if n1 in (0, int(rows.sum())):
        raise ValueError(f"no {'control' if n1 else 'treated'} unit has an outcome {outcome!r}")

    # candidates: those without missing values among these units
    z_all = data.frame[names].to_numpy(float)[rows]
    n_miss = np.isnan(z_all).sum(0)
    if n_miss.any():
        bad = [f"{c} ({k})" for c, k in zip(names, n_miss) if k]
        warnings.warn(f"{len(bad)} candidates have missing values and are excluded from the pool: "
                      f"{', '.join(bad[:10])}{', ...' if len(bad) > 10 else ''}", stacklevel=2)

    levels = {k: codes(v.to_numpy()[rows]) for k, v in data.levels.items()}
    assign = des.assigned_at
    if des.sampled_by and is_coarser(levels[des.sampled_by], levels[assign]):
        assign = des.sampled_by
    d = dict(y=y_all[rows], t=t_all[rows], z=z_all, names=names, levels=levels,
             order=list(levels), assign=assign)
    declared = [data.column_source[c].support[c] in MASS_SUPPORTS for c in names]
    # the rule per declared level, each detecting among the levels nested with it
    keys, parts, info = [None] * len(names), {}, [None] * len(names)
    by_level: dict[str, list[int]] = {}
    for j, c in enumerate(names):
        by_level.setdefault(data.column_source[c].level, []).append(j)
    for lv, js in by_level.items():
        sub = dict(d, z=d["z"][:, js], names=[names[j] for j in js], order=nested_order(levels, lv))
        k_, p_, i_ = build_rule(sub, min_support, mass_point=[declared[j] for j in js])
        for jj, j in enumerate(js):
            keys[j], info[j] = k_[jj], i_[jj]
        parts.update(p_)
    pool = [j for j, i in enumerate(info) if i["testable"]]
    if not pool:
        raise ValueError("no candidate passes the support rule")
    few = [f"{names[j]} ({info[j]['G']} {keys[j]})" for j in pool
           if parts[keys[j]] is not None and info[j]["G"] < MIN_CLUSTERS]
    if few:
        warnings.warn(f"{len(few)} tested candidates are clustered with fewer than {MIN_CLUSTERS} "
                      f"clusters, where cluster-robust p-values are unreliable: "
                      f"{', '.join(few[:10])}{', ...' if len(few) > 10 else ''}", stacklevel=2)

    fe = block_fe(levels[des.stratified_by]) if des.stratified_by else None
    fn = LevelAwareTest(keys, parts, fe=fe).subset(pool)
    # standardised over the analysed units: OLS t-statistics are invariant to an affine
    # change of a column, and the tolerances of the test become scale-free (a column in
    # the span of the conditioning set then reliably gets p = 1 whatever its units)
    y, t, z = d["y"], d["t"], d["z"][:, pool]
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    sel = nexis(y, t, z, alpha=alpha, max_rounds=MAX_ROUNDS, rho=RHO, adjust="FWER",
                backward=False, terminal_filter=True, pvalue_fn=fn)
    sel.feature_names = [names[j] for j in pool]
    S = list(sel.selected)

    # p-value of every tested candidate given the selected set (minus itself)
    p = np.full(len(names), np.nan)
    others = [j for j in range(len(pool)) if j not in S]
    if others:
        p[np.asarray(pool)[others]] = fn(y, t, z, S, others)[others]
    for j in S:
        p[pool[j]] = fn(y, t, z, [k for k in S if k != j], [j])[j]

    chosen = {pool[j] for j in S}
    recs = []
    for j, (c, i) in enumerate(zip(names, info)):
        src = data.column_source[c]
        hc1 = parts[keys[j]] is None
        if j in chosen:
            status = "selected"
        elif i["testable"]:
            status = "not selected"
        elif n_miss[j]:
            status = f"excluded: {n_miss[j]} missing values"
        elif np.nanstd(d["z"][:, j]) == 0:
            status = "dropped: constant"
        else:
            status = f"dropped: support {i['support']} < {min_support} clusters"
        recs.append(dict(covariate=c, source=src.name, level=src.level, varies_at=i["level"],
                         time=src.time, kind=src.support[c], origin=src.origin[c], domain=src.domain[c],
                         clustered_at="none (HC1)" if hc1 else keys[j].replace("+", " + "),
                         n_clusters=int(len(y)) if hc1 else i["G"], support=i["support"],
                         missing=int(n_miss[j]), p_value=p[j], status=status))
    if during and not include_during:
        va = data.varies_at(during)
        for c in during:
            src = data.column_source[c]
            recs.append(dict(covariate=c, source=src.name, level=src.level, varies_at=va[c], time=src.time,
                             kind=src.support[c], origin=src.origin[c], domain=src.domain[c], clustered_at=None,
                             n_clusters=None, support=None, missing=int(data.frame.loc[rows, c].isna().sum()),
                             p_value=np.nan, status="excluded: measured during treatment"))
    for r in recs:
        src = data.column_source[r["covariate"]]
        r.update(access=src.access, modality=src.modality)
    tab = pd.DataFrame(recs).astype({"n_clusters": "Int64", "support": "Int64"})
    return Result(data=data, outcome=outcome, alpha=alpha, min_support=min_support,
                  modifiers=[names[pool[j]] for j in S], selection=sel,
                  tested=[names[j] for j in pool], units=data.units.index[rows],
                  n_treated=n1, n_control=int(rows.sum()) - n1, _table=tab)
