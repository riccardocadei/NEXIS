"""discover(): search a Dataset for the effect modifiers of its treatment on an outcome.

The search is the paper's algorithm: nexis(backward=False, terminal_filter=True) with
the FWER (Bonferroni) gates, rho = 0.5 and the linear T x Z_j interaction test, made
level-aware by the design (nexis.design.LevelAwareTest):

  * candidate j is clustered at the coarsest of the assignment level, the sampling level
    and the level where Z_j varies (HC1 when that is the unit itself);
  * strata of a stratified randomization enter as fixed effects;
  * support rule: a candidate with a mass point whose minority side holds fewer than
    `min_support` clusters in either arm is dropped before the search (nexis.design.
    build_rule), as are candidates with missing values; Bonferroni counts the rest.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset
from .design import MIN_SUPPORT, LevelAwareTest, block_fe, build_rule, codes, is_coarser
from .search import SelectionResult, nexis

logger = logging.getLogger("nexis")

RHO, MAX_ROUNDS = 0.5, 20


@dataclass
class Result:
    """What discover() found.  `modifiers` are the selected covariates; `selection` is
    the raw SelectionResult of the search, indexing the tested pool `tested`."""

    data: Dataset
    outcome: str
    alpha: float
    modifiers: list[str]
    selection: SelectionResult
    tested: list[str]
    _table: pd.DataFrame = field(repr=False)

    def table(self) -> pd.DataFrame:
        """One row per candidate: where it comes from, how it was tested, its p-value
        given the selected set (given the other modifiers, for a modifier) and status."""
        return self._table.copy()

    def interpret(self, k: int = 5) -> dict[str, str]:
        """Describe every selected learned modifier with its source's interpreter, from
        the raw items of the k ids (at the source's level) with the highest and the k
        with the lowest values.  {} when no learned modifier was selected."""
        learned = [c for c in self.modifiers if self.data.column_source[c].origin == "learned"]
        if not learned:
            logger.warning("interpret(): no learned modifier was selected; nothing to interpret")
            return {}
        out = {}
        for c in learned:
            src = self.data.column_source[c]
            ids = pd.Index(self.data.levels[src.level].unique())
            col = src.data.loc[src.data.index.intersection(ids), c].dropna().sort_values(kind="stable")
            top, bottom = col.index[::-1][:k], col.index[:k]
            out[c] = src.interpreter.describe(src.raw_items(top), src.raw_items(bottom))
        return out

    def __repr__(self) -> str:
        return (f"Result(outcome={self.outcome!r}, alpha={self.alpha}, m={len(self.tested)}, "
                f"modifiers={self.modifiers})")


def discover(data: Dataset, outcome: str, alpha: float = 0.05, min_support: int = MIN_SUPPORT,
             include_during: bool = False) -> Result:
    """Select the effect modifiers of the design's treatment on `outcome` (a post column).

    Units with a missing outcome are dropped.  Columns measured during treatment are
    excluded with a warning unless include_during=True: include them only if treatment
    cannot have changed them.  Family-wise error rate alpha over the tested pool.
    """
    if outcome not in data.outcomes:
        if outcome in data.frame.columns:
            raise ValueError(f"outcome {outcome!r} comes from source {data.column_source[outcome].name!r} "
                             f"(time={data.column_source[outcome].time!r}); outcomes are post columns: "
                             f"{data.outcomes}")
        raise ValueError(f"unknown outcome {outcome!r}; outcomes are the post columns: {data.outcomes}")
    during = [c for c in data.candidates(True) if data.column_source[c].time == "during"]
    if during and not include_during:
        warnings.warn(f"{len(during)} columns measured during treatment are excluded from the pool "
                      f"({', '.join(during[:5])}{', ...' if len(during) > 5 else ''}); pass "
                      "include_during=True if treatment cannot have changed them", stacklevel=2)
    names = data.candidates(include_during)
    if not names:
        raise ValueError("no candidate covariates")

    y_all = data.frame[outcome].to_numpy(float)
    rows = np.isfinite(y_all)
    if not rows.all():
        logger.warning("discover(): %d of %d units have no %s and are dropped",
                       int((~rows).sum()), len(rows), outcome)
    des = data.design
    levels = {k: codes(v.to_numpy()[rows]) for k, v in data.levels.items()}
    assign = des.assigned_at
    if des.sampled_by and is_coarser(levels[des.sampled_by], levels[assign]):
        assign = des.sampled_by
    d = dict(y=y_all[rows], t=data.design_frame[des.treatment].to_numpy(float)[rows],
             z=data.frame[names].to_numpy(float)[rows], names=names, levels=levels,
             order=list(levels), assign=assign)
    keys, parts, info = build_rule(d, min_support)
    pool = [j for j, i in enumerate(info) if i["testable"]]
    if not pool:
        raise ValueError("no candidate passes the support rule")
    fe = block_fe(levels[des.stratified_by]) if des.stratified_by else None
    test = LevelAwareTest(keys, parts, fe=fe)
    fn = test.subset(pool)
    y, t, z = d["y"], d["t"], d["z"][:, pool]
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
        elif not np.isfinite(d["z"][:, j]).all():
            status = "dropped: missing values"
        else:
            status = f"dropped: support {i['support']} < {min_support} clusters"
        recs.append(dict(covariate=c, source=src.name, level=src.level, varies_at=i["level"],
                         origin=src.origin, time=src.time,
                         clustered_at="none (HC1)" if hc1 else keys[j].replace("+", " + "),
                         n_clusters=int(len(y)) if hc1 else i["G"], p_value=p[j], status=status))
    if during and not include_during:
        for c in during:
            src = data.column_source[c]
            recs.append(dict(covariate=c, source=src.name, level=src.level, varies_at=None,
                             origin=src.origin, time=src.time, clustered_at=None, n_clusters=None,
                             p_value=np.nan, status="excluded: measured during treatment"))
    tab = pd.DataFrame(recs).astype({"n_clusters": "Int64"})
    modifiers = [names[pool[j]] for j in S]
    return Result(data=data, outcome=outcome, alpha=alpha, modifiers=modifiers, selection=sel,
                  tested=[names[j] for j in pool], _table=tab)
