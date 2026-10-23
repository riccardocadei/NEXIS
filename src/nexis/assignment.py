"""How treatment was assigned: the study designs a `nexis.Dataset` declares.

A design names the treatment and the levels it was assigned and sampled at.  It tells
`nexis.discover` how to cluster each interaction test (at the coarsest of the assignment
level, the sampling level and the level where the candidate varies) and which strata
enter as fixed effects.  Its checks run when the Dataset is built: they raise a
ValueError when the data contradict the design and warn when the design is weak (few
clusters, strata without contrast).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

MIN_CLUSTERS = 20      # fewer clusters than this: cluster-robust p-values are unreliable


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _examples(index, k: int = 5) -> str:
    ids = list(index[:k])
    return ", ".join(map(str, ids)) + (", ..." if len(index) > k else "")


class Design(ABC):
    """A treatment-assignment design: the base of Randomized and Discontinuity.

    treatment: the 0/1 treatment column.
    assigned_at: the level treatment was assigned at (a level name of the Dataset).
    sampled_by: the level units were sampled by (e.g. villages drawn first), or None.
    stratified_by: the strata of a stratified assignment, or None.
    """

    treatment: str
    assigned_at: str
    sampled_by: str | None = None
    stratified_by: str | None = None

    @property
    def ids(self) -> list[str]:
        """The level names the design refers to."""
        return [x for x in (self.assigned_at, self.sampled_by, self.stratified_by) if x]

    @property
    def variables(self) -> list[str]:
        """The non-id variables of the design (never candidates)."""
        return [self.treatment]

    def check(self, frame: pd.DataFrame) -> list[str]:
        """Check the design on `frame` (one row per unit: the design variables and ids,
        no missing values).  Raise ValueError on failure, warn on a weak design, and
        return the checks that passed, one line each."""
        t = frame[self.treatment]
        done = [self._check_treatment(t)]
        done += self._check_specific(frame)
        done.append(self._check_constant_within(t, frame[self.assigned_at]))
        if self.sampled_by:
            done.append(self._check_nested(frame, self.assigned_at, self.sampled_by))
        if self.stratified_by:
            done.append(self._check_strata(frame))
        done.append(self._check_clusters(frame))
        return done

    def _check_specific(self, frame: pd.DataFrame) -> list[str]:
        return []

    def _check_treatment(self, t: pd.Series) -> str:
        bad = ~t.isin([0, 1])
        if bad.any():
            raise ValueError(f"treatment {self.treatment!r} must be 0/1; found values "
                             f"{sorted(t[bad].unique())[:5]} (units {_examples(t.index[bad])})")
        n1 = int((t == 1).sum())
        if n1 in (0, len(t)):
            raise ValueError(f"treatment {self.treatment!r} has no {'control' if n1 else 'treated'} "
                             f"units: both arms are needed")
        return f"treatment 0/1: {n1:,} treated, {len(t) - n1:,} control"

    def _check_constant_within(self, t: pd.Series, by: pd.Series) -> str:
        n = t.groupby(by.values).nunique()
        bad = n[n > 1]
        if len(bad):
            raise ValueError(f"treatment {self.treatment!r} varies within {len(bad)} {self.assigned_at} "
                             f"units (e.g. {_examples(bad.index)}); it must be constant within the "
                             f"assignment level {self.assigned_at!r}")
        return f"treatment constant within {self.assigned_at}"

    @staticmethod
    def _check_nested(frame: pd.DataFrame, a: str, b: str) -> str:
        a_in_b = (frame[b].groupby(frame[a].values).nunique() == 1).all()
        b_in_a = (frame[a].groupby(frame[b].values).nunique() == 1).all()
        if not (a_in_b or b_in_a):
            raise ValueError(f"assigned_at {a!r} and sampled_by {b!r} are not nested: some {a} "
                             f"units span several {b} units and vice versa")
        return f"{a} nested in {b}" if a_in_b else f"{b} nested in {a}"

    def _check_strata(self, frame: pd.DataFrame) -> str:
        a, s = self.assigned_at, self.stratified_by
        n = frame.groupby(a)[s].nunique()
        if (n > 1).any():
            raise ValueError(f"{int((n > 1).sum())} {a} units span several {s} strata "
                             f"(e.g. {_examples(n[n > 1].index)})")
        arms = frame.groupby(s)[self.treatment].nunique()
        flat = arms[arms < 2]
        if len(flat):
            warnings.warn(f"{len(flat)} of {len(arms)} {s} strata hold a single arm "
                          f"(e.g. {_examples(flat.index)}): with strata fixed effects they carry "
                          "no treatment contrast", stacklevel=4)
        return f"each {a} lies in one {s} stratum"

    def _check_clusters(self, frame: pd.DataFrame) -> str:
        """Warn when the base clustering (the coarser of assigned_at and sampled_by) has
        fewer than MIN_CLUSTERS clusters."""
        lv = self.assigned_at
        if self.sampled_by and frame[self.sampled_by].nunique() < frame[lv].nunique():
            lv = self.sampled_by
        G = frame[lv].nunique()
        if G < MIN_CLUSTERS:
            warnings.warn(f"only {G} {lv} clusters (< {MIN_CLUSTERS}): cluster-robust p-values "
                          "are unreliable with this few clusters", stacklevel=4)
        return f"{G:,} {lv} clusters"

    @abstractmethod
    def describe(self) -> str:
        """The design in one paragraph of plain English."""

    def info(self) -> None:
        """Print the design and its description."""
        print(f"{self!r}\n  {self.describe()}")

    def _args(self) -> dict:
        return dict(treatment=self.treatment, assigned_at=self.assigned_at,
                    stratified_by=self.stratified_by, sampled_by=self.sampled_by)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._args().items() if v is not None)
        return f"{type(self).__name__}({args})"


class Randomized(Design):
    """Treatment randomized across `assigned_at` units, optionally within strata
    (`stratified_by`, entered as fixed effects in every test), on units possibly
    sampled by `sampled_by` clusters."""

    def __init__(self, treatment: str, assigned_at: str, stratified_by: str | None = None,
                 sampled_by: str | None = None):
        self.treatment, self.assigned_at = treatment, assigned_at
        self.stratified_by, self.sampled_by = stratified_by, sampled_by

    def describe(self) -> str:
        s = f"Treatment {self.treatment!r} randomized across {self.assigned_at} units"
        if self.stratified_by:
            s += f" within {self.stratified_by} strata (fixed effects in every test)"
        if self.sampled_by:
            s += f"; units sampled by {self.sampled_by}"
        return s + "."


class Discontinuity(Design):
    """Sharp regression discontinuity: treatment is decided by `running_variable`
    crossing a cutoff, at the `assigned_at` level, on units possibly sampled by
    `sampled_by` clusters.  The cutoff interval and the direction (treated below or
    above) are inferred from the data; a given `cutoff` must lie in that interval."""

    def __init__(self, treatment: str, running_variable: str, assigned_at: str,
                 sampled_by: str | None = None, cutoff: float | None = None):
        self.treatment, self.running_variable = treatment, running_variable
        self.assigned_at, self.sampled_by, self.cutoff = assigned_at, sampled_by, cutoff
        self.direction: str | None = None              # "<=" or ">=", set by check()
        self.cutoff_interval: tuple[float, float] | None = None

    @property
    def variables(self) -> list[str]:
        return [self.treatment, self.running_variable]

    def _check_specific(self, frame: pd.DataFrame) -> list[str]:
        t, r, rv = frame[self.treatment], frame[self.running_variable], self.running_variable
        if not np.isfinite(r).all():
            raise ValueError(f"running variable {rv!r} is infinite for units {_examples(r.index[~np.isfinite(r)])}")
        r1, r0 = r[t == 1], r[t == 0]
        if r1.median() <= r0.median():               # treated below the cutoff
            direction, lo, hi = "<=", r1.max(), r0.min()
        else:
            direction, lo, hi = ">=", r0.max(), r1.min()
        if lo >= hi:
            bad = _misclassified(r, t, direction)
            n1 = int((t[bad.index] == 1).sum())
            ex = ", ".join(f"{i} ({rv}={_fmt(v)}, {self.treatment}={int(t[i])})" for i, v in bad.head(5).items())
            raise ValueError(f"not a sharp discontinuity in {rv!r}: treated and control overlap on "
                             f"[{_fmt(hi)}, {_fmt(lo)}]; the best cutoff still misclassifies "
                             f"{len(bad)} units ({n1} treated, {len(bad) - n1} control), e.g. {ex}. "
                             "Fix or drop these units (a fuzzy design is not supported)")
        interval = f"[{_fmt(lo)}, {_fmt(hi)})" if direction == "<=" else f"({_fmt(lo)}, {_fmt(hi)}]"
        msg = f"sharp: treated iff {rv} {direction} cutoff, cutoff in {interval}"
        if self.cutoff is not None:
            c = self.cutoff
            ok = lo <= c < hi if direction == "<=" else lo < c <= hi
            if not ok:
                at = r[r == c]
                note = (f"; {len(at)} units lie exactly at {rv} = {c}, all "
                        f"{'treated' if (t[at.index] == 1).all() else 'control'}" if len(at) else "")
                raise ValueError(f"cutoff {c} disagrees with the data: treated iff {rv} {direction} "
                                 f"cutoff needs a cutoff in {interval}{note}")
            n_at = int((r == c).sum())
            msg += f" (given {c}" + (f"; {n_at} units exactly at the cutoff, treated" if n_at else "") + ")"
        self.direction, self.cutoff_interval = direction, (float(lo), float(hi))
        return [msg]

    def describe(self) -> str:
        s = (f"Treatment {self.treatment!r} decided by {self.running_variable!r} crossing a cutoff "
             f"(sharp), assigned at the {self.assigned_at} level")
        if self.direction:
            lo, hi = self.cutoff_interval
            interval = f"[{_fmt(lo)}, {_fmt(hi)})" if self.direction == "<=" else f"({_fmt(lo)}, {_fmt(hi)}]"
            s += f"; treated iff {self.running_variable} {self.direction} cutoff, cutoff in {interval}"
        if self.sampled_by:
            s += f"; units sampled by {self.sampled_by}"
        return s + "."

    def _args(self) -> dict:
        return dict(treatment=self.treatment, running_variable=self.running_variable,
                    assigned_at=self.assigned_at, sampled_by=self.sampled_by, cutoff=self.cutoff)


def _misclassified(r: pd.Series, t: pd.Series, direction: str) -> pd.Series:
    """Units on the wrong side of the cutoff that misclassifies the fewest units."""
    below = t if direction == "<=" else 1 - t           # 1 where the unit should lie below
    o = np.argsort(r.to_numpy(), kind="stable")
    b = below.to_numpy()[o]
    # cut after position k: errors = (should-be-below units after k) + (above units up to k)
    err = np.concatenate([[b.sum()], b.sum() - np.cumsum(b) + np.cumsum(1 - b)])
    k = int(err.argmin())
    wrong = np.zeros(len(b), bool)
    wrong[o[k:]] = b[k:] == 1
    wrong[o[:k]] = b[:k] == 0
    return r[wrong]
