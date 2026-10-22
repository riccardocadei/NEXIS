"""How treatment was assigned: the study designs a `nexis.Dataset` declares.

A design names the treatment and the levels it was assigned and sampled at.  It tells
`nexis.discover` how to cluster each interaction test (at the coarsest of the assignment
level, the sampling level and the level where the candidate varies) and which strata
enter as fixed effects, and its checks run when the Dataset is built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _check_binary(t: pd.Series, name: str) -> None:
    if t.isna().any():
        raise ValueError(f"treatment {name!r} is missing for {int(t.isna().sum())} units")
    bad = ~t.isin([0, 1])
    if bad.any():
        raise ValueError(f"treatment {name!r} must be 0/1; found {sorted(t[bad].unique())[:5]}")
    if t.nunique() < 2:
        raise ValueError(f"treatment {name!r} takes a single value: no treated or no control units")


def _check_constant_within(t: pd.Series, by: pd.Series, name: str, level: str) -> str:
    n = t.groupby(by.values).nunique()
    bad = n[n > 1]
    if len(bad):
        raise ValueError(f"treatment {name!r} varies within {len(bad)} {level} units "
                         f"(e.g. {list(bad.index[:5])}); it must be constant within the "
                         f"assignment level {level!r}")
    return f"treatment constant within {level}"


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


class Design(ABC):
    """A treatment-assignment design.

    Subclasses: Randomized, Discontinuity.  (Observational designs, a bandwidth around
    the cutoff and fuzzy discontinuities are not implemented.)
    """

    treatment: str
    assigned_at: str
    sampled_by: str | None
    stratified_by: str | None = None

    @property
    def ids(self) -> list[str]:
        """The level names the design refers to."""
        return [x for x in (self.assigned_at, self.sampled_by, self.stratified_by) if x]

    @property
    def variables(self) -> list[str]:
        """The non-id variables of the design (never candidates)."""
        return [self.treatment]

    @abstractmethod
    def check(self, frame: pd.DataFrame) -> list[str]:
        """Check the design on `frame` (one row per unit: the design variables and ids).
        Raise ValueError on failure; return the checks that passed, one line each."""

    @abstractmethod
    def describe(self) -> str:
        """The design in one paragraph of plain English."""

    def info(self) -> None:
        print(f"{self!r}\n  {self.describe()}")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if v is not None)
        return f"{type(self).__name__}({args})"


class Randomized(Design):
    """Treatment randomized across `assigned_at` units, optionally within strata
    (`stratified_by`, entered as fixed effects), on units possibly sampled by
    `sampled_by` clusters."""

    def __init__(self, treatment: str, assigned_at: str, stratified_by: str | None = None,
                 sampled_by: str | None = None):
        self.treatment, self.assigned_at = treatment, assigned_at
        self.stratified_by, self.sampled_by = stratified_by, sampled_by

    def check(self, frame: pd.DataFrame) -> list[str]:
        t = frame[self.treatment]
        _check_binary(t, self.treatment)
        done = [_check_constant_within(t, frame[self.assigned_at], self.treatment, self.assigned_at)]
        if self.stratified_by:
            n = frame.groupby(self.assigned_at)[self.stratified_by].nunique()
            if (n > 1).any():
                raise ValueError(f"{int((n > 1).sum())} {self.assigned_at} units span several "
                                 f"{self.stratified_by} strata (e.g. {list(n[n > 1].index[:5])})")
            done.append(f"each {self.assigned_at} lies in one {self.stratified_by} stratum")
        return done

    def describe(self) -> str:
        s = f"Treatment {self.treatment!r} randomized across {self.assigned_at} units"
        if self.stratified_by:
            s += f" within {self.stratified_by} strata (fixed effects in every test)"
        if self.sampled_by:
            s += f"; units sampled by {self.sampled_by}"
        return s + "."


class Discontinuity(Design):
    """Sharp regression discontinuity: treatment is decided by `running_variable`
    crossing a cutoff.  The cutoff and its direction are inferred from the data; a
    given `cutoff` must agree with them."""

    def __init__(self, treatment: str, running_variable: str, assigned_at: str,
                 sampled_by: str | None = None, cutoff: float | None = None):
        self.treatment, self.running_variable = treatment, running_variable
        self.assigned_at, self.sampled_by, self.cutoff = assigned_at, sampled_by, cutoff
        self.direction: str | None = None              # "<=" or ">=", set by check()
        self.cutoff_interval: tuple[float, float] | None = None

    @property
    def variables(self) -> list[str]:
        return [self.treatment, self.running_variable]

    def check(self, frame: pd.DataFrame) -> list[str]:
        t, r, rv = frame[self.treatment], frame[self.running_variable], self.running_variable
        _check_binary(t, self.treatment)
        if r.isna().any():
            raise ValueError(f"running variable {rv!r} is missing for {int(r.isna().sum())} units")
        r1, r0 = r[t == 1], r[t == 0]
        if r1.median() <= r0.median():               # treated below the cutoff
            direction, lo, hi = "<=", r1.max(), r0.min()
        else:
            direction, lo, hi = ">=", r0.max(), r1.min()
        if lo >= hi:
            bad = _misclassified(r, t, direction)
            ex = ", ".join(f"{i} ({rv}={_fmt(v)}, {self.treatment}={int(t[i])})" for i, v in bad.head(5).items())
            raise ValueError(f"not a sharp discontinuity in {rv!r}: treated and control overlap on "
                             f"[{_fmt(hi)}, {_fmt(lo)}]; the best cutoff still misclassifies "
                             f"{len(bad)} units, e.g. {ex}")
        side = "[lo, hi)" if direction == "<=" else "(lo, hi]"
        if self.cutoff is not None:
            c = self.cutoff
            ok = lo <= c < hi if direction == "<=" else lo < c <= hi
            if not ok:
                raise ValueError(f"cutoff {c} disagrees with the data: treated iff {rv} {direction} "
                                 f"cutoff needs a cutoff in {side.replace('lo', _fmt(lo)).replace('hi', _fmt(hi))}")
        self.direction, self.cutoff_interval = direction, (float(lo), float(hi))
        interval = side.replace("lo", _fmt(lo)).replace("hi", _fmt(hi))
        msg = f"sharp: treated iff {rv} {direction} cutoff, cutoff in {interval}"
        if self.cutoff is not None:
            msg += f" (given {self.cutoff})"
        return [msg, _check_constant_within(t, frame[self.assigned_at], self.treatment, self.assigned_at)]

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

    def __repr__(self) -> str:
        args = dict(treatment=self.treatment, running_variable=self.running_variable,
                    assigned_at=self.assigned_at, sampled_by=self.sampled_by, cutoff=self.cutoff)
        return "Discontinuity(" + ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None) + ")"

