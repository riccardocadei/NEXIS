"""Sources of covariates and the Dataset that joins them onto the units of a study.

A Source is a table indexed by the ids of one level (its index name, e.g. household_id
or village_id) and measured at one time relative to treatment:

  pre      before treatment: candidate effect modifiers
  during   while treatment ran: excluded unless discover(..., include_during=True)
  post     after treatment: outcomes, never searched

Its origin (measured | predicted | learned) is metadata and never changes the search;
learned features (e.g. sparse-autoencoder atoms of satellite images) carry the raw items
behind each id and an interpreter that describes a selected feature.

Missing values never change the analysis silently:

  * a missing id (in a source index or a level column of the units), or a unit whose id
    has no row in a source, is an error (add a row of NaN to declare missing values);
  * a unit with a missing treatment or running variable is dropped, with a warning;
  * a candidate with a missing value (among the units analysed) is excluded from the
    pool, listed with its count in Dataset.info() and Result.table(), with a warning
    at discover();
  * a unit with a missing outcome is dropped by discover(), with a warning;
  * infinite values are an error.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .assignment import Design

TIMES = ("pre", "during", "post")
ORIGINS = ("measured", "predicted", "learned")


class Interpreter(Protocol):
    """Anything with describe(top, bottom) -> str."""

    def describe(self, top: list, bottom: list) -> str:
        """Describe what the raw items of the highest-valued ids (`top`) share against
        those of the lowest-valued ids (`bottom`)."""


class Source:
    """One table of variables at one level and one time.

    data: DataFrame indexed by the ids of its level; the index name is the level and
        must be the index name or a column of the Dataset's units.  Numeric columns;
        NaN marks a missing value.
    time: "pre" | "during" | "post".
    origin: "measured" | "predicted" | "learned"; learned needs `raw` (a mapping or
        callable id -> raw item, e.g. an image path, read lazily) and `interpreter`
        (an object with describe(top, bottom) -> str).
    name: shown in tables; defaults to "<level>_<time>".
    """

    def __init__(self, data: pd.DataFrame | pd.Series, time: str, origin: str = "measured",
                 raw: Mapping | Callable[[Any], Any] | None = None, interpreter: Interpreter | None = None,
                 name: str | None = None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Source data must be a pandas DataFrame, got {type(data).__name__}")
        if time not in TIMES:
            raise ValueError(f"time must be one of {TIMES}, got {time!r}")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        if data.index.name is None:
            raise ValueError("Source data needs a named index: its name is the level of the "
                             "source (e.g. data.index.name = 'village_id')")
        if data.index.hasnans:
            raise ValueError(f"source indexed by {data.index.name!r} has {int(data.index.isna().sum())} "
                             "missing ids")
        if data.index.has_duplicates:
            dup = data.index[data.index.duplicated()].unique()[:5]
            raise ValueError(f"source indexed by {data.index.name!r} repeats ids {list(dup)}")
        if data.columns.has_duplicates:
            raise ValueError(f"duplicate columns {list(data.columns[data.columns.duplicated()])}")
        nonnum = [c for c in data.columns if not (pd.api.types.is_numeric_dtype(data[c])
                                                  or pd.api.types.is_bool_dtype(data[c]))]
        if nonnum:
            raise ValueError(f"columns must be numeric (encode categories as dummies): {nonnum}")
        data = data.astype(float)
        inf = [c for c in data.columns if np.isinf(data[c]).any()]
        if inf:
            raise ValueError(f"infinite values in {inf}")
        if origin == "learned":
            if raw is None or interpreter is None:
                raise ValueError("a learned source needs raw= (id -> raw item) and interpreter=")
            if not callable(getattr(interpreter, "describe", None)):
                raise TypeError("interpreter must have a describe(top, bottom) -> str method")
        elif raw is not None or interpreter is not None:
            raise ValueError("raw= and interpreter= apply to learned sources only")
        self.data, self.time, self.origin = data, time, origin
        self.raw, self.interpreter = raw, interpreter
        self.level: str = data.index.name
        self.name = name or f"{self.level}_{time}"

    def raw_items(self, ids) -> list:
        """The raw items behind `ids` (learned sources only)."""
        get = self.raw if callable(self.raw) else self.raw.__getitem__
        return [get(i) for i in ids]

    def __repr__(self) -> str:
        return (f"Source({self.name!r}: {self.data.shape[1]} columns x {len(self.data)} "
                f"{self.level}, time={self.time!r}, origin={self.origin!r})")


class Dataset:
    """Units of a study, the sources joined onto them, and the assignment design.

    units: DataFrame with one row per unit; its index name is the unit level, and it
        holds the ids of every coarser level a source or the design refers to (and, if
        not in a source, the treatment and running variable).
    sources: Source objects; each is broadcast onto the units through its level.
    design: a nexis Design (Randomized, Discontinuity).

    Every unit must have a row in every source; source rows of ids outside the units
    are ignored.  Units with a missing design variable are dropped with a warning
    (`dropped_units`); the design checks then run on the rest (`checks`).  Design
    variables are taken out of the candidate pool (`removed`).
    """

    def __init__(self, units: pd.DataFrame, sources: list[Source], design: Design):
        if not isinstance(design, Design):
            raise TypeError("design must be a nexis Design (Randomized, Discontinuity)")
        if units.index.name is None:
            raise ValueError("units needs a named index: its name is the unit level")
        if units.index.hasnans:
            raise ValueError(f"units have {int(units.index.isna().sum())} missing ids")
        if units.index.has_duplicates:
            raise ValueError(f"units repeat ids {list(units.index[units.index.duplicated()].unique()[:5])}")
        self.sources, self.design = list(sources), design
        self.unit: str = units.index.name

        # ids of every level: the unit, the design ids and every source level
        ids = {}
        for lv in dict.fromkeys([self.unit] + design.ids + [s.level for s in self.sources]):
            if lv == self.unit:
                ids[lv] = pd.Series(units.index, index=units.index)
            elif lv in units.columns:
                if units[lv].isna().any():
                    raise ValueError(f"units miss the {lv!r} id of {int(units[lv].isna().sum())} units "
                                     f"(e.g. {list(units.index[units[lv].isna()][:5])})")
                ids[lv] = units[lv]
            else:
                raise ValueError(f"level {lv!r} is neither the index name nor a column of units "
                                 f"(columns: {list(units.columns)})")

        # broadcast every source onto the units
        seen: dict[str, str] = {}
        cols, self.column_source = [], {}
        for s in self.sources:
            for c in s.data.columns:
                if c in seen:
                    raise ValueError(f"column {c!r} is in sources {seen[c]!r} and {s.name!r}")
                if c in units.columns:
                    raise ValueError(f"column {c!r} of source {s.name!r} is also a column of units")
                seen[c] = s.name
            key = ids[s.level]
            missing = pd.Index(key.unique()).difference(s.data.index)
            if len(missing):
                raise ValueError(f"source {s.name!r} has no row for {len(missing)} of {key.nunique()} "
                                 f"{s.level} ids (e.g. {list(missing[:5])}); add NaN rows to declare "
                                 "missing values")
            cols.append(s.data.reindex(key.values).set_axis(units.index))
            self.column_source.update({c: s for c in s.data.columns})
        frame = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=units.index)

        # design variables, from units or a source
        fr = {}
        for v in design.variables:
            if v in units.columns:
                if not (pd.api.types.is_numeric_dtype(units[v]) or pd.api.types.is_bool_dtype(units[v])):
                    raise ValueError(f"design variable {v!r} must be numeric, got dtype {units[v].dtype}")
                fr[v] = units[v].astype(float)
            elif v in frame.columns:
                fr[v] = frame[v]
            else:
                raise ValueError(f"design variable {v!r} is neither a column of units nor of a source")
        design_frame = pd.DataFrame(fr, index=units.index)

        # units with a missing design variable are dropped
        self.dropped_units: dict[str, pd.Index] = {}
        gone = design_frame.isna().any(axis=1)
        if gone.any():
            notes = []
            for v in design.variables:
                miss = design_frame.index[design_frame[v].isna()]
                if len(miss):
                    self.dropped_units[f"missing {v}"] = miss
                    notes.append(f"{len(miss)} miss {v!r}{_arms(design_frame, miss, design.treatment, v)}")
            warnings.warn(f"{int(gone.sum())} of {len(units)} units are dropped: {'; '.join(notes)} "
                          f"(e.g. {list(units.index[gone][:5])})", stacklevel=2)
            units, frame, design_frame = units[~gone], frame[~gone], design_frame[~gone]
            ids = {k: v[~gone.to_numpy()] for k, v in ids.items()}
        for lv in design.ids:
            design_frame[lv] = ids[lv]
        self.units, self.frame, self.design_frame = units, frame, design_frame
        order = sorted(ids, key=lambda k: (k == self.unit, ids[k].nunique(), k))
        self.levels: dict[str, pd.Series] = {k: ids[k] for k in order}   # coarse -> fine
        self.removed = [v for v in design.variables if v in frame.columns]
        self.checks = design.check(design_frame)

    # ── columns by role ──────────────────────────────────────────────────────

    def _columns(self, times: tuple[str, ...]) -> list[str]:
        return [c for c in self.frame.columns
                if self.column_source[c].time in times and c not in self.removed]

    @property
    def outcomes(self) -> list[str]:
        """The post columns."""
        return self._columns(("post",))

    def candidates(self, include_during: bool = False) -> list[str]:
        """The candidate pool: pre columns (and during ones if asked), minus design
        variables.  Columns with missing values are listed here; discover() excludes them."""
        return self._columns(("pre", "during") if include_during else ("pre",))

    @property
    def missing(self) -> pd.Series:
        """Number of units with a missing value, per pre/during/post column that has any."""
        n = self.frame[self._columns(TIMES)].isna().sum()
        return n[n > 0]

    # ── description ──────────────────────────────────────────────────────────

    def info(self) -> None:
        """Print units per level, sources, candidates, outcomes, missing values and the
        design with the checks it passed."""
        print(f"Dataset: {len(self.units):,} {self.unit}")
        for why, idx in self.dropped_units.items():
            print(f"  dropped: {len(idx):,} units ({why})")
        print("  units per level: " + ", ".join(f"{k} {v.nunique():,}" for k, v in self.levels.items()))
        print("  sources:")
        w = max([len(s.name) for s in self.sources] + [4])
        for s in self.sources:
            print(f"    {s.name:{w}s}  level {s.level:12s} {s.time:7s} {s.origin:9s} {s.data.shape[1]:5d} cols")
        for time in ("pre", "during"):
            cs = self._columns((time,))
            by = pd.Series([self.column_source[c].origin for c in cs], dtype=object).value_counts()
            note = "" if time == "pre" else " (excluded unless discover(include_during=True))"
            print(f"  candidates {time}: {len(cs)}" + (f" ({', '.join(f'{k} {v}' for k, v in by.items())})" if len(cs) else "") + note)
        print(f"  outcomes (post): {', '.join(self.outcomes) or 'none'}")
        miss = self.missing
        cand = [c for c in miss.index if c not in self.outcomes]
        if cand:
            print(f"  candidates with missing values (excluded from the pool): "
                  + ", ".join(f"{c} ({miss[c]:,})" for c in cand))
        for c in self.outcomes:
            if c in miss.index:
                print(f"  outcome {c} missing for {miss[c]:,} units (dropped when it is the outcome)")
        print(f"  design: {self.design!r}")
        for c in self.checks:
            print(f"    passed: {c}")
        if self.removed:
            print(f"  removed from the pool (design variables): {', '.join(self.removed)}")

    def __repr__(self) -> str:
        return (f"Dataset({len(self.units)} {self.unit}, {len(self.sources)} sources, "
                f"{len(self.candidates())} candidates, {len(self.outcomes)} outcomes, {self.design!r})")


def _arms(frame: pd.DataFrame, idx: pd.Index, treatment: str, missing: str) -> str:
    """' (a treated, b control)' for the units idx, unless the treatment is what is missing."""
    if missing == treatment:
        return ""
    t = frame.loc[idx, treatment]
    return f" ({int((t == 1).sum())} treated, {int((t == 0).sum())} control)"
