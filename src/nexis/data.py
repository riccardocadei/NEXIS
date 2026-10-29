"""Sources of covariates and the Dataset that joins them onto the units of a study.

A Source is a table of variables at one level, measured at one time relative to
treatment, with a declared support.  These three attributes change the analysis:

  level    the ids its rows are indexed by (e.g. village_id): it is joined onto the
           units through the column of that name in `units` (or their index).  The
           level where each candidate actually varies is detected from the data and
           reported (Dataset.info(), Result.table() `varies_at`); the clustering uses
           the coarsest one.  A covariate shared by groups that are not an existing
           level (a market catchment, a raster cell) gets its own level: an id column
           in `units` and a Source indexed by it.
  time     pre      before treatment: candidate effect modifiers
           during   while treatment ran: excluded unless discover(..., include_during=True)
           post     after treatment: outcomes, never searched
  support  the values a column can take, checked at construction (NaN ignored):
           binary {0, 1}, count (integers >= 0), nonnegative (reals >= 0), sparse
           (>= 0, zero the most frequent value, e.g. SAE activations), continuous
           (any finite real).  Binary, count and sparse columns always get the
           support rule of nexis.discover; the others get it when the data show a
           mass point.

Its origin (measured | predicted | learned), modality, access and reference are
descriptive and never change the search; learned features (e.g. sparse-autoencoder
atoms of satellite images) carry the raw items behind each id and an interpreter that
describes a selected feature.

What is checked against the data at construction: support (against the values), level
(against the index name here, against the id column of the units in Dataset) and
origin="learned" (needs raw and interpreter).  Time, modality and access are
declarations only: the data cannot tell when a variable was measured, so declare it
with care (a "pre" column caused by treatment biases the search).

A list of Sources is the single declaration of a study's data: an application keeps
one Source per source and builds the Dataset from that list.

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
from .design import codes, variation_level

TIMES = ("pre", "during", "post")
ORIGINS = ("measured", "predicted", "learned")
ACCESS = ("public", "restricted", "proprietary")
SUPPORTS = ("binary", "count", "nonnegative", "sparse", "continuous")
MASS_SUPPORTS = ("binary", "count", "sparse")      # always get the support rule


class Interpreter(Protocol):
    """Anything with describe(top, bottom) -> str."""

    def describe(self, top: list, bottom: list) -> str:
        """Describe what the raw items of the highest-valued ids (`top`) share against
        those of the lowest-valued ids (`bottom`)."""


class Source:
    """One table of variables at one level and one time.

    data: DataFrame indexed by the ids of `level`, numeric columns; NaN marks a missing
        value.  Its index must be named `level` or be unnamed (it is then named so).
    level: the level of the rows; the index name or a column of the Dataset's units.
    time: "pre" | "during" | "post".
    support: one of SUPPORTS for every column, or a dict giving one per column (every
        column exactly); a column whose values violate it is an error.
    origin: "measured" | "predicted" | "learned"; learned needs `raw` (a mapping or
        callable id -> raw item, e.g. an image path, read lazily) and `interpreter`
        (an object with describe(top, bottom) -> str).
    modality: what the raw data are, e.g. "survey", "image", "raster", "tabular" (an
        open vocabulary, any non-empty str), or None.
    access: "public" | "restricted" | "proprietary", or None (unknown).
    reference: where the source is documented (e.g. a section of a data README), or None.
        modality, access and reference are shown by Dataset.info() and Result.table()
        and never used by the search.
    name: shown in tables; defaults to "<level>_<time>".
    """

    def __init__(self, data: pd.DataFrame | pd.Series, *, level: str, time: str,
                 support: str | Mapping[str, str], origin: str = "measured",
                 modality: str | None = None, access: str | None = None, reference: str | None = None,
                 raw: Mapping | Callable[[Any], Any] | None = None, interpreter: Interpreter | None = None,
                 name: str | None = None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Source data must be a pandas DataFrame, got {type(data).__name__}")
        if not isinstance(level, str) or not level:
            raise TypeError(f"level must be a level name (str), got {level!r}")
        if time not in TIMES:
            raise ValueError(f"time must be one of {TIMES}, got {time!r}")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")
        if data.index.name is None:
            data = data.rename_axis(level)
        elif data.index.name != level:
            raise ValueError(f"source data is indexed by {data.index.name!r} but level={level!r}: "
                             "index the table by the ids of its level (or leave the index unnamed)")
        if data.index.hasnans:
            raise ValueError(f"source indexed by {level!r} has {int(data.index.isna().sum())} missing ids")
        if data.index.has_duplicates:
            dup = data.index[data.index.duplicated()].unique()[:5]
            raise ValueError(f"source indexed by {level!r} repeats ids {list(dup)}")
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
        if isinstance(support, str):
            support = dict.fromkeys(data.columns, support)
        elif isinstance(support, Mapping):
            extra, lack = set(support) - set(data.columns), [c for c in data.columns if c not in support]
            if extra or lack:
                raise ValueError(f"support must cover every column exactly: missing {lack[:10]}, "
                                 f"unknown {sorted(map(str, extra))[:10]}")
            support = {c: support[c] for c in data.columns}
        else:
            raise TypeError(f"support must be a str or a dict column -> str, got {type(support).__name__}")
        for c, kind in support.items():
            _check_support(data[c], kind)
        if modality is not None and not (isinstance(modality, str) and modality):
            raise ValueError(f"modality must be a non-empty str (e.g. 'survey', 'image'), got {modality!r}")
        if access is not None and access not in ACCESS:
            raise ValueError(f"access must be one of {ACCESS} or None, got {access!r}")
        if reference is not None and not isinstance(reference, str):
            raise TypeError(f"reference must be a str, got {type(reference).__name__}")
        if origin == "learned":
            if raw is None or interpreter is None:
                raise ValueError("a learned source needs raw= (id -> raw item) and interpreter=")
            if not callable(getattr(interpreter, "describe", None)):
                raise TypeError("interpreter must have a describe(top, bottom) -> str method")
        elif raw is not None or interpreter is not None:
            raise ValueError("raw= and interpreter= apply to learned sources only")
        self.data, self.level, self.time, self.origin = data, level, time, origin
        self.support: dict[str, str] = support
        self.modality, self.access, self.reference = modality, access, reference
        self.raw, self.interpreter = raw, interpreter
        self.name = name or f"{level}_{time}"

    def raw_items(self, ids) -> list:
        """The raw items behind `ids` (learned sources only)."""
        get = self.raw if callable(self.raw) else self.raw.__getitem__
        return [get(i) for i in ids]

    def __repr__(self) -> str:
        kinds = ", ".join(sorted(set(self.support.values())))
        return (f"Source({self.name!r}: {self.data.shape[1]} columns x {len(self.data)} "
                f"{self.level}, time={self.time!r}, support={kinds!r}, origin={self.origin!r})")


def _check_support(col: pd.Series, kind: str) -> None:
    """Raise ValueError when the non-missing values of col violate the support `kind`."""
    if kind not in SUPPORTS:
        raise ValueError(f"support of {col.name!r} must be one of {SUPPORTS}, got {kind!r}")
    x = col.dropna()
    if kind == "binary":
        bad = ~x.isin([0, 1])
    elif kind == "count":
        bad = (x < 0) | (x != np.round(x))
    elif kind in ("nonnegative", "sparse"):
        bad = x < 0
    else:
        return
    if bad.any():
        vals = sorted(x[bad].unique())[:5]
        raise ValueError(f"column {col.name!r} is declared {kind} but {int(bad.sum())} values are not "
                         f"(e.g. {vals})")
    if kind == "sparse" and len(x):
        freq = x.value_counts()
        if freq.get(0.0, 0) < freq.iloc[0]:
            raise ValueError(f"column {col.name!r} is declared sparse but zero ({freq.get(0.0, 0)} rows) "
                             f"is not its most frequent value ({freq.index[0]:g}: {freq.iloc[0]} rows)")


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
        cand = set(self._columns(("pre", "during")))
        flat = []
        for s in self.sources:                        # at the source's own level, its ids in use
            rows_ = s.data.loc[pd.Index(self.levels[s.level].unique())]
            flat += [f"{c} ({s.support[c]})" for c in s.data.columns if c in cand
                     and s.support[c] not in MASS_SUPPORTS and _repeated_mode(rows_[c])]
        if flat:
            warnings.warn(f"{len(flat)} columns declared without a mass point repeat a value over >= 10% "
                          f"of their ids, so they get the support rule from the data: {', '.join(flat[:10])}"
                          f"{', ...' if len(flat) > 10 else ''}; declare them binary, count or sparse",
                          stacklevel=2)

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

    def varies_at(self, columns: list[str] | None = None) -> pd.Series:
        """The coarsest level within which each column (default: every candidate,
        during ones included) is constant, as discover() detects it."""
        cols = self.candidates(True) if columns is None else columns
        lv = {k: codes(v.to_numpy()) for k, v in self.levels.items()}
        return pd.Series({c: variation_level(self.frame[c].to_numpy(), lv, list(lv)) for c in cols},
                         dtype=object)

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
            kinds = pd.Series(s.support).value_counts()
            sup = ", ".join(k if len(kinds) == 1 else f"{k} {v}" for k, v in kinds.items())
            meta = ", ".join(f"{k} {v}" for k, v in (("modality", s.modality), ("access", s.access),
                                                       ("ref", s.reference)) if v is not None)
            print(f"    {s.name:{w}s}  level {s.level:12s} {s.time:7s} {s.origin:9s} {s.data.shape[1]:5d} cols"
                  f"  support {sup}" + (f"  [{meta}]" if meta else ""))
        for time in ("pre", "during"):
            cs = self._columns((time,))
            by = pd.Series([self.column_source[c].origin for c in cs], dtype=object).value_counts()
            note = "" if time == "pre" else " (excluded unless discover(include_during=True))"
            print(f"  candidates {time}: {len(cs)}" + (f" ({', '.join(f'{k} {v}' for k, v in by.items())})" if len(cs) else "") + note)
        va = self.varies_at()
        coarser = [f"{c} ({self.column_source[c].level} -> {v})" for c, v in va.items()
                   if v != self.column_source[c].level]
        if coarser:
            print(f"  candidates varying at a coarser level than declared (detected; the clustering "
                  f"uses it): {len(coarser)}: {', '.join(coarser[:10])}{', ...' if len(coarser) > 10 else ''}")
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


def _repeated_mode(col: pd.Series) -> bool:
    """True when the modal value of col (NaN ignored) is repeated and holds >= 10% of it."""
    n = col.value_counts()
    return len(n) > 0 and n.iloc[0] >= max(2, 0.1 * n.sum())


def _arms(frame: pd.DataFrame, idx: pd.Index, treatment: str, missing: str) -> str:
    """' (a treated, b control)' for the units idx, unless the treatment is what is missing."""
    if missing == treatment:
        return ""
    t = frame.loc[idx, treatment]
    return f" ({int((t == 1).sum())} treated, {int((t == 0).sum())} control)"
