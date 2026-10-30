"""Sources of covariates and the Dataset that joins them onto the units of a study.

A Source is one data source (a survey, a satellite product, a climate or conflict
database): a table of variables at one level, measured at one time relative to
treatment.  Every column is characterised by eight attributes:

  name      the data source, e.g. "CHIRPS", "UNICEF household survey" (unique in a
            Dataset)
  domain    what the variable is about (demographics, housing, economy, accessibility,
            urbanization, environment, security, health, ...; an open vocabulary of
            lowercase identifiers)
  level     the ids its rows are indexed by (e.g. village_id)
  support   the values a column can take
  origin    raw | hand_crafted | learned
  access    public | restricted | proprietary
  time      historic | pre | during | post
  modality  the kind of raw data the columns are extracted from (optional)

Three of them change the analysis:

  level    the source is joined onto the units through the column of that name in
           `units` (or their index).  The level where each candidate actually varies is
           detected from the data and reported (Dataset.info(), Result.table()
           `varies_at`), among the levels nested with the declared one (nested_order),
           so it is never finer than declared; the clustering uses the coarsest of it,
           the assignment and the sampling level.  A covariate shared by
           groups that are not an existing level (a market catchment, a raster cell)
           gets its own level: an id column in `units` and a Source indexed by it.
  time     historic  long-run character predating the baseline (e.g. climate normals):
                     a candidate, exactly like pre
           pre       measured before treatment: a candidate effect modifier
           during    measured while treatment ran: excluded unless
                     discover(..., include_during=True)
           post      after treatment: outcomes, never searched
  support  binary {0, 1}, count (integers >= 0), nonnegative (reals >= 0), sparse
           (>= 0, zero the most frequent value, e.g. SAE activations), continuous (any
           finite real).  Binary and sparse columns always get the support rule of
           nexis.discover (cells split at zero); the others get it when the data show a
           mass point.

Three guide the interpretation of what is found and never enter the search: the source
`name`, the `domain` and the `origin` (raw = as recorded by the source; hand_crafted =
derived by the analyst from raw data, e.g. an index or a distance; learned = a feature
learned by a model, e.g. a sparse-autoencoder atom of satellite images, which needs the
raw items behind each id (`raw`) and an `interpreter` that describes a selected feature).
Every candidate competes in one pool whatever its origin.

Two are bookkeeping, shown in the tables only: `access` (public = downloadable without
an account; restricted = needs a registration, an account or a licence; proprietary =
obtained directly from the data owner, not downloadable) and `modality` (see
MODALITIES).

What is checked against the data: support (against the values), level (against the
index name here, against the id column of the units in Dataset), raw (covers every id of
the source when it is a mapping), and, at Dataset construction and over the units, that
no candidate is constant or identical to another (errors), and warnings when a
candidate is an exact linear combination of others (with an intercept, over the columns
without missing values: e.g. a household size that is the sum of its age-group counts;
such a column gets p = 1 once the others are selected) or when two candidates correlate
above 0.999.  Time, domain, origin, access and modality are declarations
only: the data cannot tell when a variable was measured, so declare it with care (a
"pre" column caused by treatment biases the search).

A list of Sources is the single declaration of a study's data: an application keeps one
Source per data source and builds the Dataset from that list.

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

import inspect
import re
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd
import scipy.linalg

from .assignment import Design
from .design import codes, is_coarser, variation_level

TIMES = ("historic", "pre", "during", "post")
CANDIDATE_TIMES = ("historic", "pre", "during")
ORIGINS = ("raw", "hand_crafted", "learned")
ACCESS = ("public", "restricted", "proprietary")
SUPPORTS = ("binary", "count", "nonnegative", "sparse", "continuous")
MASS_SUPPORTS = ("binary", "sparse")      # always get the support rule, split at zero
MODALITIES = {
    "survey": "answers to a questionnaire or census (households, individuals, communities)",
    "image": "images (satellite or aerial tiles, photographs), e.g. behind learned features",
    "raster": "gridded geospatial layers (climate, population, land cover, night lights) "
              "sampled at the units' locations",
    "events": "dated, geolocated event records (conflicts, disasters), aggregated per id",
    "time_series": "regular series per id or station (prices, weather-station readings)",
    "text": "documents or free text (reports, transcripts, news)",
}
CORRELATION_WARNING = 0.999
RANK_TOLERANCE = 1e-9   # a pivot of the standardised pool below this fraction of the largest is zero
_DOMAIN = re.compile(r"[a-z][a-z0-9_]*")


class Interpreter(Protocol):
    """Anything with describe(name, top, bottom) -> str."""

    def describe(self, name: str, top: list, bottom: list) -> str:
        """Describe what the raw items of the highest-valued ids (`top`) of the covariate
        `name` share against those of its lowest-valued ids (`bottom`)."""


class Source:
    """One data source: a table of variables at one level and one time.

    data: DataFrame indexed by the ids of `level`, numeric (or boolean) columns with
        non-empty str names; NaN marks a missing value.  Its index must be named `level`
        or be unnamed (it is then named so).
    name: the data source, e.g. "CHIRPS" (non-empty str, unique in a Dataset).
    level: the level of the rows; the index name or a column of the Dataset's units.
    time: "historic" | "pre" | "during" | "post".
    support: one of SUPPORTS for every column, or a dict giving one per column (every
        column exactly); a column whose values violate it is an error.
    origin: "raw" | "hand_crafted" | "learned", for every column or per column (a dict
        covering every column exactly).  Learned columns need `raw` (a mapping covering
        every id, or a callable id -> raw item, e.g. an image path, read lazily) and
        `interpreter` (an object with describe(name, top, bottom) -> str).
    domain: a lowercase identifier (e.g. "economy", "environment"), for every column or
        per column (a dict covering every column exactly).
    access: "public" | "restricted" | "proprietary".
    modality: the kind of raw data the columns are extracted from, one of MODALITIES,
        or None.
    render: optional, learned columns only: a callable raw item -> displayable image
        (H x W or H x W x 3 array, floats in [0, 1] or uint8), used by
        Interpretation.show() for raw items matplotlib cannot read (e.g. a multi-band
        GeoTIFF tile: pick three bands and rescale).  None: items are drawn as they are
        (image arrays or paths to image files).
    """

    def __init__(self, data: pd.DataFrame | pd.Series, *, name: str, level: str, time: str,
                 support: str | Mapping[str, str], origin: str | Mapping[str, str],
                 domain: str | Mapping[str, str], access: str, modality: str | None = None,
                 raw: Mapping | Callable[[Any], Any] | None = None,
                 interpreter: Interpreter | None = None,
                 render: Callable[[Any], np.ndarray] | None = None):
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"name must be the name of the data source (a non-empty str), got {name!r}")
        where = f"source {name!r}"
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{where}: data must be a pandas DataFrame, got {type(data).__name__}")
        if not isinstance(level, str) or not level:
            raise TypeError(f"{where}: level must be a level name (a non-empty str), got {level!r}")
        _choice(time, TIMES, "time", where)
        _choice(access, ACCESS, "access", where)
        if modality is not None:
            _choice(modality, tuple(MODALITIES), "modality", where, extra=" or None")
        if isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"{where}: data has a MultiIndex; index it by the ids of one level")
        if data.index.name is None:
            data = data.rename_axis(level)
        elif data.index.name != level:
            raise ValueError(f"{where} is indexed by {data.index.name!r} but level={level!r}: "
                             "index the table by the ids of its level (or leave the index unnamed)")
        if data.shape[1] == 0:
            raise ValueError(f"{where} has no columns")
        if len(data) == 0:
            raise ValueError(f"{where} has no rows")
        if data.index.hasnans:
            raise ValueError(f"{where} (indexed by {level!r}) has {int(data.index.isna().sum())} missing ids")
        if data.index.has_duplicates:
            dup = data.index[data.index.duplicated()].unique()[:5]
            raise ValueError(f"{where} (indexed by {level!r}) repeats ids {list(dup)}")
        badname = [c for c in data.columns if not isinstance(c, str) or not c]
        if badname:
            raise TypeError(f"{where}: column names must be non-empty str, got {badname[:10]}")
        if data.columns.has_duplicates:
            raise ValueError(f"{where}: duplicate columns {list(data.columns[data.columns.duplicated()].unique())}")
        nonnum = [c for c in data.columns
                  if not (pd.api.types.is_numeric_dtype(data[c]) or pd.api.types.is_bool_dtype(data[c]))
                  or pd.api.types.is_complex_dtype(data[c])]
        if nonnum:
            raise ValueError(f"{where}: columns must be real numeric or boolean (encode categories as "
                             f"dummies): {nonnum[:10]}")
        data = data.astype(float)
        inf = [c for c in data.columns if np.isinf(data[c]).any()]
        if inf:
            raise ValueError(f"{where}: infinite values in {inf[:10]}")

        support = _per_column(support, data.columns, "support", where)
        for c, kind in support.items():
            _check_support(data[c], kind, where)
        origin = _per_column(origin, data.columns, "origin", where)
        for c, o in origin.items():
            _choice(o, ORIGINS, f"origin of {c!r}", where)
        domain = _per_column(domain, data.columns, "domain", where)
        for c, d in domain.items():
            if not _DOMAIN.fullmatch(d):
                raise ValueError(f"{where}: domain of {c!r} must be a lowercase identifier "
                                 f"(e.g. 'economy', 'environment'), got {d!r}")

        learned = [c for c, o in origin.items() if o == "learned"]
        if learned:
            if raw is None or interpreter is None:
                raise ValueError(f"{where}: learned columns ({_few(learned)}) need raw= (id -> raw item) "
                                 "and interpreter=")
            if not (isinstance(raw, Mapping) or callable(raw)):
                raise TypeError(f"{where}: raw must be a mapping or a callable id -> raw item, "
                                f"got {type(raw).__name__}")
            if isinstance(raw, Mapping):
                lack = [i for i in data.index if i not in raw]
                if lack:
                    raise ValueError(f"{where}: raw has no item for {len(lack)} of {len(data)} {level} ids "
                                     f"(e.g. {lack[:5]})")
            _check_interpreter(interpreter, where)
            if render is not None and not callable(render):
                raise TypeError(f"{where}: render must be a callable raw item -> image array, "
                                f"got {type(render).__name__}")
        elif raw is not None or interpreter is not None or render is not None:
            raise ValueError(f"{where}: raw=, interpreter= and render= apply to learned columns only, "
                             "and no column has origin 'learned'")

        self.data, self.name, self.level, self.time = data, name, level, time
        self.support: dict[str, str] = support
        self.origin: dict[str, str] = origin
        self.domain: dict[str, str] = domain
        self.access, self.modality = access, modality
        self.raw, self.interpreter, self.render = raw, interpreter, render

    def raw_items(self, ids) -> list:
        """The raw items behind `ids` (sources with learned columns only)."""
        get = self.raw if callable(self.raw) else self.raw.__getitem__
        return [get(i) for i in ids]

    def __repr__(self) -> str:
        return (f"Source({self.name!r}: {self.data.shape[1]} columns x {len(self.data)} {self.level}, "
                f"time={self.time!r}, support={_summary(self.support)!r}, "
                f"origin={_summary(self.origin)!r}, domain={_summary(self.domain)!r}, "
                f"access={self.access!r}, modality={self.modality!r})")


def _choice(value, allowed: tuple, what: str, where: str, extra: str = "") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{where}: {what} must be one of {allowed}{extra}, got {type(value).__name__} {value!r}")
    if value not in allowed:
        raise ValueError(f"{where}: {what} must be one of {allowed}{extra}, got {value!r}")


def _per_column(value, columns: pd.Index, what: str, where: str) -> dict[str, str]:
    """{column: str} from one str for every column or a dict covering every column."""
    if isinstance(value, str):
        return dict.fromkeys(columns, value)
    if isinstance(value, Mapping):
        extra = [k for k in value if k not in columns]
        lack = [c for c in columns if c not in value]
        if extra or lack:
            raise ValueError(f"{where}: {what} must cover every column exactly: missing {lack[:10]}, "
                             f"unknown {sorted(map(str, extra))[:10]}")
        out = {c: value[c] for c in columns}
        bad = {c: v for c, v in out.items() if not isinstance(v, str)}
        if bad:
            raise TypeError(f"{where}: {what} values must be str, got {dict(list(bad.items())[:5])}")
        return out
    raise TypeError(f"{where}: {what} must be a str or a dict column -> str, got {type(value).__name__}")


def _check_interpreter(interpreter, where: str) -> None:
    describe = getattr(interpreter, "describe", None)
    if not callable(describe):
        raise TypeError(f"{where}: interpreter must have a describe(name, top, bottom) -> str method")
    try:
        sig = inspect.signature(describe)
    except (TypeError, ValueError):            # builtins without a signature: not checkable
        return
    try:
        sig.bind("name", [], [])
    except TypeError:
        raise TypeError(f"{where}: interpreter.describe must accept (name, top, bottom), "
                        f"its signature is {sig}") from None


def _check_support(col: pd.Series, kind: str, where: str = "source") -> None:
    """Raise ValueError when the non-missing values of col violate the support `kind`."""
    _choice(kind, SUPPORTS, f"support of {col.name!r}", where)
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
        raise ValueError(f"{where}: column {col.name!r} is declared {kind} but {int(bad.sum())} values "
                         f"are not (e.g. {vals})")
    if kind == "sparse" and len(x):
        freq = x.value_counts()
        if freq.get(0.0, 0) < freq.iloc[0]:
            raise ValueError(f"{where}: column {col.name!r} is declared sparse but zero "
                             f"({freq.get(0.0, 0)} rows) is not its most frequent value "
                             f"({freq.index[0]:g}: {freq.iloc[0]} rows)")


class Dataset:
    """Units of a study, the sources joined onto them, and the assignment design.

    units: DataFrame with one row per unit; its index name is the unit level, and it
        holds the ids of every coarser level a source or the design refers to (and, if
        not in a source, the treatment and running variable).
    sources: Source objects with distinct names; each is broadcast onto the units
        through its level.
    design: a nexis Design (Randomized, Discontinuity).

    Every unit must have a row in every source; source rows of ids outside the units
    are ignored.  Units with a missing design variable are dropped with a warning
    (`dropped_units`); the design checks then run on the rest (`checks`).  Design
    variables are taken out of the candidate pool (`removed`).  Over the remaining
    units, a candidate (historic, pre or during column) that is constant or identical to
    another candidate is an error; a candidate that is an exact linear combination of
    other candidates and an intercept (among those without missing values), and two
    candidates with |correlation| above 0.999, give warnings.
    """

    def __init__(self, units: pd.DataFrame, sources: list[Source], design: Design):
        if not isinstance(units, pd.DataFrame):
            raise TypeError(f"units must be a pandas DataFrame, got {type(units).__name__}")
        if not isinstance(design, Design):
            raise TypeError("design must be a nexis Design (Randomized, Discontinuity)")
        if isinstance(sources, Source) or not all(isinstance(s, Source) for s in sources):
            raise TypeError("sources must be a list of nexis Source objects")
        if units.index.name is None:
            raise ValueError("units needs a named index: its name is the unit level")
        if units.index.name in units.columns:
            raise ValueError(f"units has a column named like its index ({units.index.name!r}): "
                             "the unit level is ambiguous; drop or rename the column")
        if units.index.hasnans:
            raise ValueError(f"units have {int(units.index.isna().sum())} missing ids")
        if units.index.has_duplicates:
            raise ValueError(f"units repeat ids {list(units.index[units.index.duplicated()].unique()[:5])}")
        self.sources, self.design = list(sources), design
        names = pd.Index([s.name for s in self.sources])
        if names.has_duplicates:
            raise ValueError(f"source names must be unique, repeated: {list(names[names.duplicated()].unique())}")
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
                if c in units.columns or c == self.unit:
                    raise ValueError(f"column {c!r} of source {s.name!r} is also a column (or the index) "
                                     "of units")
                seen[c] = s.name
            key = ids[s.level]
            missing = pd.Index(key.unique()).difference(s.data.index)
            if len(missing):
                hint = ""
                if len(missing) == key.nunique() and key.dtype != s.data.index.dtype:
                    hint = (f" (the {s.level} ids of units are {key.dtype}, the source index is "
                            f"{s.data.index.dtype})")
                raise ValueError(f"source {s.name!r} has no row for {len(missing)} of {key.nunique()} "
                                 f"{s.level} ids (e.g. {list(missing[:5])}){hint}; add NaN rows to "
                                 "declare missing values")
            cols.append(s.data.reindex(key.values).set_axis(units.index))
            self.column_source.update({c: s for c in s.data.columns})
        frame = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=units.index)

        # design variables, from units or a source
        fr = {}
        for v in design.variables:
            if v in units.columns:
                if not (pd.api.types.is_numeric_dtype(units[v]) or pd.api.types.is_bool_dtype(units[v])) \
                        or pd.api.types.is_complex_dtype(units[v]):
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
        self._check_pool()
        cand = set(self._columns(CANDIDATE_TIMES))
        flat = []
        for s in self.sources:                        # at the source's own level, its ids in use
            rows_ = s.data.loc[pd.Index(self.levels[s.level].unique())]
            flat += [f"{c} ({s.support[c]})" for c in s.data.columns if c in cand
                     and s.support[c] in ("continuous", "nonnegative") and _repeated_mode(rows_[c])]
        if flat:
            warnings.warn(f"{len(flat)} columns declared without a mass point repeat a value over >= 10% "
                          f"of their ids, so they get the support rule from the data: {', '.join(flat[:10])}"
                          f"{', ...' if len(flat) > 10 else ''}; declare them binary or sparse if they are",
                          stacklevel=2)

    def _check_pool(self) -> None:
        """Over the units: no constant candidate, no two identical ones, none identical
        to the treatment (errors); warnings for exact linear dependences and for pairs with
        |correlation| > CORRELATION_WARNING."""
        cand = self._columns(CANDIDATE_TIMES)
        if not cand:
            return
        x = self.frame[cand].to_numpy(float) + 0.0          # a copy; -0.0 becomes 0.0
        nan = np.isnan(x)
        x[nan] = np.nan                                      # one NaN bit pattern
        where = lambda c: f"{c} ({self.column_source[c].name})"   # noqa: E731
        const = []
        for j, c in enumerate(cand):
            v = x[~nan[:, j], j]
            if len(v) == 0:
                const.append(f"{where(c)}: all missing")
            elif v.min() == v.max():
                const.append(f"{where(c)}: always {v[0]:g}" + (f", {int(nan[:, j].sum())} missing"
                                                                 if nan[:, j].any() else ""))
        if const:
            raise ValueError(f"{len(const)} candidate columns are constant over the {len(x):,} units, so "
                             f"they cannot modify the effect: {'; '.join(const[:10])}"
                             f"{'; ...' if len(const) > 10 else ''}; remove them")
        groups: dict[bytes, list[str]] = {}
        for j, c in enumerate(cand):
            groups.setdefault(x[:, j].tobytes(), []).append(c)
        t = self.design_frame[self.design.treatment].to_numpy(float) + 0.0
        as_t = groups.get(t.tobytes(), [])
        if as_t:
            raise ValueError(f"candidate columns identical to the treatment {self.design.treatment!r} over the "
                             f"units (their interaction with it is collinear): {', '.join(map(where, as_t))}")
        same = [g for g in groups.values() if len(g) > 1]
        if same:
            raise ValueError(f"{len(same)} groups of candidate columns are identical over the {len(x):,} "
                             f"units: {'; '.join(' = '.join(map(where, g)) for g in same[:10])}"
                             f"{'; ...' if len(same) > 10 else ''}; keep one of each")
        full = ~nan.any(axis=0)
        sets = dependent_sets(x[:, full], [c for c, f in zip(cand, full) if f])
        if sets:
            warnings.warn(f"{len(sets)} candidate columns are exact linear combinations of other candidates "
                          f"and an intercept over the {len(x):,} units; minimal dependent sets: "
                          f"{'; '.join('{' + ', '.join(map(where, g)) + '}' for g in sets[:5])}"
                          f"{'; ...' if len(sets) > 5 else ''}.  They all stay in the pool and count in "
                          "the number of tested candidates m: once the other columns of a set are "
                          "selected, the remaining one is in the span of the conditioning set, so "
                          "nexis treats it as uninformative (p = 1); which column of a set enters "
                          "first is left to the search", stacklevel=3)
        if nan.any():
            r = pd.DataFrame(x, columns=cand).corr().to_numpy()
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.corrcoef(x, rowvar=False).reshape(len(cand), len(cand))
        hi = np.argwhere(np.triu(np.abs(np.nan_to_num(r)) > CORRELATION_WARNING, k=1))
        if len(hi):
            pairs = [f"{where(cand[a])} ~ {where(cand[b])} (r = {r[a, b]:.4f})" for a, b in hi[:10]]
            warnings.warn(f"{len(hi)} pairs of candidate columns have |correlation| > {CORRELATION_WARNING} "
                          f"over the units: {'; '.join(pairs)}{'; ...' if len(hi) > 10 else ''}; they "
                          "compete for the same signal", stacklevel=3)

    # ── columns by role ──────────────────────────────────────────────────────

    def _columns(self, times: tuple[str, ...]) -> list[str]:
        return [c for c in self.frame.columns
                if self.column_source[c].time in times and c not in self.removed]

    @property
    def outcomes(self) -> list[str]:
        """The post columns."""
        return self._columns(("post",))

    def candidates(self, include_during: bool = False) -> list[str]:
        """The candidate pool: historic and pre columns (and during ones if asked), minus
        design variables.  Columns with missing values are listed here; discover()
        excludes them."""
        return self._columns(CANDIDATE_TIMES if include_during else ("historic", "pre"))

    def varies_at(self, columns: list[str] | None = None) -> pd.Series:
        """The coarsest level within which each column (default: every candidate,
        during ones included) is constant, as discover() detects it: among the levels
        nested with the column's declared level (see nested_order)."""
        cols = self.candidates(True) if columns is None else columns
        lv = {k: codes(v.to_numpy()) for k, v in self.levels.items()}
        order = {}
        for c in cols:
            d = self.column_source[c].level
            if d not in order:
                order[d] = nested_order(lv, d)
        return pd.Series({c: variation_level(self.frame[c].to_numpy(), lv, order[self.column_source[c].level])
                          for c in cols}, dtype=object)

    @property
    def missing(self) -> pd.Series:
        """Number of units with a missing value, per column (design variables aside) that
        has any."""
        n = self.frame[self._columns(TIMES)].isna().sum()
        return n[n > 0]

    # ── description ──────────────────────────────────────────────────────────

    def sources_table(self) -> pd.DataFrame:
        """One row per source: name, level (declared), varies_at (levels its candidate
        columns vary at, detected, with counts when they differ), time, columns,
        support, origin, domain (counts when they differ), access, modality."""
        va = self.varies_at()
        rows = []
        for s in self.sources:
            cand = [c for c in s.data.columns if c in va.index]
            rows.append(dict(source=s.name, level=s.level,
                             varies_at=_summary(va[cand].to_dict()) if cand else "",
                             time=s.time, columns=s.data.shape[1], support=_summary(s.support),
                             origin=_summary(s.origin), domain=_summary(s.domain), access=s.access,
                             modality=s.modality or ""))
        return pd.DataFrame(rows, columns=["source", "level", "varies_at", "time", "columns", "support",
                                           "origin", "domain", "access", "modality"])

    def info(self) -> None:
        """Print units per level, sources, candidates, outcomes, missing values and the
        design with the checks it passed."""
        print(f"Dataset: {len(self.units):,} {self.unit}")
        for why, idx in self.dropped_units.items():
            print(f"  dropped: {len(idx):,} units ({why})")
        print("  units per level: " + ", ".join(f"{k} {v.nunique():,}" for k, v in self.levels.items()))
        print("  sources:")
        tab = self.sources_table()
        if len(tab):
            print("\n".join("    " + line for line in tab.to_string(index=False).splitlines()))
        for time in CANDIDATE_TIMES:
            cs = self._columns((time,))
            if time == "historic" and not cs:
                continue
            by = pd.Series([self.column_source[c].origin[c] for c in cs], dtype=object).value_counts()
            note = " (excluded unless discover(include_during=True))" if time == "during" else ""
            print(f"  candidates {time}: {len(cs)}" + (f" ({', '.join(f'{k} {v}' for k, v in by.items())})"
                                                       if len(cs) else "") + note)
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
            print("  candidates with missing values (excluded from the pool): "
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


def dependent_sets(x: np.ndarray, names: list[str], tol: float = RANK_TOLERANCE) -> list[list[str]]:
    """Minimal sets of columns of x (n x p, no NaN, none constant) that are linearly
    dependent together with an intercept, one set per redundant column.

    The columns are centred (the intercept) and scaled to unit norm, then a QR
    decomposition with column pivoting picks a basis: pivots below tol times the largest
    are zero, and the columns behind them redundant.  Each redundant column is regressed
    on the basis; it and the basis columns with a nonzero coefficient form its set (the
    fundamental circuit: minimal, no column of it can be left out).  [] when the columns
    are linearly independent."""
    n, p = x.shape
    if p == 0 or n == 0:
        return []
    z = x - x.mean(axis=0)
    z = z / np.linalg.norm(z, axis=0)
    _, r, piv = scipy.linalg.qr(z, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int((d > tol * d[0]).sum()) if len(d) else 0
    if rank == p:
        return []
    basis = piv[:rank]
    coef, *_ = np.linalg.lstsq(z[:, basis], z[:, piv[rank:]], rcond=None)
    coef = np.atleast_2d(coef.reshape(rank, -1))
    resid = z[:, piv[rank:]] - z[:, basis] @ coef
    sets = []
    for k, j in enumerate(piv[rank:]):
        if np.linalg.norm(resid[:, k]) > 1e-6:                # not an exact dependence after all
            continue
        c = np.abs(coef[:, k])
        members = sorted([int(j)] + [int(b) for b, v in zip(basis, c) if v > 1e-6 * max(1.0, c.max())])
        sets.append([names[i] for i in members])
    return sets


def nested_order(levels: dict[str, np.ndarray], declared: str) -> list[str]:
    """The levels (coarse -> fine, as in `levels`) a column declared at `declared` may be
    detected to vary at: those whose cells are unions of `declared` cells (coarser or
    equal) and those nested in them (finer).  A level that cuts across `declared` is left
    out: a column constant within its cells by coincidence would otherwise be clustered
    by a partition that splits cells of its declared level."""
    d = levels[declared]
    return [k for k in levels if is_coarser(levels[k], d) or is_coarser(d, levels[k])]


def _summary(values: Mapping) -> str:
    """'a' when every value is a, else 'a 3, b 1' (most frequent first)."""
    n = pd.Series(list(values.values()), dtype=object).value_counts()
    return n.index[0] if len(n) == 1 else ", ".join(f"{k} {v}" for k, v in n.items())


def _few(cols: list, k: int = 5) -> str:
    return ", ".join(map(str, cols[:k])) + (", ..." if len(cols) > k else "")


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
