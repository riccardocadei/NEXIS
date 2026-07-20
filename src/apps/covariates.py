"""Shared across apps (currently used by Ghana; Uganda/CelebA have their own,
differently-shaped data.py and haven't been migrated yet) — the Covariate
registry: one object describing every input NEXIS can see, replacing the old
per-app parallel/hand-synced lists (e.g. Ghana's NUMERIC_W, BINARY_W,
ENGINEERED_W, HOUSEHOLD_W, COMMUNITY_W, W_LABELS).

Every covariate is tagged along three independent axes:

    level    — household / community / district: where it's measured. Pure
               metadata, not a routing decision -- every covariate, at any
               level, feeds the SAME single `w` matrix nexis() takes (see
               src/method/nexis.py). There is no household-vs-community
               split at the API level: "W" means every pre-treatment
               candidate NEXIS can search over, full stop. "Z" is not a
               covariate pool at all here -- if you see it used elsewhere in
               this codebase, it either means a raw neural/learned
               representation (e.g. an SAE activation) before it's been
               registered as a Covariate and merged into W, or it's the
               unrelated, standard statistical notation for a conditioning
               set inside nexis.py's own GCM/interaction-test math -- neither
               usage denotes a distinct covariate pool the way the old
               two-argument `nexis(y, t, w, z)` API's w/z split once did.
    origin   — hand_crafted (name explains it) / learned (an SAE neuron —
               meaningless until a VLM interprets it a posteriori). This is
               what actually drives staging: nexis() screens hand_crafted
               columns in a cheaper preliminary phase first (regardless of
               level), then lets everything compete symmetrically in the
               main round (see nexis()'s own docstring for the mechanics).
    support  — binary / count / continuous / positive_continuous /
               sparse_nonneg: determines the binarization rule used for
               NEXIS's GATE-style split test (zero-threshold vs. median-split).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class Level(str, Enum):
    HOUSEHOLD  = 'household'
    COMMUNITY  = 'community'
    DISTRICT   = 'district'
    INDIVIDUAL = 'individual'   # Uganda: household-member-level survey answers
    GROUP      = 'group'        # Uganda: RCT randomization unit / site


class Origin(str, Enum):
    HAND_CRAFTED = 'hand_crafted'   # formula/lookup — self-explanatory from the name
    LEARNED      = 'learned'        # SAE neuron — needs a posteriori interpretation


class Support(str, Enum):
    BINARY              = 'binary'               # {0, 1}
    COUNT               = 'count'                 # non-negative integers
    CONTINUOUS          = 'continuous'             # unrestricted real-valued
    POSITIVE_CONTINUOUS = 'positive_continuous'     # real-valued, non-negative (e.g. mm, km)
    SPARSE_NONNEG        = 'sparse_nonneg'           # non-negative, mostly zero (SAE activations)


@dataclass(frozen=True)
class Covariate:
    name: str
    label: str
    level: Level
    origin: Origin = Origin.HAND_CRAFTED
    support: Support = Support.CONTINUOUS
    source: str = 'survey'   # matches the source names in data/ghana/README.md

    @property
    def needs_interpretation(self) -> bool:
        """True only for learned features (SAE neurons) — everything
        hand-crafted is already self-explanatory from its name/label."""
        return self.origin is Origin.LEARNED

    def binarize(self, col):
        """Dichotomize a column per its support, for NEXIS's GATE-style test.

        binary/sparse_nonneg -> active (>0) vs inactive (=0), matching how
        SAE activations and Yes/No survey items are naturally split.
        Everything else -> above vs below the sample median.
        """
        if self.support in (Support.BINARY, Support.SPARSE_NONNEG):
            return col > 0
        return col > col.median()


@dataclass
class Dataset:
    """Every pre-treatment observation NEXIS can search over, as one object.

    No more W-vs-Z split: `X` is a single (n, p) matrix with real column
    names (`X.columns == [c.name for c in covariates]`), and each column's
    `Level` (household/community/district/...) is just metadata carried by
    its Covariate -- not a structural fork into two separate arguments.
    `nexis()` reads `X.attrs['origin']` off this object directly to decide
    its own hand_crafted-first screening phase (see src/method/nexis.py's
    docstring), and `X.attrs['cluster']` for CR1S standard errors -- so
    there is nothing besides this object to pass to `nexis(y, t, dataset.X)`.
    """

    X: pd.DataFrame
    covariates: List[Covariate]
    cluster: Optional[np.ndarray] = None

    def __post_init__(self):
        names = [c.name for c in self.covariates]
        if list(self.X.columns) != names:
            raise ValueError(
                "Dataset.X columns must match covariates' names, in the same "
                f"order (X has {len(self.X.columns)} cols, covariates has {len(names)})."
            )
        self.X.attrs["origin"] = [c.origin for c in self.covariates]
        self.X.attrs["cluster"] = self.cluster

    def __len__(self) -> int:
        return len(self.covariates)

    @property
    def names(self) -> List[str]:
        return list(self.X.columns)

    @property
    def labels(self) -> List[str]:
        """Human-readable display name per column, same order as X.columns."""
        return [c.label for c in self.covariates]

    def label_of(self, name: str) -> str:
        """Display label for one column, looked up by its canonical name."""
        return next(c.label for c in self.covariates if c.name == name)

    @property
    def labels_dict(self) -> dict:
        """name -> label for every column, e.g. for balance_tests()/plot_love()-
        style helpers that take a `labels=` dict rather than a parallel list."""
        return {c.name: c.label for c in self.covariates}

    def covariate_of(self, name: str) -> Covariate:
        return next(c for c in self.covariates if c.name == name)

    def subset(self, *, level: Optional[Level] = None,
               origin: Optional[Origin] = None,
               predicate=None) -> "Dataset":
        """New Dataset with only the covariates matching level/origin/predicate.

        `predicate`, if given, is a Callable[[Covariate], bool] for filters
        beyond level/origin (e.g. by `.source` or `.support`) -- combine with
        level/origin freely; all given conditions must hold (AND)."""
        keep = [
            i for i, c in enumerate(self.covariates)
            if (level is None or c.level == level)
            and (origin is None or c.origin == origin)
            and (predicate is None or predicate(c))
        ]
        return Dataset(
            X=self.X.iloc[:, keep].copy(),
            covariates=[self.covariates[i] for i in keep],
            cluster=self.cluster,
        )
