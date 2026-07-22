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
    origin   — raw (an untouched source column, just renamed/binarized, not
               computed) / hand_crafted (a formula combining multiple
               inputs, e.g. dependency_ratio or a haversine distance) /
               learned (an SAE neuron — meaningless until a VLM interprets
               it a posteriori). raw and hand_crafted are both already
               self-explanatory from their name; only learned needs a
               posteriori interpretation. This is what actually drives
               staging: nexis() screens every non-learned column (raw or
               hand_crafted alike) in a cheaper preliminary phase first
               (regardless of level), then lets everything compete
               symmetrically in the main round (see nexis()'s own
               docstring for the mechanics).
    support  — binary / count / continuous / positive_continuous /
               sparse_nonneg: determines the binarization rule used for
               NEXIS's GATE-style split test (zero-threshold vs. median-split).

A fourth axis, `domain`, tags what a covariate is *about* (demographics,
housing, economy, accessibility, urbanization, environment, security,
health) rather than how/where it was measured -- cross-cutting level and
source on purpose. E.g. `maize_price_2015` (a market price) and `has_farm`
(household livelihood behaviour) are both ECONOMY despite one being
community-level/WFP-sourced and the other household-level/survey-sourced;
`dist_nearest_market_km` sits in ACCESSIBILITY instead, alongside every
other distance/travel-time covariate regardless of what it's a distance
*to* -- empirically, these behave as one latent "remoteness" factor (r=0.5-
0.8 pairwise among dist_to_capital_km/travel_time_to_city_min/
dist_nearest_market_km/dist_nearest_light_km), not independent stories.
Defaults to UNKNOWN for apps that haven't tagged their covariates yet
(Uganda/CelebA); every Ghana covariate in data.py is tagged.

A fifth axis, `access`, tags how the *source* was actually obtained --
proprietary (not ours to redistribute, e.g. UNICEF's LEAP-1000 survey),
public (downloadable with no account at all, e.g. HDX/public S3/WCS), or
restricted (needed a free account/registration, e.g. Google Earth Engine,
ACLED). This is a property of the source, not the individual covariate, so
every covariate from the same source shares one access tag. Also defaults
to UNKNOWN for untagged apps.

A sixth axis, `timing`, tags a covariate's relationship to the treatment
window -- pre (measured before/at baseline, the overwhelming majority: every
covariate NEXIS actually searches over, since post-treatment values risk
collider bias -- see the note above COVARIATES in data.py), during
(measured within the treatment window, e.g. a midline observation that
survives the same exogeneity argument as the acled/rainfall study-window
columns), or post (measured after the endline, e.g. a 2017 market price used
to deflate/actualize expenditures downstream, NOT registered as a NEXIS
covariate at all -- see download_market_prices.py). Defaults to UNKNOWN for
untagged apps, same as domain/access; no real Ghana covariate should ever
carry UNKNOWN since every one of them has a definite relationship to the
baseline by construction.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class Level(str, Enum):
    HOUSEHOLD  = 'household'
    COMMUNITY  = 'community'
    REGIONAL   = 'regional'      # shared by a market-catchment-sized cluster of communities,
                                  # coarser than community, not aligned to administrative districts
                                  # (e.g. maize_price_2015: 9 markets serve all 162 communities)
    DISTRICT   = 'district'
    INDIVIDUAL = 'individual'   # Uganda: household-member-level survey answers
    GROUP      = 'group'        # Uganda: RCT randomization unit / site


class Origin(str, Enum):
    RAW          = 'raw'            # untouched source column (renamed/binarized, not computed)
    HAND_CRAFTED = 'hand_crafted'   # formula/lookup combining multiple inputs — self-explanatory
    LEARNED      = 'learned'        # SAE neuron — needs a posteriori interpretation


class Support(str, Enum):
    BINARY              = 'binary'               # {0, 1}
    COUNT               = 'count'                 # non-negative integers
    CONTINUOUS          = 'continuous'             # unrestricted real-valued
    POSITIVE_CONTINUOUS = 'positive_continuous'     # real-valued, non-negative (e.g. mm, km)
    SPARSE_NONNEG        = 'sparse_nonneg'           # non-negative, mostly zero (SAE activations)


class Domain(str, Enum):
    DEMOGRAPHICS  = 'demographics'    # household composition (size, age structure, head traits)
    HOUSING       = 'housing'         # dwelling materials, water, electricity, crowding
    ECONOMY       = 'economy'         # livelihoods, business/farm/livestock ownership, prices
    ACCESSIBILITY = 'accessibility'   # distance/travel-time to anywhere (capital, city, market, light)
    URBANIZATION  = 'urbanization'    # own-place settlement character (population, built-up, lit)
    ENVIRONMENT   = 'environment'     # climate, land cover, vegetation (rainfall + satellite)
    SECURITY      = 'security'        # conflict, violence, unrest
    HEALTH        = 'health'          # disease burden
    UNKNOWN       = 'unknown'         # not yet tagged (Uganda/CelebA, pre-migration)


class Access(str, Enum):
    PROPRIETARY = 'proprietary'   # not ours to redistribute (e.g. UNICEF's LEAP-1000 survey)
    PUBLIC      = 'public'        # downloadable with no account at all (HDX, public S3/WCS)
    RESTRICTED  = 'restricted'    # needed a free account/registration (Earth Engine, ACLED)
    UNKNOWN     = 'unknown'       # not yet tagged (Uganda/CelebA, pre-migration)


class Timing(str, Enum):
    PRE    = 'pre'      # measured before/at baseline -- what NEXIS actually searches over
    DURING = 'during'   # measured within the treatment window (e.g. a midline observation)
    POST   = 'post'     # measured after endline -- not a NEXIS covariate (collider risk);
                         # e.g. a 2017 price used to deflate/actualize expenditures downstream
    UNKNOWN = 'unknown' # not yet tagged (Uganda/CelebA, pre-migration)


@dataclass(frozen=True)
class Covariate:
    name: str
    label: str
    level: Level
    origin: Origin = Origin.HAND_CRAFTED
    support: Support = Support.CONTINUOUS
    domain: Domain = Domain.UNKNOWN
    access: Access = Access.UNKNOWN
    timing: Timing = Timing.UNKNOWN
    source: str = 'survey'   # matches the source names in data/ghana/README.md

    @property
    def needs_interpretation(self) -> bool:
        """True only for learned features (SAE neurons) — raw and
        hand-crafted covariates are already self-explanatory from their
        name/label."""
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
    its own non-learned-first screening phase (see src/method/nexis.py's
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
               domain: Optional[Domain] = None,
               access: Optional[Access] = None,
               timing: Optional[Timing] = None,
               predicate=None) -> "Dataset":
        """New Dataset with only the covariates matching level/origin/domain/access/timing/predicate.

        `predicate`, if given, is a Callable[[Covariate], bool] for filters
        beyond level/origin/domain/access/timing (e.g. by `.source` or
        `.support`) -- combine with the others freely; all given conditions
        must hold (AND)."""
        keep = [
            i for i, c in enumerate(self.covariates)
            if (level is None or c.level == level)
            and (origin is None or c.origin == origin)
            and (domain is None or c.domain == domain)
            and (access is None or c.access == access)
            and (timing is None or c.timing == timing)
            and (predicate is None or predicate(c))
        ]
        return Dataset(
            X=self.X.iloc[:, keep].copy(),
            covariates=[self.covariates[i] for i in keep],
            cluster=self.cluster,
        )
