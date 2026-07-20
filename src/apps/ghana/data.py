"""Ghana LEAP 1000 — data loading and variable definitions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import covariates.py via its fully-qualified path (src.apps.covariates),
# NOT a bare `from covariates import ...` -- Python treats the same file
# imported under two different names as two independent modules with two
# independent classes, so a bare import here would give a *different* Level
# enum than interpret.py's `from src.apps.covariates import Level`, silently
# breaking every `c.level is Level.HOUSEHOLD` identity check downstream.
# Adding the repo root to sys.path guarantees this resolves to the exact
# same module interpret.py (and anything else) already imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.apps.covariates import Covariate, Level, Origin, Support
# Same reasoning for external_data.py: a bare `from external_data import ...`
# only resolves when src/apps/ghana happens to already be on sys.path (e.g.
# a script run from that directory) -- it silently ModuleNotFoundErrors from
# any other cwd/PYTHONPATH, including the notebooks/ directory. Qualified
# import works unconditionally once the repo root is on sys.path (above).
from src.apps.ghana.external_data import load_effect_modifiers

DATA_DIR = Path('../data/ghana')

# ── Covariate registry ────────────────────────────────────────────────────────
# The single source of truth for every covariate known upfront (i.e.
# everything except SAE neurons/spectral indices, which don't exist until a
# model is trained/run — those are registered dynamically in interpret.py).
# NUMERIC_W/BINARY_W/ENGINEERED_W/HOUSEHOLD_W/COMMUNITY_W/W_LABELS below are all
# DERIVED views over this list, kept only because every existing consumer
# (analysis.py, visualize.py, interpret.py, interaction_regression.py, the
# notebook) already imports them — they can't drift out of sync anymore
# since there's nothing left to hand-maintain in parallel.
#
# A covariate is excluded from Level.HOUSEHOLD/COMMUNITY only if treatment
# could plausibly have caused it (post-treatment bias/collider risk — e.g. a
# household covariate a cash transfer could itself change, which is why
# every survey covariate here is baseline-2015-only). Exogenous sources like
# weather don't carry that risk regardless of when they're measured, since
# treatment cannot cause rainfall — see external_data.py::load_effect_modifiers.
COVARIATES: list[Covariate] = [
    # ── Household-level (raw survey) ──────────────────────────────────────────
    Covariate('hhsize',       'Household size',    Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('children_u5',  'Children 0–5',       Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('children_6_17','Children 6–17',      Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('adults',       'Adults 18–64',       Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('elderly',      'Elderly 65+',        Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('head_age',     'Head age',           Level.HOUSEHOLD, support=Support.COUNT),
    Covariate('rooms',        'Rooms',              Level.HOUSEHOLD, support=Support.COUNT),

    Covariate('head_married',   'Head married',            Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('head_female',    'Female head',             Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('head_schooled',  'Head attended school',    Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('head_formal',    'Head in formal sector',   Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('no_electricity', 'No electricity',          Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('mud_walls',      'Mud walls',               Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('thatch_roof',    'Thatch roof',             Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('mud_floor',      'Mud floor',               Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('improved_water', 'Improved water',          Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('has_poultry',    'Has poultry',             Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('has_livestock',  'Has livestock',           Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('has_business',   'Has business',            Level.HOUSEHOLD, support=Support.BINARY),
    Covariate('farms',          'Farming household',       Level.HOUSEHOLD, support=Support.BINARY),

    # ── Household-level (engineered from the survey) ──────────────────────────
    Covariate('livelihood_diversity', 'Livelihood diversity',        Level.HOUSEHOLD,
              support=Support.COUNT, source='survey_engineered'),
    Covariate('dependency_ratio',     'Dependency ratio',            Level.HOUSEHOLD,
              support=Support.POSITIVE_CONTINUOUS, source='survey_engineered'),
    Covariate('rooms_per_person',     'Rooms per person',            Level.HOUSEHOLD,
              support=Support.POSITIVE_CONTINUOUS, source='survey_engineered'),
    Covariate('housing_depriv',       'Housing deprivation index',   Level.HOUSEHOLD,
              support=Support.COUNT, source='survey_engineered'),

    # ── Community-level (engineered from the survey's GPS/hhid) ───────────────
    Covariate('dist_to_capital_km', 'Distance to district capital (km)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='survey_engineered'),
    Covariate('comm_size',          'Community size',                    Level.COMMUNITY,
              support=Support.COUNT, source='survey_engineered'),

    # ── Community-level (rainfall, see external_data.py) ──────────────────────
    Covariate('rainfall_mean_pre2015', 'Mean annual rainfall, 2000–2014 (mm)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('rainfall_std_pre2015',  'Std. annual rainfall, 2000–2014 (mm)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('drought_freq_pre2015',  'Drought frequency, 2000–2014 (share of years)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('rainfall_2015', 'Annual rainfall, 2015 (mm)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('rainfall_2016', 'Annual rainfall, 2016 (mm)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('rainfall_2017', 'Annual rainfall, 2017 (mm)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='rainfall'),
    Covariate('cdd_1517', 'Max consecutive dry days, 2015–2017', Level.COMMUNITY,
              support=Support.COUNT, source='rainfall'),

    # ── Community-level (market access, see external_data.py) ──────────────────
    Covariate('travel_time_to_city_min', 'Travel time to nearest city, 2015 (min)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='market_access'),

    # ── Community-level (conflict/protest events, ACLED, see external_data.py) ─
    Covariate('dist_nearest_conflict_km', 'Distance to nearest conflict event, 2015–2017 (km)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='acled'),
    Covariate('political_violence_25km', 'Political violence events within 25km, 2015–2017', Level.COMMUNITY,
              support=Support.SPARSE_NONNEG, source='acled'),
    Covariate('demonstrations_25km', 'Demonstrations within 25km, 2015–2017', Level.COMMUNITY,
              support=Support.SPARSE_NONNEG, source='acled'),

    # ── Community-level (market prices, WFP, see external_data.py) ──────────────
    Covariate('dist_nearest_market_km', 'Distance to nearest priced market (km)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='market_prices'),
    Covariate('maize_price_2014_2015', 'Maize price at nearest market, 2014–2015 (GHS)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='market_prices'),

    # ── Community-level (nighttime lights, VIIRS, see external_data.py) ─────────
    Covariate('night_light_radiance', 'Nighttime light radiance, 2015 (own community)', Level.COMMUNITY,
              support=Support.SPARSE_NONNEG, source='nightlights'),
    Covariate('dist_nearest_light_km', 'Distance to nearest detectable light, 2015 (km)', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='nightlights'),

    # ── Community-level (population density, WorldPop, see external_data.py) ───
    Covariate('pop_density_2km', 'Population within 2km, 2015', Level.COMMUNITY,
              support=Support.COUNT, source='worldpop'),

    # ── Community-level (urbanization degree, GHSL, see external_data.py) ──────
    Covariate('urbanization_degree', 'Settlement urbanization degree, 2015', Level.COMMUNITY,
              support=Support.COUNT, source='ghsl'),

    # ── Community-level (malaria, Malaria Atlas Project, see external_data.py) ─
    Covariate('pf_mortality_rate_2015', 'Malaria mortality rate, 2015', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='malaria'),
    Covariate('pf_incidence_rate_2015', 'Malaria incidence rate, 2015', Level.COMMUNITY,
              support=Support.POSITIVE_CONTINUOUS, source='malaria'),

    # Mobile coverage (OpenCellID) and mobile usage (Ookla Speedtest) were
    # both explored and rejected as community-level covariates — see
    # data/ghana/README.md's "Explored and rejected" section. Both are
    # present-day-only sources with no way to reconstruct 2015-2017
    # conditions, unlike rainfall, which is stable across a decade.
]

# ── Derived views (kept for every existing consumer — do not hand-edit these;
#    edit COVARIATES above instead) ────────────────────────────────────────────
# All of these are W: every pre-treatment covariate NEXIS can search over,
# whatever level it's measured at. HOUSEHOLD_W/COMMUNITY_W split by level
# purely because that's a useful grouping for some consumers (e.g. picking
# controls, or matching NEXIS's hand_crafted-first screening order) -- it is
# NOT a w-vs-z pool distinction; there is no such distinction in this
# codebase (see covariates.py's module docstring).
NUMERIC_W = [c.name for c in COVARIATES
             if c.level is Level.HOUSEHOLD and c.source == 'survey'
             and c.support in (Support.COUNT, Support.POSITIVE_CONTINUOUS, Support.CONTINUOUS)]
BINARY_W = [c.name for c in COVARIATES
            if c.level is Level.HOUSEHOLD and c.source == 'survey' and c.support is Support.BINARY]
ENGINEERED_W = [c.name for c in COVARIATES
                if c.level is Level.HOUSEHOLD and c.source == 'survey_engineered']
HOUSEHOLD_W = [c.name for c in COVARIATES if c.level is Level.HOUSEHOLD]
COMMUNITY_W = [c.name for c in COVARIATES if c.level is Level.COMMUNITY]
W_LABELS: dict[str, str] = {c.name: c.label for c in COVARIATES}

# ── District capital GPS (WGS-84) ─────────────────────────────────────────────
_DISTRICT_CAPITALS: dict[str, tuple[float, float]] = {
    'East Mamprusi': (10.5285, -0.4156),   # Gambaga
    'Karaga':        (10.1003, -0.5070),   # Karaga
    'Yendi':         ( 9.4412,  0.0138),   # Yendi
    'Bongo':         (10.9019, -0.8149),   # Bongo
    'Garu-Tempane':  (10.8824, -0.1724),   # Garu
}


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def load_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load LEAP 1000 panel data with cleaned variable names.

    Core variables created:
        T            — treatment assignment (1 = Treatment, 0 = Comparison)
        Y            — adult-equivalent expenditure per month (GH₵, constant Aug-2017)
        wave         — survey wave (0 = Baseline 2015, 1 = Endline 2017)
        comm         — community identifier (162 unique values); GPS centroid shared by
                       all households in the same community.  Note: T and C households
                       can appear within the same comm code — comm is a geographic area
                       marker, NOT the randomisation unit — but it provides 162 clusters
                       for variance estimation, far more than the 5 available districts.
        gps_latitude / gps_longitude — community-level centroid coordinates.

    All Yes/No covariates are binarised (1/0) and given readable names.
    Continuous covariates are renamed for clarity.
    """
    df = pd.read_stata(Path(data_dir) / 'survey' / 'LEAP1000 2015-2017 household data++.dta')

    # Core identifiers
    df['T']    = (df['tac'] == 'Treatment').astype(int)
    df['Y']    = df['aeexp_r'].astype(float)
    df['wave'] = df['time'].map({'Baseline': 0, 'Endline': 1}).astype('int64')

    # RDD running variable: lower PMT score = poorer = Treatment-eligible.
    # Not a NEXIS covariate (near-perfect separation by construction, see
    # data/ghana/README.md's "Known caveat") -- kept only to illustrate/check
    # the assignment rule around its cutoff.
    df['pmt_score'] = df['pmtscore'].astype(float)

    # Community identifier and GPS coordinates (community-level centroids)
    df['comm']          = df['comm'].astype(int)
    df['gps_latitude']  = df['gps_latitude'].astype(float)
    df['gps_longitude'] = df['gps_longitude'].astype(float)

    # Rename continuous covariates for readability
    df = df.rename(columns={
        'chn05':   'children_u5',
        'chn617':  'children_6_17',
        'adult':   'adults',
        'headage': 'head_age',
    })

    # Rooms: 'Yes' means the household confirmed >1 room but gave no count — treat as 1
    df['rooms'] = pd.to_numeric(
        df['room'].astype(str).replace('Yes', '1'), errors='coerce'
    )

    # Binarise Yes/No columns with clean names
    _binary_map = {
        'headmarried':  'head_married',
        'headfemale':   'head_female',
        'headschool':   'head_schooled',
        'headformal':   'head_formal',
        'noelec':       'no_electricity',
        'mudwall':      'mud_walls',
        'thatchroof':   'thatch_roof',
        'mudfloor':     'mud_floor',
        'water':        'improved_water',
        'anypoultry':   'has_poultry',
        'anylivestock': 'has_livestock',
        'anybusiness':  'has_business',
        'anyfarming':   'farms',
    }
    for raw, clean in _binary_map.items():
        df[clean] = (df[raw] == 'Yes').astype(int)

    # ── Engineered features ───────────────────────────────────────────────────
    # Count of distinct income/livelihood channels (0–5)
    df['livelihood_diversity'] = (
        df['farms'] + df['has_livestock'] + df['has_poultry']
        + df['has_business'] + df['head_formal']
    )

    # Share of household members who are economically dependent
    df['dependency_ratio'] = (
        (df['children_u5'] + df['children_6_17'] + df['elderly']) / df['hhsize']
    )

    # Housing space per person (rooms fixed: 'Yes' → 1, so no NAs)
    df['rooms_per_person'] = df['rooms'] / df['hhsize']

    # Count of housing deprivation dimensions (0–4)
    df['housing_depriv'] = (
        df['mud_walls'] + df['thatch_roof'] + df['mud_floor'] + df['no_electricity']
    )

    # Haversine distance (km) from community centroid to district capital
    district_str = df['district'].astype(str)
    cap_lat = district_str.map({d: c[0] for d, c in _DISTRICT_CAPITALS.items()}).values.astype(float)
    cap_lon = district_str.map({d: c[1] for d, c in _DISTRICT_CAPITALS.items()}).values.astype(float)
    df['dist_to_capital_km'] = _haversine_km(
        df['gps_latitude'].values, df['gps_longitude'].values, cap_lat, cap_lon
    )

    # Number of unique sampled households per community (proxy for community size)
    df['comm_size'] = df.groupby('comm')['hhid'].transform('nunique')

    # ── External effect modifiers ─────────────────────────────────────────────
    # See external_data.py — merged on comm only, one row per community.
    df = df.merge(load_effect_modifiers(Path(data_dir)), on='comm', how='left')

    return df


def load_satellite_covariates(
    data_dir: Path | str = DATA_DIR, min_activations: int = 5,
) -> tuple[pd.DataFrame, list[Covariate]]:
    """Lightweight satellite-feature loader for exploration (not NEXIS itself).

    Reads the already-computed SAE activations and spectral indices for the
    162 LEAP communities directly off disk -- no SAE/foundation-model forward
    pass, no national-grid contrast pool. That heavier path (needed for VLM
    neuron interpretation) lives in interpret.py::load_nexis_inputs; this one
    only needs enough to look at the candidate pool's distribution/geography.

    Returns
    -------
    (comm_df, covariates) : comm_df indexed by `comm`, one column per
    covariate (spectral means + SAE neurons active in >= min_activations
    communities); covariates is the matching list of Covariate metadata.
    """
    sat_dir = Path(data_dir) / 'satellite'

    spectral = pd.read_csv(sat_dir / 'spectral_indices.csv').rename(columns={'comm_id': 'comm'})
    mean_cols = [c for c in spectral.columns if c.endswith('_mean')]
    spectral_names = [c[:-5] for c in mean_cols]                    # ndvi_mean -> ndvi
    spectral_df = spectral.set_index('comm')[mean_cols].rename(
        columns=dict(zip(mean_cols, spectral_names))
    )
    spectral_covariates = [
        Covariate(name, name.upper(), Level.COMMUNITY, Origin.HAND_CRAFTED,
                  Support.CONTINUOUS, source='satellite_spectral')
        for name in spectral_names
    ]

    codes    = np.load(sat_dir / 'sae_activations.npy')             # (162, 4096)
    comm_ids = np.load(sat_dir / 'sae_comm_ids.npy')
    live_idx = np.where((codes > 0).sum(axis=0) >= min_activations)[0]
    sae_df = pd.DataFrame(
        codes[:, live_idx], index=pd.Index(comm_ids, name='comm'),
        columns=[f'neuron_{int(i)}' for i in live_idx],
    )
    sae_covariates = [
        Covariate(f'neuron_{int(i)}', f'Neuron {int(i)}', Level.COMMUNITY,
                  Origin.LEARNED, Support.SPARSE_NONNEG, source='satellite_sae')
        for i in live_idx
    ]

    comm_df    = spectral_df.join(sae_df, how='inner')
    covariates = spectral_covariates + sae_covariates
    return comm_df[[c.name for c in covariates]], covariates
