"""Ghana LEAP 1000 — data loading and variable definitions."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('../data/ghana')

# ── Covariate groups ──────────────────────────────────────────────────────────
NUMERIC_W = [
    'hhsize', 'children_u5', 'children_6_17',
    'adults', 'elderly', 'head_age', 'rooms',
]

BINARY_W = [
    'head_married', 'head_female', 'head_schooled', 'head_formal',
    'no_electricity', 'mud_walls', 'thatch_roof', 'mud_floor',
    'improved_water', 'has_poultry', 'has_livestock', 'has_business',
    'farms',
]

ENGINEERED_W = [
    'livelihood_diversity',
    'dependency_ratio',
    'rooms_per_person',
    'housing_depriv',
]

# Community-level features — go into Z alongside SAE neurons and spectral indices
COMMUNITY_Z = [
    'dist_to_capital_km',
    'comm_size',
]

W_ALL = NUMERIC_W + BINARY_W + ENGINEERED_W

# ── Human-readable labels ─────────────────────────────────────────────────────
W_LABELS: dict[str, str] = {
    'hhsize':               'Household size',
    'children_u5':          'Children 0–5',
    'children_6_17':        'Children 6–17',
    'adults':               'Adults 18–64',
    'elderly':              'Elderly 65+',
    'head_age':             'Head age',
    'rooms':                'Rooms',
    'head_married':         'Head married',
    'head_female':          'Female head',
    'head_schooled':        'Head attended school',
    'head_formal':          'Head in formal sector',
    'no_electricity':       'No electricity',
    'mud_walls':            'Mud walls',
    'thatch_roof':          'Thatch roof',
    'mud_floor':            'Mud floor',
    'improved_water':       'Improved water',
    'has_poultry':          'Has poultry',
    'has_livestock':        'Has livestock',
    'has_business':         'Has business',
    'farms':                'Farming household',
    'livelihood_diversity': 'Livelihood diversity',
    'dependency_ratio':     'Dependency ratio',
    'rooms_per_person':     'Rooms per person',
    'housing_depriv':       'Housing deprivation index',
    'dist_to_capital_km':   'Distance to district capital (km)',
    'comm_size':            'Community size',
}

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
    df = pd.read_stata(Path(data_dir) / 'LEAP1000 2015-2017 household data++.dta')

    # Core identifiers
    df['T']    = (df['tac'] == 'Treatment').astype(int)
    df['Y']    = df['aeexp_r'].astype(float)
    df['wave'] = df['time'].map({'Baseline': 0, 'Endline': 1}).astype('int64')

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

    return df
