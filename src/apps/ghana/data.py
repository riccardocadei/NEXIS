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

W_ALL = NUMERIC_W + BINARY_W

# ── Human-readable labels ─────────────────────────────────────────────────────
W_LABELS: dict[str, str] = {
    'hhsize':         'Household size',
    'children_u5':    'Children 0–5',
    'children_6_17':  'Children 6–17',
    'adults':         'Adults 18–64',
    'elderly':        'Elderly 65+',
    'head_age':       'Head age',
    'rooms':          'Rooms',
    'head_married':   'Head married',
    'head_female':    'Female head',
    'head_schooled':  'Head attended school',
    'head_formal':    'Head in formal sector',
    'no_electricity': 'No electricity',
    'mud_walls':      'Mud walls',
    'thatch_roof':    'Thatch roof',
    'mud_floor':      'Mud floor',
    'improved_water': 'Improved water',
    'has_poultry':    'Has poultry',
    'has_livestock':  'Has livestock',
    'has_business':   'Has business',
    'farms':          'Farming household',
}


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

    # Rooms: coerce to numeric (raw data has a spurious 'Yes' category)
    df['rooms'] = pd.to_numeric(
        df['room'].astype(str).replace('Yes', np.nan), errors='coerce'
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

    return df
