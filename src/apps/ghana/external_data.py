"""Ghana LEAP 1000 — loaders for external, community-GPS-keyed data sources.

Every external source belongs to exactly one of two lanes, chosen by whether
it is a stable pre-treatment trait or a time-varying event during the study
window (2015-2017) — see notebooks/ghana.ipynb, section "Effect modifiers vs.
DiD controls", for the reasoning:

    load_effect_modifiers(comm)       -> NEXIS Z candidates (data.py::COMMUNITY_Z)
    load_did_controls(comm, year)     -> analysis.py::regression_did(controls=...)

Add one column-producing block per source to the matching function below
rather than inventing new merge logic per source. Rainfall (CHIRPS, via
download_rainfall.py) is the first source and populates both lanes.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path('../data/ghana')


def load_effect_modifiers(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Stable, pre-treatment community traits — candidate NEXIS Z moderators.

    Returns a DataFrame indexed by `comm` (one row per community). Merge new
    sources with `.merge(other_df, on='comm', how='outer')` and append their
    column name(s) to COMMUNITY_Z in data.py.
    """
    data_dir = Path(data_dir)
    columns = ['rainfall_mean_pre2015', 'rainfall_std_pre2015', 'drought_freq_pre2015']

    rainfall_path = data_dir / 'rainfall' / 'rainfall_climatology.csv'
    if rainfall_path.exists():
        return pd.read_csv(rainfall_path)[['comm', *columns]]
    # rainfall not yet downloaded (run download_rainfall.py) — callers still
    # see the expected columns, filled with NaN, rather than a KeyError.
    return pd.DataFrame(columns=['comm', *columns]).astype({'comm': 'int64'})


def load_did_controls(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Time-varying, study-window (2015-2017) shocks — DiD robustness controls.

    Returns a DataFrame indexed by (`comm`, `year`). NOT a source of NEXIS Z
    moderators — pass column names to analysis.py::regression_did(controls=...)
    after merging on comm + year->wave.

    `rainfall_anomaly_1517_mean` is constant across all 3 rows for a given
    `comm` (mean over 2015/2016/2017) — the household panel has no 2016 wave
    to attach a per-year 2016 value to, so this is how 2016's realized
    rainfall still gets used rather than sitting unused in the CSV.
    """
    data_dir = Path(data_dir)
    columns = ['rainfall_mm', 'rainfall_anomaly', 'rainfall_anomaly_1517_mean']

    rainfall_path = data_dir / 'rainfall' / 'rainfall_annual.csv'
    if rainfall_path.exists():
        rainfall = pd.read_csv(rainfall_path)[['comm', 'year', 'rainfall_mm', 'rainfall_anomaly']]
        study_mean = (
            rainfall.groupby('comm')['rainfall_anomaly'].mean()
            .rename('rainfall_anomaly_1517_mean').reset_index()
        )
        return rainfall.merge(study_mean, on='comm')[['comm', 'year', *columns]]
    # rainfall not yet downloaded (run download_rainfall.py) — callers still
    # see the expected columns, filled with NaN, rather than a KeyError.
    return pd.DataFrame(columns=['comm', 'year', *columns]).astype(
        {'comm': 'int64', 'year': 'int64'}
    )
