"""Ghana LEAP 1000 — loaders for external, community-GPS-keyed data sources.

Every external source is a candidate community-level covariate, keyed by
`comm` — one row per community. A covariate only needs to be *excluded* if
treatment could plausibly have caused it (post-treatment bias/collider risk,
e.g. household business/livestock ownership, which a cash transfer can
change). Exogenous sources like weather aren't at risk of this regardless of
whether they're measured before or during the study window, since treatment
cannot cause rainfall — so there's no separate "control-only" lane here.

    load_effect_modifiers(comm) -> community-level covariates (data.py::COMMUNITY_Z)

Add one column-producing block per source to this function rather than
inventing new merge logic per source. Rainfall (CHIRPS, via
download_rainfall.py) is the first source; market access (travel time to
cities, via download_market_access.py) is the second; conflict/protest
events (ACLED, via download_acled.py) are the third.

Mobile-network coverage (OpenCellID) and mobile usage (Ookla Speedtest) were
both explored and rejected — see data/ghana/README.md's "Explored and
rejected" section. Both are present-day-only sources (OpenCellID's registry
is dominated by towers logged 2025-2026; Ookla's open archive starts 2019),
with no way to reconstruct 2015-2017 conditions, so neither passed the same
temporal-validity bar rainfall clears easily (climate is stable across a
decade; Ghana's mobile infrastructure/usage was not, over this exact
window). Don't re-add either without a source that actually covers
2015-2017 (e.g. a licensed GSMA historical coverage layer).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path('../data/ghana')


def load_effect_modifiers(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Community-level candidate covariates for NEXIS.

    Returns a DataFrame indexed by `comm` (one row per community). Merge new
    sources with `.merge(other_df, on='comm', how='outer')` and append their
    column name(s) to COMMUNITY_Z in data.py.

    Rainfall contributes 7 columns, all raw mm/days (no z-scoring):
      - rainfall_mean_pre2015, rainfall_std_pre2015, drought_freq_pre2015:
        climatology (2000-2014, strictly pre-baseline).
      - rainfall_2015, rainfall_2016, rainfall_2017: the 3 realized annual
        totals during the study window itself, kept separate rather than
        averaged into one scalar (2016 correlates -0.66 with 2015 and -0.52
        with 2017, so a mean would mostly cancel out real year-to-year
        signal).
      - cdd_1517: max consecutive dry days (CDD) over the whole 2015-2017
        window, a drought-severity index a raw annual total can mask (two
        communities can have identical yearly rainfall with very different
        mid-season dry-spell exposure). Computed once over the full window
        rather than per calendar year: a calendar-year split is
        agronomically arbitrary and would understate a drought straddling
        a year boundary (e.g. Nov 2016-Jan 2017) by resetting at Dec 31.
      All of the above are legitimate effect modifiers despite overlapping
      the treatment period — the cash transfer cannot cause rainfall, so
      there's no post-treatment-bias risk, unlike a household covariate
      that could genuinely be changed by receiving the transfer.

    Market access contributes 1 column (see download_market_access.py):
      - travel_time_to_city_min: motorized travel time (minutes) to the
        nearest city of >50,000 population, Weiss et al. 2018 (Malaria
        Atlas Project), explicitly dated 2015 — an exact match to the LEAP
        baseline year, unlike OpenCellID/Ookla which were present-day-only
        snapshots (see the "Explored and rejected" note below). A cash
        transfer cannot build roads or move cities, so this is exogenous
        regardless of timing, same test as rainfall.

    ACLED contributes 3 columns (see download_acled.py), from the full
    Ghana event-level export for 2015-2017 (the study window itself —
    genuinely dated historical events, not a present-day snapshot, so no
    temporal-mismatch risk the way OpenCellID/Ookla had):
      - dist_nearest_conflict_km: distance to the nearest event of any
        type, always defined (uncapped by radius).
      - political_violence_25km: count of Battles/Violence-against-
        civilians within 25km — exposure most directly relevant to
        household welfare/safety.
      - demonstrations_25km: count of Riots/Protests within 25km — civil
        unrest, a qualitatively different exposure than political violence.
      A household-level cash transfer cannot cause a riot or a battle, so
      this is a valid community-level covariate despite overlapping the
      treatment period, same test as rainfall's study-window columns.
    """
    data_dir = Path(data_dir)
    annual_columns = ['rainfall_2015', 'rainfall_2016', 'rainfall_2017']
    rainfall_columns = [
        'rainfall_mean_pre2015', 'rainfall_std_pre2015', 'drought_freq_pre2015',
        *annual_columns, 'cdd_1517',
    ]
    market_access_columns = ['travel_time_to_city_min']
    acled_columns = ['dist_nearest_conflict_km', 'political_violence_25km', 'demonstrations_25km']

    climatology_path = data_dir / 'rainfall' / 'rainfall_climatology.csv'
    annual_path = data_dir / 'rainfall' / 'rainfall_annual.csv'
    cdd_path = data_dir / 'rainfall' / 'rainfall_cdd.csv'
    if climatology_path.exists() and annual_path.exists() and cdd_path.exists():
        climatology = pd.read_csv(climatology_path)[
            ['comm', 'rainfall_mean_pre2015', 'rainfall_std_pre2015', 'drought_freq_pre2015']
        ]
        annual_wide = (
            pd.read_csv(annual_path).pivot(index='comm', columns='year', values='rainfall_mm')
            .rename(columns={2015: 'rainfall_2015', 2016: 'rainfall_2016', 2017: 'rainfall_2017'})
            .reset_index()
        )
        cdd = pd.read_csv(cdd_path)[['comm', 'cdd_1517']]
        rainfall = climatology.merge(annual_wide, on='comm').merge(cdd, on='comm')
    else:
        # rainfall not yet downloaded (run download_rainfall.py) — callers
        # still see the expected columns, filled with NaN, not a KeyError.
        rainfall = pd.DataFrame(columns=['comm', *rainfall_columns])

    market_access_path = data_dir / 'market_access' / 'market_access_community.csv'
    if market_access_path.exists():
        market_access = pd.read_csv(market_access_path)[['comm', *market_access_columns]]
    else:
        # not yet processed (run download_market_access.py) — same NaN-fill
        # convention as rainfall above.
        market_access = pd.DataFrame(columns=['comm', *market_access_columns])

    acled_path = data_dir / 'conflicts' / 'acled_community.csv'
    if acled_path.exists():
        acled = pd.read_csv(acled_path)[['comm', *acled_columns]]
    else:
        # not yet processed (run download_acled.py) — same NaN-fill
        # convention as rainfall/market access above.
        acled = pd.DataFrame(columns=['comm', *acled_columns])

    merged = (
        rainfall.merge(market_access, on='comm', how='outer')
                .merge(acled, on='comm', how='outer')
                .astype({'comm': 'int64'})
    )
    return merged[['comm', *rainfall_columns, *market_access_columns, *acled_columns]]
