"""Download CHIRPS rainfall exposure for Ghana LEAP 1000 community centroids.

Produces three files, all feeding NEXIS Z effect-modifier candidates (see
external_data.py::load_effect_modifiers):

  1. rainfall_climatology.csv — mean/std annual rainfall and drought
     frequency over `--climatology-start`..`--climatology-end` (strictly
     before the 2015 baseline). Internally z-scores each climatology year
     against the community's own mean/std purely to define "drought year"
     (below `--drought-z-threshold`); that z-score itself isn't exposed
     downstream, only the resulting frequency.

  2. rainfall_annual.csv — realized annual rainfall (raw mm, no z-scoring)
     for the study years (2015-2017). Kept as 3 separate downstream columns
     (rainfall_2015/2016/2017), not averaged into one scalar. Legitimate as
     an effect modifier despite overlapping the treatment period: a cash
     transfer cannot cause rainfall, so there's no post-treatment-bias risk
     here, unlike a household covariate the transfer could itself change.

  3. rainfall_cdd.csv — max_dry_days_2015_2017, max consecutive dry days (CDD, a day
     counts as dry below `--dry-day-mm`) over the whole 2015-2017 window.
     A standard drought-severity index that a raw annual total can mask
     (two communities can have identical yearly rainfall with very
     different mid-season dry-spell exposure). Computed once over the full
     window rather than per calendar year: a calendar-year split is
     agronomically arbitrary (growing seasons don't align with Jan 1
     anyway) and would understate a drought straddling a year boundary
     (e.g. Nov 2016-Jan 2017) by resetting the streak at Dec 31.

Setup (one-time, same as download_satellite_images.py):
    pip install earthengine-api
    earthengine authenticate --auth_mode notebook
    earthengine set_project <your-project-id>

Usage:
    python download_rainfall.py [--out-dir ../../data/ghana/rainfall]
                                [--climatology-start 2000]
                                [--climatology-end 2014]
                                [--study-years 2015 2016 2017]
                                [--dry-day-mm 1.0]
                                [--scale 5566]
                                [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHIRPS_COLLECTION = 'UCSB-CHG/CHIRPS/DAILY'
CHIRPS_NATIVE_SCALE_M = 5566  # ~0.05 deg


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/rainfall',
                   help='Local directory to save CSVs (default: ../../../data/ghana/rainfall)')
    p.add_argument('--climatology-start', type=int, default=2000,
                   help='First year of the pre-baseline climatology window (default: 2000)')
    p.add_argument('--climatology-end', type=int, default=2014,
                   help='Last year of the pre-baseline climatology window (default: 2014, '
                        'i.e. strictly before the 2015 survey baseline)')
    p.add_argument('--study-years', type=int, nargs='+', default=[2015, 2016, 2017],
                   help='Study-window years to compute realized annual rainfall/anomaly for '
                        '(default: 2015 2016 2017)')
    p.add_argument('--drought-z-threshold', type=float, default=-1.0,
                   help='Z-score below which a climatology year counts as a drought year '
                        '(default: -1.0, i.e. 1 std below the community mean)')
    p.add_argument('--dry-day-mm', type=float, default=1.0,
                   help='Daily rainfall (mm) below which a day counts as dry, for the '
                        'consecutive-dry-day (CDD) columns (default: 1.0, the standard '
                        'WMO dry-day threshold)')
    p.add_argument('--scale', type=int, default=CHIRPS_NATIVE_SCALE_M,
                   help=f'Sampling resolution in metres (default: {CHIRPS_NATIVE_SCALE_M} = CHIRPS native)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without querying Earth Engine')
    p.add_argument('--data-path',
                   default='../../../data/ghana/survey/LEAP1000 2015-2017 household data++.dta',
                   help='Path to household .dta file')
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_community_centroids(data_path: str) -> pd.DataFrame:
    df = pd.read_stata(data_path)
    centroids = (
        df.dropna(subset=['gps_latitude', 'gps_longitude'])
          .groupby('comm')[['gps_latitude', 'gps_longitude']]
          .first()
          .reset_index()
    )
    centroids.columns = ['comm_id', 'lat', 'lon']
    n_missing = df['comm'].nunique() - len(centroids)
    log(f"Loaded {len(centroids)} community centroids ({n_missing} missing GPS skipped)")
    return centroids


# ── GEE helpers ───────────────────────────────────────────────────────────────

def make_feature_collection(centroids: pd.DataFrame):
    import ee
    features = [
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]),
                   {'comm_id': int(row.comm_id)})
        for _, row in centroids.iterrows()
    ]
    return ee.FeatureCollection(features)


def annual_rainfall_by_community(fc, year: int, scale: int) -> pd.DataFrame:
    """Total CHIRPS rainfall (mm) for `year`, reduced at each community point."""
    import ee
    annual_sum = (
        ee.ImageCollection(CHIRPS_COLLECTION)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .select('precipitation')
          .sum()
    )
    reduced = annual_sum.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=scale,
    ).getInfo()
    rows = [
        {'comm': f['properties']['comm_id'], 'year': year,
         'rainfall_mm': f['properties'].get('mean')}
        for f in reduced['features']
    ]
    return pd.DataFrame(rows)


def daily_precip_by_community(fc, start_date: str, end_date: str, scale: int) -> pd.DataFrame:
    """Per-day CHIRPS rainfall (mm) for each community, long format (comm, date, rainfall_mm).

    Stacks the whole date range into one multi-band image (one band per day)
    so a single reduceRegions call returns the full daily series per
    community, rather than one Earth Engine call per day.
    """
    import ee
    coll = (
        ee.ImageCollection(CHIRPS_COLLECTION)
          .filterDate(start_date, end_date)
          .select('precipitation')
    )
    stacked    = coll.toBands()
    band_names = stacked.bandNames().getInfo()
    reduced = stacked.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=scale,
    ).getInfo()
    rows = []
    for f in reduced['features']:
        comm = f['properties']['comm_id']
        for band in band_names:
            date = band.split('_')[0]  # CHIRPS daily band ids are 'YYYYMMDD_precipitation'
            rows.append({
                'comm': comm,
                'date': pd.to_datetime(date, format='%Y%m%d'),
                'rainfall_mm': f['properties'].get(band),
            })
    return pd.DataFrame(rows)


def max_consecutive_dry_days(daily: pd.DataFrame, dry_day_mm: float) -> pd.DataFrame:
    """Longest run of consecutive dry days (rainfall_mm < dry_day_mm) per community.

    Spans the whole date range in `daily` with no reset at year boundaries,
    so a drought straddling Dec 31 counts as one continuous streak.
    """
    daily = daily.sort_values(['comm', 'date']).reset_index(drop=True)
    is_dry  = daily['rainfall_mm'] < dry_day_mm
    wet_run  = (~is_dry).groupby(daily['comm']).cumsum()
    run_len  = is_dry.groupby([daily['comm'], wet_run]).cumsum()
    return run_len.groupby(daily['comm']).max().rename('cdd').reset_index()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    centroids = load_community_centroids(str(data_path))
    climatology_years = list(range(args.climatology_start, args.climatology_end + 1))

    if args.dry_run:
        log(f"\n[DRY RUN] {len(centroids)} communities")
        log(f"  Climatology years: {climatology_years[0]}-{climatology_years[-1]} "
            f"({len(climatology_years)} years, pre-2015 baseline)")
        log(f"  Study years:       {args.study_years}")
        log(f"  Scale: {args.scale} m  |  Drought z-threshold: {args.drought_z_threshold}  "
            f"|  Dry-day threshold: {args.dry_day_mm} mm")
        log(f"  Output: {out_dir}")
        log(f"    - rainfall_climatology.csv  (comm, rainfall_mean_pre2015, "
            f"rainfall_std_pre2015, drought_freq_pre2015)  → NEXIS Z effect-modifier candidates")
        log(f"    - rainfall_annual.csv       (comm, year, rainfall_mm)  "
            f"→ pivoted into rainfall_2015/2016/2017, also Z effect-modifier candidates")
        log(f"    - rainfall_cdd.csv          (comm, max_dry_days_2015_2017)  "
            f"→ max consecutive dry days over the whole study window, also a Z candidate")
        for _, row in centroids.iterrows():
            log(f"  comm{int(row.comm_id):04d}  lat={row.lat:.4f}  lon={row.lon:.4f}")
        return

    import ee
    ee.Initialize()
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = make_feature_collection(centroids)

    log(f"\nFetching {len(climatology_years)} climatology years "
        f"({climatology_years[0]}-{climatology_years[-1]}) …")
    climatology_frames = []
    for i, year in enumerate(climatology_years, 1):
        frame = annual_rainfall_by_community(fc, year, args.scale)
        climatology_frames.append(frame)
        log(f"  [{i}/{len(climatology_years)}]  {year}  "
            f"{frame['rainfall_mm'].notna().sum()}/{len(frame)} communities OK")
    climatology_long = pd.concat(climatology_frames, ignore_index=True)

    climatology = (
        climatology_long.groupby('comm')['rainfall_mm']
        .agg(rainfall_mean_pre2015='mean', rainfall_std_pre2015='std')
        .reset_index()
    )
    merged = climatology_long.merge(climatology, on='comm')
    merged['z'] = (
        (merged['rainfall_mm'] - merged['rainfall_mean_pre2015'])
        / merged['rainfall_std_pre2015']
    )
    drought_freq = (
        merged.assign(is_drought=merged['z'] < args.drought_z_threshold)
        .groupby('comm')['is_drought'].mean()
        .rename('drought_freq_pre2015')
        .reset_index()
    )
    climatology = climatology.merge(drought_freq, on='comm')
    climatology_path = out_dir / 'rainfall_climatology.csv'
    climatology.to_csv(climatology_path, index=False)
    log(f"Wrote {climatology_path}  ({len(climatology)} communities)")

    log(f"\nFetching {len(args.study_years)} study-window years {args.study_years} …")
    study_frames = []
    for i, year in enumerate(args.study_years, 1):
        frame = annual_rainfall_by_community(fc, year, args.scale)
        study_frames.append(frame)
        log(f"  [{i}/{len(args.study_years)}]  {year}  "
            f"{frame['rainfall_mm'].notna().sum()}/{len(frame)} communities OK")
    annual = pd.concat(study_frames, ignore_index=True)[['comm', 'year', 'rainfall_mm']]
    annual_path = out_dir / 'rainfall_annual.csv'
    annual.to_csv(annual_path, index=False)
    log(f"Wrote {annual_path}  ({len(annual)} rows)")

    study_start = f'{args.study_years[0]}-01-01'
    study_end   = f'{args.study_years[-1] + 1}-01-01'  # filterDate end is exclusive
    log(f"\nFetching daily CHIRPS series {study_start} to {study_end} "
        f"for consecutive-dry-day (CDD, <{args.dry_day_mm}mm) counts …")
    daily = daily_precip_by_community(fc, study_start, study_end, args.scale)
    log(f"  {daily['rainfall_mm'].notna().sum()}/{len(daily)} community-days OK")

    cdd = max_consecutive_dry_days(daily, args.dry_day_mm).rename(columns={'cdd': 'max_dry_days_2015_2017'})
    cdd_path = out_dir / 'rainfall_cdd.csv'
    cdd.to_csv(cdd_path, index=False)
    log(f"Wrote {cdd_path}  ({len(cdd)} communities)")


if __name__ == '__main__':
    main()
