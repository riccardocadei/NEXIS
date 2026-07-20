"""Process ACLED conflict/protest event data for Ghana LEAP 1000 community
centroids, into community-level covariates for NEXIS (see
external_data.py::load_effect_modifiers).

Unlike CHIRPS rainfall (download_rainfall.py), this is *not* pulled from an
API here: ACLED's event-level export is account-gated (a free academic/
non-profit registration at acleddata.com gets access to their Data Export
Tool), so download the Ghana event-level CSV yourself — dyadic (one row per
event, the default), text interaction values, standard comma delimiter —
and point this script at the local file with --input.

Raw columns include (per event): event_date, year, disorder_type,
event_type, sub_event_type, actor1, admin1/admin2/admin3, location,
latitude, longitude, geo_precision, fatalities. Only event_type, latitude,
longitude, and fatalities are used here; the rest (actors, notes, sources)
aren't needed once events are reduced to community-level distance/counts.

Produces one file, community-level, feeding NEXIS covariates:
    conflicts/acled_community.csv
        comm, dist_nearest_conflict_km, political_violence_25km,
        demonstrations_25km

Covariate design (see data/ghana/README.md for the full writeup):
  - dist_nearest_conflict_km: distance to the nearest event of any type,
    always defined (uncapped by radius).
  - political_violence_25km: count of Battles/Violence-against-civilians
    within 25km — the event types most directly relevant to household
    welfare/safety exposure.
  - demonstrations_25km: count of Riots/Protests within 25km — civil
    unrest, a qualitatively different kind of exposure than political
    violence.
  Fatality-weighted sums were considered and dropped: on top of already-
  sparse event counts, fatalities are heavily zero-inflated (75th
  percentile is 0 nationally), so a sum would mostly reflect a handful of
  outlier high-fatality events rather than a stable community trait.

Usage:
    python download_acled.py --input "path/to/ACLED Data.csv" [--radius-km 25.0] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POLITICAL_VIOLENCE_TYPES = ['Battles', 'Violence against civilians']
DEMONSTRATION_TYPES = ['Riots', 'Protests']


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True,
                   help='Path to the ACLED Ghana event-level export (.csv), '
                        'downloaded manually from acleddata.com\'s Data Export Tool')
    p.add_argument('--out-dir', default='../../../data/ghana/conflicts',
                   help='Local directory to save the community-level CSV '
                        '(default: ../../../data/ghana/conflicts)')
    p.add_argument('--radius-km', type=float, default=25.0,
                   help='Radius around each community centroid for the '
                        'event-count covariates (default: 25.0, matching '
                        'the radius already used for market-access checks)')
    p.add_argument('--data-path',
                   default='../../../data/ghana/survey/LEAP1000 2015-2017 household data++.dta',
                   help='Path to household .dta file, for community centroids')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without reading/writing files')
    return p.parse_args()


def load_community_centroids(data_path: str) -> pd.DataFrame:
    df = pd.read_stata(data_path)
    centroids = (
        df.dropna(subset=['gps_latitude', 'gps_longitude'])
          .groupby('comm')[['gps_latitude', 'gps_longitude']]
          .first()
          .reset_index()
    )
    centroids.columns = ['comm', 'lat', 'lon']
    n_missing = df['comm'].nunique() - len(centroids)
    log(f"Loaded {len(centroids)} community centroids ({n_missing} missing GPS skipped)")
    return centroids


def load_events(input_path: str) -> pd.DataFrame:
    events = pd.read_csv(input_path, usecols=['event_type', 'latitude', 'longitude', 'fatalities'])
    return events.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def community_conflict_stats(centroids: pd.DataFrame, events: pd.DataFrame,
                              radius_km: float) -> pd.DataFrame:
    pv = events[events['event_type'].isin(POLITICAL_VIOLENCE_TYPES)]
    demo = events[events['event_type'].isin(DEMONSTRATION_TYPES)]

    rows = []
    for _, comm in centroids.iterrows():
        dist_km = _haversine_km(comm['lat'], comm['lon'], events['latitude'], events['longitude'])
        dist_pv = _haversine_km(comm['lat'], comm['lon'], pv['latitude'], pv['longitude'])
        dist_demo = _haversine_km(comm['lat'], comm['lon'], demo['latitude'], demo['longitude'])
        rows.append({
            'comm': comm['comm'],
            'dist_nearest_conflict_km': dist_km.min() if len(dist_km) else np.nan,
            f'political_violence_{radius_km:g}km': (dist_pv <= radius_km).sum(),
            f'demonstrations_{radius_km:g}km': (dist_demo <= radius_km).sum(),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    input_path = Path(args.input).resolve()

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Input:  {input_path}")
        log(f"  Radius: {args.radius_km} km")
        log(f"  Output: {out_dir / 'acled_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log(f"Reading events from {input_path} ...")
    events = load_events(str(input_path))
    log(f"  {len(events)} Ghana events with valid coordinates")

    stats = community_conflict_stats(centroids, events, args.radius_km)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'acled_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    pv_col = f'political_violence_{args.radius_km:g}km'
    demo_col = f'demonstrations_{args.radius_km:g}km'
    log(f"  dist_nearest_conflict_km: min={stats['dist_nearest_conflict_km'].min():.1f}  "
        f"median={stats['dist_nearest_conflict_km'].median():.1f}  "
        f"max={stats['dist_nearest_conflict_km'].max():.1f}")
    log(f"  {(stats[pv_col] > 0).sum()}/{len(stats)} communities have political violence within {args.radius_km}km")
    log(f"  {(stats[demo_col] > 0).sum()}/{len(stats)} communities have demonstrations within {args.radius_km}km")


if __name__ == '__main__':
    main()
