"""Download GHSL Settlement Model (degree of urbanization) for Ghana LEAP
1000 community centroids, into a community-level covariate for NEXIS (see
external_data.py::load_effect_modifiers).

Source: JRC/GHSL/P2023A/GHS_SMOD_V2-0/2015, via Google Earth Engine -- same
authentication already used for rainfall/satellite/nightlights/WorldPop, no
new account needed. 1km resolution categorical settlement classification
(smod_code), monotonically ordered by urbanization degree: 11 (very-low-
density rural) < 12 (low-density rural) < 13 (rural cluster) < 21
(suburban/peri-urban) < 22 (semi-dense urban cluster) < 23 (dense urban
cluster) < 30 (urban centre).

Checked for degeneracy before adopting: genuine spread across the 162
communities (92 at code 12, 35 at 11, 24 at 13, and a handful reaching
21/22/23/30), not a single dominant category. Moderately correlated with
pop_density_2km (r=0.68, expected -- GHSL's own classification is partly
built from population density) but not so high as to be a pure duplicate,
and weakly correlated with dist_to_capital_km/travel_time_to_city_min
(r=-0.17).

Usage:
    python download_ghsl.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/ghsl',
                   help='Local directory to save the community-level CSV '
                        '(default: ../../../data/ghana/ghsl)')
    p.add_argument('--data-path',
                   default='../../../data/ghana/survey/LEAP1000 2015-2017 household data++.dta',
                   help='Path to household .dta file, for community centroids')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without querying Earth Engine')
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


def urbanization_degree(centroids: pd.DataFrame) -> pd.DataFrame:
    import ee
    ee.Initialize()
    img = ee.Image('JRC/GHSL/P2023A/GHS_SMOD_V2-0/2015').select('smod_code')
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {'comm_id': int(row.comm)})
        for _, row in centroids.iterrows()
    ])
    reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=1000).getInfo()
    rows = [
        {'comm': f['properties']['comm_id'], 'urbanization_degree': f['properties'].get('first')}
        for f in reduced['features']
    ]
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Collection: JRC/GHSL/P2023A/GHS_SMOD_V2-0/2015")
        log(f"  Output: {out_dir / 'ghsl_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log("Querying Earth Engine for GHSL settlement classification ...")
    stats = urbanization_degree(centroids)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ghsl_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  urbanization_degree distribution:\n{stats['urbanization_degree'].value_counts().sort_index()}")


if __name__ == '__main__':
    main()
