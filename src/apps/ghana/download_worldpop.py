"""Download WorldPop gridded population data for Ghana LEAP 1000 community
centroids, into a community-level covariate for NEXIS (see
external_data.py::load_effect_modifiers).

Source: WorldPop/GP/100m/pop, 2015, via Google Earth Engine -- same
authentication already used for rainfall/satellite/nightlights, no new
account needed. ~100m resolution gridded population estimate.

Produces one covariate: pop_density_2km, the summed estimated population
within 2km of each community centroid (not a raster point-value -- a single
100m pixel's population estimate is too granular/noisy; summing over a
small neighbourhood gives a locally-smoothed density). Checked for
degeneracy before adopting: 0/162 communities are zero at this radius
(range 209-9271), and it's only weakly correlated with comm_size (r=0.04,
confirming it captures true local population density rather than
duplicating the LEAP sample count) and moderately with
dist_to_capital_km/travel_time_to_city_min (r=-0.48, expected).

Usage:
    python download_worldpop.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

BUFFER_M = 2000


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/worldpop',
                   help='Local directory to save the community-level CSV '
                        '(default: ../../../data/ghana/worldpop)')
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


def population_density(centroids: pd.DataFrame) -> pd.DataFrame:
    import ee
    ee.Initialize()
    img = (ee.ImageCollection('WorldPop/GP/100m/pop')
             .filterDate('2015-01-01', '2016-01-01')
             .filter(ee.Filter.eq('country', 'GHA'))
             .first())
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]).buffer(BUFFER_M),
                   {'comm_id': int(row.comm)})
        for _, row in centroids.iterrows()
    ])
    reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()
    rows = [
        {'comm': f['properties']['comm_id'], 'pop_density_2km': f['properties'].get('sum')}
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
        log(f"  Collection: WorldPop/GP/100m/pop  year=2015  buffer={BUFFER_M}m")
        log(f"  Output: {out_dir / 'worldpop_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log("Querying Earth Engine for population density ...")
    stats = population_density(centroids)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'worldpop_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  pop_density_2km: min={stats['pop_density_2km'].min():.0f}  "
        f"median={stats['pop_density_2km'].median():.0f}  "
        f"max={stats['pop_density_2km'].max():.0f}")


if __name__ == '__main__':
    main()
