"""Download travel-time-to-city market-access data for Ghana LEAP 1000
community centroids, into a community-level covariate for NEXIS (see
external_data.py::load_effect_modifiers).

Source: Weiss et al. 2018 (Nature), "A global map of travel time to cities
to assess inequalities in accessibility in 2015" — Malaria Atlas Project,
public WCS endpoint (no account/API key). Unlike OpenCellID/Ookla (both
explored and rejected, see data/ghana/README.md), this dataset is
*explicitly dated 2015* (confirmed via its WCS metadata: temporalExtent
2015) — an exact match to the LEAP baseline year, not a present-day
snapshot standing in for one.

The WCS GetCoverage request is clipped to Ghana's bounding box server-side,
so this pulls a small (~700KB) GeoTIFF rather than the ~1km global raster.

Produces two files:
    market_access/travel_time_2015_ghana.tif
        Raw clipped raster (minutes to nearest city of >50,000 population,
        motorized travel time, 1km resolution) — kept for reference/re-use.
    market_access/market_access_community.csv
        comm, travel_time_to_city_min — raster value sampled at each
        community centroid (rasterio point sample, no representation
        learning needed — a raster value at a point, same pattern as
        CHIRPS rainfall).

Usage:
    python download_market_access.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import rasterio
import requests

WCS_URL = 'https://data.malariaatlas.org/geoserver/ows'
COVERAGE_ID = 'Accessibility__201501_Global_Travel_Time_to_Cities'
# Generous bbox around the 5 LEAP districts (Northern/Upper East regions),
# a few degrees of padding either side.
GHANA_BBOX = dict(lat_min=9.0, lat_max=11.5, lon_min=-1.5, lon_max=0.8)


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/market_access',
                   help='Local directory to save the raster + community CSV '
                        '(default: ../../../data/ghana/market_access)')
    p.add_argument('--data-path',
                   default='../../../data/ghana/survey/LEAP1000 2015-2017 household data++.dta',
                   help='Path to household .dta file, for community centroids')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without downloading/reading/writing files')
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


def download_raster(out_path: Path):
    if out_path.exists():
        log(f"  SKIP download (cached at {out_path})")
        return
    params = {
        'service': 'WCS', 'version': '2.0.1', 'request': 'GetCoverage',
        'coverageId': COVERAGE_ID,
        'subset': [f"Lat({GHANA_BBOX['lat_min']},{GHANA_BBOX['lat_max']})",
                   f"Long({GHANA_BBOX['lon_min']},{GHANA_BBOX['lon_max']})"],
        'format': 'image/geotiff',
    }
    log(f"  Requesting {COVERAGE_ID} clipped to Ghana bbox ...")
    r = requests.get(WCS_URL, params=params, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"  Wrote {out_path} ({len(r.content) / 1e3:.0f} KB)")


def sample_at_centroids(raster_path: Path, centroids: pd.DataFrame) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        vals = [v[0] for v in src.sample(zip(centroids['lon'], centroids['lat']))]
    return pd.DataFrame({'comm': centroids['comm'], 'travel_time_to_city_min': vals})


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    raster_path = out_dir / 'travel_time_2015_ghana.tif'

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Coverage: {COVERAGE_ID}")
        log(f"  Bbox: {GHANA_BBOX}")
        log(f"  Output: {out_dir / 'market_access_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    download_raster(raster_path)

    stats = sample_at_centroids(raster_path, centroids)
    out_path = out_dir / 'market_access_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  travel_time_to_city_min: min={stats['travel_time_to_city_min'].min():.0f}  "
        f"median={stats['travel_time_to_city_min'].median():.0f}  "
        f"max={stats['travel_time_to_city_min'].max():.0f}")


if __name__ == '__main__':
    main()
