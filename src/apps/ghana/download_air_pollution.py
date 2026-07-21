"""Download satellite-derived PM2.5 air pollution data for Ghana LEAP 1000
community centroids, into a community-level covariate for NEXIS (see
external_data.py::load_effect_modifiers).

Source: Washington University ACAG SatPM2.5 (V6.GL.03) -- fine particulate
matter estimates fusing satellite aerosol optical depth, GEOS-Chem
simulation, and ground monitors. Public AWS Open Data (s3://satpmdata/),
no account/API key, CC-BY 4.0. Annual Africa-clipped composite, 0.1deg
resolution, includes 2015 exactly -- no temporal-mismatch trade-off, same
as market access/malaria.

Produces one covariate: pm25_2015, the nearest-gridcell PM2.5 concentration
(ug/m3) at each community centroid. Checked for degeneracy before adopting:
0/162 communities hit the fill value, range 32-37 ug/m3 (std 1.3). Notably
correlated with rainfall_mean_pre2015 (r=0.75) -- physically sensible, this
Sahel region's air quality is Harmattan-dust-driven and rainfall tracks
distance from the Sahara -- comparable in magnitude to the already-accepted
rainfall/malaria correlation (r=0.87), not a reason to drop.

Usage:
    python download_air_pollution.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PM25_URL = ('https://satpmdata.s3.amazonaws.com/V6GL03/CoarseResolution/AF/Annual/'
            'V6GL03.CNNPM25.0p10.AF.201501-201512.nc')
FILL_VALUE = -999.0


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/air_pollution',
                   help='Local directory to save the raster + community CSV '
                        '(default: ../../../data/ghana/air_pollution)')
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
    log(f"  Downloading {PM25_URL} ...")
    r = requests.get(PM25_URL, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"  Wrote {out_path} ({len(r.content) / 1e3:.0f} KB)")


def sample_at_centroids(nc_path: Path, centroids: pd.DataFrame) -> pd.DataFrame:
    """Nearest-gridcell lookup (0.1deg resolution) -- no interpolation,
    same simple point-sample convention as the rasterio-based sources."""
    import netCDF4 as nc
    ds = nc.Dataset(nc_path)
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    pm25 = ds.variables['PM25'][:]

    rows = []
    for _, comm in centroids.iterrows():
        i = np.abs(lat - comm['lat']).argmin()
        j = np.abs(lon - comm['lon']).argmin()
        val = float(pm25[i, j])
        rows.append({'comm': comm['comm'], 'pm25_2015': np.nan if val == FILL_VALUE else val})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    nc_path    = out_dir / 'pm25_2015_africa.nc'

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Source: {PM25_URL}")
        log(f"  Output: {out_dir / 'air_pollution_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    download_raster(nc_path)

    stats = sample_at_centroids(nc_path, centroids)
    out_path = out_dir / 'air_pollution_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  pm25_2015: min={stats['pm25_2015'].min():.2f}  "
        f"median={stats['pm25_2015'].median():.2f}  "
        f"max={stats['pm25_2015'].max():.2f} ug/m3")


if __name__ == '__main__':
    main()
