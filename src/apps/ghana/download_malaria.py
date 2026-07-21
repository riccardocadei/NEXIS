"""Download Plasmodium falciparum (malaria) mortality/incidence rate for
Ghana LEAP 1000 community centroids, into community-level covariates for
NEXIS (see external_data.py::load_effect_modifiers).

Source: Malaria Atlas Project, via the same public WCS endpoint already
used for market access (data.malariaatlas.org), no account/API key needed.
Unlike the city-accessibility layer (a single 2015 snapshot), this
collection is a genuine annual time series (2000-2022, 5km resolution), so
2015 is selected via a WCS time subset -- an exact match to the LEAP
baseline year, same as market access, not a present-day-only snapshot the
way OpenCellID/Ookla were.

Motivation: malaria is a leading cause of under-5 mortality in Ghana, and
LEAP-1000 specifically targets pregnant women and children under 1 -- a
directly relevant "children's care" exposure proxy, checked for degeneracy
before adopting (0/162 communities at zero for either variable).

Two covariates, not one: mortality and incidence turned out to be
*negatively* correlated (r=-0.62) -- plausibly because higher-incidence
areas sometimes have better malaria program targeting/case management,
lowering mortality despite more cases -- so they capture genuinely
different aspects of the local disease environment, not one redundant with
the other.

Produces two raw rasters (Ghana-clipped, ~100KB each) and one community CSV:
    malaria/pf_mortality_2015_ghana.tif
    malaria/pf_incidence_2015_ghana.tif
    malaria/malaria_community.csv
        comm, malaria_mortality_rate_2015, malaria_incidence_rate_2015

Usage:
    python download_malaria.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import rasterio
import requests

WCS_URL = 'https://data.malariaatlas.org/geoserver/ows'
MORTALITY_COVERAGE = 'Malaria__202406_Global_Pf_Mortality_Rate'
INCIDENCE_COVERAGE = 'Malaria__202406_Global_Pf_Incidence_Rate'
YEAR = 2015
GHANA_BBOX = dict(lat_min=9.0, lat_max=11.5, lon_min=-1.5, lon_max=0.8)


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/malaria',
                   help='Local directory to save rasters + community CSV '
                        '(default: ../../../data/ghana/malaria)')
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


def download_raster(coverage_id: str, out_path: Path):
    if out_path.exists():
        log(f"  SKIP download (cached at {out_path})")
        return
    params = {
        'service': 'WCS', 'version': '2.0.1', 'request': 'GetCoverage',
        'coverageId': coverage_id,
        'subset': [f"Lat({GHANA_BBOX['lat_min']},{GHANA_BBOX['lat_max']})",
                   f"Long({GHANA_BBOX['lon_min']},{GHANA_BBOX['lon_max']})",
                   f'time("{YEAR}-01-01T00:00:00.000Z")'],
        'format': 'image/geotiff',
    }
    log(f"  Requesting {coverage_id} ({YEAR}) clipped to Ghana bbox ...")
    r = requests.get(WCS_URL, params=params, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"  Wrote {out_path} ({len(r.content) / 1e3:.0f} KB)")


def sample_at_centroids(raster_path: Path, centroids: pd.DataFrame, colname: str) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        vals = [v[0] for v in src.sample(zip(centroids['lon'], centroids['lat']))]
    return pd.DataFrame({'comm': centroids['comm'], colname: vals})


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    mortality_path = out_dir / f'pf_mortality_{YEAR}_ghana.tif'
    incidence_path = out_dir / f'pf_incidence_{YEAR}_ghana.tif'

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Coverages: {MORTALITY_COVERAGE}, {INCIDENCE_COVERAGE}  year={YEAR}")
        log(f"  Output: {out_dir / 'malaria_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    download_raster(MORTALITY_COVERAGE, mortality_path)
    download_raster(INCIDENCE_COVERAGE, incidence_path)

    mortality = sample_at_centroids(mortality_path, centroids, 'malaria_mortality_rate_2015')
    incidence = sample_at_centroids(incidence_path, centroids, 'malaria_incidence_rate_2015')
    stats = mortality.merge(incidence, on='comm')

    out_path = out_dir / 'malaria_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  malaria_mortality_rate_2015: min={stats['malaria_mortality_rate_2015'].min():.5f}  "
        f"median={stats['malaria_mortality_rate_2015'].median():.5f}  "
        f"max={stats['malaria_mortality_rate_2015'].max():.5f}")
    log(f"  malaria_incidence_rate_2015: min={stats['malaria_incidence_rate_2015'].min():.3f}  "
        f"median={stats['malaria_incidence_rate_2015'].median():.3f}  "
        f"max={stats['malaria_incidence_rate_2015'].max():.3f}")


if __name__ == '__main__':
    main()
