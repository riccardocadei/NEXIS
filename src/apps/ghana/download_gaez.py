"""Download FAO GAEZ v4 agroecological suitability data for Ghana LEAP 1000
community centroids, into a community-level covariate for NEXIS (see
external_data.py::load_effect_modifiers).

Source: FAO GAEZ v4 (Global Agro-Ecological Zones), via its public,
unauthenticated ArcGIS ImageServer (res05 = Theme 4, Suitability and
Attainable Yield) -- no account/API key. Layer selected via a mosaic-rule
`where` clause on the catalog's `name` attribute: `suHr0_mze` = rainfed,
high-input-level maize suitability index (continuous, 0-10000, higher =
more suitable land). Maize, matching the staple-crop choice already made
for market prices (see download_market_prices.py) -- Northern Ghana's
actual staple, not an arbitrary pick.

Genuinely time-invariant, unlike every other "should we worry about the
year" source in this project: GAEZ v4's baseline represents 1981-2010
climate normals, not a rolling annual series, so there's no year-matching
question to resolve at all -- soil/agroclimatic suitability doesn't change
year to year the way a market price or a cell-tower registry does.
Timing.HISTORIC (same role as rainfall's 2000-2014 climatology): "what
this land is normally like," not "right before treatment."

Distinct from rainfall (realized weather/shocks, varies year to year) and
from market prices (what a household can currently buy) -- this captures
the land's *ceiling*: whether a farming household's soil can support
intensified maize cultivation at all, regardless of a given year's weather.

The exportImage request is clipped to Ghana's bounding box server-side, so
this pulls a small raster rather than the ~9km-resolution global one.

Produces two files:
    gaez/maize_suitability_ghana.tif
        Raw clipped raster (rainfed high-input maize suitability index,
        0-10000, ~9km/5-arcmin resolution) -- kept for reference/re-use.
    gaez/gaez_community.csv
        comm, maize_suitability_index -- raster value sampled at each
        community centroid (rasterio point sample, no representation
        learning needed -- a raster value at a point, same pattern as
        CHIRPS rainfall / market access).

Usage:
    python download_gaez.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import rasterio
import requests

IMAGE_SERVER_URL = 'https://gaez-services.fao.org/server/rest/services/res05/ImageServer/exportImage'
LAYER_NAME = 'suHr0_mze'   # rainfed, high-input maize suitability index (0-10000)
# Generous bbox around the 5 LEAP districts (Northern/Upper East regions),
# a few degrees of padding either side -- same box as download_market_access.py.
GHANA_BBOX = dict(lat_min=9.0, lat_max=11.5, lon_min=-1.5, lon_max=0.8)


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/gaez',
                   help='Local directory to save the raster + community CSV '
                        '(default: ../../../data/ghana/gaez)')
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
        'bbox': f"{GHANA_BBOX['lon_min']},{GHANA_BBOX['lat_min']},"
                f"{GHANA_BBOX['lon_max']},{GHANA_BBOX['lat_max']}",
        'bboxSR': 4326, 'imageSR': 4326,
        'size': '400,400',
        'format': 'tiff', 'pixelType': 'F32',
        'noDataInterpretation': 'esriNoDataMatchAny',
        'mosaicRule': f'{{"where":"name=\'{LAYER_NAME}\'"}}',
        'f': 'image',
    }
    log(f"  Requesting GAEZ layer {LAYER_NAME} clipped to Ghana bbox ...")
    r = requests.get(IMAGE_SERVER_URL, params=params, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"  Wrote {out_path} ({len(r.content) / 1e3:.0f} KB)")


def sample_at_centroids(raster_path: Path, centroids: pd.DataFrame) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        vals = [v[0] for v in src.sample(zip(centroids['lon'], centroids['lat']))]
    return pd.DataFrame({'comm': centroids['comm'], 'maize_suitability_index': vals})


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    raster_path = out_dir / 'maize_suitability_ghana.tif'

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Layer: {LAYER_NAME} (rainfed, high-input maize suitability, 0-10000)")
        log(f"  Bbox: {GHANA_BBOX}")
        log(f"  Output: {out_dir / 'gaez_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    download_raster(raster_path)

    stats = sample_at_centroids(raster_path, centroids)
    out_path = out_dir / 'gaez_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  maize_suitability_index: min={stats['maize_suitability_index'].min():.0f}  "
        f"median={stats['maize_suitability_index'].median():.0f}  "
        f"max={stats['maize_suitability_index'].max():.0f}")


if __name__ == '__main__':
    main()
