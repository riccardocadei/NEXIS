"""Download 2017 Landsat 8 composites for the 6 LEAP communities activated
by SAE neuron 3821 (ephemeral waterways).

Usage:
    python download_waterway_2017.py [--dry-run]

Output:
    data/ghana/satellite/tif_2017/ghana_comm{id:04d}.tif
"""

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "data" / "ghana" / "satellite" / "tif_2017"
YEAR     = 2017
BANDS    = ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6', 'SR_B7']
SCALE    = 30
TILE_KM  = 5.0
MAX_CLOUD = 70.0

# Neuron 3821 active communities (comm_id → lat, lon)
COMMUNITIES = {
    951:  (10.291571, -0.280648),
    675:  (10.860487, -0.733824),
    395:  (10.844102, -0.839544),
    1265: (10.886461, -0.812530),
    655:  (10.941933, -0.765417),
    624:  (10.878216, -0.165861),
}


def log(msg):
    print(msg, flush=True)


def make_tile(lon, lat, half_m):
    import ee
    return ee.Geometry.Point([float(lon), float(lat)]).buffer(half_m).bounds()


def build_composite(year, tile):
    import ee
    return (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(tile)
          .filter(ee.Filter.lt('CLOUD_COVER', MAX_CLOUD))
          .select(BANDS)
          .map(lambda img: img.multiply(0.0000275).add(-0.2)
                             .copyProperties(img, img.propertyNames()))
          .median()
          .clip(tile)
    )


def download_tile(composite, tile, comm_id, out_dir, retries=3):
    dest = out_dir / f'ghana_comm{comm_id:04d}.tif'
    if dest.exists():
        log(f"  SKIP  comm{comm_id:04d} (exists)")
        return True
    params = {'scale': SCALE, 'crs': 'EPSG:4326', 'region': tile, 'format': 'GEO_TIFF'}
    for attempt in range(1, retries + 1):
        try:
            log(f"  GET   comm{comm_id:04d} attempt {attempt} ...")
            url = composite.getDownloadURL(params)
            r   = requests.get(url, timeout=180)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log(f"  OK    comm{comm_id:04d} ({len(r.content)/1024:.0f} KB)")
            return True
        except Exception as exc:
            if attempt < retries:
                wait = 15 * attempt
                log(f"  RETRY comm{comm_id:04d} — {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                log(f"  FAIL  comm{comm_id:04d} — {exc}")
                return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    half_m = TILE_KM * 1000 / 2

    if args.dry_run:
        log(f"[DRY RUN] Would download {len(COMMUNITIES)} tiles for year {YEAR}")
        for comm_id, (lat, lon) in COMMUNITIES.items():
            log(f"  comm{comm_id:04d}  lat={lat:.6f}  lon={lon:.6f}")
        log(f"  Output dir: {OUT_DIR}")
        return

    import ee
    ee.Initialize()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for comm_id, (lat, lon) in COMMUNITIES.items():
        tile      = make_tile(lon, lat, half_m)
        composite = build_composite(YEAR, tile)
        if download_tile(composite, tile, comm_id, OUT_DIR):
            ok += 1
        else:
            fail += 1

    log(f"\nDone: {ok} downloaded, {fail} failed → {OUT_DIR}")


if __name__ == '__main__':
    main()
