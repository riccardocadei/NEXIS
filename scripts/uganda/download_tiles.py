"""Download Landsat 5 TM tiles for Uganda via Google Earth Engine.

Two modes:
  --mode rct       → 331 RCT site tiles (centred on UgandaDataProcessed.csv centroids)
  --mode national  → regular 5×5 km grid covering all of Uganda (SAE training corpus)

Satellite source:
  LANDSAT/LT05/C02/T1_L2  (Landsat 5 TM, Collection 2 Tier 1 Level 2)
  Cloud-free median composite over 2005–2007 (pre-treatment; YOP disbursements began 2008).
  6 optical bands, in this TIF order:
    0: SR_B1  Blue   (0.45–0.52 µm)
    1: SR_B2  Green  (0.52–0.60 µm)
    2: SR_B3  Red    (0.63–0.69 µm)
    3: SR_B4  NIR    (0.76–0.90 µm)
    4: SR_B5  SWIR-1 (1.55–1.75 µm)
    5: SR_B7  SWIR-2 (2.08–2.35 µm)
  Scale factor: ×0.0000275 − 0.2  (standard Landsat Collection 2 Level 2 SR)

Layout on disk:
  data/uganda/satellite/tif_rct/        ← RCT tiles   uganda_rct{key:06d}.tif
  data/uganda/satellite/tif_national/   ← national grid uganda_grid{gid:06d}.tif

Usage:
  python scripts/uganda/download_tiles.py --mode rct
  python scripts/uganda/download_tiles.py --mode national [--workers 32]
  python scripts/uganda/download_tiles.py --mode rct --dry-run

GEE cost: none — free for non-commercial research.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local as thread_local

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

# ── Tile / composite spec ──────────────────────────────────────────────────────
TILE_KM    = 5.0
YEAR_START = 2005   # pre-treatment: YOP disbursements began 2008
YEAR_END   = 2007
BANDS      = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
SCALE      = 30          # metres per pixel
MAX_CLOUD  = 70.0        # % cloud cover filter

# Uganda bounding box (WGS-84); GADM boundary used for national grid filtering
LON_MIN, LON_MAX = 29.5, 35.1
LAT_MIN, LAT_MAX = -1.5, 4.25

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(1.4))   # ~central Uganda latitude
STEP_LON = TILE_KM / KM_PER_DEG_LON
STEP_LAT = TILE_KM / KM_PER_DEG_LAT

_thread = thread_local()


def _session() -> requests.Session:
    if not hasattr(_thread, 'session'):
        _thread.session = requests.Session()
    return _thread.session


def log(msg: str):
    print(msg, flush=True)


# ── GEE composite ─────────────────────────────────────────────────────────────

def build_composite(year_start: int, year_end: int, bands: list[str], max_cloud: float):
    """Landsat 5 median composite over [year_start, year_end] (lazy GEE object)."""
    import ee
    bbox = ee.Geometry.BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    return (
        ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
          .filterDate(f'{year_start}-01-01', f'{year_end}-12-31')
          .filterBounds(bbox)
          .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
          .select(bands)
          .map(lambda img: img.multiply(0.0000275).add(-0.2)
                             .copyProperties(img, img.propertyNames()))
          .median()
    )


def make_tile_geom(lon: float, lat: float, half_m: float):
    import ee
    return ee.Geometry.Point([lon, lat]).buffer(half_m).bounds()


# ── Single-tile download ──────────────────────────────────────────────────────

def download_one(
    lon: float, lat: float, tile_id: str,
    composite,           # shared GEE image (lazy, thread-safe to read)
    scale: int,
    out_dir: Path,
    retries: int,
) -> tuple[str, bool, str]:
    dest = out_dir / f'{tile_id}.tif'
    if dest.exists():
        return tile_id, True, 'SKIP'

    half_m    = (TILE_KM * 1000) / 2
    tile_geom = make_tile_geom(lon, lat, half_m)
    tile_img  = composite.clip(tile_geom)
    params    = {'scale': scale, 'crs': 'EPSG:4326',
                 'region': tile_geom, 'format': 'GEO_TIFF'}

    wait = 10
    for attempt in range(1, retries + 1):
        try:
            url = tile_img.getDownloadURL(params)
            r   = _session().get(url, timeout=180)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return tile_id, True, f'OK {len(r.content)//1024}KB'
        except Exception as exc:
            if attempt < retries:
                time.sleep(wait)
                wait *= 2
            else:
                return tile_id, False, f'FAIL: {exc}'

    return tile_id, False, 'FAIL'


# ── Tile lists ────────────────────────────────────────────────────────────────

def load_rct_sites(data_dir: Path) -> list[tuple[float, float, str]]:
    """Return (lon, lat, tile_id) for all 331 RCT site centroids."""
    df = pd.read_csv(data_dir / 'UgandaDataProcessed.csv', low_memory=False,
                     usecols=['geo_long_lat_key', 'geo_long_center', 'geo_lat_center'])
    sites = (df.drop_duplicates('geo_long_lat_key')
               .dropna(subset=['geo_long_center', 'geo_lat_center'])
               .sort_values('geo_long_lat_key'))
    return [(row.geo_long_center, row.geo_lat_center,
             f'uganda_rct{int(row.geo_long_lat_key):06d}')
            for _, row in sites.iterrows()]


def build_national_grid(data_dir: Path) -> list[tuple[float, float, str]]:
    """Return (lon, lat, tile_id) for a 5×5 km grid covering all of Uganda."""
    gadm    = gpd.read_file(data_dir / 'map' / 'gadm41_UGA_1.json')
    uganda  = gadm.union_all()

    lons   = np.arange(LON_MIN + STEP_LON / 2, LON_MAX, STEP_LON)
    lats   = np.arange(LAT_MIN + STEP_LAT / 2, LAT_MAX, STEP_LAT)
    points = []
    gid    = 0
    for lat in lats:
        for lon in lons:
            if uganda.contains(Point(lon, lat)):
                points.append((lon, lat, f'uganda_grid{gid:06d}'))
            gid += 1
    return points


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',     choices=['rct', 'national'], required=True,
                   help='rct → 331 experimental sites; national → full Uganda grid')
    p.add_argument('--data-dir', default='../../data/uganda',
                   help='Path to data/uganda (default: relative to this script)')
    p.add_argument('--out-dir',  default=None,
                   help='Override output directory (default: data/uganda/satellite/tif_{mode})')
    p.add_argument('--workers',  type=int, default=32,
                   help='Parallel GEE workers (GEE limit is 40, default 32)')
    p.add_argument('--scale',    type=int, default=SCALE,
                   help='Pixel resolution in metres (default 30)')
    p.add_argument('--retries',  type=int, default=4)
    p.add_argument('--dry-run',  action='store_true',
                   help='Print plan without downloading')
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent
    data_dir   = (script_dir / args.data_dir).resolve()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = data_dir / 'satellite' / ('tif_rct' if args.mode == 'rct' else 'tif_national')

    log(f'Mode      : {args.mode}')
    log(f'Composite : Landsat 5 TM  {YEAR_START}–{YEAR_END} median  '
        f'(cloud < {MAX_CLOUD}%)')
    log(f'Bands     : {BANDS}')
    log(f'Scale     : {args.scale} m  |  Tile: {TILE_KM}×{TILE_KM} km')

    if args.mode == 'rct':
        tiles = load_rct_sites(data_dir)
        log(f'RCT sites : {len(tiles)}')
    else:
        log('Building national grid …')
        tiles = build_national_grid(data_dir)
        log(f'National  : {len(tiles)} tiles inside Uganda boundary')

    tile_px  = int(TILE_KM * 1000 / args.scale)
    tile_mb  = (tile_px ** 2 * len(BANDS) * 4) / 1e6
    total_gb = len(tiles) * tile_mb / 1024
    log(f'Storage   : ~{tile_mb:.1f} MB/tile  ×  {len(tiles)} = ~{total_gb:.1f} GB')
    log(f'Workers   : {args.workers}  |  '
        f'Est. time: {len(tiles)*8/args.workers/3600:.1f}–'
        f'{len(tiles)*12/args.workers/3600:.1f}h')
    log(f'Output    : {out_dir}')

    if args.dry_run:
        log(f'\n[DRY RUN] first 3 tiles: {tiles[:3]}')
        return

    import ee
    ee.Initialize()
    out_dir.mkdir(parents=True, exist_ok=True)

    log('\nBuilding Landsat 5 composite (lazy GEE object) …')
    composite = build_composite(YEAR_START, YEAR_END, BANDS, MAX_CLOUD)
    log('Composite ready. Starting downloads …\n')

    done_set = {p.stem for p in out_dir.glob('*.tif')}
    pending  = [(lon, lat, tid) for lon, lat, tid in tiles if tid not in done_set]
    skipped  = len(tiles) - len(pending)
    if skipped:
        log(f'Resuming: {skipped} already done, {len(pending)} remaining.\n')

    ok   = skipped
    fail = 0
    t0   = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, lon, lat, tid,
                        composite, args.scale, out_dir, args.retries): tid
            for lon, lat, tid in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            tid, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                log(f'  FAIL {tid}: {msg}')

            if i % 50 == 0 or i == len(pending):
                elapsed = time.time() - t0
                rate    = i / max(elapsed, 1)
                eta     = (len(pending) - i) / max(rate, 1e-9)
                log(f'  [{i:5d}/{len(pending)}]  {ok} ok  {fail} fail  '
                    f'{rate:.2f} t/s  ETA {eta/60:.0f} min')

    elapsed = time.time() - t0
    log(f'\nDone in {elapsed/3600:.2f}h: {ok} ok, {fail} failed → {out_dir}')
    if fail:
        log('Re-run to retry failed tiles (completed tiles are skipped automatically).')


if __name__ == '__main__':
    main()
