"""Download Landsat 8 tiles for a regular grid covering all of Ghana.

These tiles are the SAE *training* corpus — kept separate from the 162 LEAP
community tiles used in the experiment analysis.

Layout on disk:
  data/ghana/satellite/tif/            ← LEAP experiment tiles (do not touch)
  data/ghana/satellite/tif_national/   ← national grid tiles (this script)

Grid spec:
  - 5×5 km tiles at 30 m resolution (identical to LEAP tiles)
  - 5 km centre-to-centre spacing → edge-to-edge, full national coverage
  - 2015 annual cloud-free Landsat 8 SR median composite
  - 6 bands: SR_B4, SR_B3, SR_B2, SR_B5, SR_B6, SR_B7

Speed design:
  The national composite (Landsat 8 median for 2015, all of Ghana) is computed
  ONCE as a lazy GEE image object in the main thread.  Each worker thread only
  clips that image to its tile region and calls getDownloadURL — no redundant
  collection filtering or compositing per tile.  Combined with 32 parallel
  workers (well within GEE's 40-request limit) this typically achieves <1h.

GEE cost: none — free for non-commercial research.

Usage:
  python download_national_grid.py [--workers 32] [--dry-run]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local as thread_local

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Point

# ── Tile spec (must match LEAP tiles) ─────────────────────────────────────────
TILE_KM   = 5.0
YEAR      = 2015
BANDS     = ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6', 'SR_B7']
SCALE     = 30
MAX_CLOUD = 70.0

LON_MIN, LON_MAX = -3.26, 1.19
LAT_MIN, LAT_MAX =  4.74, 11.17

KM_PER_DEG_LON = 111.0 * np.cos(np.radians(8.0))
KM_PER_DEG_LAT = 111.0
STEP_LON = TILE_KM / KM_PER_DEG_LON
STEP_LAT = TILE_KM / KM_PER_DEG_LAT

# Per-thread requests session for connection reuse
_thread = thread_local()


def _session() -> requests.Session:
    if not hasattr(_thread, 'session'):
        _thread.session = requests.Session()
    return _thread.session


def log(msg: str):
    print(msg, flush=True)


# ── Grid ──────────────────────────────────────────────────────────────────────

def build_grid(data_dir: Path) -> list[tuple[float, float, int]]:
    gadm       = gpd.read_file(data_dir / 'gadm41_GHA_1.json')
    ghana_poly = gadm.union_all()

    lons   = np.arange(LON_MIN + STEP_LON / 2, LON_MAX, STEP_LON)
    lats   = np.arange(LAT_MIN + STEP_LAT / 2, LAT_MAX, STEP_LAT)
    points = []
    gid    = 0
    for lat in lats:
        for lon in lons:
            if ghana_poly.contains(Point(lon, lat)):
                points.append((lon, lat, gid))
            gid += 1
    return points


# ── GEE ───────────────────────────────────────────────────────────────────────

def build_national_composite(year: int, bands: list[str], max_cloud: float):
    """Build a single Landsat 8 median composite for all of Ghana (lazy GEE object).

    Workers share this object and only call .clip() + getDownloadURL per tile,
    avoiding redundant collection filtering and compositing.
    """
    import ee
    ghana_bbox = ee.Geometry.BBox(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    return (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(ghana_bbox)
          .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
          .select(bands)
          .map(lambda img: img.multiply(0.0000275).add(-0.2)
                             .copyProperties(img, img.propertyNames()))
          .median()
    )


def make_tile_geom(lon: float, lat: float, half_m: float):
    import ee
    return ee.Geometry.Point([lon, lat]).buffer(half_m).bounds()


def download_one(
    lon: float, lat: float, gid: int,
    composite,           # shared GEE image (lazy, thread-safe to read)
    scale: int,
    out_dir: Path,
    retries: int,
) -> tuple[int, bool, str]:
    dest = out_dir / f'ghana_grid{gid:06d}.tif'
    if dest.exists():
        return gid, True, 'SKIP'

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
            return gid, True, f'OK {len(r.content)//1024}KB'
        except Exception as exc:
            if attempt < retries:
                time.sleep(wait)
                wait *= 2          # exponential backoff
            else:
                return gid, False, f'FAIL: {exc}'

    return gid, False, 'FAIL'


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir',  default='../../data/ghana/satellite/tif_national')
    p.add_argument('--data-dir', default='../../data/ghana')
    p.add_argument('--workers',  type=int, default=32,
                   help='Parallel workers (default 32; GEE limit is 40)')
    p.add_argument('--scale',    type=int, default=SCALE)
    p.add_argument('--retries',  type=int, default=4)
    p.add_argument('--dry-run',  action='store_true')
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent
    out_dir    = (script_dir / args.out_dir).resolve()
    data_dir   = (script_dir / args.data_dir).resolve()

    log('Building Ghana grid …')
    grid     = build_grid(data_dir)
    tile_mb  = (int(TILE_KM * 1000 / SCALE) ** 2 * len(BANDS) * 8) / 1e6
    total_gb = len(grid) * tile_mb / 1024

    log(f'Grid: {len(grid)} tiles  |  {TILE_KM}×{TILE_KM} km  |  {SCALE} m  '
        f'|  {len(BANDS)} bands')
    log(f'Storage: ~{total_gb:.1f} GB   Workers: {args.workers}   '
        f'Est. time: ~{len(grid)*8/args.workers/3600:.1f}–'
        f'{len(grid)*12/args.workers/3600:.1f}h')
    log(f'Output: {out_dir}')

    if args.dry_run:
        log(f'\n[DRY RUN] first 3 points: {grid[:3]}')
        return

    import ee
    ee.Initialize()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build composite once — workers share this lazy GEE object
    log('\nBuilding national composite (lazy GEE object) …')
    composite = build_national_composite(YEAR, BANDS, MAX_CLOUD)
    log('Composite defined. Starting downloads …\n')

    done_set = {p.stem for p in out_dir.glob('*.tif')}
    pending  = [(lon, lat, gid) for lon, lat, gid in grid
                if f'ghana_grid{gid:06d}' not in done_set]
    skipped  = len(grid) - len(pending)
    if skipped:
        log(f'Resuming: {skipped} already done, {len(pending)} remaining.\n')

    ok   = skipped
    fail = 0
    t0   = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, lon, lat, gid,
                        composite, args.scale, out_dir, args.retries): gid
            for lon, lat, gid in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            gid, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                log(f'  FAIL grid{gid:06d}: {msg}')

            # Print progress every 50 tiles
            if i % 50 == 0 or i == len(pending):
                elapsed = time.time() - t0
                rate    = i / max(elapsed, 1)
                eta     = (len(pending) - i) / max(rate, 1e-9)
                log(f'  [{i:5d}/{len(pending)}]  {ok} ok  {fail} fail  '
                    f'{rate:.2f} t/s  ETA {eta/60:.0f} min')

    elapsed = time.time() - t0
    log(f'\nDone in {elapsed/3600:.2f}h: {ok} downloaded, {fail} failed → {out_dir}')
    if fail:
        log('Re-run to retry failed tiles (completed tiles are skipped automatically).')


if __name__ == '__main__':
    main()
