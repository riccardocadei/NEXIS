"""Download Landsat 8 satellite images for Ghana LEAP 1000 community centroids.

For each community, builds a 5×5 km cloud-free annual composite (2015) on
GEE and downloads it directly as a multi-band GeoTIFF — no Google Drive needed.

Setup (one-time):
    pip install earthengine-api requests
    earthengine authenticate --auth_mode notebook
    earthengine set_project <your-project-id>

Usage:
    python download_satellite_images.py [--out-dir ../../data/ghana/satellite]
                                        [--tile-km 5]
                                        [--year 2015]
                                        [--bands B4 B3 B2 B5 B6 B7]
                                        [--scale 30]
                                        [--max-cloud 70]
                                        [--dry-run]

Bands (Landsat 8 SR):
    B2  Blue              482 nm
    B3  Green             562 nm
    B4  Red               655 nm
    B5  NIR               865 nm
    B6  SWIR-1           1609 nm
    B7  SWIR-2           2201 nm
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../data/ghana/satellite',
                   help='Local directory to save GeoTIFFs (default: ../../data/ghana/satellite)')
    p.add_argument('--tile-km', type=float, default=5.0,
                   help='Tile side length in km (default: 5)')
    p.add_argument('--year', type=int, default=2015,
                   help='Year for annual composite (default: 2015)')
    p.add_argument('--bands', nargs='+',
                   default=['B4', 'B3', 'B2', 'B5', 'B6', 'B7'],
                   help='Landsat 8 bands to include (default: B4 B3 B2 B5 B6 B7)')
    p.add_argument('--scale', type=int, default=30,
                   help='Resolution in metres (default: 30 = Landsat native)')
    p.add_argument('--max-cloud', type=float, default=70.0,
                   help='Max scene-level cloud cover %% to include (default: 70)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print plan without downloading anything')
    p.add_argument('--data-path',
                   default='../../data/ghana/LEAP1000 2015-2017 household data++.dta',
                   help='Path to household .dta file')
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_community_centroids(data_path: str) -> pd.DataFrame:
    df = pd.read_stata(data_path)
    centroids = (
        df.dropna(subset=['gps_latitude', 'gps_longitude'])
          .groupby('comm')[['gps_latitude', 'gps_longitude']]
          .first()
          .reset_index()
    )
    centroids.columns = ['comm_id', 'lat', 'lon']
    n_missing = df['comm'].nunique() - len(centroids)
    print(f"Loaded {len(centroids)} community centroids ({n_missing} missing GPS skipped)")
    return centroids


# ── GEE helpers ───────────────────────────────────────────────────────────────

def make_tile(lon: float, lat: float, half_m: float):
    import ee
    centre = ee.Geometry.Point([float(lon), float(lat)])
    return centre.buffer(half_m).bounds()


def build_composite(year: int, tile, bands: list[str], max_cloud: float):
    import ee
    collection = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(tile)
          .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
          .select(bands)
          .map(lambda img: img.multiply(0.0000275).add(-0.2)
                             .copyProperties(img, img.propertyNames()))
    )
    return collection.median().clip(tile)


# ── Download ──────────────────────────────────────────────────────────────────

def download_tile(composite, tile, comm_id: int, scale: int,
                  out_dir: Path, retries: int = 3) -> bool:
    import ee
    dest = out_dir / f'ghana_comm{comm_id:04d}.tif'
    if dest.exists():
        print(f"  SKIP  comm{comm_id:04d}  (already exists)")
        return True

    params = {
        'scale':  scale,
        'crs':    'EPSG:4326',
        'region': tile,
        'format': 'GEO_TIFF',
    }

    for attempt in range(1, retries + 1):
        try:
            url = composite.getDownloadURL(params)
            r   = requests.get(url, timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"  OK    comm{comm_id:04d}  ({len(r.content) / 1024:.0f} KB)")
            return True
        except Exception as exc:
            if attempt < retries:
                wait = 10 * attempt
                print(f"  RETRY comm{comm_id:04d}  attempt {attempt} — {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  FAIL  comm{comm_id:04d}  — {exc}")
                return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    centroids = load_community_centroids(str(data_path))

    if args.dry_run:
        tile_px = int((args.tile_km * 1000) / args.scale)
        size_kb = tile_px * tile_px * len(args.bands) * 4 / 1024
        print(f"\n[DRY RUN] {len(centroids)} communities")
        print(f"  Year: {args.year}  |  Tile: {args.tile_km}×{args.tile_km} km  "
              f"|  {tile_px}×{tile_px} px  |  ~{size_kb:.0f} KB/tile")
        print(f"  Bands: {args.bands}  |  Scale: {args.scale} m  "
              f"|  Total: ~{size_kb * len(centroids) / 1024:.0f} MB")
        print(f"  Output: {out_dir}")
        for _, row in centroids.iterrows():
            print(f"  comm{int(row.comm_id):04d}  lat={row.lat:.4f}  lon={row.lon:.4f}")
        return

    import ee
    ee.Initialize()
    out_dir.mkdir(parents=True, exist_ok=True)

    half_m  = (args.tile_km * 1000) / 2
    ok, fail = 0, 0

    print(f"\nDownloading {len(centroids)} tiles → {out_dir}\n")
    for i, (_, row) in enumerate(centroids.iterrows(), 1):
        tile      = make_tile(row.lon, row.lat, half_m)
        composite = build_composite(args.year, tile, args.bands, args.max_cloud)
        success   = download_tile(composite, tile, int(row.comm_id),
                                  args.scale, out_dir)
        if success:
            ok += 1
        else:
            fail += 1
        print(f"  [{i}/{len(centroids)}]  {ok} ok  {fail} failed", end='\r')

    print(f"\n\nDone: {ok} downloaded, {fail} failed → {out_dir}")
    if fail:
        print("Re-run the script to retry failed tiles (completed tiles are skipped).")


if __name__ == '__main__':
    main()
