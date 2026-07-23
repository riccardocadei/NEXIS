"""Download VIIRS nighttime-lights data for Ghana LEAP 1000 community
centroids, into community-level covariates for NEXIS (see
external_data.py::load_effect_modifiers).

Source: NOAA/VIIRS/DNB/ANNUAL_V21 (stray-light-corrected annual composite),
via Google Earth Engine -- same authentication already used for rainfall/
satellite imagery, no new account needed. 2015 composite, matching the LEAP
baseline year exactly.

Two covariates, deliberately not one: they answer different questions and
turned out to be only weakly correlated with each other (r=-0.25) and with
existing covariates.

  - night_light_radiance: mean radiance within 1km of the community centroid
    -- "is this specific community itself electrified/economically active".
    Sparse: only ~15/162 communities show any detectable light at this
    scale (checked at multiple buffer radii up to 10km -- genuinely a fact
    about these deep-rural communities, not a radius artifact), so this is
    registered as Support.SPARSE_NONNEG, same convention as SAE activations.
    Weakly correlated with dist_to_capital_km/community_size/travel_time_to_city_min
    (|r| <= 0.22) -- a fairly independent fact (a specific market centre or
    school having solar/grid lighting doesn't track general remoteness).

  - dist_nearest_light_km: distance from the community centroid to the
    nearest pixel with radiance > 1.0 (a standard "detectable urban light"
    threshold), searched across the whole Ghana bounding box, always
    defined. Answers "how far is this community from an electrified area" --
    more of a remoteness/access proxy, correlated with dist_to_capital_km
    (r=0.66) and travel_time_to_city_min (r=0.52), but not so high as to be
    a pure re-derivation.

A third covariate, night_light_trend, answers a different question again:
not "how lit is this community" (a level) but "was it getting more lit in
the run-up to treatment" (a trend) -- a local economic-momentum proxy
(Henderson, Storeygard & Weil 2012, AER, is the canonical satellite-
nightlights-as-growth-proxy citation), distinct from a snapshot level.
Computed as night_light_radiance(2015) - night_light_radiance(TREND_YEAR).

TREND_YEAR=2013, not 2012: checked against the collection's own catalog
metadata first -- NOAA/GEE explicitly flag 2012 as inconsistent ("2012
data are not yet included because of differences in processing"), so the
clean annual series only starts at 2013. Cross-year radiometric-consistency
caveat also checked: the catalog doesn't carry an explicit DMSP-OLS-style
intercalibration warning, but does expose a `cf_cvg` (cloud-free
observation count) band as a quality signal. Checked per-community cf_cvg
for both 2013 and 2015 before trusting the difference -- every one of the
162 communities has cf_cvg > 80 in both years (median ~91-94), so no
masking was actually needed for this specific set of communities (unlike,
say, a rainforest-zone study area might require).

Usage:
    python download_nightlights.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

VIIRS_COLLECTION = 'NOAA/VIIRS/DNB/ANNUAL_V21'
YEAR = 2015
TREND_YEAR = 2013             # earliest year with a clean (non-2012-flagged) annual composite
LIGHT_THRESHOLD = 1.0        # radiance above which a pixel counts as "lit"
OWN_COMMUNITY_BUFFER_M = 1000
GHANA_BBOX = dict(lat_min=9.0, lat_max=11.5, lon_min=-1.5, lon_max=0.8)


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/nightlights',
                   help='Local directory to save the raster + community CSV '
                        '(default: ../../../data/ghana/nightlights)')
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
    import ee
    ee.Initialize()
    img = (ee.ImageCollection(VIIRS_COLLECTION)
             .filterDate(f'{YEAR}-01-01', f'{YEAR + 1}-01-01')
             .first().select('average_masked'))
    region = ee.Geometry.Rectangle([GHANA_BBOX['lon_min'], GHANA_BBOX['lat_min'],
                                     GHANA_BBOX['lon_max'], GHANA_BBOX['lat_max']])
    url = img.clip(region).getDownloadURL({
        'scale': 500, 'crs': 'EPSG:4326', 'region': region, 'format': 'GEO_TIFF',
    })
    log(f"  Downloading VIIRS {YEAR} annual composite (Ghana bbox) ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"  Wrote {out_path} ({len(r.content) / 1e3:.0f} KB)")


def own_community_radiance(centroids: pd.DataFrame, year: int, col: str) -> pd.DataFrame:
    """Mean radiance within OWN_COMMUNITY_BUFFER_M of each centroid, via GEE
    reduceRegions -- direct measurement of the community's own detectable
    light, not a proxy sampled from a downloaded raster."""
    import ee
    ee.Initialize()
    img = (ee.ImageCollection(VIIRS_COLLECTION)
             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
             .first().select('average_masked'))
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)])
                     .buffer(OWN_COMMUNITY_BUFFER_M),
                   {'comm_id': int(row.comm)})
        for _, row in centroids.iterrows()
    ])
    reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500).getInfo()
    rows = [
        {'comm': f['properties']['comm_id'], col: f['properties'].get('mean')}
        for f in reduced['features']
    ]
    return pd.DataFrame(rows)


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def dist_to_nearest_light(centroids: pd.DataFrame, raster_path: Path) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        rows_idx, cols_idx = np.where(band > LIGHT_THRESHOLD)
        lons, lats = rasterio.transform.xy(src.transform, rows_idx, cols_idx)
    lit_lons, lit_lats = np.array(lons), np.array(lats)

    rows = []
    for _, comm in centroids.iterrows():
        dist_km = _haversine_km(comm['lat'], comm['lon'], lit_lats, lit_lons)
        rows.append({
            'comm': comm['comm'],
            'dist_nearest_light_km': dist_km.min() if len(dist_km) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    raster_path = out_dir / f'viirs_{YEAR}_ghana.tif'

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Collection: {VIIRS_COLLECTION}  year={YEAR}  trend_year={TREND_YEAR}")
        log(f"  Output: {out_dir / 'nightlights_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    download_raster(raster_path)

    radiance = own_community_radiance(centroids, YEAR, 'night_light_radiance')
    trend_radiance = own_community_radiance(centroids, TREND_YEAR, 'night_light_radiance_trend_year')
    dist = dist_to_nearest_light(centroids, raster_path)
    stats = radiance.merge(trend_radiance, on='comm').merge(dist, on='comm')
    stats['night_light_trend'] = stats['night_light_radiance'] - stats['night_light_radiance_trend_year']
    stats = stats.drop(columns=['night_light_radiance_trend_year'])

    out_path = out_dir / 'nightlights_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    n_lit = (stats['night_light_radiance'] > 0.1).sum()
    log(f"  {n_lit}/{len(stats)} communities detectably lit (radiance > 0.1) at own location")
    log(f"  dist_nearest_light_km: min={stats['dist_nearest_light_km'].min():.2f}  "
        f"median={stats['dist_nearest_light_km'].median():.2f}  "
        f"max={stats['dist_nearest_light_km'].max():.2f}")
    log(f"  night_light_trend ({YEAR} - {TREND_YEAR}): min={stats['night_light_trend'].min():.3f}  "
        f"median={stats['night_light_trend'].median():.3f}  "
        f"max={stats['night_light_trend'].max():.3f}")


if __name__ == '__main__':
    main()
