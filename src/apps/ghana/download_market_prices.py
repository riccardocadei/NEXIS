"""Download WFP food-price data for Ghana LEAP 1000 community centroids, into
community-level covariates for NEXIS (see external_data.py::load_effect_modifiers).

Source: WFP Food Prices for Ghana, via the Humanitarian Data Exchange (HDX)
-- fully public, no account/API key (unlike OpenCellID/ACLED). Two files:
market list (91 markets, with lat/lon) and monthly price observations by
market/commodity since 2006.

Commodity: Maize, not milk -- Ghana's WFP price monitoring doesn't track
milk/dairy at all (no local fresh-milk market; Ghana is import-dependent for
dairy). Maize is Northern Ghana's actual staple crop and has the best market
coverage near the 5 LEAP districts: Nalerigu (East Mamprusi), Yendi, Garu
(Garu-Tempane), Bongo, and several others within reach (Gushegu, Bunkprugu,
Bolga, Tamale, Zabzugu).

Period: 2014-2015 (pre/at-baseline), not later years -- deliberately
avoiding the classic "cash transfer causes local price inflation" general-
equilibrium debate. Using pre-treatment prices sidesteps it entirely, same
logic as rainfall's pre-2015 climatology.

Produces one file, community-level, feeding NEXIS covariates:
    market_prices/market_prices_community.csv
        comm, dist_nearest_market_km, maize_price_2014_2015

Usage:
    python download_market_prices.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PRICES_URL = ('https://data.humdata.org/dataset/626e809c-c4fc-467b-a60c-129acb5e9320/'
              'resource/e877350b-146f-4fa7-8690-db9605eea78c/download/wfp_food_prices_gha.csv')
MARKETS_URL = ('https://data.humdata.org/dataset/626e809c-c4fc-467b-a60c-129acb5e9320/'
               'resource/e674838c-87ba-4c7c-afc5-614226842768/download/wfp_markets_gha.csv')
COMMODITY = 'Maize'
PERIOD_START = '2014-01-01'
PERIOD_END = '2015-12-31'


def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default='../../../data/ghana/market_prices',
                   help='Local directory to save the community-level CSV '
                        '(default: ../../../data/ghana/market_prices)')
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


def load_maize_markets() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_URL, skiprows=[1])
    markets = pd.read_csv(MARKETS_URL)
    prices['date'] = pd.to_datetime(prices['date'])

    maize = prices[
        (prices['commodity'] == COMMODITY)
        & (prices['date'] >= PERIOD_START) & (prices['date'] <= PERIOD_END)
    ]
    by_market = maize.groupby('market_id').agg(
        price=('price', 'mean'), n_obs=('price', 'size'),
    ).reset_index()
    return by_market.merge(markets, on='market_id')


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def community_market_stats(centroids: pd.DataFrame, maize_markets: pd.DataFrame) -> pd.DataFrame:
    """One row per community: distance to the nearest market with a
    {COMMODITY} price in {PERIOD_START}-{PERIOD_END}, and that market's
    mean price over the period (assigned, not interpolated)."""
    rows = []
    for _, comm in centroids.iterrows():
        dist_km = _haversine_km(comm['lat'], comm['lon'],
                                 maize_markets['latitude'], maize_markets['longitude'])
        idx = dist_km.values.argmin()
        rows.append({
            'comm': comm['comm'],
            'dist_nearest_market_km': dist_km.values[idx],
            'maize_price_2014_2015': maize_markets['price'].iloc[idx],
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Commodity: {COMMODITY}  |  Period: {PERIOD_START} to {PERIOD_END}")
        log(f"  Output: {out_dir / 'market_prices_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log(f"Downloading WFP Ghana market/price data ...")
    maize_markets = load_maize_markets()
    log(f"  {len(maize_markets)} markets with {COMMODITY} price data in {PERIOD_START[:4]}-{PERIOD_END[:4]}")

    stats = community_market_stats(centroids, maize_markets)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'market_prices_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  dist_nearest_market_km: min={stats['dist_nearest_market_km'].min():.1f}  "
        f"median={stats['dist_nearest_market_km'].median():.1f}  "
        f"max={stats['dist_nearest_market_km'].max():.1f}")
    log(f"  maize_price_2014_2015: min={stats['maize_price_2014_2015'].min():.1f}  "
        f"median={stats['maize_price_2014_2015'].median():.1f}  "
        f"max={stats['maize_price_2014_2015'].max():.1f} GHS")


if __name__ == '__main__':
    main()
