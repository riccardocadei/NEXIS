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
logic as rainfall's pre-2015 climatology. Named maize_price_2015 (not
maize_price_2014_2015) since 2014 is only pooled in to get enough price
observations per market -- the intent is "the price as of baseline", same
naming convention as every other 2015-dated covariate in this project.

2014 isn't optional pooling for stability -- it's load-bearing. Restricted
to 2015 alone, 7 of the 9 markets currently assigned as someone's nearest
have ZERO Maize observations that year (their only observation is from
2014); median distance to the nearest priced market jumps from 13.7km to
75.3km and only 2 distinct markets cover all 162 communities (vs. 9).

Also produces maize_price_2017 (2016-2017 pooled, same per-market-
observation-count logic), NOT fed into NEXIS's COVARIATES registry --
it's post-endline, for deflating/actualizing expenditures downstream, not
a covariate NEXIS should search over (post-treatment collider risk, same
reasoning as every other Timing.POST source -- see covariates.py). Market
coverage near the LEAP districts largely collapsed after 2015: only 3
distinct nearest-markets (vs. 9 pre-period), median distance 62.1km (vs.
13.7km) -- pooling 2016 in doesn't recover any closer markets, it's the
identical 3 markets/distances as 2017 alone, just more observations to
average per market.

Produces one file, community-level:
    market_prices/market_prices_community.csv
        comm, dist_nearest_market_km, maize_price_2015, maize_price_2017

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
POST_PERIOD_START = '2016-01-01'
POST_PERIOD_END = '2017-12-31'


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


def load_maize_markets(prices: pd.DataFrame, markets: pd.DataFrame,
                       start: str, end: str) -> pd.DataFrame:
    maize = prices[
        (prices['commodity'] == COMMODITY)
        & (prices['date'] >= start) & (prices['date'] <= end)
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


def nearest_market_price(centroids: pd.DataFrame, maize_markets: pd.DataFrame,
                         price_col: str, dist_col: str = None) -> pd.DataFrame:
    """One row per community: that period's nearest-market mean {COMMODITY}
    price (assigned, not interpolated), plus the distance to it if dist_col
    is given (each period can have a different nearest market, since not
    every market reports in every period)."""
    rows = []
    for _, comm in centroids.iterrows():
        dist_km = _haversine_km(comm['lat'], comm['lon'],
                                 maize_markets['latitude'], maize_markets['longitude'])
        idx = dist_km.values.argmin()
        row = {'comm': comm['comm'], price_col: maize_markets['price'].iloc[idx]}
        if dist_col:
            row[dist_col] = dist_km.values[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Commodity: {COMMODITY}")
        log(f"  Pre-treatment period (maize_price_2015): {PERIOD_START} to {PERIOD_END}")
        log(f"  Post-treatment period (maize_price_2017): {POST_PERIOD_START} to {POST_PERIOD_END}")
        log(f"  Output: {out_dir / 'market_prices_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log(f"Downloading WFP Ghana market/price data ...")
    prices = pd.read_csv(PRICES_URL, skiprows=[1])
    markets = pd.read_csv(MARKETS_URL)
    prices['date'] = pd.to_datetime(prices['date'])

    pre_markets = load_maize_markets(prices, markets, PERIOD_START, PERIOD_END)
    log(f"  {len(pre_markets)} markets with {COMMODITY} price data in {PERIOD_START[:4]}-{PERIOD_END[:4]}")
    post_markets = load_maize_markets(prices, markets, POST_PERIOD_START, POST_PERIOD_END)
    log(f"  {len(post_markets)} markets with {COMMODITY} price data in {POST_PERIOD_START[:4]}-{POST_PERIOD_END[:4]}")

    pre = nearest_market_price(centroids, pre_markets, 'maize_price_2015', dist_col='dist_nearest_market_km')
    post = nearest_market_price(centroids, post_markets, 'maize_price_2017')
    stats = pre.merge(post, on='comm')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'market_prices_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  dist_nearest_market_km: min={stats['dist_nearest_market_km'].min():.1f}  "
        f"median={stats['dist_nearest_market_km'].median():.1f}  "
        f"max={stats['dist_nearest_market_km'].max():.1f}")
    log(f"  maize_price_2015: min={stats['maize_price_2015'].min():.1f}  "
        f"median={stats['maize_price_2015'].median():.1f}  "
        f"max={stats['maize_price_2015'].max():.1f} GHS")
    log(f"  maize_price_2017 (post-endline, NOT a NEXIS covariate -- for expenditure "
        f"deflation only): min={stats['maize_price_2017'].min():.1f}  "
        f"median={stats['maize_price_2017'].median():.1f}  "
        f"max={stats['maize_price_2017'].max():.1f} GHS")


if __name__ == '__main__':
    main()
