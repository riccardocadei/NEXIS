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

Extracts one price per YEAR, 2010-2017, each independently (no pooling
across years) -- nearest-market assignment is recomputed per year, since not
every market reports Maize every year. All are Level.REGIONAL: WFP coverage
near the LEAP districts is sparse enough that a "nearest priced market"
value is really shared by a whole catchment of communities, not a
per-community measurement.

IMPORTANT caveat, checked before extracting: outside of 2014, WFP Maize
coverage near the LEAP districts collapses to just 2 distinct markets,
median distance ~75km (max ~116km) -- 2014 is the one anomalous year with
rich nearby coverage (9 markets, median 13.7km), which is why the ORIGINAL
version of this script pooled 2014+2015 into a single "maize_price_2015"
covariate (see git history). That pooling was deliberately dropped in favor
of a clean per-year series -- maize_price_2015 is now 2015 ALONE, sharing
the same degenerate 2-market/75km-median coverage as every other non-2014
year. This is an interim choice, not a data-driven one: it's simpler and
consistent with the rest of the per-year series, at the cost of reverting a
previously-fixed sparsity problem in the one covariate that still feeds
NEXIS. Revisit if maize_price_2015 turns out to matter for the results.

Of the 8 yearly columns, only maize_price_2015 (paired with
dist_nearest_market_km, both Timing.PRE) is registered as a NEXIS covariate
in data.py's COVARIATES. The other 7 are diagnostics only, not fed into W:
    - maize_price_2010..maize_price_2014 (Timing.HISTORIC): long-run price
      history predating baseline, analogous to rainfall's 2000-2014
      climatology (rainfall_mean_pre2015/rainfall_std_pre2015/
      drought_freq_pre2015, also Timing.HISTORIC -- see covariates.py).
    - maize_price_2016, maize_price_2017 (Timing.POST): after endline,
      potentially useful for deflating/actualizing expenditures downstream,
      but a post-treatment value is not something NEXIS should search over
      (collider-bias risk, same reasoning as every other Timing.POST
      source).

dist_nearest_market_km is paired only with maize_price_2015 (the one W
covariate) -- the other 7 years don't get their own distance column, same
reasoning as before: they're diagnostics, not paired effect-modifier
candidates.

Produces one file, community-level:
    market_prices/market_prices_community.csv
        comm, dist_nearest_market_km, maize_price_2010, maize_price_2011,
        maize_price_2012, maize_price_2013, maize_price_2014,
        maize_price_2015, maize_price_2016, maize_price_2017

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
YEARS = list(range(2010, 2018))   # 2010-2017 inclusive
W_YEAR = 2015                     # the only year fed into NEXIS's COVARIATES


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


def load_maize_markets(prices: pd.DataFrame, markets: pd.DataFrame, year: int) -> pd.DataFrame:
    maize = prices[
        (prices['commodity'] == COMMODITY)
        & (prices['date'] >= f'{year}-01-01') & (prices['date'] <= f'{year}-12-31')
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
    """One row per community: that year's nearest-market mean {COMMODITY}
    price (assigned, not interpolated), plus the distance to it if dist_col
    is given (each year can have a different nearest market, since not
    every market reports every year)."""
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
        log(f"  Years: {YEARS[0]}-{YEARS[-1]} (each extracted independently)")
        log(f"  W covariate: maize_price_{W_YEAR} (+ dist_nearest_market_km)")
        log(f"  Output: {out_dir / 'market_prices_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log(f"Downloading WFP Ghana market/price data ...")
    prices = pd.read_csv(PRICES_URL, skiprows=[1])
    markets = pd.read_csv(MARKETS_URL)
    prices['date'] = pd.to_datetime(prices['date'])

    stats = centroids[['comm']].copy()
    for year in YEARS:
        year_markets = load_maize_markets(prices, markets, year)
        log(f"  {len(year_markets)} markets with {COMMODITY} price data in {year}")
        price_col = f'maize_price_{year}'
        dist_col = 'dist_nearest_market_km' if year == W_YEAR else None
        year_stats = nearest_market_price(centroids, year_markets, price_col, dist_col=dist_col)
        stats = stats.merge(year_stats, on='comm')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'market_prices_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  dist_nearest_market_km: min={stats['dist_nearest_market_km'].min():.1f}  "
        f"median={stats['dist_nearest_market_km'].median():.1f}  "
        f"max={stats['dist_nearest_market_km'].max():.1f}")
    for year in YEARS:
        col = f'maize_price_{year}'
        tag = ' (W covariate)' if year == W_YEAR else ''
        log(f"  {col}{tag}: min={stats[col].min():.1f}  median={stats[col].median():.1f}  "
            f"max={stats[col].max():.1f} GHS")


if __name__ == '__main__':
    main()
