"""Download WFP food-price data for Ghana LEAP 1000 community centroids, into
community-level covariates for NEXIS (see external_data.py::load_effect_modifiers).

Source: WFP Food Prices for Ghana, via the Humanitarian Data Exchange (HDX)
-- fully public, no account/API key (unlike OpenCellID/ACLED). Two files:
market list (91 markets, with lat/lon) and monthly price observations by
market/commodity since 2006.

Extracts a STAPLE PRICE INDEX per YEAR, 2010-2017, each year independently
(no pooling across years) -- an equal-weighted average of z-scored
nearest-market prices across 7 staples: Maize, Rice (imported), Rice
(local), Millet, Sorghum, Yam, Plantains. This replaced an earlier
single-commodity (Maize-only) version entirely -- see git history --
because a single crop's price answers "what does maize cost here", not the
actual construct of interest: purchasing power / cost-of-living, i.e. how
far a fixed-GHS LEAP transfer goes locally. Z-scoring first is necessary
because raw price levels aren't comparable across commodities (a GHS/kg of
rice isn't the same unit as a GHS/kg of yam); the index is each community's
average standardized price across the basket, so higher = locally more
expensive staples overall, not any one commodity's literal price.

Checked before adopting this basket, for every year 2010-2017: all 7
commodities have real market coverage in every year (2014 is unusually
richer for Maize/Rice specifically, same anomaly documented below, but
none of the 7 is ever absent). Two commodities were checked and excluded:
Cassava resolves to a different, farther nearest-market catchment (median
129km vs. 75km for every other staple in 2015) -- mixing it in would
silently swap the geographic assignment underlying part of the index.
Rice (paddy) was dropped as redundant with Rice (imported)/Rice (local)
(the consumer-facing forms actually purchased, vs. the unprocessed
wholesale form).

All 8 yearly index columns are Level.REGIONAL: WFP coverage near the LEAP
districts is sparse enough that a "nearest priced market" value is really
shared by a whole catchment of communities, not a per-community
measurement.

IMPORTANT caveat, checked before extracting: outside of 2014, WFP coverage
near the LEAP districts collapses to just 2 distinct markets for this
basket, median distance ~75km (max ~116km) -- 2014 is the one anomalous
year with rich nearby coverage (9 markets, median 13.7km for Maize
specifically). This is a real, structural sparsity in the underlying WFP
data near these districts, not a basket-composition problem -- averaging
across 7 commodities reduces commodity-specific idiosyncratic noise (a bad
maize harvest alone doesn't sink the whole index) but does NOT fix it,
since every commodity in the basket shares the same market geography.
staple_price_index_2015 (the one W covariate here) is built on this same
degenerate 2-market/75km-median coverage. This is an interim situation, not
a data-driven ideal -- revisit if this covariate turns out to matter for
the results.

Of the 8 yearly columns, only staple_price_index_2015 (paired with
dist_nearest_market_km, both Timing.PRE) is registered as a NEXIS covariate
in data.py's COVARIATES. The other 7 are diagnostics only, not fed into W:
    - staple_price_index_2010..staple_price_index_2014 (Timing.HISTORIC):
      long-run price history predating baseline, analogous to rainfall's
      2000-2014 climatology (rainfall_mean_pre2015/rainfall_std_pre2015/
      drought_freq_pre2015, also Timing.HISTORIC -- see covariates.py).
    - staple_price_index_2016, staple_price_index_2017 (Timing.POST): after
      endline, potentially useful for deflating/actualizing expenditures
      downstream, but a post-treatment value is not something NEXIS should
      search over (collider-bias risk, same reasoning as every other
      Timing.POST source).

dist_nearest_market_km is paired only with staple_price_index_2015 (the one
W covariate) -- the other 7 years don't get their own distance column,
same reasoning as before: they're diagnostics, not paired effect-modifier
candidates. It's computed from Maize specifically (the basket's anchor
commodity, per the module's original design) since every commodity in the
basket shares the same nearest-market geography in practice.

Produces one file, community-level:
    market_prices/market_prices_community.csv
        comm, dist_nearest_market_km, staple_price_index_2010,
        staple_price_index_2011, staple_price_index_2012,
        staple_price_index_2013, staple_price_index_2014,
        staple_price_index_2015, staple_price_index_2016,
        staple_price_index_2017

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
ANCHOR_COMMODITY = 'Maize'        # used only for dist_nearest_market_km
YEARS = list(range(2010, 2018))   # 2010-2017 inclusive
W_YEAR = 2015                     # the only year fed into NEXIS's COVARIATES

# Basket for staple_price_index_* -- see module docstring for why Cassava
# and Rice (paddy) are excluded.
STAPLE_COMMODITIES = [
    'Maize', 'Rice (imported)', 'Rice (local)', 'Millet', 'Sorghum', 'Yam', 'Plantains (apentu)',
]


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


def load_commodity_markets(prices: pd.DataFrame, markets: pd.DataFrame,
                           commodity: str, year: int) -> pd.DataFrame:
    sub = prices[
        (prices['commodity'] == commodity)
        & (prices['date'] >= f'{year}-01-01') & (prices['date'] <= f'{year}-12-31')
    ]
    by_market = sub.groupby('market_id').agg(
        price=('price', 'mean'), n_obs=('price', 'size'),
    ).reset_index()
    return by_market.merge(markets, on='market_id')


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def nearest_market_price(centroids: pd.DataFrame, commodity_markets: pd.DataFrame,
                         price_col: str, dist_col: str = None) -> pd.DataFrame:
    """One row per community: that year's nearest-market mean price for one
    commodity (assigned, not interpolated), plus the distance to it if
    dist_col is given (each year can have a different nearest market, since
    not every market reports every commodity every year)."""
    rows = []
    for _, comm in centroids.iterrows():
        dist_km = _haversine_km(comm['lat'], comm['lon'],
                                 commodity_markets['latitude'], commodity_markets['longitude'])
        idx = dist_km.values.argmin()
        row = {'comm': comm['comm'], price_col: commodity_markets['price'].iloc[idx]}
        if dist_col:
            row[dist_col] = dist_km.values[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def staple_price_index(centroids: pd.DataFrame, prices: pd.DataFrame, markets: pd.DataFrame,
                       year: int) -> pd.DataFrame:
    """Equal-weighted average of z-scored nearest-market prices across
    STAPLE_COMMODITIES for one year -- see module docstring."""
    zscored = []
    for commodity in STAPLE_COMMODITIES:
        commodity_markets = load_commodity_markets(prices, markets, commodity, year)
        log(f"    {len(commodity_markets)} markets with {commodity} price data in {year}")
        col = f'_tmp_{commodity}'
        commodity_stats = nearest_market_price(centroids, commodity_markets, col)
        z = (commodity_stats[col] - commodity_stats[col].mean()) / commodity_stats[col].std()
        zscored.append(pd.DataFrame({'comm': commodity_stats['comm'], col: z}))
    index_df = zscored[0]
    for z in zscored[1:]:
        index_df = index_df.merge(z, on='comm')
    index_cols = [c for c in index_df.columns if c != 'comm']
    price_col = f'staple_price_index_{year}'
    index_df[price_col] = index_df[index_cols].mean(axis=1)
    return index_df[['comm', price_col]]


def main():
    args = parse_args()
    script_dir = Path(__file__).parent
    data_path  = (script_dir / args.data_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    if args.dry_run:
        log(f"[DRY RUN]")
        log(f"  Staple basket: {', '.join(STAPLE_COMMODITIES)}")
        log(f"  Years: {YEARS[0]}-{YEARS[-1]} (each extracted independently)")
        log(f"  W covariates: staple_price_index_{W_YEAR} + dist_nearest_market_km")
        log(f"  Output: {out_dir / 'market_prices_community.csv'}")
        return

    centroids = load_community_centroids(str(data_path))
    log(f"Downloading WFP Ghana market/price data ...")
    prices = pd.read_csv(PRICES_URL, skiprows=[1])
    markets = pd.read_csv(MARKETS_URL)
    prices['date'] = pd.to_datetime(prices['date'])

    anchor_markets = load_commodity_markets(prices, markets, ANCHOR_COMMODITY, W_YEAR)
    log(f"  {len(anchor_markets)} markets with {ANCHOR_COMMODITY} price data in {W_YEAR} (for dist_nearest_market_km)")
    dist_stats = nearest_market_price(centroids, anchor_markets, f'_tmp_anchor_{W_YEAR}',
                                      dist_col='dist_nearest_market_km')
    stats = dist_stats[['comm', 'dist_nearest_market_km']]

    for year in YEARS:
        log(f"  Building staple_price_index_{year} from {len(STAPLE_COMMODITIES)} commodities ...")
        year_index = staple_price_index(centroids, prices, markets, year)
        stats = stats.merge(year_index, on='comm')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'market_prices_community.csv'
    stats.to_csv(out_path, index=False)
    log(f"Wrote {out_path}  ({len(stats)} communities)")
    log(f"  dist_nearest_market_km: min={stats['dist_nearest_market_km'].min():.1f}  "
        f"median={stats['dist_nearest_market_km'].median():.1f}  "
        f"max={stats['dist_nearest_market_km'].max():.1f}")
    for year in YEARS:
        col = f'staple_price_index_{year}'
        tag = ' (W covariate)' if year == W_YEAR else ''
        log(f"  {col}{tag}: min={stats[col].min():.2f}  median={stats[col].median():.2f}  "
            f"max={stats[col].max():.2f}")


if __name__ == '__main__':
    main()
