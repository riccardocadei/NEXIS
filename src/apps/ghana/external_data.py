"""Ghana LEAP 1000 — loaders for external, community-GPS-keyed data sources.

Every external source is a candidate community-level covariate, keyed by
`comm` — one row per community. A covariate only needs to be *excluded* if
treatment could plausibly have caused it (post-treatment bias/collider risk,
e.g. household business/livestock ownership, which a cash transfer can
change). Exogenous sources like weather aren't at risk of this regardless of
whether they're measured before or during the study window, since treatment
cannot cause rainfall — so there's no separate "control-only" lane here.

    load_effect_modifiers(comm) -> community-level covariates (data.py::COMMUNITY_W)

Add one column-producing block per source to this function rather than
inventing new merge logic per source. Rainfall (CHIRPS, via
download_rainfall.py) is the first source; market access (travel time to
cities, via download_market_access.py) is the second; conflict/protest
events (ACLED, via download_acled.py) are the third; market prices (WFP,
via download_market_prices.py) are the fourth; nighttime lights (VIIRS, via
download_nightlights.py) are the fifth; population density (WorldPop, via
download_worldpop.py) is the sixth; settlement/urbanization degree (GHSL,
via download_ghsl.py) is the seventh; malaria mortality/incidence (Malaria
Atlas Project, via download_malaria.py) is the eighth; PM2.5 air pollution
(WashU ACAG SatPM2.5, via download_air_pollution.py) is the ninth.

Mobile-network coverage (OpenCellID) and mobile usage (Ookla Speedtest) were
both explored and rejected — see data/ghana/README.md's "Explored and
rejected" section. Both are present-day-only sources (OpenCellID's registry
is dominated by towers logged 2025-2026; Ookla's open archive starts 2019),
with no way to reconstruct 2015-2017 conditions, so neither passed the same
temporal-validity bar rainfall clears easily (climate is stable across a
decade; Ghana's mobile infrastructure/usage was not, over this exact
window). Don't re-add either without a source that actually covers
2015-2017 (e.g. a licensed GSMA historical coverage layer).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path('../data/ghana')


def load_effect_modifiers(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Community-level candidate covariates for NEXIS.

    Returns a DataFrame indexed by `comm` (one row per community). Merge new
    sources with `.merge(other_df, on='comm', how='outer')` and append their
    column name(s) to COMMUNITY_W in data.py.

    Rainfall contributes 7 columns, all raw mm/days (no z-scoring):
      - rainfall_mean_pre2015, rainfall_std_pre2015, drought_freq_pre2015:
        climatology (2000-2014, strictly pre-baseline).
      - rainfall_2015, rainfall_2016, rainfall_2017: the 3 realized annual
        totals during the study window itself, kept separate rather than
        averaged into one scalar (2016 correlates -0.66 with 2015 and -0.52
        with 2017, so a mean would mostly cancel out real year-to-year
        signal).
      - max_dry_days_2015_2017: max consecutive dry days (CDD) over the whole 2015-2017
        window, a drought-severity index a raw annual total can mask (two
        communities can have identical yearly rainfall with very different
        mid-season dry-spell exposure). Computed once over the full window
        rather than per calendar year: a calendar-year split is
        agronomically arbitrary and would understate a drought straddling
        a year boundary (e.g. Nov 2016-Jan 2017) by resetting at Dec 31.
      All of the above are legitimate effect modifiers despite overlapping
      the treatment period — the cash transfer cannot cause rainfall, so
      there's no post-treatment-bias risk, unlike a household covariate
      that could genuinely be changed by receiving the transfer.

    Market access contributes 1 column (see download_market_access.py):
      - travel_time_to_city_min: motorized travel time (minutes) to the
        nearest city of >50,000 population, Weiss et al. 2018 (Malaria
        Atlas Project), explicitly dated 2015 — an exact match to the LEAP
        baseline year, unlike OpenCellID/Ookla which were present-day-only
        snapshots (see the "Explored and rejected" note below). A cash
        transfer cannot build roads or move cities, so this is exogenous
        regardless of timing, same test as rainfall.

    ACLED contributes 3 columns (see download_acled.py), from the full
    Ghana event-level export for 2015-2017 (the study window itself —
    genuinely dated historical events, not a present-day snapshot, so no
    temporal-mismatch risk the way OpenCellID/Ookla had):
      - dist_nearest_conflict_km: distance to the nearest event of any
        type, always defined (uncapped by radius).
      - political_violence_25km: count of Battles/Violence-against-
        civilians within 25km — exposure most directly relevant to
        household welfare/safety.
      - demonstrations_25km: count of Riots/Protests within 25km — civil
        unrest, a qualitatively different exposure than political violence.
      A household-level cash transfer cannot cause a riot or a battle, so
      this is a valid community-level covariate despite overlapping the
      treatment period, same test as rainfall's study-window columns.

    Market prices contribute 2 columns to NEXIS's covariate pool (see
    download_market_prices.py), from WFP Ghana food prices (via HDX, public,
    no account needed):
      - dist_nearest_market_km: distance to the nearest market with a Maize
        price observation in 2015, always defined. Maize, not milk --
        Ghana's WFP monitoring doesn't track milk/dairy at all (no local
        fresh-milk market); maize is Northern Ghana's actual staple crop.
      - maize_price_2015: that nearest market's mean Maize price over 2015
        alone (pre/at-baseline, not later years -- deliberately sidesteps
        the "cash transfer causes local price inflation" general-
        equilibrium debate by using a pre-treatment price, same logic as
        rainfall's pre-2015 climatology). Level.REGIONAL, not COMMUNITY:
        only 2 distinct markets nationally report Maize in 2015 near the
        LEAP districts (median distance 75km) -- a real step function
        shared across a market's catchment area, not per-community
        variation. This is an interim, non-data-driven choice: an earlier
        version of this covariate pooled 2014+2015 to fix exactly this
        sparsity (9 markets, 13.7km median instead) -- see git history and
        download_market_prices.py's module docstring for the tradeoff.
      A household-level cash transfer cannot retroactively change a 2015
      market price, so this is exogenous by construction regardless.

      download_market_prices.py also writes 6 more yearly columns to the
      same CSV, deliberately NOT loaded here / not part of COVARIATES:
      maize_price_2010..maize_price_2014 (Timing.HISTORIC -- long-run price
      history predating baseline, same role as rainfall's 2000-2014
      climatology) and maize_price_2016/maize_price_2017 (Timing.POST --
      after endline, for deflating/actualizing expenditures downstream, not
      a NEXIS covariate: using a post-treatment price as a "pre-treatment"
      covariate would risk exactly the collider bias every other exclusion
      in this file avoids). Outside of 2014, WFP Maize coverage near the
      LEAP districts collapses to the same 2 distant markets every year, so
      these 6 diagnostics are largely near-duplicates of each other and of
      maize_price_2015, tracking the same 2 markets' price over time.

    Nighttime lights contribute 2 columns (see download_nightlights.py),
    from the VIIRS 2015 annual composite (Google Earth Engine, same
    authentication as rainfall/satellite -- explicitly dated 2015, an exact
    match to the LEAP baseline, unlike OpenCellID/Ookla):
      - night_light_radiance: mean radiance within 1km of the community
        centroid -- "is this specific community itself electrified/
        economically active". Sparse (~15/162 communities show any
        detectable light at this scale -- checked at multiple buffer radii,
        genuinely a fact about these deep-rural communities, not a radius
        artifact), registered as sparse_nonneg like SAE activations.
      - dist_nearest_light_km: distance to the nearest pixel with radiance
        above a standard "detectable urban light" threshold, always
        defined -- a remoteness/access proxy, weakly correlated with
        night_light_radiance itself (r=-0.25, answers a different question).
      A cash transfer cannot build a power grid or move a town, so both are
      exogenous regardless of timing.

    Population density contributes 1 column (see download_worldpop.py),
    from WorldPop's 2015 100m gridded population estimate (Google Earth
    Engine, same authentication as the sources above):
      - pop_density_2km: summed estimated population within 2km of the
        community centroid. Weakly correlated with community_size (r=0.04,
        confirming it captures true local population density rather than
        duplicating LEAP's own household sample count) and moderately with
        dist_to_capital_km/travel_time_to_city_min (r=-0.48, expected).
      A household-level cash transfer cannot move the local population, so
      this is exogenous regardless of timing.

    Settlement/urbanization degree contributes 1 column (see
    download_ghsl.py), from the GHSL Settlement Model's 2015 classification
    (Google Earth Engine, same authentication as the sources above):
      - urbanization_degree: categorical code (11-30), monotonically
        ordered by urbanization (very-low-density rural through urban
        centre). Moderately correlated with pop_density_2km (r=0.68,
        expected -- GHSL's own classification is partly built from
        population density) but not a pure duplicate, and weakly correlated
        with dist_to_capital_km/travel_time_to_city_min (r=-0.17).
      A household-level cash transfer cannot reclassify a settlement's
      urbanization degree, so this is exogenous regardless of timing.

    Malaria contributes 2 columns (see download_malaria.py), from the
    Malaria Atlas Project's Pf (Plasmodium falciparum) mortality/incidence
    time series -- a genuine annual series (2000-2022), so 2015 is
    selected directly via a WCS time subset, an exact match to the LEAP
    baseline year, same as market access:
      - malaria_mortality_rate_2015, malaria_incidence_rate_2015: raster values
        sampled at each community centroid. Negatively correlated with each
        other (r=-0.62) -- plausibly because higher-incidence areas
        sometimes have better malaria program targeting/case management,
        lowering mortality despite more cases -- so they capture genuinely
        different aspects of the local disease environment, not one
        redundant with the other. A directly relevant "children's care"
        exposure proxy: malaria is a leading cause of under-5 mortality in
        Ghana, and LEAP-1000 specifically targets pregnant women and
        children under 1.
      A household-level cash transfer cannot move a district's malaria
      burden, so this is exogenous regardless of timing.

    Air pollution contributes 1 column (see download_air_pollution.py),
    from Washington University's ACAG SatPM2.5 (V6.GL.03) -- satellite
    aerosol optical depth fused with GEOS-Chem simulation and ground
    monitors, public AWS Open Data, no account needed. Annual Africa
    composite, explicitly dated 2015 -- an exact match to the LEAP
    baseline year, same as market access/malaria:
      - pm25_2015: nearest-gridcell PM2.5 concentration (ug/m3), range
        32-37 across the 162 communities. Correlated with
        rainfall_mean_pre2015 (r=0.75) -- physically sensible, this Sahel
        region's air quality is Harmattan-dust-driven and rainfall tracks
        distance from the Sahara -- comparable in magnitude to the
        already-accepted rainfall/malaria correlation (r=0.87), not a
        reason to drop. Weakly correlated with accessibility/urbanization
        covariates (|r| <= 0.23), confirming this is dust-driven rather
        than urban-pollution-driven in this rural Sahel setting.
      A household-level cash transfer cannot move regional air quality, so
      this is exogenous regardless of timing.
    """
    data_dir = Path(data_dir)
    annual_columns = ['rainfall_2015', 'rainfall_2016', 'rainfall_2017']
    rainfall_columns = [
        'rainfall_mean_pre2015', 'rainfall_std_pre2015', 'drought_freq_pre2015',
        *annual_columns, 'max_dry_days_2015_2017',
    ]
    market_access_columns = ['travel_time_to_city_min']
    acled_columns = ['dist_nearest_conflict_km', 'political_violence_25km', 'demonstrations_25km']
    market_prices_columns = ['dist_nearest_market_km', 'maize_price_2015']
    nightlights_columns = ['night_light_radiance', 'dist_nearest_light_km']
    worldpop_columns = ['pop_density_2km']
    ghsl_columns = ['urbanization_degree']
    malaria_columns = ['malaria_mortality_rate_2015', 'malaria_incidence_rate_2015']
    air_pollution_columns = ['pm25_2015']

    climatology_path = data_dir / 'rainfall' / 'rainfall_climatology.csv'
    annual_path = data_dir / 'rainfall' / 'rainfall_annual.csv'
    cdd_path = data_dir / 'rainfall' / 'rainfall_cdd.csv'
    if climatology_path.exists() and annual_path.exists() and cdd_path.exists():
        climatology = pd.read_csv(climatology_path)[
            ['comm', 'rainfall_mean_pre2015', 'rainfall_std_pre2015', 'drought_freq_pre2015']
        ]
        annual_wide = (
            pd.read_csv(annual_path).pivot(index='comm', columns='year', values='rainfall_mm')
            .rename(columns={2015: 'rainfall_2015', 2016: 'rainfall_2016', 2017: 'rainfall_2017'})
            .reset_index()
        )
        cdd = pd.read_csv(cdd_path)[['comm', 'max_dry_days_2015_2017']]
        rainfall = climatology.merge(annual_wide, on='comm').merge(cdd, on='comm')
    else:
        # rainfall not yet downloaded (run download_rainfall.py) — callers
        # still see the expected columns, filled with NaN, not a KeyError.
        rainfall = pd.DataFrame(columns=['comm', *rainfall_columns])

    market_access_path = data_dir / 'market_access' / 'market_access_community.csv'
    if market_access_path.exists():
        market_access = pd.read_csv(market_access_path)[['comm', *market_access_columns]]
    else:
        # not yet processed (run download_market_access.py) — same NaN-fill
        # convention as rainfall above.
        market_access = pd.DataFrame(columns=['comm', *market_access_columns])

    acled_path = data_dir / 'conflicts' / 'acled_community.csv'
    if acled_path.exists():
        acled = pd.read_csv(acled_path)[['comm', *acled_columns]]
    else:
        # not yet processed (run download_acled.py) — same NaN-fill
        # convention as rainfall/market access above.
        acled = pd.DataFrame(columns=['comm', *acled_columns])

    market_prices_path = data_dir / 'market_prices' / 'market_prices_community.csv'
    if market_prices_path.exists():
        market_prices = pd.read_csv(market_prices_path)[['comm', *market_prices_columns]]
    else:
        # not yet processed (run download_market_prices.py) — same NaN-fill
        # convention as the sources above.
        market_prices = pd.DataFrame(columns=['comm', *market_prices_columns])

    nightlights_path = data_dir / 'nightlights' / 'nightlights_community.csv'
    if nightlights_path.exists():
        nightlights = pd.read_csv(nightlights_path)[['comm', *nightlights_columns]]
    else:
        # not yet processed (run download_nightlights.py) — same NaN-fill
        # convention as the sources above.
        nightlights = pd.DataFrame(columns=['comm', *nightlights_columns])

    worldpop_path = data_dir / 'worldpop' / 'worldpop_community.csv'
    if worldpop_path.exists():
        worldpop = pd.read_csv(worldpop_path)[['comm', *worldpop_columns]]
    else:
        # not yet processed (run download_worldpop.py) — same NaN-fill
        # convention as the sources above.
        worldpop = pd.DataFrame(columns=['comm', *worldpop_columns])

    ghsl_path = data_dir / 'ghsl' / 'ghsl_community.csv'
    if ghsl_path.exists():
        ghsl = pd.read_csv(ghsl_path)[['comm', *ghsl_columns]]
    else:
        # not yet processed (run download_ghsl.py) — same NaN-fill
        # convention as the sources above.
        ghsl = pd.DataFrame(columns=['comm', *ghsl_columns])

    malaria_path = data_dir / 'malaria' / 'malaria_community.csv'
    if malaria_path.exists():
        malaria = pd.read_csv(malaria_path)[['comm', *malaria_columns]]
    else:
        # not yet processed (run download_malaria.py) — same NaN-fill
        # convention as the sources above.
        malaria = pd.DataFrame(columns=['comm', *malaria_columns])

    air_pollution_path = data_dir / 'air_pollution' / 'air_pollution_community.csv'
    if air_pollution_path.exists():
        air_pollution = pd.read_csv(air_pollution_path)[['comm', *air_pollution_columns]]
    else:
        # not yet processed (run download_air_pollution.py) — same NaN-fill
        # convention as the sources above.
        air_pollution = pd.DataFrame(columns=['comm', *air_pollution_columns])

    merged = (
        rainfall.merge(market_access, on='comm', how='outer')
                .merge(acled, on='comm', how='outer')
                .merge(market_prices, on='comm', how='outer')
                .merge(nightlights, on='comm', how='outer')
                .merge(worldpop, on='comm', how='outer')
                .merge(ghsl, on='comm', how='outer')
                .merge(malaria, on='comm', how='outer')
                .merge(air_pollution, on='comm', how='outer')
                .astype({'comm': 'int64'})
    )
    return merged[['comm', *rainfall_columns, *market_access_columns, *acled_columns,
                    *market_prices_columns, *nightlights_columns,
                    *worldpop_columns, *ghsl_columns, *malaria_columns,
                    *air_pollution_columns]]
