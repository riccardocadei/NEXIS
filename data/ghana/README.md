# Ghana LEAP-1000 — data sources

Tracks every dataset used by `src/apps/ghana/`, where it comes from, and its status. Keep this updated whenever a new source is added (see `src/apps/ghana/external_data.py` for the code-side convention).

## Household survey (core, restricted)

| | |
|---|---|
| **File** | `LEAP1000 2015-2017 household data++.dta` |
| **Origin** | UNICEF Ghana / ISSER (Univ. of Ghana) / Carolina Population Center (UNC-Chapel Hill) / Navrongo Health Research Centre — LEAP 1000 impact evaluation. **Not publicly available**; obtained directly from UNICEF Ghana. |
| **Coverage** | 5 districts, Northern Ghana: East Mamprusi, Karaga, Yendi (Northern region), Bongo, Garu-Tempane (Upper East region). |
| **Years** | 2 waves — Baseline 2015 (n=2,497), Endline 2017 (n=2,331). |
| **Size** | 4,828 rows × 30 columns, 83 KB. 162 communities. |
| **Description** | Household panel: PMT eligibility score, treatment status, adult-equivalent expenditure, demographics, housing, livelihoods. See `variable_description.csv` for the full column dictionary. |
| **Known caveat** | `pmtscore` is the eligibility/targeting score (RDD running variable) — do not use as an outcome or NEXIS effect modifier (near-perfect separation between arms by construction). |
| **Reference reports** | `LEAP-1000_Baseline-Survey_2015.pdf`, `Ghana-LEAP-1000-Endline-Household-Survey-v8.pdf` — full UNICEF/ISSER/UNC endline evaluation report (June 2018), used for context (e.g. Table 4.2.7 "Shocks in the community" motivated the rainfall covariate below). |

## Satellite imagery (Landsat 8, via Google Earth Engine)

| | |
|---|---|
| **Origin** | `LANDSAT/LC08/C02/T1_L2` (USGS Collection 2, Surface Reflectance), pulled via `earthengine-api`. |
| **Produced by** | `src/apps/ghana/download_satellite_images.py` (LEAP community tiles), `download_waterway_2017.py` (2017 pilot), `download_national_grid.py` (national training grid). |

| Subfolder | Years | Tiles | Size | Status |
|---|---|---|---|---|
| `satellite/tif/` | 2015 | 162 (all LEAP communities) | 120 MB | ✅ complete |
| `satellite/tif_2017/` | 2017 | 6 (neuron-3821 pilot only — `download_waterway_2017.py`'s hardcoded `COMMUNITIES` dict) | 4.7 MB | ⚠️ partial — full 162-community 2017 pass not yet run |
| `satellite/tif_national/` | 2015 | 9,592 (national grid, SAE pretraining corpus) | 6.7 GB | ✅ complete |
| `satellite/previews/` | 2015 | PNG previews per community | 203 MB | ✅ complete |

Each tile: 5×5 km, 30 m/px, 6 bands (B2–B7, i.e. blue/green/red/NIR/SWIR-1/SWIR-2), cloud-filtered annual median composite. Filename convention: `ghana_comm{id:04d}.tif`.

## Derived embeddings / SAE artifacts

| | |
|---|---|
| **Origin** | Computed from the satellite tiles above by `src/apps/ghana/extract_satellite_features.py` (Prithvi-EO-1.0-100M, HLS-pretrained ViT-MAE) and `train_sae.py` (TopK Sparse Autoencoder, Gao et al. 2024). |

| File | Description | Size |
|---|---|---|
| `satellite/prithvi_embeddings.npy` + `prithvi_comm_ids.npy` | 768-d Prithvi embedding per LEAP community (2015), 162 rows | 476 KB |
| `satellite/national/` | Same, for the 9,592-tile national grid (SAE training corpus) | 28 MB |
| `satellite/spectral_indices.csv` | 6 hand-crafted spectral indices per LEAP-community tile | 28 KB |
| `satellite/sae_model.pt` | Trained TopK SAE weights (trained on national grid, evaluated on LEAP communities) | 23 MB |
| `satellite/sae_whiten_mean.npy` / `sae_whiten_std.npy` | Whitening stats used before SAE encoding | — |
| `satellite/sae_activations.npy` + `sae_comm_ids.npy` | SAE neuron activations per LEAP community — this is the `Z` block NEXIS searches | 74 KB |
| `satellite/sae_cv_results.csv`, `sae_reconstruction_error.csv` | SAE training diagnostics | — |
| `satellite/vit_embeddings.npy` / `vit_comm_ids.npy` | **Legacy/unused** — no current script produces or reads these (predates the Prithvi-only refactor); safe to ignore or remove | 238 KB |

## Administrative boundaries & basemaps (plotting only)

| File | Origin | Description |
|---|---|---|
| `gadm41_GHA_1.json`, `gadm41_GHA_2.json` | [GADM](https://gadm.org) v4.1 | Ghana region (level 1) / district (level 2) boundaries — used for map figures. |
| `ne_10m_admin_0_countries.*` | [Natural Earth](https://www.naturalearthdata.com), 1:10m cultural | Country boundaries — basemap context. |
| `ne_10m_lakes.*` | Natural Earth, 1:10m physical | Lakes (e.g. Lake Volta) — basemap context. |

No survey or outcome data — purely cartographic, not versioned/updated independently.

## Rainfall / drought exposure (CHIRPS, via Google Earth Engine)

| | |
|---|---|
| **Origin** | `UCSB-CHG/CHIRPS/DAILY` (Climate Hazards Center InfraRed Precipitation with Station data), via `earthengine-api`. |
| **Produced by** | `src/apps/ghana/download_rainfall.py` |
| **Motivation** | LEAP 1000 endline evaluation report, Table 4.2.7: drought (74% of communities, 2015–2017) and floods (57%) are the dominant self-reported community shocks in this exact sample. |
| **Status** | ✅ **Downloaded** — 162/162 communities, all 18 years (2000–2017). |

Split into two files, deliberately kept in separate roles (see `notebooks/ghana.ipynb`, "Effect modifiers vs. DiD controls", and `external_data.py`):

| File | Years | Columns (per community) | Role |
|---|---|---|---|
| `rainfall/rainfall_climatology.csv` | 2000–2014 (pre-baseline only) | 1 row × 3 cols: `rainfall_mean_pre2015`, `rainfall_std_pre2015`, `drought_freq_pre2015` | Stable community traits → all 3 are NEXIS `Z` effect-modifier candidates (`COMMUNITY_Z` in `data.py`) |
| `rainfall/rainfall_annual.csv` | 2015–2017 (study window) | 3 rows × 2 cols: `rainfall_mm`, `rainfall_anomaly` | Realized annual rainfall + anomaly vs. the same climatology → `analysis.py::regression_did(controls=[...])` robustness check, **not** a NEXIS moderator |

Example (community 14, Garu-Tempane): mean 945mm/yr, std 111mm, drought in 2/15 pre-2015 years (13.3%); realized 2015/2016/2017 anomalies of +0.62σ / −0.52σ / −0.77σ against that same baseline.

## Planned / candidate future sources

Not yet started — tracked here so scope stays visible. Add one at a time; classify each as an effect-modifier (`external_data.py::load_effect_modifiers`, time-invariant trait) or a DiD control (`load_did_controls`, time-varying study-window event) before wiring in, same as rainfall above.

- **Market access** (travel-time-to-market rasters, e.g. Malaria Atlas Project / JRC accessibility layers) — time-invariant trait → effect-modifier lane.
- **EM-DAT / ACLED** disaster & conflict event records — time-varying → DiD-control lane.
- **OpenCellID** mobile network coverage/density (suggested by UNICEF colleagues) — likely time-invariant infrastructure trait → effect-modifier lane, needs scoping.
- **Original community questionnaire microdata** (if UNICEF can share it) — the *ground-truth* version of the rainfall/shocks proxy above (Table 4.2.7 was computed from this); should take priority over the modeled CHIRPS proxy if it becomes available.
- Any additional UNICEF data drop — extend `load_data()` / `external_data.py` following the same pattern, not a rewrite.
