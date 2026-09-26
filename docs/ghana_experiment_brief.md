# Ghana LEAP 1000 experiment brief

> **Purpose.** The Ghana case study as it stands in the final paper
> (`paper/ICLR'27/main.tex`, Section "Case study 2", and `appendix.tex`, Appendix E):
> design, the choices behind it, every number the paper quotes, and the file or script
> each number comes from. Paths are relative to the repo root; `results/` and `data/` are
> untracked (`data/ghana/survey/` is restricted: do not redistribute). Top-level
> commands: `README.md`, section Ghana; run notes in Section 11.
> Updated 2026-09-26 from the June (NeurIPS) version, which used the interleaved-backward
> algorithm, a pool of 155 (no spectral indices), stale p-values (2.1×10⁻⁸, 3.7×10⁻⁷, from
> a 3-neuron run), an "RCT"/"ATE = ITT = ATT" framing, and a "3 discoveries" line that
> contradicted its own 2-row table. The unverifiable citation of that version is kept
> flagged in Section 8.

---

## 1. Programme and design

**Programme.** LEAP 1000 (Livelihood Empowerment Against Poverty), Ghana's Department of
Social Welfare: bimonthly cash transfers to extremely poor households with children aged
0–1. Evaluation by UNICEF Innocenti and ISSER (University of Ghana), with the Carolina
Population Center and the Navrongo Health Research Centre.

**Design: regression discontinuity, not an RCT.** Eligibility by a proxy means test (PMT)
score: T = 1 if the PMT score is at or below the cutoff (poorer households are treated).
In the analytic sample every treated household has `pmtscore` ≤ 7.158 and every
comparison household > 7.158 (T from `tac == "Treatment"`, `src/apps/ghana/data.py`;
checked 2026-09-26, 0 of 2,331 households misclassified). 154 of 162 communities hold both
arms, median within-community treated share 0.50, so the community is not the assignment
unit.

**Sample** (balanced panel, baseline 2015, endline 2017): 2,331 households, 162
communities with GPS centroids, 5 districts (East Mamprusi, Karaga, Yendi, Bongo,
Garu-Tempane), Northern and Upper East regions; 1,185 treated (50.8 %), 1,146 comparison.
Baseline adult-equivalent expenditure 120.9 GH₵/month (treated 117.8, comparison 124.0).
Source: `src/apps/ghana/table_gate.py` (n, communities); treated count, shares and
baseline means recomputed 2026-09-26 from `load_data("data/ghana")` on the balanced panel
(no committed script writes them).

**Outcome.** Adult-equivalent household consumption expenditure per month (`aeexp_r`),
constant Greater Accra August-2017 prices; NEXIS uses the first difference
ΔY = Y₂₀₁₇ − Y₂₀₁₅.

**Estimand.** Local ATT for the marginal eligible population near the PMT cutoff,
assuming ignorable noncompliance; balanced-panel selection is a limitation.
DiD estimate **+7.35 GH₵/month** (CR1S s.e. 3.84): `src/apps/ghana/table_gate.py` →
`results/ghana/paper_numbers/table_gate.md`. Both arms fall in nominal terms from 2015 to
2017 because 2017 values are in constant 2017 prices; the DiD is the causal estimate.
*Why NEXIS applies:* Appendix E.2 extends Theorem 1 under local as-if randomization and a
subset-wise conditional parallel-trends condition on ΔY (paper, Eq. `eq:cpt`).

---

## 2. Candidate pool (m = 167)

| Block | Count | Level |
|---|---|---|
| SAE atoms active in ≥ 5 of the 162 communities | 131 of 4,096 | community |
| Spectral indices (NDVI, NDWI, MNDWI, NDBI, EVI, BSI; mean and std) | 12 | community |
| Survey covariates | 24 | household |

Built by `scripts/realworld_clustered_nexis.py::ghana` (131 + 24 = 155) and
`scripts/realworld_final_runs.py::ghana_with_spectral` (+ 12 from
`data/ghana/satellite/spectral_indices.csv`) = 167, searched in one pass.
*Why the 12 spectral indices:* the main text describes the pool with them, as in Uganda;
the June runs used 155. Selections are identical on both pools (`report.json`, runs
`pool 155 | …` and `pool 167 | …`); the spectral indices are far from entering (smallest
marginal p 0.087, `S_mndwi_std`; `report.json` → `spectral_indices_pool167`).
*Why ≥ 5 communities:* rarer atoms carry too little variation for an interaction; the
filter uses Z only, so α/m stays valid.

**Survey covariates (24).** Household composition (size, children 0–5 and 6–17, adults
18–64, elderly 65+), head of household (age, female, marital status, schooling, formal
employment), housing and WASH (rooms, rooms per person, mud walls, thatch roof, mud floor,
no electricity, improved water), livelihoods (farming household, livestock, poultry,
business, livelihood diversity, dependency ratio), housing-deprivation index
(`W_ALL` in `src/apps/ghana/data.py`).

---

## 3. Satellite pipeline

**Imagery.** Landsat 8 OLI surface reflectance, 2015 cloud-free median composite, 30 m,
5×5 km tiles at the 162 community centroids (`src/apps/ghana/download_satellite_images.py`)
and a national 5 km grid for the SAE corpus (`download_national_grid.py`), bands SR_B4,
B3, B2, B5, B6, B7. VLM images: false colour NIR/Green/SWIR2, 2–98 percentile stretch.

**Embeddings.** Prithvi-EO-1.0-100M, 768-d mean of patch tokens
(`src/apps/ghana/extract_satellite_features.py`, which reorders the bands to Prithvi's
blue-to-SWIR2 order and repeats the tile over the 3 time steps). As for Uganda, the code
takes the last encoder block, while the paper says "layer 5" (see the Uganda brief).

**SAE.** TopK, 768 → 4,096, k = 25, 2,000 epochs, batch 256, lr 2×10⁻⁴, trained on the
national grid with the 162 LEAP sites held out; whitening fit on the national corpus.
Outputs `data/ghana/satellite/{sae_model.pt, sae_activations.npy, sae_comm_ids.npy}`
(not in the repo; GPU training is not bit-reproducible, restore the original rather than
retrain).
*Why 4,096 atoms (vs 1,024 for Uganda):* a larger and more diverse national corpus.
The SLURM wrapper `scripts/ghana/slurm_train_sae.sh` calls the removed
`scripts/ghana/train_sae.py`; run `src/apps/ghana/train_sae.py` directly (see `README.md`)
until it is fixed after the ghana merge.

---

## 4. NEXIS configuration

`nexis(backward=False, terminal_filter=True, alpha=0.05, adjust="FWER", rho=0.5,
max_rounds=20)`, linear T × Z_j test on ΔY with CR1S standard errors clustered by
community (G = 162) and a t(G−1) reference (`scripts/realworld_final_runs.py`, run
`pool 167 | published test | new default`; the test is
`realworld_clustered_nexis.plain_test(cluster=community)`, identical to
`nexis(cluster=...)`). Terminal level α/m = 0.05/167 ≈ 3.0×10⁻⁴.

*Why cluster by community:* the satellite features are community constants; treating
households as independent would understate their variance. Community is a geographic
grouping, not the assignment unit (households are assigned by their own PMT score), so
the clustered p-values are not design-based (Section 7). Under HC1 without clustering, the
June NeurIPS-configuration run selected nothing (`results/ghana/codes/nexis_fwer_hc1/
result.json`); not rerun with the paper's algorithm.
*Why the linear test:* power with 162 clusters; binary survey covariates make linearity
exact and sparse atoms make it a reasonable approximation; counts, ratios, head age and the
12 spectral indices carry the assumption (paper limitation).

---

## 5. Results (Table `tab:ghana_nexis`)

S̃ = {Z_3821, Z_2095}; both pass the terminal step, so both are certified and there are no
candidates. p-values: `results/realworld_final/report.json` → `ghana/consumption` →
`runs` → `pool 167 | published test | new default` → `coords` (`p_marginal`,
`worst_subset_p`). GATEs (GH₵/month): OLS of ΔY on (1, T) within active (Z_j > 0) and
inactive households, CR1S s.e. by community, from `src/apps/ghana/table_gate.py` →
`results/ghana/paper_numbers/table_gate.{md,tex,json}`.

| Modifier | Coord | Active communities / households | GATE active | GATE inactive | Δ | Marginal p | Cert. p |
|---|---|---|---|---|---|---|---|
| Ephemeral waterways | `Z_3821` | 6 / 83 | +42.9 (15.8) | +6.0 (3.9) | +36.9 | 3.2×10⁻⁸ | 3.2×10⁻⁸ |
| Closed-canopy forest | `Z_2095` | 5 / 42 | +56.2 (19.7) | +6.4 (3.9) | +49.8 | 2.0×10⁻⁵ | 2.0×10⁻⁵ |

The certification p equals the marginal p because the worst subset is A = ∅. The GATE in
active communities is 5.8× and 7.7× the local ATT (the paper's "6–8×").

No survey covariate or spectral index is selected. *Reading in the paper:* LEAP 1000
targets a homogeneous, deeply poor population, so residual heterogeneity is environmental
rather than demographic; "this does not establish invariance".

**Marginal screening (main text).** 18 of the 167 candidates pass an uncorrected marginal
test (CR1S by community, p ≤ 0.05): 17 SAE atoms, 1 survey covariate (farming household),
0 spectral indices. Source: `src/apps/ghana/table_gate.py` →
`results/ghana/paper_numbers/table_gate.md`.

---

## 6. Interpretation

**VLM protocol** (`src/apps/ghana/interpret.py`, `scripts/ghana/slurm_interpret.sh`):
Qwen2.5-VL-72B-Instruct, 4-bit, one H100; rank the national-grid tiles (not only the 162
LEAP tiles, for a larger contrast pool) by Z_j, show the top 12 and bottom 12 side by
side with the prompt quoted in the appendix, record a short label and a confidence.

- **Z_3821, ephemeral waterways** (confidence high): narrow seasonal streams and wetland
  corridors with riparian vegetation; inactive tiles show uniform land cover. Hypothesis:
  complementarity, water-adjacent smallholders invest transfers in seasonal cultivation.
- **Z_2095, closed-canopy forest** (confidence high): dense continuous canopy; rare
  endowment in a savannah landscape. Hypothesis: complementarity with forest-based
  livelihoods; the large GATE may also reflect selection.

**Temporal check (Table `tab:ghana_temporal`).** Paired 2015/2017 composites of the
waterway-active communities shown to the VLM (`src/apps/ghana/interpret_temporal_waterways.py`;
table from `src/apps/ghana/table_temporal.py --paper` →
`results/ghana/paper_numbers/table_temporal.{md,tex}`): waterway structure unchanged in 6
of 6, cropland expansion with denser vegetation next to the waterway in 3 of 6 (951,
1265, 624). **Discrepancy, being fixed (see open issues):** neuron 3821 is active in
communities 951, 675, 395, 1265, 655 and 624; the paper's table lists 311 and 1613 (activation
0) instead of 395 and 655, and gives 675 "increased biomass" where the VLM artifact says no
change. Figure: `bash scripts/ghana/run_figure_neural.sh` (VLM temporal labels for the
4 most-activated communities, `interpret_temporal_changes.py`, then
`figure_neural_combined.py` → `results/ghana/figures/figure_neural_ghana_combined.pdf`);
teaser tile `src/apps/figure1_tiles.py` → `results/figures/figure1/waterways.pdf`;
districts map `src/apps/ghana/figure_maps.py`.

---

## 7. Limitations as reported

- **Active-community counts.** 6 and 5 active communities. A restricted wild cluster
  bootstrap-t by community (9,999 draws), each atom given the other, gives p = 0.015
  (`Z_3821`) and p = 0.056 (`Z_2095`), far above α/m (`report.json` →
  `ghana/consumption` → `posthoc`, pool 167, `p_wild`: 0.0145 and 0.0561).
- **Local identification.** RDD: the results describe households near the PMT cutoff,
  the least deprived among the eligible; the evaluation report calls RDD estimates likely
  lower bounds.
- **Community-level exposure.** One tile per community; no household-level position.
- **Linear test.** See Section 4.

---

## 8. Exploratory analysis (Appendix `sec:ghana:exploratory`)

NEXIS without multiple-testing correction on an earlier, smaller pool (72 atoms active in
≥ 10 communities, 24 survey covariates through `nexis(w=...)`, 6 spectral `*_mean`
indices; the two certified atoms are not in it), in the configuration of 6 May 2026
(NeurIPS defaults: interleaved backward step, `adjust=None`, CR1S by community). Source:
`src/apps/ghana/exploratory_run.py` → `results/ghana/paper_numbers/exploratory_run.md`,
which reproduces `results/ghana/mact10/codes/nexis_no_adj/result.json` exactly.

Selected: atoms 2252 (p = 0.0014), 1777 (0.0085), 3331 (0.0044), 1046 (0.0066),
3976 (0.0348) and farming household (0.0175). The paper discusses only atom 1777, labelled
*sparse burn scar presence* (confidence medium): active in 12 communities (228 households)
concentrated in East Mamprusi and Karaga; GATE +26.3 vs +5.6 GH₵/month; conditional
p = 0.0085; GATE contrast +20.7 (s.e. 19.9), p = 0.30. Figure:
`src/apps/ghana/figure_neural_1777.py` → `results/ghana/figures/figure_neural_1777.pdf`.
Hypothesis: transfers act as insurance after fire shocks; too local to generalise.

**External anchor: citation concern, unresolved.** The June brief supported the burn-scar
reading with "the Ghana Statistical Service's 2015 District and Regional Social
Development Profile, Chapter 4, Section 4.4.10 ('Natural Disasters, Risks and
Vulnerability'): 151 MMDAs affected by natural disasters in 2015, especially flooding and
bush fire, bulk of incidents in the Eastern and Northern Regions". Two independent searches
(2026-07-23) could not find that document, and the real GSS district reports have no such
section. The paper now attributes the same statement to a different source, the National
Development Planning Commission's *2015 Annual Progress Report* on the GSGDA II
(`NDPC2016GSGDA2015APR` in `refs.bib`, Section 4.4.10). Neither attribution has been
verified against the document here. Check the NDPC report before relying on it.

---

## 9. Compute (Table `tab:ghana_compute`)

GEE extraction (LEAP + national grid) ~1–2 h (cloud CPU); Prithvi embeddings ~30 min
(H100); SAE ~2 h (H100); NEXIS < 5 min (CPU); VLM, 2 atoms × top/bottom 12, ~30 min (H100);
VLM temporal, 6 communities × 2 years, ~15 min (H100). Estimates carried over from the June
brief; not re-measured.

---

## 10. Open issues

- `tab:ghana_temporal` rows 311/1613 should be 395/655 and row 675 should read no change
  (Section 6); the corrected rows are in `results/ghana/paper_numbers/table_temporal.tex`.
- The burn-scar citation (Section 8) is unverified.
- Paper says Prithvi "layer 5"; the code uses the last block (Section 3).
- Treated count, baseline means and within-community shares have no committed producer
  script (recomputed 2026-09-26, Section 1).
- The broken SLURM wrappers `scripts/ghana/{slurm_train_sae,run_temporal_waterways,
  slurm_temporal_waterways}.sh` are to be fixed after the ghana merge.

---

## 11. Run notes

- `scripts/realworld_final_runs.py` without `--only` runs both applications and writes
  `results/realworld_final/report.json`. The paper's Ghana rows are the runs
  `pool 167 | published test | new default`. The script first checks that the NeurIPS
  configuration reproduces the published set, read from
  `results/ghana/codes/nexis_fwer_crve/result.json` (untracked).
- The download, extraction and training scripts default to paths relative to
  `src/apps/ghana/` (`../../data/ghana/...`): run `download_satellite_images.py`,
  `download_national_grid.py` and `extract_satellite_features.py` from that folder, or pass
  `--out-dir`/`--tif-dir`; pass explicit paths to `train_sae.py` (as in `README.md`).
- The SLURM wrappers `scripts/ghana/slurm_train_sae.sh`, `run_temporal_waterways.sh` and
  `slurm_temporal_waterways.sh` call `scripts/ghana/*.py` files that no longer exist (the
  code moved to `src/apps/ghana/`); they will be fixed after the ghana merge.
- `notebooks/ghana.ipynb` is exploratory and predates the paper's configuration.
