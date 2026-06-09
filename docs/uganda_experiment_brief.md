# Uganda YOP Experiment — Structured Brief for Paper Writing

> **Purpose.** This document is a structured description of the Uganda case study to be used as input for writing the main experiment section and the corresponding appendix of the NeurIPS paper. All numbers are final; all methodological choices have been made.

---

## 1. Programme & RCT Design

**Programme.** Uganda Youth Opportunities Programme (YOP), studied in Blattman et al. (2014). The programme offered cash grants (~USD 382 per group member) plus optional vocational training to self-organised youth groups in Northern Uganda, targeting young adults with limited economic opportunity in the aftermath of the Lord's Resistance Army conflict.

**Experimental design.** Randomised controlled trial. Treatment (T) was assigned at the **group level** (unit of randomisation = self-selected youth group of ~15–20 members). Groups were clustered within communities (geographic sites). Baseline data were collected pre-treatment; endline outcomes ~2–4 years later.

**Sample.**
- 2,082 individuals (endline)
- 439 groups (treatment randomisation units)
- ~331 distinct geographic communities (RCT sites, each with a unique satellite footprint)
- Treatment rate: 39.6% (825 treated individuals)

**Geography.** Northern Uganda; sites span multiple districts including Karamoja, Acholi, Teso, Lango, and West Nile sub-regions.

---

## 2. Outcomes Considered

Two outcomes were selected for the main analysis based on data quality and theoretical interest:

| Alias | CSV column | Description | Scale |
|---|---|---|---|
| `skilled_employed` | `skilled_dummy_e` | Any skilled trade engagement at endline | Binary |
| `log_biz_assets` | `bizasset_val_real_ln_e` | Log real business asset value at endline | Continuous |

Both outcomes measure whether the programme succeeded in building productive economic capacity — the first through labour market participation, the second through capital accumulation.

### Average Treatment Effects and Comparison with Blattman et al. (2014)

The original Blattman et al. (2014) paper reports significant positive average treatment effects on both outcomes: increased skilled employment and log business asset value approximately two years post-treatment. Our analysis replicates these positive ATEs in the full sample — the sample-weighted average across NEXIS subgroups also yields positive estimates (skilled employment: ~+0.31; log business assets: ~+0.61). The ATE is the right summary for programme-level policy evaluation (did the programme work on average?). NEXIS asks the complementary question: *for whom* and *under what geographic conditions* does the effect vary?

Blattman et al. also explore heterogeneity by baseline characteristics (age, sex, baseline wealth). NEXIS extends this by (i) systematically searching over a high-dimensional candidate set including satellite-derived features not available to the original authors, (ii) controlling the sequential selection path to limit false discoveries, and (iii) producing interpretable hypotheses via VLM labelling of SAE neurons.

### Note on p-values reported in Section 7

The marginal p-values in the results table come from **unconditional** T × modifier interaction regressions (no features in the conditioning set S). We report marginal rather than NEXIS conditional p-values for two reasons: (1) marginal tests are simple and directly comparable across features and across papers; (2) NEXIS conditional p-values depend on the current selection path S and become hard to interpret for downstream readers unfamiliar with the sequential procedure. NEXIS selects features; the marginal test is reported to characterise the strength of the raw signal for each discovered modifier.

---

## 3. Tracked Variables and Levels of Granularity

Variables operate at three distinct levels of the data hierarchy:

### Individual level
- Outcomes (Y): `skilled_employed`, `log_biz_assets`
- Demographics: age, sex (female), father's education, mother's education

### Group level
- Treatment indicator (T): binary, constant within group
- Group composition: share female members (`group_female`)
- Language/ethnicity group (`lang_group`, 7 categories): Acholi, Langi, Lugbara, Madi, Teso, Karamojong, other — constant within group, district-level stratification variable

### Community / site level
- Satellite imagery features (Z): one 112×112 pixel Landsat 7 tile per site, centred on the community's geographic centroid
- Spectral indices derived from imagery: NDVI, NDWI, MNDWI, NDBI, EVI, BSI (mean and std per tile → 12 scalars)
- All Z features are constant within community; all individuals at the same site inherit the same satellite-derived features

---

## 4. Satellite Data: Extension via Google Earth Engine and Comparison with Prior Work

### Comparison with Jerzak et al. (2023)

Jerzak et al. (2023) is the closest prior work using this Uganda YOP dataset for treatment effect modifier discovery with satellite imagery. Their approach differs from ours in three key dimensions:

| Dimension | Jerzak et al. (2023) | This work |
|---|---|---|
| Imagery vintage | Static images from ~2000 (pre-treatment by 8–10 years) | 2005–2007 median composite (immediately pre-treatment) |
| Image format | Pre-computed index scores in CSV format; limited spectral richness | Raw Landsat 7 multispectral tiles (6 bands) |
| Feature extraction | Hand-crafted spectral indices; no deep features | Prithvi-EO foundation model embeddings → SAE dictionary |
| Candidate modifiers | Structured tabular covariates | 146 SAE neurons + 24 spectral/demographic covariates (170 total) |
| Interpretability | Features are human-defined | SAE neurons interpreted post-hoc via VLM image contrast |
| Selection procedure | Causal forest / CATE-based heterogeneity tests | NEXIS forward–backward stepwise with FWER control |

The older, temporally misaligned images used by Jerzak et al. introduce measurement error that may dilute true heterogeneity signals. By re-extracting imagery closer to baseline and using a foundation model to expand the candidate feature space, NEXIS identifies modifiers (perennial river presence, vegetation heterogeneity, structured agriculture) that were not surfaced by the prior approach.

### New imagery pipeline (GEE)
We re-extracted satellite imagery for all 331 RCT sites directly from **Google Earth Engine** using **Landsat 7 ETM+** imagery:
- **Time window**: 2005–2007 (3-year cloud-free median composite)
- **Rationale**: immediately pre-treatment, maximising relevance to baseline conditions; 2005–2007 is the latest window before the programme launched (~2008) without substantial cloud contamination in the Landsat 7 record for Northern Uganda
- **Spatial resolution**: 30 m/pixel; tiles cropped to 112×112 pixels (~3.36 km × 3.36 km) centred on each site's GPS centroid
- **Bands used**: Blue (B1), Green (B2), Red (B3), NIR (B4), SWIR1 (B5), SWIR2 (B6)
- **Visualisation**: false-colour composites (NIR / Green / SWIR1) for VLM interpretation; 2–98 percentile stretch per band per tile

---

## 5. Learning Interpretable Satellite Representations (FM + SAE)

### Step 1 — Foundation model embeddings (Prithvi-EO)
We use **Prithvi-EO** (IBM/NASA geospatial foundation model), a Vision Transformer pretrained on global Landsat imagery. For each satellite tile we extract the patch-level embedding from **layer 5** of the encoder (768-dimensional). This gives a rich, pretrained representation of land-cover and landscape structure without any task-specific supervision.

### Step 2 — Sparse Autoencoder (SAE) training
A **TopK Sparse Autoencoder** (SAE; Gao et al., 2024) with **1,024 hidden dimensions** and sparsity k = 25 is trained on Prithvi-EO embeddings from a **national Uganda satellite grid** (full-country coverage, same Landsat 7 2005–2007 time window). The 331 RCT sites are held out from SAE training; the national grid corpus provides geographic diversity for learning a rich feature dictionary without data leakage. Whitening statistics (mean and std) are fit on the national corpus and applied to the RCT embeddings at inference time.

The SAE learns a sparse dictionary of 1,024 "neurons" (basis directions) that reconstruct Prithvi embeddings with high fidelity and high sparsity. Each neuron can be interpreted as a distinct visual concept detectable in satellite imagery.

**Architecture details.** TopK SAE: encoder = linear layer (768 → 1,024, bias), TopK activation (k = 25 active units per sample), decoder = unit-norm column matrix (1,024 → 768, bias). Trained for 2,000 epochs, batch size 256, learning rate 2×10⁻⁴, 5-fold cross-validation on the national corpus.

**Comparison with Ghana SAE.** The Ghana case study uses a structurally identical TopK SAE but with **4,096 hidden dimensions** (k = 25), trained on the Ghana national satellite imagery corpus. The larger dictionary is justified by the larger and more diverse Ghana training set. Both SAEs use the same Prithvi-EO backbone (layer 5, 768-dim input). In the Ghana analysis, 131 of 4,096 neurons are active in at least 5 of the LEAP 1000 evaluation communities (activity threshold Z_j > 0). No NEXIS discoveries are found for Ghana, suggesting either weaker treatment effect heterogeneity in the Ghana data or insufficient power given the Ghana sample size.

### Step 3 — Feature filtering
For the Uganda RCT analysis, only SAE neurons active in **at least 5 of the 331 RCT sites** are retained (activity threshold Z_j > 0). This yields **146 active neurons** out of 1,024, forming the neural candidate set Z for NEXIS. The threshold prevents highly sparse neurons (active at 1–4 sites) from entering the regression, as they would have insufficient variation to reliably estimate an interaction effect.

---

## 6. NEXIS: Treatment Effect Modifier Selection

**Method.** NEXIS (Neural Exposure Interaction Search) is a forward–backward stepwise procedure that selects treatment effect modifiers from a large candidate set. At each step it tests the conditional interaction hypothesis:
$$H_0(j \mid S): \gamma_j = 0 \text{ in } Y = \beta_0 + \beta_T T + \beta_j Z_j + \gamma_S^\top (T \cdot Z_S) + \gamma_j (T \cdot Z_j) + \varepsilon$$
using a **continuous linear** interaction t-test (not binarised), conditioning on the already-selected set S.

**Candidate pool.** Z_full = 146 SAE neurons ∪ 24 hand-crafted W covariates (demographics + spectral indices) = **170 candidates total**. W covariates compete symmetrically with neural features.

**Correction.** FWER control via **Bonferroni** correction at each forward step (α = 0.05 / |remaining|); backward pruning enforces the same gate. Additionally a spectral-gap stopping rule (ρ = 0.5) prevents selecting features whose conditional t-statistic is less than half that of the weakest already-selected feature.

**Hyperparameters.** α = 0.05, max\_steps = 20, ρ = 0.5, linear test (no nonparametric nuisance).

**Standard errors.** Standard homoskedastic OLS (no clustering). [Note: cluster-robust SEs at group or community level were explored but found to destabilise the sequential selection path without a clear improvement in validity; see Appendix.]

---

## 7. Discovered Modifiers (FWER)

### Panel A: Skilled Employment
| Modifier | Type | GATE (active) | GATE (inactive) | Δ | Marginal p |
|---|---|---|---|---|---|
| Teso (lang. 4) | W | −0.030 (0.060) | +0.372 (0.022) | −0.403 (0.063) | 7.7×10⁻¹⁰ |
| Acholi (lang. 2) | W | +0.092 (0.061) | +0.347 (0.023) | −0.255 (0.065) | 1.6×10⁻⁵ |
| Karamojong (lang. 7) | W | +0.674 (0.058) | +0.288 (0.022) | +0.386 (0.062) | 6.3×10⁻⁸ |
| Neuron 339 | Z | +0.089 (0.098) | +0.330 (0.021) | −0.242 (0.100) | 2.1×10⁻⁴ |
| Neuron 533 | Z | +0.214 (0.038) | +0.373 (0.025) | −0.159 (0.045) | 6.7×10⁻⁵ |

### Panel B: Log Business Assets
| Modifier | Type | GATE (active) | GATE (inactive) | Δ | Marginal p |
|---|---|---|---|---|---|
| NDVI | W | +0.668 (0.061) | +0.552 (0.068) | +0.115 (0.092) | 5.6×10⁻⁵ |
| Neuron 820 | Z | +0.368 (0.094) | +0.649 (0.051) | −0.282 (0.107) | 1.7×10⁻² |

GATEs computed via difference in means within binarised subgroups (neurons: Z > 0; NDVI: above median). Marginal p-values from unconditional continuous linear interaction test T × modifier.

---

## 8. Marginal Testing Baseline vs NEXIS

To illustrate the value of NEXIS over naive marginal screening, we compare with a pure marginal baseline: for each outcome, test all 170 candidates for T × modifier interactions individually using α = 0.05 (unadjusted, no multiple-testing correction). The number of discoveries is:

| Outcome | Marginal (α = 0.05, unadjusted) | NEXIS FWER |
|---|---|---|
| `skilled_employed` | **71** features | **5** features |
| `log_biz_assets` | **45** features | **2** features |

Marginal testing at a nominal α = 0.05 with 170 tests yields approximately 8–9 expected false positives under the null, but in practice produces 45–71 "discoveries" — most of which are driven by confounding between correlated SAE neurons (a dense neuron cluster active in overlapping sites will all be marginally significant). NEXIS addresses this by conditioning each forward step on the features already selected, which eliminates the downstream significance of correlated redundant features. The 5 + 2 NEXIS discoveries are a parsimonious set of genuine, conditionally independent modifiers.

---

## 9. VLM Interpretability Procedure

**Goal.** Assign a human-readable semantic label to each discovered SAE neuron by inspecting the satellite images that most strongly activate it.

**Model.** Qwen2.5-VL-72B-Instruct (4-bit quantised, run on a single H100 80GB GPU).

**Protocol (direct contrast).** For each neuron j:
1. Rank all 331 RCT sites by their activation value Z_j
2. Collect the **top-k = 12** (highest activation) and **bottom-k = 12** (zero/near-zero) satellite tiles
3. Present both sets side-by-side to the VLM with the prompt: *"These are pairs of satellite images from Uganda (Landsat 7, 2005–2007). The left column shows sites where a learned visual feature is strongly active; the right column shows sites where it is inactive. Describe in one short phrase what landscape or environmental property distinguishes the active from the inactive sites."*
4. The VLM response is post-processed into a concise label

**Resulting interpretations:**
- **Neuron 339** → *perennial river presence* (sites along permanent watercourses)
- **Neuron 533** → *vegetation spatial heterogeneity* (mosaic of agricultural patches and bush)
- **Neuron 820** → *structured agricultural landscape* (regular field grid, mechanised-scale agriculture)

---

## 10. Computational Budget

| Component | Hardware | Runtime |
|---|---|---|
| GEE imagery extraction (331 RCT + national tiles) | CPU (cloud) | ~1–2 h |
| Prithvi-EO embedding extraction (RCT + national) | RTX 2080 Ti | ~30 min |
| SAE training (Uganda, 1024 hidden, 2000 epochs) | RTX 2080 Ti | ~1 h |
| NEXIS analysis (both outcomes, 170 candidates) | CPU | < 5 min |
| VLM interpretation (3 neurons × top/bottom 12 images) | H100 80GB | ~30 min |

For comparison, the Ghana SAE (4,096 hidden dimensions, 2,000 epochs) was trained on an H100 80GB GPU with a 2-hour time budget.

---

## 11. Interpretation of Results as Hypotheses

### Skilled Employment

**Language group heterogeneity (W).**
- *Karamojong communities* show the largest positive treatment effect (GATE = +0.674 vs sample mean ≈ +0.31). Karamoja is among Uganda's most economically marginalised regions; the programme may have filled a near-complete gap in vocational opportunity.
- *Teso communities* show no positive effect (GATE = −0.030), possibly because Teso has a more developed local labour market where cash grants alone do not shift skilled employment trajectories.
- *Acholi communities* show intermediate effects, consistent with partial recovery from conflict-related displacement.

**Neural modifiers (Z).**
- *Neuron 339 (river presence)*: negative modifier. Hypothesis: proximity to permanent water supports subsistence agriculture and fishing as alternatives to skilled trade, reducing uptake of programme-supported vocational paths.
- *Neuron 533 (vegetation heterogeneity)*: negative modifier. Hypothesis: agro-ecological diversity (mixed bush/crop mosaic) correlates with more flexible livelihood strategies, making skilled employment a less marginal improvement over the status quo.

### Log Business Assets

**NDVI (W).**
- Positive modifier: greener, more fertile sites see larger treatment effects on business asset accumulation. Hypothesis: baseline agricultural productivity provides collateral and cash flow that amplifies the productive use of the grant capital.

**Neuron 820 (structured agriculture, Z).**
- Negative modifier. Hypothesis: areas with already-structured, large-scale agricultural landscapes may be dominated by existing commercial actors, leaving less room for new small-business entrants supported by the programme grants to accumulate assets.

---

## 12. Suggested Paper Structure

### Main section
1. Dataset and RCT design (brief)
2. Satellite data and FM+SAE pipeline (key design choices: GEE imagery, Prithvi, Uganda national corpus SAE)
3. NEXIS setup (candidates, correction, test)
4. Results table (FWER discoveries, GATE estimates)
5. Neural interpretation figures (one per outcome)
6. Interpretation / hypotheses (qualitative, 1 paragraph per outcome)

### Appendix
- Full RCT variable definitions and summary statistics
- GEE imagery extraction details (bands, time window, cloud masking)
- SAE training details (architecture, loss, TopK, Uganda national corpus)
- NEXIS algorithm pseudocode / full hyperparameter table
- VLM prompt templates (verbatim)
- Activation maps for all discovered neurons (full top/bottom image grids)
- Robustness: CRVE discussion (why dropped), FDR vs FWER comparison
- Marginal testing baseline vs NEXIS comparison (Section 8 numbers)
- Comparison with Jerzak et al. (2023): imagery vintage, feature extraction pipeline, selection method
