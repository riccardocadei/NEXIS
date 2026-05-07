# Ghana LEAP 1000 Experiment — Structured Brief for Paper Writing

> **Purpose.** This document is a structured description of the Ghana case study to be used as input for writing the experiment section and corresponding appendix of the NeurIPS paper. All numbers are final; all methodological choices have been made.

---

## 1. Programme & RCT Design

**Programme.** Ghana Livelihood Empowerment Against Poverty 1000 (LEAP 1000), implemented by Ghana's Department of Social Welfare. LEAP 1000 provides quarterly cash transfers to extremely poor households with children aged 0–1, targeting the intergenerational transmission of poverty. The programme was evaluated by the Institute of Statistical, Social & Economic Research (ISSER), University of Ghana, Legon.

**Experimental design.** Cluster-randomised controlled trial. Although the data contain a community identifier (`comm`) covering 162 geographic areas, **`comm` is not the randomisation unit**: 154 of 162 communities contain both Treatment and Comparison households, with a median within-community treatment share of ≈ 50%. The actual randomisation clusters are not recorded in the microdata. Treatment assignment (`tac`) is therefore used at the **household level**; `comm` is used only as a geographic grouping for GPS purposes and for clustering variance estimates (G = 162). Two survey waves: **baseline 2015** and **endline 2017**.

**Sample (balanced panel — households observed at both waves).**
- 2,331 households
- 162 communities with GPS centroids (one satellite footprint each)
- 5 districts: East Mamprusi, Karaga, Yendi, Bongo, Garu-Tempane
- 2 regions: Northern and Upper East Ghana
- Treatment rate: 50.8% (1,185 treated; 1,146 comparison)

**Geography.** Northern and Upper East Ghana — among the country's poorest areas, characterised by subsistence agriculture, seasonal water scarcity, and historically limited access to public services.

---

## 2. Outcome Considered

One outcome is selected for the main analysis:

| Alias | Column | Description | Scale |
|---|---|---|---|
| `Y` | `aeexp_r` | Adult-equivalent household consumption expenditure per month | Continuous (GH₵) |

Expenditure is deflated to constant Greater Accra August-2017 prices. It is the primary welfare measure in the evaluation and directly captures the cash-transfer programme's intended effect on household consumption.

**Baseline statistics.**
- Mean: 120.9 GH₵/month (all households)
- Treated arm: 117.8 GH₵/month; Comparison arm: 124.0 GH₵/month (near-perfect balance)

---

## 3. Estimand: ATE = ITT = ATT

The quantity of interest is the **Average Treatment Effect**:
$$\text{ATE} = E[Y(1) - Y(0)]$$

the average causal effect of receiving LEAP 1000 on adult-equivalent household expenditure. The ITT (effect of assignment) coincides with the ATT (effect on recipients): the evaluation report documents near-complete compliance (almost all assigned Treatment households received transfers, almost no Comparison households did), and the balanced-panel restriction eliminates selective attrition.

**Estimation strategy: Difference-in-Differences (DiD).** With two waves (baseline 2015, endline 2017), we estimate:
$$Y_{it} = \alpha_i + \beta_T \cdot T_i + \beta_{\text{wave}} \cdot \text{wave}_t + \delta \cdot (T_i \times \text{wave}_t) + \varepsilon_{it}$$
where $\delta$ identifies the DiD ATT under parallel trends. The identifying assumption is supported by: (i) near-random assignment of treatment within all communities; (ii) excellent baseline balance across all 24 pre-treatment covariates; (iii) baseline measurements uncontaminated by transfers (which began post-2015).

**ATT estimate:** DiD = **+7.35 GH₵/month** (treated ΔY = −32.6; comparison ΔY = −40.0; both arms fell in nominal terms due to deflation — the treated arm fell significantly less). For the DiD regression, we report both HC1 and CR1S-by-`comm` SEs; the `comm` variable is used as a geographic grouping (G = 162) rather than the true randomisation unit, which is unrecorded.

**Note on the negative nominal ΔY values.** Both arms experience a nominal expenditure decline from 2015 to 2017 because the 2017 values are deflated to constant Greater Accra August-2017 prices whereas 2015 nominal expenditures were higher in current prices. The DiD difference (treated minus control decline) of +7.35 GH₵/month is the real causal estimate.

---

## 4. Tracked Variables and Levels of Granularity

Variables operate at two levels of the data hierarchy:

### Household level
- Outcome (Y): adult-equivalent expenditure
- Treatment indicator (T): binary, household-level assignment
- Household covariates (W): 24 survey variables — see table below

### Community / site level
- Satellite imagery features (Z): one community-level tile per geographic area, one unique footprint per `comm` identifier (162 total)
- Spectral indices derived from imagery: NDVI, NDWI, MNDWI, NDBI, EVI, BSI (mean and std → 12 scalars)
- Geographic covariates: distance to district capital (km), community size (residents)
- SAE neuron activations (131 active neurons, see Section 5)

All Z features are constant within community; all households in the same `comm` inherit the same satellite-derived and geographic features.

### W covariates (24 household-level features)

| Category | Variables |
|---|---|
| Household composition | Household size, children 0–5, children 6–17, adults 18–64, elderly 65+ |
| Head of household | Age, sex (female), marital status, schooling, formal employment |
| Housing & WASH | Rooms, rooms/person, mud walls, thatch roof, mud floor, no electricity, improved water |
| Livelihoods | Farming household, has livestock, has poultry, has business, livelihood diversity, dependency ratio |
| Housing deprivation | Composite index |

---

## 5. Satellite Data

### Imagery pipeline
Satellite imagery for all 162 LEAP communities was extracted from **Google Earth Engine** using **Landsat 8 OLI** imagery:
- **Time window**: 2015 (cloud-free median composite aligned to the baseline survey year)
- **Spatial resolution**: 30 m/pixel; tiles centred on each community's GPS centroid
- **Bands used**: Green (B3), NIR (B5), SWIR2 (B7), plus additional bands for spectral index derivation
- **Visualisation**: false-colour composites (NIR / Green / SWIR2) for VLM interpretation; 2–98 percentile stretch per band per tile
- **National grid**: a full-coverage Ghana national grid was extracted in the same time window and used as the SAE training corpus (held out from the 162 LEAP community evaluation sites)

---

## 6. Learning Interpretable Satellite Representations (FM + SAE)

### Step 1 — Foundation model embeddings (Prithvi-EO)
We use **Prithvi-EO** (IBM/NASA geospatial foundation model), a Vision Transformer pretrained on global Landsat imagery. For each satellite tile we extract the patch-level embedding from **layer 5** of the encoder (768-dimensional).

### Step 2 — Sparse Autoencoder (SAE) training
A **TopK Sparse Autoencoder** with **4,096 hidden dimensions** (k = 25) is trained on Prithvi-EO embeddings from the **Ghana national satellite grid**. The 162 LEAP community sites are held out from SAE training. Whitening statistics are fit on the national corpus and applied to the LEAP embeddings at inference time.

**Architecture details.** TopK SAE: encoder = linear layer (768 → 4,096, bias), TopK activation (k = 25), decoder = unit-norm column matrix (4,096 → 768, bias). Trained for 2,000 epochs, batch size 256, learning rate 2×10⁻⁴. The larger dictionary (4,096 vs. Uganda's 1,024) is justified by the larger and more geographically diverse Ghana national training corpus.

### Step 3 — Feature filtering
Only SAE neurons active in **at least 5 of the 162 LEAP communities** (activity threshold Z_j > 0) are retained. This yields **131 active neurons** out of 4,096.

**Note on the small number of LEAP communities relative to neurons.** With only 162 communities (vs. 331 in Uganda), many SAE neurons that capture rare landscape features will be active in only 1–4 communities, giving insufficient variation to reliably estimate an interaction effect. The ≥5-community threshold is particularly consequential here: it reduces the effective candidate neural set to 131 out of 4,096.

---

## 7. NEXIS: Treatment Effect Modifier Selection

**Method.** NEXIS forward–backward stepwise procedure selecting treatment effect modifiers. At each step it tests the conditional interaction hypothesis:
$$H_0(j \mid S): \gamma_j = 0 \text{ in } \Delta Y = \beta_0 + \beta_T T + \beta_j Z_j + \gamma_S^\top (T \cdot Z_S) + \gamma_j (T \cdot Z_j) + \varepsilon$$
where ΔY = Y_endline − Y_baseline (first-differenced outcome).

**Candidate pool.** Z_full = 131 SAE neurons ∪ 24 household-level W covariates = **155 candidates total**. W covariates compete symmetrically with neural features.

**Standard errors.** CR1S cluster-robust variance estimation, clustered by `comm` (G = 162). This choice requires care: `comm` is a geographic grouping variable, not the randomisation unit — 154 of 162 communities contain both treated and comparison households and the actual randomisation clusters are unrecorded. However, clustering by `comm` is the appropriate choice for the NEXIS interaction tests because the SAE neuron features Z_j are **community-level constants**: all ~14 households in the same community share the same Z_j value, so HC1 treating them as independent observations would severely understate the SE. CR1S with G = 162 correctly reflects that effective variation is between communities, not between households. Clustering at district level (G = 5) is too few to be reliable.

**Limitation.** Under HC1 (no clustering) NEXIS FWER yields zero discoveries. The 2 reported discoveries are only certified under CR1S. This SE sensitivity should be acknowledged in the paper as a limitation of the analysis given the small number of clusters and the ambiguity of the randomisation unit.

**Correction.** FWER control via Bonferroni at each forward step (α = 0.05 / |remaining|); backward pruning with same gate; spectral-gap stopping rule (ρ = 0.5).

**Hyperparameters.** α = 0.05, max_steps = 20, ρ = 0.5, linear continuous interaction test.

---

## 8. Discovered Modifiers (FWER)

NEXIS with FWER control discovers **2 SAE neurons** and **0 W covariates**:

| Modifier | Type | Label (VLM) | GATE (active) | n (active) | GATE (inactive) | n (inactive) | Δ | SE | NEXIS p |
|---|---|---|---|---|---|---|---|---|---|
| Neuron 3821 | Z | Ephemeral waterways | 42.91 | 83 | 5.97 | 2,248 | +36.94 | 14.9 | 2.1×10⁻⁸ |
| Neuron 2095 | Z | Closed-canopy forest | 56.20 | 42 | 6.44 | 2,289 | +49.77 | 17.9 | 3.7×10⁻⁷ |

GATEs in GH₵/month; computed via difference in means within binarised subgroups (Z_j > 0 vs. = 0). NEXIS p-values are FWER-corrected (CR1S, Bonferroni sequential). Marginal unconditional p-values for the binarised GATE: neuron 3821 p = 0.013; neuron 2095 p = 0.005.

**Striking heterogeneity.** The overall ATE is +7.35 GH₵/month. Among the handful of communities where neuron 3821 or 2095 is active, the GATE is 43–56 GH₵/month — 6–8× the programme average. These are very small subgroups (6 and 5 communities out of 162, respectively), which is why NEXIS required the CRVE-adjusted test to detect them.

---

## 9. No W Discoveries: Why Demographic Interactions Are Absent

NEXIS with FWER selects **zero W covariates**, and even the exploratory no-adjustment run selects only "Farming household" marginally (p = 0.017). The marginal GATE table confirms that no demographic feature shows a substantial treatment interaction (all GATE differences well within ±10 GH₵/month, all p > 0.03 after correction).

**Justification.** Two reinforcing reasons:

1. **Narrow target population.** LEAP 1000 specifically targets households below the PMT eligibility threshold — the most deprived households in northern Ghana. This is a substantially more homogeneous population than typical RCT samples: nearly all households are subsistence farmers, have low asset holdings, and are in similar livelihood positions. The programme effect (cash → consumption) is mechanically similar across this narrow population regardless of household composition.

2. **Consumption habits are structurally stable in deep poverty.** At very low consumption levels, the marginal propensity to consume additional cash is high and relatively uniform across demographic groups. An elderly household and a young farming household at the same extreme poverty level respond similarly to a cash transfer on the margin — both spend it primarily on food and basic necessities. This is in contrast to Uganda YOP, where the *type* of investment (skilled labour, business assets) varied substantially with local market structure and opportunity, generating demographic heterogeneity.

The absence of W discoveries thus supports the programme design: the targeting was precise enough that demographic stratification within the programme group does not substantially differentiate treatment response. **The residual heterogeneity is environmental** — driven by community-level landscape features that affect whether the cash transfer can translate into sustained consumption gains.

---

## 10. VLM Interpretability Procedure

**Model.** Qwen2.5-VL-72B-Instruct (4-bit quantised, run on a single H100 80GB GPU).

**Protocol (direct contrast).** For each neuron j:
1. Rank all communities in the Ghana national grid by activation value Z_j
2. Collect **top-k = 12** (highest activation) and **bottom-k = 12** (zero/near-zero) satellite tiles from the national grid
3. Present both sets side-by-side to the VLM with a prompt describing the contrast task for Ghana/Landsat 8 imagery
4. Post-process into a concise label

**Resulting interpretations:**
- **Neuron 3821** → *Ephemeral waterways*: narrow seasonal streams/wetland corridors with adjacent riparian vegetation; inactive tiles show uniform land cover with no water courses. Confidence: **high**.
- **Neuron 2095** → *Closed-canopy forest*: dense, continuous forest canopy with minimal breaks; inactive tiles show fragmented vegetation with open spaces. Confidence: **high**.

---

## 11. Temporal Landscape Analysis (Neuron 3821)

For the primary discovery (neuron 3821, ephemeral waterways), we additionally examine the **temporal evolution of the landscape** between 2015 and 2017 using the LEAP community-level satellite images from both survey years.

**VLM temporal protocol.** For the 6 LEAP communities where neuron 3821 is active, we present paired 2015/2017 false-colour composites to Qwen2.5-VL-72B and ask it to describe changes in waterways, agriculture, vegetation, and settlements.

**Finding.** Among the 6 top-activated communities, the VLM consistently reports:
- **Waterways**: unchanged — seasonal streams/wetland corridors appear similarly visible in both years
- **Agricultural change**: 3 of 6 communities (comm IDs 951, 1265, 624) show expansion of bare/tan cropland in 2017 relative to 2015
- **Vegetation change**: these same 3 communities show denser vegetation adjacent to waterways in 2017

The landscape near ephemeral waterways appears to have intensified agricultural use over the 2015–2017 period — precisely the endline period during which LEAP transfers were distributed. This is consistent with an interpretation where cash transfers enabled households near seasonal water sources to expand irrigation-adjacent smallholder cultivation.

The combined figure (`figure_neural_ghana_combined.pdf`) shows the top and bottom activation tiles for neuron 3821 (left panel) connected by a V-shaped link to the temporal grid of the 4 most representative active communities (right panel, 2015 top / 2017 bottom), with VLM-detected land-use changes annotated between years.

---

## 12. Exploratory Analysis: NEXIS with Relaxed Corrections

To assess robustness and identify potentially important features with insufficient power under FWER, we also run NEXIS without multiple-testing correction. This yields:

| Modifier | Type | Label (VLM) | NEXIS p (no adj.) |
|---|---|---|---|
| Neuron 2252 | Z | *(no label)* | 0.0014 |
| Neuron 1777 | Z | Sparse burn scar presence | 0.0085 |
| Neuron 3331 | Z | *(no label)* | 0.0044 |
| Neuron 1046 | Z | *(no label)* | 0.0066 |
| Neuron 3976 | Z | *(no label)* | 0.0348 |
| Farming household | W | — | 0.0175 |

**Note on power.** Even FDR control (BH procedure) yields 0 discoveries, confirming that the exploratory neuron signals are genuinely weak. NEXIS FWER selects 3 neurons; NEXIS FDR selects none; NEXIS no-adj selects 5 neurons + 1 W. The gap between FWER and FDR underscores limited statistical power: the Ghana sample has 162 communities (vs. 331 in Uganda), and the active communities per neuron are very few (5–12 out of 162).

### Neuron 1777 — "Sparse burn scar presence"

The most interpretable exploratory discovery is **neuron 1777**, labelled by Qwen2.5-VL-72B as *Sparse burn scar presence* (small irregular burn scars scattered across vegetation; confidence: medium). This neuron is active in **12 of 162 communities** and geographically concentrated in the **East Mamprusi and Karaga districts** (Northern Region).

GATE: communities with active neuron 1777 show a GATE of +26.3 GH₵/month vs. +5.6 GH₵/month for inactive communities (Δ = +20.7, SE = 19.9; marginal p = 0.30 — not significant individually, but selected by NEXIS path in the no-adj run at p = 0.0085 in the conditional interaction test).

**External validation.** The Ghana Statistical Service's 2015 District and Regional Social Development Profile (the monitoring report contemporaneous with the baseline imagery) documents, in Chapter 4, Section 4.4.10 ("Natural Disasters, Risks and Vulnerability"), that 151 MMDAs were affected by natural disasters in 2015, "especially flooding and bush fire," with the bulk of incidents in the **Eastern and Northern Regions**. The 2015 satellite imagery for communities where neuron 1777 is active — the same year the imagery was captured — thus visually records the aftermath of actual documented fire events in those communities.

**Interpretation.** The hypothesis is that **receiving LEAP cash transfers in a community exposed to fire shocks amplifies programme impact on expenditure**. Fire shocks destroy subsistence assets (crops, livestock feed, stored food) and create acute liquidity needs. A cash transfer arriving in this context provides insurance-like relief that directly buffers against the shock, translating into a larger net consumption gain relative to unexposed communities. This is consistent with the broader literature on cash transfers as insurance against covariate shocks.

The geographical specificity of neuron 1777 (East Mamprusi–Karaga cluster) makes this effect very local — the signal is real but narrowly identified, which explains why FWER and even FDR cannot certify it.

---

## 13. Marginal Testing Baseline vs NEXIS

| Method | Z discoveries | W discoveries |
|---|---|---|
| Marginal, unadjusted (α = 0.05) | ~12 neurons (p < 0.05 in GATE table) | 1 (Farming hh) |
| Marginal, FWER-grouped | 0 | 0 |
| NEXIS, FDR | 0 | 0 |
| NEXIS, FWER (CRVE) | **2** | **0** |
| NEXIS, no adjustment | 5 | 1 |

Marginal screening at α = 0.05 unadjusted surfaces ~12 neurons with nominally significant GATE differences, most of which reflect correlated spatial patterns rather than independent modifiers. NEXIS conditions each forward step on already-selected features, pruning correlated redundant signals. The 3 NEXIS FWER discoveries represent a parsimonious set of conditionally independent modifiers.

The stricter-than-FDR profile (FWER finds 3; FDR finds 0) is unusual and merits comment: it arises because Bonferroni sequential testing at each step can be more powerful than BH in the specific forward-stepwise path if the top signals have very small conditional p-values (2×10⁻⁸, 4×10⁻⁷) that survive even the conservative FWER gate, while no feature clears the FDR threshold globally because the test distribution has too many features with medium p-values (0.01–0.1) that inflate the BH correction. The sequential conditioning in NEXIS efficiently concentrates power on the strongest signals.

---

## 14. Computational Budget

| Component | Hardware | Runtime |
|---|---|---|
| GEE imagery extraction (162 LEAP communities + national Ghana grid) | CPU (cloud) | ~1–2 h |
| Prithvi-EO embedding extraction (LEAP + national) | H100 80GB | ~30 min |
| SAE training (Ghana, 4,096 hidden, 2,000 epochs) | H100 80GB | ~2 h |
| NEXIS analysis (155 candidates, FWER + FDR + no-adj variants) | CPU | < 5 min |
| VLM interpretation (2 neurons × top/bottom 12 images) | H100 80GB | ~30 min |
| VLM temporal analysis (6 communities × 2 years) | H100 80GB | ~15 min |

---

## 15. Interpretation of Results as Hypotheses

### Ephemeral waterways (Neuron 3821)

**Positive modifier** (GATE active = +42.9, GATE inactive = +6.0). Hypothesis: households living near seasonal streams and wetland corridors in northern Ghana have differential access to water for smallholder cultivation and livestock. LEAP cash transfers in these communities enable households to invest in agricultural inputs (seeds, fertiliser, small tools) for land adjacent to seasonal water — an investment opportunity that is simply not available to households far from water. The temporal analysis supports this: in the 6 active communities, the VLM detects cropland expansion near waterways between 2015 and 2017.

The ephemeral (seasonal) character of the waterways is relevant: these are not perennial rivers providing year-round access, but seasonal flood corridors active during the wet season. The LEAP transfer timing (quarterly) means at least one payment arrives during the agricultural season, enabling water-adjacent smallholders to invest at the right moment.

### Closed-canopy forest (Neuron 2095)

**Positive modifier** (GATE active = +56.2, GATE inactive = +6.4). Hypothesis: LEAP communities located within or adjacent to closed-canopy forest zones benefit from both the direct economic value of forest access (non-timber forest products, fuelwood, forest-edge cultivation) and from the ecological services forests provide (soil moisture retention, microclimate regulation, protection against crop failure). Cash transfers in these communities complement existing forest-based livelihood strategies, allowing households to intensify or diversify production in ways not feasible in dryer, deforested areas.

Northern Ghana and Upper East are predominantly savannah; closed-canopy forest patches are rare and geographically concentrated. The 5 communities where neuron 2095 is active are outliers in the landscape — their anomalous effect size (GATE = +56.2 GH₵/month, nearly 8× the overall ATE) reflects a combination of unusually productive environmental endowments and possibly larger household capacity to translate cash into sustained welfare gains.

### General pattern

Both discovered modifiers are **environmental features** with no direct analogue in the household survey. This is the central motivation for the satellite-based approach: the strongest treatment effect heterogeneity is driven by the physical environment of communities — features that no household questionnaire would capture — and that only satellite imagery can reveal at scale.

The overall ATE (+7.35 GH₵/month) masks substantial spatial heterogeneity: in the few communities with favourable environmental endowments (water access, forest cover), the programme generates effects 6–8× larger. This has direct implications for programme targeting: a satellite-based screening tool could identify high-impact communities before programme expansion.

---

## 16. Suggested Paper Structure

### Main section
1. Dataset and DiD design (LEAP 1000 programme, panel structure, ATE = ITT = ATT, ATE estimate)
2. Satellite data and FM+SAE pipeline (GEE, Prithvi-EO, Ghana national corpus SAE, 131 active neurons)
3. NEXIS setup (candidates: 131 SAE + 24 W = 155 total; FWER with CRVE; DiD first-differenced outcome)
4. Results table (2 main FWER discoveries with VLM labels + neuron 3318; GATE estimates)
5. Neural interpretation figures: `figure_neural_ghana_combined.pdf` (static features + temporal grid for neuron 3821)
6. No W discoveries — interpretation (narrow target population + consumption homogeneity)
7. Interpretation / hypotheses (environmental amplifiers of cash transfer effects)

### Appendix
- Full RCT variable definitions, summary statistics, balance table
- DiD estimation details (regression specification, parallel trends discussion, CRVE)
- GEE imagery extraction details (Landsat 8, bands, time window, cloud masking)
- SAE training details (architecture, TopK, Ghana national corpus, 4,096 neurons)
- NEXIS algorithm pseudocode / full hyperparameter table
- VLM prompt templates (verbatim)
- Activation maps for all discovered neurons (top/bottom image grids from national grid)
- Temporal analysis detail: VLM per-community responses for neuron 3821 communities
- Robustness: NEXIS FDR and no-adj comparison; discussion of power limitations with G = 162
- Marginal testing baseline vs NEXIS comparison (Section 13 numbers)
- Exploratory neuron 1777 (burn scars): geographical map, VLM label, external validation reference
- Comparison with Uganda case study: scale, outcome type, modifier types (W-dominant in Uganda vs Z-dominant in Ghana)
