# Uganda YOP: case study 1

> **Purpose.** The Uganda case study as it stands in the paper (Section "Case study 1"
> and Appendix D): design, the choices behind it, every number the paper quotes, and the file or script
> each number comes from. Paths are relative to the repo root; `results/` and `data/` are
> untracked. Top-level commands: `README.md`, section Uganda; run notes in Section 10;
> an earlier, broader clustered and multilevel inference analysis in Section 11.
> Updated 2026-09-26 from the June version, which used the interleaved-backward
> algorithm and reported 5 + 2 discoveries without the certified/candidate split.

---

## 1. Programme and trial

**Programme.** Youth Opportunities Program (YOP), Blattman, Fiala & Martinez (2014): cash
grants of about USD 382 per group member, plus optional vocational training, to
self-organised youth groups in post-conflict Northern Uganda (launched 2008).

**Randomization.** By group (self-selected youth groups of ~15–20 members), within
district; 535 groups randomized, 439 in our analytic sample. Baseline before treatment,
endline 2–4 years later.

**Sample.** 2,082 individuals, 439 groups, 327 distinct communities (331 RCT sites), 825
treated (39.6 %). Sub-regions: Karamoja, Teso, Lango, West Nile.
Source: `results/realworld_final/report.json` (`uganda/<outcome>` → `n`, `pool`) and
`src/apps/uganda/multilevel_groupcluster.py` log (`groups=439 districts=14 communities=327`,
`results/realworld_uganda_groupcluster/run.log`); treated count and rate recomputed from
`src/apps/uganda/pool.py::uganda` (`t.sum() = 825`, mean 0.396).

**Treatment.** T = grant received (`Wobs`, as-treated), not the lottery assignment. Of
the 265 groups assigned to the grant, 29 never received it (21 administrative, 8 theft or
diversion; Blattman et al. 2014). The paper treats them as untreated and assumes
non-receipt is as good as random, which may fail (receiving groups were slightly more
educated and wealthier); it is stated as a limitation. In the data `Wobs != assigned` in
122 of 2,082 rows (`report.json` → `n_rows_wobs_ne_assigned`).
*Why:* the published selections were computed with `Wobs`. `src/apps/realworld_final_runs.py` also
runs T = assigned (intent-to-treat) for reference; the paper reports `Wobs`.

**Outcomes.**

| Alias | CSV column | Description |
|---|---|---|
| `skilled_employed` | `skilled_dummy_e` | Any skilled trade at endline (binary) |
| `log_biz_assets` | `bizasset_val_real_ln_e` | Log real business assets at endline |

Both measure productive capacity (labour market and capital). Difference in means:
+0.32 (skilled employment), +0.61 (log business assets), replicating the positive
headline effect of Blattman et al. Source: `src/apps/uganda/table_gate.py` →
`results/uganda/paper_numbers/table_gate.md` (+0.321, +0.610).

---

## 2. Data hierarchy and candidate pool (m = 170)

| Level | Variables |
|---|---|
| Individual | outcomes; age, female, father's and mother's education |
| Group | T; share of female members (`group_female`) |
| Community / site | language group (7 dummies); 12 spectral indices; 146 SAE atoms |

**Pool.** 146 SAE atoms + 24 hand-crafted covariates = 170 candidates, searched in one
pass in which covariates and atoms compete symmetrically (they are all columns of `w`).
The 24 covariates are `W_age, W_female, W_father_educ, W_mother_educ, W_group_female`,
`W_lang_1..7` and `W_{ndvi,ndwi,mndwi,ndbi,evi,bsi}_{mean,std}` (names in `report.json`
→ `candidates`). Pool built by `src/apps/uganda/pool.py::uganda`.

**Language groups.** 7 ethnolinguistic clusters inherited from Blattman et al.
(dominant language of each district): Alur, Langi, Lugbara (`W_lang_2`), Madi, Teso,
Karamojong (`W_lang_4`), Pallisa (`W_lang_7`). Pallisa is geographic rather than
linguistic: all communities of Pallisa district, a mix of Iteso and Bagwere/Banyole.
*Why clusters and not districts:* ≈47 communities per cluster against ≈24 per district,
so enough support for an interaction. District-dummy sensitivity
(`src/apps/uganda/district_sensitivity.py` → `results/realworld_final/uganda_districts.json`;
the 7 language dummies replaced in place by 14 district dummies, m = 177, skilled
employment): Pallisa survives at the single-district level, while Karamojong and Lugbara
span several small districts and lose significance, so the cluster representation is
load-bearing for those two. **Fixed:** the paper now quotes the certification p under the
final configuration, p ≈ 7.9×10⁻⁵ (α/m = 2.8×10⁻⁴; the kind of p the main table reports).
With the paper's algorithm S̃ is {Z_859, Z_551, Z_339, Z_306, Pallisa}; Pallisa's marginal
p is 6.3×10⁻⁸, p | S̃ 7.7×10⁻⁵ (a June run with the earlier configuration gave
p(Pallisa | rest of the final set) ≈ 7.7×10⁻⁵ from
`results/uganda/prithvi_l5_1024/skilled_employed_districts/nexis_result.json`,
`nexis_fwer`; the script reproduces it to 1e-14 — kept here as provenance, not what the
paper cites). No Karamojong (Kotido, Moroto, Nakapiripirit) or Lugbara (Arua, Yumbe)
district enters S̃; given the final set their p-values are 2.7×10⁻³ (Arua) to 0.99
(Moroto). Kotido alone is marginally below α/m (1.2×10⁻⁴).

---

## 3. Satellite pipeline

**Imagery** (`src/apps/uganda/download_tiles.py`): Landsat 7 ETM+ Collection 2 Level 2
surface reflectance, 2005–2007 cloud-free median composite, 30 m, 5×5 km tiles centred on
each site (`TILE_KM = 5.0`), bands SR_B1, B2, B3, B4, B5, B7 (blue, green, red, NIR,
SWIR1, SWIR2). VLM images: false colour NIR/Green/SWIR1, 2–98 percentile stretch.
*Why 2005–2007:* the latest pre-treatment window (disbursements began 2008) without heavy
cloud cover; the median also removes the Landsat 7 SLC-off stripes. Landsat 5 has no
coverage of Uganda in 2003–2007.
*Why not the prior imagery:* Jerzak et al. (2023) used images ~7 years before the
programme with 3 bands and a binary clustering of embeddings.

**Embeddings** (`src/apps/uganda/extract_satellite_features.py`): Prithvi-EO-1.0-100M,
mean over the patch tokens, 768-d. **Fixed:** the appendix now says the embedding
averages the patch tokens of the last (12th) encoder layer, matching
`forward_features(x)[-1]` in the code. The `l5` in `prithvi_l5` refers to Landsat, not to
a layer; the extractor's docstring and comments now say Landsat 7.

**SAE** (`src/apps/uganda/train_sae.py` via `scripts/uganda/submit_train_sae.sh`,
output `results/uganda/prithvi_l5_1024/`): TopK SAE, 768 → 1,024, k = 25, unit-norm
decoder, 2,000 epochs, lr 2×10⁻⁴, 5-fold CV, trained on the national grid with the 331
RCT sites held out; whitening fit on the national corpus. *Why a national corpus:*
geographic diversity for the dictionary, and no leakage from the trial sites.
**Batch size 64** (the script default; the SLURM script passes none). The checkpoint
(`sae_model.pt`, 2026-05-05 11:35) was trained outside SLURM with no recorded command, so
this was checked on 2026-09-27 by retraining on a GPU at 64 and at 256
(`results/uganda/batch_size_check/`). Batch 64 reproduces the original log
(`logs/train_sae_l5.log`, archived): the loss trajectory matches at every logged epoch to
about 1e-4 (e.g. 0.023912 vs 0.023921 at epoch 400, including the bump at epoch 1200), and
the CV fold MSEs and live latents (167 vs 161) are close; batch 256 ends at a clearly lower
loss (0.02176 vs 0.02290) with 186 live latents. GPU training is not bit-reproducible:
restore the original `results/uganda/prithvi_l5_1024/` rather than retraining.

**Feature filter.** Atoms active (Z_j > 0) in at least 5 of the 331 sites: 146 of 1,024.
*Why:* atoms active at 1–4 sites have too little variation to estimate an interaction.
The filter uses Z only, so the terminal level α/m with m = 170 stays valid (Appendix B).

---

## 4. NEXIS configuration

`nexis(backward=False, terminal_filter=True, alpha=0.05, adjust="FWER", rho=0.5,
max_rounds=20)`, linear T × Z_j test, homoskedastic OLS, no clustering
(`src/apps/realworld_final_runs.py`, `VARIANTS["new default"]`, run
`Wobs | published test | new default`).

- Forward step admits the best candidate if p_j(S) ≤ α/|S̄| and it passes the spectral
  gap ρ = 0.5.
- Terminal backward step keeps j ∈ S̃ only if p_j(A) ≤ α/m for every A ⊆ S̃ \ {j};
  α/m = 0.05/170 ≈ 2.9×10⁻⁴.
- *Certified* = survives the terminal step; *candidate* = in S̃ only.
- *Marginal p* = unconditional T × Z_j test (`p_marginal`); *certification p* = the
  largest p_j(A) over the tested subsets (`worst_subset_p`).
- `max_rounds = 20` never binds (|S̃| ≤ 5).

*Why the linear test:* power at n = 2,082 with 170 candidates; GCM and PCM lose power when
the CATE is close to linear (CelebA ablation). Language dummies and binary covariates make
linearity exact; sparse SAE atoms make it a reasonable approximation; NDVI and the other
continuous indices carry the assumption (paper limitation).
*Why homoskedastic individual-level SEs:* the guarantee is stated for i.i.d. units and the
individual-level test gives the power to generate hypotheses; the multilevel check is in
Section 7.
*Why the terminal step and not the earlier interleaved step:* the interleaved step tests
at data-dependent conditioning sets; the terminal step tests the fixed null
H0(j | S*) on the recall event and gives the precision guarantee.

---

## 5. Results (Table `tab:nexis_appendix`)

p-values: `results/realworld_final/report.json` → `uganda/<outcome>` → `runs` →
`Wobs | published test | new default` → `coords`. GATEs (s.e.): difference in means within
active/inactive subgroups, HC1 s.e., Δ s.e. = √(se_a² + se_i²); active = Z_j > 0 for atoms,
dummy = 1 for language groups, above the sample median for NDVI; from
`src/apps/uganda/table_gate.py` → `results/uganda/paper_numbers/table_gate.{md,tex,json}`.

**Panel A: skilled employment.** S̃ = {Karamojong, Lugbara, Z_339, Z_533, Pallisa}.

| Tier | Modifier | Coord | GATE active | GATE inactive | Δ | Marginal p | Cert. p |
|---|---|---|---|---|---|---|---|
| Certified | Karamojong | `W_lang_4` | −0.030 (0.060) | +0.372 (0.022) | −0.403 (0.063) | 7.7×10⁻¹⁰ | 7.6×10⁻⁸ |
| Certified | Pallisa | `W_lang_7` | +0.674 (0.058) | +0.288 (0.022) | +0.386 (0.062) | 6.3×10⁻⁸ | 1.0×10⁻⁴ |
| Certified | Vegetation spatial heterogeneity | `Z_533` | +0.214 (0.038) | +0.373 (0.025) | −0.159 (0.045) | 6.7×10⁻⁵ | 2.7×10⁻⁴ |
| Candidate | Lugbara | `W_lang_2` | +0.092 (0.061) | +0.347 (0.023) | −0.255 (0.065) | 1.6×10⁻⁵ | 3.2×10⁻⁴ |
| Candidate | Perennial river presence | `Z_339` | +0.089 (0.097) | +0.330 (0.021) | −0.242 (0.100) | 2.1×10⁻⁴ | 4.4×10⁻⁴ |

**Panel B: log business assets.** S̃ = {NDVI, Z_820}.

| Tier | Modifier | Coord | GATE active | GATE inactive | Δ | Marginal p | Cert. p |
|---|---|---|---|---|---|---|---|
| Certified | NDVI (`ndvi_mean`) | `W_ndvi_mean` | +0.668 (0.061) | +0.552 (0.068) | +0.115 (0.092) | 5.6×10⁻⁵ | 5.6×10⁻⁵ |
| Candidate | Structured agricultural landscape | `Z_820` | +0.368 (0.094) | +0.649 (0.051) | −0.282 (0.107) | 1.7×10⁻² | 1.7×10⁻² |

The June brief's s.e. 0.098 for Z_339 was the unpooled Neyman s.e.; the paper's 0.097 is
HC1 (`table_gate.py` docstring). No individual demographic (age, sex, parental
education) enters S̃ for either outcome.

**Marginal screening (Table `tab:uganda_marginal`, main text).** Uncorrected marginal
test at 0.05: 71 of 170 (skilled employment) and 45 of 170 (log business assets), against
3 certified + 2 candidates and 1 + 1 for NEXIS. Source: recomputed on 2026-09-26 with
`causality.multilevel.plain_test()` at S = ∅ on the pool of
`apps.uganda.pool.uganda()` (71 and 45).
*Reading:* ~8–9 false positives are expected under the global null; the excess comes from
correlated SAE atoms that proxy the same few modifiers, the experimental power paradox.

---

## 6. VLM interpretation

Qwen2.5-VL-72B-Instruct, 4-bit, one H100 (`src/apps/uganda/interpret.py`, wrapper
`scripts/uganda/submit_interpret.sh`, pipeline `qwen72b`). Direct contrast: rank the 331
sites by Z_j, show the top 12 and bottom 12 tiles side by side with the prompt quoted in
the appendix, post-process into a short label. *Why contrast:* describing top tiles alone
gave generic descriptions ("~60 % vegetation").

The `feature` field of the per-outcome `interpretations.json` indexes the 146 active atoms
(50 → 339, 67 → 533, 122 → 820); `extra_atoms` uses raw SAE indices.

| Atom | Label | Source |
|---|---|---|
| 339 | perennial river presence | `results/uganda/prithvi_l5_1024/skilled_employed/qwen72b/interpretations.json` |
| 533 | vegetation spatial heterogeneity | same |
| 820 | structured agricultural landscape | `results/uganda/prithvi_l5_1024/log_biz_assets/qwen72b/interpretations.json` |
| 261 | perennial water presence (confidence high) | `results/uganda/prithvi_l5_1024/extra_atoms/qwen72b/interpretations.json` (`--extra-atoms 261`) |

Figures: `src/apps/uganda/figure_neural.py` →
`results/uganda/figures/figure_neural_{skilled_employed,log_biz_assets}.pdf` (map plus two
top and two bottom tiles per atom); `figure_maps.py` → `figure_districts.pdf`,
`figure_languages.pdf`; teaser tile `src/apps/figure1_tiles.py` →
`results/figures/figure1/river.pdf`.

---

## 7. Limitations as reported

**Multilevel inference** (`src/apps/uganda/multilevel_groupcluster.py`, run 2, →
`results/realworld_uganda_groupcluster/{report.json,summary.csv,run.log}`). Same algorithm
with the test clustered by treatment-assignment group (CR1S, G = 439, t(G−1)) and 14
district fixed effects, as in Blattman et al., after a support gate that keeps the 128
candidates with at least 5 active groups per arm (the gate uses Z and T only).
- Skilled employment certifies Karamojong and Lugbara only.
- Log business assets certifies one atom, Z_261 (perennial water presence), m = 128.
- The four environmental modifiers of Section 5 (Z_339, Z_533, NDVI, Z_820), each given the
  rest of its outcome's S̃, get randomization-inference p-values (group lottery replayed
  within district, 9,999 permutations) between 1.1×10⁻³ (Z_533) and 1.3×10⁻² (NDVI):
  Z_339 7.3×10⁻³, Z_820 8.5×10⁻³ (`run.log`, "RI|published").
- The river atom Z_339 is active in 4 treated groups (and 16 control groups).
*Reading:* the loss comes from fewer effective units (individuals → groups; community
atoms active in few groups), not from smaller effects; clustering leaves point estimates
unchanged. Clustering at the language-group level (7 clusters) would ask a different
question (generalisation to other regions). The modifiers are therefore presented as
suggestive hypotheses. (`src/apps/realworld_final_runs.py` repeats the RI with 1,999 permutations;
the paper quotes the 9,999-permutation run.)

**Community-level exposure.** All individuals of a site share one tile; within-site
exposure (e.g. distance to the river) is not measured.

**Linear test.** See Section 4.

---

## 8. Compute (Table `tab:uganda_compute`)

GEE extraction ~1–2 h (cloud CPU); Prithvi embeddings ~30 min (RTX 2080 Ti); SAE ~1 h
(RTX 2080 Ti); NEXIS < 5 min (CPU); VLM (4 atoms, top/bottom 12) ~30 min (H100). Source:
estimates carried over from the June brief; not re-measured.

---

## 9. Open issues

- None.

---

## 10. Run notes

- `src/apps/realworld_final_runs.py` without `--only` runs both applications and writes
  `results/realworld_final/report.json`, the file the paper numbers were read from
  (`--out-dir` writes elsewhere). The paper's Uganda rows are the runs
  `Wobs | published test | new default`. The Uganda grid is in
  `src/apps/uganda/final_runs.py`, the pool in `src/apps/uganda/pool.py` and the
  level-aware test in `src/causality/multilevel.py`.
- The script first checks that the earlier configuration (interleaved backward step)
  reproduces the published sets, read from
  `results/uganda/prithvi_l5_1024/<outcome>/nexis_result.json` (untracked; comes with the
  original SAE outputs).
- VLM labels: `src/apps/uganda/interpret.py` via `scripts/uganda/submit_interpret.sh`
  (Section 6); add `--extra-atoms 261` for an atom outside the selected sets.
- Website data: `src/apps/uganda/export_website_data.py` rebuilds
  `docs/assets/uganda_communities.json` and the Uganda `act` dicts of
  `docs/assets/nexis_activations.json` (`--out-dir` writes elsewhere).
- The older multi-backbone pipeline that wrote the published sets (the launcher
  `scripts/uganda/run.sh` and the steps `train.py`, `summarize.py`, `plot_features.py`)
  is not needed for the paper numbers and was removed from the tree; it is at git tag
  `pre-cleanup-2026-09`. Its `analyze.py` stays: the pool uses its `build_covariates`.

---

## 11. Clustered and multilevel inference (earlier analysis)

> **Status.** An earlier, broader robustness analysis of the June results (the
> interleaved-backward algorithm and its seven published modifiers), made before the
> paper's final configuration. The paper's multilevel limitation is Section 7
> (`src/apps/uganda/multilevel_groupcluster.py`). "Brief" and "appendix" below refer to the
> June versions; § references point inside this section.

The code of this analysis (`robustness_clustering.py`: variance-estimator sweep and nested
mixed model; `multilevel_inference.py`: level-aware clustering, wild cluster bootstrap and
randomization inference, run with `--embed-model prithvi_l5 --sae-dim 1024 --outcomes
skilled_employed,log_biz_assets --n-boot 99999 --n-perm 99999`) was removed from the
tree; it is at git tag `pre-cleanup-2026-09`. Its outputs are in
`results/uganda/prithvi_l5_1024/{robustness_clustering,multilevel_inference}/`
(untracked). Nothing was retrained; the frozen SAE artifacts were read as-is. The
level-aware test the paper uses is `src/causality/multilevel.py`.

---

### 11.1 The published SEs are homoskedastic OLS with no clustering

`nexis()` defaults to `cluster=None, hc1=False` (`src/method/nexis.py`) and
`analyze.py` never passes either. The June brief's §6 and the appendix `\paragraph{Standard errors}`
are accurate as written. The CR1S path already existed in the core but was never
wired into the Uganda app; it is validated here against `statsmodels`
`cov_type='cluster'`/`'HC1'` (exact agreement on the t-statistic).

### 11.2 Level of each candidate — corrected

**The `lang_*` dummies are region-level, not community-level.** Appendix §*Data
hierarchy* and the June brief's §3 both place them at community level; that is wrong, and it
matters more than any other single point in this document. `lang_group` *defines* a
partition of the sample into **7 regions**; each dummy is a one-hot encoding of that
partition. A variable constant within region is trivially constant within every site
inside it, so a level check that only scans group and community mislabels them. The
corrected scan runs coarsest-first:

| Level | Clusters | Count | Members |
|---|---|---|---|
| **Region** (`lang_group`) | **7** | **7** | **all `lang_*` dummies** |
| District / block | 14 | 0 | — |
| Community (site) | 327 | 158 | 146 SAE neurons + 12 spectral indices |
| Group (randomisation unit) | 439 | 1 | `group_female` |
| Individual | 2082 | 4 | `age`, `female`, `father_educ`, `mother_educ` |

Region ⊃ district ⊃ community is a strict chain here (no community spans a district,
no district spans a language group). Groups are the exception: 30 of 439 draw members
from more than one site, so group is *not* nested in community.

### 11.3 Design facts

| Partition | Clusters | With within-cluster T variation | Used as a cluster level? |
|---|---|---|---|
| Region | 7 | 7 | diagnostic only (see §11.4b) |
| Block / district | 14 | 14 | **no — it is a blocking variable** |
| Community | 327 | 49 | yes |
| Group | 439 | 0 (T assigned at group level) | yes |

Treatment was randomised over groups **within 14 district blocks** (recovered as
`ceil(strata/2)`; each block has both arms and sits inside one district and one
region). T is constant within 278 of 327 communities, so for a community-level
modifier the T×Z contrast is almost entirely between-cluster.

**District is deliberately not a clustering level.** Clustering is called for by
clustered *sampling* or clustered *assignment* (Abadie, Athey, Imbens & Wooldridge
2023); a stratification variable is neither. Blocking is handled by *conditioning* —
block fixed effects and block-stratified re-randomisation — not by clustering. With
G = 14 its CRVE is also anti-conservative, and empirically it was the single estimator
that disagreed with all others, "rescuing" `Z_339` (6.8e-05) and `W_lang_7` (7.0e-05)
after every other estimator rejected them. With it removed, the three remaining CR
variants agree unanimously (3/5 and 0/2 features clearing their gate).

**Block treatment propensity varies from 0.20 to 0.62 across the 14 blocks.** With a
blocked design and propensity that unequal, block fixed effects are needed for
*unbiasedness*, not merely efficiency — without them the pooled estimate absorbs the
correlation between a block's treatment propensity and its outcome level. The
published specification includes none. This, rather than clustering, is the correct
way to respect the design, and §11.6 reports every test with and without them.

One structural consequence: with block FE the `lang_*` **main effects are exactly
collinear** with the district dummies — a language group *is* a set of districts. The
T × lang_j interactions remain identified, but the design is rank-deficient and must
be rank-filtered before any variance is computed (`rank_filter`), or `inv(X'X)`
silently returns garbage.

### 11.4 Two distinct degeneracies, in opposite directions

Neither is a bug — both reproduce in `statsmodels`.

**(a) Sparse neurons at community level → anti-conservative.** The T×Z_j interaction
is identified off clusters that are *both* active and treated. With one or two such
cells the sandwich meat collapses while the t-test still uses df = G−1:

| Feature | active communities | of which treated | p (OLS) | p (HC1) | p (CR-community) |
|---|---|---|---|---|---|
| `Z_909` | 7 | 1 | 3.3e-04 | 5.6e-47 | 8.7e-20 |
| `Z_177` | 6 | 1 | **0.51** | 0.53 | **2.3e-07** |

`Z_177` is null under OLS and 7-sigma under clustering. Classic few-treated-clusters
failure (MacKinnon & Webb 2017). The published `--active-threshold 5` (≥5 active
*sites*) is far too weak for sandwich inference.

**(b) Language dummies at region level → no power at all.** Each `lang_*` dummy is
active in **exactly 1 of 7 regions**, by construction. This is the one-treated-cluster
case, where the restricted wild bootstrap is provably degenerate. The exhaustive
enumeration makes it visible: `W_lang_4` and `W_lang_2` return *identical* p-values of
0.2969 = 38/128, because with G=7 the bootstrap distribution is driven by the sign
flip of the single active region and carries almost no information from the data.
p = 0.30 here is not evidence of absence — it is the test having nothing to work with.

### 11.5 The method

There is a standard answer, and it does not need new methodology. Cameron & Miller
(2015, §II.C): **cluster at the coarsest level at which the regressor of interest
varies.** For a pool spanning levels, that makes the cluster level a property of the
*candidate*, not of the analysis. Concretely, four tiers:

1. **Level-aware CR1S** — each candidate clustered at its own level (region for
   `lang_*`, community for neurons/spectral, group for `group_female`, HC1 for
   individual-level). Cheap enough to run inside the selection loop.
2. **Support gate at the candidate's own level** — require ≥10 active-treated and ≥10
   active-control clusters, or the sandwich is not usable (§11.4a). 98 of 170 pass;
   **all 7 region-level candidates are dropped**, along with 65 community-level ones.
3. **Restricted wild cluster bootstrap-t** (Rademacher, null-imposed; Cameron, Gelbach
   & Miller 2008) where clusters are few, enumerated exhaustively when 2^G ≤ 2^14 so
   the p-value is exact and the resolution floor is explicit.
4. **Design-based randomization inference** — re-randomise T over groups within the 14
   real blocks under a constant-effect null. Uses only the randomness the
   experimenters created; immune to arbitrary correlation in Y and to the
   few-clusters problem. **This is the only valid test for the `lang_*` modifiers.**

NEXIS needed no redesign to accept this. It consumes only p-values and t-statistics,
so the level-aware test drops in through a new `nexis(pvalue_fn=...)` hook
(`src/method/nexis.py`) — which simply realises the claim the appendix already makes
("any valid p-value-returning test can be plugged in").

### 11.6 Results

`--n-boot 99999 --n-perm 99999`. Gate = the Bonferroni threshold NEXIS actually
applies on the realised path (≈3.0e-04).

| Feature | Level | active reg/blk/comm | OLS | CR (own level) | WCB | RI | WCB **+block FE** | **RI +block FE** |
|---|---|---|---|---|---|---|---|---|
| `W_lang_4` | region | 1/3/36 | 3.7e-10 | *degenerate* | 0.2969 *(no power)* | 1.0e-05 | 0.359 *(no power)* | **1.0e-05** ✓ |
| `W_lang_2` | region | 1/3/48 | 2.8e-07 | *degenerate* | 0.2969 *(no power)* | 2.8e-04 | 0.344 *(no power)* | **2.0e-05** ✓ |
| `W_lang_7` | region | 1/1/29 | 1.0e-04 | *degenerate* | 0.6094 *(no power)* | 0.170 | 0.719 *(no power)* | 0.225 ✗ |
| `Z_339` | community | 6/7/21 | 2.8e-06 | 4.1e-03 ✗ | 0.104 ✗ | 0.019 | 0.091 ✗ | 5.4e-03 ✗ |
| `Z_533` | community | 7/14/99 | 8.7e-06 | 1.8e-04 ✓ | 1.6e-04 ✓ | 8.7e-04 | 5.6e-04 ✗ | 1.8e-03 ✗ |
| `W_ndvi_mean` | community | 7/14/327 | 5.9e-08 | 4.6e-04 ✗ | 6.0e-05 ✓ | 3.5e-04 | 1.6e-02 ✗ | 1.3e-02 ✗ |
| `Z_820` | community | 4/7/40 | 9.5e-05 | 4.2e-04 ✗ | 4.0e-03 ✗ | 4.5e-04 | 1.3e-02 ✗ | 3.6e-03 ✗ |

The rightmost column is the correct specification: block fixed effects for the blocked
design, design-based inference for the variance. Under it exactly **two** of the seven
published modifiers survive — `W_lang_4` and `W_lang_2`.

Adding block FE moves results in both directions, which is the signature of a real
omitted-variable effect rather than noise. It *strengthens* `W_lang_2` (2.8e-04 →
2.0e-05, from borderline to clear) and `Z_339` (0.019 → 5.4e-03). It *destroys*
Panel B: `W_ndvi_mean` degrades 37× (3.5e-04 → 1.3e-02) and `Z_820` 8× (4.5e-04 →
3.6e-03). NDVI is strongly geographic, so without block FE a large part of the
apparent "NDVI modifies the treatment effect" signal was the varying block propensity
being read as environmental heterogeneity. **Panel B fails for a reason that has
nothing to do with clustering.**

Monte Carlo resolution: B = 99999, so p̂ = 1e-05 and 2e-05 rest on 0 and 1 exceedances
(at the floor); the failures at 1e-02 and 1e-03 are resolved to within a few percent.

**Level-aware NEXIS re-selection** (support-gated pool, so no `lang_*` candidates
exist): `skilled_employed` → `Z_323`, `Z_859`; `log_biz_assets` → `Z_323`,
`W_female`. None of the published modifiers is recovered, and `Z_323` fails its own
Bonferroni gate conditional on the published set (3.6e-03), so it is not a replacement
discovery either.

**Nested mixed model** (community + group random intercepts; REML, converged) — note
it omits the region level, so it is not a substitute for the region-level question:

| Outcome | Term | coef | SE | p |
|---|---|---|---|---|
| `skilled_employed` | T × `W_lang_4` | −0.378 | 0.078 | 1.2e-06 |
| | T × `W_lang_2` | −0.313 | 0.071 | 9.7e-06 |
| | T × `Z_339` | −0.091 | 0.024 | 1.8e-04 |
| | T × `Z_533` | −0.106 | 0.032 | 8.8e-04 |
| | T × `W_lang_7` | +0.234 | 0.091 | 1.0e-02 |
| `log_biz_assets` | T × `W_ndvi_mean` | +2.100 | 0.572 | 2.4e-04 |
| | T × `Z_820` | −0.276 | 0.102 | 6.6e-03 |

ICCs: 0.068 community / 0.066 group (`skilled_employed`); 0.058 / **0.181**
(`log_biz_assets`). Signs and magnitudes are stable across every estimator — this is
an inference question, not an estimation question.

#### The RI-vs-clustering gap is substantive, not an artifact

`W_lang_4` gives RI p = 1e-05 and region-clustered p = 0.30. Both are correct answers
to different questions:

- **RI (internal validity):** given the randomisation actually performed, is the
  Karamojong-vs-rest difference in treatment effect larger than chance assignment
  produces? **Decisively yes.**
- **Region clustering (external validity):** would this replicate over a fresh draw of
  regions? **Unanswerable from 7 regions, 1 of them Karamojong.**

NEXIS's claim is about *this* trial — for whom did YOP work — so the design-based
answer is the appropriate primary one. But the generalisability caveat is real and
currently unstated: a language-group modifier estimated from one region cannot be
projected to regions outside the trial.

### 11.7 Is this in agreement with the appendix?

The appendix limitations paragraph makes four claims. One holds, three do not.

| Appendix claim | Verdict |
|---|---|
| "clustering at the group level ignores the community-level dependence induced by site-constant satellite features" | **Correct.** |
| language dummies are "community-level" | **Wrong.** They are region-level (7 clusters) — §11.2. Appendix §*Data hierarchy* needs the same fix. |
| "a community-level cluster bootstrap is degenerate for those same features (all within-community observations share the same regressor value)" | **Wrong.** Constancy within cluster is the *canonical* case for a cluster bootstrap, not a degeneracy; between-cluster variation identifies the coefficient. The community-level WCB runs fine and returns p = 1.6e-04 (`Z_533`) and 6.0e-05 (`W_ndvi_mean`). The real degeneracy is at *region* level for the language dummies, for a different reason (one active cluster) — §11.4b. |
| "Neither standard clustered approach is well-suited… We leave the design of an appropriate multilevel test to future work." | **Overstated.** The standard approach is per-candidate clustering plus WCB plus design-based RI; it is ~250 lines and needs no extension to NEXIS beyond a test hook the appendix already claims exists. |

Also affected: the power argument at appendix §*Representations* ("mean ≈47
communities per cluster vs ≈24 per individual district") counts the wrong unit. For
inference on a T×language interaction, the relevant count is the number of language
clusters (**7**), not communities within them. Aggregating districts into language
groups *reduces* the number of independent units for the modifier from 14 to 7; it
buys precision only under the assumption that treatment-effect shocks do not vary at
the region level, which is exactly what is untestable here.

### 11.8 Are the results robust?

| Feature | RI + block FE (primary) | Cluster-robust at own level | Verdict |
|---|---|---|---|
| `W_lang_4` (Karamojong) | **1.0e-05** ✓ | not testable (1 of 7 regions) | **Robust internally**; not generalisable beyond the trial's regions |
| `W_lang_2` (Lugbara) | **2.0e-05** ✓ | not testable | **Robust internally**; same caveat |
| `W_lang_7` (Pallisa) | 0.225 ✗ | not testable | **Not robust** |
| `Z_533` (vegetation heterogeneity) | 1.8e-03 ✗ | 5.6e-04 ✗ | **Not robust** (borderline pre-FE) |
| `Z_339` (perennial river) | 5.4e-03 ✗ | 0.091 ✗ | **Not robust** |
| `W_ndvi_mean` | 1.3e-02 ✗ | 1.6e-02 ✗ | **Not robust** — driven by omitted block FE |
| `Z_820` (structured agriculture) | 3.6e-03 ✗ | 1.3e-02 ✗ | **Not robust** |

At the corrected multiplicity threshold, **two of seven** published modifiers survive:
`W_lang_4` and `W_lang_2`, both by design-based inference, both region-level with one
active region out of seven. The published OLS p-values of 1e-10 to 1e-04 overstate the
evidence by treating 2082 individuals as independent when the modifiers vary across 7
regions or 327 sites; Panel B additionally rests on an unblocked specification.

**Recommendation.** Report `W_lang_4` and `W_lang_2` as the heterogeneity result, with
block fixed effects and design-based p-values, and state the 7-region generalisability
limit explicitly. Demote `W_lang_7`, `Z_339`, `Z_533` and all of Panel B to
exploratory. Replace the appendix limitations paragraph: the multilevel test is not
future work, and the stated reason for avoiding a community-level bootstrap is
incorrect.

Note this leaves **no surviving SAE-neuron modifier** — the two survivors are
hand-crafted covariates inherited from Blattman et al. That is a material change to
what the Uganda case study demonstrates and should be stated plainly rather than
worked around.

**Caveats.** The mixed model omits the region level (7 units will not identify a
region random effect; this is why the design-based route is used instead). RI imposes
a constant-effect null with a single pooled τ̂; a block-specific τ̂ is the natural
sensitivity check. Region-level WCB is reported to show the test has no power, not as
evidence of absence.
