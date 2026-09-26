# Uganda YOP experiment brief

> **Purpose.** The Uganda case study as it stands in the final paper
> (`paper/iclr27/main.tex`, Section "Case study 1", and `appendix.tex`, Appendix D):
> design, the choices behind it, every number the paper quotes, and the file or script
> each number comes from. Paths are relative to the repo root; `results/` and `data/` are
> untracked. Top-level commands: `README.md`, section Uganda; run notes in Section 10.
> Updated 2026-09-26 from the June (NeurIPS) version, which used the interleaved-backward
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
`scripts/realworld_uganda_groupcluster.py` log (`groups=439 districts=14 communities=327`,
`results/realworld_uganda_groupcluster/run.log`); treated count and rate recomputed from
`scripts/realworld_clustered_nexis.py::uganda` (`t.sum() = 825`, mean 0.396).

**Treatment.** T = grant received (`Wobs`, as-treated), not the lottery assignment. Of
the 265 groups assigned to the grant, 29 never received it (21 administrative, 8 theft or
diversion; Blattman et al. 2014). The paper treats them as untreated and assumes
non-receipt is as good as random, which may fail (receiving groups were slightly more
educated and wealthier); it is stated as a limitation. In the data `Wobs != assigned` in
122 of 2,082 rows (`report.json` → `n_rows_wobs_ne_assigned`).
*Why:* the published selections were computed with `Wobs`. `realworld_final_runs.py` also
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
pass in which covariates and atoms compete symmetrically (they are all columns of `z`).
The 24 covariates are `W_age, W_female, W_father_educ, W_mother_educ, W_group_female`,
`W_lang_1..7` and `W_{ndvi,ndwi,mndwi,ndbi,evi,bsi}_{mean,std}` (names in `report.json`
→ `candidates`). Pool built by `scripts/verify_new_default_realworld.py::uganda_data`.

**Language groups.** 7 ethnolinguistic clusters inherited from Blattman et al.
(dominant language of each district): Alur, Langi, Lugbara (`W_lang_2`), Madi, Teso,
Karamojong (`W_lang_4`), Pallisa (`W_lang_7`). Pallisa is geographic rather than
linguistic: all communities of Pallisa district, a mix of Iteso and Bagwere/Banyole.
*Why clusters and not districts:* ≈47 communities per cluster against ≈24 per district,
so enough support for an interaction. District-dummy sensitivity
(`scripts/realworld_uganda_districts.py` → `results/realworld_final/uganda_districts.json`;
the 7 language dummies replaced in place by 14 district dummies, m = 177, skilled
employment): Pallisa survives at the single-district level, while Karamojong and Lugbara
span several small districts and lose significance, so the cluster representation is
load-bearing for those two. The paper's p ≈ 7.7×10⁻⁵ is p(Pallisa | rest of the final set)
from a June run with the NeurIPS configuration
(`results/uganda/prithvi_l5_1024/skilled_employed_districts/nexis_result.json`,
`nexis_fwer`; the script reproduces it to 1e-14). With the paper's algorithm S̃ is the same
{Z_859, Z_551, Z_339, Z_306, Pallisa}; Pallisa is certified with certification p
7.9×10⁻⁵ (α/m = 2.8×10⁻⁴; the kind of p the main table reports), marginal p 6.3×10⁻⁸,
p | S̃ 7.7×10⁻⁵. No Karamojong (Kotido, Moroto, Nakapiripirit) or Lugbara (Arua, Yumbe)
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
mean over the patch tokens, 768-d. **Discrepancy:** the paper says "layer 5 of the
encoder"; the code takes `forward_features(x)[-1]`, the output of the last (12th) block
after the final LayerNorm. The `l5` in `prithvi_l5` appears to refer to Landsat (the
docstrings still say Landsat 5), not to a layer. Not changed; see open issues.

**SAE** (`src/apps/uganda/train_sae.py` via `scripts/uganda/train_sae_slurm.sh`,
output `results/uganda/prithvi_l5_1024/`): TopK SAE, 768 → 1,024, k = 25, unit-norm
decoder, 2,000 epochs, lr 2×10⁻⁴, 5-fold CV, trained on the national grid with the 331
RCT sites held out; whitening fit on the national corpus. *Why a national corpus:*
geographic diversity for the dictionary, and no leakage from the trial sites.
**Discrepancy:** the paper says batch size 256; the SLURM script does not pass
`--batch-size`, so `train_sae.py` uses its default 64. The original checkpoint
(`sae_model.pt`, 2026-05-05 11:35) stores no config, and the last logged SLURM run
(`logs/slurm-sae-58521261`) hit its time limit at 08:20, so the call that wrote the
checkpoint is not traceable. GPU training is not bit-reproducible: restore the original
`results/uganda/prithvi_l5_1024/` rather than retraining.

**Feature filter.** Atoms active (Z_j > 0) in at least 5 of the 331 sites: 146 of 1,024.
*Why:* atoms active at 1–4 sites have too little variation to estimate an interaction.
The filter uses Z only, so the terminal level α/m with m = 170 stays valid (Appendix B).

---

## 4. NEXIS configuration

`nexis(backward=False, terminal_filter=True, alpha=0.05, adjust="FWER", rho=0.5,
max_rounds=20)`, linear T × Z_j test, homoskedastic OLS, no clustering
(`scripts/realworld_final_runs.py`, `VARIANTS["new default"]`, run
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
*Why the terminal step and not the NeurIPS interleaved step:* the interleaved step tests
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
3 certified + 2 candidates and 1 + 1 for NEXIS. Source: no committed script prints these
counts yet; recomputed on 2026-09-26 with `realworld_clustered_nexis.plain_test()` at
S = ∅ on the pool of `realworld_clustered_nexis.uganda()` (71 and 45). See open issues.
*Reading:* ~8–9 false positives are expected under the global null; the excess comes from
correlated SAE atoms that proxy the same few modifiers, the experimental power paradox.

---

## 6. VLM interpretation

Qwen2.5-VL-72B-Instruct, 4-bit, one H100 (`src/apps/uganda/interpret.py`, wrapper
`scripts/uganda/slurm_interpret.sh`, pipeline `qwen72b`). Direct contrast: rank the 331
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

**Multilevel inference** (`scripts/realworld_uganda_groupcluster.py`, run 2, →
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
suggestive hypotheses. (`realworld_final_runs.py` repeats the RI with 1,999 permutations;
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

- Paper says Prithvi "layer 5"; the code uses the last block (Section 3).
- Paper says SAE batch size 256; the scripted default is 64 and the checkpoint's training
  call is not traceable (Section 3).
- The 71/170 and 45/170 marginal counts have no committed producer script (Section 5).
- The district-dummy p ≈ 7.7×10⁻⁵ comes from a June run with the NeurIPS configuration;
  the paper's algorithm gives certification p 7.9×10⁻⁵ (Section 2).
- Treated count 825 and sub-region list are consistent with the data but are not written
  by any paper-number script.

---

## 10. Run notes

- `scripts/realworld_final_runs.py` without `--only` runs both applications and writes
  `results/realworld_final/report.json`, the file the paper numbers were read from. The
  paper's Uganda rows are the runs `Wobs | published test | new default`.
- The script first checks that the NeurIPS configuration reproduces the published sets,
  read from `results/uganda/prithvi_l5_1024/<outcome>/nexis_result.json` (untracked; comes
  with the original SAE outputs).
- VLM labels: `src/apps/uganda/interpret.py` via `scripts/uganda/slurm_interpret.sh`
  (Section 6); add `--extra-atoms 261` for an atom outside the selected sets.
- `scripts/uganda/run.sh` and `reanalyze.sh` are the older multi-backbone pipeline that
  wrote the published sets; they are not needed for the paper numbers.
- `notebooks/uganda.ipynb` is exploratory and predates the paper's configuration.
