# CelebA semi-synthetic experiment brief

> **Purpose.** The CelebA benchmark as it stands in the final paper
> (`paper/iclr27/main.tex`, Section 3 and Figure 3; `appendix.tex`, Appendix C and the
> run statistics of Appendix B): design, the choices behind it, every number the paper
> quotes, and the file or script each number comes from. Paths are relative to the repo
> root; `results/` and `data/` are untracked. Top-level commands: `README.md`, section
> CelebA; the per-figure map is in Section 7.
> "NEXIS-v2" in file and method names is the paper's algorithm.

---

## 1. Pool and representation

**Pool.** The 19,867 images of the official CelebA validation split (aligned and
cropped), streamed from the Hugging Face mirror `flwrlabs/celeba`
(`src/apps/celeba/embed.py` and `iclr/` both pin revision
`2d738f56e0e7f925ea36ae7c808ea925264aacec`, the revision behind the paper's embeddings). The same images train the SAE, define the ground truth
and are the units the benchmark samples from.

**Encoder.** SigLIP 2 ViT-B/16, timm `vit_base_patch16_siglip_224.v2_webli`
(`src/apps/celeba/backbones.py`, pinned 2026-09-26; the untagged name resolved to the same
weights), 224 px, the 196 final-layer patch tokens (768-d), no attention-pooling head.
*Why patch tokens:* the SAE recipe follows Mencattini et al. (2025), which trains on
individual patch tokens (timm `forward_features`, as `backbones.py` notes).

**Dictionary.** TopK SAE (`overcomplete` 0.3.0: linear encoder, ReLU, TopK, unit-norm
dictionary), m = 9,216 = 12 × 768, trained on the 19,867 × 196 ≈ 3.89×10⁶ patch tokens:
20 epochs, batches of 20 images, Adam lr 5×10⁻⁴, gradient-norm clip 1, seed 0, MSE loss
(`src/apps/celeba/train_sae.py` via `scripts/celeba/submit_sae.sh`; checkpoints
`results/celeba/sae_siglip_k{5,20}.pt`). Each image is encoded from its mean patch token
into two views:
- **Z**: sparse post-TopK codes (near-orthogonal, L0 ≤ k), `data/celeba/embeddings/sae_k{5,20}.npy`;
- **Z_pre**: dense pre-activations before ReLU and TopK, `sae_precode_k{5,20}.npy`.
Main setting: k = 20 on Z. Ablations: k = 5 on Z; k = 20 on Z_pre.
*Why train on the benchmark images:* the SAE never sees labels, treatments or outcomes;
the replica below checks that this does not drive the results.

**Replica dictionary.** Same recipe (k = 20), trained on an independent random sample of
19,867 train-split images (seed 1; the CelebA splits share no identities), then used to
encode the same validation pool (`scripts/celeba/submit_resample_{embed,sae}.sh b1`;
`results/celeba/resample_b1/sae_siglip_k20.pt`, codes in
`data/celeba_resample_b1/eval/embeddings/`). Trained on one H100 (the others on an
RTX 2080 Ti). Background: `docs/celeba_sae_resampling.md`.

**Principal coordinates (ground truth S\*).** For each attribute, argmax over j of the
best-threshold F1 of Z_pre^j for the label, over all 19,867 images, computed once and held
fixed. *Why Z_pre:* continuous scores give a smoother, threshold-free alignment score.

| Dictionary | Wearing_Hat | Eyeglasses | Sideburns | Source |
|---|---|---|---|---|
| main, k = 20 (Z and Z_pre) | 5348 | 5537 | 1683 | `results/celeba/experiment/k20/{sae,sae_precode}/ground_truth.json`, `experiment_r3/k20/sae/ground_truth.json` |
| main, k = 5 | 7044 | 5732 | | `results/celeba/experiment/k5/sae/ground_truth.json` |
| replica, k = 20 | 197 | 4833 | | `results/celeba/experiment_resample_b1/k20/sae/ground_truth.json` |

These indices exist only for the original checkpoints (not in the repo): GPU training is
not bit-reproducible and a retrained SAE has other indices. To run `iclr/` on the original
artefacts, link them into its layout as `~/.cache/nexis_cap` did:
`data/celeba/embeddings/{siglip.npy, siglip_patches.npy → siglip_patches.f16, sae*_k*.npy}`,
`data/celeba_resample_b1/eval/embeddings/sae[_precode]_k20.npy → sae[_precode]_replica_k20.npy`,
`results/celeba/sae_siglip_k{5,20}.pt → models/sae_main_k{5,20}.pt`,
`results/celeba/resample_b1/sae_siglip_k20.pt → models/sae_replica_k20.pt`,
plus `labels.parquet` and `images.npy`; `prepare` then only writes the ground-truth files.

---

## 2. Data-generating process

For each unit: W_k ~ Bernoulli(p̂_k) independently at the CelebA prevalence,
T ~ Bernoulli(0.5), an image drawn without replacement from the pool cell matching
(W_1, …, W_r), and

  Y = Σ_k β_k W_k + T · [τ₀ + η · Σ_k γ_k W_k] + ε,  ε ~ N(0, 1), τ₀ = 0.5

(`src/apps/celeba/scm.py`, `generate_celeba_rct`). The seed fixes the whole draw, so all
methods, and all DGPs sampling the same attributes, see the same units.

| DGP | Modifiers (γ) | Prognostic β | S\* | Sweep |
|---|---|---|---|---|
| main, r = 2 | Wearing_Hat (+1, prevalence 4.7 %), Eyeglasses (−1, 7.0 %) | 0.3, −0.2 | {5348, 5537} | `submit_experiment_v2.sh` |
| r = 1 | Wearing_Hat (+1); Eyeglasses sampled, γ = 0 | 0.3, −0.2 | {5348} | `submit_dgp_extra_v2.sh r1` |
| r = 3 | + Sideburns (+1) | 0.3, −0.2, 0.3 | {5348, 5537, 1683} | `submit_dgp_extra_v2.sh r3` |
| r = 0 | none, τ = τ₀ | 0.3, −0.2 (fixed) | ∅ | `submit_dgp_extra_v2.sh r0`, n sweep only, 200 seeds |
| U-shape | τ = τ₀ + η (g₁(Z_pre^5348) − g₂(Z_pre^5537)) | 0.3, −0.2 | {5348, 5537} on Z_pre | `submit_experiment_v2_ushape.sh` (`--effect-form ortho_quadratic`) |
| violation | main DGP; Z^5348 split into U·Z and (1−U)·Z, U ~ Bern(0.5) per image | 0.3, −0.2 | {5348, 9216 (new), 5537} | `python iclr/run.py violation` |

g_k is the squared standardised pre-activation, residualised on (1, z) over the pool and
scaled to unit variance, so τ has zero covariance with each principal coordinate.
*Why prognostic effects in every DGP:* the CATE test targets the treatment interaction, so
a coordinate carrying only prognostic signal satisfies the null; r = 1 and r = 0 check it.
*Why r = 0 with fixed β and only an n sweep:* with γ = 0, η multiplies nothing and a larger
τ₀ leaves the interaction statistics unchanged, so the four DGP rows collapse to one; 200
seeds make the per-n false-discovery share readable against α (seeds 0–49 draw the main
setting's units). The earlier variant where η scales β is `experiment_v2_r0`, unused.
*Why Sideburns as the third modifier:* `src/apps/celeba/third_modifier_candidates.py` →
`results/celeba/figures_v2/third_modifier_candidates.md`. Attributes with a higher F1 are
high-prevalence labels whose sparse code almost never fires (e.g. Male, code recall 0.03)
or are weakly separated from the runner-up (Smiling); the better-aligned Blond_Hair and
Bangs have one image with hat and glasses, so the independent sampler fails even at
n = 50. Sideburns supports every n up to 10,000. Its code AUC is 0.805.

---

## 3. Grid, metrics, methods

**Grid.** η ∈ {1, …, 10} at n ∈ {500, 2000}; n ∈ {50, 100, 200, 350, 500, 750, 1000,
2000, 3500, 5000, 10000} at η ∈ {2, 5}; 50 seeds per cell (200 per n at r = 0). The U-shape
runs η at n = 2000 and n at η = 5 only.

**Metrics.** Precision, recall and IoU of Ŝ against S\*, seed means with ±1.96 SE bands.
*Macro* = mean over the 42 cells of an appendix figure (Figure 3: 21 cells). *First
0.95* = smallest grid value where the seed mean reaches 0.95.

**Methods.** Baselines: marginal testing of every coordinate alone with the same linear
interaction test, uncorrected, FDR (BH) or FWER (Bonferroni). NEXIS-v2 =
`nexis(rho=0.5, backward=False, terminal_filter=True)`, α = 0.05, FWER forward gate,
linear test, terminal gate α/m with m = 9,216 (`V2_METHODS` in
`src/apps/celeba/experiment.py`; one-axis variants such as `NEXIS-v2 (rho=0.8)`).
GCM/PCM nuisances cross-fitted with 3 folds (`--gcm-splits 3`); LightGBM with 50 trees of
depth ≤ 4 and ≤ 15 leaves (the `nexis()` defaults); the PCM combines the two split
directions as min{1, 2 min(p₁, p₂)} (default since `dd18f32`).
*Why marginal screening is the only baseline:* forests, meta-learners, causal trees and
rule ensembles return effects or subgroups, not a set of coordinates.
**max_rounds.** Every src v2 sweep ran with `--max-steps 10` (a cap of 10 forward
rounds), although the paper describes a forward step without a cap. The cap binds only
where the forward step would exceed 10 rounds (ablations without forward correction,
ρ ∈ {0, 0.2}, the U-shape tests, the violation; `iclr/README.md`, Setting); in the main
setting |S̃| ≤ 7 (Section 4.6).

---

## 4. Numbers quoted in the paper

### 4.1 Main setting (Figure 3, `dgp.pdf`)

Source: `results/celeba/figures_v2/dgp_extra.md` (tables for NEXIS-v2, main SAE) and
`results/celeba/iclr_local_runs/results/figures/summary.md` (identical, from `iclr/`).

| Quantity | NEXIS | Best marginal |
|---|---|---|
| macro precision / recall / IoU | 0.808 / 0.750 / 0.745 | FWER 0.345 / 0.759 / 0.300 |
| IoU at n = 2000, η = 5 | 1.000 (all 50 runs return exactly {5348, 5537}) | FWER 0.205 |
| first recall 0.95 | n = 750 (η = 5), n = 2000 (η = 2), η = 2 (n = 2000), η = 7 (n = 500) | |
| first precision 0.95 | n = 350 (η = 5), η = 4 (n = 500) | never |

Mean IoU 0.999 over η ∈ {3, …, 10} at n = 2000 (recomputed from
`results/celeba/experiment_v2/k20/sae/effect_sweep.parquet`: 0.99875).

### 4.2 Reproducibility: replica SAE (`replica_k20.pdf`)

`dgp_extra.md`: macro precision / recall / IoU 0.745 / 0.729 / 0.680 (main 0.808 / 0.750 /
0.745); IoU at n = 2000, η = 5 0.920 (main 1.000); best marginal macro precision 0.312
(FWER). The precision gap comes from replica coordinates that carry residual signal of the
same concepts and count as false discoveries against the coordinate-level S\*.

### 4.3 Principal Alignment (Table `tab:celeba_pa_screening`, `pa_spectrum.pdf`, `pa_top_images.pdf`)

Source: `results/celeba/figures_v2/principal_alignment/pa_summary.json`,
`pa_screening.{md,tex}`, `pa_screening_splits.csv` (`src/apps/celeba/alignment_appendix.py`).

- Sign-free AUC of the principal: 0.969 (hat), 0.956 (glasses); runner-up 0.755 (coord.
  2815), 0.714 (coord. 7706); gaps 0.21, 0.24.
- Top-100 purity of the principal: 90 % and 95 % (prevalences 4.7 %, 7.0 %).
- Coordinates with AUC ≥ 0.6: 9 (hat) and 11 (glasses) besides the principal.
- Screening-off probe (20 random 50/50 splits, L2 logistic C = 1, companions chosen on the
  training half): AUC from 0.970 to at most 0.988 (hat) and 0.956 to at most 0.989
  (glasses) with K ∈ {10, 50, 200} extra coordinates; the principal alone carries 94–96 %
  of the augmented probe's held-out log-loss reduction; the likelihood-ratio test still
  rejects exact conditional independence (p < 10⁻¹⁴ in every split; largest log10 p −14.8,
`lrt_log10p_max`). Principal Alignment
  therefore holds only approximately.

### 4.4 Controlled violation (`violation.pdf`)

Source: `results/celeba/iclr_local_runs/results/figures/{summary.md, violation_table.tex}`
(`python iclr/run.py violation`), η = 5. NEXIS selects both halves in 48 of 50 runs at
n = 2000 (share 0.96) and returns exactly {5348, 9216, 5537} in all 50 at n = 10⁴; in the
other 2 runs at n = 2000 the spectral gap stops the forward step before either half enters.
Marginal testing (FWER) keeps the three target coordinates but selects 47.9 coordinates on
average at n = 10⁴ (precision 0.06). NEXIS first reaches mean IoU 0.95 at n = 3500 (0.97)
against n = 750 without the split.

### 4.5 Ablations (Appendix C.4–C.5)

Thresholds recomputed 2026-09-26 from `results/celeba/experiment_v2/<tree>/{n,effect}_sweep.parquet`;
they match the paper. (The PCM rows of `results/celeba/figures_v2/comparison.md` predate
the calibrated 2·min PCM rule and are stale; the other rows agree.)

| Ablation | Paper statement |
|---|---|
| k = 5 (`model_k5.pdf`) | recall 0.95 unchanged (η = 2 at n = 2000; n = 750 at η = 5); precision needs η = 2 at n = 2000 and n = 5000 at η = 5 (main: n = 350) |
| Z_pre (`model_precode.pdf`) | recall at n = 750 (η = 5); precision n = 500 vs 350; FWER baseline selects ~600 coordinates (607 on average over all runs) |
| Test (`method_test.pdf`) | linear: recall 0.95 at n = 750 (η = 5), η = 2 (n = 2000); GCM (both): n = 2000, η = 4; PCM (both): n = 3500, never at n = 2000 (recall 0.60 / 0.59 at n = 2000, η = 5); at n = 500 recall ≤ 0.04 up to η = 10 |
| Test, U-shape (`tab:celeba_test_story`) | n\* (IoU ≥ 0.95 at η = 5) and IoU at n = 2000: linear 750 / 1.00 vs — / 0.00; GCM quad. 2000 / 0.98 vs — / 0.00; GCM LightGBM 2000 / 0.98 vs — / 0.00; PCM quad. 3500 / 0.60 vs 3500 / 0.77; PCM LightGBM 3500 / 0.59 vs 1000 / 1.00. Source `figures_v2/test_story.{md,tex}` |
| Correction (`method_adjust.pdf`) | none: precision 0.95 at η = 2 (n = 2000), n = 500 (η = 5); recall n = 1000 (vs 750), η = 8 (vs 7); FDR ≡ FWER |
| ρ (`method_rho.pdf`) | ρ = 0.8 reaches recall 0.95 only at η = 7 (n = 2000) |
| Backward (`method_backward.pdf`) | terminal step: precision 0.95 at n = 350 instead of 2000 (η = 5), ≥ 0.96 at n = 500 for η ≥ 4 (forward alone 0.72 at η = 10); recall n = 750 vs 500 (η = 5), η = 7 vs 5 (n = 500); the interleaved step changes no run |

### 4.6 More modifiers and none (Appendix C.6)

Source: `dgp_extra.md` (macro metrics, first 0.95, r = 0 table; `dgp_r0_table.tex`) and
`results/celeba/per_modifier/r{1,2,3}.parquet` (`per_modifier_recall.py`).

- **r = 1:** macro recall / IoU 0.749 / 0.734 (r = 2: 0.750 / 0.745); IoU at n = 2000, η = 5
  1.000; macro precision 0.734 (0.808). Per-modifier macro recall of Wearing_Hat 0.713 at
  r = 2 (Eyeglasses 0.787) and 0.749 at r = 1. Coordinate 5537 is selected in none of the
  2,100 runs.
- **r = 3:** recall 0.95 at n = 1000 (η = 5) and n = 3500 (η = 2), never at n = 500; macro
  precision 0.761, recall 0.660, IoU 0.644; IoU at n = 2000, η = 5 0.953; per-modifier
  macro recall Sideburns 0.597, Wearing_Hat 0.657, Eyeglasses 0.726.
- **r = 0 (Table `tab:celeba_r0`):** share of runs with any false discovery ≤ 0.015 at every
  n for NEXIS (0.010 at n = 750, 0.015 at 1000, 0.005 at 3500), 0.003 pooled over 2,200
  runs; uncorrected marginal 0.995–1.000; NEXIS errs in exactly the runs where marginal
  FWER does (its first forward step is that test).
- **Run statistics (Appendix B):** over the 6,300 runs of the three dictionaries at
  ρ = 0.5, |S̃| median 2, max 7; the forward step alone on the main setting also median 2,
  max 7; terminal tests median 4, max 192 (bound 7·2⁶ = 448). Source:
  `src/apps/celeba/run_statistics.py` → `results/celeba/paper_numbers/run_statistics.md`
  (inputs from `src/apps/celeba/ablation_rho_filter.py`).

### 4.7 Compute (Table `tab:celeba_compute`)

SigLIP embedding ~10 min and SAEs ~1 h on one GPU; principal coordinates and alignment
tests minutes on CPU; all sweeps ~400 CPU-h (~10 h on 40 cores). Measured per block in
`iclr/README.md` (Compute).

---

## 5. Design choices and why (Appendix B and C)

- **Linear test by default.** The CATE is linear in the binary attributes and the principal
  coordinates act as near-indicators, so the linear test is close to well specified and the
  most sample-efficient. GCM pays for flexibility the DGP does not reward and is blind to
  zero-covariance (U-shaped) modifiers; the PCM sees them but halves the sample. Use the GCM
  for nonlinear monotone heterogeneity, the PCM when non-monotone heterogeneity cannot be
  ruled out and n is large.
- **FWER forward gate.** FDR and FWER coincide here (the BH rejection set at the leading
  discoveries is a singleton); FWER is the more conservative. No correction delays recall.
- **ρ = 0.5.** Lower ρ admits correlated companions (precision loss at large n or η);
  ρ = 0.8 blocks the weaker principal. Recommend a sweep over {0.2, 0.5, 0.8} in applications.
- **Terminal backward step.** Carries the precision guarantee (tests the fixed null
  H0(j | S\*) on the recall event); costs at most |S̃|·2^(|S̃|−1) tests, small here.
  The interleaved step is kept only as an option for more entangled settings.
- **k = 20 and sparse codes Z.** k = 5 spreads attributes over correlated coordinates
  (precision later); Z_pre is dense and correlated (more data needed).
- **Principal coordinate by F1 on Z_pre** (Section 1) and a **replica SAE** on
  identity-disjoint images to rule out training-set effects.

---

## 6. Open issues

- The paper describes the forward step without a cap; the src sweeps used 10 rounds
  (Section 3). `iclr/` has no cap by default and applies 10 only in the listed runs.
- Of `iclr/`, only `main` and `violation` have been checked against the paper; the other
  blocks are expected to match but are unverified.
- `results/celeba/figures_v2/comparison.md` has stale PCM rows (Section 4.5).

---

## 7. Commands and figure map

The commands are in `README.md` (section CelebA). The paper's CelebA numbers and figures
come from the `src/` chain ("NEXIS-v2" = `nexis(rho=0.5, backward=False,
terminal_filter=True)`), except `violation.pdf`, which comes from `iclr/`. Of `iclr/`,
only `main` and `violation` are checked against the paper: `main` matches the `src`
main-setting runs run by run (`results/celeba/paper_numbers/run_statistics.md`). The
figures are copied under the same name to `paper/iclr27/figures/`.

| Paper item | File in `results/celeba/` | Made by |
|---|---|---|
| Figure 3 | `figures_v2/figure_main.pdf` | `figure_main.py --nexis-key NEXIS-v2` |
| `dgp`, `model_k5`, `model_precode`, `method_{test,adjust,rho,backward}` | `figures_v2/*.pdf` | `figure_appendix.py --variant v2` |
| `replica_k20`, `dgp_r1`, `dgp_r3`, r = 0 table | `figures_v2/` (`dgp_r0_table.tex`, `dgp_extra.md`) | `figure_dgp_extra.py` |
| Test comparison table | `figures_v2/test_story.tex` | `figure_test_story_v2.py` |
| `pa_spectrum`, `pa_top_images`, screening table | `figures_v2/principal_alignment/` | `alignment_appendix.py` |
| Choice of the r = 3 modifier | | `interleaved_backward_align.py sae_precode_k20`, then `third_modifier_candidates.py` |
| Run statistics (Appendix B) | `paper_numbers/run_statistics.md` | `run_statistics.py` |
| `violation` | `iclr_local_runs/results/figures/violation.pdf` | `python iclr/run.py violation`, run in `iclr/` and its outputs moved there |

**Ground truth.** The sweeps read the principal coordinates from `ground_truth.json` files
written by earlier runs (`--gt-json results/celeba/experiment{,_r3,_resample_b1,_ushape}/…`,
untracked). Without them, drop `--gt-json`: `run_experiment.py` then recomputes the same
rule (argmax best-threshold F1 on Z_pre, Section 1).

**Not in the paper.** The DINOv2 backbone ablation (`backbone=dinov2` in every stage,
`compare_backbones.py`), the NeurIPS-era sweeps (`submit_experiment.sh`,
`submit_resample_experiment.sh`, `run_experiment_*.sh`) and `notebooks/celeba.ipynb`.
