# CelebA: semi-synthetic benchmark

> **Purpose.** The CelebA benchmark as it stands in the paper (Section 3 and Figure 3;
> Appendix C and the run statistics of Appendix B): design, the choices behind it, every number the paper
> quotes, and the file or script each number comes from. Paths are relative to the repo
> root; `results/` and `data/` are untracked. Top-level commands: `README.md`, section
> CelebA; the per-figure map is in Section 7. Section 8 checks the robustness of every
> experiment to the SAE training sample. The self-contained package in `benchmark/`
> reproduces every experiment with one command per block (its own `README.md`).
> "NEXIS-v2" in file and method names is the paper's algorithm.

---

## 1. Pool and representation

**Pool.** The 19,867 images of the official CelebA validation split (aligned and
cropped), streamed from the Hugging Face mirror `flwrlabs/celeba`
(`src/apps/celeba/embed.py` and `benchmark/config.py` both pin revision
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
RTX 2080 Ti). Background: Section 8.

**Principal coordinates (ground truth S\*).** For each attribute, argmax over j of the
best-threshold F1 of Z_pre^j for the label, over all 19,867 images, computed once and held
fixed. *Why Z_pre:* continuous scores give a smoother, threshold-free alignment score.

| Dictionary | Wearing_Hat | Eyeglasses | Sideburns | Source |
|---|---|---|---|---|
| main, k = 20 (Z and Z_pre) | 5348 | 5537 | 1683 | `results/celeba/experiment/k20/{sae,sae_precode}/ground_truth.json`, `experiment_r3/k20/sae/ground_truth.json` |
| main, k = 5 | 7044 | 5732 | | `results/celeba/experiment/k5/sae/ground_truth.json` |
| replica, k = 20 | 197 | 4833 | | `results/celeba/experiment_resample_b1/k20/sae/ground_truth.json` |

These indices exist only for the original checkpoints (not in the repo): GPU training is
not bit-reproducible and a retrained SAE has other indices. To run the benchmark package
(`src/apps/celeba/benchmark/`) on the original artefacts, link them into its data layout:
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
| violation | main DGP; Z^5348 split into U·Z and (1−U)·Z, U ~ Bern(0.5) per image | 0.3, −0.2 | {5348, 9216 (new), 5537} | `benchmark/run.py violation` |

g_k is the squared standardised pre-activation, residualised on (1, z) over the pool and
scaled to unit variance, so τ has zero covariance with each principal coordinate.
*Why prognostic effects in every DGP:* the CATE test targets the treatment interaction, so
a coordinate carrying only prognostic signal satisfies the null; r = 1 and r = 0 check it.
*Why r = 0 with fixed β and only an n sweep:* with γ = 0, η multiplies nothing and a larger
τ₀ leaves the interaction statistics unchanged, so the four DGP rows collapse to one; 200
seeds make the per-n false-discovery share readable against α (seeds 0–49 draw the main
setting's units). An earlier variant where η scales β is not used.
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
ρ ∈ {0, 0.2}, the U-shape tests, the violation; `benchmark/README.md`, Setting); in the main
setting |S̃| ≤ 7 (Section 4.6).

---

## 4. Numbers quoted in the paper

### 4.1 Main setting (Figure 3, `dgp.pdf`)

Source: `results/celeba/figures_v2/dgp_extra.md` (tables for NEXIS-v2, main SAE) and
`results/celeba/benchmark/results/figures/summary.md` (identical, from `benchmark/`).

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

Source: `results/celeba/benchmark/results/figures/{summary.md, violation_table.tex}`
(`python run.py violation` in `benchmark/`), η = 5. NEXIS selects both halves in 48 of 50 runs at
n = 2000 (share 0.96) and returns exactly {5348, 9216, 5537} in all 50 at n = 10⁴; in the
other 2 runs at n = 2000 the spectral gap stops the forward step before either half enters.
Marginal testing (FWER) keeps the three target coordinates but selects 47.9 coordinates on
average at n = 10⁴ (precision 0.06). NEXIS first reaches mean IoU 0.95 at n = 3500 (0.97)
against n = 750 without the split.

### 4.5 Ablations (Appendix C.4–C.5)

Thresholds recomputed 2026-09-26 from `results/celeba/experiment_v2/<tree>/{n,effect}_sweep.parquet`;
they match the paper.

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
`benchmark/README.md` (Compute).

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

- Of the `benchmark/` blocks, only `main` and `violation` have been checked against the
  paper; the other blocks are expected to match but are unverified.

---

## 7. Commands and figure map

The commands are in `README.md` (section CelebA). The paper's CelebA numbers and figures
come from the `src/` chain ("NEXIS-v2" = `nexis(rho=0.5, backward=False,
terminal_filter=True)`), except `violation.pdf`, which comes from `benchmark/`. Of
`benchmark/`, only `main` and `violation` are checked against the paper: `main` matches the
`src` main-setting runs run by run (`results/celeba/paper_numbers/run_statistics.md`). The
figures are copied under the same name into the paper's figure folder.

| Paper item | File in `results/celeba/` | Made by |
|---|---|---|
| Figure 3 | `figures_v2/figure_main.pdf` | `figure_main.py --nexis-key NEXIS-v2` |
| `dgp`, `model_k5`, `model_precode`, `method_{test,adjust,rho,backward}` | `figures_v2/*.pdf` | `figure_appendix.py --variant v2` |
| `replica_k20`, `dgp_r1`, `dgp_r3`, r = 0 table | `figures_v2/` (`dgp_r0_table.tex`, `dgp_extra.md`) | `figure_dgp_extra.py` |
| Test comparison table | `figures_v2/test_story.tex` | `figure_test_story_v2.py` |
| `pa_spectrum`, `pa_top_images`, screening table | `figures_v2/principal_alignment/` | `alignment_appendix.py` |
| Choice of the r = 3 modifier | | `interleaved_backward_align.py sae_precode_k20`, then `third_modifier_candidates.py` |
| Run statistics (Appendix B) | `paper_numbers/run_statistics.md` | `run_statistics.py` |
| `violation` | `benchmark/results/figures/violation.pdf` | `python run.py violation` in `src/apps/celeba/benchmark/` (`--data-dir`/`--results-dir` point it at `results/celeba/benchmark/`) |

**Ground truth.** The sweeps read the principal coordinates from `ground_truth.json` files
written by earlier runs (`--gt-json results/celeba/experiment{,_r3,_resample_b1,_ushape}/…`,
untracked). Without them, drop `--gt-json`: `run_experiment.py` then recomputes the same
rule (argmax best-threshold F1 on Z_pre, Section 1).

**Not in the paper.** The DINOv2 backbone ablation (`backbone=dinov2` in every stage,
`compare_backbones.py`) and the earlier sweeps (`submit_experiment.sh`,
`submit_resample_experiment.sh`, `run_experiment_*.sh`).

---

## 8. Robustness of every experiment to the SAE training sample

> **Status.** An earlier, more extensive analysis than the paper's replica figure. The
> replica dictionary built here (steps 1-3 of §8.7) and its ground truth feed the
> reproducibility figure `replica_k20.pdf` (`scripts/celeba/submit_dgp_extra_v2.sh`, then
> `src/apps/celeba/figure_dgp_extra.py`).
> The agreement analyses (§8.1, §8.4-§8.6: `sae_agreement.py`, `agreement_rates.py`,
> `concept_agreement.py`, `submit_concept_agreement.sh`) are not in the paper and were
> removed from the tree; they are at git tag `pre-cleanup-2026-09`.

**Question.** The paper's CelebA experiments all rest on two TopK SAEs (k=5, k=20) trained
on one particular corpus. If those dictionaries had been learned from a *different* sample
of the same size, would the reported conclusions change?

**Answer.** Recall is essentially unchanged — averaged over all 12 methods the paired
difference is within ±1.2 pp in three of the four configs, and +7.7 pp *in arm B's favour*
on k5/z_pre. Index-level precision drops
(−5 pp at k=20, −27 pp at k=5 at the main design point) for a single, identifiable and
*a-priori diagnosable* reason: the independently trained dictionary splits the Eyeglasses
concept over two coordinates, and the ground-truth set S\* — defined as the F1-argmax
coordinate per attribute — names only one of them. Scored against the concept set instead
of the coordinate indices, the two dictionaries are statistically indistinguishable
(k=20: 0.975 vs 1.000; k=5: 0.991 vs 0.991). Every recovered feature matches a feature of
the other dictionary with the same semantics (concept agreement 1.00) and near-identical
CATE profiles (0.92–1.00). The leakage diagnostic ε̂, computable from the dictionary alone
before any experiment is run, orders the four dictionaries exactly as their precision does.

---

### 8.1 The reported table (main setting: k=20, sparse codes z, NEXIS, n=2000, η=5, 50 seeds)

| SAE training                | Precision | Recall | IoU | Concept agreement | Matched-CATE correlation |
| --------------------------- | --------: | -----: | --: | ----------------: | -----------------------: |
| Original training sample    | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |
| Independent training sample | 0.893 ± 0.028 | 1.000 ± 0.000 | 0.893 ± 0.028 | 1.00 | 1.00 |

Same 50 selections per arm, scored against the concept set {Wearing_Hat, Eyeglasses}
instead of the F1-argmax coordinate indices (removes the split-coordinate artefact):

| SAE training                | Precision | Recall | IoU |
| --------------------------- | --------: | -----: | --: |
| Original training sample    | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Independent training sample | 0.975 ± 0.011 | 1.000 ± 0.000 | 0.963 ± 0.016 |

And the k=5 ablation (same design point):

| SAE training                | Precision | Recall | IoU | Concept agreement | Matched-CATE correlation | Precision (concept) | IoU (concept) |
| --------------------------- | --------: | -----: | --: | ----------------: | -----------------------: | ------------------: | ------------: |
| Original training sample    | 0.935 ± 0.024 | 1.000 ± 0.000 | 0.935 ± 0.024 | — | — | 0.991 ± 0.006 | 0.987 ± 0.009 |
| Independent training sample | 0.666 ± 0.019 | 1.000 ± 0.000 | 0.666 ± 0.019 | 1.00 | 0.92 | 0.991 ± 0.006 | 0.987 ± 0.009 |

Mean number of features selected per run: k=20 — 2.00 (original) vs 2.44 (independent);
k=5 — 2.26 vs 3.12. |S\*| = 2 in every dictionary.

### 8.2 Design

| | Original training sample (arm A) | Independent training sample (arm B) |
|---|---|---|
| SAE training corpus | 19,867 CelebA **valid**-split images × 196 SigLIP patches | 19,867 CelebA **train**-split images × 196 SigLIP patches, drawn at random (seed 1) |
| Corpus overlap | — | none (CelebA splits are identity-disjoint) |
| Architecture / schedule | TopK SAE, hidden 9,216, 20 epochs, batch 20 images, lr 5e-4, init seed 0 | identical |
| Encoded (experiment) data | valid-split mean-pooled SigLIP embeddings | **the same file** (byte-identical, symlinked and verified) |
| Design grid | 10 η values × {n=500, 2000}; 11 n values × {η=2, 5}; 50 seeds; 12 methods | identical |
| Monte Carlo draws | seeds 0–49 | identical seeds → identical datasets |

Only the SAE training corpus changes. The experiment data, the DGP, the design grid and
the Monte Carlo draws are held fixed, which makes every comparison **paired at the
(grid point, seed) level**: `generate_celeba_rct` seeds the image draw, treatment and
outcome noise from `seed` alone and never from the features, so at every cell the two arms
see the *same* units, the same T and the same Y — only Z differs. The concept-agreement
script asserts this equality run-by-run (`arms disagree on the sampled dataset` would raise).

**Exactness of the parallel rerun.** The rerun is sharded 96 ways
(config × sweep × fixed value × method group × seed block). To prove the sharding does not
perturb anything, the identical pipeline was run over arm A's own features and compared to
the stored paper results: **0 differing cells out of 18,000** (3,000 rows × 6 metrics), same
S\*, F1 spectra equal to 1e-4. NEXIS and the GCM variants take all randomness from explicit
`random_state`, so results are independent of how work is split across processes.

### 8.3 Cost

| Stage | Hardware | Wall clock |
|---|---|---|
| Embed 19,867 train-split images (SigLIP, patches cached, 5.7 GB) | 1× H100 | **1.5 min** |
| Train SAE k=5 and k=20 + encode the valid split | 2× H100 (parallel) | **6 min** |
| Rerun all experiments: 4 configs × 2 sweeps × 12 methods × 50 seeds | 96 CPU jobs (2,560 cores) | **1 h 38 min** |
| Merge + agreement metrics + concept agreement + ε̂ + all figures | CPU | **~15 min** |
| **Total** | | **≈ 2 h** |

An earlier draft of the paper estimated ~8 h per k for SAE training plus ~1–2 days for the
sweeps; the H100 and the 96-way sharding are what bring it inside a single sitting. (That
draft's compute table — 8 h/SAE, m=13,824, 729 patches, 1,152-d — did not describe the runs
actually on disk: the checkpoints are 768→9,216 with 196 patches and train in ~6 min. The
paper's compute table now reports the measured runs, Section 4.7.)

### 8.4 Metric definitions

**Concept agreement.** Each recovered feature is described by its top-M activating images
(M=100) over the same 19,867 evaluation images and labelled with the CelebA attribute those
images share — the attribute maximising top-M *lift* subject to purity ≥ 0.30, with CelebA's
40 binary attributes serving as the semantic vocabulary in place of a VLM caption. Features
are matched across dictionaries by top-M image-set Jaccard (greedy, highest overlap first,
floor 0.05). Concept agreement is the share of arm-B recoveries whose matched arm-A partner
carries the same label, weighted by how often the feature was actually recovered. This is
the natural index-free comparison: two SAEs have no shared coordinate system, but their
features can be identified by what they fire on.

**Matched-CATE correlation.** For a matched pair (j_A, j_B) and each seed, the per-unit CATE
implied by that feature is fitted in each arm — τ̂_i = τ̂_T + β̂_TZ · Z_ij from
Y ~ 1 + T + Z_j + T·Z_j — and the two profiles are correlated across units. Because both
arms see the same units, this is a paired, unit-level comparison of the recovered
heterogeneity map. Reported as the Fisher-z mean over pairs and seeds.

**Sweep-level agreement** (`sae_agreement.py`, all 12 methods × 4 configs × both sweeps):
paired difference Δ = mean(m_B − m_A) with a seed-clustered bootstrap CI; TOST equivalence at
δ = 5 pp; curve discrepancy MAD = mean_g|D(g)| and max_g|D(g)|; and mean |t| = the
disagreement in units of the Monte Carlo standard error of the paper's own curves (SE floored
at 0.2 pp so near-deterministic cells do not produce meaningless t's).

### 8.5 Results across all experiments

Aggregated over all 12 methods and both sweeps (mean over cells):

| config | metric | mean Δ (B−A) | mean MAD | worst \|D\| | mean \|t\| | TOST-equivalent cells |
|---|---|---|---|---|---|---|
| k20/sae (main)   | recall | −0.012 | 0.013 | 0.100 | 0.6 | 88% |
| k20/sae (main)   | precision | −0.048 | 0.050 | 0.454 | 1.8 | 44% |
| k20/sae_precode  | recall | +0.001 | 0.012 | 0.170 | 0.5 | 94% |
| k20/sae_precode  | precision | +0.002 | 0.014 | 0.111 | 0.5 | 96% |
| k5/sae           | recall | −0.007 | 0.013 | 0.100 | 0.4 | 98% |
| k5/sae           | precision | −0.079 | 0.085 | 0.479 | 3.3 | 27% |
| k5/sae_precode   | recall | +0.077 | 0.079 | 0.470 | 1.5 | 29% |
| k5/sae_precode   | precision | +0.050 | 0.053 | 0.424 | 1.2 | 33% |

#### Agreement rates (preferred framing — averages over heterogeneous design cells are hard to read)

Each replication is a (design cell, seed) pair where both dictionaries see the *identical*
dataset, so their IoUs are directly comparable. The agreement rate is the share of
replications with |IoU_B − IoU_A| ≤ τ. It is calibrated against a **same-dictionary
reference**: the paper's own dictionary compared with itself across two different Monte Carlo
draws of the same cell (seed s vs seed s+25). That reference is the agreement one gets from
sampling noise alone with the dictionary held fixed.

| SAE | replications | comparable IoU (\|Δ\|≤0.1) | same-dictionary reference | mean \|ΔIoU\| | reference |
|---|---:|---:|---:|---:|---:|
| k=20, z (main) | 2,100 | **0.78** | 0.70 | 0.096 | 0.140 |
| k=20, z_pre | 2,100 | **0.89** | 0.73 | 0.046 | 0.135 |
| k=5, z | 2,100 | 0.47 | 0.63 | 0.198 | 0.168 |
| k=5, z_pre | 2,100 | **0.69** | 0.57 | 0.178 | 0.317 |

In three of the four dictionaries — including the main setting — swapping the training corpus
perturbs the recovered set *less* than redrawing the Monte Carlo sample does with the corpus
fixed. The exception is k=5 on sparse codes, the one dictionary whose ε̂ = 0.637 exceeds
ρ = 0.5, and whose disagreement is the split Eyeglasses coordinate (§8.6); scored at concept
level that config is identical across arms (0.991 vs 0.991 precision, 0.987 vs 0.987 IoU).

Seed-averaged (curve-level, 42 cells) the ordering reverses, as it must: averaging 50 seeds
removes the Monte Carlo noise that dominates single runs and leaves the systematic dictionary
difference, so 0.69 (k=20, z) / 1.00 (k=20, z_pre) of cells agree within 0.1 IoU against
references of 0.98 / 0.93. Both levels were written to `agreement_rates_iou.md` by
`agreement_rates.py --tag b1` (`--metric precision|recall` for the other metrics; the script
is at git tag `pre-cleanup-2026-09`).

Conclusion-level agreement (NEXIS vs the FWER baseline, the paper's headline claim):

* **Sign-flip rate of the precision gap: 0.00–0.30**, and 0.00 in the main setting at both
  n=500 and n=2000 — the ordering "NEXIS dominates every marginal baseline on precision" is
  never reversed.
* **Kendall τ between method rankings: 0.76–0.99** (mean ≈ 0.89 for precision, 0.89 for
  recall) — the ablation *orderings* the appendix reports survive.
* **Detection thresholds** (smallest n or η at which NEXIS reaches recall 0.9) agree within
  a factor 0.96–1.27 in the k=20 configs; on k5/sae_precode arm B is *faster* (ratio
  0.29–0.49), i.e. resampling helped there.
* Every qualitative ablation finding is reproduced on arm B: k=5 gives the same recall with
  lower precision; z_pre needs more data; linear dominates GCM; FDR ≡ FWER; ρ=0.5 jointly
  optimal; the backward step is neutral (the appendix figures regenerated on arm B, step 5
  of §8.7).

### 8.6 Why index-level precision drops — and why it is predictable

The two dictionaries are equally *selective*: the principal coordinate's best-threshold F1 is
0.871 / 0.943 (arm A, W1/W2) vs 0.868 / 0.940 (arm B) at k=20, and 0.864 / 0.912 vs
0.855 / 0.896 at k=5. What differs is how many coordinates carry the concept.

In arm B, coordinate **2608** is a second Eyeglasses feature: 99% of its top-100 images wear
glasses (lift 14.2). NEXIS selects it in 12/50 runs at k=20 and 45/50 at k=5. It is a
correct discovery of a real modifier, but S\* names only the F1-argmax coordinate, so it
scores as a false positive. That is exactly the "target misspecification, not algorithmic
error" case discussed under Principal Alignment (Section 4.3, `principal_alignment.py`).

The leakage diagnostic ε̂ = max_{j∉S\*} c_j / min_k c_{j_k} identifies this **before** any
experiment is run, and orders the four dictionaries exactly as their precision does:

| dictionary | ε̂ (vs ρ = 0.5) | max-leak coordinate | precision at the main point |
|---|---|---|---|
| A, k=20 | **0.183** ✓ | 7706 | 1.000 |
| A, k=5  | **0.345** ✓ | 5537 | 0.935 |
| B, k=20 | **0.413** ✓ | **2608** | 0.893 |
| B, k=5  | **0.637** ✗ (> ρ) | **2608** | 0.666 |

The one dictionary whose ε̂ exceeds ρ is the one whose index-level precision degrades
materially — and its "false positives" are the leaking coordinate itself. Raising ρ, or
scoring against concepts rather than coordinates, restores agreement.

### 8.7 Reproduction

```bash
# 1. same-size, disjoint SAE training corpus (H100, ~2 min)
sbatch scripts/celeba/submit_resample_embed.sh b1 1 19867 siglip

# 2. two SAEs, trained on that corpus, encoding the ORIGINAL valid embeddings (2× H100, ~6 min)
sbatch scripts/celeba/submit_resample_sae.sh  5 b1 siglip
sbatch scripts/celeba/submit_resample_sae.sh 20 b1 siglip

# 3. rerun every experiment, 96-way sharded (~1.5 h)
bash scripts/celeba/submit_resample_experiment.sh b1
python src/apps/celeba/merge_shards.py --tag b1          # validates completeness

# 4. agreement analysis (scripts at git tag pre-cleanup-2026-09)
python src/apps/celeba/sae_agreement.py --tag b1                             # sweeps, all methods
sbatch scripts/celeba/submit_concept_agreement.sh 20 b1 2000 5.0             # concept + CATE, k=20
sbatch scripts/celeba/submit_concept_agreement.sh  5 b1 2000 5.0             # concept + CATE, k=5

# 5. paper figures on the resampled dictionaries
python src/apps/celeba/figure_appendix.py \
    --experiment-dir results/celeba/experiment_resample_b1 \
    --out-dir        results/celeba/appendix_resample_b1

# optional: proof that the 96-way sharding is exact (reruns arm A through the same pipeline)
sbatch --cpus-per-task=40 --mem=100G \
    scripts/celeba/run_resample_shard.sh a0 sae 20 effect 2000 g1 0 50
```

Artifacts: the new checkpoints `results/celeba/resample_b1/sae_siglip_k{5,20}.pt`. The
agreement outputs (REPORT.md, per-cell CSVs, `agreement_curves.png`, concept-agreement
tables, `matches.csv`, `selections.csv`, top-activating contact sheets and the arm A/B
leakage diagnostics) and the regenerated appendix figures of step 5 are not kept in the
working tree; rerun steps 4-5 to recreate them (the `brief.md` that `figure_appendix.py`
writes there is hard-coded prose from the figure template, not arm-B facts).

### 8.8 Caveats

* One resample (one alternative corpus). It establishes that the conclusions are not an
  artefact of the particular training sample, not a full distribution over corpora; a second
  seed would let arm B be compared to an equally out-of-sample twin.
* Arm A's SAEs are trained on the same images the experiments draw from, arm B's are not, so
  the contrast mixes "different sample" with "in-sample vs out-of-sample". The matched F1
  values (0.871/0.943 vs 0.868/0.940) suggest this is not what drives the results.
* Concept labels come from CelebA's 40 attributes, not from a VLM. The vocabulary is
  therefore closed: a feature encoding something CelebA does not annotate is labelled by its
  closest annotated correlate or left `unlabelled`. Contact sheets are written for every
  recovered feature so the labels can be audited or replaced by VLM captions.
* Concept agreement and matched-CATE correlation are computed at the main design point
  (n=2000, η=5), because the sweep parquets record only counts, not selected indices.
* **The agreement level is a property of this task's difficulty, not of the method alone.**
  Wearing_Hat and Eyeglasses are visually salient, spatially localised, high-contrast
  attributes: any competent encoder + TopK SAE allocates a crisp coordinate to them, which is
  precisely why two independently trained dictionaries can be matched at all. The evidence is
  visible inside our own run: every feature that matched across dictionaries has top-100 lift
  ≥ 12.6 (purity ≥ 0.69), whereas the arm-B recoveries with no counterpart are mostly low-lift,
  diffuse concepts (median lift 2.0 at k=20 — Attractive, Male, Pointy_Nose — each selected in
  1/50 runs). For latent modifiers that are subtle, distributed, or entangled with other
  factors, a resampled dictionary is far less likely to devote a dedicated coordinate to them,
  and both concept matching and recovery should be expected to degrade. These numbers therefore
  support the salient-modifier regime and should not be read as a general guarantee.
* Exact dictionary identifiability is nevertheless *not* what NEXIS requires. The two
  dictionaries here index entirely different coordinates (S\*_A = {5348, 5537} vs
  S\*_B = {197, 4833}) and still recover the same concepts with the same CATE profiles: what
  the theory needs is approximate alignment (ε̂ < ρ), not a shared basis. Recent work toward
  identifiable SAEs argues that this weaker, more realistic regime is the attainable one
  [3], which is the setting our results occupy.

  [3] Nelson W., Karaletsos T., Locatello F., *Toward Identifiable Sparse Autoencoders*,
  ICML 2026. (Citation reproduced as supplied; not independently verified here.)
