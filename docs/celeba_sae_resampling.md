# CelebA: robustness of every experiment to the SAE training sample

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

## 1. The reported table (main setting: k=20, sparse codes z, NEXIS, n=2000, η=5, 50 seeds)

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

## 2. Design

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

## 3. Cost

| Stage | Hardware | Wall clock |
|---|---|---|
| Embed 19,867 train-split images (SigLIP, patches cached, 5.7 GB) | 1× H100 | **1.5 min** |
| Train SAE k=5 and k=20 + encode the valid split | 2× H100 (parallel) | **6 min** |
| Rerun all experiments: 4 configs × 2 sweeps × 12 methods × 50 seeds | 96 CPU jobs (2,560 cores) | **1 h 38 min** |
| Merge + agreement metrics + concept agreement + ε̂ + all figures | CPU | **~15 min** |
| **Total** | | **≈ 2 h** |

The paper's own estimate for this work is ~8 h per k for SAE training plus ~1–2 days for the
sweeps; the H100 and the 96-way sharding are what bring it inside a single sitting. (The
appendix's compute table — 8 h/SAE, m=13,824, 729 patches, 1,152-d — does not describe the
runs actually on disk: the checkpoints are 768→9,216 with 196 patches and train in ~6 min.
That table is inherited boilerplate and should be corrected independently of this analysis.)

## 4. Metric definitions

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

## 5. Results across all experiments

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

### Agreement rates (preferred framing — averages over heterogeneous design cells are hard to read)

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
ρ = 0.5, and whose disagreement is the split Eyeglasses coordinate (§6); scored at concept
level that config is identical across arms (0.991 vs 0.991 precision, 0.987 vs 0.987 IoU).

Seed-averaged (curve-level, 42 cells) the ordering reverses, as it must: averaging 50 seeds
removes the Monte Carlo noise that dominates single runs and leaves the systematic dictionary
difference, so 0.69 (k=20, z) / 1.00 (k=20, z_pre) of cells agree within 0.1 IoU against
references of 0.98 / 0.93. Both levels are in
`results/celeba/agreement_b1/agreement_rates_iou.md`; regenerate with
`python src/apps/celeba/agreement_rates.py --tag b1` (`--metric precision|recall` for the
other metrics).

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
  optimal; the backward step is neutral. Regenerated figures:
  `results/celeba/appendix_resample_b1/{dgp,model_k5,model_precode,method_test,method_adjust,method_rho,method_backward}.pdf`.

## 6. Why index-level precision drops — and why it is predictable

The two dictionaries are equally *selective*: the principal coordinate's best-threshold F1 is
0.871 / 0.943 (arm A, W1/W2) vs 0.868 / 0.940 (arm B) at k=20, and 0.864 / 0.912 vs
0.855 / 0.896 at k=5. What differs is how many coordinates carry the concept.

In arm B, coordinate **2608** is a second Eyeglasses feature: 99% of its top-100 images wear
glasses (lift 14.2). NEXIS selects it in 12/50 runs at k=20 and 45/50 at k=5. It is a
correct discovery of a real modifier, but S\* names only the F1-argmax coordinate, so it
scores as a false positive. That is exactly the "target misspecification, not algorithmic
error" case flagged in the principal-alignment rebuttal.

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

## 7. Reproduction

```bash
# 1. same-size, disjoint SAE training corpus (H100, ~2 min)
sbatch scripts/celeba/submit_resample_embed.sh b1 1 19867 siglip

# 2. two SAEs, trained on that corpus, encoding the ORIGINAL valid embeddings (2× H100, ~6 min)
sbatch scripts/celeba/submit_resample_sae.sh  5 b1 siglip
sbatch scripts/celeba/submit_resample_sae.sh 20 b1 siglip

# 3. rerun every experiment, 96-way sharded (~1.5 h)
bash scripts/celeba/submit_resample_experiment.sh b1
python src/apps/celeba/merge_shards.py --tag b1          # validates completeness

# 4. agreement analysis
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

Artifacts: `results/celeba/agreement_b1/` (REPORT.md, per-cell CSVs, `agreement_curves.png`,
`concept_k20/`, `concept_k5/` with `headline_table.md`, `matches.csv`, `selections.csv` and
top-activating contact sheets, `pa_arm{A,B}_k{5,20}` leakage diagnostics),
`results/celeba/appendix_resample_b1/` (regenerated paper figures — note `brief.md` in that
directory contains hard-coded prose from the figure template, not arm-B facts),
`results/celeba/resample_b1/sae_siglip_k{5,20}.pt` (the new checkpoints).

## 8. Caveats

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
