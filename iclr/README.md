# NEXIS: semi-synthetic CelebA experiments

Code for *From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects
Identification*, anonymous submission to ICLR 2027.

This package reproduces every semi-synthetic CelebA experiment of the paper: Figure 3
(the experimental power paradox) and all the figures and tables of Appendix C. It is
self-contained. `nexis/` implements the method, `celeba/` implements the benchmark, and
`run.py` runs one block of experiments per command.

## What it reproduces

| Paper item | Output (in `results/figures/`) | Command |
|---|---|---|
| Data: embeddings, SAEs, principal coordinates | `data/` (see Layout) | `python run.py prepare` |
| Figure 3 | `figure_main.pdf` | `python run.py main` |
| Appendix C, reference figure | `dgp.pdf` | (same run) |
| Appendix C, reproducibility: replica SAE, and its macro IoU with the interleaved instead of the terminal backward step | `replica_k20.pdf`, `summary.md` | `python run.py replica` |
| Appendix C, supervised test of Principal Alignment | `pa_spectrum.pdf`, `pa_top_images.pdf`, `pa_screening.tex`, `pa_summary.json` | `python run.py alignment` |
| Appendix C, SAE ablations (k = 5; Z_pre) | `model_k5.pdf`, `model_precode.pdf` | `python run.py ablations-model` |
| Appendix C, NEXIS ablations (test, correction, ρ, backward steps) | `method_test.pdf`, `method_adjust.pdf`, `method_rho.pdf`, `method_backward.pdf` | `python run.py ablations-method` |
| Appendix C, test comparison under a linear and a U-shape CATE (table, panel B) | `test_comparison.tex`, `ushape.pdf` | `python run.py ushape` (after `ablations-method`) |
| Appendix C, r = 1 and r = 3 direct modifiers; r = 0 table | `dgp_r1.pdf`, `dgp_r3.pdf`, `dgp_r0_table.tex`, `dgp_r0.pdf` | `python run.py dgp-extra` |
| Appendix C, controlled violation of Principal Alignment (split principal coordinate) | `violation.pdf`, `violation_grid.pdf`, `violation_table.tex`, `summary.md` | `python run.py violation` |
| Everything above | all of the above | `python run.py all` |

Every sweep figure uses the 4 x 3 layout of Appendix C. The rows are the n sweeps at
η ∈ {5, 2} and the η sweeps at n ∈ {2000, 500}, and the columns are precision, recall and
IoU. `summary.md` gives, for each figure, the macro metrics (the mean over the 42 cells),
the IoU at n = 2000, η = 5, the first grid value at which each curve reaches 0.95, and the
per-modifier recall at r = 1, 2, 3. These are the numbers quoted in the text.

## Compute

We measured these numbers on the runs that produced the paper's results. The CPU blocks
ran on 40-core nodes with 40 runs in parallel (`--jobs 40`). CPU-hours are core-hours on
such a node. Peak memory is the largest resident set of the job.

| Block | Hardware | CPU-hours | Wall time on the reference machine | Peak memory |
|---|---|---|---|---|
| `prepare` | 1 GPU (RTX 2080 Ti) | – | ~1.2 h: pool embedding ~7 min, the two main SAEs ~38 min, replica embedding and SAE ~25 min, CPU encoding and principal coordinates ~15 min; plus downloading CelebA (~11 GB with the train split) | 32 GB host RAM |
| `main` | 40-core CPU node | 13 | 20 min | 160 GB |
| `ablations-model` | 40-core CPU node | 24 | 40 min | 160 GB |
| `ablations-method` | 40-core CPU node | 124 | 3.1 h (the LightGBM PCM alone: 50 min) | 160 GB |
| `replica` | 40-core CPU node | 22 | 35 min | 150 GB |
| `dgp-extra` | 40-core CPU node | 56 | 1.4 h | 160 GB |
| `ushape` | 40-core CPU node | 132 | 3.3 h | 160 GB |
| `alignment` | 16-core CPU node | 1.5 | 6 min | 2 GB |
| `violation` | 40-core CPU node | 15 (9.5 if `main` exists) | 45 min (31 min if `main` exists) | 150 GB |
| `all` | 1 GPU, then one 40-core node | ~385 | ~1.2 h GPU + ~10 h CPU | 160 GB |

We measured `violation` with this package, as a job array of its 18 units with one
40-core node and 40 runs per unit. Its wall time is the sum of the unit wall times, which
is the time on a single node. It includes the `main` units (the no-split control), which
are skipped when `main` has already run.

The n sweeps set the peak memory. A run at n = 10,000 holds about 4 GB (the 10,000 x 9,216
design and its residualised copies), so peak memory is about 4 GB per concurrent run, and
the η sweeps (n ≤ 2000) need about a quarter of it. `--jobs N` sets the number of
concurrent runs. Wall time scales as 1/N and peak memory as about 4N GB: for example,
`--jobs 8` needs about 35 GB and takes five times longer than 40 jobs. On a cluster, the
units of a block can also run as a job array (see below). The U-shape sweep, for example,
finished in 20 minutes as 40 array tasks.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

The CPU experiments need only numpy, scipy, pandas, pyarrow, scikit-learn, lightgbm,
joblib and matplotlib. `torch`, `torchvision`, `timm`, `overcomplete`, `datasets` and
`pillow` are used by `prepare` only.

## Quick start

```bash
python run.py prepare                 # one GPU: CelebA, SigLIP 2 embeddings, SAEs, S*
python run.py all --smoke             # every CPU block, 2 seeds and 2 grid points per sweep
python run.py all                     # the full experiments (skips outputs that exist)
```

`--smoke` writes to `results/smoke/`, never to the full results. It takes about 15 minutes
on a single CPU core (`--jobs 1`), or 7 minutes on a multi-core machine. The smoke figures
have the right layout but only two points per curve. Other options:

* `--seeds N` changes the number of Monte Carlo seeds; results go to `results/seedsN/`.
* `--jobs N` sets the number of runs executed in parallel (threads; default: all cores).
* `--overwrite` recomputes outputs that exist. By default every step and unit whose output
  exists is skipped, so interrupted runs resume.
* `python run.py figures` redraws every figure and table from the results on disk.
* `--data-dir` and `--results-dir` (or the environment variables `NEXIS_DATA_DIR` and
  `NEXIS_RESULTS_DIR`) move the data and the results.

The method can also be used on its own:

```python
from nexis import nexis
result = nexis(y, t, z)            # y: (n,) outcome, t: (n,) binary treatment, z: (n, m) representation
result.selected                    # the selected coordinates of z
```

### On a cluster

A block is a list of *units*, one per (experiment, sweep, method), and each unit writes one
parquet file. `python run.py BLOCK --list-units` prints the units, and
`python run.py BLOCK --unit I` runs one of them. `slurm/run_units.sbatch` maps a SLURM job
array onto the units, and `slurm/prepare.sbatch` runs `prepare` on one GPU. Add your
cluster's partition and account options to both.

## Layout

```
run.py              command line (blocks, units, figures)
config.py           ALL paths, grids, seeds, DGP parameters, methods and experiments
nexis/
  cate_tests.py     CATE-equivalence tests: linear, GCM (quadratic / LightGBM), PCM (quadratic / LightGBM)
  search.py         NEXIS (forward step, spectral-gap gate, interleaved and terminal backward
                    steps, FWER / FDR / no correction) and the marginal baselines
celeba/
  prepare.py        CelebA download, SigLIP 2 embeddings, SAE training, encoding, S*
  sae.py            TopK SAE training on patch tokens and encoding
  scm.py            the semi-synthetic randomized experiment (sampler and outcome model)
  experiment.py     sweeps: draw, run a method, score against S*, write parquet
  figures.py        every sweep figure and table
  alignment.py      the supervised Principal Alignment check
  violation.py      the controlled violation of Principal Alignment (figures and tables)
slurm/              generic job templates
```

`prepare` writes to `data/`:

```
celeba/labels.parquet, celeba/images.npy          attributes and 128 x 128 thumbnails of the pool
celeba/embeddings/siglip.npy, siglip_patches.f16  mean-pooled and per-patch SigLIP 2 tokens
celeba/embeddings/sae[_precode][_replica]_k{5,20}.npy   Z and Z_pre of the pool
celeba/ground_truth/{main,replica}_k{5,20}.json   principal coordinate and F1 spectrum per attribute
celeba_train_sample/                              the replica's SAE training images (tokens)
models/sae_{main,replica}_k{5,20}.pt              SAE checkpoints
```

`config.py` holds every setting that the paper reports. Several of them are choices you
may want to change, and they are named options there:

* The r = 3 DGP. `R3_ATTRS`, `R3_GAMMAS` and `R3_BETAS` set the attributes, interactions
  and main effects (default: Wearing_Hat +1, Eyeglasses −1, Sideburns +1, betas 0.3, −0.2,
  0.3). `prepare` computes the principal coordinate of every attribute listed there. If you
  change the attributes after `prepare`, rerun `python run.py prepare --steps ground-truth`.
* The r = 0 layout, `R0_MODE`. The default, `"fixed_beta"`, fixes the prognostic effects
  at the main setting's betas, runs the n sweep only, and uses 200 seeds (`R0_SEEDS`).
  `"scale_beta"` instead lets η scale the prognostic effects and runs the four rows of the
  main grid.
* The NEXIS default (`NEXIS_DEFAULT`) and its one-axis variants (`NEXIS_VARIANTS`).

## Setting (Appendix C.1)

* **Pool.** The 19,867 images of the official CelebA validation split (aligned and cropped),
  from the Hugging Face mirror `flwrlabs/celeba` at a pinned revision.
* **Encoder.** SigLIP 2 ViT-B/16 (timm `vit_base_patch16_siglip_224.v2_webli`) at 224 px.
  The encoder keeps the 196 final-layer patch tokens (768-d) and does not use the attention
  pooling head.
* **Dictionary.** A TopK SAE (`overcomplete` library: a linear encoder, ReLU and TopK, and a
  unit-norm dictionary) with m = 9,216 codes and k ∈ {5, 20}. It is trained on the
  individual patch tokens of the pool: 20 epochs, batches of 20 images, Adam at learning
  rate 5·10⁻⁴, MSE loss. It encodes the mean patch token of each image into Z (the sparse
  post-TopK codes) and Z_pre (the pre-activations).
* **Replica dictionary.** The same k = 20 SAE, trained on an independent random sample of
  19,867 train-split images (seed 1). The CelebA splits are identity-disjoint. It encodes
  the same validation pool.
* **Ground truth.** For each attribute, the principal coordinate is argmax_j of the
  best-threshold F1 of Z_pre^j over the pool (at k = 20: Wearing_Hat → 5348,
  Eyeglasses → 5537, Sideburns → 1683).
* **DGP.** W_k ~ Bernoulli(prevalence) independently, T ~ Bernoulli(0.5), an image drawn
  without replacement from the matching attribute cell, and
  Y = Σ β_k W_k + T (τ₀ + η Σ γ_k W_k) + ε with τ₀ = 0.5, γ = (+1, −1), β = (0.3, −0.2)
  and ε ~ N(0, 1). The seed fixes the whole draw, so all methods, and all DGPs that sample
  the same attributes, see the same units.
* **Controlled violation of Principal Alignment.** The main DGP, with the Wearing_Hat
  principal coordinate j1 split in two complementary halves: U ~ Bernoulli(0.5) is drawn
  once per pool image, Z^{j1} ← U Z^{j1}, and a new coordinate
  Z^{j_new} = (1 − U) Z^{j1} is appended (`DGP_VIOLATION` in `config.py`). Neither half
  screens the modifier off alone, so the target becomes S* = {j1, j_new, j2}. W, T, the
  images and Y are those of the main setting, which serves as the no-split control. The
  block also runs NEXIS with ρ = 0, to show whether the spectral-gap gate stops the search
  before a half enters.
* **Grid.** η ∈ {1, …, 10} at n ∈ {500, 2000}; n ∈ {50, 100, 200, 350, 500, 750, 1000,
  2000, 3500, 5000, 10000} at η ∈ {2, 5}; 50 seeds per cell (200 per n at r = 0).
* **Methods.** Marginal testing (no correction, FDR, FWER) and NEXIS. The NEXIS default is
  α = 0.05, the linear test, a Bonferroni forward gate, spectral gap ρ = 0.5, no
  interleaved backward step, and the terminal backward step at α/m. The GCM/PCM nuisances
  are cross-fitted with 3 folds. The forward step has no cap on its number of rounds,
  except in the runs where it would exceed 10 rounds and make the terminal step
  (|S̃| 2^(|S̃|−1) tests) costly: there it stops after 10 rounds (`max_rounds` in
  `config.py`). These runs are the ablations with no forward correction, ρ = 0 and
  ρ = 0.2, the U-shape experiment (all tests except the LightGBM PCM) and the controlled
  violation of Principal Alignment.

## Determinism

Every run is seeded: the sampler, the cross-fitting folds, the PCM split and LightGBM. The
sweeps are therefore deterministic given the encoded data. Two steps can differ at the
level of floating-point rounding across hardware:

* GPU embedding and SAE training (`prepare`) are not bit-reproducible.
* Encoding with a trained SAE, and the F1 spectra, can differ in the last float32 digits
  on a different CPU. The principal coordinates did not change in our checks.

`prepare` therefore cannot reproduce the paper's dictionaries bit for bit: a retrained SAE
has other coordinate indices (the paper's 5348, 5537, 1683, ...) and slightly different
numbers. To reproduce the paper exactly, start from the original artefacts instead: put
(or symlink) the original pool files `celeba/labels.parquet`, `celeba/images.npy`,
`celeba/embeddings/siglip.npy`, `siglip_patches.f16` and `sae[_precode][_replica]_k*.npy`,
the replica corpus files in `celeba_train_sample/`, and the checkpoints
`models/sae_{main,replica}_k*.pt` at the paths of the Layout above. Every step whose output
exists is skipped, so `python run.py prepare` then only computes the ground-truth files
(on CPU). The checkpoints use the format written by `celeba/sae.py`.

## Real-world applications

The paper also applies NEXIS to two anti-poverty programs: a youth cash-transfer program
in Uganda and the LEAP 1000 program in Ghana. Both analyses use household microdata that
are sensitive or held under data-use agreements. They also use satellite features
geolocated to the program communities. Neither the data nor the derived features can be
released, so this package contains no data or code for these applications. The analyses
did not use this package: they ran the same algorithm (forward step, terminal backward
step, α = 0.05, ρ = 0.5, linear test) with the research implementation of NEXIS in the
parent repository (`src/method/nexis.py`, called by `scripts/realworld_final_runs.py`),
which adds what the applications need and `nexis/` lacks: cluster-robust (CR1S) standard
errors and a hook for a custom conditional test (`cluster`, `pvalue_fn`).

## Assets

* CelebA: Liu et al., *Deep Learning Face Attributes in the Wild*, ICCV 2015 (non-commercial
  research license), through the `flwrlabs/celeba` Hugging Face mirror.
* SigLIP 2: Tschannen et al., 2025, through `timm`.
* TopK SAE: Gao et al., 2024, implemented with the `overcomplete` library.
* GCM: Shah and Peters, 2020. PCM: Lundborg, Kim, Shah and Samworth, *Annals of
  Statistics*, 2024. LightGBM: Ke et al., 2017.
