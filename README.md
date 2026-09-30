# NEXIS: Neural EXposure Interaction Search

**From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification**
· Riccardo Cadei, Frank Otchere, Nyasha Tirivayi, Gustavo Angeles Tagliaferro, Falco J. Bargagli-Stoffi, Francesco Locatello
· [arXiv](https://arxiv.org/abs/2606.17010) · [Website](https://riccardocadei.github.io/NEXIS/)

NEXIS finds *which* pre-treatment features drive the heterogeneity of a treatment effect
in a randomized experiment. Given an outcome `y`, a binary treatment `t` and a
representation `z` of rich pre-treatment measurements (for example the atoms of a sparse
autoencoder (SAE) trained on a foundation-model embedding of images), it returns the
minimal set of coordinates of `z` that characterizes the conditional average treatment
effect (CATE): a forward search with conditional interaction tests, then a terminal
backward certification step, which, under the paper's assumptions, recovers the true set
with probability at least 1 − α asymptotically. The selected coordinates are interpreted afterwards (for example by
showing their most activating images to a vision-language model) and can guide how the
treatment is assigned next time.

![NEXIS pipeline](docs/assets/pipeline.gif)

This repository is first a small Python library for the method (`src/method`), kept
minimal and app-agnostic, and second the code that reproduces the paper's experiments
(`src/apps`, with shell launchers in `scripts`).

## Install

```bash
pip install -e .            # the method and the CPU experiments (Python >= 3.11)
pip install -e ".[gpu]"     # + embeddings, SAE training, VLM interpretation
pip install -e ".[geo]"     # + Google Earth Engine downloads and maps (Uganda, Ghana)
```

Versions are pinned to the environment the paper's runs used (Python 3.11.8). The install
exposes the folders of `src/` as top-level modules (`method`, `causality`, `train`,
`apps`), so `from method import nexis` works from any directory.

## Using NEXIS

```python
import numpy as np
from method import nexis

rng = np.random.default_rng(0)
n, m = 2000, 100
w = rng.normal(size=(n, m))          # pre-treatment candidates (e.g. SAE codes, covariates)
t = rng.binomial(1, 0.5, size=n)     # randomized binary treatment
y = w[:, 0] + t * (1.0 + 0.5 * w[:, 3] - 0.5 * w[:, 7]) + rng.normal(size=n)

res = nexis(y=y, t=t, w=w, alpha=0.05,
            backward=False, terminal_filter=True,  # the paper's Algorithm 1
            rho=0.5, max_rounds=None)
res.selected                          # [7, 3]: certified coordinates (columns of w)
res.feature_names                     # "0" ... "99" for an ndarray, column names for a DataFrame
res.metadata["terminal_candidates"]   # forward-step output S~; S~ minus selected = candidates
res.metadata["terminal_log"]          # per j in S~: the certification p-value (worst_p)
```

`w` holds every candidate, learned or hand-crafted, in one matrix: all its columns compete
in one search and the Bonferroni gates count all of them. It can be an ndarray or a pandas
DataFrame; for a DataFrame, `res.feature_names` are its column names and
`w.attrs["cluster"]` supplies cluster labels when `cluster=` is not passed.
`marginal_select(y, t, w, ...)` takes the same `w`.

The defaults of `nexis()` are those of an earlier version of the method (interleaved
backward step, no terminal step) and are kept so that older results reproduce. Always pass
`backward=False, terminal_filter=True` to run the paper's algorithm.

| Parameter | Default | Paper | Meaning |
|---|---|---|---|
| `alpha` | `0.05` | `0.05` | Level α: forward gate α/\|S̄\|, terminal gate α/m (m = columns of `w`) |
| `backward` | `True` | `False` | Interleaved backward step after every forward addition |
| `terminal_filter` | `False` | `True` | Terminal certification: keep j ∈ S̃ only if p_j(A) ≤ α/m for every A ⊆ S̃ \ {j} |
| `rho` | `0.5` | `0.5` | Spectral-gap stop: new \|t\| < ρ · min \|t\| of the selected; `0`/`None` disables |
| `adjust` | `"FWER"` | `"FWER"` | Forward gate: Bonferroni, `"FDR"` (Benjamini–Hochberg) or `None` |
| `max_rounds` | `20` | no cap | Cap on forward rounds (CelebA sweeps: 10; applications: 20, never binding) |
| `test` | `"linear"` | `"linear"` | T × Z_j interaction t-test; also `"GCM: ..."` and `"PCM: ..."` (nonparametric) |
| `cluster`, `hc1`, `pvalue_fn` | `None` | Ghana: clusters | Cluster-robust or HC1 errors, or a custom conditional test |

See the docstring of [`src/method/nexis.py`](src/method/nexis.py) for all options.

## Repository layout

```
src/method/        the NEXIS method (nexis.py); the library, app-agnostic
src/causality/     HC1-robust OLS, ATE and GATE/CATE reporting (estimation.py), and the
                   level-aware clustered CATE test for multilevel data (multilevel.py)
src/train/         TopK SAE training (overcomplete), used by CelebA
src/apps/<app>/    one pipeline per application: celeba, uganda, ghana
                   (data, embeddings, SAE, NEXIS runs, VLM interpretation, figures)
src/apps/realworld_final_runs.py
                   the Uganda and Ghana runs behind the paper's application numbers
src/apps/celeba/benchmark/
                   self-contained CelebA benchmark package, one command per block
scripts/<app>/     shell launchers: submit_<step>.sh (SLURM), run_<step>.sh (bash)
animations/        Manim scenes for the website videos
docs/              project website (GitHub Pages: index.html, assets/)
```

Library code (`src/method`, `src/causality`) stays minimal, app-agnostic and free of
experiment-specific options; application logic belongs in `src/apps`.

## Reproducing the paper

Scripts target SLURM but also run with `bash` from the repo root; logs go to `logs/`,
outputs to the untracked `data/` and `results/`. GPUs are needed for embeddings, SAE
training and VLM interpretation; NEXIS itself runs on CPU in minutes. GPU training is not
bit-reproducible, so a retrained SAE gives other coordinate indices than the ones quoted in
the paper. Each application has a `README.md` in its folder (`src/apps/<app>/`) with every
number the paper quotes, its source file and script, and the design choices behind it.

### CelebA (semi-synthetic benchmark)

Data: the 19,867 validation images of CelebA, streamed from the Hugging Face mirror
`flwrlabs/celeba` at a pinned revision (no manual download). Compute: embeddings and SAEs
~1 h on one GPU, sweeps ~400 CPU-hours. Figure map: [`src/apps/celeba/README.md`](src/apps/celeba/README.md) (Section 7).

```bash
# GPU: SigLIP 2 embeddings and TopK SAEs (m = 9,216, k = 5 and 20), plus the replica SAE
bash scripts/celeba/submit_embed.sh
bash scripts/celeba/submit_sae.sh
sbatch scripts/celeba/submit_resample_embed.sh b1 1
sbatch scripts/celeba/submit_resample_sae.sh 20 b1
# CPU sweeps, 50 seeds (jobs whose output exists are skipped)
bash scripts/celeba/submit_experiment.sh          # main, k=5, Z_pre, method ablations
bash scripts/celeba/submit_experiment_ushape.sh   # U-shape DGP
bash scripts/celeba/submit_dgp_extra.sh           # replica, r = 3, 1, 0
bash scripts/celeba/submit_per_modifier_recall.sh
bash scripts/celeba/submit_alignment_appendix.sh     # Principal Alignment check
# Figures and tables -> results/celeba/figures_v2/
python src/apps/celeba/figure_main.py --data-dir results/celeba/experiment_v2/k20/sae \
    --out-path results/celeba/figures_v2/figure_main.pdf --nexis-key NEXIS-v2
python src/apps/celeba/figure_appendix.py --experiment-dir results/celeba/experiment_v2 \
    --out-dir results/celeba/figures_v2 --variant v2
python src/apps/celeba/figure_dgp_extra.py
python src/apps/celeba/figure_test_story_v2.py
python src/apps/celeba/interleaved_backward_align.py sae_precode_k20 && \
    python src/apps/celeba/third_modifier_candidates.py
python src/apps/celeba/run_statistics.py
(cd src/apps/celeba/benchmark && python run.py violation)   # the controlled-violation figure
```

[`src/apps/celeba/benchmark/`](src/apps/celeba/benchmark/README.md) is a self-contained
package that reproduces every CelebA experiment (`python run.py <block>` from that folder); only its `main` and `violation` blocks are
checked against the paper so far. It will be folded into `src/apps/celeba` as the
canonical CelebA pipeline, running on the single `src/method/nexis.py`.

### Uganda YOP (Youth Opportunities Program)

Data: survey microdata from the public replication release of Jerzak et al. (2023) of the
Blattman, Fiala & Martinez (2014) trial
([Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O8XOSF),
[Hugging Face](https://huggingface.co/datasets/cjerzak/ImageHeterogeneity)), as
`data/uganda/UgandaDataProcessed.csv`; we do not redistribute it. Landsat 7 tiles are
re-extracted from Google Earth Engine (`earthengine authenticate`); its imagery cannot be
redistributed, so the scripts download it. Compute: SAE ~1 h on one GPU,
VLM ~30 min on one H100. Details: [`src/apps/uganda/README.md`](src/apps/uganda/README.md).

```bash
python src/apps/uganda/download_tiles.py --mode rct        # 331 trial sites
python src/apps/uganda/download_tiles.py --mode national   # national grid (SAE corpus)
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_rct --out-dir data/uganda/satellite/rct
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_national --out-dir data/uganda/satellite/national
sbatch scripts/uganda/submit_train_sae.sh                  # -> results/uganda/prithvi_l5_1024/
python src/apps/realworld_final_runs.py --only uganda      # paper numbers
python src/apps/uganda/multilevel_groupcluster.py          # multilevel limitation
python src/apps/uganda/district_sensitivity.py             # district-dummy sensitivity
python src/apps/uganda/table_gate.py                       # GATE columns
sbatch scripts/uganda/submit_interpret.sh                  # VLM labels (Qwen2.5-VL-72B)
python src/apps/uganda/figure_neural.py
python src/apps/uganda/figure_maps.py
python src/apps/figure1_tiles.py                           # Figure 1 tiles
```

### Ghana LEAP 1000 (Livelihood Empowerment Against Poverty)

Data: the LEAP 1000 household panel (2015–2017) is restricted (contact UNICEF Ghana) and
must not be redistributed; Landsat 8 tiles from Google Earth Engine. Compute: SAE ~2 h and
VLM ~45 min on one H100. Details: [`src/apps/ghana/README.md`](src/apps/ghana/README.md)
(Section 11 for path conventions).

```bash
python src/apps/ghana/download_satellite_images.py --year 2015   # 162 communities
python src/apps/ghana/download_national_grid.py                  # national grid (SAE corpus)
python src/apps/ghana/extract_satellite_features.py
python src/apps/ghana/train_sae.py --d-hidden 4096 --k 25 --epochs 2000 --batch-size 256 \
    --train-embeddings data/ghana/satellite/national/prithvi_embeddings.npy \
    --eval-embeddings data/ghana/satellite/prithvi_embeddings.npy \
    --eval-ids data/ghana/satellite/prithvi_comm_ids.npy --out-dir data/ghana/satellite
python src/apps/realworld_final_runs.py --only ghana             # paper numbers
python src/apps/ghana/table_gate.py                              # GATEs
bash scripts/ghana/run_figure_neural.sh                          # VLM temporal step (GPU) and figure
python src/apps/ghana/figure_neural_1777.py                      # exploratory figure
python src/apps/ghana/figure_maps.py
```

## Citation

```bibtex
@article{cadei2026nexis,
  title   = {From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification},
  author  = {Cadei, Riccardo and Otchere, Frank and Tirivayi, Nyasha and Angeles Tagliaferro, Gustavo and Bargagli-Stoffi, Falco J. and Locatello, Francesco},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.17010},
  note    = {Under review}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
