# NEXIS — Neural EXposure Interaction Search

**From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification**

*Riccardo Cadei*, Frank Otchere, Nyasha Tirivayi, Gustavo Angeles Tagliaferro, Falco J. Bargagli-Stoffi, Francesco Locatello · *Under review, 2026* · [Website](https://riccardocadei.github.io/NEXIS/)

**TL;DR:** We introduce NEXIS, an iterative procedure over sufficient and principally aligned representations for effect heterogeneity, to identify its causal characterization, i.e., answering questions as "*what if* or *how* should I modify my treatment assignement policy?".

## Causal and Interpretable Heterogeneous Treatment Effects Identification

![NEXIS pipeline](docs/assets/pipeline.gif)

<ol type="i">
  <li>
    <strong>Experiment.</strong> Design a controlled experiment, run and measure any candidate treatment interactors.
  </li>

  <li>
    <strong>Represent.</strong> Represent such pre-treatment measurements in an interpretable dictionary with minimal or no human supervision, e.g., training a Sparse Autoencoder (SAE) on a frozen domain-specific foundation model.
  </li>

  <li>
    <strong>Identify.</strong> Neural EXposure Interactors Search (NEXIS): retrieve the minimal and sufficient heterogeneous effect characterization, by iterative CATE-equivalence testing on the representation coordinates. Interpret the selected coordinates <em>a-posteriori</em>, e.g., querying a VLM on the most activating observations.
  </li>

  <li>
    <strong>Policy.</strong> Optimize the treatment assignment accordingly.
  </li>
</ol>


---

## Install

```bash
pip install -e .            # the method and the CPU experiments (Python 3.11)
pip install -e ".[gpu]"     # + embeddings, SAE training, VLM interpretation
pip install -e ".[geo]"     # + Google Earth Engine downloads and maps (Uganda, Ghana)
```

The versions in `pyproject.toml` are those of the conda env every script uses,
`/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl` (Python 3.11.8). The install exposes
the folders of `src/` as top-level modules (`method`, `causality`, `apps`, `train`), so
`from method import nexis` works from any directory. The repo's own scripts also import
`src.method...` and `src.apps...`, which resolve only from the repo root (the scripts put
the root and `src/` on `sys.path`). Run everything from the repo root.

## Quick start

The paper's algorithm (Algorithm 1 of the ICLR 2027 version) is a forward step with a
Bonferroni gate, followed by one terminal backward step at level α/m, with no interleaved
backward step:

```python
from method import nexis

res = nexis(y=Y, t=T, z=Z, alpha=0.05,
            backward=False, terminal_filter=True,   # the paper's algorithm, NOT the src defaults
            rho=0.5, max_rounds=None)
res.selected                          # certified coordinates (column indices of Z)
res.metadata["terminal_candidates"]   # forward-step output S~; S~ minus res.selected = candidates
res.metadata["terminal_log"]          # per j in S~: worst_p (the certification p-value) and worst_A
```

The defaults in `src/method/nexis.py` are still those of the NeurIPS 2026 submission
(interleaved backward step, no terminal step), so that older results reproduce. Always pass
`backward=False, terminal_filter=True` to run the method of the paper.

| Parameter | src default | Paper | Description |
|---|---|---|---|
| `alpha` | `0.05` | `0.05` | Level α: forward gate α/\|S̄\|, terminal gate α/m (m = number of columns of `z`) |
| `backward` | `True` | `False` | Interleaved backward step after every forward addition (Inter-IAMB); optional, off in the paper |
| `terminal_filter` | `False` | `True` | Terminal backward step: keep j ∈ S̃ only if p_j(A) ≤ α/m for every A ⊆ S̃ \ {j}. Carries the precision guarantee; costs at most \|S̃\|·2^(\|S̃\|−1) tests |
| `terminal_max_size` | `None` | `None` | Raise instead of enumerating when \|S̃\| exceeds it |
| `rho` | `0.5` | `0.5` | Spectral-gap gate: stop when the new candidate's \|t\| < ρ · min \|t\| of the selected ones; `0` or `None` disables it |
| `adjust` | `"FWER"` | `"FWER"` | Forward gate: `"FWER"` (Bonferroni α/\|S̄\|), `"FDR"` (Benjamini–Hochberg) or `None` (α). The terminal gate stays α/m |
| `max_rounds` | `20` | no cap | Cap on forward rounds. The CelebA sweeps ran with 10 (`--max-steps 10`), which binds only in the ablations listed in [`iclr/README.md`](iclr/README.md) (Setting, Methods); the applications ran with 20, which never binds (\|S̃\| ≤ 5) |
| `test` | `"linear"` | `"linear"` | CATE-equivalence test: `"linear"` (T × Z_j interaction t-test), `"GCM: quadratic"`, `"GCM: lgbm"`, `"PCM: quadratic"`, `"PCM: lgbm"` |
| `n_splits` | `5` | `3` | Cross-fitting folds of the GCM/PCM nuisances; unused by the linear test |
| `n_estimators`, `max_depth` | `100`, `None` | defaults | LightGBM nuisances: the defaults map to 50 trees of depth 4 with 15 leaves, the paper's setting |
| `pcm_*` | | defaults | PCM projection, polynomial order, LightGBM screen (32 candidates) and the 2·min combination of the two split directions |
| `cluster` | `None` | Ghana: community | CR1S cluster-robust s.e. for the linear test, t(G−1) reference |
| `hc1` | `False` | `False` | HC1 robust s.e. |
| `pvalue_fn` | `None` | applications | Custom conditional test; overrides `test`, `cluster` and `hc1`. The application scripts pass their test through it (for Ghana, CR1S by community, identical to `cluster=`) |
| `w`, `w_names` | `None` | not used | Legacy two-phase search on structured covariates. The applications append the covariates as extra columns of `z` instead |
| `backward_gate`, `alpha_spending` | `"standard"` | not used | Pathwise-Bonferroni variant of the interleaved step (NeurIPS rebuttal) |

`res.pvalues` holds p(j | S \ {j}) for the selected coordinates and the last forward
p-values for the others. See [`src/method/nexis.py`](src/method/nexis.py) for all options.

---

## Reproducing experiments

Scripts target SLURM but run equally with `bash` from the repo root, and logs go to
`logs/`. GPU is required for foundation-model embeddings, SAE training and VLM
interpretation; NEXIS itself is CPU-only. The paper's figures are the files named below,
copied under the same name to `paper/ICLR'27/figures/`. `results/` and `data/` are not
tracked; see [Archived artifacts](#archived-artifacts).

### CelebA — semi-synthetic benchmark

**Data.** No manual download: `src/apps/celeba/embed.py` streams the images from the
Hugging Face mirror `flwrlabs/celeba` (`datasets.load_dataset(..., streaming=True)`; the
replica corpus is read from the local Arrow cache). The pool is the official validation
split (19,867 images). `src/` does not pin the dataset revision; the local cache
(`~/.cache/huggingface/datasets/flwrlabs___celeba`) holds revision `2d738f5…`, the one
`iclr/config.py` pins.

**Two pipelines.** The paper's CelebA numbers and figures were produced by the `src/`
chain below ("NEXIS-v2" = `nexis(rho=0.5, backward=False, terminal_filter=True)`, the
paper's algorithm), except `violation.pdf`. [`iclr/`](iclr/README.md) is a self-contained
reproduction package for all CelebA experiments, one command per block
(`python iclr/run.py <block>`). After the ghana merge it becomes the canonical CelebA
pipeline, moved into `src/apps/celeba` under a neutral name and running on the single
`src/method/nexis.py`. So far only its `main` and `violation` blocks have been checked
against the paper: `violation.pdf` comes from `python iclr/run.py violation`, and `main`
matches the `src` main-setting runs run by run (`results/celeba/paper_numbers/run_statistics.md`).

```bash
# 1. GPU: SigLIP 2 embeddings and TopK SAEs (m = 9,216, k = 5 and 20)
bash scripts/celeba/submit_embed.sh                # data/celeba/{labels.parquet, embeddings/siglip*.npy}
bash scripts/celeba/submit_sae.sh                  # results/celeba/sae_siglip_k{5,20}.pt, data/celeba/embeddings/sae[_precode]_k{5,20}.npy
sbatch scripts/celeba/submit_resample_embed.sh b1 1   # replica corpus: 19,867 train-split images, seed 1
sbatch scripts/celeba/submit_resample_sae.sh 20 b1    # replica SAE: results/celeba/resample_b1/, data/celeba_resample_b1/eval/

# 2. CPU sweeps, 50 seeds, all with --max-steps 10 (jobs whose output exists are skipped)
bash scripts/celeba/submit_experiment_v2.sh        # main, k=5, Z_pre, method ablations → results/celeba/experiment_v2/
bash scripts/celeba/submit_experiment_v2_ushape.sh # U-shape DGP → results/celeba/experiment_v2_ushape/
bash scripts/celeba/submit_dgp_extra_v2.sh         # replica, r=3, r=1, r=0 → results/celeba/experiment_v2_{resample_b1,r3,r1,r0_fixbeta}/
bash scripts/celeba/submit_per_modifier_recall.sh  # per-modifier recall → results/celeba/per_modifier/
bash scripts/celeba/submit_alignment_appendix.sh   # Principal Alignment check → results/celeba/figures_v2/principal_alignment/

# 3. Figures and tables → results/celeba/figures_v2/
python src/apps/celeba/figure_main.py --data-dir results/celeba/experiment_v2/k20/sae \
    --out-path results/celeba/figures_v2/figure_main.pdf --nexis-key NEXIS-v2
python src/apps/celeba/figure_appendix.py --experiment-dir results/celeba/experiment_v2 \
    --out-dir results/celeba/figures_v2 --variant v2
python src/apps/celeba/figure_dgp_extra.py         # replica_k20, dgp_r1, dgp_r3, dgp_r0_table.tex, dgp_extra.md
python src/apps/celeba/figure_test_story_v2.py     # test_story.tex (linear vs U-shape table)
python src/apps/celeba/interleaved_backward_align.py sae_precode_k20 && \
    python src/apps/celeba/third_modifier_candidates.py   # choice of the r = 3 modifier
python src/apps/celeba/run_statistics.py           # |S~| and terminal-test counts (Appendix B)
```

| Paper item | File in `results/celeba/` | Made by |
|---|---|---|
| Figure 3 | `figures_v2/figure_main.pdf` | `figure_main.py --nexis-key NEXIS-v2` |
| `dgp`, `model_k5`, `model_precode`, `method_{test,adjust,rho,backward}` | `figures_v2/*.pdf` | `figure_appendix.py --variant v2` |
| `replica_k20`, `dgp_r1`, `dgp_r3`, r = 0 table | `figures_v2/` | `figure_dgp_extra.py` |
| Test comparison table | `figures_v2/test_story.tex` | `figure_test_story_v2.py` |
| `pa_spectrum`, `pa_top_images`, screening table | `figures_v2/principal_alignment/` | `alignment_appendix.py` |
| `violation` | `iclr_local_runs/results/figures/violation.pdf` | `python iclr/run.py violation`, run in `iclr/` and its outputs moved there |

The sweeps read the principal coordinates from `ground_truth.json` files written by
earlier runs (`--gt-json results/celeba/experiment{,_r3,_resample_b1,_ushape}/…`, in the
archive). Without them, drop `--gt-json`: `run_experiment.py` then recomputes the same
rule (argmax best-threshold F1 on Z_pre). The numbers are listed with their sources in
[`docs/celeba_experiment_brief.md`](docs/celeba_experiment_brief.md).

Not in the paper: the DINOv2 backbone ablation (`backbone=dinov2` in every stage,
`compare_backbones.py`), the NeurIPS-era sweeps (`submit_experiment.sh`,
`submit_resample_experiment.sh`, `run_experiment_*.sh`) and
[`notebooks/celeba.ipynb`](notebooks/celeba.ipynb).

### Uganda YOP — real-world application

**Data.** Survey microdata: `data/uganda/UgandaDataProcessed.csv`, from the public
replication release of Jerzak et al. (2023) of the Blattman, Fiala & Martinez (2014) trial.
Satellite: Landsat 7 2005–2007 median composites, 5×5 km tiles, from Google Earth Engine
(`pip install earthengine-api; earthengine authenticate; earthengine set_project <id>`).
The paper's model is `prithvi_l5` with SAE `prithvi_l5_1024` (1,024 atoms, k = 25).

```bash
python src/apps/uganda/download_tiles.py --mode rct        # 331 RCT sites
python src/apps/uganda/download_tiles.py --mode national   # national grid (SAE training corpus)
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_rct --out-dir data/uganda/satellite/rct
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_national --out-dir data/uganda/satellite/national
sbatch scripts/uganda/train_sae_slurm.sh                   # src/apps/uganda/train_sae.py → results/uganda/prithvi_l5_1024/

python scripts/realworld_final_runs.py --only uganda       # paper numbers → results/realworld_final/report_uganda.json
python scripts/realworld_uganda_groupcluster.py            # multilevel limitation → results/realworld_uganda_groupcluster/
python src/apps/uganda/table_gate.py                       # GATE columns → results/uganda/paper_numbers/
python src/apps/uganda/figure_neural.py                    # figure_neural_{skilled_employed,log_biz_assets}.pdf
python src/apps/uganda/figure_maps.py                      # figure_districts.pdf, figure_languages.pdf
python src/apps/figure1_tiles.py                           # Figure 1 tiles (river.pdf, waterways.pdf)
```

Without `--only`, `realworld_final_runs.py` runs both applications and writes
`report.json`, the file the paper numbers were read from. The paper's rows are the runs
`Wobs | published test | new default` (Uganda) and `pool 167 | published test | new default`
(Ghana). The script first checks that the NeurIPS configuration reproduces the published
sets, read from `results/uganda/prithvi_l5_1024/<outcome>/nexis_result.json` and
`results/ghana/codes/nexis_fwer_crve/result.json` (both archived). VLM labels come from
`src/apps/uganda/interpret.py` via `scripts/uganda/slurm_interpret.sh` (Qwen2.5-VL-72B,
4-bit, H100; add `--extra-atoms 261` for an atom outside those sets). `scripts/uganda/run.sh`
and `reanalyze.sh` are the older multi-backbone pipeline that wrote those published sets;
they are not needed for the paper numbers. Details and numbers:
[`docs/uganda_experiment_brief.md`](docs/uganda_experiment_brief.md).

### Ghana LEAP 1000 — real-world application

**Data.** The LEAP 1000 2015–2017 household panel is not public (contact UNICEF Ghana).
Satellite: Landsat 8 2015 composites from Google Earth Engine, same setup as Uganda; SAE
with 4,096 atoms (k = 25), outputs in `data/ghana/satellite/`.

```bash
python src/apps/ghana/download_satellite_images.py --year 2015   # 162 LEAP communities
python src/apps/ghana/download_national_grid.py                  # national grid (SAE corpus)
python src/apps/ghana/extract_satellite_features.py              # Prithvi embeddings, spectral_indices.csv
python src/apps/ghana/train_sae.py --d-hidden 4096 --k 25 --epochs 2000 --batch-size 256 \
    --train-embeddings data/ghana/satellite/national/prithvi_embeddings.npy \
    --eval-embeddings data/ghana/satellite/prithvi_embeddings.npy \
    --eval-ids data/ghana/satellite/prithvi_comm_ids.npy --out-dir data/ghana/satellite

python scripts/realworld_final_runs.py --only ghana              # paper numbers → results/realworld_final/report_ghana.json
python src/apps/ghana/table_gate.py                              # GATEs and the 18-of-167 count → results/ghana/paper_numbers/
bash scripts/ghana/run_figure_neural.sh                          # VLM temporal step (GPU), then figure_neural_combined.py
python src/apps/ghana/figure_neural_1777.py                      # exploratory burn-scar figure
python src/apps/ghana/figure_maps.py                             # figure_districts.pdf
```

The download, extraction and training scripts default to paths relative to
`src/apps/ghana/` (`../../data/ghana/...`): run the first three from that folder, or pass
`--out-dir`/`--tif-dir`, and pass explicit paths to `train_sae.py` as above. The
SLURM wrappers `scripts/ghana/slurm_train_sae.sh`, `run_temporal_waterways.sh` and
`slurm_temporal_waterways.sh` call `scripts/ghana/*.py` files that no longer exist (the
code moved to `src/apps/ghana/`); they will be fixed after the ghana merge. Details and
numbers: [`docs/ghana_experiment_brief.md`](docs/ghana_experiment_brief.md).
[`notebooks/uganda.ipynb`](notebooks/uganda.ipynb) and
[`notebooks/ghana.ipynb`](notebooks/ghana.ipynb) are exploratory and predate the paper's
configuration.

### Archived artifacts

`/fs3/group/locatgrp/rcadei/nexis-archived_exp/` (see its `README.md` and
`MANIFEST.sha256`) holds a copy of the untracked `data/`, `results/` and `logs/` as of tag
`neurips-rebuttal-final`, plus the rebuttal caches (`cache/nexis_cert`, `nexis_cap`). It
includes the SAE checkpoints and codes behind every published coordinate index: CelebA
`results/celeba/sae_siglip_k{5,20}.pt`, the replica `results/celeba/resample_b1/`, Uganda
`results/uganda/prithvi_l5_1024/` and Ghana `data/ghana/satellite/sae_model.pt`. GPU
embedding and SAE training are not bit-reproducible, so a retrained SAE gives other
coordinate indices (the paper's 5348, 5537, 533, 3821, ... would change): to reproduce the
paper exactly, restore these files instead of retraining. The LEAP 1000 survey data in
`data/ghana/survey/` are restricted; do not redistribute them.

---

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
