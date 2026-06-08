# NEXIS — Neural EXposure Interaction Search

**From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification**

Cadei et al. · *Under review* · [Website](https://riccardocadei.github.io/NEIS/) · [Workshop paper](assets/aistats26-workshop.pdf)

---

**TL;DR:** NEXIS finds which features *causally* drive treatment effect heterogeneity — combining foundation-model representations of complex observations (satellite imagery, medical imaging) with a statistically rigorous sequential selection that provably controls false discoveries.

---

## Pipeline

![NEXIS pipeline: raw observations → foundation model → sparse autoencoder → NEXIS → VLM interpreter → policy guidelines](assets/pipeline.png)

Given pre-treatment observations **X** (e.g. satellite imagery) and a randomised experiment **(Y, T)**, NEXIS:
1. Extracts dense embeddings via a **foundation model** (Prithvi, DINOv2, DINOv3)
2. Decomposes them into sparse, near-monosemantic neurons via a **Sparse Autoencoder** (SAE), yielding a candidate matrix **Z ∈ Rⁿˣᵐ** (m ~ 10⁴)
3. Runs **NEXIS** — a forward-backward Markov-blanket discovery loop — to select S* ⊂ [m], the principal proxies of the direct effect modifiers, with FWER ≤ α
4. Passes top/bottom activating patches of each selected neuron through a **VLM** (Qwen-VL, GeoChat) to produce human-readable descriptions
5. Returns **causal and interpretable policy guidelines**

## Problem setting

<table>
<tr>
<td valign="middle" width="58%">

HTE is causally explained by **direct effect modifiers** W<sup>dir</sup>, but these are rarely measured and typically entangled in complex observations. NEXIS targets W<sup>dir</sup> — the only modifiers that license policy intervention — while provably excluding indirect modifiers (W<sup>ind</sup>), proxies (W<sup>prx</sup>), and common-cause spurious correlates (W<sup>cc</sup>).

Under Measurement and Representation Sufficiency, the NEXIS output S* satisfies:

```
τ(W^dir) = E[τ | Z^{S*}]   a.s.    with P(Ŝ_n = S*) ≥ 1 − α
```

*Gray: observed. White: latent.*

</td>
<td valign="middle" align="center" width="42%">
<img src="assets/causal_model.png" width="270" alt="Effect modification taxonomy: T, Y, X, Z observed; W^dir, W^ind, W^prx, W^cc latent"/>
<br/>
<sub>Effect modification taxonomy (VanderWeele 2007). Only <b>W<sup>dir</sup></b> has a causal pathway to τ.</sub>
</td>
</tr>
</table>

---

## Install

```bash
pip install -e .
```

## Quick start

```python
from src.method import nexis

result = nexis(y=Y, t=T, z=Z, alpha=0.05)
print(result.selected)        # indices of selected neurons in Z
print(result.pvalues)         # Bonferroni-gated p-values at each step
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.05` | FWER significance level |
| `test` | `"parametric"` | Test type: `"parametric"` (linear interaction) or `"gcm"` (GCM, model-free) |
| `adjust` | `True` | Forward-backward pruning (backward step) |
| `rho` | `0.5` | Cross-fitting ratio for nuisance estimation |

See `src/method/nexis.py` for all options and method variants.

---

## Reproducing experiments

All experiments require a GPU. Scripts are designed for SLURM but also run via `bash` locally.

### CelebA — semi-synthetic benchmark

**Data**: CelebA face images (download from the [official source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html); place at `data/celeba/`).

```bash
# 1. Extract SigLIP embeddings and train SAE
bash scripts/celeba/submit_embed.sh
bash scripts/celeba/submit_sae.sh

# 2. Run NEXIS experiments (effect size × sample size sweeps)
bash scripts/celeba/submit_experiment.sh

# 3. Generate figures
python src/apps/celeba/figure_main.py
python src/apps/celeba/figure_appendix.py
```

See `notebooks/celeba.ipynb` for interactive exploration.

### Uganda YOP — real-world application

**Satellite data**: Landsat 7 via Google Earth Engine. One-time setup:
```bash
pip install earthengine-api
earthengine authenticate --auth_mode notebook
earthengine set_project <your-gee-project-id>
python src/apps/uganda/download_tiles.py --mode rct
```

**Survey data**: Available from the [World Bank Microdata Library](https://microdata.worldbank.org/) — search "Uganda Youth Opportunities Programme".

```bash
# Full pipeline (embedding → SAE → NEXIS → VLM interpretation → figures)
bash scripts/uganda/run.sh --models=prithvi,dinov2,dinov3 --all-outcomes

# Skip embedding/SAE training (use existing embeddings)
bash scripts/uganda/reanalyze.sh --models=prithvi --all-outcomes
```

Available backbone presets for `--models=`: `prithvi`, `dinov2`, `dinov3`, `dinov2_large`.  
Available outcome aliases for `--outcomes=`: `log_skilled_hours`, `skilled_employed`, `skilled_fulltime`, `log_training_hours`, `log_earnings`, `log_biz_assets`, `wellbeing`, `wealth_index`.

See `notebooks/uganda.ipynb` for interactive visualisation.

### Ghana LEAP 1000 — real-world application

**Satellite data**: Landsat 8 via Google Earth Engine (same setup as Uganda):
```bash
python src/apps/ghana/download_satellite_images.py --year 2015
```

**Survey data**: The LEAP 1000 2015–2017 household panel is not publicly available. Contact UNICEF Ghana directly if you are interested in access.

```bash
# Train SAE and run NEXIS
bash scripts/ghana/slurm_train_sae.sh
bash scripts/ghana/slurm_stats.sh

# VLM interpretation
bash scripts/ghana/slurm_interpret_7b.sh   # Qwen-VL 7B
bash scripts/ghana/slurm_interpret.sh      # Qwen-VL 72B

# Generate figures
bash scripts/ghana/run_figure_neural.sh
```

See `notebooks/ghana.ipynb` for interactive visualisation.

---

## Repository structure

```
src/
  method/
    nexis.py          # core algorithm, CATE-equivalence tests, evaluation utilities
  causality/
    estimation.py     # ATE/GATE estimation
  apps/
    celeba/           # CelebA benchmark (embed, SAE, experiments, figures)
    ghana/            # Ghana LEAP 1000 pipeline
    synthetic/        # synthetic DGP utilities
    uganda/           # Uganda YOP pipeline

notebooks/
  celeba.ipynb        # CelebA semi-synthetic benchmark
  synthetic.ipynb     # synthetic sweeps
  uganda.ipynb        # Uganda YOP analysis
  ghana.ipynb         # Ghana LEAP 1000 analysis

scripts/
  celeba/             # SLURM/bash scripts for CelebA
  ghana/              # SLURM/bash scripts for Ghana
  uganda/
    run.sh            # full Uganda pipeline
    reanalyze.sh      # re-run analysis steps only (skip embedding/SAE)

assets/
  pipeline.png              # pipeline overview diagram
  causal_model.png          # effect modification taxonomy (TikZ)
  aistats26-workshop.pdf    # CauScale @ AISTATS 2026 workshop version

results/
  uganda/prithvi_1024/      # selected Uganda results (narratives + figures)

docs/
  index.html                # project website (GitHub Pages)
  assets/                   # figures served by the website

data/                       # local only — not tracked
```

---

## Citation

```bibtex
@article{cadei2025nexis,
  title   = {From Tokens to Policy: Causal and Interpretable
             Heterogeneous Treatment Effects Identification},
  author  = {Cadei, Riccardo and Otchere, Frank and Tirivayi, Nyasha and
             Angeles Tagliaferro, Gustavo and Bargagli-Stoffi, Falco J.
             and Locatello, Francesco},
  year    = {2025},
  note    = {Under review. Preprint coming soon.}
}
```
