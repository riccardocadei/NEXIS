# NEMS — Neuron Effect Modifier Selection

**How does a treatment affect different people in different places — and can satellite imagery tell us why?**

Large pre-trained vision models produce rich, high-dimensional representations of remote-sensing imagery. These representations encode thousands of latent features, many of which capture geographic, environmental, or economic characteristics relevant to policy evaluation. **NEMS** is a method that systematically searches these representations to identify which learned features act as *effect modifiers* of a binary treatment — that is, which features predict treatment effect heterogeneity — while controlling the family-wise error rate.

The core application we have in mind is causal inference from randomised experiments paired with satellite imagery: given an RCT with outcomes `Y` and treatment `T`, and a high-dimensional image embedding `Z` (e.g. from a Sparse Autoencoder trained on Prithvi or DINOv2 features), NEMS answers the question *"which neurons in Z interact with T to change outcomes?"*

---

## Method

Given an RCT or observational dataset `(Y, T, Z)`, NEMS iteratively selects neurons `j` by testing the conditional interaction hypothesis

```
H0(j | S) : γ_j = 0   in   Y ~ 1 + T + Z_S + T·Z_S + Z_j + T·Z_j
```

conditioning on the already-selected set `S`. At each step a Bonferroni gate is applied over all remaining candidates, so the family-wise error rate is controlled throughout. Selection stops when no remaining neuron clears the gate.

The procedure is designed for the high-dimensional regime (`p >> n`) where a naïve interaction screen would produce far too many false discoveries. By conditioning on the growing selected set and gating with Bonferroni, NEMS achieves valid sequential selection without requiring post-hoc adjustment.

```python
from src import nems_select

result = nems_select(y=Y, t=T, z=Z, alpha=0.05)
print(result.selected)   # list of selected neuron indices
```

---

## Experiments

### Uganda Youth Opportunities Programme

We apply NEMS to the Uganda Youth Opportunities Programme (YOP), a large-scale cash-and-training RCT studied in [Blattman, Fiala & Martinez (2014)](#references). The experiment randomised cash grants to applicant groups across northern Uganda, with outcomes measured at a 4-year endline in 2012.

Following [Jerzak, Johansson & Daoud (2023)](#references), we pair each participant with the satellite imagery of their village from the year 2000 (pre-treatment), and ask whether image-derived features predict who benefits most. We extract patch embeddings using **Prithvi** (IBM/NASA geospatial foundation model) and **DINOv2**, then train a Sparse Autoencoder (SAE) on top to obtain interpretable monosemantic features, which are passed to NEMS.

The map below shows the spatial distribution of study sites across Uganda:

![Uganda study sites](results/uganda/map.png)

#### Primary outcome: log skilled-trade hours (`log_skilled_hours`)

The primary outcome from Jerzak et al. (2023) is log(skilled-trade hrs/wk + 100). NEMS (Prithvi, SAE dim = 1024) identifies **2 effect modifiers**:

| Rank | Feature | Interpretation | GATE (low) | GATE (high) | Δ CATE | p-value |
|------|---------|----------------|-----------|------------|--------|---------|
| 1 | SAE\_659 | **No perennial water source** (Open water / wetland) | −0.001 | +0.038 | **+0.039\*** | 3.0 × 10⁻⁶ |
| 2 | lang\_6 | Lugbara language region | +0.030 | −0.003 | **−0.033\*** | 1.8 × 10⁻³ |

> \* 95% CI of Δ CATE excludes zero. ATE = +0.020 (SE = 0.006, p = 0.0004, n = 2 372).

The programme was significantly more effective in drier areas with no perennial water source (GATE ≈ +0.038) than in wetter areas (GATE ≈ −0.001). Participants from Lugbara-speaking communities showed lower treatment response, suggesting geographic and ethno-linguistic targeting could improve programme efficiency.

The figure below summarises the GATE decomposition for this outcome:

![Summary illustration — log skilled hours](results/uganda/prithvi_1024/log_skilled_hours/summary_illustration.png)

#### Available outcomes

Eight endline outcomes are available (all from `UgandaDataProcessed.csv`; see `src/uganda.py`):

**Labour outcomes** — Blattman et al. (2014) Table III

| Alias | Description |
|-------|-------------|
| `log_skilled_hours` | log(skilled-trade hrs/wk + 100) — primary outcome |
| `skilled_employed` | Any skilled-trade engagement in past month (binary) |
| `skilled_fulltime` | ≥ 30 hrs/week in skilled trade (binary) |
| `log_training_hours` | log vocational training hours received |

**Economic outcomes** — Blattman et al. (2014) Tables IV & VI

| Alias | Description |
|-------|-------------|
| `log_earnings` | log real 4-week cash earnings |
| `log_biz_assets` | log real business asset value |
| `wealth_index` | Household wealth / durable assets index |
| `wellbeing` | Subjective wellbeing ladder (1–9) |

#### Running the pipeline

```bash
# Full pipeline for the primary outcome (default)
bash scripts/run.sh --models=prithvi

# All outcomes, all models
bash scripts/run.sh --models=dinov2,prithvi --all-outcomes

# Re-run analysis steps only (skip embedding/SAE training)
bash scripts/reanalyze.sh --models=prithvi --outcomes=log_skilled_hours,log_earnings,log_biz_assets
```

---

### Synthetic benchmark

We validate NEMS on synthetic data where the true set of effect modifiers is known. The benchmark sweeps over effect size and sample size, comparing NEMS against marginal interaction testing (raw and Bonferroni-adjusted). NEMS consistently achieves higher power at controlled FWER across all settings.

See [notebooks/synthetic.ipynb](notebooks/synthetic.ipynb) to reproduce the benchmark.

---

## Repository structure

```
src/
  nems.py          # core selection algorithm and evaluation utilities
  synthetic.py     # synthetic DGP (loading matrix, RCT generator)
  uganda.py        # Uganda YOP helpers (outcome aliases, mapping, causal utilities)
  train.py         # DINOv2/Prithvi patch embedding extraction + SAE training
  analyze.py       # NEMS feature selection for a given outcome
  interpret.py     # VLM→LLM interpretation of selected SAE features
  summarize.py     # ATE + CATE/GATE summary for a given outcome
  plot_features.py # feature image grids

notebooks/
  synthetic.ipynb  # synthetic benchmark (effect size & sample size sweeps)
  uganda.ipynb     # Uganda YOP real-data analysis

scripts/
  run.sh           # full pipeline (embedding → SAE → NEMS → interpret → summarize → plot)
  reanalyze.sh     # re-run analysis steps only (skips embedding/SAE training)

papers/
  aistats26-workshop.pdf  # NEMS workshop paper (AISTATS 2026)

results/
  uganda/
    map.png
    {model}_{dim}/{outcome}/   # NEMS results per (model, outcome)
  synthetic/
    linear/        # saved PDF figures — linear DGP
    quadratic/     # saved PDF figures — quadratic DGP

data/              # real-world datasets (not tracked by git)
```

---

## References

<a name="references"></a>

Blattman, C., Fiala, N., & Martinez, S. (2014). Generating skilled self-employment in developing countries: Experimental evidence from Uganda. *Quarterly Journal of Economics*, 129(2), 697–752.

Jerzak, C. T., Johansson, F., & Daoud, A. (2023). Image-based treatment effect heterogeneity. *Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR)*, PMLR 213:531–552.

---

## Citation

If you use NEMS, please cite our paper (see [papers/aistats26-workshop.pdf](papers/aistats26-workshop.pdf)):

```bibtex
@article{nems2025,
  title   = {},
  author  = {},
  journal = {},
  year    = {2025},
  note    = {Preprint coming soon}
}
```

```bibtex
@article{blattman2014uganda,
  title   = {Generating Skilled Self-Employment in Developing Countries:
             Experimental Evidence from {Uganda}},
  author  = {Blattman, Christopher and Fiala, Nathan and Martinez, Sebastian},
  journal = {Quarterly Journal of Economics},
  volume  = {129},
  number  = {2},
  pages   = {697--752},
  year    = {2014}
}
```

```bibtex
@inproceedings{jerzak2023image,
  title     = {Image-Based Treatment Effect Heterogeneity},
  author    = {Jerzak, Connor T. and Johansson, Fredrik and Daoud, Adel},
  booktitle = {Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {213},
  pages     = {531--552},
  year      = {2023},
  publisher = {PMLR}
}
```
