# NEMS — Neuron Effect Modifier Selection

**NEMS** is a sequential conditional testing procedure for identifying the embedding neurons (features) that act as *effect modifiers* of a binary treatment, given a high-dimensional representation `Z`.

## Method

Given an RCT or observational dataset `(Y, T, Z)`, NEMS iteratively selects neurons `j` by testing the conditional interaction hypothesis

```
H0(j | S) : γ_j = 0   in   Y ~ 1 + T + Z_S + T·Z_S + Z_j + T·Z_j
```

using a Bonferroni gate over the remaining candidates at each step. Selection stops when no remaining neuron clears the gate.

## Repository structure

```
src/
  nems.py        # core selection algorithm and evaluation utilities
  synthetic.py   # synthetic DGP (loading matrix, RCT generator)
  uganda.py      # Uganda YOP helpers (outcome aliases, mapping, causal utilities)
  train.py       # DINOv2/Prithvi patch embedding extraction + SAE training
  analyze.py     # NEMS feature selection for a given outcome
  interpret.py   # VLM→LLM interpretation of selected SAE features
  summarize.py   # ATE + CATE/GATE summary for a given outcome
  plot_features.py  # feature image grids

notebooks/
  synthetic.ipynb   # synthetic benchmark (effect size & sample size sweeps)
  uganda.ipynb      # Uganda YOP real-data analysis

scripts/
  run.sh         # full pipeline (embedding → SAE → NEMS → interpret → summarize → plot)
  reanalyze.sh   # re-run analysis steps only (skips embedding/SAE training)

results/
  synthetic/
    linear/      # saved PDF figures — linear DGP
    quadratic/   # saved PDF figures — quadratic DGP
  uganda/
    {model}_{dim}/{outcome}/   # NEMS results per (model, outcome)

data/            # real-world datasets (not tracked by git)
```

## Quickstart

```python
from src import nems_select

result = nems_select(y=Y, t=T, z=Z, alpha=0.05)
print(result.selected)   # list of selected neuron indices
```

## Uganda YOP application

The Uganda application uses satellite imagery from 2000 (pre-treatment) and individual-level outcomes from the 4-year endline (2012) of the Youth Opportunities Programme (YOP) experiment described in Blattman et al. (2014, QJE). The primary reference for the image-based heterogeneity analysis is Jerzak, Johansson & Daoud (2023, CLeaR).

### Outcomes

Nine endline outcomes are available (all from `UgandaDataProcessed.csv`; see `src/uganda.py: OUTCOME_ALIASES`):

**Labour outcomes** — Blattman et al. (2014) Table III; `skilled_hours` is the primary outcome of Jerzak et al. (2023)

| Alias | CSV column | Description |
|---|---|---|
| `skilled_hours` | `Yobs` | log(skilled-trade hrs/wk + 100) — primary outcome |
| `skilled_employed` | `skilled_dummy_e` | Any skilled-trade engagement in past month (binary) |
| `skilled_fulltime` | `fulltimeskill_e` | ≥ 30 hrs/week in skilled trade (binary) |
| `log_training_hours` | `training_hours_ln_e` | log vocational training hours received |
| `employ_hours` | `employhours_e` | Total employment hours/week — **missing at endline** |

**Economic outcomes** — Blattman et al. (2014) Tables IV & VI

| Alias | CSV column | Description |
|---|---|---|
| `log_earnings` | `profits4w_real_ln_e` | log real 4-week cash earnings (000s 2008 UGX) |
| `log_biz_assets` | `bizasset_val_real_ln_e` | log real business asset value (000s 2008 UGX) |
| `wealth_index` | `wealthindex_e` | Household wealth / durable assets index |
| `wellbeing` | `wealthladder_e` | Subjective wellbeing ladder (1–9) |

Notes:
- `skilled_hours` (`Yobs`) uses a log(x + 100) floor to handle zeros; values range ~4.6–5.4. Effects are interpretable as approximate % changes via exp(β) − 1. Same transformation applies to `log_biz_assets` and `log_earnings`.
- `employ_hours` maps to `employhours_e`, which is entirely missing in the distributed dataset (Blattman et al. 2014 Table III reports it but it is not recomputed in the Jerzak et al. 2023 release).

### Running the pipeline

```bash
# Full pipeline for the primary outcome (default)
bash scripts/run.sh --models=dinov2

# All outcomes, all models
bash scripts/run.sh --models=dinov2,prithvi --all-outcomes

# Re-run analysis steps only (skip embedding/SAE training)
bash scripts/reanalyze.sh --models=dinov2 --outcomes=skilled_hours,log_earnings,log_biz_assets
```

## Synthetic experiments

Open [notebooks/synthetic.ipynb](notebooks/synthetic.ipynb) to reproduce the benchmark sweeps over effect size and sample size, comparing NEMS against marginal interaction testing (raw and Bonferroni-adjusted).

## References

- Blattman, Fiala & Martinez (2014). "Generating Skilled Self-Employment in Developing Countries." *Quarterly Journal of Economics*, 129(2), 697–752.
- Jerzak, Johansson & Daoud (2023). "Image-based Treatment Effect Heterogeneity." *Proceedings of CLeaR*, PMLR 213:531–552.

## Citation

If you use this code, please cite the accompanying paper.
