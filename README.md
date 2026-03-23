# NEMS — Neuron Effect Modifier Selection

**Identifying treatment effect modifiers in high-dimensional learned representations.**

Treatment effects vary across individuals — but detecting *which features* drive this heterogeneity is challenging when you have thousands of candidate variables. **NEMS** is a method for identifying effect modifiers (features that predict differential treatment response) from pre-trained neural network representations, while controlling the family-wise error rate. It is designed for the regime where [features >> samples], where naive interaction screening fails.

### Motivation: satellite imagery and RCTs

A motivating application combines randomised experiments with satellite imagery: suppose you have an RCT with treatment `T` and outcomes `Y`, paired with pre-treatment satellite imagery from the locations where treated units live. Modern vision models (e.g. Prithvi, DINOv2) extract rich spatial features from that imagery, and Sparse Autoencoders can map these to interpretable individual features. Given these high-dimensional embeddings `Z`, the question becomes: *which learned features interact with treatment to drive outcome differences?* NEMS provides a principled, multiply-tested answer.

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

## Example application: Uganda Youth Opportunities Programme

We apply NEMS to the Uganda Youth Opportunities Programme (YOP), a cash-and-training RCT in northern Uganda ([Blattman, Fiala & Martinez, 2014](https://doi.org/10.1093/qje/qju003)). Following prior work by [Jerzak, Johansson & Daoud (2023)](https://proceedings.mlr.press/v213/jerzak23a.html), we pair each participant with pre-treatment satellite imagery (2000) and extract learned features using **Prithvi** and **DINOv2**. We train a Sparse Autoencoder on top for interpretability, then apply NEMS.

For the primary outcome (log skilled-trade hours), NEMS identifies 2 effect modifiers:

| Rank | Feature | Interpretation | Effect (high) | Effect (low) | Δ CATE | p-value |
|------|---------|----------------|--------------|-------------|--------|---------|
| 1 | SAE\_659 | No perennial water source | +0.038 | −0.001 | **+0.039\*** | 3.0 × 10⁻⁶ |
| 2 | lang\_6 | Lugbara language region | −0.003 | +0.030 | **−0.033\*** | 1.8 × 10⁻³ |

> \* 95% CI of CATE difference excludes zero.

The programme is substantially more effective in drier areas without perennial water. This demonstrates how satellite imagery + NEMS can identify actionable geographic targeting to improve programme efficiency.

![Uganda study sites](results/uganda/map.png)

**To reproduce this example**, see `src/uganda.py` for outcome definitions and [notebooks/uganda.ipynb](notebooks/uganda.ipynb) for detailed analysis. Eight outcomes are supported (labour, earnings, assets, wellbeing); run the pipeline with:

```bash
bash scripts/run.sh --models=prithvi,dinov2 --all-outcomes
```

---

## Method validation: synthetic benchmarks

To validate that NEMS correctly controls the family-wise error rate and achieves good power, we run experiments on synthetic data where the ground truth is known. The benchmark sweeps over effect size and sample size, comparing NEMS against marginal interaction testing (raw and Bonferroni-adjusted). Across all settings, NEMS achieves higher power at controlled FWER.

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


