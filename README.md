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
  __init__.py

notebooks/
  synthetic.ipynb   # synthetic benchmark (effect size & sample size sweeps)

results/
  synthetic/
    linear/      # saved PDF figures — linear DGP
    quadratic/   # saved PDF figures — quadratic DGP

data/            # real-world datasets (not tracked by git)
```

## Quickstart

```python
from src import nems_select

result = nems_select(y=Y, t=T, z=Z, alpha=0.05)
print(result.selected)   # list of selected neuron indices
```

## Synthetic experiments

Open [notebooks/synthetic.ipynb](notebooks/synthetic.ipynb) to reproduce the benchmark sweeps over effect size and sample size, comparing NEMS against marginal interaction testing (raw and Bonferroni-adjusted).

## Citation

If you use this code, please cite the accompanying paper.
