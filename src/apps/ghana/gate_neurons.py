#!/usr/bin/env python3
"""Marginal GATE table of every community-level candidate Z (gate_Z.csv).

Produces results/ghana/gate/gate_Z.csv: one row per community-level feature, with the
GATE among active and inactive households and a test of the T x Z interaction. The
figure scripts figure_neural.py, figure_neural_1777.py and figure_neural_combined.py
read the sign of its `diff` column to print "(+impact)" or "(-impact)" in each
neuron's title.

Computation (as in the exploratory analysis that first wrote this file):
  * sample: balanced LEAP 1000 panel (households seen in both waves), n = 2,331
    households in 162 communities; outcome dY = endline - baseline monthly
    adult-equivalent consumption; T = LEAP treatment arm (`tac`);
  * Z = [SAE neurons active (> 0) in at least 5 communities | spectral indices |
    distance to district capital, community size], all at community level;
  * each Z_j is binarized: SAE neurons at > 0, the other columns at their median
    across households;
  * GATE(=1), GATE(=0): difference in mean dY between treated and control households
    among active (Z_j = 1) and inactive (Z_j = 0) households; diff = GATE(=1) - GATE(=0);
  * SE and p-value: OLS dY ~ 1 + T + Z_j + T*Z_j, CR1S standard errors clustered by
    community, two-sided t test with n - 4 degrees of freedom
    (src/apps/ghana/analysis.py::gate_modification_table).

The GATE point estimates of neurons 3821 and 2095 equal those of the GATE table
written by table_gate.py (+42.9 vs +6.0 and +56.2 vs +6.4), and those of neuron 1777
are the exploratory numbers (+26.3 vs +5.6, contrast p = 0.30). The SE here is that
of the binarized interaction, not the per-subgroup s.e. of that table, and the p-value
is not the table's marginal p (linear interaction in the continuous Z_j).

The feature names are "z_<k> [neuron <idx>]" for SAE
neurons, where <idx> is the SAE coordinate, then the spectral index names, then the
geographic labels. Rows are sorted by p-value.

Inputs: data/ghana (survey, via data.load_data), data/ghana/satellite/
{sae_activations.npy, sae_comm_ids.npy, spectral_indices.csv}.

Command (repo root, CPU, a few seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/gate_neurons.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.apps.ghana.analysis import gate_modification_table  # noqa: E402
from src.apps.ghana.data import COMMUNITY_Z, W_LABELS, load_data  # noqa: E402

DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR = DATA_DIR / "satellite"
OUT = ROOT / "results" / "ghana" / "gate" / "gate_Z.csv"
MIN_ACTIVE_COMMUNITIES = 5
SHOW = (3821, 2095, 1777)


def main() -> None:
    df = load_data(DATA_DIR)
    waves = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(waves[waves == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]

    merged = (df0.set_index("hhid")[["T", "comm", "Y"]]
              .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
              .dropna(subset=["Y1"]))
    y = (merged["Y1"] - merged["Y"]).to_numpy(float)
    t = merged["T"].to_numpy(float)
    comm = merged["comm"].to_numpy()

    act_full = np.load(SAT_DIR / "sae_activations.npy")
    sae_ids = np.load(SAT_DIR / "sae_comm_ids.npy")
    live = np.where((act_full > 0).sum(axis=0) >= MIN_ACTIVE_COMMUNITIES)[0]
    act = act_full[:, live]

    sp = (pd.read_csv(SAT_DIR / "spectral_indices.csv")
          .rename(columns={"comm_id": "comm"}).set_index("comm"))
    spectral = list(sp.columns)
    sp_comm = sp.reindex(sae_ids)[spectral].to_numpy(float)
    geo_comm = df0.groupby("comm")[COMMUNITY_Z].first().reindex(sae_ids).to_numpy(float)

    Z_comm = np.hstack([act, sp_comm, geo_comm])
    row = pd.Series(np.arange(len(sae_ids)), index=sae_ids)
    Z = Z_comm[row.reindex(comm).to_numpy()]
    names = ([f"z_{k} [neuron {j}]" for k, j in enumerate(live)] + spectral
             + [W_LABELS[c] for c in COMMUNITY_Z])

    gate = gate_modification_table(y, t, Z, z_names=names, n_sae=len(live), cluster=comm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    gate.to_csv(OUT)

    print(f"n = {len(y)} households, {len(np.unique(comm))} communities; "
          f"Z = {len(live)} SAE neurons + {len(spectral)} spectral + {len(COMMUNITY_Z)} geographic")
    idx = gate.index.str.extract(r"neuron (\d+)", expand=False).astype(float)
    print(gate[idx.isin(SHOW)].to_string())
    print(f"Saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
