#!/usr/bin/env python3
"""Ghana exploratory NEXIS run without multiple-testing correction (paper item).

Produces the numbers of the appendix section "Exploratory analysis"
(sec:ghana:exploratory, paper/iclr27/appendix.tex):
  "an earlier, smaller pool, which keeps the 72 neurons active in at least 10
   communities together with the 24 survey covariates and 6 spectral indices [...]
   excludes the two certified neurons [...] sparse burn scar presence [neuron 1777]
   active in 12 communities [...] GATE +26.3 in active communities vs +5.6 in inactive
   ones; the conditional p = 0.0085 in this uncorrected run, but the GATE contrast is
   not significant (p = 0.30)".

The run replays the 6 May 2026 configuration behind results/ghana/mact10/codes/
nexis_no_adj/result.json (src/apps/ghana/interpret.py --min-activations 10, codes): Z =
the SAE codes of the 72 neurons active in >= 10 of the 162 LEAP communities plus the 6
spectral *_mean indices; the 24 survey covariates enter through nexis(w=...);
nexis(adjust=None, cluster=community) with the other arguments at their defaults. It
reproduces that file's selection and p-values exactly (asserted), without writing to it.

GATE of neuron 1777: OLS of dY on (1, T) within active (code > 0) / inactive
households; the contrast, its s.e. and p come from dY ~ 1 + T + A + T x A with CR1S by
community (t(G-1) and normal p reported).

Inputs: data/ghana (load_data), data/ghana/satellite/{sae_activations.npy,
prithvi_comm_ids.npy, spectral_indices.csv}; results/ghana/mact10/codes/nexis_no_adj/
result.json (for the check only).

Command (repo root, CPU, seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/exploratory_run.py

Output: results/ghana/paper_numbers/exploratory_run.{md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.apps.ghana.data import load_data, W_ALL, W_LABELS  # noqa: E402
from src.method.nexis import nexis  # noqa: E402

SAT = ROOT / "data/ghana/satellite"
SAVED = ROOT / "results/ghana/mact10/codes/nexis_no_adj/result.json"
OUT = ROOT / "results/ghana/paper_numbers"
MIN_ACT, NEURON = 10, 1777


def ols_cr1s(X, y, g):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    meat = np.zeros((k, k))
    cl = np.unique(g)
    for c in cl:
        s = X[g == c].T @ e[g == c]
        meat += np.outer(s, s)
    G = len(cl)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), G


def main() -> None:
    df = load_data(ROOT / "data/ghana")
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    sp = pd.read_csv(SAT / "spectral_indices.csv").rename(columns={"comm_id": "comm"})
    spec = [c for c in sp.columns if c.endswith("_mean")]
    m = (df0.set_index("hhid")[["T", "comm"] + W_ALL + ["Y"]]
         .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"})))
    m["dY"] = m["Y1"] - m["Y"]
    m = m.reset_index().merge(sp, on="comm", how="left").set_index("hhid")
    y, t = m["dY"].values.astype(float), m["T"].values.astype(float)
    W = m[W_ALL].values.astype(float)
    comm = m["comm"].values

    codes = np.load(SAT / "sae_activations.npy")
    ids = np.load(SAT / "prithvi_comm_ids.npy")
    live = np.where((codes > 0).sum(axis=0) >= MIN_ACT)[0]
    hh = m["comm"].map(dict(zip(ids, range(len(ids))))).values
    Z = np.concatenate([codes[:, live][hh], m[spec].values.astype(float)], axis=1)
    z_names = [None] * len(live) + [c[:-5] for c in spec]

    res = nexis(y, t, Z, w=W, w_names=[W_LABELS.get(c, c) for c in W_ALL], z_names=z_names,
                alpha=0.05, adjust=None, cluster=comm)
    sel_z, sel_w = [], []
    for i in res.selected:
        name, p = res.feature_names[i], float(res.pvalues[i])
        if name.startswith("z_") and name[2:].isdigit():
            sel_z.append({"neuron_idx": int(live[int(name[2:])]), "pvalue": p})
        else:
            sel_w.append({"label": name, "pvalue": p})

    saved = json.loads(SAVED.read_text()) if SAVED.exists() else None
    if saved is not None:
        a = {e["neuron_idx"]: e["pvalue"] for e in saved["selected_z"]}
        b = {e["neuron_idx"]: e["pvalue"] for e in sel_z}
        assert a.keys() == b.keys() and all(np.isclose(a[k], b[k], rtol=1e-8) for k in a), (a, b)

    act = (codes[:, NEURON] > 0)[hh]
    one = np.ones_like(y)
    ga = ols_cr1s(np.column_stack([one[act], t[act]]), y[act], comm[act])[0][1]
    gi = ols_cr1s(np.column_stack([one[~act], t[~act]]), y[~act], comm[~act])[0][1]
    b, se, G = ols_cr1s(np.column_stack([one, t, act, t * act]), y, comm)
    tstat = b[3] / se[3]
    out = dict(
        pool=dict(neurons=int(len(live)), spectral_indices=len(spec), survey_covariates=len(W_ALL),
                  certified_3821_in_pool=bool(3821 in live), certified_2095_in_pool=bool(2095 in live),
                  n_communities_active={str(k): int((codes[:, k] > 0).sum()) for k in (3821, 2095)}),
        selected_z=sel_z, selected_w=sel_w, matches_saved=saved is not None,
        neuron_1777=dict(p_conditional=next(e["pvalue"] for e in sel_z if e["neuron_idx"] == NEURON),
                         active_communities=int(len(np.unique(comm[act]))),
                         active_households=int(act.sum()), gate_active=float(ga),
                         gate_inactive=float(gi), contrast=float(b[3]), contrast_se=float(se[3]),
                         contrast_p_t=float(2 * stats.t.sf(abs(tstat), G - 1)),
                         contrast_p_normal=float(2 * stats.norm.sf(abs(tstat)))))
    n = out["neuron_1777"]
    md = ["# Ghana exploratory run (no multiple-testing correction)", "",
          f"Pool: {out['pool']}", "",
          "Selected neurons: " + ", ".join(f"{e['neuron_idx']} (p={e['pvalue']:.4f})" for e in sel_z),
          "Selected survey covariates: " + ", ".join(f"{e['label']} (p={e['pvalue']:.4f})" for e in sel_w),
          f"Matches {SAVED.relative_to(ROOT)}: {out['matches_saved']}", "",
          f"Neuron 1777: conditional p = {n['p_conditional']:.4f}; active in "
          f"{n['active_communities']} communities ({n['active_households']} households); GATE "
          f"{n['gate_active']:+.1f} active vs {n['gate_inactive']:+.1f} inactive; contrast "
          f"{n['contrast']:+.2f} (CR1S s.e. {n['contrast_se']:.2f}), p = {n['contrast_p_t']:.3f} "
          f"(t(G-1)), {n['contrast_p_normal']:.3f} (normal)."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exploratory_run.json").write_text(json.dumps(out, indent=2) + "\n")
    (OUT / "exploratory_run.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
