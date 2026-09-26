#!/usr/bin/env python3
"""GATE (s.e.) columns of tab:ghana_nexis and the Ghana counts quoted in the paper.

Produces (paper/ICLR'27):
  appendix, Table tab:ghana_nexis: GATE among active / inactive households with s.e.
      clustered by community, the contrast Delta, and the Marginal / Certification
      p-values, for ephemeral waterways (neuron 3821) and closed-canopy forest (2095);
  appendix, Sec. "Discovered effect modifiers": the local ATT (+7.35 GH cedi/month) and
      the number of active communities / households of each discovery (6 / 83, 5 / 42);
  main.tex, Ghana results: "18 of 167 coordinates pass an uncorrected marginal test".

Estimators. Outcome: first-differenced monthly consumption (2017 - 2015), T = LEAP
household. GATE = OLS of dY on (1, T) within the subgroup (active: Z_j > 0), with CR1S
standard errors clustered by community, (G/(G-1))(n-1)/(n-k), the same factor as the
realworld scripts' test. Delta s.e. = sqrt(se_a^2 + se_i^2). The marginal count applies
the paper's test (linear T x Z_j interaction, CR1S by community, t(G-1)) to each of
the 167 candidates alone and counts p <= 0.05. The table's p-values are read from
results/realworld_final/report.json, run "pool 167 | published test | new default":
Marginal = p_marginal, Certification = worst_subset_p.

Inputs: data/ghana (load_data), data/ghana/satellite/spectral_indices.csv and the SAE
pool, loaded through scripts/realworld_clustered_nexis.py::ghana and
scripts/realworld_final_runs.py::ghana_with_spectral (the 167-candidate pool NEXIS runs
on); results/realworld_final/report.json.

Command (repo root, CPU, under a minute):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/table_gate.py

Output: results/ghana/paper_numbers/table_gate.{tex,md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import realworld_clustered_nexis as R  # noqa: E402
import realworld_final_runs as F  # noqa: E402

REPORT = ROOT / "results/realworld_final/report.json"
RUN = "pool 167 | published test | new default"
OUT = ROOT / "results/ghana/paper_numbers"
ROWS = [("Z_3821", "Ephemeral waterways"), ("Z_2095", "Closed-canopy forest")]


def gate_cr1s(y, t, g):
    """OLS y ~ 1 + t, coefficient on t and its CR1S s.e. clustered by g."""
    X = np.column_stack([np.ones_like(t), t])
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    meat = np.zeros((k, k))
    clusters = np.unique(g)
    for c in clusters:
        s = X[g == c].T @ e[g == c]
        meat += np.outer(s, s)
    G = len(clusters)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return float(b[1]), float(np.sqrt(V[1, 1]))


def ptex(p: float) -> str:
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def main() -> None:
    d = F.ghana_with_spectral(R.ghana())
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    comm = d["levels"]["community"]
    m = z.shape[1]
    ate, ate_se = gate_cr1s(y, t, comm)

    # marginal test of every candidate alone (the paper's test)
    fn = R.plain_test(cluster=comm, m=m)
    pm_all = np.asarray(fn(y, t, z, [], list(range(m))), dtype=float)
    kinds = {"SAE neurons": "Z_", "survey covariates": "W_", "spectral indices": "S_"}
    marg = {"m": int(m), "n_pass_0.05": int((pm_all <= 0.05).sum()),
            "by_kind": {k: {"m": int(sum(n.startswith(p) for n in names)),
                            "n_pass": int(sum(n.startswith(p) and q <= 0.05
                                              for n, q in zip(names, pm_all)))}
                        for k, p in kinds.items()},
            "passing": {names[j]: float(pm_all[j]) for j in np.argsort(pm_all)
                        if pm_all[j] <= 0.05}}

    coords = {c["name"]: c for c in json.loads(REPORT.read_text())["ghana/consumption"]
              ["runs"][RUN]["coords"]}
    rows, tex = [], []
    for name, label in ROWS:
        act = z[:, names.index(name)] > 0
        ga, sa = gate_cr1s(y[act], t[act], comm[act])
        gi, si = gate_cr1s(y[~act], t[~act], comm[~act])
        c = coords[name]
        rows.append(dict(name=name, label=label, n_active_households=int(act.sum()),
                         n_active_communities=int(len(np.unique(comm[act]))),
                         gate_active=ga, se_active=sa, gate_inactive=gi, se_inactive=si,
                         delta=ga - gi, se_delta=float(np.hypot(sa, si)),
                         p_marginal=c["p_marginal"], p_certification=c["worst_subset_p"],
                         p_marginal_recomputed=float(pm_all[names.index(name)]),
                         ratio_to_att=ga / ate))
        tex.append(f"{label:20s} & $+{ga:.1f}\\ ({sa:.1f})$ & $+{gi:.1f}\\ ({si:.1f})$ & "
                   f"${ga - gi:+.1f}$ & {ptex(c['p_marginal'])} & "
                   f"{ptex(c['worst_subset_p'])} \\\\")

    res = dict(n=int(len(y)), n_communities=int(len(np.unique(comm))),
               att=ate, att_se=ate_se, rows=rows, marginal=marg)
    md = [f"# Ghana LEAP 1000 paper numbers", "",
          f"n = {len(y)} households, {len(np.unique(comm))} communities; local ATT "
          f"(DiD) {ate:+.2f} (CR1S s.e. {ate_se:.2f}).", "",
          "## tab:ghana_nexis", "",
          "| modifier | coord | active comm. / hh | GATE active | GATE inactive | Delta | "
          "marginal p | certification p | GATE active / ATT |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['name']} | {r['n_active_communities']} / "
                  f"{r['n_active_households']} | {r['gate_active']:+.2f} ({r['se_active']:.2f}) | "
                  f"{r['gate_inactive']:+.2f} ({r['se_inactive']:.2f}) | {r['delta']:+.2f} "
                  f"({r['se_delta']:.2f}) | {r['p_marginal']:.2e} | {r['p_certification']:.2e} | "
                  f"{r['ratio_to_att']:.1f}x |")
    md += ["", "## Uncorrected marginal test (main text)", "",
           f"{marg['n_pass_0.05']} of {m} candidates have marginal p <= 0.05 "
           f"(CR1S by community, t(G-1)); by kind: {marg['by_kind']}.", "",
           "Passing: " + ", ".join(f"{k} {v:.1e}" for k, v in marg["passing"].items()), "",
           "## LaTeX rows", "", "```", *tex, "```"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_gate.json").write_text(json.dumps(res, indent=2) + "\n")
    (OUT / "table_gate.tex").write_text("\n".join(tex) + "\n")
    (OUT / "table_gate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
