#!/usr/bin/env python3
"""GATE (s.e.) columns of the Uganda YOP table tab:nexis_appendix (paper item).

Produces, for every coordinate of Table tab:nexis_appendix (paper/ICLR'27/appendix.tex,
"Coordinates selected by NEXIS on YOP"): GATE among active and inactive units with its
s.e., the contrast Delta = GATE_active - GATE_inactive with s.e. sqrt(se_a^2 + se_i^2),
and the two p-value columns; plus the full-sample difference in means quoted in the
text (+0.32 skilled employment, +0.61 log business assets).

Estimator: within each subgroup, OLS of Y on (1, T) with HC1 standard errors
(src/causality/estimation.py::ate_ols, no covariates), i.e. the difference in means
with HC1 s.e. T = Wobs (grant received), as in the NEXIS runs. Subgroups: active =
Z_j > 0 for SAE neurons, Z_j = 1 for language groups, NDVI above its sample median.
(The June brief's 0.098 for neuron 339 is the unpooled Neyman s.e.; HC1 gives 0.097.)

p-values: read from results/realworld_final/report.json (scripts/realworld_final_runs.py),
run "Wobs | published test | new default": Marginal = p_marginal (unconditional
T x Z_j test), Certification = worst_subset_p (largest p_j(A) over the subsets tested by
the terminal backward step).

Inputs: data/uganda/UgandaDataProcessed.csv and
results/uganda/prithvi_l5_1024/individual_features.npz, loaded through
scripts/realworld_clustered_nexis.py::uganda (the pool NEXIS runs on);
results/realworld_final/report.json.

Command (repo root, CPU, seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/uganda/table_gate.py

Output: results/uganda/paper_numbers/table_gate.{tex,md,json}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import realworld_clustered_nexis as R  # noqa: E402
from causality.estimation import ate_ols  # noqa: E402

REPORT = ROOT / "results/realworld_final/report.json"
RUN = "Wobs | published test | new default"
OUT = ROOT / "results/uganda/paper_numbers"

# (outcome, panel title, [(tier, name, paper label)])
PANELS = [
    ("skilled_employed", "Panel A: Skilled Employment", [
        ("Certified", "W_lang_4", "Karamojong"),
        ("Certified", "W_lang_7", "Pallisa"),
        ("Certified", "Z_533", "Vegetation spatial heterogeneity"),
        ("Candidates", "W_lang_2", "Lugbara"),
        ("Candidates", "Z_339", "Perennial river presence")]),
    ("log_biz_assets", "Panel B: Log Business Assets", [
        ("Certified", "W_ndvi_mean", "NDVI"),
        ("Candidates", "Z_820", "Structured agricultural landscape")]),
]


def active_mask(name: str, z: np.ndarray) -> np.ndarray:
    if name.startswith("Z_"):
        return z > 0
    if len(np.unique(z)) == 2:
        return z == z.max()
    return z > np.median(z)


def gate(y, t):
    est, se, _ = ate_ols(y, t, None)
    return est, se


def ptex(p: float) -> str:
    m, e = f"{p:.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def num(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    report = json.loads(REPORT.read_text())
    res, tex, md = {}, [], []
    for outcome, title, rows in PANELS:
        d = R.uganda(outcome)
        y, t = d["y"], d["t"]
        ate, ate_se = gate(y, t)
        coords = {c["name"]: c for c in report[f"uganda/{outcome}"]["runs"][RUN]["coords"]}
        res[outcome] = {"n": int(len(y)), "difference_in_means": ate, "se": ate_se, "rows": []}
        md += [f"## {title} (n = {len(y)}, difference in means {ate:+.3f} ({ate_se:.3f}))", "",
               "| tier | modifier | coord | n active | GATE active | GATE inactive | Delta | "
               "marginal p | certification p |", "|---|---|---|---|---|---|---|---|---|"]
        tex.append(f"  \\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\[2pt]")
        tier_prev = None
        for tier, name, label in rows:
            z = d["z"][:, d["names"].index(name)]
            act = active_mask(name, z)
            ga, sa = gate(y[act], t[act])
            gi, si = gate(y[~act], t[~act])
            dl, sd = ga - gi, float(np.hypot(sa, si))
            c = coords[name]
            pm, pc = c["p_marginal"], c["worst_subset_p"]
            res[outcome]["rows"].append(dict(tier=tier, name=name, label=label, n_active=int(act.sum()),
                                             gate_active=ga, se_active=sa, gate_inactive=gi,
                                             se_inactive=si, delta=dl, se_delta=sd,
                                             p_marginal=pm, p_certification=pc,
                                             passes_terminal=c["passes_terminal"]))
            md.append(f"| {tier} | {label} | {name} | {int(act.sum())} | {num(ga)} ({sa:.3f}) | "
                      f"{num(gi)} ({si:.3f}) | {num(dl)} ({sd:.3f}) | {pm:.1e} | {pc:.1e} |")
            if tier != tier_prev:
                if tier_prev is not None:
                    tex[-1] += "[2pt]"
                tex.append(f"  \\multicolumn{{6}}{{@{{}}l}}{{\\quad\\textit{{{tier}}}}} \\\\")
                tier_prev = tier
            tex.append(f"    {label} & ${num(ga)}$ ({sa:.3f}) & ${num(gi)}$ ({si:.3f}) & "
                       f"${num(dl)}$ ({sd:.3f}) & {ptex(pm)} & {ptex(pc)} \\\\")
        md.append("")
        tex.append("  \\midrule")
    tex = tex[:-1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_gate.json").write_text(json.dumps(res, indent=2) + "\n")
    (OUT / "table_gate.tex").write_text("\n".join(tex) + "\n")
    (OUT / "table_gate.md").write_text("# tab:nexis_appendix (Uganda YOP)\n\n" + "\n".join(md))
    print("\n".join(md))
    print("\n".join(tex))


if __name__ == "__main__":
    main()
