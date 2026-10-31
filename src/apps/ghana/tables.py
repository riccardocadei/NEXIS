#!/usr/bin/env python3
"""Ghana LEAP 1000 tables and numbers of the paper computed on CPU, one subcommand each.

  gate          GATE (s.e.) columns of Table tab:ghana_nexis, the local ATT, the
                active communities / households and the uncorrected marginal count of
                the main text -> results/ghana/paper_numbers/table_gate.{tex,md,json}
  temporal      Table tab:ghana_temporal from the VLM temporal artifact
                (pipeline/interpret_temporal_changes.py) and the local NDVI increase
                (ndvi_change.py) -> results/ghana/paper_numbers/table_temporal.{md,tex}
                with --paper
  exploratory   the exploratory run without multiple-testing correction (Appendix
                sec:ghana:exploratory) -> results/ghana/paper_numbers/exploratory_run.{md,json}
  gate-neurons  marginal GATE of every community-level candidate
                -> results/ghana/gate/gate_Z.csv (read by figures.py for the sign of the
                GATE contrast in the titles)

The NEXIS p-values of the main table come from src/apps/realworld_final_runs.py, which
`gate` reads.  Run from the repo root (CPU, seconds each); `<command> --help` gives the
options.

    python src/apps/ghana/tables.py gate
    python src/apps/ghana/tables.py temporal --paper
    python src/apps/ghana/tables.py exploratory
    python src/apps/ghana/tables.py gate-neurons
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.ghana.analysis import gate_modification_table  # noqa: E402
from apps.ghana.data import COMMUNITY_Z, W_ALL, W_LABELS, load_data  # noqa: E402
from apps.ghana.ndvi_change import increase_area_pct  # noqa: E402
from apps.ghana.pool import ghana, ghana_with_spectral  # noqa: E402
from nexis import nexis  # noqa: E402
from nexis.design import plain_test  # noqa: E402

DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR = DATA_DIR / "satellite"
PAPER_NUMBERS = ROOT / "results/ghana/paper_numbers"


# ===========================================================================
# gate
# ===========================================================================

GATE_DOC = """\
GATE (s.e.) columns of tab:ghana_nexis and the Ghana counts quoted in the paper.

Produces (paper items):
  appendix, Table tab:ghana_nexis: GATE among active / inactive households with s.e.
      clustered by community, the contrast Delta, and the Marginal / Certification
      p-values, for ephemeral waterways (neuron 3821) and closed-canopy forest (2095);
  appendix, Sec. "Discovered effect modifiers": the local ATT (+7.35 GH cedi/month) and
      the number of active communities / households of each discovery (6 / 83, 5 / 42);
  main text, Ghana results: "18 of 167 coordinates pass an uncorrected marginal test".

Estimators. Outcome: first-differenced monthly consumption (2017 - 2015), T = LEAP
household. GATE = OLS of dY on (1, T) within the subgroup (active: Z_j > 0), with CR1S
standard errors clustered by community, (G/(G-1))(n-1)/(n-k), the same factor as the
realworld scripts' test. Delta s.e. = sqrt(se_a^2 + se_i^2). The marginal count applies
the paper's test (linear T x Z_j interaction, CR1S by community, t(G-1)) to each of
the 167 candidates alone and counts p <= 0.05. The table's p-values are read from
results/realworld_final/report.json, run "pool 167 | published test | new default":
Marginal = p_marginal, Certification = worst_subset_p.

Inputs: data/ghana (load_data), data/ghana/satellite/spectral_indices.csv and the SAE
pool, loaded through src/apps/ghana/pool.py::ghana and ::ghana_with_spectral (the
167-candidate pool NEXIS runs on); results/realworld_final/report.json.

Command (repo root, CPU, under a minute):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/tables.py gate

Output: results/ghana/paper_numbers/table_gate.{tex,md,json} (--out-dir to change)
"""

REPORT = ROOT / "results/realworld_final/report.json"
RUN = "pool 167 | published test | new default"
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


def cmd_gate(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py gate", description=GATE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PAPER_NUMBERS)
    out = ap.parse_args(argv).out_dir
    d = ghana_with_spectral(ghana())
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    comm = d["levels"]["community"]
    m = z.shape[1]
    ate, ate_se = gate_cr1s(y, t, comm)

    # marginal test of every candidate alone (the paper's test)
    fn = plain_test(cluster=comm, m=m)
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_gate.json").write_text(json.dumps(res, indent=2) + "\n")
    (out / "table_gate.tex").write_text("\n".join(tex) + "\n")
    (out / "table_gate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ===========================================================================
# temporal
# ===========================================================================

TEMPORAL_DOC = """\
Emit the per-community temporal table straight from the VLM artifact.

Source
------
`results/ghana/temporal/neuron_3821_temporal_neutral.json`, written by
`pipeline/interpret_temporal_changes.py` (Qwen2.5-VL-72B-Instruct, 4-bit, greedy
decoding). The VLM sees the 2015 and 2017 false-colour composites of one
community side by side, with a colour key, and is asked to list every
land-cover change as "+/- <category>" plus a 1-2 sentence description. It is
not told which feature the community was selected for (neutral prompt).

The artifact covers the six communities where neuron 3821 (ephemeral
waterways) fires, `[951, 675, 395, 1265, 655, 624]` (activation > 0 in
data/ghana/satellite/sae_activations.npy; `--check` verifies this).

An earlier run, `results/ghana/temporal/neuron_3821_temporal.json`
(a script since removed), told the model that the tile contains
ephemeral waterways and asked about changes near them. That prompt is leading,
so the table no longer uses it.

Paper item: Table tab:ghana_temporal (paper appendix, "Per-community VLM
temporal analysis for the six waterway-active LEAP communities").

Columns: community, cropland change (from the "+/- cropland" line; "not listed"
when the VLM gives no cropland line), and local NDVI
increase (% area): the area of the patches where NDVI rose between the 2015 and
2017 composites by more than 0.05 beyond the tile's median change, as % of the
tile's non-water area (rule in `ndvi_change.py`; the same patches are boxed in
Figure 5). The vegetation calls are printed as a note, not as a column.

Usage
-----
    python src/apps/ghana/tables.py temporal               # markdown + latex to stdout
    python src/apps/ghana/tables.py temporal --check       # verify against the activations, exit 1 on drift
    python src/apps/ghana/tables.py temporal --paper       # write results/ghana/paper_numbers/table_temporal.{md,tex}
"""

ARTIFACT = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_temporal_neutral.json"
TEMPORAL_NEURON = 3821


def change(entry: dict, category: str) -> str:
    """'+', '-' or '' for one category of the VLM's change list."""
    syms = {c["symbol"] for c in entry["changes"] if c["label"] == category}
    if len(syms) > 1:
        return "+/-"
    return syms.pop() if syms else ""


LATEX = {"+": r"$\uparrow$ increase", "-": r"$\downarrow$ decrease",
         "+/-": "mixed", "": "not listed"}
MD = {"+": "↑ increase", "-": "↓ decrease", "+/-": "mixed", "": "not listed"}


def load_rows():
    data = json.loads(ARTIFACT.read_text())
    rows = [dict(v, comm_id=int(k)) for k, v in data.items()]
    rows.sort(key=lambda e: -e["activation"])
    return rows


def check(rows) -> int:
    """The artifact must cover exactly the communities where the neuron is active."""
    acts = np.load(SAT_DIR / "sae_activations.npy")[:, TEMPORAL_NEURON]
    ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    active = {int(ids[i]) for i in np.flatnonzero(acts > 0)}
    listed = {int(e["comm_id"]) for e in rows}
    ok = active == listed
    print(f"active communities for neuron {TEMPORAL_NEURON}: {sorted(active)}")
    print(f"communities in the artifact         : {sorted(listed)}")
    if not ok:
        print(f"  MISSING from artifact: {sorted(active - listed)}")
        print(f"  NOT ACTUALLY ACTIVE  : {sorted(listed - active)}")
    print("OK" if ok else "DRIFT")
    return 0 if ok else 1


def render(rows):
    md = ["| Community | Cropland change | Local NDVI increase (% area) |", "|---|---|---|"]
    tex = [r"\toprule",
           r"Community & Cropland change & Local NDVI increase (\% area) \\",
           r"\midrule"]
    for e in rows:
        c = change(e, "cropland")
        cell_md, cell_tex = MD[c], LATEX[c]
        pct = increase_area_pct(e["comm_id"])
        pct_md = "n/a" if pct is None else f"{pct:.1f}"
        pct_tex = "n/a" if pct is None else f"${pct:.1f}$"
        md.append(f"| {e['comm_id']} | {cell_md} | {pct_md} |")
        tex.append(f"{e['comm_id']:<5}& {cell_tex} & {pct_tex} \\\\")
    tex.append(r"\bottomrule")
    n_crop = sum(change(e, "cropland") == "+" for e in rows)
    veg = ", ".join(f"{e['comm_id']}: {MD[change(e, 'vegetation')]}" for e in rows)
    note = (f"cropland increase in {n_crop} of {len(rows)} communities; "
            f"vegetation (not tabulated): {veg}")
    return "\n".join(md), "\n".join(tex), note


def cmd_temporal(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="tables.py temporal", description=TEMPORAL_DOC,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Only verify the artifact against the activations.")
    p.add_argument("--paper", action="store_true",
                   help="Write table_temporal.{md,tex} to --out-dir.")
    p.add_argument("--out-dir", type=Path, default=PAPER_NUMBERS,
                   help="where --paper writes (default: %(default)s)")
    args = p.parse_args(argv)

    if not ARTIFACT.exists():
        sys.exit(f"missing artifact: {ARTIFACT}\n"
                 f"run: sbatch scripts/ghana/submit_temporal_changes.sh")
    rows = load_rows()
    if args.check:
        sys.exit(check(rows))

    md, tex, note = render(rows)
    print(md, "\n"); print(tex, "\n"); print(note)
    for e in rows:
        print(f"  {e['comm_id']}: {e['description']}")
    if args.paper:
        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "table_temporal.md").write_text(md + "\n\n" + note + "\n")
        (out / "table_temporal.tex").write_text(tex + "\n")
        print(f"\nwrote {out/'table_temporal.md'} and {out/'table_temporal.tex'}")


# ===========================================================================
# exploratory
# ===========================================================================

EXPLORATORY_DOC = """\
Ghana exploratory NEXIS run without multiple-testing correction (paper item).

Produces the numbers of the appendix section "Exploratory analysis"
(sec:ghana:exploratory of the paper appendix):
  "an earlier, smaller pool, which keeps the 72 neurons active in at least 10
   communities together with the 24 survey covariates and 6 spectral indices [...]
   excludes the two certified neurons [...] sparse burn scar presence [neuron 1777]
   active in 12 communities [...] GATE +26.3 in active communities vs +5.6 in inactive
   ones; the conditional p = 0.0085 in this uncorrected run, but the GATE contrast is
   not significant (p = 0.30)".

The run replays the 6 May 2026 configuration behind results/ghana/mact10/codes/
nexis_no_adj/result.json (src/apps/ghana/pipeline/interpret.py --min-activations 10, codes): Z =
the SAE codes of the 72 neurons active in >= 10 of the 162 LEAP communities plus the 6
spectral *_mean indices, plus the 24 survey covariates, all in one candidate pool;
nexis(adjust=None, cluster=community) with the other arguments at their defaults. The
6 May run passed the survey covariates through a former two-phase form of nexis() that
screened them first and seeded the search with the one it selected (farming household);
with adjust=None no gate depends on the pool size, and the one-pool search selects the
same set with the same p-values. It reproduces that file's selection and p-values
(asserted), without writing to it.

GATE of neuron 1777: OLS of dY on (1, T) within active (code > 0) / inactive
households; the contrast, its s.e. and p come from dY ~ 1 + T + A + T x A with CR1S by
community (t(G-1) and normal p reported).

Inputs: data/ghana (load_data), data/ghana/satellite/{sae_activations.npy,
prithvi_comm_ids.npy, spectral_indices.csv}; results/ghana/mact10/codes/nexis_no_adj/
result.json (for the check only).

Command (repo root, CPU, seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/tables.py exploratory

Output: results/ghana/paper_numbers/exploratory_run.{md,json} (--out-dir to change)
"""

SAVED = ROOT / "results/ghana/mact10/codes/nexis_no_adj/result.json"
MIN_ACT, EXPLORATORY_NEURON = 10, 1777


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


def cmd_exploratory(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py exploratory", description=EXPLORATORY_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=PAPER_NUMBERS)
    out_dir = ap.parse_args(argv).out_dir
    df = load_data(ROOT / "data/ghana")
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    sp = pd.read_csv(SAT_DIR / "spectral_indices.csv").rename(columns={"comm_id": "comm"})
    spec = [c for c in sp.columns if c.endswith("_mean")]
    m = (df0.set_index("hhid")[["T", "comm"] + W_ALL + ["Y"]]
         .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"})))
    m["dY"] = m["Y1"] - m["Y"]
    m = m.reset_index().merge(sp, on="comm", how="left").set_index("hhid")
    y, t = m["dY"].values.astype(float), m["T"].values.astype(float)
    W = m[W_ALL].values.astype(float)
    comm = m["comm"].values

    codes = np.load(SAT_DIR / "sae_activations.npy")
    ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    live = np.where((codes > 0).sum(axis=0) >= MIN_ACT)[0]
    hh = m["comm"].map(dict(zip(ids, range(len(ids))))).values
    Z = np.concatenate([codes[:, live][hh], m[spec].values.astype(float)], axis=1)
    # column names as the saved result.json labels them: z_<j> for the j-th live neuron,
    # z_<index> for the spectral indices, w_<label> for the survey covariates
    names = ([f"z_{j}" for j in range(len(live))] + [f"z_{c[:-5]}" for c in spec]
             + [f"w_{W_LABELS.get(c, c)}" for c in W_ALL])
    pool = pd.DataFrame(np.hstack([Z, W]), columns=names)

    res = nexis(y, t, pool, alpha=0.05, adjust=None, cluster=comm)
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

    act = (codes[:, EXPLORATORY_NEURON] > 0)[hh]
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
        neuron_1777=dict(p_conditional=next(e["pvalue"] for e in sel_z if e["neuron_idx"] == EXPLORATORY_NEURON),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exploratory_run.json").write_text(json.dumps(out, indent=2) + "\n")
    (out_dir / "exploratory_run.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ===========================================================================
# gate-neurons
# ===========================================================================

GATE_NEURONS_DOC = """\
Marginal GATE table of every community-level candidate Z (gate_Z.csv).

Produces results/ghana/gate/gate_Z.csv: one row per community-level feature, with the
GATE among active and inactive households and a test of the T x Z interaction. The
figures (figures.py neural and neural-1777)
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
written by `gate` (+42.9 vs +6.0 and +56.2 vs +6.4), and those of neuron 1777
are the exploratory numbers (+26.3 vs +5.6, contrast p = 0.30). The SE here is that
of the binarized interaction, not the per-subgroup s.e. of that table, and the p-value
is not the table's marginal p (linear interaction in the continuous Z_j).

The feature names are "z_<k> [neuron <idx>]" for SAE
neurons, where <idx> is the SAE coordinate, then the spectral index names, then the
geographic labels. Rows are sorted by p-value.

Inputs: data/ghana (survey, via data.load_data), data/ghana/satellite/
{sae_activations.npy, sae_comm_ids.npy, spectral_indices.csv}.

Command (repo root, CPU, a few seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/ghana/tables.py gate-neurons
"""

GATE_Z_OUT = ROOT / "results" / "ghana" / "gate" / "gate_Z.csv"
MIN_ACTIVE_COMMUNITIES = 5
SHOW = (3821, 2095, 1777)


def cmd_gate_neurons(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py gate-neurons", description=GATE_NEURONS_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=GATE_Z_OUT, help="output CSV (default: %(default)s)")
    out = ap.parse_args(argv).out
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
    out.parent.mkdir(parents=True, exist_ok=True)
    gate.to_csv(out)

    print(f"n = {len(y)} households, {len(np.unique(comm))} communities; "
          f"Z = {len(live)} SAE neurons + {len(spectral)} spectral + {len(COMMUNITY_Z)} geographic")
    idx = gate.index.str.extract(r"neuron (\d+)", expand=False).astype(float)
    print(gate[idx.isin(SHOW)].to_string())
    print(f"Saved -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


COMMANDS = {"gate": cmd_gate, "temporal": cmd_temporal, "exploratory": cmd_exploratory,
            "gate-neurons": cmd_gate_neurons}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
