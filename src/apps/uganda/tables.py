#!/usr/bin/env python3
"""Uganda YOP tables and numbers of the paper computed on CPU, one subcommand each.

  gate        GATE (s.e.) columns of Table tab:nexis_appendix and the difference in
              means of the text -> results/uganda/paper_numbers/table_gate.{tex,md,json}
  districts   district-dummy sensitivity of the language groups (Section 2 of README.md)
              -> results/realworld_final/uganda_districts.json
  multilevel  the multilevel limitation: NEXIS with the CATE test clustered by
              treatment-assignment group and district fixed effects, plus randomization
              inference -> results/realworld_uganda_groupcluster/{report.json,summary.csv}

The NEXIS p-values of the main table come from src/apps/realworld_final_runs.py, which
`gate` reads.  Run from the repo root; `<command> --help` gives the options.

    python src/apps/uganda/tables.py gate
    python src/apps/uganda/tables.py districts [--overwrite]
    python src/apps/uganda/tables.py multilevel | tee results/realworld_uganda_groupcluster/run.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.uganda.data import resolve_outcome  # noqa: E402
from apps.uganda.pool import MODEL_DIR, uganda  # noqa: E402
from apps.realworld_harness import (  # noqa: E402
    ALPHA, NEW_DEFAULT, PUBLISHED_CFG, RHO, VARIANTS, kept_dropped, run)
from nexis.checks import randomization_test  # noqa: E402
from nexis.design import LevelAwareTest, block_fe, plain_test  # noqa: E402
from nexis.effects import ate_ols  # noqa: E402

OUTCOMES = ["skilled_employed", "log_biz_assets"]


# ===========================================================================
# gate
# ===========================================================================

GATE_DOC = """\
GATE (s.e.) columns of the Uganda YOP table tab:nexis_appendix (paper item).

Produces, for every coordinate of Table tab:nexis_appendix (paper appendix,
"Coordinates selected by NEXIS on YOP"): GATE among active and inactive units with its
s.e., the contrast Delta = GATE_active - GATE_inactive with s.e. sqrt(se_a^2 + se_i^2),
and the two p-value columns; plus the full-sample difference in means quoted in the
text (+0.32 skilled employment, +0.61 log business assets).

Estimator: within each subgroup, OLS of Y on (1, T) with HC1 standard errors
(src/nexis/effects.py::ate_ols, no covariates), i.e. the difference in means
with HC1 s.e. T = Wobs (grant received), as in the NEXIS runs. Subgroups: active =
Z_j > 0 for SAE neurons, Z_j = 1 for language groups, NDVI above its sample median.
(The June brief's 0.098 for neuron 339 is the unpooled Neyman s.e.; HC1 gives 0.097.)

p-values: read from results/realworld_final/report.json (src/apps/realworld_final_runs.py),
run "Wobs | published test | new default": Marginal = p_marginal (unconditional
T x Z_j test), Certification = worst_subset_p (largest p_j(A) over the subsets tested by
the terminal backward step).

Inputs: data/uganda/UgandaDataProcessed.csv and
results/uganda/prithvi_l5_1024/individual_features.npz, loaded through
src/apps/uganda/pool.py::uganda (the pool NEXIS runs on);
results/realworld_final/report.json.

Command (repo root, CPU, seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/uganda/tables.py gate

Output: results/uganda/paper_numbers/table_gate.{tex,md,json} (--out-dir to change)
"""

REPORT = ROOT / "results/realworld_final/report.json"
RUN = "Wobs | published test | new default"
GATE_OUT = ROOT / "results/uganda/paper_numbers"

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


def cmd_gate(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py gate", description=GATE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=GATE_OUT)
    out = ap.parse_args(argv).out_dir
    report = json.loads(REPORT.read_text())
    res, tex, md = {}, [], []
    for outcome, title, rows in PANELS:
        d = uganda(outcome)
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_gate.json").write_text(json.dumps(res, indent=2) + "\n")
    (out / "table_gate.tex").write_text("\n".join(tex) + "\n")
    (out / "table_gate.md").write_text("# tab:nexis_appendix (Uganda YOP)\n\n" + "\n".join(md))
    print("\n".join(md))
    print("\n".join(tex))


# ===========================================================================
# districts
# ===========================================================================

DISTRICTS_DOC = """\
Uganda YOP district-dummy sensitivity analysis, with the paper's NEXIS configuration.

The Uganda pool of the paper (src/apps/uganda/pool.py::uganda, m = 170)
holds 7 language-group dummies W_lang_1..7.  This sensitivity analysis replaces them,
in place, with the 14 district dummies W_district_<NAME> (m = 177), everything else
unchanged: 146 SAE atoms, age, female, father's and mother's education, group_female
and the 12 spectral indices; T = grant received (Wobs); linear T x Z_j test with
homoskedastic OLS SEs, no clustering, no support gate (the paper's
`Wobs | published test` run of src/apps/realworld_final_runs.py).

It runs both NEXIS variants, rho = 0.5, alpha = 0.05, FWER forward gate:

  published   nexis(backward=True,  terminal_filter=False)   (the earlier, June run)
  new default nexis(backward=False, terminal_filter=True)    (the paper's algorithm)

The published variant must reproduce the June run behind the brief's p ~ 7.7e-5,
src/apps/uganda/analyze.py --district-dummies --out-suffix _districts
(results/uganda/prithvi_l5_1024/<outcome>_districts/nexis_result.json, `nexis_fwer`),
whose p-values are p(j | S \\ {j}) on the final set S.  The script checks the pool
names, the selected set and those p-values against that file when it exists.

For every district dummy it also reports the marginal p, the p given the final set
and the p given the candidate set S~, against the forward gate alpha/m and the
terminal gate alpha/m.  Language group -> districts: Karamojong = KOTIDO, MOROTO,
NAKAPIRIPIRIT; Lugbara = ARUA, YUMBE; Pallisa = PALLISA.

CPU only, under a minute.  Reads data/ and results/ (local, not tracked); writes
results/realworld_final/uganda_districts.json, or the file given by --out (refuses to
overwrite without --overwrite).

    python src/apps/uganda/tables.py districts [--overwrite] [--out FILE]
"""

DISTRICTS_OUT = ROOT / "results" / "realworld_final" / "uganda_districts.json"
LANG_DISTRICTS = {"Karamojong": ["KOTIDO", "MOROTO", "NAKAPIRIPIRIT"],
                  "Lugbara": ["ARUA", "YUMBE"], "Pallisa": ["PALLISA"]}


def uganda_districts(outcome):
    """The paper's Uganda pool with W_lang_* replaced in place by W_district_*."""
    d = uganda(outcome)
    df = pd.read_csv(ROOT / "data" / "uganda" / "UgandaDataProcessed.csv", low_memory=False)
    Zall = np.load(MODEL_DIR / "individual_features.npz")["features"]
    df = df[df[resolve_outcome(outcome)].notna() & np.isfinite(Zall[:, 0])].reset_index(drop=True)
    assert len(df) == len(d["y"]) and np.array_equal(df["Wobs"].values, d["t"])
    dist = pd.get_dummies(df["district"], prefix="district", dtype=float)
    lang = [j for j, n in enumerate(d["names"]) if n.startswith("W_lang_")]
    assert lang == list(range(lang[0], lang[-1] + 1)) and len(lang) == 7
    a, b = lang[0], lang[-1] + 1
    z = np.hstack([d["z"][:, :a], dist.values, d["z"][:, b:]])
    names = d["names"][:a] + [f"W_{c}" for c in dist.columns] + d["names"][b:]
    return dict(d, z=z, names=names)


def june_run(outcome):
    f = MODEL_DIR / f"{outcome}_districts" / "nexis_result.json"
    return json.loads(f.read_text()) if f.exists() else None


def district_table(d, fn, S_final, S_tilde, m):
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    idx = {n: j for j, n in enumerate(names)}
    cols = [j for j, n in enumerate(names) if n.startswith("W_district_")]
    fin = [idx[n] for n in S_final]
    til = [idx[n] for n in S_tilde]
    pm = fn(y, t, z, [], cols)
    rows = {}
    for j in cols:
        pf = float(fn(y, t, z, [k for k in fin if k != j], [j])[j])
        ps = float(fn(y, t, z, [k for k in til if k != j], [j])[j])
        rows[names[j]] = dict(p_marginal=float(pm[j]), p_given_final=pf, p_given_Stilde=ps,
                              in_Stilde=names[j] in S_tilde, selected=names[j] in S_final,
                              below_alpha_over_m=bool(min(pf, ps) <= ALPHA / m))
    return rows


def cmd_districts(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py districts", description=DISTRICTS_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=DISTRICTS_OUT)
    args = ap.parse_args(argv)
    out = args.out
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    report = dict(alpha=ALPHA, rho=RHO, test="linear, homoskedastic OLS, no clustering",
                  treatment="Wobs (grant received)", lang_districts=LANG_DISTRICTS)
    fn = plain_test()
    for outcome in OUTCOMES:
        d = uganda_districts(outcome)
        m = d["z"].shape[1]
        print(f"\n{'=' * 100}\nuganda / {outcome}   n={len(d['y'])}   pool={m}   alpha/m={ALPHA / m:.2e}")
        june = june_run(outcome)
        block = dict(n=len(d["y"]), pool=m, names=d["names"], runs={})
        if june is not None:
            meta = june["feature_meta"]
            june_names = [f"Z_{k}" for k in meta["sae_active_idx"]] + [f"W_{w}" for w in meta["w_names"]]
            assert june_names == d["names"], "pool differs from the June run"
            print("  pool: identical names and order to the June run")
        for vname, cfg in VARIANTS.items():
            print(f"  -- {vname}")
            r = run(d, vname, fn, cfg)
            r["districts"] = district_table(d, fn, r["selected"], r["S_tilde"], m)
            if cfg is PUBLISHED_CFG and june is not None:
                jsel = {s["label"]: s["pvalue"] for s in june["nexis_fwer"]["selected"]}
                ours = {c["name"]: c["p_cond_final"] for c in r["coords"] if c["selected"]}
                same = sorted(jsel) == sorted(ours)
                rel = max(abs(ours[k] / jsel[k] - 1) for k in jsel) if same else None
                r["june_check"] = dict(june_selected=jsel, same_set=same, max_rel_diff_p=rel)
                print(f"        June run (nexis_fwer): {jsel}\n        same set: {same}"
                      + (f"   max relative p difference: {rel:.1e}" if same else ""))
            block["runs"][vname] = r
            print(f"        district dummies (p marginal | p given final set | p given S~):")
            for n, row in r["districts"].items():
                flag = "selected" if row["selected"] else ("in S~" if row["in_Stilde"] else "")
                print(f"          {n:26s} {row['p_marginal']:.1e} | {row['p_given_final']:.1e} | "
                      f"{row['p_given_Stilde']:.1e}  {flag}")
        report[f"uganda/{outcome}"] = block

    print(f"\n{'=' * 100}\nSUMMARY (skilled employment, the outcome of the paper's sentence)")
    b = report["uganda/skilled_employed"]
    for vname, r in b["runs"].items():
        print(f"\n  {vname}: S~ = {r['S_tilde']}  selected = {r['selected']}")
        for c in r["coords"]:
            if c["name"] == "W_district_PALLISA":
                pf = f"{c['p_cond_final']:.2e}" if c["p_cond_final"] is not None else "-"
                print(f"    PALLISA: p | rest of final set {pf}   marginal {c['p_marginal']:.2e}   "
                      f"certification (worst subset) {c['worst_subset_p']:.2e} "
                      f"(alpha/m = {r['gate']:.2e})")
        for lang, dists in LANG_DISTRICTS.items():
            for dn in dists:
                row = r["districts"][f"W_district_{dn}"]
                print(f"    {lang:10s} {dn:14s} marginal {row['p_marginal']:.1e}  "
                      f"| final {row['p_given_final']:.1e}  "
                      f"{'selected' if row['selected'] else ('in S~' if row['in_Stilde'] else 'not in S~')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {out}")


# ===========================================================================
# multilevel
# ===========================================================================

MULTILEVEL_DOC = """\
Uganda YOP NEXIS with the CATE test clustered at the treatment-assignment unit.

Blattman, Fiala & Martinez (2014) randomise the grant across applicant groups within
district, cluster standard errors by group and include district fixed effects.  This
script runs NEXIS with that specification of the linear T x Z_j interaction test for
every candidate: CR1S clustered by GROUP (G = 439), 14 district fixed effects in the
nuisance design, t(G - 1) reference.  One clustering for every candidate, not the
level-aware join of src/nexis/design.py.

T = Wobs (grant received), as published.  rho = 0.5, alpha = 0.05, FWER forward gate.
Runs, for each outcome (skilled employment, log business assets):

  0a  sanity: published variant, published test (homoskedastic OLS); must reproduce
      the published set
  0b  sanity: new default, published test; must certify the sets stated
      previously
  1   new default nexis(backward=False, terminal_filter=True), group-clustered test,
      all m = 170 candidates
  2   as 1, after a support gate that drops, before the search and without Y, every
      candidate with fewer than K = 5 active groups (rows off the column's mode) in
      either arm.  Continuous columns (no mass point) are never dropped.
  3   published variant nexis(backward=True, terminal_filter=False), group-clustered
      test, all 170 candidates (reference)

For every coordinate of S~ (the forward set) and of the final set: marginal p, p given
the rest of S~, worst-subset p over A in S~ \\ {j} with its argmax, and the active
groups / communities in the treated and control arms.  Post hoc, for every coordinate
in S~: randomization inference replaying the group lottery within district (9,999
permutations), conditional on the rest of S~; its statistic is the same group-clustered
district-FE t, so the analytic twin is the "cond" p.  Last, each published modifier,
selected or not: its group-clustered p (marginal, given the rest of the published set,
given the run-2 set) and its RI p given the rest of the published set.

CPU only, under a few minutes.  Reads data/ and results/ (local, not tracked); writes
<out-dir>/{report.json,summary.csv}, by default in results/realworld_uganda_groupcluster/
(tee stdout to run.log).

    python src/apps/uganda/tables.py multilevel [--n-perm 9999] [--out-dir DIR]
"""

K = 5
MULTILEVEL_OUT = ROOT / "results" / "realworld_uganda_groupcluster"
# sets stated previously (new default, published homoskedastic test)
NEW_DEFAULT_EXPECTED = {"skilled_employed": ["W_lang_4", "W_lang_7", "Z_533"],
                        "log_biz_assets": ["W_ndvi_mean"]}
LABELS = {"W_lang_2": "Lugbara", "W_lang_4": "Karamojong", "W_lang_7": "Pallisa",
          "W_ndvi_mean": "NDVI", "Z_339": "river"}


def label(n):
    return f"{n} ({LABELS[n]})" if n in LABELS else n


def support(d, j):
    """Active (off the mode) groups and communities per arm; continuous columns have no
    mass point and every row counts as active."""
    col, t = d["z"][:, j], d["t"]
    grp, com = d["levels"]["group"], d["levels"]["community"]
    fin = np.isfinite(col)
    vals, c = np.unique(col[fin], return_counts=True)
    mass = bool(c.max() >= 0.1 * fin.sum())
    act = fin & (col != vals[c.argmax()]) if mass else fin
    cnt = lambda lv, msk: int(len(np.unique(lv[msk])))   # noqa: E731
    return dict(mass_point=mass, mode=float(vals[c.argmax()]) if mass else None,
                n_active_rows=int(act.sum()),
                groups_T=cnt(grp, act & (t == 1)), groups_C=cnt(grp, act & (t == 0)),
                communities_T=cnt(com, act & (t == 1)), communities_C=cnt(com, act & (t == 0)),
                finite=bool(fin.all()))


def sup_str(s):
    kind = "" if s["mass_point"] else "continuous, "
    return (f"{kind}groups {s['groups_T']}/{s['groups_C']}  "
            f"communities {s['communities_T']}/{s['communities_C']} (T/C)")


def run_with_support(d, lab, fn, cfg, pool, sup):
    out = run(d, lab, fn, cfg, pool)
    idx = {n: j for j, n in enumerate(d["names"])}
    for row in out["coords"]:
        row["support"] = sup[idx[row["name"]]]
        row["in_S_tilde"] = row["name"] in out["S_tilde"]
        print(f"        {row['name']:14s} {sup_str(row['support'])}")
    return out


def block(outcome, n_perm):
    d = uganda(outcome)
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    n, m = z.shape
    grp, dist = d["levels"]["group"], d["levels"]["district"]
    G = int(len(np.unique(grp)))
    print(f"\n{'=' * 100}\nuganda / {outcome}   n={n}   m={m}   groups={G}   "
          f"districts={len(np.unique(dist))}   communities={len(np.unique(d['levels']['community']))}")
    assert (pd.Series(t).groupby(grp).nunique() == 1).all(), "T varies within group"

    fe = block_fe(dist)
    test = LevelAwareTest(["group"] * m, {"group": grp}, fe=fe)   # CR1S by group, district FE, t(G-1)
    sup = [support(d, j) for j in range(m)]
    full = list(range(m))
    gated = [j for j in full if sup[j]["finite"] and min(sup[j]["groups_T"], sup[j]["groups_C"]) >= K]
    removed = [names[j] for j in full if j not in gated]
    print(f"  support gate (>= {K} active groups per arm): keeps {len(gated)}/{m}; removed {len(removed)}: {removed}")
    print(f"  published removed by the gate: {[n_ for n_ in removed if n_ in d['published']]}")

    runs = {}
    print("  -- 0a published variant, published test (homoskedastic OLS)")
    runs["0a"] = run_with_support(d, "0a", plain_test(), PUBLISHED_CFG, full, sup)
    if sorted(runs["0a"]["selected"]) != sorted(d["published"]):
        raise SystemExit(f"published set NOT reproduced: {runs['0a']['selected']} vs {d['published']}")
    print("  -- 0b new default, published test (homoskedastic OLS)")
    runs["0b"] = run_with_support(d, "0b", plain_test(), NEW_DEFAULT, full, sup)
    if sorted(runs["0b"]["selected"]) != sorted(NEW_DEFAULT_EXPECTED[outcome]):
        raise SystemExit(f"new-default set NOT reproduced: {runs['0b']['selected']}")
    print("  -- 1 new default, group-clustered test + district FE, full pool")
    runs["1"] = run_with_support(d, "1", test, NEW_DEFAULT, full, sup)
    print(f"  -- 2 new default, group-clustered test + district FE, support gate k={K}")
    runs["2"] = run_with_support(d, "2", test, NEW_DEFAULT, gated, sup)
    runs["2"]["gated_out"] = removed
    print("  -- 3 published variant, group-clustered test + district FE, full pool")
    runs["3"] = run_with_support(d, "3", test, PUBLISHED_CFG, full, sup)
    for r in runs.values():
        r.update(kept_dropped(r["selected"], d["published"]))

    # post hoc RI for every coordinate of S~ in runs 1-3, given the rest of S~
    idx = {n_: j for j, n_ in enumerate(names)}
    cache, t0 = {}, time.time()
    print(f"\n  RI: group lottery within district, n_perm={n_perm}, given S~ \\ {{j}}")
    for rk in ["1", "2", "3"]:
        S_t = runs[rk]["S_tilde"]
        for row in runs[rk]["coords"]:
            if not row["in_S_tilde"]:
                continue
            S = tuple(sorted(idx[x] for x in S_t if x != row["name"]))
            key = (idx[row["name"]], S)
            if key not in cache:
                ri = randomization_test(d, list(S), idx[row["name"]], n_perm)
                cache[key] = ri
                print(f"    {row['name']:14s} | {[names[k] for k in S]}  RI p={ri['p']:.1e} "
                      f"(floor {ri['floor']:.0e}, degenerate draws {ri['degenerate_draws']})  "
                      f"analytic p={row['p_cond_Stilde']:.1e}", flush=True)
            row["ri"] = cache[key]
    # the published modifiers under the group-clustered test, selected or not
    print("\n  published modifiers under the group-clustered test (+ RI given the rest of the published set)")
    pub = [idx[x] for x in d["published"]]
    fin2 = [idx[x] for x in runs["2"]["selected"]]
    pubrows = []
    for j in pub:
        S_pub = [k for k in pub if k != j]
        S_2 = [k for k in fin2 if k != j]
        ri = randomization_test(d, S_pub, j, n_perm)
        row = dict(name=names[j], gated_out=names[j] in removed,
                   p_marginal=float(test(y, t, z, [], [j])[j]),
                   p_given_published=float(test(y, t, z, S_pub, [j])[j]),
                   p_given_run2_final=float(test(y, t, z, S_2, [j])[j]),
                   p_homoskedastic_given_published=float(plain_test()(y, t, z, S_pub, [j])[j]),
                   ri_given_published=ri, support=sup[j],
                   status={rk: ("certified" if names[j] in runs[rk]["selected"] else
                                "candidate" if names[j] in runs[rk]["S_tilde"] else "not selected")
                           for rk in ["1", "2", "3"]})
        pubrows.append(row)
        print(f"    {label(names[j]):22s} marg {row['p_marginal']:.1e}  |published {row['p_given_published']:.1e} "
              f"(homosk. {row['p_homoskedastic_given_published']:.1e})  |run-2 set {row['p_given_run2_final']:.1e}  "
              f"RI|published {ri['p']:.1e} (degen {ri['degenerate_draws']})  gated out: {row['gated_out']}  "
              f"{sup_str(sup[j])}  {row['status']}", flush=True)
    print(f"  RI: {time.time() - t0:.0f}s")
    return dict(n=n, m=m, G=G, published=d["published"], min_active_groups=K,
                gated_out=removed, m_gated=len(gated), support={names[j]: sup[j] for j in full},
                runs=runs, published_modifiers=pubrows)


def summary_rows(outcome, b):
    rows = []
    for rk, r in b["runs"].items():
        for c in r["coords"]:
            s = c["support"]
            rows.append(dict(outcome=outcome, run=rk, config=r["config"], m=r["m"], alpha_over_m=r["gate"],
                             name=c["name"], label=LABELS.get(c["name"], ""),
                             status="certified" if c["selected"] else "candidate",
                             published=c["name"] in b["published"],
                             p_marginal=c["p_marginal"], p_cond_Stilde=c["p_cond_Stilde"],
                             p_cond_final=c["p_cond_final"], worst_subset_p=c["worst_subset_p"],
                             worst_A=" ".join(c["worst_A"]), p_ri=c.get("ri", {}).get("p"),
                             groups_T=s["groups_T"], groups_C=s["groups_C"],
                             communities_T=s["communities_T"], communities_C=s["communities_C"],
                             mass_point=s["mass_point"]))
    return rows


def cmd_multilevel(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="tables.py multilevel", description=MULTILEVEL_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--out-dir", type=Path, default=MULTILEVEL_OUT)
    args = ap.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    report = dict(alpha=ALPHA, rho=RHO, treatment="Wobs", n_perm=args.n_perm,
                  test="linear T x Z_j, CR1S clustered by group, district FE, t(G-1)")
    rows = []
    for o in OUTCOMES:
        report[o] = block(o, args.n_perm)
        rows += summary_rows(o, report[o])

    print(f"\n{'=' * 100}\nSUMMARY  (status | worst-subset p | RI p | active groups T/C)")
    for o in OUTCOMES:
        b = report[o]
        print(f"\n{o}   published: {[label(x) for x in b['published']]}   gate removed "
              f"{len(b['gated_out'])}, published among them: {[x for x in b['gated_out'] if x in b['published']]}")
        for rk in ["1", "2", "3"]:
            r = b["runs"][rk]
            print(f"  run {rk} ({r['config']}), m={r['m']}, alpha/m={r['gate']:.2e}: "
                  f"S~={r['S_tilde']} -> {r['selected']}   [published kept {r['kept']} / dropped {r['dropped']}]")
            for c in r["coords"]:
                ri = f"{c['ri']['p']:.1e}" if "ri" in c else "   -   "
                print(f"      {label(c['name']):22s} {'certified' if c['selected'] else 'candidate':9s}  "
                      f"worst {c['worst_subset_p']:.1e}  RI {ri}  groups {c['support']['groups_T']}/{c['support']['groups_C']}")
        for p in b["published_modifiers"]:
            print(f"  published {label(p['name']):22s} {p['status']}  p|published {p['p_given_published']:.1e}  "
                  f"RI|published {p['ri_given_published']['p']:.1e}")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {out / 'report.json'}, {out / 'summary.csv'}")


COMMANDS = {"gate": cmd_gate, "districts": cmd_districts, "multilevel": cmd_multilevel}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
