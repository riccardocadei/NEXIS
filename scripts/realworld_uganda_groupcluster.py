"""Uganda YOP NEXIS with the CATE test clustered at the treatment-assignment unit.

Blattman, Fiala & Martinez (2014) randomise the grant across applicant groups within
district, cluster standard errors by group and include district fixed effects.  This
script runs NEXIS with that specification of the linear T x Z_j interaction test for
every candidate: CR1S clustered by GROUP (G = 439), 14 district fixed effects in the
nuisance design, t(G - 1) reference.  One clustering for every candidate, not the
level-aware join of scripts/realworld_clustered_nexis.py.

T = Wobs (grant received), as published.  rho = 0.5, alpha = 0.05, FWER forward gate.
Runs, for each outcome (skilled employment, log business assets):

  0a  sanity: published variant, published test (homoskedastic OLS); must reproduce
      the published set
  0b  sanity: new default, published test; must certify the sets stated in the
      rebuttal
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
results/realworld_uganda_groupcluster/{report.json,summary.csv} (tee stdout to run.log).

    python scripts/realworld_uganda_groupcluster.py [--n-perm 9999]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import realworld_clustered_nexis as R  # noqa: E402
import realworld_final_runs as F  # noqa: E402  (patches R._cr_last: singular RI draws get t = 0)

K = 5
OUT = ROOT / "results" / "realworld_uganda_groupcluster"
OUTCOMES = ["skilled_employed", "log_biz_assets"]
# sets stated for the rebuttal (new default, published homoskedastic test)
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


def run(d, lab, fn, cfg, pool, sup):
    out = R.run(d, lab, fn, cfg, pool)
    idx = {n: j for j, n in enumerate(d["names"])}
    for row in out["coords"]:
        row["support"] = sup[idx[row["name"]]]
        row["in_S_tilde"] = row["name"] in out["S_tilde"]
        print(f"        {row['name']:14s} {sup_str(row['support'])}")
    return out


def block(outcome, n_perm):
    d = R.uganda(outcome)
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    n, m = z.shape
    grp, dist = d["levels"]["group"], d["levels"]["district"]
    G = int(len(np.unique(grp)))
    print(f"\n{'=' * 100}\nuganda / {outcome}   n={n}   m={m}   groups={G}   "
          f"districts={len(np.unique(dist))}   communities={len(np.unique(d['levels']['community']))}")
    assert (pd.Series(t).groupby(grp).nunique() == 1).all(), "T varies within group"

    fe = R.block_fe(dist)
    test = R.LevelAwareTest(["group"] * m, {"group": grp}, fe=fe)   # CR1S by group, district FE, t(G-1)
    sup = [support(d, j) for j in range(m)]
    full = list(range(m))
    gated = [j for j in full if sup[j]["finite"] and min(sup[j]["groups_T"], sup[j]["groups_C"]) >= K]
    removed = [names[j] for j in full if j not in gated]
    print(f"  support gate (>= {K} active groups per arm): keeps {len(gated)}/{m}; removed {len(removed)}: {removed}")
    print(f"  published removed by the gate: {[n_ for n_ in removed if n_ in d['published']]}")

    runs = {}
    print("  -- 0a published variant, published test (homoskedastic OLS)")
    runs["0a"] = run(d, "0a", R.plain_test(), R.PUBLISHED_CFG, full, sup)
    if sorted(runs["0a"]["selected"]) != sorted(d["published"]):
        raise SystemExit(f"published set NOT reproduced: {runs['0a']['selected']} vs {d['published']}")
    print("  -- 0b new default, published test (homoskedastic OLS)")
    runs["0b"] = run(d, "0b", R.plain_test(), R.NEW_DEFAULT, full, sup)
    if sorted(runs["0b"]["selected"]) != sorted(NEW_DEFAULT_EXPECTED[outcome]):
        raise SystemExit(f"new-default set NOT reproduced: {runs['0b']['selected']}")
    print("  -- 1 new default, group-clustered test + district FE, full pool")
    runs["1"] = run(d, "1", test, R.NEW_DEFAULT, full, sup)
    print(f"  -- 2 new default, group-clustered test + district FE, support gate k={K}")
    runs["2"] = run(d, "2", test, R.NEW_DEFAULT, gated, sup)
    runs["2"]["gated_out"] = removed
    print("  -- 3 published variant, group-clustered test + district FE, full pool")
    runs["3"] = run(d, "3", test, R.PUBLISHED_CFG, full, sup)
    for r in runs.values():
        r.update(F.kept_dropped(r["selected"], d["published"]))

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
                F.DEGENERATE[0] = 0
                ri = R.randomization_test(d, list(S), idx[row["name"]], n_perm)
                ri["degenerate_draws"] = F.DEGENERATE[0]
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
        F.DEGENERATE[0] = 0
        ri = R.randomization_test(d, S_pub, j, n_perm)
        ri["degenerate_draws"] = F.DEGENERATE[0]
        row = dict(name=names[j], gated_out=names[j] in removed,
                   p_marginal=float(test(y, t, z, [], [j])[j]),
                   p_given_published=float(test(y, t, z, S_pub, [j])[j]),
                   p_given_run2_final=float(test(y, t, z, S_2, [j])[j]),
                   p_homoskedastic_given_published=float(R.plain_test()(y, t, z, S_pub, [j])[j]),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=9999)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(alpha=R.ALPHA, rho=R.RHO, treatment="Wobs", n_perm=args.n_perm,
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
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {OUT / 'report.json'}, {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
