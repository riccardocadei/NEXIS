"""Final set of real-world NEXIS runs behind the paper's application results.

Builds on scripts/realworld_clustered_nexis.py (c48dca3), whose level-aware clustered
CATE test, candidate-level rule and post-hoc checks are reused unchanged.  What differs:
the support gate is k = 5 clusters per (side of Z_j) x (arm) cell instead of 10, and the
grid of runs below.  Both NEXIS variants run everywhere:

  published   nexis(backward=True,  terminal_filter=False)
  new default nexis(backward=False, terminal_filter=True)

with rho = 0.5, alpha = 0.05, the FWER forward gate and the linear T x Z_j test.

A. Uganda YOP (skilled employment, log business assets), for T in
   {Wobs: grant received (published, as-treated), assigned: lottery (intent-to-treat)}:
     (i)  published test: homoskedastic OLS, no clustering, all 170 candidates;
     (ii) level-aware test: CR1S at L_j (join of the group and Z_j's level), t(G-1),
          district FE, support gate k = 5 (recomputed for each T, it depends on T).
B. Ghana LEAP 1000 (first-differenced consumption):
     (i)  published test (CR1S by community for every candidate) on the joint pool of
          131 SAE neurons + 24 survey covariates = 155, as the appendix counts it;
     (ii) level-aware test (HC1 for the household covariates, CR1S by community for the
          neurons), support gate k = 5, same pool;
     (i') and (ii') the same on the pool that also holds the 12 community-level spectral
          indices (167), as the main text describes it.
C. Post-hoc design-based table for every published modifier and every new selection,
   each conditional on the rest of the set it was selected with:
     Uganda: CR1S by group with district FE (t(G-1)) and randomization inference
             replaying the group lottery within district, under T = Wobs and T = assigned;
     Ghana:  CR1S by community (t(G-1)) and a restricted wild cluster bootstrap-t by
             community.
   Each row also carries the number of clusters (at the candidate's L_j) in each
   (active / inactive) x (treated / control) cell.

CPU only.  Reads data/ and results/ (local, not tracked); writes
results/realworld_final/report.json.

    python scripts/realworld_final_runs.py [--n-perm 1999] [--n-boot 9999]
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

K = 5
ALPHA = R.ALPHA
OUT = ROOT / "results" / "realworld_final"
VARIANTS = {"published NEXIS": R.PUBLISHED_CFG, "new default": R.NEW_DEFAULT}
UG_OUTCOMES = ["skilled_employed", "log_biz_assets"]


# ── Rule with the k = 5 gate, and per-cell cluster counts ─────────────────────

def rule_k(d, k=K):
    """R.build_rule, with the support gate re-applied at k clusters per cell."""
    keys, parts, info = R.build_rule(d)
    for j, i in enumerate(info):
        i["testable"] = bool(np.isfinite(d["z"][:, j]).all()) and (i["support"] is None or i["support"] >= k)
    return keys, parts, info


def cells(d, keys, parts, j):
    """Clusters at L_j in each (active = off the mode / inactive) x (treated / control) cell.
    For a column without a mass point (continuous), only the clusters per arm."""
    col, t = d["z"][:, j], d["t"]
    g = parts[keys[j]]
    g = np.arange(len(t)) if g is None else g
    cnt = lambda msk: int(len(np.unique(g[msk])))       # noqa: E731
    vals, c = np.unique(col[np.isfinite(col)], return_counts=True)
    out = dict(cluster=keys[j], n_clusters=int(len(np.unique(g))),
               treated=cnt(t == 1), control=cnt(t == 0))
    if c.max() >= 0.1 * np.isfinite(col).sum():
        act = np.isfinite(col) & (col != vals[c.argmax()])
        out.update(mode=float(vals[c.argmax()]),
                   active_treated=cnt(act & (t == 1)), active_control=cnt(act & (t == 0)),
                   inactive_treated=cnt(~act & (t == 1)), inactive_control=cnt(~act & (t == 0)))
    return out


def cell_str(c):
    if "active_treated" in c:
        return f"active {c['active_treated']}/{c['active_control']} of {c['treated']}/{c['control']} {c['cluster']} (T/C)"
    return f"continuous; {c['treated']}/{c['control']} {c['cluster']} (T/C)"


def do_run(d, label, fn, cfg, pool=None, info=None, rule=None):
    """R.run plus, for every reported coordinate, its cluster counts per cell."""
    out = R.run(d, label, fn, cfg, pool, info)
    idx = {n: j for j, n in enumerate(d["names"])}
    keys, parts, _ = rule
    for row in out["coords"]:
        row["cells"] = cells(d, keys, parts, idx[row["name"]])
    for row in out["coords"]:
        if row["selected"]:
            print(f"        cells {row['name']:14s} {cell_str(row['cells'])}")
    return out


# ── Data ──────────────────────────────────────────────────────────────────────

def ghana_with_spectral(d):
    """Ghana pool of 167: the 155 of R.ghana() plus 12 community-level spectral indices
    (data/ghana/satellite/spectral_indices.csv, the file behind the 6 May notebook run)."""
    from src.apps.ghana.data import load_data
    spec = pd.read_csv(ROOT / "data" / "ghana" / "satellite" / "spectral_indices.csv").set_index("comm_id")
    df = load_data(ROOT / "data" / "ghana")
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    merged = (df0.set_index("hhid")[["comm", "Y"]]
              .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
              .dropna(subset=["Y1"]))
    assert np.allclose((merged["Y1"] - merged["Y"]).values, d["y"])
    S = spec.loc[merged["comm"].values].values.astype(float)
    assert np.isfinite(S).all()
    return dict(d, z=np.hstack([d["z"], S]), names=d["names"] + [f"S_{c}" for c in spec.columns])


# ── Post-hoc design-based checks ─────────────────────────────────────────────

_cr_last_orig = R._cr_last
DEGENERATE = [0]


def _cr_last_safe(X, y, C, factor_G):
    """R._cr_last, but a draw whose design is singular (a permutation that leaves no
    treated or no control cluster on the active side of a sparse Z_j, so T x Z_j is
    collinear) gets t = 0, i.e. it never counts as at least as extreme.  Counted in
    DEGENERATE and reported."""
    try:
        return _cr_last_orig(X, y, C, factor_G)
    except np.linalg.LinAlgError:
        DEGENERATE[0] += 1
        return np.nan, 0.0


R._cr_last = _cr_last_safe

def uganda_group_fe_p(d, S, j):
    """CR1S by group with district FE, t(G-1): the analytic twin of the RI statistic."""
    fe = R.block_fe(d["levels"]["district"])
    m = d["z"].shape[1]
    fn = R.LevelAwareTest(["g"] * m, {"g": d["levels"]["group"]}, fe=fe)
    return float(fn(d["y"], d["t"], d["z"], S, [j])[j])


def posthoc_uganda(d_w, d_a, rules, S, j, n_perm):
    row = {}
    for tag, dd in [("Wobs", d_w), ("assigned", d_a)]:
        keys, parts, _ = rules[tag]
        DEGENERATE[0] = 0
        ri = R.randomization_test(dd, S, j, n_perm)
        row[tag] = dict(p_group_fe=uganda_group_fe_p(dd, S, j), p_ri=ri["p"], ri_floor=ri["floor"],
                        ri_degenerate_draws=DEGENERATE[0],
                        t_obs=ri["t_obs"], cells=cells(dd, keys, parts, j))
    return row


def posthoc_ghana(d, rule, pub_fn, S, j, n_boot):
    keys, parts, _ = rule
    DEGENERATE[0] = 0
    wb = R.wild_cluster_bootstrap(d, S, j, d["levels"]["community"], n_boot)
    return dict(p_community=float(pub_fn(d["y"], d["t"], d["z"], S, [j])[j]),
                p_wild=wb["p"], wild_floor=wb["floor"], wild_degenerate_draws=DEGENERATE[0], t_obs=wb["t_obs"],
                cells=cells(d, keys, parts, j))


# ── Main ──────────────────────────────────────────────────────────────────────

def kept_dropped(sel, published):
    return dict(kept=[n for n in published if n in sel], dropped=[n for n in published if n not in sel],
                new=[n for n in sel if n not in published])


def uganda_block(outcome, args):
    d_w = R.uganda(outcome)
    d_a = dict(d_w, t=d_w["assigned"])
    print(f"\n{'=' * 100}\nuganda / {outcome}   n={len(d_w['y'])}   pool={d_w['z'].shape[1]}   "
          f"groups={len(np.unique(d_w['levels']['group']))}   "
          f"Wobs != assigned in {int((d_w['t'] != d_w['assigned']).sum())} rows")
    fe = R.block_fe(d_w["levels"]["district"])
    rules, runs = {}, {}
    for tag, d in [("Wobs", d_w), ("assigned", d_a)]:
        rule = rules[tag] = rule_k(d)
        keys, parts, info = rule
        pool = [j for j, i in enumerate(info) if i["testable"]]
        removed = [i["name"] for i in info if not i["testable"]]
        print(f"\n  T = {tag}: support gate k={K} keeps {len(pool)}/{len(info)}; "
              f"published removed: {[n for n in removed if n in d['published']]}")
        for vname, cfg in VARIANTS.items():
            lab = f"{tag} | published test | {vname}"
            print(f"  -- {lab}")
            runs[lab] = do_run(d, "i", R.plain_test(), cfg, rule=rule)
            if tag == "Wobs" and cfg is R.PUBLISHED_CFG and sorted(runs[lab]["selected"]) != sorted(d["published"]):
                raise SystemExit(f"published set NOT reproduced: {runs[lab]['selected']} vs {d['published']}")
            lab = f"{tag} | level-aware k={K} | {vname}"
            print(f"  -- {lab}")
            runs[lab] = do_run(d, "ii", R.LevelAwareTest(keys, parts, fe=fe), cfg, pool, info, rule=rule)
            runs[lab]["gated_out"] = removed
    for r in runs.values():
        r.update(kept_dropped(r["selected"], d_w["published"]))

    # C. post hoc on every published modifier and every new selection
    idx = {n: j for j, n in enumerate(d_w["names"])}
    todo = {}
    sets = [("published", d_w["published"])] + [(k, r["selected"]) for k, r in runs.items()]
    for src, sel in sets:
        for n in sel:
            key = (n, tuple(sorted(x for x in sel if x != n)))
            todo.setdefault(key, []).append(src)
    post, t0 = [], time.time()
    print(f"\n  post hoc ({len(todo)} (modifier, conditioning set) pairs, RI n_perm={args.n_perm})")
    for (n, S), srcs in todo.items():
        r = posthoc_uganda(d_w, d_a, rules, [idx[x] for x in S], idx[n], args.n_perm)
        post.append(dict(name=n, given=list(S), from_runs=srcs, **r))
        print(f"    {n:14s} | {list(S)}\n"
              f"        Wobs:     group+FE p={r['Wobs']['p_group_fe']:.1e}  RI p={r['Wobs']['p_ri']:.1e} (degen {r['Wobs']['ri_degenerate_draws']})   "
              f"{cell_str(r['Wobs']['cells'])}\n"
              f"        assigned: group+FE p={r['assigned']['p_group_fe']:.1e}  RI p={r['assigned']['p_ri']:.1e} (degen {r['assigned']['ri_degenerate_draws']})   "
              f"{cell_str(r['assigned']['cells'])}")
    print(f"  post hoc: {time.time() - t0:.0f}s", flush=True)
    return dict(n=len(d_w["y"]), pool=d_w["z"].shape[1], published=d_w["published"],
                n_rows_wobs_ne_assigned=int((d_w["t"] != d_w["assigned"]).sum()),
                candidates={tag: rules[tag][2] for tag in rules}, runs=runs, posthoc=post)


def ghana_block(args):
    d155 = R.ghana()
    d167 = ghana_with_spectral(d155)
    runs, rules, pubfns, datas = {}, {}, {}, {"155": d155, "167": d167}
    spec_p = {}
    for pname, d in datas.items():
        m = d["z"].shape[1]
        print(f"\n{'=' * 100}\nghana / consumption   n={len(d['y'])}   pool={m}")
        rule = rules[pname] = rule_k(d)
        keys, parts, info = rule
        pool = [j for j, i in enumerate(info) if i["testable"]]
        removed = [i["name"] for i in info if not i["testable"]]
        print(f"  levels: {pd.Series([i['level'] for i in info]).value_counts().to_dict()}  "
              f"support gate k={K} keeps {len(pool)}/{m}; published removed: "
              f"{[n for n in removed if n in d['published']]}")
        pubfns[pname] = R.plain_test(cluster=d["levels"]["community"], m=m)
        if pname == "167":
            # where the spectral indices stand under the published test
            spec = [j for j, n in enumerate(d["names"]) if n.startswith("S_")]
            pub = [d["names"].index(n) for n in d["published"]]
            pm = pubfns[pname](d["y"], d["t"], d["z"], [], spec)
            pc = pubfns[pname](d["y"], d["t"], d["z"], pub, spec)
            spec_p = {d["names"][j]: dict(p_marginal=float(pm[j]), p_given_published=float(pc[j])) for j in spec}
            best = min(spec_p, key=lambda n: spec_p[n]["p_marginal"])
            print(f"  spectral indices: smallest marginal p {best} {spec_p[best]['p_marginal']:.1e}, "
                  f"smallest p | published set {min(v['p_given_published'] for v in spec_p.values()):.1e}  "
                  f"(alpha/m = {ALPHA / m:.1e})")
        for vname, cfg in VARIANTS.items():
            lab = f"pool {pname} | published test | {vname}"
            print(f"  -- {lab}")
            runs[lab] = do_run(d, "i", pubfns[pname], cfg, rule=rule)
            if pname == "155" and cfg is R.PUBLISHED_CFG and sorted(runs[lab]["selected"]) != sorted(d["published"]):
                raise SystemExit(f"published set NOT reproduced: {runs[lab]['selected']} vs {d['published']}")
            lab = f"pool {pname} | level-aware k={K} | {vname}"
            print(f"  -- {lab}")
            runs[lab] = do_run(d, "ii", R.LevelAwareTest(keys, parts), cfg, pool, info, rule=rule)
            runs[lab]["gated_out"] = removed
    for r in runs.values():
        r.update(kept_dropped(r["selected"], d155["published"]))

    todo = {}
    sets = [("published", "155", d155["published"])] + \
           [(k, k.split()[1], r["selected"]) for k, r in runs.items()]
    for src, pname, sel in sets:
        for n in sel:
            key = (pname, n, tuple(sorted(x for x in sel if x != n)))
            todo.setdefault(key, []).append(src)
    post, t0 = [], time.time()
    print(f"\n  post hoc ({len(todo)} pairs, wild bootstrap n_boot={args.n_boot})")
    for (pname, n, S), srcs in todo.items():
        d = datas[pname]
        idx = {x: j for j, x in enumerate(d["names"])}
        r = posthoc_ghana(d, rules[pname], pubfns[pname], [idx[x] for x in S], idx[n], args.n_boot)
        post.append(dict(pool=pname, name=n, given=list(S), from_runs=srcs, **r))
        print(f"    [{pname}] {n:14s} | {list(S)}  community CR1S p={r['p_community']:.1e}  "
              f"wild p={r['p_wild']:.1e}   {cell_str(r['cells'])}")
    print(f"  post hoc: {time.time() - t0:.0f}s")
    return dict(n=len(d155["y"]), pools={k: v["z"].shape[1] for k, v in datas.items()},
                published=d155["published"], spectral_indices_pool167=spec_p,
                candidates={k: rules[k][2] for k in rules}, runs=runs, posthoc=post)


def summary(block):
    print(f"\n  {'configuration':52s} {'m':>4s}  selected   [published kept / dropped]")
    for k, r in block["runs"].items():
        print(f"  {k:52s} {r['m']:4d}  {r['selected']}   [kept {r['kept']} / dropped {r['dropped']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1999)
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--only", choices=["uganda", "ghana"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(alpha=ALPHA, rho=R.RHO, min_support=K, n_perm=args.n_perm, n_boot=args.n_boot)
    if args.only in (None, "uganda"):
        for o in UG_OUTCOMES:
            report[f"uganda/{o}"] = uganda_block(o, args)
    if args.only in (None, "ghana"):
        report["ghana/consumption"] = ghana_block(args)
    print(f"\n{'=' * 100}\nSUMMARY")
    for k, b in report.items():
        if isinstance(b, dict):
            print(f"\n{k}   published: {b['published']}")
            summary(b)
    name = "report.json" if args.only is None else f"report_{args.only}.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {OUT / name}")


if __name__ == "__main__":
    main()
