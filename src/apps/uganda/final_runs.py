"""Uganda YOP block of the final real-world runs (src/apps/realworld_final_runs.py).

For each outcome (skilled employment, log business assets) and each
T in {Wobs: grant received (published, as-treated), assigned: lottery (intent-to-treat)},
both NEXIS variants with
  (i)  the published test: homoskedastic OLS, no clustering, all 170 candidates;
  (ii) the level-aware test: CR1S at L_j (join of the group and Z_j's level), t(G-1),
       district FE, support gate k = 5 (recomputed for each T, it depends on T).
Post hoc, for every published modifier and every new selection, each conditional on the
rest of the set it was selected with: CR1S by group with district FE (t(G-1)) and
randomization inference replaying the group lottery within district, under T = Wobs and
T = assigned.
"""

from __future__ import annotations

import time

import numpy as np

from apps.uganda.pool import uganda
from causality.multilevel import (MIN_SUPPORT, PUBLISHED_CFG, VARIANTS, LevelAwareTest, block_fe,
                                  build_rule, cell_str, cells, kept_dropped, plain_test,
                                  randomization_test, run)

OUTCOMES = ["skilled_employed", "log_biz_assets"]


def uganda_group_fe_p(d, S, j):
    """CR1S by group with district FE, t(G-1): the analytic twin of the RI statistic."""
    fe = block_fe(d["levels"]["district"])
    m = d["z"].shape[1]
    fn = LevelAwareTest(["g"] * m, {"g": d["levels"]["group"]}, fe=fe)
    return float(fn(d["y"], d["t"], d["z"], S, [j])[j])


def posthoc_uganda(d_w, d_a, rules, S, j, n_perm):
    row = {}
    for tag, dd in [("Wobs", d_w), ("assigned", d_a)]:
        keys, parts, _ = rules[tag]
        ri = randomization_test(dd, S, j, n_perm)
        row[tag] = dict(p_group_fe=uganda_group_fe_p(dd, S, j), p_ri=ri["p"], ri_floor=ri["floor"],
                        ri_degenerate_draws=ri["degenerate_draws"],
                        t_obs=ri["t_obs"], cells=cells(dd, keys, parts, j))
    return row


def uganda_block(outcome, args):
    d_w = uganda(outcome)
    d_a = dict(d_w, t=d_w["assigned"])
    print(f"\n{'=' * 100}\nuganda / {outcome}   n={len(d_w['y'])}   pool={d_w['z'].shape[1]}   "
          f"groups={len(np.unique(d_w['levels']['group']))}   "
          f"Wobs != assigned in {int((d_w['t'] != d_w['assigned']).sum())} rows")
    fe = block_fe(d_w["levels"]["district"])
    rules, runs = {}, {}
    for tag, d in [("Wobs", d_w), ("assigned", d_a)]:
        rule = rules[tag] = build_rule(d)
        keys, parts, info = rule
        pool = [j for j, i in enumerate(info) if i["testable"]]
        removed = [i["name"] for i in info if not i["testable"]]
        print(f"\n  T = {tag}: support gate k={MIN_SUPPORT} keeps {len(pool)}/{len(info)}; "
              f"published removed: {[n for n in removed if n in d['published']]}")
        for vname, cfg in VARIANTS.items():
            lab = f"{tag} | published test | {vname}"
            print(f"  -- {lab}")
            runs[lab] = run(d, "i", plain_test(), cfg, rule=rule)
            if tag == "Wobs" and cfg is PUBLISHED_CFG and sorted(runs[lab]["selected"]) != sorted(d["published"]):
                raise SystemExit(f"published set NOT reproduced: {runs[lab]['selected']} vs {d['published']}")
            lab = f"{tag} | level-aware k={MIN_SUPPORT} | {vname}"
            print(f"  -- {lab}")
            runs[lab] = run(d, "ii", LevelAwareTest(keys, parts, fe=fe), cfg, pool, info, rule=rule)
            runs[lab]["gated_out"] = removed
    for r in runs.values():
        r.update(kept_dropped(r["selected"], d_w["published"]))

    # post hoc on every published modifier and every new selection
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
