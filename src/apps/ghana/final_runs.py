"""Ghana LEAP 1000 block of the final real-world runs (src/apps/realworld_final_runs.py).

First-differenced consumption; both NEXIS variants with
  (i)  the published test (CR1S by community for every candidate) on the joint pool of
       131 SAE neurons + 24 survey covariates = 155, as the appendix counts it;
  (ii) the level-aware test (HC1 for the household covariates, CR1S by community for
       the neurons), support gate k = 5, same pool;
  (i') and (ii') the same on the pool that also holds the 12 community-level spectral
       indices (167), as the main text describes it.
Post hoc, for every published modifier and every new selection, each conditional on the
rest of the set it was selected with: CR1S by community (t(G-1)) and a restricted wild
cluster bootstrap-t by community.
"""

from __future__ import annotations

import time

import pandas as pd

from apps.ghana.pool import ghana, ghana_with_spectral
from causality.multilevel import (ALPHA, MIN_SUPPORT, PUBLISHED_CFG, VARIANTS, LevelAwareTest,
                                  build_rule, cell_str, cells, kept_dropped, plain_test, run,
                                  wild_cluster_bootstrap)


def posthoc_ghana(d, rule, pub_fn, S, j, n_boot):
    keys, parts, _ = rule
    wb = wild_cluster_bootstrap(d, S, j, d["levels"]["community"], n_boot)
    return dict(p_community=float(pub_fn(d["y"], d["t"], d["z"], S, [j])[j]),
                p_wild=wb["p"], wild_floor=wb["floor"], wild_degenerate_draws=wb["degenerate_draws"],
                t_obs=wb["t_obs"], cells=cells(d, keys, parts, j))


def ghana_block(args):
    d155 = ghana()
    d167 = ghana_with_spectral(d155)
    runs, rules, pubfns, datas = {}, {}, {}, {"155": d155, "167": d167}
    spec_p = {}
    for pname, d in datas.items():
        m = d["z"].shape[1]
        print(f"\n{'=' * 100}\nghana / consumption   n={len(d['y'])}   pool={m}")
        rule = rules[pname] = build_rule(d)
        keys, parts, info = rule
        pool = [j for j, i in enumerate(info) if i["testable"]]
        removed = [i["name"] for i in info if not i["testable"]]
        print(f"  levels: {pd.Series([i['level'] for i in info]).value_counts().to_dict()}  "
              f"support gate k={MIN_SUPPORT} keeps {len(pool)}/{m}; published removed: "
              f"{[n for n in removed if n in d['published']]}")
        pubfns[pname] = plain_test(cluster=d["levels"]["community"], m=m)
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
            runs[lab] = run(d, "i", pubfns[pname], cfg, rule=rule)
            if pname == "155" and cfg is PUBLISHED_CFG and sorted(runs[lab]["selected"]) != sorted(d["published"]):
                raise SystemExit(f"published set NOT reproduced: {runs[lab]['selected']} vs {d['published']}")
            lab = f"pool {pname} | level-aware k={MIN_SUPPORT} | {vname}"
            print(f"  -- {lab}")
            runs[lab] = run(d, "ii", LevelAwareTest(keys, parts), cfg, pool, info, rule=rule)
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
