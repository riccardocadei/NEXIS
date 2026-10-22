"""Shared harness of the real-world NEXIS runs behind the paper's application results.

The run configurations (alpha, rho, the two NEXIS variants), one reported NEXIS run
with per-coordinate p-values (`run`) and the cluster-count reporting helpers.  Used by
src/apps/realworld_final_runs.py, src/apps/{uganda,ghana}/final_runs.py,
src/apps/uganda/multilevel_groupcluster.py and src/apps/uganda/district_sensitivity.py.
The statistics live in the library: nexis.design (level-aware test, support rule) and
nexis.checks (post-hoc checks).
"""

from __future__ import annotations

import numpy as np

from nexis import nexis
from nexis.checks import worst_subset
from nexis.design import LevelAwareTest

ALPHA, RHO = 0.05, 0.5
PUBLISHED_CFG = dict(backward=True, terminal_filter=False)
NEW_DEFAULT = dict(backward=False, terminal_filter=True)
VARIANTS = {"published NEXIS": PUBLISHED_CFG, "new default": NEW_DEFAULT}


# ── Reporting helpers ─────────────────────────────────────────────────────────

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


def kept_dropped(sel, published):
    return dict(kept=[n for n in published if n in sel], dropped=[n for n in published if n not in sel],
                new=[n for n in sel if n not in published])


def run(d, label, fn, cfg, pool=None, info=None, rule=None):
    """One NEXIS run on the columns `pool` of d["z"] with the test `fn`, and for every
    coordinate of S~ and of the final set its marginal p, its p given the rest of S~
    and of the final set, and its worst-subset p.  `info` (from build_rule) adds each
    coordinate's level and cluster; `rule` (the output of build_rule) adds its cluster
    counts per cell."""
    y, t = d["y"], d["t"]
    pool = list(range(d["z"].shape[1])) if pool is None else list(pool)
    z = d["z"][:, pool]
    if isinstance(fn, LevelAwareTest):
        fn = fn.subset(pool)
    nm = [d["names"][k] for k in pool]
    r = nexis(y, t, z, alpha=ALPHA, max_rounds=20, adjust="FWER", rho=RHO,
              pvalue_fn=fn, **cfg)
    m = z.shape[1]
    gate = ALPHA / m
    sel = list(r.selected)
    S_tilde = list(r.metadata["terminal_candidates"]) if cfg["terminal_filter"] else sel
    rows = []
    for j in sorted(set(S_tilde) | set(sel), key=lambda k: S_tilde.index(k) if k in S_tilde else 99):
        p_marg = float(fn(y, t, z, [], [j])[j])
        p_cond = float(fn(y, t, z, [k for k in S_tilde if k != j], [j])[j])
        p_final = float(fn(y, t, z, [k for k in sel if k != j], [j])[j]) if j in sel else None
        wp, wa, exact = worst_subset(fn, y, t, z, S_tilde, j, gate)
        row = dict(name=nm[j], selected=j in sel, p_marginal=p_marg,
                   p_cond_Stilde=p_cond, p_cond_final=p_final,
                   worst_subset_p=wp, worst_A=[nm[k] for k in wa], worst_exact=exact,
                   passes_terminal=bool(wp <= gate))
        if info is not None:
            ii = info[pool[j]]
            row.update(level=ii["level"], cluster=ii["cluster"], G=ii["G"])
        rows.append(row)
    out = dict(run=label, config="published (interleaved backward)" if cfg["backward"]
               else "new default (forward + terminal)", m=m, gate=gate,
               S_tilde=[nm[k] for k in S_tilde], selected=[nm[k] for k in sel], coords=rows)
    print(f"  [{label:3s}] m={m:3d} a/m={gate:.2e}  S~={out['S_tilde']}  ->  {out['selected']}")
    for rw in rows:
        lvl = f" {rw.get('cluster', ''):>16s} G={rw.get('G', '')}" if info is not None else ""
        pf = f"{rw['p_cond_final']:.1e}" if rw["p_cond_final"] is not None else "   -   "
        print(f"        {rw['name']:18s}{lvl}  marg {rw['p_marginal']:.1e}  cond|S~ {rw['p_cond_Stilde']:.1e}"
              f"  cond|final {pf}  worst {'' if rw['worst_exact'] else '>='}{rw['worst_subset_p']:.1e} at {rw['worst_A']}"
              f"  {'sel' if rw['selected'] else ''}")
    if rule is not None:
        idx = {n: j for j, n in enumerate(d["names"])}
        keys, parts, _ = rule
        for row in rows:
            row["cells"] = cells(d, keys, parts, idx[row["name"]])
        for row in rows:
            if row["selected"]:
                print(f"        cells {row['name']:14s} {cell_str(row['cells'])}")
    return out
