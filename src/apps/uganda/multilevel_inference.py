"""
Multilevel inference for the Uganda YOP NEXIS analysis.

The appendix (sec:uganda:limitations) argues that no standard clustered approach
suits a candidate pool spanning several levels of nesting, and leaves a multilevel
test to future work.  There is in fact a standard answer, and this module implements
it.  The rule (Cameron & Miller 2015, §II.C) is to cluster at the *coarsest level at
which the regressor of interest varies* — which, for a pool that spans levels, means
the cluster level is a property of the candidate, not of the analysis.  NEXIS already
accommodates this: it consumes only p-values and t-statistics from its conditional
test, so a level-aware test drops straight in via `nexis(pvalue_fn=...)`.

Three tests are provided, in increasing order of robustness and cost:

  (1) `levelwise_pvalues`   CR1S cluster-robust, each candidate clustered at its own
                            level.  Cheap enough to run inside the selection loop.
  (2) `wild_cluster_boot`   Restricted (null-imposed) wild cluster bootstrap-t with
                            Rademacher weights — the standard remedy when the
                            candidate's own level has few clusters.  Exhaustively
                            enumerated when 2^G is small, so the p-value is exact.
  (3) `randomization_test`  Design-based.  Re-randomises T over groups within the 14
                            district blocks that Blattman et al. actually used, under
                            a constant-effect null.  Makes no assumption whatever
                            about the correlation structure of Y.

The level hierarchy on the analysis sample (n = 2082), coarsest first:

    region  (lang_group)        7      \\
    district / block           14       |  strict chain
    community (site)          327      /
    group (randomisation)     439      crosses community for 30 of 439 groups
    individual               2082

Usage
-----
    python src/apps/uganda/multilevel_inference.py \\
        --embed-model prithvi_l5 --sae-dim 1024 \\
        --outcomes skilled_employed,log_biz_assets
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from method.nexis import nexis                                   # noqa: E402
from apps.uganda.robustness_clustering import build_dataset      # noqa: E402

# Coarsest → finest.  'region' is lang_group; 'block' is the randomisation block,
# recovered as ceil(strata/2) and coinciding with district.
LEVEL_ORDER = ["region", "block", "community", "group", "individual"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default="prithvi_l5")
    p.add_argument("--sae-dim", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--active-threshold", type=int, default=5)
    p.add_argument("--outcomes", default="skilled_employed,log_biz_assets")
    p.add_argument("--n-boot", type=int, default=9999)
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Levels
# ──────────────────────────────────────────────────────────────────────────────
def build_levels(df):
    """Return {level_name: integer cluster code array} on the analysis sample."""
    block = np.ceil(df["strata"].values / 2).astype(int)
    return {
        "region":     pd.factorize(df["lang_group"].astype(str))[0],
        "block":      pd.factorize(pd.Series(block).astype(str))[0],
        "community":  pd.factorize(df["geo_long_lat_key"].astype(str))[0],
        "group":      pd.factorize(df["groupid"].astype(str))[0],
        "individual": np.arange(len(df)),
    }


def assign_levels(data, levels, tol=1e-10):
    """Coarsest level at which each candidate is constant within every cell.

    A variable constant within a coarse cell is trivially constant within every
    finer cell, so scanning coarse→fine and taking the first hit gives the level
    at which the variable genuinely varies.  This is what determines the correct
    cluster level for that candidate.
    """
    Z = data["Z_full"]
    out = []
    for j, lab in enumerate(data["labels"]):
        col = Z[:, j]
        scale = np.nanstd(col) or 1.0
        lvl = "individual"
        for name in LEVEL_ORDER:
            g = levels[name]
            s = pd.Series(col).groupby(g).transform("std").fillna(0.0).max()
            if s / scale < tol:
                lvl = name
                break
        out.append(lvl)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared design construction
# ──────────────────────────────────────────────────────────────────────────────
def block_dummies(block_codes):
    """Block fixed effects, first level dropped (intercept is already in design)."""
    u = np.unique(block_codes)
    return np.column_stack([(block_codes == b).astype(float) for b in u[1:]])


def design(y, t, Z, S, j, fe=None):
    """Full design [1, T, FE, Z_S, T*Z_S, Z_j, T*Z_j]; interaction is the last column.

    `fe` holds block fixed effects.  Treatment was randomised within 14 district
    blocks whose treated share ranges from 0.20 to 0.62, so block indicators are
    needed for unbiasedness, not merely efficiency: without them the pooled estimate
    absorbs the correlation between a block's treatment propensity and its outcome
    level.  This — not clustering — is the correct way to respect a blocked design.
    """
    cols = [np.ones(len(y)), t]
    if fe is not None and fe.shape[1]:
        cols.extend(fe.T)
    for k in S:
        cols.append(Z[:, k])
    for k in S:
        cols.append(t * Z[:, k])
    cols.append(Z[:, j])
    cols.append(t * Z[:, j])
    return np.column_stack(cols)


def rank_filter(X, tol=1e-9):
    """Indices of a maximal independent column subset, always keeping the last column.

    With block fixed effects the design is rank-deficient by construction: every
    `lang_*` main effect is an exact sum of district dummies (a language group *is* a
    set of districts), so those main effects are absorbed.  The T x lang_j interaction
    is not absorbed and stays identified, but `inv(X'X)` on the singular matrix
    silently returns garbage and statsmodels returns NaN, so the redundant columns
    must be dropped before any variance is computed.  Computed once per specification
    and reused across bootstrap / permutation draws.
    """
    k = X.shape[1]
    keep = []
    for c in range(k - 1):
        trial = keep + [c]
        if np.linalg.matrix_rank(X[:, trial], tol=tol) == len(trial):
            keep.append(c)
    keep.append(k - 1)
    if np.linalg.matrix_rank(X[:, keep], tol=tol) != len(keep):
        return None                       # interaction itself absorbed: not testable
    return np.array(keep, dtype=int)


def _cluster_index(g):
    """Pre-sort rows by cluster so score sums become one reduceat call."""
    order = np.argsort(g, kind="stable")
    gs = np.asarray(g)[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    return order, starts, len(starts)


def _cr_t(y, X, g, idx, pre=None):
    """OLS coefficient idx and its CR1S cluster-robust t-statistic.

    `pre` is an optional (order, starts, G, XtXi) tuple so the bootstrap loop can
    hoist the sort and the (X'X)^-1 out of the inner iteration — X is fixed across
    wild-bootstrap draws, only y changes.
    """
    n, k = X.shape
    if pre is None:
        order, starts, G = _cluster_index(g)
        XtXi = np.linalg.pinv(X.T @ X)
    else:
        order, starts, G, XtXi = pre
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    sc = X * e[:, None]
    sums = np.add.reduceat(sc[order], starts, axis=0)       # (G, k)
    meat = sums.T @ sums
    meat *= (G / (G - 1)) * ((n - 1) / (n - k))
    V = XtXi @ meat @ XtXi
    se = np.sqrt(max(V[idx, idx], 1e-300))
    return float(b[idx]), float(b[idx] / se), G


# ──────────────────────────────────────────────────────────────────────────────
# (1) Level-aware CR1S test — plugs into nexis(pvalue_fn=...)
# ──────────────────────────────────────────────────────────────────────────────
def make_levelwise_pvalue_fn(level_of, levels):
    """Build a pvalue_fn that clusters each candidate at its own level.

    Candidates are partitioned by level and the vectorised core test is called once
    per level, so cost is ~#levels x the single-cluster cost rather than per-candidate.
    """
    from method.nexis import conditional_interaction_pvalues as cip

    lvl_arr = np.asarray(level_of)

    def fn(y, t, z, S, candidates, return_tstats=False):
        m = z.shape[1]
        pv = np.ones(m)
        ts = np.zeros(m)
        cand = np.asarray(list(candidates), dtype=int)
        if cand.size == 0:
            return (pv, ts) if return_tstats else pv
        for name in set(lvl_arr[cand]):
            sub = cand[lvl_arr[cand] == name]
            kw = {} if name == "individual" else {"cluster": levels[name]}
            if name == "individual":
                kw = {"hc1": True}      # no clustering possible; use HC1
            p, tt = cip(y=y, t=t, z=z, S=S, candidates=sub.tolist(),
                        return_tstats=True, **kw)
            pv[sub] = p[sub]
            ts[sub] = tt[sub]
        return (pv, ts) if return_tstats else pv

    return fn


# ──────────────────────────────────────────────────────────────────────────────
# (2) Restricted wild cluster bootstrap-t (WCR), Rademacher
# ──────────────────────────────────────────────────────────────────────────────
def wild_cluster_boot(y, t, Z, S, j, g, n_boot=9999, rng=None, fe=None):
    """Null-imposed wild cluster bootstrap-t for H0: gamma_j = 0.

    Cameron, Gelbach & Miller (2008); MacKinnon & Webb (2018).  Residuals are drawn
    under the restricted model (interaction excluded) and perturbed by a single
    Rademacher weight per cluster.  When 2^G <= 2^14 every sign vector is enumerated,
    making the p-value exact rather than simulated — and exposing the resolution
    floor: with G clusters no such test can return a p-value below 2^-(G-1).
    """
    rng = rng or np.random.default_rng(0)
    X = design(y, t, Z, S, j, fe)
    keep = rank_filter(X)
    if keep is None:
        return dict(t_obs=np.nan, G=0, n_draws=0, exact=False,
                    pvalue=np.nan, resolution_floor=np.nan,
                    note="interaction absorbed by fixed effects")
    X = X[:, keep]
    idx = X.shape[1] - 1
    order, starts, G = _cluster_index(g)
    pre = (order, starts, G, np.linalg.pinv(X.T @ X))
    _, t_obs, _ = _cr_t(y, X, g, idx, pre)

    Xr = X[:, :idx]                                    # restricted: drop T*Z_j
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
    fit = Xr @ br
    resid = y - fit

    uniq = np.unique(g)
    codes = np.searchsorted(uniq, g)

    exact = G <= 14
    if exact:
        signs = np.array(list(product([-1.0, 1.0], repeat=G)))
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_boot, G))

    count = 0
    for v in signs:
        y_star = fit + v[codes] * resid
        _, t_star, _ = _cr_t(y_star, X, g, idx, pre)
        if abs(t_star) >= abs(t_obs) - 1e-12:
            count += 1
    B = len(signs)
    p = count / B if exact else (1 + count) / (B + 1)
    return dict(t_obs=t_obs, G=int(G), n_draws=int(B), exact=bool(exact),
                pvalue=float(p), resolution_floor=float(1.0 / B))


# ──────────────────────────────────────────────────────────────────────────────
# (3) Design-based randomization inference
# ──────────────────────────────────────────────────────────────────────────────
def randomization_test(y, t, Z, S, j, group_codes, block_codes,
                       n_perm=9999, rng=None, fe=None):
    """Re-randomise T over groups within randomisation blocks.

    Blattman et al. assigned treatment to groups within 14 district blocks, so the
    exact design permutes group-level assignment holding the number of treated
    groups per block fixed.  We test the constant-effect null: Y_i(0) is imputed as
    Y_i - tau_hat * T_i with tau_hat the pooled OLS effect, T is re-randomised, the
    outcome is rebuilt, and the studentised interaction t-statistic is recomputed.
    This makes no assumption about the correlation structure of Y — the only
    randomness used is the one the experimenters created.
    """
    rng = rng or np.random.default_rng(0)
    n = len(y)

    # group-level table
    gu, ginv = np.unique(group_codes, return_inverse=True)
    g_block = np.array([block_codes[group_codes == g][0] for g in gu])
    g_treat = np.array([t[group_codes == g][0] for g in gu])

    # Pre-slice the parts of the design that do not depend on T.
    Z_S = Z[:, list(S)] if len(S) else np.empty((n, 0))
    z_j = Z[:, j]
    FE = fe if fe is not None else np.empty((n, 0))
    nf, ns = FE.shape[1], Z_S.shape[1]
    k = 2 + nf + 2 * ns + 2
    idx = k - 1
    ones = np.ones(n)
    o_fe, o_zs = 2, 2 + nf                    # column offsets, fixed across draws

    # Collinear columns (lang main effects vs block FE) are dropped once, from the
    # observed design, and the same specification is reused for every permutation.
    keep = rank_filter(design(y, t, Z, S, j, fe))
    if keep is None:
        return dict(t_obs=np.nan, n_perm=0, pvalue=np.nan,
                    resolution_floor=np.nan, tau_hat=np.nan,
                    note="interaction absorbed by fixed effects")
    kk = len(keep)
    idx_k = kk - 1

    def build(tt):
        XX = np.empty((n, k))
        XX[:, 0] = ones
        XX[:, 1] = tt
        if nf:
            XX[:, o_fe:o_fe + nf] = FE
        if ns:
            XX[:, o_zs:o_zs + ns] = Z_S
            XX[:, o_zs + ns:o_zs + 2 * ns] = tt[:, None] * Z_S
        XX[:, idx - 1] = z_j
        XX[:, idx] = tt * z_j
        return XX[:, keep]

    def tstat(yy, tt):
        XX = build(tt)
        G_ = XX.T @ XX
        try:
            Gi = np.linalg.inv(G_)
        except np.linalg.LinAlgError:                       # pragma: no cover
            Gi = np.linalg.pinv(G_)
        b = Gi @ (XX.T @ yy)
        e = yy - XX @ b
        s2 = (e @ e) / (n - kk)
        se = np.sqrt(max(s2 * Gi[idx_k, idx_k], 1e-300))
        return b[idx_k] / se

    t_obs = tstat(y, t)

    # constant-effect null: impute Y(0)
    Xt = np.column_stack([np.ones(n), t])
    tau = float(np.linalg.lstsq(Xt, y, rcond=None)[0][1])
    y0 = y - tau * t

    blocks = [np.where(g_block == b)[0] for b in np.unique(g_block)]
    n_tr = [int(g_treat[b].sum()) for b in blocks]

    count = 0
    for _ in range(n_perm):
        gt = np.zeros(len(gu))
        # NB: do not bind `k` here — `tstat` closes over the design width `k`.
        for blk, n_t in zip(blocks, n_tr):
            gt[rng.choice(blk, size=n_t, replace=False)] = 1.0
        t_star = gt[ginv]
        y_star = y0 + tau * t_star
        if abs(tstat(y_star, t_star)) >= abs(t_obs) - 1e-12:
            count += 1
    return dict(t_obs=float(t_obs), n_perm=int(n_perm),
                pvalue=float((1 + count) / (n_perm + 1)),
                resolution_floor=float(1.0 / (n_perm + 1)),
                tau_hat=tau)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"
        / "multilevel_inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    PUBLISHED = {
        "skilled_employed": ["W_lang_4", "W_lang_2", "Z_339", "Z_533", "W_lang_7"],
        "log_biz_assets":   ["W_ndvi_mean", "Z_820"],
    }
    report = {"embed_model": args.embed_model, "sae_dim": args.sae_dim,
              "outcomes": {}}

    for outcome in [o.strip() for o in args.outcomes.split(",")]:
        data = build_dataset(args, outcome)
        if data is None:
            continue
        df = data["df"]
        levels = build_levels(df)
        level_of = assign_levels(data, levels)
        lab2idx = {l: i for i, l in enumerate(data["labels"])}
        m = data["Z_full"].shape[1]

        print("=" * 78)
        print(f"OUTCOME {outcome}   n={len(df)}   candidates={m}")
        print("  cluster counts:",
              {k: int(len(np.unique(v))) for k, v in levels.items()})
        lv = pd.Series(level_of).value_counts().to_dict()
        print("  candidate levels (coarsest at which each varies):", lv)

        res = {"n": int(len(df)),
               "cluster_counts": {k: int(len(np.unique(v)))
                                  for k, v in levels.items()},
               "candidate_levels": {k: int(v) for k, v in lv.items()},
               "levels_by_feature": {data["labels"][i]: level_of[i]
                                     for i in range(m)}}

        # ── support at each candidate's OWN level ─────────────────────────────
        # A sandwich estimator at level L needs enough clusters that are active-and-
        # treated and active-and-control; below ~10 the meat matrix collapses and the
        # SE is meaningless (see robustness_clustering.py).  Evaluating the gate at
        # each candidate's own level is what makes the level-aware run interpretable.
        T = data["T"]
        sup_ok = np.zeros(m, dtype=bool)
        sup_detail = {}
        for j in range(m):
            g = levels[level_of[j]]
            act = data["Z_full"][:, j] != 0
            n_t = len(np.unique(g[act & (T == 1)]))
            n_c = len(np.unique(g[act & (T == 0)]))
            sup_ok[j] = (n_t >= 10) and (n_c >= 10)
            sup_detail[data["labels"][j]] = [int(n_t), int(n_c)]
        cols = np.flatnonzero(sup_ok)
        dropped_lvl = pd.Series([level_of[j] for j in np.flatnonzero(~sup_ok)]
                                ).value_counts().to_dict()
        print(f"  support gate at own level: {len(cols)}/{m} pass; "
              f"dropped by level: {dropped_lvl}")
        res["support_gate"] = {
            "n_pass": int(len(cols)), "n_total": int(m),
            "dropped_by_level": {k: int(v) for k, v in dropped_lvl.items()},
            "published_support": {l: sup_detail[l] for l in PUBLISHED[outcome]}}

        # ── level-aware NEXIS re-selection ────────────────────────────────────
        fn = make_levelwise_pvalue_fn(level_of, levels)
        sel = {}
        for pool, cc in [("full", None), ("supported", cols)]:
            Zp = data["Z_full"] if cc is None else data["Z_full"][:, cc]
            fnp = fn if cc is None else make_levelwise_pvalue_fn(
                [level_of[j] for j in cc], levels)
            for adjust in ["FWER", "FDR"]:
                r = nexis(data["Y"], data["T"], Zp, alpha=args.alpha,
                          max_rounds=args.max_steps, adjust=adjust,
                          pvalue_fn=fnp, verbose=False)
                back = (lambda k: k) if cc is None else (lambda k: int(cc[k]))
                sel[f"{pool}_{adjust}"] = [
                    {"label": data["labels"][back(k)],
                     "level": level_of[back(k)],
                     "pvalue": float(r.pvalues[k])} for k in r.selected]
                print(f"  level-aware NEXIS [{pool:9s}] {adjust}: "
                      f"{[s['label'] for s in sel[f'{pool}_{adjust}']]}")
        res["levelwise_nexis"] = sel

        # ── confirmatory tests on the published features ──────────────────────
        pub = PUBLISHED[outcome]
        S_pub = [lab2idx[l] for l in pub]
        FE = block_dummies(levels["block"])
        rows = []
        for lab in pub:
            j = lab2idx[lab]
            S = [k for k in S_pub if k != j]
            lvl = level_of[j]
            g = levels[lvl]
            gate = 0.05 / (m - len(S))

            zj = data["Z_full"][:, j] != 0
            n_act = {nm: int(len(np.unique(levels[nm][zj])))
                     for nm in ["region", "block", "community"]}

            rng = np.random.default_rng(args.seed)
            wcb = wild_cluster_boot(data["Y"], data["T"], data["Z_full"],
                                    S, j, g, n_boot=args.n_boot, rng=rng)
            rng = np.random.default_rng(args.seed + 1)
            ri = randomization_test(data["Y"], data["T"], data["Z_full"], S, j,
                                    levels["group"], levels["block"],
                                    n_perm=args.n_perm, rng=rng)
            # Blocked design handled by conditioning, not clustering: add block
            # fixed effects and re-run both tests.
            rng = np.random.default_rng(args.seed)
            wcb_fe = wild_cluster_boot(data["Y"], data["T"], data["Z_full"],
                                       S, j, g, n_boot=args.n_boot, rng=rng, fe=FE)
            rng = np.random.default_rng(args.seed + 1)
            ri_fe = randomization_test(data["Y"], data["T"], data["Z_full"], S, j,
                                       levels["group"], levels["block"],
                                       n_perm=args.n_perm, rng=rng, fe=FE)
            rows.append(dict(feature=lab, level=lvl, n_clusters=wcb["G"],
                             n_active_regions=n_act["region"],
                             n_active_blocks=n_act["block"],
                             n_active_communities=n_act["community"],
                             bonferroni_gate=gate,
                             wcb_p=wcb["pvalue"], wcb_exact=wcb["exact"],
                             wcb_floor=wcb["resolution_floor"],
                             wcb_passes=bool(wcb["pvalue"] <= gate),
                             ri_p=ri["pvalue"], ri_floor=ri["resolution_floor"],
                             ri_passes=bool(ri["pvalue"] <= gate),
                             wcb_blockfe_p=wcb_fe["pvalue"],
                             ri_blockfe_p=ri_fe["pvalue"],
                             ri_blockfe_passes=bool(ri_fe["pvalue"] <= gate)))
            print(f"    {lab:12s} level={lvl:10s} G={wcb['G']:4d} "
                  f"active(reg/blk/comm)={n_act['region']}/{n_act['block']}/"
                  f"{n_act['community']}  WCB={wcb['pvalue']:.4g}"
                  f"{'*' if wcb['exact'] else ''}  RI={ri['pvalue']:.4g}  "
                  f"|+blockFE| WCB={wcb_fe['pvalue']:.4g} RI={ri_fe['pvalue']:.4g}  "
                  f"gate={gate:.2e}")
        tab = pd.DataFrame(rows)
        tab.to_csv(out_dir / f"{outcome}_multilevel.csv", index=False)
        res["confirmatory"] = rows
        report["outcomes"][outcome] = res
        print()

    path = out_dir / "multilevel_inference.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
