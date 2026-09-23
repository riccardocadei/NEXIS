#!/usr/bin/env python3
"""Diagnose what the spectral-gap gate rho removes in NEXIS on the CelebA benchmark.

Companion of ablation_rho_filter.py (same SCM, seeds, truth, alpha, max_rounds).  For a
few high-power cells it re-runs, on the same simulated data,

  rho05_filter  nexis(rho=0.5, terminal_filter=True)
  rho0_filter   nexis(rho=0,   terminal_filter=True)

and logs S~, S_hat, the terminal log and the forward path (|t| of every admitted
candidate and of the candidate stopped by the rho gate, recorded through a pvalue_fn
wrapper around the default linear test; nexis.py is untouched).

"rho-blocked FP" = j in S_hat(rho0_filter) minus S_hat(rho05_filter) minus S*.

Per run it also stores t(j | S*) for every coordinate j (the test the terminal filter
must reject on the recall event), so the recurring coordinates can be tracked in every
run, not only where they were selected.

Phase `run` (heavy; one pickle per cell) and phase `analyze` (population diagnostics,
CelebA attribute AUCs, CATE-leak test, candidate replacement rules, summary.md).

Usage:
  diagnose_rho.py run <tree> <sweep> <fixed> <param> [--seeds N] [--overwrite]
  diagnose_rho.py analyze
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.scm import build_buckets, generate_celeba_rct  # noqa: E402
from method.nexis import conditional_interaction_pvalues, nexis  # noqa: E402

ALPHA, MAX_STEPS, P_TREAT = 0.05, 10, 0.5
OUT = ROOT / "results/celeba/ablation_rho_filter/diagnose_rho"
SCM = dict(w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
           tau_0=0.5, gamma_w1=1.0, gamma_w2=-1.0, noise_sd=1.0)
BETA = (0.3, -0.2)  # scm defaults beta_w1, beta_w2


def load_tree(tree):
    kdir, ftype = tree.split("/")
    k = int(kdir[1:])
    fname = f"sae_k{k}.npy" if ftype == "sae" else f"sae_precode_k{k}.npy"
    features = np.load(ROOT / "data/celeba/embeddings" / fname)
    gt = json.load(open(ROOT / f"results/celeba/experiment/{tree}/ground_truth.json"))
    return features, gt


def ols_coefs(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def interaction_gamma(y, t, Z, S, j):
    """gamma_j in Y ~ 1 + T + Z_{S+j} + T*Z_{S+j} (the working model of the linear test)."""
    cols = list(S) + [j]
    X = np.column_stack([np.ones_like(t), t, Z[:, cols], t[:, None] * Z[:, cols]])
    b = ols_coefs(y, X)
    return float(b[2 + 2 * len(cols) - 1])


def hc3_t(y, t, Z, S, j):
    """HC3 t-statistic of gamma_j in Y ~ 1 + T + Z_{S+j} + T*Z_{S+j}."""
    cols = list(S) + [j]
    X = np.column_stack([np.ones_like(t), t, Z[:, cols], t[:, None] * Z[:, cols]])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    u = e / np.clip(1 - h, 1e-8, None)
    V = XtXi @ (X.T * u ** 2) @ X @ XtXi
    k = X.shape[1] - 1
    return float(b[k] / np.sqrt(V[k, k])) if V[k, k] > 0 else float("nan")


def hc3_vec(y, t, Z, S, cands):
    """Vectorised HC3 t-statistics of gamma_j (T*Z_j) given S, for j in cands (FWL)."""
    n = len(y)
    D = np.column_stack([np.ones(n), t] + [Z[:, k] for k in S] + [t * Z[:, k] for k in S])
    Q, _ = np.linalg.qr(D)
    hD = (Q ** 2).sum(1)
    res = lambda A: A - Q @ (Q.T @ A)
    yt = res(y)
    Zc = Z[:, cands]
    zt = res(Zc)
    xt = res(t[:, None] * Zc)
    zz, xx, zx = (zt * zt).sum(0), (xt * xt).sum(0), (zt * xt).sum(0)
    zy, xy = (zt * yt[:, None]).sum(0), (xt * yt[:, None]).sum(0)
    det = zz * xx - zx ** 2
    ok = det > 1e-12
    det = np.where(ok, det, 1.0)
    bx = (zz * xy - zx * zy) / det
    bz = (xx * zy - zx * xy) / det
    e = yt[:, None] - zt * bz - xt * bx
    h = hD[:, None] + (xx * zt ** 2 - 2 * zx * zt * xt + zz * xt ** 2) / det
    a = (zz * xt - zx * zt) / det          # gamma_hat = sum_i a_i y_i
    v = (a ** 2 * e ** 2 / np.clip(1 - h, 1e-8, None) ** 2).sum(0)
    tt = np.where(ok & (v > 0), bx / np.sqrt(np.where(v > 0, v, 1.0)), 0.0)
    return tt


def hc3_terminal_filter(y, t, Z, St, tcrit):
    """Subset-robust terminal filter (as in nexis) with the HC3 test instead."""
    from itertools import combinations
    keep = []
    for j in St:
        others = [s for s in St if s != j]
        ok = True
        for r in range(len(others), -1, -1):
            for A in combinations(others, r):
                if abs(hc3_vec(y, t, Z, list(A), [j])[0]) <= tcrit:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            keep.append(int(j))
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# phase: run
# ─────────────────────────────────────────────────────────────────────────────

def run_cell(tree, sweep, fixed, param, n_seeds, overwrite):
    tag = f"{tree.replace('/', '_')}_{sweep}_{fixed:g}_{param:g}"
    out_path = OUT / "runs" / f"{tag}.pkl"
    if out_path.exists() and not overwrite:
        print(f"exists, skipping: {out_path}")
        return
    features, gt = load_tree(tree)
    truth = sorted(set(gt["truth"]))
    labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
    M = features.shape[1]
    n = int(param) if sweep == "n" else int(fixed)
    effect = float(param) if sweep == "effect" else float(fixed)
    ref = None
    ref_path = (ROOT / "results/celeba/ablation_rho_filter"
                / f"rf_{tree.replace('/', '_')}_{sweep}_{fixed:g}.csv")
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref = ref[ref.param == param]

    def one(seed):
        t0 = time.perf_counter()
        try:
            d = generate_celeba_rct(n=n, features=features, labels_df=labels_df,
                                    buckets=buckets, effect_scale=effect, seed=seed, **SCM)
        except ValueError:
            return None
        y, t, Z = d.Y, d.T, d.Z

        def run(rho):
            fwd = []   # forward calls: (S, j_star, |t|, p) or (S, None, ...)

            def pfn(y, t, z, S, candidates, return_tstats=False):
                res = conditional_interaction_pvalues(y=y, t=t, z=z, S=S,
                                                      candidates=candidates,
                                                      return_tstats=return_tstats)
                if return_tstats:   # only the forward step asks for t-statistics
                    p, ts = res
                    rem = [j for j in candidates if j not in S]
                    gate = ALPHA / len(rem)
                    el = [j for j in rem if p[j] <= gate]
                    if el:
                        js = max(el, key=lambda j: abs(ts[j]))
                        fwd.append((list(S), int(js), float(abs(ts[js])), float(p[js])))
                    else:
                        fwd.append((list(S), None, float("nan"), float("nan")))
                return res

            # nexis requests t-statistics only when rho is set, so for rho=0 the
            # forward path is not logged (it is not needed there).
            r = nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS, rho=rho,
                      backward=True, terminal_filter=True, terminal_max_size=14,
                      pvalue_fn=pfn)
            md = r.metadata
            # replay the rho gate on the logged forward path
            t_sel, stop = [], None
            for S, js, tv, pv in fwd:
                if js is None:
                    continue
                if t_sel and tv < rho * min(t_sel):
                    stop = dict(S=S, j=js, t_new=tv, t_min=min(t_sel),
                                ratio=tv / min(t_sel))
                    break
                t_sel.append(tv)
            return dict(St=[int(j) for j in md["terminal_candidates"]],
                        S=[int(j) for j in r.selected],
                        tlog=md["terminal_log"], fwd=fwd, t_sel=t_sel, stop=stop)

        r05 = run(0.5)
        r0 = run(0.0)

        # t(j | S*) for all j
        p_star, t_star = conditional_interaction_pvalues(y=y, t=t, z=Z, S=truth,
                                                         return_tstats=True)
        _, t_star_hc1 = conditional_interaction_pvalues(y=y, t=t, z=Z, S=truth,
                                                        return_tstats=True, hc1=True)
        n1 = int(t.sum())
        nnz1 = (Z[t == 1] != 0).sum(0)
        nnz0 = (Z[t == 0] != 0).sum(0)

        def diag(j, other):
            """Diagnostics of coordinate j (not in S*)."""
            pj_s, tj_s = conditional_interaction_pvalues(y=y, t=t, z=Z, S=truth,
                                                         candidates=[j],
                                                         return_tstats=True)
            cond = sorted(set(truth) | set(other))
            pj_o, tj_o = conditional_interaction_pvalues(y=y, t=t, z=Z, S=cond,
                                                         candidates=[j],
                                                         return_tstats=True)
            ph, th = conditional_interaction_pvalues(y=y, t=t, z=Z, S=truth,
                                                     candidates=[j], return_tstats=True,
                                                     hc1=True)
            tk = {}
            for k in truth:
                Sk = [s for s in truth if s != k] + [j]
                _, tt = conditional_interaction_pvalues(y=y, t=t, z=Z, S=Sk,
                                                        candidates=[k],
                                                        return_tstats=True)
                tk[k] = float(abs(tt[k]))
            g = interaction_gamma(y, t, Z, truth, j)
            nzv = Z[:, j][Z[:, j] != 0]
            return dict(j=int(j), p_S=float(pj_s[j]), t_S=float(tj_s[j]),
                        p_S_hc1=float(ph[j]), t_S_hc1=float(th[j]),
                        p_So=float(pj_o[j]), t_So=float(tj_o[j]),
                        t_principals=tk, ratio_S=float(abs(tj_s[j]) / min(tk.values())),
                        gamma=g, gamma_x_mean_active=g * (float(nzv.mean()) if nzv.size else 0.0),
                        nnz1=int(nnz1[j]), nnz0=int(nnz0[j]))

        blocked = sorted(set(r0["S"]) - set(r05["S"]) - set(truth))
        bl = [diag(j, [s for s in r0["St"] if s != j]) for j in blocked]
        # per-element quantities on the rho0_filter output for replacement rules
        elem = []
        for j in r0["S"]:
            others = [s for s in r0["S"] if s != j]
            g = interaction_gamma(y, t, Z, others, j)
            nzv = Z[:, j][Z[:, j] != 0]
            elem.append(dict(j=int(j), gamma=g, sd=float(Z[:, j].std()),
                             mean_active=float(nzv.mean()) if nzv.size else 0.0,
                             nnz1=int(nnz1[j]), nnz0=int(nnz0[j]),
                             t_S=float(t_star[j]) if j not in truth else float("nan")))
        # sigma of the working model given S* (for noncentrality predictions)
        X = np.column_stack([np.ones(n), t, Z[:, truth], t[:, None] * Z[:, truth]])
        res = y - X @ ols_coefs(y, X)
        rec = dict(tree=tree, sweep=sweep, fixed=fixed, param=param, seed=seed, n=n,
                   effect=effect, n1=n1, truth=truth, r05=r05, r0=r0, blocked=bl,
                   elem=elem, sigma_S=float(res.std()),
                   n_rej_S=int((p_star <= ALPHA / M).sum() - 0),
                   rej_S=[int(j) for j in np.where(p_star <= ALPHA / M)[0]
                          if j not in truth],
                   t_star=t_star.astype(np.float32),
                   t_star_hc1=t_star_hc1.astype(np.float32), sec=time.perf_counter() - t0)
        if ref is not None:
            rr = ref[ref.seed == seed].set_index("variant")["selected"].fillna("")
            chk = {}
            for v, key in [("rho05_filter", ("r05", "S")), ("rho05", ("r05", "St")),
                           ("rho0_filter", ("r0", "S")), ("rho0", ("r0", "St"))]:
                if v in rr.index:
                    csv = sorted(int(x) for x in str(rr[v]).split())
                    chk[v] = csv == sorted(rec[key[0]][key[1]])
            rec["match_csv"] = chk
        print(f"{tag} seed={seed} S05={r05['S']} S0={r0['S']} blocked={blocked} "
              f"{rec['sec']:.1f}s", flush=True)
        return rec

    from joblib import Parallel, delayed
    nj = int(os.environ.get("NJOBS", "8"))
    recs = Parallel(n_jobs=nj, prefer="threads")(delayed(one)(s) for s in range(n_seeds))
    recs = [r for r in recs if r is not None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(recs, f)
    tmp.replace(out_path)
    mism = sum(1 for r in recs if not all(r.get("match_csv", {}).values()))
    print(f"wrote {len(recs)} runs -> {out_path}; csv mismatches={mism}")


# ─────────────────────────────────────────────────────────────────────────────
# phase: analyze
# ─────────────────────────────────────────────────────────────────────────────

def auc_matrix(x, L):
    """AUC of score x for each binary column of L (rank formula, ties averaged)."""
    from scipy.stats import rankdata
    r = rankdata(x)
    out = []
    for c in L.T:
        n1 = c.sum()
        n0 = len(c) - n1
        out.append((r[c == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.array(out)


def population(tree, features, labels_df, truth):
    """Population (CelebA-weighted) projection of the CATE on Z.

    Units are drawn as W1 ~ Bern(p1), W2 ~ Bern(p2) independently, then an image
    uniformly from the (W1, W2) bucket, so image i has weight P(cell_i)/|cell_i|.
    CATE = tau_0 + e*(W1 - W2); per unit effect u = W1 - W2.
    Returns per-coordinate: gamma_j (coef of Z_j in the weighted projection of u on
    [1, Z_S*, Z_j]), partial R^2 of Z_j for u given Z_S*, Var_perp(Z_j | Z_S*),
    and the residual variances needed to predict the noncentrality."""
    W1 = labels_df["Wearing_Hat"].values.astype(int)
    W2 = labels_df["Eyeglasses"].values.astype(int)
    p1, p2 = W1.mean(), W2.mean()
    cell = W1 * 2 + W2
    pc = np.array([(1 - p1) * (1 - p2), (1 - p1) * p2, p1 * (1 - p2), p1 * p2])
    cnt = np.bincount(cell, minlength=4)
    w = pc[cell] / cnt[cell]
    w = w / w.sum()
    u = (W1 - W2).astype(float)
    bw = BETA[0] * W1 + BETA[1] * W2
    Z = features.astype(np.float64)
    D = np.column_stack([np.ones(len(u)), Z[:, truth]])
    sw = np.sqrt(w)

    def resid(v):
        Dw = D * sw[:, None]
        coef = np.linalg.lstsq(Dw, v * (sw[:, None] if v.ndim == 2 else sw), rcond=None)[0]
        return v - D @ coef

    ru = resid(u)
    rb = resid(bw)
    RZ = resid(Z)
    vz = (w[:, None] * RZ ** 2).sum(0)
    cuz = (w[:, None] * RZ * ru[:, None]).sum(0)
    vu = float((w * ru ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(vz > 1e-12, cuz / vz, 0.0)
        pr2 = np.where(vz > 1e-12, cuz ** 2 / (vz * vu), 0.0)
    varu = float((w * (u - (w * u).sum()) ** 2).sum())
    r2_S = 1 - vu / varu
    freq = (w[:, None] * (Z != 0)).sum(0)
    return dict(w=w, gamma=gamma, pr2=pr2, vz=vz, vu_perp=vu, var_u=varu, r2_S=r2_S,
                vb_perp=float((w * rb ** 2).sum()), cov_bu_perp=float((w * rb * ru).sum()),
                freq=freq)


def analyze():
    runs = []
    for f in sorted((OUT / "runs").glob("*.pkl")):
        runs += pickle.load(open(f, "rb"))
    if not runs:
        sys.exit("no runs")
    labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    attrs = [c for c in labels_df.columns if c != "celeb_id"]
    L = labels_df[attrs].values.astype(int)
    lines = ["# What does rho remove? (CelebA, rho05_filter vs rho0_filter)\n"]
    lines.append("Script: src/apps/celeba/diagnose_rho.py.  rho-blocked FP = j kept by "
                 "rho0_filter, not by rho05_filter, not in S*.  All numbers measured.\n")

    # ── per cell counts ──
    cells = pd.DataFrame([dict(tree=r["tree"], sweep=r["sweep"], fixed=r["fixed"],
                               param=r["param"], seed=r["seed"],
                               n_blocked=len(r["blocked"]),
                               fp05=len(set(r["r05"]["S"]) - set(r["truth"])),
                               fp0=len(set(r["r0"]["S"]) - set(r["truth"])),
                               miss05=len((set(r["truth"]) & set(r["r0"]["S"]))
                                          - set(r["r05"]["S"])),
                               miss0=len((set(r["truth"]) & set(r["r05"]["S"]))
                                         - set(r["r0"]["S"])),
                               n_rej_S=len(r["rej_S"]),
                               csv_ok=all(r.get("match_csv", {"x": True}).values()))
                          for r in runs])
    cells.to_csv(OUT / "runs_summary.csv", index=False)
    g = cells.groupby(["tree", "sweep", "fixed", "param"]).agg(
        runs=("seed", "size"), blocked=("n_blocked", "sum"), fp05=("fp05", "mean"),
        fp0=("fp0", "mean"), miss05=("miss05", "sum"), miss0=("miss0", "sum"),
        runs_with_rej_S=("n_rej_S", lambda s: int((s > 0).sum())),
        csv_ok=("csv_ok", "mean")).reset_index()
    lines.append("## Cells\n")
    lines.append("fp05/fp0 = mean false positives per run of rho05_filter / rho0_filter; "
                 "miss05 = runs where rho05_filter misses a principal that rho0_filter "
                 "keeps; runs_with_rej_S = runs where some j not in S* has "
                 "p(j|S*) <= alpha/m (the filter cannot remove such j on the recall "
                 "event); csv_ok = fraction of runs whose selections equal the ablation "
                 "CSV.\n")
    lines.append(g.to_markdown(index=False, floatfmt=".3g") + "\n")

    # ── blocked FPs ──
    B = []
    for r in runs:
        for b in r["blocked"]:
            B.append(dict(tree=r["tree"], sweep=r["sweep"], param=r["param"],
                          seed=r["seed"], n=r["n"], effect=r["effect"],
                          n1=r["n1"], sigma_S=r["sigma_S"],
                          stop_j=(r["r05"]["stop"] or {}).get("j"),
                          stop_ratio=(r["r05"]["stop"] or {}).get("ratio"),
                          **{k: v for k, v in b.items() if k != "t_principals"},
                          min_t_principal=min(b["t_principals"].values())))
    B = pd.DataFrame(B)

    # population quantities per tree, and leverage-robust (HC3) re-test of every
    # blocked FP on regenerated data (same seed -> same sample)
    pops, hc3 = {}, {}
    buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
    for tree in sorted(set(r["tree"] for r in runs)):
        features, gt = load_tree(tree)
        truth = sorted(set(gt["truth"]))
        pop = population(tree, features, labels_df, truth)
        pops[tree] = {k: v for k, v in pop.items() if k != "w"}
        if len(B):
            for (sd, n_, e_), grp in B[B.tree == tree].groupby(["seed", "n", "effect"]):
                d = generate_celeba_rct(n=int(n_), features=features, labels_df=labels_df,
                                        buckets=buckets, effect_scale=float(e_),
                                        seed=int(sd), **SCM)
                for j in grp.j.unique():
                    hc3[(tree, sd, n_, e_, j)] = hc3_t(d.Y, d.T, d.Z, truth, int(j))
        del features
    if len(B):
        B["t_S_hc3"] = [hc3[(r.tree, r.seed, r.n, r.effect, r.j)] for r in B.itertuples()]
        B["pop_freq"] = [pops[r.tree]["freq"][r.j] for r in B.itertuples()]
        B["pop_pr2"] = [pops[r.tree]["pr2"][r.j] for r in B.itertuples()]
        B["pred_t"] = [r.effect * abs(pops[r.tree]["gamma"][r.j]) * np.sqrt(
            r.n * P_TREAT * (1 - P_TREAT) * pops[r.tree]["vz"][r.j]) / r.sigma_S
            for r in B.itertuples()]
    B.to_csv(OUT / "blocked_fps.csv", index=False)
    lines.append(f"## rho-blocked FPs\n\n{len(B)} rho-blocked FPs in {len(runs)} runs; "
                 f"{B.groupby('tree').j.nunique().to_dict()} distinct coordinates per "
                 f"tree.\n")
    from scipy.stats import norm
    tcrit_m = norm.isf(ALPHA / 9216 / 2)
    if len(B):
        lines.append(f"- p(j|S*) <= alpha/m in {(B.p_S <= ALPHA / 9216).mean():.3f} of "
                     f"them (median p(j|S*) {B.p_S.median():.2e}); "
                     f"median p(j|S* + rest of S~) {B.p_So.median():.2e}.\n"
                     f"- |t(j|S*)| / min_k |t(k|S*-k+j)|: median "
                     f"{B.ratio_S.median():.3f}, 90th pct {B.ratio_S.quantile(.9):.3f}, "
                     f"max {B.ratio_S.max():.3f}.\n"
                     f"- HC3 (leverage-robust) |t(j|S*)| > {tcrit_m:.2f} in "
                     f"{(B.t_S_hc3.abs() > tcrit_m).mean():.3f} of them (median "
                     f"|t_hc3|/|t_homo| {(B.t_S_hc3.abs() / B.t_S.abs()).median():.3f}).\n"
                     f"- predicted noncentrality of t(j|S*) from the population "
                     f"projection: median {B.pred_t.median():.2f}; >= 3 in "
                     f"{(B.pred_t >= 3).mean():.3f}, < 1 in {(B.pred_t < 1).mean():.3f}"
                     f"; pop partial R^2 median {B.pop_pr2.median():.4f}; active on < 1% "
                     f"of CelebA in {(B.pop_freq < 0.01).mean():.3f}.\n"
                     f"- rho05 gate stopped on this same j in "
                     f"{(B.stop_j == B.j).mean():.3f} of cases; gate ratio "
                     f"|t_new|/min|t_sel| median {B.stop_ratio.median():.3f}.\n"
                     f"- nonzeros per arm: min(nnz1,nnz0) median "
                     f"{np.minimum(B.nnz1, B.nnz0).median():.0f}, min "
                     f"{np.minimum(B.nnz1, B.nnz0).min()} ; <20 in "
                     f"{(np.minimum(B.nnz1, B.nnz0) < 20).mean():.3f}.\n")

    # ── per coordinate characterisation ──
    rows = []
    t_track = []
    for tree in sorted(set(r["tree"] for r in runs)):
        features, gt = load_tree(tree)
        truth = sorted(set(gt["truth"]))
        pop = pops[tree]
        tr_runs = [r for r in runs if r["tree"] == tree]
        # coordinates to characterise: all blocked + all rej_S ones
        Bt = B[B.tree == tree] if len(B) else B
        cnt_b = Bt.j.value_counts() if len(Bt) else pd.Series(dtype=int)
        cnt_rej = pd.Series([j for r in tr_runs for j in r["rej_S"]]).value_counts()
        coords = sorted(set(cnt_b.index) | set(cnt_rej.index[:20]))
        # rank of partial R^2 among all coords
        order = np.argsort(-pop["pr2"])
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        Zf = features
        for j in coords:
            x = Zf[:, j].astype(float)
            aucs = auc_matrix(x, L)
            dev = np.abs(aucs - 0.5)
            top = np.argsort(-dev)[:3]
            cors = {k: float(np.corrcoef(x, Zf[:, k])[0, 1]) for k in truth}
            cos = {k: float(x @ Zf[:, k] / (np.linalg.norm(x) * np.linalg.norm(Zf[:, k])
                                              + 1e-12)) for k in truth}
            # co-activation: P(j active | principal active)
            coact = {k: float(((x != 0) & (Zf[:, k] != 0)).sum() / max((Zf[:, k] != 0).sum(), 1))
                     for k in truth}
            # observed vs predicted |t(j|S*)| across all runs of each cell
            for (sw, prm), grp in pd.DataFrame(
                    [dict(sweep=r["sweep"], param=r["param"], n=r["n"], e=r["effect"],
                          sig=r["sigma_S"], t=float(r["t_star"][j]))
                     for r in tr_runs]).groupby(["sweep", "param"]):
                n_, e_ = grp.n.iloc[0], grp.e.iloc[0]
                pred = (e_ * abs(pop["gamma"][j]) * np.sqrt(n_ * P_TREAT * (1 - P_TREAT)
                                                             * pop["vz"][j]) / grp.sig.mean())
                t_track.append(dict(tree=tree, j=int(j), sweep=sw, param=prm, n=n_, e=e_,
                                    mean_abs_t=grp.t.abs().mean(),
                                    mean_signed_t=grp.t.mean(),
                                    frac_rej=(2 * (1 - __import__("scipy").stats.norm.cdf(
                                        grp.t.abs())) <= ALPHA / len(pop["gamma"])).mean(),
                                    pred_t=pred))
            rows.append(dict(tree=tree, j=int(j),
                             n_blocked=int(cnt_b.get(j, 0)),
                             n_rej_S=int(cnt_rej.get(j, 0)),
                             pop_freq=float(pop["freq"][j]),
                             pop_gamma_u=float(pop["gamma"][j]),
                             pop_pr2=float(pop["pr2"][j]),
                             pr2_rank=int(rank[j]),
                             **{f"corr_{k}": v for k, v in cors.items()},
                             **{f"cos_{k}": v for k, v in cos.items()},
                             **{f"coact_{k}": v for k, v in coact.items()},
                             auc_hat=float(aucs[attrs.index("Wearing_Hat")]),
                             auc_glasses=float(aucs[attrs.index("Eyeglasses")]),
                             top_attrs="; ".join(f"{attrs[i]} {aucs[i]:.3f}" for i in top)))
        # calibration: rejections of H0(j|S*) among coordinates that are (near) null in
        # population (predicted noncentrality < 0.5), homoskedastic vs HC1
        from scipy.stats import norm
        mm = len(pop["gamma"])
        tcrit = norm.isf(ALPHA / mm / 2)
        cal = []
        for r in tr_runs:
            pred = (r["effect"] * np.abs(pop["gamma"]) * np.sqrt(
                r["n"] * P_TREAT * (1 - P_TREAT) * pop["vz"]) / r["sigma_S"])
            null = (pred < 0.5) & (pop["vz"] > 1e-12)
            null[truth] = False
            sparse = pop["freq"] * r["n"] / 2 < 20   # < 20 expected nonzeros per arm
            for lab, ts in (("homo", r["t_star"]), ("hc1", r["t_star_hc1"])):
                rej = null & (np.abs(ts) > tcrit)
                nonnull_rej = ~null & (np.abs(ts) > tcrit)
                nonnull_rej[truth] = False
                cal.append(dict(test=lab, sweep=r["sweep"], param=r["param"],
                                n_null=int(null.sum()), null_rej=int(rej.sum()),
                                null_rej_sparse=int((rej & sparse).sum()),
                                null_rej_supported=int((rej & ~sparse).sum()),
                                n_null_supported=int((null & ~sparse).sum()),
                                null_sd_t_supported=float(np.std(ts[null & ~sparse])),
                                nonnull_rej=int(nonnull_rej.sum()),
                                null_sd_t=float(np.std(ts[null])),
                                null_frac_gt3=float((np.abs(ts[null]) > 3).mean())))
        cal = pd.DataFrame(cal)
        cal.to_csv(OUT / f"calibration_{tree.replace('/', '_')}.csv", index=False)
        cg = cal.groupby(["test", "sweep", "param"]).agg(
            runs=("n_null", "size"), n_null=("n_null", "mean"),
            null_rej_per_run=("null_rej", "mean"),
            runs_with_null_rej=("null_rej", lambda s: (s > 0).mean()),
            null_rej_sparse=("null_rej_sparse", "sum"),
            n_null_supported=("n_null_supported", "mean"),
            null_rej_supported=("null_rej_supported", "sum"),
            null_sd_t_supported=("null_sd_t_supported", "mean"),
            nonnull_rej_per_run=("nonnull_rej", "mean"),
            null_sd_t=("null_sd_t", "mean"),
            null_frac_gt3=("null_frac_gt3", "mean")).reset_index()
        lines.append(f"### {tree}: calibration of the test of H0(j|S*)\n\n"
                     f"'null' = coordinates whose predicted noncentrality under the "
                     f"population projection is < 0.5; rejection = |t| > {tcrit:.2f} "
                     f"(two-sided alpha/m).  Under a calibrated test runs_with_null_rej "
                     f"<= ~0.05 and null_sd_t ~ 1 (null_frac_gt3 ~ 0.0027).  "
                     f"null_rej_sparse / _supported = total null rejections on coordinates "
                     f"with < / >= 20 expected nonzeros per arm.  HC1 is shown for "
                     f"reference only: it collapses on columns with a handful of nonzeros "
                     f"(residuals of perfectly fitted leverage points are ~0).  nonnull_rej = rejections among the other (non-null, "
                     f"not S*) coordinates.\n")
        lines.append(cg.to_markdown(index=False, floatfmt=".3g") + "\n")
        # baseline: population partial R^2 distribution and alignment
        lines.append(f"### {tree}: population alignment\n\n"
                     f"S* = {truth}; R^2 of CATE on [1, Z_S*] (CelebA-weighted) = "
                     f"{pop['r2_S']:.4f}; partial R^2 of best other coordinate = "
                     f"{pop['pr2'][order[0]]:.4f} (j={order[0]}), 10th best "
                     f"{pop['pr2'][order[9]]:.4f}, median over all "
                     f"{np.median(pop['pr2']):.2e}.\n")
    C = pd.DataFrame(rows).sort_values(["tree", "n_blocked"], ascending=[True, False])
    C.to_csv(OUT / "coords.csv", index=False)
    T = pd.DataFrame(t_track)
    T.to_csv(OUT / "t_given_Sstar_tracking.csv", index=False)
    lines.append("## Coordinates (blocked FPs and top recurring rejections of H0(j|S*))\n")
    lines.append("n_blocked = times it is a rho-blocked FP; n_rej_S = runs with "
                 "p(j|S*) <= alpha/m; pop_gamma_u = coefficient of Z_j in the "
                 "CelebA-weighted projection of (W1-W2) on [1, Z_S*, Z_j]; pop_pr2 = "
                 "partial R^2 (fraction of CATE residual variance given Z_S* removed by "
                 "Z_j); pr2_rank among all coordinates; coact_k = P(j active | principal "
                 "k active).\n")
    lines.append(C.to_markdown(index=False, floatfmt=".3g") + "\n")
    lines.append("## Observed vs predicted |t(j|S*)| for these coordinates\n")
    lines.append("pred_t = e*|gamma_pop|*sqrt(n p(1-p) Var_perp(Z_j|Z_S*)) / sigma "
                 "(noncentrality under the population projection); mean over all runs "
                 "of the cell, selected or not.\n")
    top = C.groupby("tree").head(6)[["tree", "j"]]
    lines.append(T.merge(top).to_markdown(index=False, floatfmt=".3g") + "\n")

    # ── HC3: calibration on population-null coordinates and HC3 terminal filter ──
    from scipy.stats import norm as _norm
    tc = _norm.isf(ALPHA / 9216 / 2)
    H3 = []
    for tree in sorted(set(r["tree"] for r in runs)):
        features, gt = load_tree(tree)
        truth = sorted(set(gt["truth"]))
        pop = pops[tree]
        for r in [r for r in runs if r["tree"] == tree]:
            d = generate_celeba_rct(n=r["n"], features=features, labels_df=labels_df,
                                    buckets=buckets, effect_scale=r["effect"],
                                    seed=r["seed"], **SCM)
            pred = (r["effect"] * np.abs(pop["gamma"]) * np.sqrt(
                r["n"] * P_TREAT * (1 - P_TREAT) * pop["vz"]) / r["sigma_S"])
            null = (pred < 0.5) & (pop["vz"] > 1e-12)
            null[truth] = False
            sparse = pop["freq"] * r["n"] / 2 < 20
            idx = np.where(null)[0]
            th = hc3_vec(d.Y, d.T, d.Z, truth, idx)
            rej = np.abs(th) > tc
            k0 = hc3_terminal_filter(d.Y, d.T, d.Z, r["r0"]["St"], tc)
            k5 = hc3_terminal_filter(d.Y, d.T, d.Z, r["r05"]["St"], tc)
            mnz = np.minimum((d.Z[d.T == 1] != 0).sum(0), (d.Z[d.T == 0] != 0).sum(0))
            k0s = [j for j in k0 if mnz[j] >= 20]
            H3.append(dict(tree=tree, sweep=r["sweep"], param=r["param"], seed=r["seed"],
                           null_rej=int(rej.sum()),
                           null_rej_sparse=int((rej & sparse[idx]).sum()),
                           null_rej_supported=int((rej & ~sparse[idx]).sum()),
                           null_sd_t_supported=float(np.std(th[~sparse[idx]])),
                           tp_rho0_hc3=len(set(k0) & set(truth)),
                           fp_rho0_hc3=len(set(k0) - set(truth)),
                           tp_rho0_hc3_supp=len(set(k0s) & set(truth)),
                           fp_rho0_hc3_supp=len(set(k0s) - set(truth)),
                           tp_rho05_hc3=len(set(k5) & set(truth)),
                           fp_rho05_hc3=len(set(k5) - set(truth)),
                           tp_rho0=len(set(r["r0"]["S"]) & set(truth)),
                           fp_rho0=len(set(r["r0"]["S"]) - set(truth)),
                           tp_rho05=len(set(r["r05"]["S"]) & set(truth)),
                           fp_rho05=len(set(r["r05"]["S"]) - set(truth))))
        del features
    H3 = pd.DataFrame(H3)
    H3.to_csv(OUT / "hc3_filter.csv", index=False)
    lines.append("## HC3 (leverage-robust) test\n")
    lines.append("Calibration on population-null coordinates (same definition as above) "
                 "with HC3 standard errors, and the subset-robust terminal filter "
                 "re-run with the HC3 test on the same S~ (rho0 and rho05 paths). "
                 "tp/fp = totals over the runs of the cell (2 principals per run).  "
                 "Caveat: HC3 is degenerate on columns with ~1 nonzero per arm (the "
                 "column fits those points exactly, residual ~0, |t| explodes), which "
                 "is what null_rej_sparse counts; *_hc3_supp adds a floor of >= 20 "
                 "nonzeros per arm to the HC3 filter.\n")
    h3g = H3.groupby(["tree", "sweep", "param"]).agg(
        runs=("seed", "size"),
        runs_with_null_rej=("null_rej", lambda s: (s > 0).mean()),
        null_rej_sparse=("null_rej_sparse", "sum"),
        null_rej_supported=("null_rej_supported", "sum"),
        null_sd_t_supported=("null_sd_t_supported", "mean"),
        tp_rho05=("tp_rho05", "sum"), fp_rho05=("fp_rho05", "sum"),
        tp_rho0=("tp_rho0", "sum"), fp_rho0=("fp_rho0", "sum"),
        tp_rho0_hc3=("tp_rho0_hc3", "sum"), fp_rho0_hc3=("fp_rho0_hc3", "sum"),
        tp_rho05_hc3=("tp_rho05_hc3", "sum"), fp_rho05_hc3=("fp_rho05_hc3", "sum"),
        tp_rho0_hc3_supp=("tp_rho0_hc3_supp", "sum"),
        fp_rho0_hc3_supp=("fp_rho0_hc3_supp", "sum"),
    ).reset_index()
    lines.append(h3g.to_markdown(index=False, floatfmt=".3g") + "\n")

    # ── rho05 hurts ──
    H = []
    for r in runs:
        lost = (set(r["truth"]) & set(r["r0"]["S"])) - set(r["r05"]["S"])
        for k in lost:
            H.append(dict(tree=r["tree"], sweep=r["sweep"], param=r["param"],
                          seed=r["seed"], k=k, S05=r["r05"]["S"], St05=r["r05"]["St"],
                          stop=r["r05"]["stop"]))
    lines.append(f"## Runs where rho05_filter misses a principal rho0_filter finds: "
                 f"{len(H)}\n")
    for h in H:
        lines.append(f"- {h}\n")

    # ── replacement rules on the rho0_filter output ──
    E = []
    for r in runs:
        el = pd.DataFrame(r["elem"])
        if el.empty:
            continue
        el["std_eff"] = (el.gamma * el.sd).abs()
        el["act_eff"] = (el.gamma * el.mean_active).abs()
        el["min_nnz"] = np.minimum(el.nnz1, el.nnz0)
        el["rel_std"] = el.std_eff / el.std_eff.max()
        el["rel_act"] = el.act_eff / el.act_eff.max()
        el["is_true"] = el.j.isin(r["truth"])
        el["tree"], el["seed"], el["param"] = r["tree"], r["seed"], r["param"]
        el["in05"] = el.j.isin(r["r05"]["S"])
        E.append(el)
    E = pd.concat(E)
    E.to_csv(OUT / "rho0_filter_elements.csv", index=False)

    def score(keep_mask):
        tp = (E.is_true & keep_mask).sum()
        fp = (~E.is_true & keep_mask).sum()
        fn_ = (E.is_true & ~keep_mask).sum()
        return tp, fp, fn_

    lines.append("## Candidate replacement rules applied to the rho0_filter output\n")
    lines.append("Counts over all runs; baseline rho05_filter shown for reference "
                 "(its own output, not a filter of rho0_filter).\n")
    rr = []
    tp5 = sum(len(set(r["truth"]) & set(r["r05"]["S"])) for r in runs)
    fp5 = sum(len(set(r["r05"]["S"]) - set(r["truth"])) for r in runs)
    rr.append(dict(rule="rho05_filter (reference)", tp=tp5, fp=fp5))
    rr.append(dict(rule="rho0_filter (no rule)", tp=int(E.is_true.sum()),
                   fp=int((~E.is_true).sum())))
    for c in (20, 50, 100):
        tp, fp, _ = score(E.min_nnz >= c)
        rr.append(dict(rule=f"min per-arm nonzeros >= {c}", tp=int(tp), fp=int(fp)))
    for c in (0.1, 0.25, 0.5):
        tp, fp, _ = score(E.rel_std >= c)
        rr.append(dict(rule=f"|gamma*sd| >= {c} x max in S", tp=int(tp), fp=int(fp)))
        tp, fp, _ = score(E.rel_act >= c)
        rr.append(dict(rule=f"|gamma*mean_active| >= {c} x max in S", tp=int(tp),
                       fp=int(fp)))
    lines.append(pd.DataFrame(rr).to_markdown(index=False) + "\n")
    lines.append("\nFP element distribution (rho0_filter output, j not in S*): "
                 + E[~E.is_true][["rel_std", "rel_act", "min_nnz"]].describe()
                 .round(3).to_markdown() + "\n\nTrue principals: "
                 + E[E.is_true][["rel_std", "rel_act", "min_nnz"]].describe()
                 .round(3).to_markdown() + "\n")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "analyze"])
    ap.add_argument("tree", nargs="?")
    ap.add_argument("sweep", nargs="?", choices=["effect", "n"])
    ap.add_argument("fixed", nargs="?", type=float)
    ap.add_argument("param", nargs="?", type=float)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.phase == "run":
        run_cell(a.tree, a.sweep, a.fixed, a.param, a.seeds, a.overwrite)
    else:
        analyze()
