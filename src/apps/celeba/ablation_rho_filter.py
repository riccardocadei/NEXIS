#!/usr/bin/env python3
"""Ablation: can the spectral-gap gate rho be dropped once NEXIS has the terminal filter?

Re-runs the CelebA semi-synthetic grid of the NeurIPS rebuttal experiment
(~/.cache/nexis_cert/run_all6.py): same SCM, grid, seeds, ground truth, support screen
and scoring.  For every (tree, sweep, param, seed) it computes on the same simulated data

  rho05            nexis(rho=0.5)                         == all6 "base"  (sanity check)
  rho05_filter     rho05 + terminal filter (alpha/m)       == all6 "opt2"  (sanity check)
  rho0             nexis(rho=0)
  rho0_filter      rho0 + terminal filter
  fwd_rho0         nexis(rho=0, backward=False)
  fwd_rho0_filter  fwd_rho0 + terminal filter
  scr_rho05[_filter], scr_rho0[_filter]
                   same on the support-screened dictionary (filter at alpha/m_scr);
                   scr_rho05_filter == all6 "scr_opt2"

Terminal filter (subset-robust certification), as built into nexis(terminal_filter=True):
keep j in S~ iff max_{A subset of S~ minus j, incl. empty} p_j(A) <= alpha/m, removals
simultaneous; subsets are tried largest first and j is dropped at its first failing
subset, so filter_tests is <= |S~| 2^(|S~|-1).  For |S~| > MAX_CERT (terminal_max_size)
the row is flagged "uncertifiable" and scored on the unfiltered S~.  S~ is read from
metadata["terminal_candidates"], so each rho/backward/dictionary setting is one nexis call.

Usage: ablation_rho_filter.py <tree> <sweep> <fixed> [--seeds N] [--params p1,p2] [--out F]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.scm import build_buckets, generate_celeba_rct  # noqa: E402
from method.nexis import nexis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("sweep", choices=["effect", "n"])
ap.add_argument("fixed", type=float)
ap.add_argument("--seeds", type=int, default=50)
ap.add_argument("--params", default=None, help="comma-separated subset of the grid")
ap.add_argument("--out", default=None)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

TREE, SWEEP, FIXED = args.tree, args.sweep, args.fixed
ALPHA, MAX_STEPS, N_SEEDS, MIN_ARM, MAX_CERT = 0.05, 10, args.seeds, 3, 14
EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]

OUT_DIR = ROOT / "results/celeba/ablation_rho_filter"
tag = f"{TREE.replace('/', '_')}_{SWEEP}_{FIXED:g}"
out_path = Path(args.out) if args.out else OUT_DIR / f"rf_{tag}.csv"
if out_path.exists() and not args.overwrite:
    print(f"exists, skipping: {out_path}")
    sys.exit(0)

kdir, ftype = TREE.split("/")
k = int(kdir[1:])
fname = f"sae_k{k}.npy" if ftype == "sae" else f"sae_precode_k{k}.npy"
features = np.load(ROOT / "data/celeba/embeddings" / fname)
labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
gt = json.load(open(ROOT / f"results/celeba/experiment/{TREE}/ground_truth.json"))
TRUTH = sorted(set(gt["truth"]))
buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
scm = dict(w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
           tau_0=0.5, gamma_w1=1.0, gamma_w2=-1.0, noise_sd=1.0)
M = features.shape[1]
print(f"{TREE} {fname} m={M} truth={TRUTH} sweep={SWEEP} fixed={FIXED:g}", flush=True)


def eff_support(Zarm):
    """Kish effective support per column: (sum|z|)^2 / sum z^2."""
    s1 = np.abs(Zarm).sum(axis=0)
    s2 = (Zarm ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(s2 > 0, s1 ** 2 / s2, 0.0)
    return ess


def score(sel):
    ss, ts = set(int(x) for x in sel), set(TRUTH)
    tp, kk = len(ss & ts), len(ss)
    den = kk + len(ts) - tp
    return {"n_selected": kk, "tp": tp, "fp": len(ss - ts),
            "recall": tp / len(ts), "precision": (tp / kk) if kk else 0.0,
            "iou": (tp / den) if den else 0.0}


def one(param, seed):
    n = int(param) if SWEEP == "n" else int(FIXED)
    effect = float(param) if SWEEP == "effect" else FIXED
    try:
        d = generate_celeba_rct(n=n, features=features, labels_df=labels_df,
                                buckets=buckets, effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return []
    y, t, Z = d.Y, d.T, d.Z

    cols = np.where((eff_support(Z[t == 1]) >= MIN_ARM)
                    & (eff_support(Z[t == 0]) >= MIN_ARM))[0]
    Zs = Z[:, cols] if len(cols) else None

    def run(Zm, rho, backward):
        """One NEXIS call with the built-in terminal filter (nexis.py, alpha/m where
        m = columns entering nexis).  Returns (S~, S_hat, n_tests, sec, uncertifiable)."""
        t0 = time.perf_counter()
        try:
            r = nexis(y=y, t=t, z=Zm, alpha=ALPHA, max_rounds=MAX_STEPS, rho=rho,
                      backward=backward, terminal_filter=True,
                      terminal_max_size=MAX_CERT)
        except ValueError as e:
            if "terminal_max_size" not in str(e):
                raise
            S = list(nexis(y=y, t=t, z=Zm, alpha=ALPHA, max_rounds=MAX_STEPS,
                           rho=rho, backward=backward).selected)
            return S, S, 0, time.perf_counter() - t0, True
        md = r.metadata
        return (list(md["terminal_candidates"]), list(r.selected),
                int(md["terminal_tests"]), time.perf_counter() - t0, False)

    specs = [("rho05", False, 0.5, True), ("rho0", False, 0.0, True),
             ("fwd_rho0", False, 0.0, False),
             ("scr_rho05", True, 0.5, True), ("scr_rho0", True, 0.0, True)]
    tagd = {"tree": TREE, "sweep": SWEEP, "fixed": FIXED, "param": param, "seed": seed,
            "n": n, "effect_scale": effect, "m": M, "m_scr": len(cols),
            "truth_kept": len(set(TRUTH) & set(cols.tolist()))}
    rows = []
    for name, scr, rho, bwd in specs:
        if scr and Zs is None:
            S, kept, n_t, dt, unc = [], [], 0, 0.0, False
            back = list
        else:
            S, kept, n_t, dt, unc = run(Zs if scr else Z, rho, bwd)
            back = ((lambda L: [int(cols[j]) for j in L]) if scr
                    else (lambda L: [int(j) for j in L]))
        S_o, kept_o = back(S), back(kept)
        # sec = wall time of the whole nexis call (forward/backward + terminal filter)
        common = {"size_pre": len(S_o), "nexis_filter_sec": dt}
        rows.append({**tagd, "variant": name, **common, "size_post": len(S_o),
                     "filter_tests": 0, "uncertifiable": False,
                     "selected": " ".join(map(str, S_o)), **score(S_o)})
        rows.append({**tagd, "variant": name + "_filter", **common,
                     "size_post": len(kept_o), "filter_tests": n_t,
                     "uncertifiable": unc,
                     "selected": " ".join(map(str, kept_o)), **score(kept_o)})
    return rows


grid = EFFECT_GRID if SWEEP == "effect" else N_GRID
if args.params:
    grid = [type(grid[0])(p) for p in args.params.split(",")]
tasks = [(p, s) for p in grid for s in range(N_SEEDS)]
print(f"{len(tasks)} runs ...", flush=True)
nj = int(os.environ.get("NJOBS", "12"))
out = Parallel(n_jobs=nj, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for rows in out for r in rows])
out_path.parent.mkdir(parents=True, exist_ok=True)
tmp = out_path.with_suffix(".csv.tmp")
df.to_csv(tmp, index=False)
tmp.replace(out_path)
print(f"wrote {len(df)} rows -> {out_path}; uncertifiable={int(df['uncertifiable'].sum())}")
