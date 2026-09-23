#!/usr/bin/env python3
"""Generalised certified run: any feature tree x any sweep from the paper.

Usage: run_certified_all.py <tree> <sweep> <fixed>
  tree  : k20/sae | k20/sae_precode | k5/sae | k5/sae_precode
  sweep : effect | n
  fixed : n value (effect sweep) or effect_scale (n sweep)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/nfs/scistore19/locatgrp/rcadei/NEXIS")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from apps.celeba.scm import build_buckets, generate_celeba_rct
from method.nexis import nexis, iou_score
from certify import certify_global, certify_subset_robust, log10_N_K

TREE, SWEEP, FIXED = sys.argv[1], sys.argv[2], float(sys.argv[3])
ALPHA, MAX_STEPS, N_SEEDS = 0.05, 10, 50
K_GRID = [3, 5, 10]
EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]

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
print(f"{TREE} {fname}  m={M}  truth={TRUTH}  sweep={SWEEP} fixed={FIXED}", flush=True)


def metrics(sel):
    ss, ts = set(int(x) for x in sel), set(TRUTH)
    tp, kk = len(ss & ts), len(ss)
    return {"n_selected": kk, "tp": tp, "fp": len(ss - ts),
            "recall": tp / len(ts), "precision": (tp / kk) if kk else 1.0,
            "iou": iou_score(ss, ts)}


def one(param, seed):
    n = int(param) if SWEEP == "n" else int(FIXED)
    effect = float(param) if SWEEP == "effect" else FIXED
    try:
        d = generate_celeba_rct(n=n, features=features, labels_df=labels_df,
                                buckets=buckets, effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return []          # bucket exhausted, same skip rule as the paper runner
    y, t, Z = d.Y, d.T, d.Z
    S = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    tag = {"tree": TREE, "sweep": SWEEP, "fixed": FIXED, "param": param,
           "seed": seed, "n": n, "effect_scale": effect, "n_cand": len(S)}
    rows = [{**tag, "variant": "NEXIS", **metrics(S), "cert_status": "ok"}]
    for K in K_GRID:
        c = certify_global(y, t, Z, S, ALPHA, K)
        rows.append({**tag, "variant": f"NEXIS + Opt1 (K={K})",
                     **metrics(c["certified"]), "cert_status": c["status"]})
    c2 = certify_subset_robust(y, t, Z, S, ALPHA)
    rows.append({**tag, "variant": "NEXIS + Opt2", **metrics(c2["certified"]),
                 "cert_status": "ok"})
    return rows


grid = EFFECT_GRID if SWEEP == "effect" else N_GRID
tasks = [(p, s) for p in grid for s in range(N_SEEDS)]
print(f"{len(tasks)} runs …", flush=True)
out = Parallel(n_jobs=int(__import__("os").environ.get("NJOBS","32")), prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for rows in out for r in rows])
tag = f"{TREE.replace('/', '_')}_{SWEEP}_{FIXED:g}"
df.to_csv(HERE / f"cert_{tag}.csv", index=False)
print(f"wrote {len(df)} rows -> cert_{tag}.csv")
