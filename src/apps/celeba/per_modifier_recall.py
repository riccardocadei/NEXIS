#!/usr/bin/env python3
"""
Per-modifier recall of NEXIS-v2 on the CelebA sweeps.

The sweep parquets store tp and recall per run, not which coordinates were selected,
so they cannot say which modifier a partially successful run missed.  This script
replays NEXIS-v2 (nexis(rho=0.5, backward=False, terminal_filter=True), alpha 0.05,
10 steps) on exactly the draws of those sweeps (same sampler call, same seeds, so the
same units, T, images and Y) and records, for each run, the selected set and whether
each modifier's ground-truth coordinate is in it.

Grid: the four rows of dgp.pdf (n in 50..10,000 at eta = 5 and 2; eta in 1..10 at
n = 2000 and 500), 50 seeds.  Writes one parquet with one row per (row, x, seed).
With --check, compares the replayed recall to the stored sweep and reports mismatches.

    python src/apps/celeba/per_modifier_recall.py --gammas 1 -1 \
        --out results/celeba/per_modifier/r2.parquet --check results/celeba/experiment_v2/k20/sae
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from method.nexis import nexis
from apps.celeba.scm import build_buckets, generate_celeba_rct
from apps.celeba.experiment import NEXIS_V2_DEFAULT
from apps.celeba.run_experiment_dgp import EFFECT_GRID, N_GRID, neurons_by_attr

ROWS = [("n", 5.0), ("n", 2.0), ("effect_scale", 2000), ("effect_scale", 500)]


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--w-attrs", nargs="+", default=["Wearing_Hat", "Eyeglasses"])
    p.add_argument("--gammas", type=float, nargs="+", required=True)
    p.add_argument("--gt-json", type=Path,
                   default="results/celeba/experiment/k20/sae/ground_truth.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check", type=Path, default=None,
                   help="sweep directory holding n_sweep/effect_sweep.parquet to compare with")
    p.add_argument("--n-seeds", type=int, default=50)
    a = p.parse_args()

    out = _abs(a.out)
    if out.exists():
        print(f"exists, skipping: {out}")
        return
    betas = [(0.3 if k % 2 == 0 else -0.2) for k in range(len(a.w_attrs))]
    by_attr = neurons_by_attr(json.loads(_abs(a.gt_json).read_text()))
    mods = [(w, by_attr[w][0]) for w, g in zip(a.w_attrs, a.gammas) if g != 0]
    truth = {j for _, j in mods}
    print(f"attrs {a.w_attrs} gammas {a.gammas}; modifiers {mods}")

    features = np.load(ROOT / "data/celeba/embeddings/sae_k20.npy")
    labels = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    buckets = build_buckets(labels, w_attrs=a.w_attrs)

    def one(row, x, seed):
        sweep, fixed = ROWS[row]
        n, eta = (int(x), fixed) if sweep == "n" else (int(fixed), x)
        d = generate_celeba_rct(n=n, features=features, labels_df=labels, buckets=buckets,
                                w_attrs=a.w_attrs, betas=betas, gammas=a.gammas,
                                tau_0=0.5, noise_sd=1.0, effect_scale=eta, seed=seed)
        sel = set(int(j) for j in nexis(y=d.Y, t=d.T, z=d.Z, alpha=0.05, max_rounds=10,
                                        **NEXIS_V2_DEFAULT).selected)
        rec = {"row": row, "sweep": sweep, "fixed": fixed, "x": x, "seed": seed,
               "selected": ",".join(map(str, sorted(sel))), "n_selected": len(sel),
               "tp": len(sel & truth), "recall": len(sel & truth) / len(truth)}
        rec.update({f"hit_{w}": int(j in sel) for w, j in mods})
        return rec

    tasks = [(r, x, s) for r, (sw, _) in enumerate(ROWS)
             for x in (N_GRID if sw == "n" else EFFECT_GRID) for s in range(a.n_seeds)]
    print(f"{len(tasks)} runs", flush=True)
    df = pd.DataFrame(Parallel(n_jobs=-1, prefer="threads")(
        delayed(one)(*t) for t in tasks))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows -> {out}")

    if a.check is not None:
        c = _abs(a.check)
        ref = []
        for sweep, fname, fcol in (("n", "n_sweep", "fixed_effect"),
                                   ("effect_scale", "effect_sweep", "fixed_n")):
            s = pd.read_parquet(c / f"{fname}.parquet")
            s = s[s.method == "NEXIS-v2"].rename(columns={sweep: "x", fcol: "fixed"})
            s["sweep"] = sweep
            ref.append(s[["sweep", "fixed", "x", "seed", "recall", "n_selected"]])
        m = df.merge(pd.concat(ref), on=["sweep", "fixed", "x", "seed"],
                     suffixes=("", "_ref"))
        bad = ((m.recall - m.recall_ref).abs() > 1e-9) | (m.n_selected != m.n_selected_ref)
        print(f"check vs {c}: {len(m)} matched runs, {int(bad.sum())} mismatches")

    hits = [c for c in df.columns if c.startswith("hit_")]
    print(df.groupby(["row", "x"])[hits + ["recall"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
