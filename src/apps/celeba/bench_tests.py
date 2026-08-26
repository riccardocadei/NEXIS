#!/usr/bin/env python3
"""
Serial wall-clock cost of one full NEXIS run, per CATE-equivalence test.

Do NOT read cost off the sweep parquets: those runs are executed 40-way in parallel
via joblib threads, and the contention is not uniform across tests (the LightGBM
projection, which issues many small fits, is penalised ~10x harder than the vectorised
paths).  The sweep also confounds test cost with selection depth, since a test with
less power terminates after fewer rounds.  This script fixes both: same n, same eta,
same number of rounds and selections for every test, run one at a time.

Writes results/celeba/appendix/test_cost.json, consumed by table_test_ablation.py.

Usage
-----
    OMP_NUM_THREADS=1 python src/apps/celeba/bench_tests.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.scm import build_buckets, generate_celeba_rct
from method.nexis import nexis

TESTS = ["linear", "GCM: quadratic", "GCM: lgbm", "PCM: quadratic", "PCM: lgbm"]
N, ETA, SEEDS = 2000, 5.0, 5


def main() -> None:
    labels = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    feat = np.asarray(np.load(ROOT / "data/celeba/embeddings/sae_k20.npy",
                              mmap_mode="r"))
    buckets = build_buckets(labels, "Wearing_Hat", "Eyeglasses")

    acc: dict[str, list[float]] = {m: [] for m in TESTS}
    depth: dict[str, list[int]] = {m: [] for m in TESTS}
    for seed in range(SEEDS):
        d = generate_celeba_rct(n=N, features=feat, labels_df=labels, buckets=buckets,
                                w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
                                effect_scale=ETA, seed=seed)
        for m in TESTS:
            t0 = time.perf_counter()
            r = nexis(y=d.Y, t=d.T, z=d.Z, alpha=0.05, max_rounds=10,
                      test=m, n_splits=3)
            acc[m].append(time.perf_counter() - t0)
            depth[m].append(len(r.selected))

    base = float(np.median(acc["linear"]))
    out = {m: {"seconds": round(float(np.median(acc[m])), 2),
               "rel_linear": round(float(np.median(acc[m])) / base, 1),
               "n_selected": sorted(set(depth[m]))}
           for m in TESTS}
    meta = {"n": N, "eta": ETA, "seeds": SEEDS, "serial": True, "tests": out}

    dest = ROOT / "results/celeba/appendix/test_cost.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"serial, n={N}, eta={ETA:g}, {SEEDS} seeds, one full NEXIS run:")
    for m in TESTS:
        o = out[m]
        print(f"  {m:16s} {o['seconds']:6.1f}s  ({o['rel_linear']}x linear)  "
              f"selected={o['n_selected']}")
    print(f"\nSaved -> {dest}")


if __name__ == "__main__":
    main()
