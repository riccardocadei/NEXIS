#!/usr/bin/env python3
"""
Merge the sharded resampled-SAE experiment rerun into the canonical results layout.

Shards are written by scripts/celeba/run_resample_shard.sh as

    results/celeba/experiment_resample_<TAG>/shards/<shard>/k<K>/<feature>/{effect,n}_sweep.parquet

each holding one (sweep × fixed value × method group × seed block) slice.  This script
concatenates them into

    results/celeba/experiment_resample_<TAG>/k<K>/<feature>/{effect,n}_sweep.parquet
    results/celeba/experiment_resample_<TAG>/k<K>/<feature>/ground_truth.json

and checks the merge is complete and non-overlapping: every (method, grid value, seed,
fixed value) cell must appear exactly once, and all shards of one config must agree on
the ground-truth neurons.

Usage
-----
    python src/apps/celeba/merge_shards.py --tag b1
    python src/apps/celeba/merge_shards.py --tag b1 --expect-seeds 50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# The design grid of the paper's CelebA experiments.
EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID      = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
FIXED_N     = [500, 2000]
FIXED_EFF   = [2.0, 5.0]
N_METHODS   = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag",          default="b1", help="Resample tag (default: b1)")
    p.add_argument("--results-root", type=Path, default="results/celeba")
    p.add_argument("--expect-seeds", type=int, default=50)
    p.add_argument("--expect-methods", type=int, default=N_METHODS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = (ROOT / args.results_root if not args.results_root.is_absolute()
            else args.results_root)
    base = root / f"experiment_resample_{args.tag}"
    shard_root = base / "shards"
    if not shard_root.is_dir():
        print(f"No shards found at {shard_root}", file=sys.stderr)
        return 1

    # (k, feature, sweep_file) → [shard parquet paths]
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    gts: dict[tuple[str, str], list[tuple[Path, dict]]] = defaultdict(list)

    for shard in sorted(shard_root.iterdir()):
        if not shard.is_dir():
            continue
        for parquet in shard.glob("k*/*/[en]*_sweep.parquet"):
            feature = parquet.parent.name          # sae | sae_precode
            kdir    = parquet.parent.parent.name   # k5 | k20
            groups[(kdir, feature, parquet.name)].append(parquet)
        for gt in shard.glob("k*/*/ground_truth.json"):
            with open(gt) as f:
                gts[(gt.parent.parent.name, gt.parent.name)].append((gt, json.load(f)))

    if not groups:
        print(f"No shard parquets under {shard_root}", file=sys.stderr)
        return 1

    ok = True

    # ── ground truth: every shard of a config must agree ─────────────────────
    for (kdir, feature), entries in sorted(gts.items()):
        truths = {tuple(d["truth"]) for _, d in entries}
        if len(truths) > 1:
            print(f"[FAIL] {kdir}/{feature}: shards disagree on truth set: {truths}")
            ok = False
            continue
        out_dir = base / kdir / feature
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "ground_truth.json", "w") as f:
            json.dump(entries[0][1], f, indent=2)
        print(f"{kdir}/{feature}: truth={list(truths)[0]}  "
              f"({len(entries)} shards agree)")

    # ── sweeps ───────────────────────────────────────────────────────────────
    for (kdir, feature, fname), paths in sorted(groups.items()):
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        sweep_col = "effect_scale" if "effect_scale" in df.columns else "n"
        fixed_col = "fixed_n" if sweep_col == "effect_scale" else "fixed_effect"

        expected_grid  = EFFECT_GRID if sweep_col == "effect_scale" else N_GRID
        expected_fixed = FIXED_N     if sweep_col == "effect_scale" else FIXED_EFF

        # run_sweep tags every row with the fixed value of the other design axis, so a
        # complete merge has exactly one row per (grid, fixed, seed, method) cell.
        dups    = int(df.duplicated(subset=[sweep_col, fixed_col, "seed", "method"]).sum())
        n_cells = (len(expected_grid) * len(expected_fixed)
                   * args.expect_seeds * args.expect_methods)
        got     = len(df)

        status = "OK"
        if fixed_col not in df.columns:
            status = f"FAIL missing {fixed_col} column"
            ok = False
        elif sorted(df[fixed_col].unique()) != sorted(expected_fixed):
            status = f"FAIL {fixed_col}={sorted(df[fixed_col].unique())}"
            ok = False
        if df["method"].nunique() != args.expect_methods:
            status = f"FAIL methods={df['method'].nunique()}"
            ok = False
        if df["seed"].nunique() != args.expect_seeds:
            status = f"FAIL seeds={df['seed'].nunique()}"
            ok = False
        if sorted(df[sweep_col].unique()) != sorted(expected_grid):
            status = f"FAIL grid={sorted(df[sweep_col].unique())}"
            ok = False
        if got != n_cells:
            status = f"FAIL rows={got} expected={n_cells}"
            ok = False
        if dups:
            status = f"FAIL {dups} duplicated (grid, fixed, seed, method) cells"
            ok = False

        out_dir = base / kdir / feature
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        df.to_parquet(out_path, index=False)
        print(f"{kdir}/{feature}/{fname}: {got} rows from {len(paths)} shards  "
              f"[{status}]  →  {out_path.relative_to(root)}")

    print("\nMerge complete." if ok else "\nMerge finished WITH PROBLEMS (see FAIL above).")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
