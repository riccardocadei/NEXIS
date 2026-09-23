#!/usr/bin/env python3
"""Summarise interleaved-backward pilots / sweeps: one row per (tag, sweep, fixed, param).

Columns: activation rate (share of runs in which the interleaved step removed >= 1
coordinate), mean removals, share of runs where the two arms' final selections differ,
mean |S~| before the terminal step in each arm, mean terminal tests in each arm, and
precision / recall / IoU of f, fb, fi, fib.

Usage: interleaved_backward_summary.py [tag ...]  (default: every tag directory)
       --by-tag   one row per tag (pooled over its grid) instead of per param
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/celeba/interleaved_backward"

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
by_tag = "--by-tag" in sys.argv
tags = argv or sorted(p.name for p in OUT.iterdir()
                      if p.is_dir() and p.name not in ("align", "smoke"))
dfs = [pd.read_csv(f) for t in tags for f in sorted((OUT / t).glob("*.csv"))]
df = pd.concat(dfs, ignore_index=True)
df["act"] = (df["fib_int_removed"] > 0).astype(int)
df["fb_Spre"] = df["f_n_sel"]
df["fib_Spre"] = df["fi_n_sel"]
keys = ["tag"] if by_tag else ["tag", "sweep", "fixed", "param"]
agg = df.groupby(keys).agg(
    runs=("seed", "size"), act=("act", "mean"), removed=("fib_int_removed", "mean"),
    rm_truth=("fib_removed_truth", "mean"), differ=("differ", "mean"),
    Spre_fb=("fb_Spre", "mean"), Spre_fib=("fib_Spre", "mean"),
    tests_fb=("fb_term_tests", "mean"), tests_fib=("fib_term_tests", "mean"),
    mix_in_Sf=("f_has_mix", "mean"), mix_in_fb=("fb_has_mix", "mean"),
    P_fb=("fb_precision", "mean"), P_fib=("fib_precision", "mean"),
    R_f=("f_recall", "mean"), R_fb=("fb_recall", "mean"), R_fib=("fib_recall", "mean"),
    IoU_f=("f_iou", "mean"), IoU_fb=("fb_iou", "mean"), IoU_fi=("fi_iou", "mean"),
    IoU_fib=("fib_iou", "mean"),
).reset_index()
pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 500)
print(agg.round(3).to_string(index=False))
