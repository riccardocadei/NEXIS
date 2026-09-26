#!/usr/bin/env python3
"""Size of the forward selection |S~| and cost of the terminal backward step (paper item).

Produces the numbers of Appendix B ("Backward steps", paper/iclr27/appendix.tex):
  "over the 6,300 CelebA runs of the three dictionaries at rho = 0.5, |S~| has median 2
   and maximum 7, and the forward step alone gives the same median and maximum on the
   main setting. [...] the same runs needed a median of 4 and a maximum of 192 tests,
   against the bound 7 * 2^6 = 448."

Definitions:
  |S~|   size of the forward selection before the terminal backward step (size_pre).
  tests  number of p-values computed by the terminal backward step of
         nexis(rho=0.5, backward=False, terminal_filter=True) (the default "NEXIS-v2"):
         subsets tried largest first, a coordinate dropped at its first failing subset
         (filter_tests).

Inputs (untracked outputs):
  results/celeba/ablation_rho_filter/rf_*.csv  variant "rho05_filter" on the 3
      dictionaries (k20/sae, k20/sae_precode, k5/sae) x 2,100 runs (n sweep at eta in
      {2, 5}, effect sweep at n in {500, 2000}, 50 seeds). Produced by
      src/apps/celeba/ablation_rho_filter.py (scripts/celeba/submit_rho_filter.sh).
      rho05_filter = nexis(rho=0.5, backward=True, terminal_filter=True): the default
      NEXIS-v2 (backward=False) plus the interleaved backward step. Its selections equal
      experiment_v2's NEXIS-v2 on 6,299 of 6,300 runs; |S~| and test counts of the pure
      default are stored only for the 2,100 main-setting runs (cross-check below).
  results/celeba/iclr_local_runs/results/runs/main/*/NEXIS.parquet (optional
      cross-check: the ICLR re-implementation's main-setting runs, n_forward and
      terminal_tests; these agree run by run with the k20/sae rows).

Command (repo root, CPU, seconds):
  /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 src/apps/celeba/run_statistics.py

Output: results/celeba/paper_numbers/run_statistics.{md,json}
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RF = ROOT / "results/celeba/ablation_rho_filter"
ICLR = ROOT / "results/celeba/iclr_local_runs/results/runs/main"
OUT = ROOT / "results/celeba/paper_numbers"
KEY = ["sweep", "fixed", "param", "seed"]


def stats(d: pd.DataFrame) -> dict:
    return {"runs": int(len(d)),
            "S_tilde_median": float(d.size_pre.median()), "S_tilde_max": int(d.size_pre.max()),
            "S_hat_median": float(d.size_post.median()), "S_hat_max": int(d.size_post.max()),
            "tests_median": float(d.filter_tests.median()), "tests_max": int(d.filter_tests.max())}


def main() -> None:
    df = pd.concat([pd.read_csv(f) for f in sorted(RF.glob("rf_*.csv"))], ignore_index=True)
    nexis = df[df.variant == "rho05_filter"]
    res = {"all_three_dictionaries": stats(nexis),
           "per_dictionary": {t: stats(g) for t, g in nexis.groupby("tree")}}
    # "forward step alone": default NEXIS without the terminal step (variant rho05)
    fwd = df[(df.variant == "rho05") & (df.tree == "k20/sae")]
    res["forward_only_main_setting"] = {"S_hat_median": float(fwd.size_post.median()),
                                        "S_hat_max": int(fwd.size_post.max())}
    s = res["all_three_dictionaries"]
    res["bound"] = int(s["S_tilde_max"] * 2 ** (s["S_tilde_max"] - 1))
    top = nexis.sort_values("filter_tests").iloc[-1]
    res["max_tests_run"] = {k: (top[k].item() if hasattr(top[k], "item") else top[k])
                            for k in ["tree", "sweep", "fixed", "param", "seed",
                                      "size_pre", "filter_tests"]}

    # Cross-check against the ICLR re-implementation (main setting only).
    if (ICLR / "n_sweep/NEXIS.parquet").exists():
        a = pd.concat([
            pd.read_parquet(ICLR / "n_sweep/NEXIS.parquet").assign(
                sweep="n", fixed=lambda x: x.fixed_effect, param=lambda x: x.n),
            pd.read_parquet(ICLR / "effect_sweep/NEXIS.parquet").assign(
                sweep="effect", fixed=lambda x: x.fixed_n, param=lambda x: x.effect_scale)])
        m = a.merge(nexis[nexis.tree == "k20/sae"], on=KEY)
        res["iclr_main_crosscheck"] = {
            "runs_matched": int(len(m)),
            "n_forward_equal": int((m.n_forward == m.size_pre).sum()),
            "terminal_tests_equal": int((m.terminal_tests == m.filter_tests).sum()),
            "n_selected_equal": int((m.n_selected_x == m.size_post).sum())}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_statistics.json").write_text(json.dumps(res, indent=2) + "\n")
    md = ["# |S~| and terminal backward tests (Appendix B, Backward steps)", "",
          "| runs | median/max \\|S~\\| | median/max \\|S_hat\\| | median/max tests |",
          "|---|---|---|---|"]
    for name, r in [("all", s)] + list(res["per_dictionary"].items()):
        md.append(f"| {name} ({r['runs']}) | {r['S_tilde_median']:g} / {r['S_tilde_max']} | "
                  f"{r['S_hat_median']:g} / {r['S_hat_max']} | "
                  f"{r['tests_median']:g} / {r['tests_max']} |")
    f = res["forward_only_main_setting"]
    md += ["", f"Forward step alone, main setting (k20/sae): |S_hat| median "
           f"{f['S_hat_median']:g}, max {f['S_hat_max']}.",
           f"Bound |S~| 2^(|S~|-1) at max |S~|: {res['bound']}.",
           f"Run with the most tests: {res['max_tests_run']}."]
    if "iclr_main_crosscheck" in res:
        md.append(f"ICLR main-setting cross-check: {res['iclr_main_crosscheck']}.")
    text = "\n".join(md) + "\n"
    (OUT / "run_statistics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
