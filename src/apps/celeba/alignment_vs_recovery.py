"""
Recovery of S* as a function of *measured* Principal Alignment quality.

Rebuttal analysis.  Assumption 3 is a property of the dictionary, hence under
the analyst's control and checkable before any experiment is run.  This script
makes that operational: for every available dictionary (backbone x SAE sparsity
x feature view) it reports

  * the alignment diagnostics of principal_alignment.py — in particular the
    scale-free leakage ratio eps_hat, the empirical counterpart of the
    approximate-alignment parameter, directly comparable to the spectral gap rho;
  * NEXIS recovery of S* at a fixed reference cell (n, eta), against the
    marginal-screening baseline.

The dictionaries span a wide alignment range without any synthetic perturbation
(SigLIP is well aligned, DINOv2 is not), which lets recovery be plotted against
a quantity an analyst can estimate rather than against an unobservable.

Caveat carried into the output: the ground-truth S* is defined as the
top-F1 coordinate per attribute.  When eps_hat > 1 some non-principal
coordinate carries more conditional heterogeneity signal than the weakest
"principal", i.e. the F1-defined target is not the minimal sufficient set of
that dictionary, and precision against it is no longer the right metric.  Such
rows are flagged.

Usage
-----
python src/apps/celeba/alignment_vs_recovery.py                  # all dictionaries
python src/apps/celeba/alignment_vs_recovery.py --n-seeds 50 --fixed-n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from method.nexis import nexis, marginal_select, iou_score
from apps.celeba.scm import build_buckets, generate_celeba_rct
from apps.celeba.experiment import compute_f1_scores
from apps.celeba.principal_alignment import leakage_spectrum
from apps.celeba.violation_sweep import population_r2, _metrics
from apps.celeba.backbones import (
    BACKBONES, get_backbone, code_path as _code_path, precode_path as _precode_path,
)


def run_one(Z, labels_df, buckets, tau_pop, truth, ref_r2, n, es, seed,
            alpha, max_rounds, scm_kwargs) -> List[Dict]:
    d = generate_celeba_rct(n=n, features=Z, labels_df=labels_df, buckets=buckets,
                            effect_scale=es, seed=seed, **scm_kwargs)
    runners = {
        "NEXIS": lambda: nexis(y=d.Y, t=d.T, z=d.Z, alpha=alpha,
                               max_rounds=max_rounds, rho=0.5),
        "NEXIS (rho=0)": lambda: nexis(y=d.Y, t=d.T, z=d.Z, alpha=alpha,
                                       max_rounds=max_rounds, rho=0.0),
        "Marginal Testing (FWER)": lambda: marginal_select(
            y=d.Y, t=d.T, z=d.Z, alpha=alpha, adjust="FWER"),
    }
    rows = []
    for name, fn in runners.items():
        sel = list(fn().selected)
        r = {"method": name, "n": n, "effect_scale": es, "seed": seed,
             "n_selected": len(sel)}
        r |= _metrics(sel, truth)
        r2 = population_r2(Z, tau_pop, sel)
        r["suff_r2"] = r2
        r["suff_ratio"] = r2 / ref_r2 if ref_r2 > 0 else np.nan
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default="data/celeba")
    p.add_argument("--out-dir", type=Path,
                   default="results/celeba/experiment/alignment")
    p.add_argument("--fixed-n", type=int, nargs="+", default=[2000])
    p.add_argument("--fixed-effect", type=float, nargs="+", default=[5.0])
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--w1-attr", default="Wearing_Hat")
    p.add_argument("--w2-attr", default="Eyeglasses")
    p.add_argument("--tau0", type=float, default=0.5)
    p.add_argument("--gamma-w1", type=float, default=1.0)
    p.add_argument("--gamma-w2", type=float, default=-1.0)
    p.add_argument("--noise-sd", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    data_dir = ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir
    out_dir = ROOT / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    W1 = labels_df[args.w1_attr].values.astype(np.float64)
    W2 = labels_df[args.w2_attr].values.astype(np.float64)
    tau_pop = args.gamma_w1 * W1 + args.gamma_w2 * W2
    buckets = build_buckets(labels_df, args.w1_attr, args.w2_attr)
    scm_kwargs = dict(w1_attr=args.w1_attr, w2_attr=args.w2_attr, tau_0=args.tau0,
                      gamma_w1=args.gamma_w1, gamma_w2=args.gamma_w2,
                      noise_sd=args.noise_sd)

    combos = [(b, k, pc) for b in sorted(BACKBONES) for k in (20, 5)
              for pc in (False, True)
              if _code_path(data_dir, get_backbone(b).key, k).exists()]

    rows: List[Dict] = []
    summary: List[Dict] = []
    for b, k, pc in combos:
        spec = get_backbone(b)
        label = f"{spec.label} k={k} {'z_pre' if pc else 'z'}"
        reg = (_precode_path if pc else _code_path)(data_dir, spec.key, k)
        pre = _precode_path(data_dir, spec.key, k)
        Z = np.load(reg)
        Zpre = np.load(pre) if pre.exists() else Z

        f1_1, f1_2 = compute_f1_scores(Zpre, W1), compute_f1_scores(Zpre, W2)
        j1, j2 = int(np.argmax(f1_1)), int(np.argmax(f1_2))
        truth = sorted({j1, j2})
        lk = leakage_spectrum(Z, tau_pop, truth)
        ref_r2 = population_r2(Z, tau_pop, truth)

        print(f"\n{'='*84}\n{label}   S*={truth}   "
              f"F1=({f1_1[j1]:.3f}, {f1_2[j2]:.3f})   "
              f"eps_hat={lk['eps_hat']:.3f}   R2(S*)={ref_r2:.3f}")
        if lk["eps_hat"] > 1.0:
            print("  ** eps_hat > 1: the F1-defined S* is NOT the minimal "
                  "sufficient set of this dictionary; precision against it is "
                  "not the right metric.")

        rec: Dict[str, Dict[str, float]] = {}
        for n in args.fixed_n:
            for es in args.fixed_effect:
                out = Parallel(n_jobs=args.n_jobs)(
                    delayed(run_one)(Z, labels_df, buckets, tau_pop, truth, ref_r2,
                                     n, es, s, args.alpha, args.max_steps, scm_kwargs)
                    for s in range(args.n_seeds))
                flat = [r | {"dictionary": label, "backbone": spec.key,
                             "sae_top_k": k, "precode": pc,
                             "eps_hat": lk["eps_hat"]}
                        for batch in out for r in batch]
                rows.extend(flat)
                agg = pd.DataFrame(flat).groupby("method")[
                    ["recall", "precision", "iou", "suff_ratio", "n_selected"]].mean()
                print(f"  n={n}  eta={es}")
                print(agg.round(3).to_string().replace("\n", "\n    "))
                for m, r in agg.iterrows():
                    rec[f"{m} (n={n},eta={es})"] = r.round(4).to_dict()

        summary.append({
            "dictionary": label, "backbone": spec.key, "sae_top_k": k,
            "precode": pc, "S_star": truth,
            "f1_w1": float(f1_1[j1]), "f1_w2": float(f1_2[j2]),
            "eps_hat": lk["eps_hat"], "max_leak": lk["max_leak"],
            "weakest_principal": lk["weakest_principal"],
            "r2_S_star": ref_r2, "recovery": rec,
            "f1_target_is_minimal_sufficient": bool(lk["eps_hat"] <= 1.0),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "alignment_vs_recovery.parquet", index=False)
    with open(out_dir / "alignment_vs_recovery.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*104}")
    print(f"{'dictionary':<26s} {'F1(W1)':>7s} {'F1(W2)':>7s} {'eps_hat':>8s} "
          f"{'NEXIS prec':>11s} {'NEXIS rec':>10s} {'NEXIS IoU':>10s} "
          f"{'Marg prec':>10s} {'|S_hat|':>8s}")
    print("-" * 104)
    key_n, key_e = args.fixed_n[0], args.fixed_effect[0]
    for s in summary:
        nx = s["recovery"][f"NEXIS (n={key_n},eta={key_e})"]
        mg = s["recovery"][f"Marginal Testing (FWER) (n={key_n},eta={key_e})"]
        flag = "" if s["f1_target_is_minimal_sufficient"] else "  **"
        print(f"{s['dictionary']:<26s} {s['f1_w1']:>7.3f} {s['f1_w2']:>7.3f} "
              f"{s['eps_hat']:>8.3f} {nx['precision']:>11.3f} {nx['recall']:>10.3f} "
              f"{nx['iou']:>10.3f} {mg['precision']:>10.3f} "
              f"{nx['n_selected']:>8.2f}{flag}")
    print("\n**  eps_hat > 1: F1-defined S* is not this dictionary's minimal "
          "sufficient set (see note above).")
    print(f"wrote {out_dir}/alignment_vs_recovery.{{parquet,json}}")


if __name__ == "__main__":
    main()
