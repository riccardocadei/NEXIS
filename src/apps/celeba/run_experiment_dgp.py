#!/usr/bin/env python3
"""
CelebA sweeps for alternative data-generating processes: r = 0 or 1 direct modifiers.

run_experiment.py needs r >= 2 modifier attributes and makes every sampled attribute a
modifier.  This runner separates the two: the *sampled* attributes (--w-attrs) fix how
images are drawn, and --gammas says which of them modify the effect.  S* is the set of
ground-truth coordinates of the attributes with gamma != 0.  With the default
--w-attrs Wearing_Hat Eyeglasses the sampler is the published one, so for a given seed
every DGP draws the same units, T and images as the main setting; only Y changes.

  r = 1:  --gammas 1 0     tau = tau_0 + eta * W_hat             S* = {hat coordinate}
                           (Eyeglasses stays a prognostic, non-modifying attribute)
  r = 0:  --gammas 0 0     tau = tau_0 (constant)                S* = {}
          --scale beta     eta multiplies the prognostic main effects instead:
                           Y = eta * (0.3 W_hat - 0.2 W_glasses) + tau_0 T + noise

Why --scale beta at r = 0: with gamma = 0 an effect-scale multiplier on gamma does
nothing, and a larger constant effect tau_0 does not change the linear test at all
(T is in the nuisance design, so adding c*T to Y leaves every interaction t-statistic
unchanged).  Scaling the prognostic strength of the attributes is the axis along which
the null actually becomes harder: the SAE coordinates that encode hats and glasses
predict Y more and more strongly, without modifying the effect.

Writes out-dir/k{K}/sae/{effect_sweep,n_sweep}.parquet (same schema as run_experiment.py,
"effect_scale" = eta), ground_truth.json (copied from --gt-json) and dgp.json.

    python src/apps/celeba/run_experiment_dgp.py --gammas 1 0 --out-dir results/celeba/experiment_v2_r1
    python src/apps/celeba/run_experiment_dgp.py --gammas 0 0 --scale beta \
        --out-dir results/celeba/experiment_v2_r0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.scm import build_buckets, generate_celeba_rct
from apps.celeba.experiment import run_sweep, evaluate_methods_on_dataset

EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default="data/celeba")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sae-top-k", type=int, default=20)
    p.add_argument("--gt-json", type=Path,
                   default="results/celeba/experiment/k20/sae/ground_truth.json",
                   help="ground_truth.json of the dictionary; maps attributes to coordinates")
    p.add_argument("--w-attrs", nargs="+", default=["Wearing_Hat", "Eyeglasses"],
                   help="Sampled attributes (default: the published pair)")
    p.add_argument("--gammas", type=float, nargs="+", required=True,
                   help="T x W_k coefficients, one per sampled attribute; 0 = not a modifier")
    p.add_argument("--betas", type=float, nargs="+", default=None,
                   help="Main effects of W_k on Y (default: 0.3/-0.2 alternating)")
    p.add_argument("--scale", choices=["gamma", "beta"], default="gamma",
                   help="What eta multiplies: the interactions (default, as the published "
                        "DGP) or the prognostic main effects (for r = 0)")
    p.add_argument("--tau0", type=float, default=0.5)
    p.add_argument("--noise-sd", type=float, default=1.0)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--gcm-splits", type=int, default=3)
    p.add_argument("--fixed-n", type=int, nargs="+", default=[500, 2000])
    p.add_argument("--fixed-effect", type=float, nargs="+", default=[2.0, 5.0])
    p.add_argument("--effect-grid", type=float, nargs="+", default=EFFECT_GRID)
    p.add_argument("--n-grid", type=int, nargs="+", default=N_GRID)
    p.add_argument("--sweep", choices=["effect", "n", "both"], default="both")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def neurons_by_attr(gt: dict) -> dict[str, list[int]]:
    """Attribute -> ground-truth coordinates, for either ground_truth.json layout."""
    if "neurons_per_attr" in gt and "w_attrs" in gt:
        return {a: list(map(int, v)) for a, v in zip(gt["w_attrs"], gt["neurons_per_attr"])}
    return {gt["w1_attr"]: list(map(int, gt["w1_neurons"])),
            gt["w2_attr"]: list(map(int, gt["w2_neurons"]))}


def sweep_beta(features, labels_df, buckets, truth, sweep_param, param_grid, *,
               fixed_n=None, fixed_effect=None, n_seeds, alpha, max_rounds, methods,
               gcm_splits, w_attrs, betas, gammas, tau_0, noise_sd) -> pd.DataFrame:
    """run_sweep with eta multiplying the main effects betas (gamma held fixed)."""
    def one(pv, seed):
        n = int(pv) if sweep_param == "n" else fixed_n
        eta = pv if sweep_param == "effect_scale" else fixed_effect
        try:
            d = generate_celeba_rct(
                n=n, features=features, labels_df=labels_df, buckets=buckets,
                w_attrs=w_attrs, betas=[eta * b for b in betas], gammas=gammas,
                tau_0=tau_0, noise_sd=noise_sd, effect_scale=1.0, seed=seed)
        except ValueError:
            return pv, seed, None   # bucket exhausted
        return pv, seed, evaluate_methods_on_dataset(
            y=d.Y, t=d.T, z=d.Z, truth=truth, alpha=alpha, max_rounds=max_rounds,
            methods=methods, gcm_splits=gcm_splits)

    tasks = [(pv, s) for pv in param_grid for s in range(n_seeds)]
    print(f"  {len(tasks)} tasks ({len(param_grid)} {sweep_param} values x {n_seeds} seeds)",
          flush=True)
    res = Parallel(n_jobs=-1, prefer="threads")(delayed(one)(pv, s) for pv, s in tasks)
    rows = [{sweep_param: pv, "seed": s, "method": m, **met}
            for pv, s, out in res if out is not None for m, met in out.items()]
    df = pd.DataFrame(rows)
    if sweep_param == "effect_scale":
        df["fixed_n"] = fixed_n
    else:
        df["fixed_effect"] = fixed_effect
    return df


def main():
    a = parse_args()
    data_dir = _abs(a.data_dir)
    out_dir = _abs(a.out_dir) / f"k{a.sae_top_k}" / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"effect": out_dir / "effect_sweep.parquet", "n": out_dir / "n_sweep.parquet"}
    todo = [s for s in ("effect", "n") if a.sweep in (s, "both")]
    if not a.force:
        for s in list(todo):
            if paths[s].exists():
                print(f"exists, skipping (use --force): {paths[s]}")
                todo.remove(s)
    if not todo:
        return

    if len(a.gammas) != len(a.w_attrs):
        raise ValueError(f"--gammas needs {len(a.w_attrs)} values")
    betas = a.betas or [(0.3 if k % 2 == 0 else -0.2) for k in range(len(a.w_attrs))]
    if len(betas) != len(a.w_attrs):
        raise ValueError(f"--betas needs {len(a.w_attrs)} values")

    gt = json.loads(_abs(a.gt_json).read_text())
    by_attr = neurons_by_attr(gt)
    missing = [w for w, g in zip(a.w_attrs, a.gammas) if g != 0 and w not in by_attr]
    if missing:
        raise ValueError(f"no ground-truth coordinate for {missing} in {a.gt_json}")
    truth = sorted({j for w, g in zip(a.w_attrs, a.gammas) if g != 0 for j in by_attr[w]})
    modifiers = [w for w, g in zip(a.w_attrs, a.gammas) if g != 0]
    print(f"sampled attributes {a.w_attrs}; modifiers (r={len(modifiers)}) {modifiers}; "
          f"S* = {truth}; eta scales {a.scale}")

    feat_file = data_dir / "embeddings" / f"sae_k{a.sae_top_k}.npy"
    features = np.load(feat_file)
    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    print(f"features {features.shape} from {feat_file}")
    buckets = build_buckets(labels_df, w_attrs=a.w_attrs)

    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out_dir / "dgp.json").write_text(json.dumps({
        "sampled_attrs": a.w_attrs, "gammas": a.gammas, "betas": betas,
        "modifiers": modifiers, "r": len(modifiers), "truth": truth,
        "eta_scales": a.scale, "tau_0": a.tau0, "noise_sd": a.noise_sd,
        "features": str(feat_file.relative_to(ROOT)), "gt_json": str(a.gt_json),
        "methods": a.methods, "n_seeds": a.n_seeds, "alpha": a.alpha,
        "max_steps": a.max_steps}, indent=2))

    common = dict(n_seeds=a.n_seeds, alpha=a.alpha, max_rounds=a.max_steps,
                  methods=a.methods, gcm_splits=a.gcm_splits)
    scm = dict(w_attrs=a.w_attrs, betas=betas, gammas=a.gammas, tau_0=a.tau0,
               noise_sd=a.noise_sd)

    def run(sweep_param, grid, **fixed):
        if a.scale == "gamma":   # the published code path, eta multiplies gammas
            return run_sweep(features, labels_df, buckets, truth, sweep_param=sweep_param,
                             param_grid=grid, **fixed, **common, **scm)
        return sweep_beta(features, labels_df, buckets, truth, sweep_param, grid,
                          **fixed, **common, **scm)

    if "effect" in todo:
        dfs = []
        for n in a.fixed_n:
            print(f"\n=== eta sweep, n={n} ===", flush=True)
            dfs.append(run("effect_scale", a.effect_grid, fixed_n=n))
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(paths["effect"], index=False)
        print(f"{len(df)} rows -> {paths['effect']}")
    if "n" in todo:
        dfs = []
        for eta in a.fixed_effect:
            print(f"\n=== n sweep, eta={eta} ===", flush=True)
            dfs.append(run("n", a.n_grid, fixed_effect=eta))
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(paths["n"], index=False)
        print(f"{len(df)} rows -> {paths['n']}")


if __name__ == "__main__":
    main()
