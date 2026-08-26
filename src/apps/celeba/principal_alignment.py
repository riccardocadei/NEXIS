"""
Empirical diagnostics for Principal Alignment (Assumption 3) on CelebA.

Rebuttal analysis: Principal Alignment is *not* monosemanticity.  This script
quantifies, on the very dictionary used in the paper, (i) how far the principal
coordinates are from a one-to-one concept mapping, and (ii) the only quantity
that actually matters for Theorem 4.1.1, namely the residual conditional
heterogeneity signal ("leakage") left in the non-principal coordinates once the
principals are conditioned on.

Three diagnostics
-----------------
1. Selectivity spectrum
   Best-threshold F1 of every coordinate against each ground-truth attribute.
   F1 < 1 for the principal, and non-trivial F1 for many companions, is direct
   evidence that the dictionary is *not* monosemantic.

2. Polysemanticity of the principal coordinate
   How many other CelebA attributes the principal coordinate also predicts.
   Principal Alignment places no constraint in this direction: a coordinate may
   encode arbitrarily many concepts.

3. Leakage spectrum  (the quantity in Lemma B.1(b) / the rho-gate, Eq. C.3)
       c_j := || E[tau | Z^{S* u {j}}] - E[tau | Z^{S*}] ||_2      j not in S*
       c_k := || E[tau | Z^{S*}] - E[tau | Z^{S* \ {j_k}}] ||_2    j_k in S*
   with the linear projection used by the default CATE-equivalence test.  Under
   exact Principal Alignment c_j = 0 for all j not in S*.  The scale-free ratio
       eps_hat := max_{j not in S*} c_j / min_k c_k
   is the empirical counterpart of the approximate-alignment parameter: it is
   free of n and sigma, hence directly comparable to the spectral gap rho.
   eps_hat > 0 means Assumption 3 is violated; eps_hat < rho means the relaxed
   condition under which exact recovery still holds is satisfied.

4. Conditional-independence probe (optional, --probe)
   Held-out check of Assumption 3 as stated: is W^k independent of the rest of
   the dictionary given the principal coordinate?  Compares the AUC of a probe
   on the principal alone against a probe on the principal plus the top-K other
   coordinates.  Delta-AUC > 0 is a direct violation.

Usage
-----
python src/apps/celeba/principal_alignment.py                     # main setting
python src/apps/celeba/principal_alignment.py --sae-top-k 5
python src/apps/celeba/principal_alignment.py --precode
python src/apps/celeba/principal_alignment.py --backbone dinov2
python src/apps/celeba/principal_alignment.py --all --probe       # every dictionary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.experiment import compute_f1_scores
from apps.celeba.backbones import (
    BACKBONES, DEFAULT_BACKBONE, get_backbone,
    code_path as _code_path, precode_path as _precode_path,
)


# ── linear-projection leakage ─────────────────────────────────────────────────

def _residualize(M: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Residualize the columns of M on the columns of A (intercept included)."""
    A1 = np.column_stack([np.ones(len(A)), A])
    coef, *_ = np.linalg.lstsq(A1, M, rcond=None)
    return M - A1 @ coef


def leakage_spectrum(
    Z: np.ndarray,
    tau: np.ndarray,
    principals: Sequence[int],
    chunk: int = 2048,
) -> Dict[str, np.ndarray | float]:
    """Conditional heterogeneity signal of every coordinate given the principals.

    Returns, for each j not in S*, the incremental projection norm
    c_j = |cov(r_j, r_tau)| / sd(r_j) where r denotes residuals on Z^{S*}, and
    for each principal j_k the same quantity conditioning on S* \\ {j_k}.
    """
    P = [int(j) for j in principals]
    n, m = Z.shape

    # --- non-principal coordinates: condition on the full S*
    A = Z[:, P]
    r_tau = _residualize(tau.reshape(-1, 1).astype(np.float64), A).ravel()
    sd_tau = float(r_tau.std())

    c = np.zeros(m, dtype=np.float64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        R = _residualize(Z[:, s:e].astype(np.float64), A)
        sd = R.std(axis=0)
        cov = (R * r_tau[:, None]).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c[s:e] = np.where(sd > 1e-10, np.abs(cov) / sd, 0.0)
    c[P] = 0.0  # exactly collinear with the conditioning set

    # --- principals: condition on the other principals only
    c_principal: Dict[int, float] = {}
    for jk in P:
        others = [j for j in P if j != jk]
        A_k = Z[:, others] if others else np.zeros((n, 0))
        r_t = _residualize(tau.reshape(-1, 1).astype(np.float64), A_k).ravel()
        r_j = _residualize(Z[:, [jk]].astype(np.float64), A_k).ravel()
        sd = float(r_j.std())
        c_principal[jk] = (
            float(abs((r_j * r_t).mean()) / sd) if sd > 1e-10 else 0.0
        )

    weakest = min(c_principal.values()) if c_principal else float("nan")
    return {
        "c": c,
        "c_principal": c_principal,
        "weakest_principal": weakest,
        "max_leak": float(c.max()),
        "argmax_leak": int(c.argmax()),
        "eps_hat": float(c.max() / weakest) if weakest > 0 else float("inf"),
        "sd_tau_given_principals": sd_tau,
    }


# ── conditional-independence probe ────────────────────────────────────────────

def ci_probe(
    Z: np.ndarray,
    w: np.ndarray,
    principal: int,
    n_extra: int = 200,
    n_bins: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Held-out test of  W  indep  Z^{-j}  |  Z^{j}  (Assumption 3 as stated).

    Probe 1: W ~ binned(Z^j)                     (the principal alone)
    Probe 2: W ~ binned(Z^j) + top-n_extra other coordinates
    The extra coordinates are selected on the training half only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss

    rng = np.random.default_rng(seed)
    n = len(w)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    zj = Z[:, principal].astype(np.float64)

    # flexible 1-d expansion of the principal: quantile-bin dummies + linear term
    qs = np.quantile(zj[tr][zj[tr] > 0], np.linspace(0, 1, n_bins + 1)[1:-1]) \
        if (zj[tr] > 0).sum() > n_bins else np.array([])
    edges = np.unique(np.concatenate([[0.0], qs]))
    B = np.column_stack(
        [zj] + [(zj > t).astype(np.float64) for t in edges]
    )

    # Candidate companions: strongest partial correlation with the residual of W
    # given the principal.  Everything here is computed on the TRAINING half only
    # -- the residualisation of W included -- so the held-out AUC below is not
    # contaminated by the coordinate-selection step.
    r_w_tr = _residualize(w[tr].reshape(-1, 1).astype(np.float64), B[tr]).ravel()
    others = np.setdiff1d(np.arange(Z.shape[1]), [principal])
    score = np.zeros(len(others))
    for s in range(0, len(others), 2048):
        e = min(s + 2048, len(others))
        Rj = _residualize(Z[np.ix_(tr, others[s:e])].astype(np.float64), B[tr])
        sd = Rj.std(axis=0)
        cov = (Rj * r_w_tr[:, None]).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            score[s:e] = np.where(sd > 1e-10, np.abs(cov) / sd, 0.0)
    extra = others[np.argsort(-score)[:n_extra]]

    out = {}
    for name, X in [("principal", B), ("principal+rest", np.column_stack([B, Z[:, extra]]))]:
        mdl = LogisticRegression(max_iter=2000, C=1.0)
        mdl.fit(X[tr], w[tr])
        p = mdl.predict_proba(X[te])[:, 1]
        out[f"auc_{name}"] = float(roc_auc_score(w[te], p))
        out[f"logloss_{name}"] = float(log_loss(w[te], p))
    out["delta_auc"] = out["auc_principal+rest"] - out["auc_principal"]
    out["delta_logloss"] = out["logloss_principal"] - out["logloss_principal+rest"]
    out["n_extra"] = float(n_extra)
    return out


# ── per-dictionary report ─────────────────────────────────────────────────────

def analyse_dictionary(
    data_dir: Path,
    backbone: str,
    sae_top_k: int,
    precode: bool,
    w1_attr: str,
    w2_attr: str,
    gamma_w1: float,
    gamma_w2: float,
    probe: bool,
    n_extra: int,
) -> Dict:
    spec = get_backbone(backbone)
    reg_file = (_precode_path if precode else _code_path)(data_dir, spec.key, sae_top_k)
    pre_file = _precode_path(data_dir, spec.key, sae_top_k)

    label = f"{spec.label} k={sae_top_k} {'z_pre' if precode else 'z'}"
    print(f"\n{'='*78}\n{label}\n{'='*78}")

    Z = np.load(reg_file)
    Zpre = np.load(pre_file) if pre_file.exists() else Z
    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    W1 = labels_df[w1_attr].values.astype(np.float64)
    W2 = labels_df[w2_attr].values.astype(np.float64)
    print(f"dictionary: {Z.shape[0]:,} images x {Z.shape[1]:,} coordinates  "
          f"(sparsity {(Z == 0).mean():.3f})")

    # 1. selectivity spectrum — ground truth is defined on z_pre (as in the paper)
    f1 = {w: compute_f1_scores(Zpre, v) for w, v in [(w1_attr, W1), (w2_attr, W2)]}
    principals: Dict[str, int] = {w: int(np.argmax(s)) for w, s in f1.items()}
    S_star = sorted(set(principals.values()))

    rep: Dict = {"label": label, "backbone": spec.key, "sae_top_k": sae_top_k,
                 "precode": precode, "m": int(Z.shape[1]), "n_images": int(Z.shape[0]),
                 "S_star": S_star, "attrs": {}}

    for w, s in f1.items():
        order = np.argsort(-s)
        jk = int(order[0])
        rep["attrs"][w] = {
            "principal": jk,
            "f1_principal": float(s[jk]),
            "f1_runner_up": float(s[order[1]]),
            "f1_top5": [float(x) for x in s[order[:5]]],
            "n_f1_above_half_best": int((s >= 0.5 * s[jk]).sum()),
            "n_f1_above_0.3": int((s >= 0.3).sum()),
            "prevalence": float((W1 if w == w1_attr else W2).mean()),
        }
        print(f"\n[{w}]  principal = {jk}   F1 = {s[jk]:.3f}  "
              f"(runner-up {s[order[1]]:.3f})")
        print(f"   F1 top-5: {np.round(s[order[:5]], 3).tolist()}")
        print(f"   #coords with F1 >= 0.5*best: {(s >= 0.5*s[jk]).sum()}   "
              f"F1 >= 0.3: {(s >= 0.3).sum()}")

    # 2. polysemanticity of each principal: which other attributes it predicts.
    # AUC, not F1: F1 is inflated for high-prevalence attributes (a constant
    # predictor already scores 2p/(1+p)), which would overstate polysemanticity.
    from sklearn.metrics import roc_auc_score
    attr_cols = [c for c in labels_df.columns if c != "celeb_id"]
    for w, jk in principals.items():
        sel = []
        zj = Zpre[:, jk].astype(np.float64)
        for c in attr_cols:
            v = labels_df[c].values.astype(int)
            if v.std() == 0:
                continue
            auc = float(roc_auc_score(v, zj))
            # direction-free selectivity: |AUC - 0.5| + 0.5
            sel.append((c, max(auc, 1.0 - auc), float(v.mean())))
        sel.sort(key=lambda x: -x[1])
        n_poly = sum(1 for _, a, _ in sel if a >= 0.75)
        rep["attrs"][w]["principal_also_predicts_auc"] = sel[:10]
        rep["attrs"][w]["n_attrs_auc_above_0.75"] = n_poly
        print(f"\n[{w}] principal coord {jk} polysemanticity — attributes by AUC "
              f"({n_poly}/{len(sel)} attributes with AUC >= 0.75):")
        for c, a, p in sel[:8]:
            print(f"     {c:<22s} AUC={a:.3f}  (prevalence {p:.2f})"
                  f"{'   <-- target' if c == w else ''}")

    # 3. leakage spectrum
    tau = gamma_w1 * W1 + gamma_w2 * W2
    lk = leakage_spectrum(Z, tau, S_star)
    c = lk["c"]
    order = np.argsort(-c)
    rep["leakage"] = {
        "c_principal": {str(k): v for k, v in lk["c_principal"].items()},
        "weakest_principal": lk["weakest_principal"],
        "max_leak": lk["max_leak"],
        "argmax_leak": lk["argmax_leak"],
        "eps_hat": lk["eps_hat"],
        "top10_leak_coords": [int(j) for j in order[:10]],
        "top10_leak_ratio": [float(c[j] / lk["weakest_principal"]) for j in order[:10]],
        "n_leak_above_0.5": int((c >= 0.5 * lk["weakest_principal"]).sum()),
        "n_leak_above_0.2": int((c >= 0.2 * lk["weakest_principal"]).sum()),
    }
    print(f"\nleakage spectrum (linear projection, full population)")
    print(f"   principal contributions c_k : "
          f"{ {k: round(v, 4) for k, v in lk['c_principal'].items()} }")
    print(f"   weakest principal          : {lk['weakest_principal']:.4f}")
    print(f"   max non-principal leak     : {lk['max_leak']:.4f}  (coord {lk['argmax_leak']})")
    print(f"   eps_hat = max leak / weakest principal = {lk['eps_hat']:.3f}"
          f"   [rho default = 0.5]")
    print(f"   top-10 leak ratios         : "
          f"{np.round(rep['leakage']['top10_leak_ratio'], 3).tolist()}")
    print(f"   #coords with ratio >= 0.5  : {rep['leakage']['n_leak_above_0.5']}   "
          f">= 0.2: {rep['leakage']['n_leak_above_0.2']}")

    # 4. conditional-independence probe
    if probe:
        rep["ci_probe"] = {}
        for w, jk in principals.items():
            wv = (W1 if w == w1_attr else W2).astype(int)
            pr = ci_probe(Z, wv, jk, n_extra=n_extra)
            rep["ci_probe"][w] = pr
            print(f"\nCI probe [{w}]  W indep Z^-j | Z^j ?")
            print(f"   AUC(principal)      = {pr['auc_principal']:.4f}")
            print(f"   AUC(principal+rest) = {pr['auc_principal+rest']:.4f}"
                  f"   delta = {pr['delta_auc']:+.4f}")
    return rep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default="data/celeba")
    p.add_argument("--out", type=Path,
                   default="results/celeba/appendix/principal_alignment.json")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    p.add_argument("--sae-top-k", type=int, default=20)
    p.add_argument("--precode", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="Report every available dictionary (k in {5,20} x {z, z_pre} "
                        "x every backbone with cached codes).")
    p.add_argument("--probe", action="store_true",
                   help="Run the held-out conditional-independence probe (slower).")
    p.add_argument("--n-extra", type=int, default=200)
    p.add_argument("--w1-attr", default="Wearing_Hat")
    p.add_argument("--w2-attr", default="Eyeglasses")
    p.add_argument("--gamma-w1", type=float, default=1.0)
    p.add_argument("--gamma-w2", type=float, default=-1.0)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir

    if args.all:
        combos = [(b, k, pc) for b in sorted(BACKBONES)
                  for k in (20, 5) for pc in (False, True)
                  if _code_path(data_dir, get_backbone(b).key, k).exists()]
    else:
        combos = [(args.backbone, args.sae_top_k, args.precode)]

    reports = []
    for b, k, pc in combos:
        reports.append(analyse_dictionary(
            data_dir, b, k, pc, args.w1_attr, args.w2_attr,
            args.gamma_w1, args.gamma_w2, args.probe, args.n_extra,
        ))

    out = ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nwrote {out}")

    # compact summary table
    print(f"\n{'='*100}")
    print(f"{'dictionary':<34s} {'F1(W1)':>7s} {'F1(W2)':>7s} "
          f"{'runnerF1':>9s} {'eps_hat':>8s} {'#leak>0.5':>10s} {'#leak>0.2':>10s}")
    print("-" * 100)
    for r in reports:
        a = list(r["attrs"].values())
        print(f"{r['label']:<34s} {a[0]['f1_principal']:>7.3f} {a[1]['f1_principal']:>7.3f} "
              f"{max(x['f1_runner_up'] for x in a):>9.3f} "
              f"{r['leakage']['eps_hat']:>8.3f} "
              f"{r['leakage']['n_leak_above_0.5']:>10d} "
              f"{r['leakage']['n_leak_above_0.2']:>10d}")


if __name__ == "__main__":
    main()
