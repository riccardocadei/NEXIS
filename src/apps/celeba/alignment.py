#!/usr/bin/env python3
"""Principal Alignment on CelebA (Appendix C), one subcommand each.

  check           supervised check of Principal Alignment on the main dictionary (TopK
                  SAE, k = 20, codes Z): alignment spectrum (pa_spectrum.pdf), top
                  activating images of the principals (pa_top_images.pdf) and the
                  screening-off probe (Table tab:celeba_pa_screening), written to
                  results/celeba/figures_v2/principal_alignment/
  f1              attribute-to-coordinate F1 matrix of a dictionary (the ground-truth
                  rule, experiment.compute_f1_scores, for every CelebA attribute), written
                  to results/celeba/interleaved_backward/align/f1_<dict>.npz
  third-modifier  ranks the CelebA attributes as the third direct modifier of the r = 3
                  DGP from the F1 matrix of sae_precode_k20
                  (results/celeba/figures_v2/third_modifier_candidates.md)

Principal Alignment (Assumption 3): for each direct modifier W^k there is one principal
coordinate j_k with  W^k  indep  Z^{[m] minus j_k}  |  Z^{j_k}.  It does not require
monosemanticity, and it is testable only with supervision.  On CelebA the modifiers
(Wearing_Hat, Eyeglasses) are observed labels, so `check` tests it on the main dictionary
(19,867 x 9,216 codes of mean-pooled SigLIP-2 patch tokens), with the principals of
results/celeba/experiment/k20/sae/ground_truth.json (Hat 5348, Eyeglasses 5537).

`check` outputs:

  pa_spectrum.{pdf,png}     alignment spectrum: the AUC of every coordinate's
                            activation for the label, sorted; principal and runner-up
                            highlighted, with the gap between them
  pa_top_images.{pdf,png}   top-10 activating images of each principal (one row each)
                            and one row of random images for contrast, labelled with the
                            top-100 purity; framed images carry the attribute
  pa_screening.{md,tex}     screening-off probe: held-out AUC of W^k from Z^{j_k}
                            alone vs Z^{j_k} + top-K other coordinates, K in {10, 50, 200},
                            over repeated random 50/50 splits, and a held-out
                            likelihood-ratio test of the K extra coordinates
  pa_summary.json           every number above

The probe is ci_probe below (quantile-bin expansion of the principal, companions selected
on the training half by partial correlation with W given the principal, L2-logistic with
C = 1); ``probe_path`` runs it for several K from one selection and reproduces ci_probe
exactly at K = 200 (checked in ``check``).

Usage (repo root, CPU):
    python src/apps/celeba/alignment.py check [--figures-only]
    python src/apps/celeba/alignment.py f1 sae_precode_k20
    python src/apps/celeba/alignment.py third-modifier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.figures import METHOD_STYLES, _OI_ORANGE, _OI_GREEN  # noqa: E402  (also the rcParams)

F1_DIR = ROOT / "results/celeba/interleaved_backward/align"


# ── linear residualisation and the original conditional-independence probe ───

def _residualize(M: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Residualize the columns of M on the columns of A (intercept included)."""
    A1 = np.column_stack([np.ones(len(A)), A])
    coef, *_ = np.linalg.lstsq(A1, M, rcond=None)
    return M - A1 @ coef


def ci_probe(
    Z: np.ndarray,
    w: np.ndarray,
    principal: int,
    n_extra: int = 200,
    n_bins: int = 10,
    seed: int = 0,
) -> dict[str, float]:
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


CODES = ROOT / "data/celeba/embeddings/sae_k20.npy"
LABELS = ROOT / "data/celeba/labels.parquet"
IMAGES = ROOT / "data/celeba/images.npy"            # 128x128 thumbnails, same row order
GT = ROOT / "results/celeba/experiment/k20/sae/ground_truth.json"
OUT = ROOT / "results/celeba/figures_v2/principal_alignment"

ATTR_LABEL = {"Wearing_Hat": "Wearing Hat", "Eyeglasses": "Eyeglasses"}
ATTR_COLOR = {"Wearing_Hat": _OI_ORANGE, "Eyeglasses": _OI_GREEN}
BLUE = METHOD_STYLES["NEXIS-v2"]["color"]
GREY = "#9a9a9a"
KS = (10, 50, 200)
TOP_M = 100


# ── alignment spectrum ────────────────────────────────────────────────────────

def column_auc(Z: np.ndarray, w: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """AUC of each column of Z as a score for the binary label w (ties count 1/2)."""
    pos = w.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    out = np.empty(Z.shape[1])
    for s in range(0, Z.shape[1], chunk):
        R = rankdata(Z[:, s:s + chunk], axis=0)
        out[s:s + chunk] = (R[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def top_purity(z: np.ndarray, w: np.ndarray, M: int = TOP_M) -> dict:
    """Share of the M highest-activating images that carry the label, and its lift."""
    top = np.argsort(-z, kind="stable")[:M]
    pur = float(w[top].mean())
    return {"purity": pur, "lift": pur / float(w.mean()), "n_active": int((z > 0).sum())}


def spectrum(Z: np.ndarray, w: np.ndarray, j: int) -> tuple[dict, np.ndarray]:
    """Sign-free AUC, max(AUC, 1 - AUC), of every coordinate: a coordinate that switches
    OFF with the attribute is as informative as one that switches on."""
    auc_raw = column_auc(Z, w)
    auc = np.maximum(auc_raw, 1 - auc_raw)
    others = np.arange(Z.shape[1]) != j
    runner = int(np.flatnonzero(others)[np.argmax(auc[others])])
    # runner-up by top-100 purity, among coordinates active on >= 100 images
    active = (Z > 0).sum(axis=0) >= TOP_M
    cand = np.flatnonzero(others & active)
    pur = np.array([w[np.argpartition(-Z[:, c], TOP_M)[:TOP_M]].mean() for c in cand])
    runner_pur = int(cand[np.argmax(pur)])
    rep = {
        "principal": int(j), "prevalence": float(w.mean()),
        "auc_principal": float(auc[j]), "runner_up_auc": runner,
        "auc_runner_up": float(auc[runner]), "auc_gap": float(auc[j] - auc[runner]),
        "auc_runner_up_signed": float(auc_raw[runner]),
        "rank_principal_auc": int((auc > auc[j]).sum() + 1),
        "n_auc_ge_0.7": int((auc[others] >= 0.7).sum()),
        "n_auc_ge_0.6": int((auc[others] >= 0.6).sum()),
        "top100_principal": top_purity(Z[:, j], w),
        "top100_runner_up_auc": top_purity(Z[:, runner], w),
        "runner_up_purity": runner_pur,
        "top100_runner_up_purity": top_purity(Z[:, runner_pur], w),
        "auc_runner_up_purity": float(auc[runner_pur]),
    }
    return rep, auc


# ── screening-off probe ───────────────────────────────────────────────────────

def _basis(zj: np.ndarray, tr: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """ci_probe's expansion of the principal: linear term + quantile-bin indicators."""
    pos = zj[tr][zj[tr] > 0]
    qs = np.quantile(pos, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(pos) > n_bins else np.array([])
    edges = np.unique(np.concatenate([[0.0], qs]))
    return np.column_stack([zj] + [(zj > t).astype(np.float64) for t in edges])


def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def probe_path(Z: np.ndarray, w: np.ndarray, j: int, Ks=KS, seed: int = 0) -> list[dict]:
    """ci_probe for several K from a single companion selection, plus a held-out LRT.

    AUC / log-loss: fit on the training half, evaluate on the held-out half (as ci_probe).
    LRT: on the held-out half only (the companions were chosen on the training half, so
    the test is valid given the selection), unpenalised logistic fits of
    W ~ basis(Z^j)  vs  W ~ basis(Z^j) + K companions, 2 * dloglik ~ chi2_K under
    W indep companions | basis(Z^j).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    n = len(w)
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    zj = Z[:, j].astype(np.float64)
    B = _basis(zj, tr)
    r_w = _residualize(w[tr].reshape(-1, 1).astype(np.float64), B[tr]).ravel()
    others = np.setdiff1d(np.arange(Z.shape[1]), [j])
    score = np.zeros(len(others))
    for s in range(0, len(others), 2048):
        e = min(s + 2048, len(others))
        Rj = _residualize(Z[np.ix_(tr, others[s:e])].astype(np.float64), B[tr])
        sd = Rj.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            score[s:e] = np.where(sd > 1e-10, np.abs((Rj * r_w[:, None]).mean(axis=0)) / sd, 0.0)
    ranked = others[np.argsort(-score)]

    def fit_eval(X):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], w[tr])
        p = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(w[te], p), log_loss(w[te], p)

    def heldout_ll(X):
        Xs = X[te]
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(penalty=None, max_iter=5000).fit(Xs, w[te])
        return _loglik(w[te], m.predict_proba(Xs)[:, 1])

    auc0, ll0 = fit_eval(B)
    lik0 = heldout_ll(B)
    p1 = w[te].mean()
    H = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    rows = []
    for K in Ks:
        X = np.column_stack([B, Z[:, ranked[:K]]])
        auc1, ll1 = fit_eval(X)
        lr = 2 * (heldout_ll(X) - lik0)
        rows.append({"seed": seed, "K": K, "auc_principal": auc0, "auc_augmented": auc1,
                     "delta_auc": auc1 - auc0, "logloss_principal": ll0,
                     "logloss_augmented": ll1,
                     "info_share": (H - ll0) / (H - ll1),
                     "lrt_stat": lr, "lrt_logp": float(chi2.logsf(max(lr, 0.0), K)),
                     "companions_top5": [int(c) for c in ranked[:5]]})
    return rows


def summarise_probe(df: pd.DataFrame) -> pd.DataFrame:
    def q(x, a):
        return float(np.quantile(x, a))
    out = []
    for (attr, K), d in df.groupby(["attr", "K"], sort=False):
        row = {"attr": attr, "K": int(K), "n_splits": len(d)}
        for c in ("auc_principal", "auc_augmented", "delta_auc", "info_share"):
            row[c] = float(d[c].mean())
            row[c + "_lo"], row[c + "_hi"] = q(d[c], 0.025), q(d[c], 0.975)
        row["lrt_stat_median"] = float(d.lrt_stat.median())
        row["lrt_log10p_max"] = float(d.lrt_logp.max() / np.log(10))
        row["share_splits_reject_0.05"] = float((d.lrt_logp < np.log(0.05)).mean())
        out.append(row)
    return pd.DataFrame(out)


def write_tables(s: pd.DataFrame) -> None:
    fmt = lambda r, c, p=3: f"{r[c]:.{p}f} [{r[c + '_lo']:.{p}f}, {r[c + '_hi']:.{p}f}]"
    md = ["| Modifier | K | AUC, principal only | AUC, principal + K others | ΔAUC | "
          "Info. share of principal | LRT: median χ² (df = K), max p |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lrcccc}", r"\toprule",
           r"Modifier & $K$ & AUC $Z^{j_k}$ & AUC $Z^{j_k}$ + $K$ others & $\Delta$AUC & "
           r"info.\ share \\", r"\midrule"]
    for i, r in s.iterrows():
        first = i == 0 or s.attr[i - 1] != r.attr
        md.append(f"| {ATTR_LABEL[r.attr] if first else ''} | {r.K} | "
                  f"{fmt(r, 'auc_principal') if first else ''} | {fmt(r, 'auc_augmented')} | "
                  f"{fmt(r, 'delta_auc')} | {fmt(r, 'info_share', 2)} | "
                  f"{r.lrt_stat_median:.0f}, p ≤ 1e{int(np.ceil(r.lrt_log10p_max))} |")
        tex.append((rf"\multirow{{3}}{{*}}{{{ATTR_LABEL[r.attr]}}}" if first else "")
                   + f" & {r.K} & "
                   + (rf"\multirow{{3}}{{*}}{{{r.auc_principal:.3f}}}" if first else "")
                   + f" & {r.auc_augmented:.3f} & "
                   f"{r.delta_auc:+.3f} [{r.delta_auc_lo:+.3f}, {r.delta_auc_hi:+.3f}] & "
                   f"{r.info_share:.2f} \\\\")
        if not first and (i + 1 == len(s) or s.attr[i + 1] != r.attr):
            tex.append(r"\midrule" if i + 1 < len(s) else r"\bottomrule")
    tex.append(r"\end{tabular}")
    (OUT / "pa_screening.md").write_text("\n".join(md) + "\n")
    (OUT / "pa_screening.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


# ── figures ───────────────────────────────────────────────────────────────────

def figure_spectrum(aucs: dict, reps: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    for ax, (attr, auc) in zip(axes, aucs.items()):
        r, col = reps[attr], ATTR_COLOR[attr]
        s = np.sort(auc)[::-1]
        rank = np.arange(1, len(s) + 1)
        ax.scatter(rank[2:], s[2:], s=4, color=GREY, lw=0, zorder=1)
        ax.axhline(0.5, color=GREY, ls=":", lw=1)
        ax.scatter([1], [r["auc_principal"]], marker="*", s=240, color=col, ec="k", lw=0.6,
                   zorder=4, label=f"principal $j_k$ = {r['principal']}")
        ax.scatter([2], [r["auc_runner_up"]], marker="o", s=40, color=GREY, ec="k", lw=0.6,
                   zorder=4, label=f"runner-up = {r['runner_up_auc']}")
        x_gap = 0.06 * len(s)                    # the two points sit at ranks 1 and 2
        for y in (r["auc_principal"], r["auc_runner_up"]):
            ax.plot([1, x_gap], [y, y], color="k", ls=":", lw=0.8, zorder=2)
        ax.annotate("", xy=(x_gap, r["auc_runner_up"]), xytext=(x_gap, r["auc_principal"]),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1))
        ax.text(x_gap + 0.02 * len(s), (r["auc_principal"] + r["auc_runner_up"]) / 2,
                f"gap {r['auc_gap']:.2f}", va="center", fontsize=11)
        ax.set_xlim(-0.03 * len(s), 1.03 * len(s))
        ax.set_ylim(0.35, 1.02)
        ax.set_title(ATTR_LABEL[attr])
        ax.set_xlabel("coordinate rank (sorted by AUC)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", frameon=False, fontsize=10, handletextpad=0.3)
    axes[0].set_ylabel("AUC of $W^k \\sim Z^j$\n(sign-free)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pa_spectrum.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / 'pa_spectrum.pdf'}")


def figure_top_images(Z: np.ndarray, labels: pd.DataFrame, reps: dict,
                      n_show: int = 10, seed: int = 0) -> None:
    """Top-``n_show`` activating images per principal and one random row for contrast.
    Images that carry the row's attribute (both, for the random row) get a coloured
    frame."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    imgs = np.load(IMAGES, mmap_mode="r")
    rows = []
    for attr, r in reps.items():
        j = r["principal"]
        idx = np.argsort(-Z[:, j], kind="stable")[:n_show]
        p = r["top100_principal"]
        rows.append((idx, {attr: labels[attr].values},
                     f"{ATTR_LABEL[attr]}: top-{n_show} images of coordinate {j}   "
                     f"(top-{TOP_M} purity {p['purity']:.2f}; "
                     f"{100 * r['prevalence']:.1f}% of all images)"))
    rnd = np.random.default_rng(seed).choice(len(labels), n_show, replace=False)
    rows.append((rnd, {a: labels[a].values for a in reps},
                 f"{n_show} random images (contrast)"))
    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.02, len(rows) * 1.28))
    for (idx, marks, title), row in zip(rows, axes):
        for ax, i in zip(row, idx):
            ax.imshow(np.asarray(imgs[i]))
            ax.set_xticks([]); ax.set_yticks([])
            hits = [a for a, v in marks.items() if v[i]]
            for s in ax.spines.values():
                s.set_visible(False)
            if hits:                      # frame(s): outer = first attribute
                for k, a in enumerate(hits):
                    ax.add_patch(Rectangle((1.5 + 5 * k, 1.5 + 5 * k), 124 - 10 * k, 124 - 10 * k,
                                           fill=False, lw=2.6, ec=ATTR_COLOR[a]))
        row[0].set_title(title, loc="left", fontsize=10.5, pad=3)
    handles = [Rectangle((0, 0), 1, 1, fill=False, lw=2.2, ec=ATTR_COLOR[a],
                         label=f"labelled {ATTR_LABEL[a]}") for a in reps]
    fig.subplots_adjust(wspace=0.04, hspace=0.3, bottom=0.08)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.06))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"pa_top_images.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / 'pa_top_images.pdf'}")


# ── check: the appendix check ──────────────────────────────────────────────────

def cmd_check(argv: list[str]) -> None:
    """Supervised check of Principal Alignment on the main dictionary (Appendix C):
    alignment spectrum (pa_spectrum), top-activating images of the principals
    (pa_top_images), screening-off probe (pa_screening.{md,tex}, pa_screening_splits.csv)
    and pa_summary.json, in --out-dir (default
    results/celeba/figures_v2/principal_alignment/).  CPU, a few minutes."""
    global OUT
    ap = argparse.ArgumentParser(prog="alignment.py check", description=cmd_check.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-splits", type=int, default=20)
    ap.add_argument("--figures-only", action="store_true",
                    help="skip the screening-off probe (spectrum + images only)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args(argv)
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    gt = json.loads(GT.read_text())
    principals = {gt["w1_attr"]: gt["w1_neurons"][0], gt["w2_attr"]: gt["w2_neurons"][0]}
    Z = np.ascontiguousarray(np.load(CODES, mmap_mode="r"), dtype=np.float32)
    labels = pd.read_parquet(LABELS)
    assert len(labels) == len(Z) == len(np.load(IMAGES, mmap_mode="r"))
    print(f"codes {Z.shape}, principals {principals}", flush=True)

    reps, aucs = {}, {}
    for attr, j in principals.items():
        w = labels[attr].values.astype(np.float64)
        reps[attr], aucs[attr] = spectrum(Z, w, j)
        r = reps[attr]
        print(f"[{attr}] j={j} AUC {r['auc_principal']:.3f} (rank {r['rank_principal_auc']}), "
              f"runner-up {r['runner_up_auc']} {r['auc_runner_up']:.3f}, gap {r['auc_gap']:.3f}; "
              f"top-100 purity {r['top100_principal']['purity']:.2f} "
              f"(lift {r['top100_principal']['lift']:.1f}), runner-up by purity "
              f"{r['runner_up_purity']} {r['top100_runner_up_purity']['purity']:.2f}", flush=True)
    figure_spectrum(aucs, reps)
    figure_top_images(Z, labels, reps)
    summary = {"codes": str(CODES.relative_to(ROOT)), "principals": principals,
               "spectrum": reps}

    if not args.figures_only:
        rows = []
        for attr, j in principals.items():
            w = labels[attr].values.astype(np.float64)
            for s in range(args.n_splits):
                rows += [{"attr": attr, **r} for r in probe_path(Z, w, j, seed=s)]
                if s == 0:                # reproduces the original probe exactly
                    ref = ci_probe(Z, w, j, n_extra=200, seed=0)
                    mine = rows[-1]
                    assert abs(ref["auc_principal+rest"] - mine["auc_augmented"]) < 1e-9, (ref, mine)
                    assert abs(ref["auc_principal"] - mine["auc_principal"]) < 1e-9
                print(f"  {attr} split {s}: " + ", ".join(
                    f"K={r['K']} dAUC={r['delta_auc']:+.4f}" for r in rows[-len(KS):]), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "pa_screening_splits.csv", index=False)
        s = summarise_probe(df)
        write_tables(s)
        summary["screening"] = s.to_dict(orient="records")
        summary["screening_settings"] = {"Ks": list(KS), "n_splits": args.n_splits,
                                         "probe": "principal_alignment.ci_probe (10 bins, C=1)"}
    elif (OUT / "pa_summary.json").exists():
        old = json.loads((OUT / "pa_summary.json").read_text())
        summary.update({k: v for k, v in old.items() if k.startswith("screening")})
    (OUT / "pa_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT / 'pa_summary.json'}")


# ── f1: attribute-to-coordinate F1 matrix ─────────────────────────────────────

def cmd_f1(argv: list[str]) -> None:
    """Attribute-to-coordinate F1 matrix of CelebA dictionaries.

    For every CelebA attribute, computes the best-threshold F1 of each coordinate
    (experiment.compute_f1_scores, the ground-truth rule of the paper) and saves the full
    (n_attrs, m) F1 matrix to <out-dir>/f1_<dict>.npz (f1, attrs); existing files are
    skipped.  `third-modifier` reads f1_sae_precode_k20.npz.  (Written for an earlier
    interleaved-backward experiment, hence the default output path.)  NJOBS (default 8)
    sets the number of joblib workers.
    """
    from joblib import Parallel, delayed
    from apps.celeba.experiment import compute_f1_scores

    ap = argparse.ArgumentParser(prog="alignment.py f1", description=cmd_f1.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dicts", nargs="+",
                    help="sae_k20, sae_k5, sae_precode_k20, sae_precode_k5, ... "
                         "(data/celeba/embeddings/<dict>.npy)")
    ap.add_argument("--out-dir", type=Path, default=F1_DIR)
    args = ap.parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    attrs = [c for c in labels.columns if c != "celeb_id"]
    nj = int(os.environ.get("NJOBS", "8"))

    for d in args.dicts:
        out = out_dir / f"f1_{d}.npz"
        if out.exists():
            print("exists", out); continue
        F = np.load(ROOT / "data/celeba/embeddings" / f"{d}.npy", mmap_mode="r")
        f1 = Parallel(n_jobs=nj)(delayed(compute_f1_scores)(
            np.asarray(F), labels[a].values.astype(float)) for a in attrs)
        np.savez(out, f1=np.stack(f1), attrs=np.array(attrs))
        print("wrote", out, flush=True)


# ── third-modifier: candidates for the third direct modifier at r = 3 ─────────

N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
PRINCIPALS = {"Wearing_Hat": 5348, "Eyeglasses": 5537}


def cmd_third_modifier(argv: list[str]) -> None:
    """Rank CelebA attributes as the third direct modifier of the r = 3 DGP.

    For every attribute, on the main k = 20 SigLIP dictionary (19,867-image pool):

      j*        best-threshold-F1 coordinate on the pre-activations (the ground-truth rule)
      F1, gap   its F1 and the gap to the runner-up coordinate
      F1 base   F1 of predicting every image positive, 2p/(1+p): best-threshold F1 is
                inflated for high-prevalence labels, so F1 is read against this floor
      AUC       of the pre-activation z_pre[:, j*], and of the sparse code z[:, j*] that
                NEXIS actually regresses on (ties at 0 when the code does not fire)
      code P/R  precision and recall of "code fires" (z[:, j*] > 0) for the label
      lift      share of positives among the 100 images with the largest z_pre[:, j*],
                over the prevalence (its maximum is 1/p)
      distinct  j* differs from the hat and glasses principals 5348 and 5537
      n_max     the independent sampler (scm.generate_celeba_rct) draws the three
                attributes independently at CelebA prevalence and takes images without
                replacement from the matching joint cell; n_max is the largest n of the
                sweep grid for which all 50 seeds find every cell populated with
                probability >= 0.99 (binomial cell counts); "(1,1,1)" is the number of
                images with hat, glasses and the attribute.

    Paper item: the r = 3 paragraph of Appendix C, "Sideburns [...] its principal
    coordinate separates it with AUC 0.805 on Z" = column "AUC code" of the Sideburns
    row.  The comparison values "0.969 and 0.956 for the other two" (Wearing_Hat 5348,
    Eyeglasses 5537, excluded here) are auc_principal of `check` (pa_summary.json,
    "spectrum"; --figures-only suffices).

    Reads the F1 matrix of `f1 sae_precode_k20` and writes --out (default
    results/celeba/figures_v2/third_modifier_candidates.md).
    """
    ap = argparse.ArgumentParser(prog="alignment.py third-modifier",
                                 description=cmd_third_modifier.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f1", type=Path, default=F1_DIR / "f1_sae_precode_k20.npz")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/celeba/figures_v2/third_modifier_candidates.md")
    args = ap.parse_args(argv)
    f1z = np.load(args.f1 if args.f1.is_absolute() else ROOT / args.f1,
                  allow_pickle=True)
    F1, attrs = f1z["f1"], [str(a) for a in f1z["attrs"]]
    L = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    P = np.load(ROOT / "data/celeba/embeddings/sae_precode_k20.npy", mmap_mode="r")
    S = np.load(ROOT / "data/celeba/embeddings/sae_k20.npy", mmap_mode="r")
    hat, gl = L.Wearing_Hat.values.astype(int), L.Eyeglasses.values.astype(int)
    ph, pg = hat.mean(), gl.mean()

    rows = []
    for i, a in enumerate(attrs):
        if a in PRINCIPALS:
            continue
        y = L[a].values.astype(int)
        p = y.mean()
        order = np.argsort(F1[i])[::-1]
        j = int(order[0])
        zp, zc = np.asarray(P[:, j]), np.asarray(S[:, j])
        fires = zc > 0
        top = np.argsort(zp)[::-1][:100]
        p_ok = []
        for n in N_GRID:
            ok = 1.0
            for h in (0, 1):
                for g in (0, 1):
                    for v in (0, 1):
                        cnt = int(((hat == h) & (gl == g) & (y == v)).sum())
                        pc = (ph if h else 1 - ph) * (pg if g else 1 - pg) * (p if v else 1 - p)
                        ok *= binom.cdf(cnt, n, pc)
            p_ok.append(ok ** 50)
        feas = [n for n, q in zip(N_GRID, p_ok) if q >= 0.99]
        rows.append({
            "attribute": a, "prevalence": p, "j*": j, "F1": F1[i, j],
            "gap": F1[i, j] - F1[i, order[1]], "F1 base": 2 * p / (1 + p),
            "AUC z_pre": roc_auc_score(y, zp), "AUC code": roc_auc_score(y, zc),
            "code P": y[fires].mean() if fires.any() else np.nan, "code R": fires[y == 1].mean(),
            "lift": y[top].mean() / p, "max lift": 1 / p,
            "distinct": j not in PRINCIPALS.values(),
            "(1,1,1)": int(((hat == 1) & (gl == 1) & (y == 1)).sum()),
            "n_max": max(feas) if feas else 0,
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "NO"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    cols = list(df.columns)
    # The column reference names the script under its earlier name, as in the
    # published third_modifier_candidates.md (now `alignment.py third-modifier`).
    md = ["# Third direct modifier for the r = 3 DGP: alignment and feasibility", "",
          "Main k=20 SigLIP dictionary, 19,867-image pool; Wearing_Hat (5348) and "
          "Eyeglasses (5537) excluded. Columns: see src/apps/celeba/third_modifier_candidates.py. "
          "n_max = largest sweep n (grid 50..10,000) at which the independent sampler "
          "populates every (hat, glasses, attribute) cell for all 50 seeds with "
          "probability >= 0.99; 0 = not even n = 50.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Saved → {out}")


COMMANDS = {"check": cmd_check, "f1": cmd_f1, "third-modifier": cmd_third_modifier}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
