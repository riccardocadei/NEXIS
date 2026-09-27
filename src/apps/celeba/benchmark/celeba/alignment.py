"""Supervised check of Principal Alignment on the main dictionary (Appendix C).

On CelebA the direct modifiers are observed attributes, so Principal Alignment,
W^k independent of Z^{[m] minus j_k} given Z^{j_k}, can be checked on the codes Z of the
main dictionary (k = 20):

  alignment spectrum   sign-free AUC, max(AUC, 1 - AUC), of every coordinate as a score
                       for the attribute; the principal coordinate against the runner-up,
                       and the purity of its 100 most activating images   (pa_spectrum.pdf)
  top images           the 10 most activating images of each principal coordinate and 10
                       random images for contrast                         (pa_top_images.pdf)
  screening-off probe  held-out AUC of logistic probes of W^k on Z^{j_k} alone and on
                       Z^{j_k} plus the K in {10, 50, 200} other coordinates most associated
                       with W^k given Z^{j_k} (selected on the training half), over 20
                       random 50/50 splits, with a held-out likelihood-ratio test of the K
                       extra coordinates                                  (pa_screening.tex)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

import config as C
from celeba.experiment import load_ground_truth
from celeba.figures import APPENDIX_RC

ATTRS = C.DGP_MAIN["attrs"]                  # the two modifiers of the main setting
ATTR_LABEL = {"Wearing_Hat": "Wearing Hat", "Eyeglasses": "Eyeglasses"}
ATTR_COLOR = {"Wearing_Hat": "#E69F00", "Eyeglasses": "#009E73"}
GREY = "#9a9a9a"
KS = (10, 50, 200)
N_SPLITS = 20
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
    """Share of the M most activating images that carry the label, and its lift."""
    top = np.argsort(-z, kind="stable")[:M]
    pur = float(w[top].mean())
    return {"purity": pur, "lift": pur / float(w.mean()), "n_active": int((z > 0).sum())}


def spectrum(Z: np.ndarray, w: np.ndarray, j: int):
    auc_raw = column_auc(Z, w)
    auc = np.maximum(auc_raw, 1 - auc_raw)
    others = np.arange(Z.shape[1]) != j
    runner = int(np.flatnonzero(others)[np.argmax(auc[others])])
    rep = {"principal": int(j), "prevalence": float(w.mean()),
           "auc_principal": float(auc[j]), "runner_up_auc": runner,
           "auc_runner_up": float(auc[runner]), "auc_gap": float(auc[j] - auc[runner]),
           "rank_principal_auc": int((auc > auc[j]).sum() + 1),
           "n_auc_ge_0.6": int((auc[others] >= 0.6).sum()),
           "top100_principal": top_purity(Z[:, j], w)}
    return rep, auc


# ── screening-off probe ───────────────────────────────────────────────────────

def _residualize(M: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Residuals of the columns of M on [1, A]."""
    A1 = np.column_stack([np.ones(len(A)), A])
    coef, *_ = np.linalg.lstsq(A1, M, rcond=None)
    return M - A1 @ coef


def _basis(zj: np.ndarray, tr: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Expansion of the principal: linear term + indicators at the training-half deciles
    of its positive values."""
    pos = zj[tr][zj[tr] > 0]
    qs = np.quantile(pos, np.linspace(0, 1, n_bins + 1)[1:-1]) if len(pos) > n_bins else np.array([])
    edges = np.unique(np.concatenate([[0.0], qs]))
    return np.column_stack([zj] + [(zj > t).astype(np.float64) for t in edges])


def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def probe_path(Z: np.ndarray, w: np.ndarray, j: int, Ks=KS, seed: int = 0) -> list:
    """One random 50/50 split.  Companions of the principal are ranked on the training
    half by |partial correlation| with W given the basis of Z^j; L2-logistic probes
    (C = 1) are fitted on the training half and scored on the held-out half.  The LRT
    compares unpenalised fits of W ~ basis(Z^j) and W ~ basis(Z^j) + K companions on the
    held-out half: 2 dloglik ~ chi2_K under screening-off."""
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
                     "logloss_augmented": ll1, "info_share": (H - ll0) / (H - ll1),
                     "lrt_stat": lr, "lrt_logp": float(chi2.logsf(max(lr, 0.0), K))})
    return rows


def summarise_probe(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (attr, K), d in df.groupby(["attr", "K"], sort=False):
        row = {"attr": attr, "K": int(K), "n_splits": len(d)}
        for c in ("auc_principal", "auc_augmented", "delta_auc", "info_share"):
            row[c] = float(d[c].mean())
            row[c + "_lo"], row[c + "_hi"] = (float(np.quantile(d[c], 0.025)),
                                              float(np.quantile(d[c], 0.975)))
        row["lrt_log10p_max"] = float(d.lrt_logp.max() / np.log(10))
        out.append(row)
    return pd.DataFrame(out)


def screening_table(s: pd.DataFrame) -> list:
    tex = [r"\begin{tabular}{lrcccc}", r"\toprule",
           r"Modifier & $K$ & AUC $Z^{j_k}$ & AUC $Z^{j_k}$ + $K$ others & $\Delta$AUC & "
           r"info.\ share \\", r"\midrule"]
    for i, r in s.iterrows():
        first = i == 0 or s.attr[i - 1] != r.attr
        tex.append((rf"\multirow{{3}}{{*}}{{{ATTR_LABEL[r.attr]}}}" if first else "")
                   + f" & {r.K} & "
                   + (rf"\multirow{{3}}{{*}}{{{r.auc_principal:.3f}}}" if first else "")
                   + f" & {r.auc_augmented:.3f} & "
                   f"{r.delta_auc:+.3f} [{r.delta_auc_lo:+.3f}, {r.delta_auc_hi:+.3f}] & "
                   f"{r.info_share:.2f} \\\\")
        if not first and (i + 1 == len(s) or s.attr[i + 1] != r.attr):
            tex.append(r"\midrule" if i + 1 < len(s) else r"\bottomrule")
    tex.append(r"\end{tabular}")
    return tex


# ── figures ───────────────────────────────────────────────────────────────────

def figure_spectrum(aucs: dict, reps: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    with plt.rc_context(APPENDIX_RC):
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
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
    print(f"  figure -> {out}")


def figure_top_images(Z: np.ndarray, labels: pd.DataFrame, reps: dict, images: np.ndarray,
                      out: Path, n_show: int = 10, seed: int = 0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
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
    rows.append((rnd, {a: labels[a].values for a in reps}, f"{n_show} random images (contrast)"))
    with plt.rc_context(APPENDIX_RC):
        fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.02, len(rows) * 1.28))
        for (idx, marks, title), row in zip(rows, axes):
            for ax, i in zip(row, idx):
                ax.imshow(np.asarray(images[i]))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                hits = [a for a, v in marks.items() if v[i]]
                for k, a in enumerate(hits):       # one frame per attribute carried
                    ax.add_patch(Rectangle((1.5 + 5 * k, 1.5 + 5 * k), 124 - 10 * k,
                                           124 - 10 * k, fill=False, lw=2.6, ec=ATTR_COLOR[a]))
            row[0].set_title(title, loc="left", fontsize=10.5, pad=3)
        handles = [Rectangle((0, 0), 1, 1, fill=False, lw=2.2, ec=ATTR_COLOR[a],
                             label=f"labelled {ATTR_LABEL[a]}") for a in reps]
        fig.subplots_adjust(wspace=0.04, hspace=0.3, bottom=0.08)
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, 0.06))
        fig.savefig(out, bbox_inches="tight", dpi=200)
        plt.close(fig)
    print(f"  figure -> {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def run(out_dir: Path, overwrite: bool = False, n_splits: int = N_SPLITS) -> None:
    """Writes pa_spectrum.pdf, pa_top_images.pdf, pa_screening.tex and pa_summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pa_summary.json"
    if summary_path.exists() and not overwrite:
        print(f"  skip (exists): {summary_path}")
        return
    coords = load_ground_truth(20, False)
    Z = np.ascontiguousarray(np.load(C.codes_path(20, False, "z"), mmap_mode="r"),
                             dtype=np.float32)
    labels = pd.read_parquet(C.LABELS)

    reps, aucs = {}, {}
    for attr in ATTRS:
        reps[attr], aucs[attr] = spectrum(Z, labels[attr].values.astype(np.float64), coords[attr])
    figure_spectrum(aucs, reps, out_dir / "pa_spectrum.pdf")
    if C.IMAGES.exists():
        figure_top_images(Z, labels, reps, np.load(C.IMAGES, mmap_mode="r"),
                          out_dir / "pa_top_images.pdf")
    else:
        print(f"  skip pa_top_images.pdf: {C.IMAGES} missing (run `prepare --steps embed`)")
    # principal-coordinate AUC of every attribute used by an experiment (e.g. the r = 3 one)
    auc_all = {}
    for attr in C.GT_ATTRS:
        a = column_auc(Z[:, [coords[attr]]], labels[attr].values.astype(np.float64))[0]
        auc_all[attr] = {"coordinate": coords[attr], "auc": float(max(a, 1 - a))}

    rows = []
    for attr in ATTRS:
        w = labels[attr].values.astype(np.float64)
        for s in range(n_splits):
            rows += [{"attr": attr, **r} for r in probe_path(Z, w, coords[attr], seed=s)]
        print(f"  screening-off probe done: {attr}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pa_screening_splits.csv", index=False)
    s = summarise_probe(df)
    (out_dir / "pa_screening.tex").write_text("\n".join(screening_table(s)) + "\n")
    summary_path.write_text(json.dumps({"principals": {a: coords[a] for a in ATTRS},
                                        "spectrum": reps, "principal_auc_all": auc_all,
                                        "screening": s.to_dict(orient="records")}, indent=1))
    print(f"  tables -> {out_dir / 'pa_screening.tex'}, {summary_path}")
