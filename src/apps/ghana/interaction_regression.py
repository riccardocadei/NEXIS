"""
Ghana LEAP 1000 — Interaction regression for NEXIS-selected effect modifiers.

Fits a first-differences OLS model with heterogeneous treatment effects:

    dY_i = α + τ·T_i
           + β₁·T_i·z51_c  + γ₁·z51_c      (neuron 1777 — Vegetation Density)
           + β₂·T_i·z122_c + γ₂·z122_c     (neuron 3821)
           + β₃·T_i·farms_c + γ₃·farms_c   (Farming household)
           + β₄·T_i·formal_c + γ₄·formal_c (Head in formal sector)
           + W_i'δ + ε_i

Modifiers are centred (mean zero) so τ = ATE at the mean of all modifiers.
Cluster-robust SEs clustered at community level (G=162).

Outputs
-------
  results/ghana/codes/nexis_no_adj/interaction_regression.json
  results/ghana/codes/nexis_no_adj/interaction_regression.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.apps.ghana.data import load_data, W_ALL, W_LABELS

DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
OUT_DIR  = ROOT / "results" / "ghana" / "codes" / "nexis_no_adj"

MODIFIERS = {
    "z_51":       {"neuron_idx": 1777, "label": "Vegetation Density (neuron 1777)"},
    "z_122":      {"neuron_idx": 3821, "label": "Neuron 3821"},
    "farms":      {"neuron_idx": None, "label": "Farming household"},
    "head_formal":{"neuron_idx": None, "label": "Head in formal sector"},
}

TREAT_COLOR = "#e07b39"
CTRL_COLOR  = "#5b8db8"


# ── Data preparation ──────────────────────────────────────────────────────────

def _compute_sae_activations(embeddings: np.ndarray, ckpt: dict,
                              wh_mean: np.ndarray, wh_std: np.ndarray) -> np.ndarray:
    """Top-k sparse SAE codes (k=25)."""
    x = torch.from_numpy(((embeddings - wh_mean) / wh_std)).float()
    with torch.no_grad():
        acts = F.relu((x - ckpt["b_dec"]) @ ckpt["W_enc.weight"].T + ckpt["W_enc.bias"])
        topk_vals, _ = torch.topk(acts, 25, dim=-1)
        acts = acts * (acts >= topk_vals[:, -1:])
    return acts.numpy()


def load_regression_data() -> pd.DataFrame:
    """
    Returns a household-level DataFrame with dY, T, W covariates,
    and the two SAE neuron activations (community-level, mapped to households).
    """
    df_full = load_data(DATA_DIR)
    hh_both = df_full.groupby("hhid")["wave"].nunique()
    df = df_full[df_full["hhid"].isin(hh_both[hh_both == 2].index)].copy()
    df0 = df[df["wave"] == 0]
    df1 = df[df["wave"] == 1]

    merged = (
        df0.set_index("hhid")[["T", "comm"] + W_ALL + ["Y"]]
           .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
    )
    merged["dY"] = merged["Y1"] - merged["Y"]
    merged = merged.reset_index()

    # Impute rooms NaN with column mean (307 missing; rooms is a control, not a modifier)
    merged["rooms"] = merged["rooms"].fillna(merged["rooms"].mean())

    # Load SAE and compute community-level activations
    ckpt    = torch.load(SAT_DIR / "sae_model.pt", map_location="cpu", weights_only=False)
    wh_mean = np.load(SAT_DIR / "sae_whiten_mean.npy")
    wh_std  = np.load(SAT_DIR / "sae_whiten_std.npy")

    leap_embs = np.load(SAT_DIR / "prithvi_embeddings.npy")
    leap_ids  = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    leap_acts = _compute_sae_activations(leap_embs, ckpt, wh_mean, wh_std)

    # Live neurons (≥5 non-zero activations)
    live_mask = (leap_acts > 0).sum(axis=0) >= 5
    live_idx  = np.where(live_mask)[0]          # global neuron indices
    leap_live = leap_acts[:, live_mask]          # (n_comm, n_live)

    # filtered_idx 51 → global neuron 1777,  filtered_idx 122 → global neuron 3821
    col_51  = 51
    col_122 = 122

    comm_to_row = {int(cid): i for i, cid in enumerate(leap_ids)}
    merged["z_51"]  = merged["comm"].map(lambda c: float(leap_live[comm_to_row[c], col_51]))
    merged["z_122"] = merged["comm"].map(lambda c: float(leap_live[comm_to_row[c], col_122]))

    # Spectral indices (community-level) — mean columns only
    sp = pd.read_csv(SAT_DIR / "spectral_indices.csv").rename(columns={"comm_id": "comm"})
    spectral_cols = [c for c in sp.columns if c.endswith("_mean")]
    merged = merged.merge(sp[["comm"] + spectral_cols], on="comm", how="left")

    return merged, spectral_cols


# ── Cluster-robust OLS ────────────────────────────────────────────────────────

def cluster_ols(y: np.ndarray, X: np.ndarray,
                groups: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    """
    OLS with cluster-robust sandwich SEs.
    Returns DataFrame with Coef, SE, t-stat, p-value, 95% CI bounds.
    """
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta  = XtXi @ X.T @ y
    resid = y - X @ beta

    unique_g = np.unique(groups)
    G = len(unique_g)
    meat = np.zeros((k, k))
    for g in unique_g:
        mask = groups == g
        sg   = X[mask].T @ resid[mask]
        meat += np.outer(sg, sg)
    meat *= (G / (G - 1)) * ((n - 1) / (n - k))
    V   = XtXi @ meat @ XtXi
    var = np.diag(V).clip(0)   # clip tiny numerical negatives before sqrt
    se  = np.sqrt(var)

    # t-distribution with G-1 degrees of freedom (conservative, standard for CRVE)
    from scipy.stats import t as tdist
    tstat = beta / se
    pval  = 2 * tdist.sf(np.abs(tstat), df=G - 1)
    ci_lo = beta - tdist.ppf(0.975, df=G - 1) * se
    ci_hi = beta + tdist.ppf(0.975, df=G - 1) * se

    return pd.DataFrame({
        "coef":    beta,
        "se":      se,
        "t_stat":  tstat,
        "p_value": pval,
        "ci_lo":   ci_lo,
        "ci_hi":   ci_hi,
    }, index=col_names)


# ── Fit one specification ─────────────────────────────────────────────────────

def fit_spec(df_c: pd.DataFrame, mod_keys: list[str],
             ctrl_cols: list[str], label: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Build design matrix, run cluster OLS, return (results_df, groups, n_valid)."""
    n = len(df_c)
    cols = (["intercept", "T"]
            + [f"{k}_c" for k in mod_keys]
            + [f"T_x_{k}" for k in mod_keys]
            + ctrl_cols)
    X = np.column_stack([
        np.ones((n, 1)),
        df_c["T"].values[:, None],
        np.column_stack([df_c[f"{k}_c"] for k in mod_keys]),
        np.column_stack([df_c["T"] * df_c[f"{k}_c"] for k in mod_keys]),
        df_c[ctrl_cols].values,
    ]).astype(float)
    y = df_c["dY"].values.astype(float)

    valid  = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y   = X[valid], y[valid]
    groups = df_c.loc[valid, "comm"].values
    print(f"  [{label}] valid observations: {valid.sum()}")
    return cluster_ols(y, X, groups, cols), groups, int(valid.sum())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    df, _ = load_regression_data()
    n  = len(df)
    print(f"  n = {n} households, {df['comm'].nunique()} communities")

    mod_keys   = ["z_51", "z_122", "farms", "head_formal"]
    mod_labels = [MODIFIERS[k]["label"] for k in mod_keys]

    # Centre modifiers (sample mean → 0, so T coef = ATE at the mean)
    df_c = df.copy()
    for k in mod_keys:
        df_c[f"{k}_c"] = df_c[k] - df_c[k].mean()

    # Survey modifiers already in model as centred main effects — exclude from W_ctrl
    _mod_survey = {"farms", "head_formal"}
    W_ctrl = [c for c in W_ALL if c not in _mod_survey]

    # ── Specification 1: W_ALL controls only (matches NEXIS design) ───────────
    # ── Specification 2: + Y₀ (ANCOVA — absorbs baseline wealth level) ────────
    # NOTE: spectral indices are derived from the same imagery as the SAE neurons
    # and would create near-perfect multicollinearity with T×neuron terms.
    specs = [
        ("W_ALL",       W_ctrl,           "baseline (NEXIS controls)"),
        ("W_ALL + Y₀",  W_ctrl + ["Y"],   "ANCOVA: adds Y₀ (baseline consumption)"),
    ]

    print("\nFitting both specifications ...")
    results = {}
    for spec_name, ctrl_cols, desc in specs:
        res, groups, n_valid = fit_spec(df_c, mod_keys, ctrl_cols, spec_name)
        results[spec_name] = {"res": res, "groups": groups,
                              "n": n_valid, "desc": desc, "ctrl": ctrl_cols}

    # ── Print side-by-side comparison ─────────────────────────────────────────

    print("\n" + "=" * 80)
    print("INTERACTION REGRESSION — Ghana LEAP 1000 (nexis_no_adj, codes)")
    print("Cluster-robust SEs, t(G-1=161),  * p<0.05  ** p<0.01")
    print("=" * 80)

    for spec_name, d in results.items():
        res     = d["res"]
        n_valid = d["n"]
        ate_row = res.loc["T"]
        print(f"\n── Spec: {spec_name}  ({d['desc']})  n={n_valid} ──")
        print(f"  ATE  τ = {ate_row['coef']:>8.3f}  SE={ate_row['se']:.3f}"
              f"  p={ate_row['p_value']:.4f}"
              f"  95% CI [{ate_row['ci_lo']:.3f}, {ate_row['ci_hi']:.3f}]")
        print(f"  {'Modifier':<45} {'β':>8}  {'SE':>7}  {'p':>7}  {'β/SD':>8}")
        print("  " + "-" * 80)
        for k, lbl in zip(mod_keys, mod_labels):
            row  = res.loc[f"T_x_{k}"]
            sd   = float(df_c[k].std())
            sig  = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "  ")
            print(f"  {lbl:<45} {row['coef']:>8.3f}  {row['se']:>7.3f}"
                  f"  {row['p_value']:>7.4f}  {row['coef']*sd:>8.3f}  {sig}")

    # ── Primary result: W_ALL spec; secondary: ANCOVA ─────────────────────────
    primary = results["W_ALL"]
    primary_res = primary["res"]
    ate_row = primary_res.loc["T"]

    # ── Save JSON ─────────────────────────────────────────────────────────────

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _spec_payload(spec_name, d):
        res = d["res"]
        ate = res.loc["T"]
        return {
            "spec":    spec_name,
            "desc":    d["desc"],
            "n":       d["n"],
            "G":       int(np.unique(d["groups"]).size),
            "se_type": "cluster-robust (CRVE), df=G-1=161",
            "ate": {k: float(v) for k, v in ate.items()},
            "interactions": [
                {
                    "key":   k, "label": lbl,
                    "neuron_idx": MODIFIERS[k]["neuron_idx"],
                    "modifier_mean": float(df_c[k].mean()),
                    "modifier_sd":   float(df_c[k].std()),
                    **{col: float(res.loc[f"T_x_{k}", col])
                       for col in ["coef", "se", "t_stat", "p_value", "ci_lo", "ci_hi"]},
                    "beta_per_sd": float(res.loc[f"T_x_{k}", "coef"] * df_c[k].std()),
                }
                for k, lbl in zip(mod_keys, mod_labels)
            ],
        }

    payload = {
        "model": "OLS first-differences with interaction terms",
        "note":  ("Spec W_ALL matches NEXIS design and is the primary result. "
                  "Spec W_ALL+Y0 is ANCOVA — valid for SAE neurons but binary modifier "
                  "estimates are unstable due to near-constant support (farms=96%)."),
        "specs": [_spec_payload(sn, d) for sn, d in results.items()],
    }

    json_path = OUT_DIR / "interaction_regression.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {json_path}")

    # ── Plot: two-panel forest plot (primary spec only) ───────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.patch.set_facecolor("white")

    # Left panel: ATE
    ax = axes[0]
    ax.set_facecolor("white")
    coef, lo, hi = ate_row["coef"], ate_row["ci_lo"], ate_row["ci_hi"]
    ax.barh([0], [hi - lo], left=[lo], height=0.4, color=TREAT_COLOR, alpha=0.25)
    ax.scatter([coef], [0], color=TREAT_COLOR, s=80, zorder=3)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(["ATE"], fontsize=11)
    ax.set_xlabel("GH₵/month", fontsize=10)
    ax.set_title("Average Treatment Effect\n(at mean of modifiers)", fontsize=10)
    ax.text(coef, 0.28, f"{coef:.1f}\np={ate_row['p_value']:.3f}",
            ha="center", va="bottom", fontsize=9, color=TREAT_COLOR, fontweight="bold")

    # Right panel: interactions
    ax = axes[1]
    ax.set_facecolor("white")
    y_pos = list(range(len(mod_keys)))[::-1]

    for k, lbl, yp in zip(mod_keys, mod_labels, y_pos):
        row = primary_res.loc[f"T_x_{k}"]
        col = TREAT_COLOR if row["coef"] > 0 else CTRL_COLOR
        ax.barh([yp], [row["ci_hi"] - row["ci_lo"]], left=[row["ci_lo"]],
                height=0.45, color=col, alpha=0.2)
        ax.scatter([row["coef"]], [yp], color=col, s=70, zorder=3)
        sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "")
        ax.text(row["ci_hi"] + 0.5, yp,
                f"{row['coef']:.1f}  (p={row['p_value']:.3f}){sig}",
                va="center", fontsize=9)

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_labels, fontsize=9)
    ax.set_xlabel("Δ treatment effect per unit of centred modifier  (GH₵/month)", fontsize=10)
    ax.set_title("Effect-Modifier Interactions  (T × modifier_centred)", fontsize=10)

    pos_patch = mpatches.Patch(color=TREAT_COLOR, alpha=0.5, label="positive")
    neg_patch = mpatches.Patch(color=CTRL_COLOR,  alpha=0.5, label="negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    fig.suptitle(
        "Ghana LEAP 1000 — NEXIS-selected effect modifiers (nexis_no_adj, SAE codes)\n"
        f"n={primary['n']} hh, G=162 clusters, W_ALL controls,  * p<0.05  ** p<0.01",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    png_path = OUT_DIR / "interaction_regression.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {png_path}")
    plt.close(fig)

    # ── Interpretation summary ────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("INTERPRETATION (primary spec: W_ALL)")
    print("=" * 80)
    print(f"\nATE at mean of all modifiers: {ate_row['coef']:.2f} GH₵/month"
          f"  [{ate_row['ci_lo']:.2f}, {ate_row['ci_hi']:.2f}]  p={ate_row['p_value']:.4f}\n")

    interp_notes = {
        "z_51": (
            "Neuron 1777 detects dense continuous vegetation cover. "
            "Communities with denser vegetation benefit more from the transfer, "
            "consistent with greener areas having more productive agricultural land "
            "and thus higher returns to cash (e.g., input purchase)."
        ),
        "z_122": (
            "Neuron 3821 has a similar visual signature to 1777 (VLM could not "
            "distinguish them), yet its coefficient is significant and sizable. "
            "It likely captures a finer sub-dimension of vegetation structure "
            "that the VLM cannot verbalize but the SAE separates statistically."
        ),
        "farms": (
            "Farming household (96% of sample). The positive β reflects that the "
            "rare non-farming households (~93 hh) benefit substantially less — "
            "plausibly because farming provides the channel (inputs, livestock) "
            "through which cash translates to consumption gains. "
            "Interpret with caution: small minority, wide CI."
        ),
        "head_formal": (
            "Head in formal sector (9% of sample). The negative β (not significant) "
            "is consistent with diminishing marginal returns to income: wage-employed "
            "heads already have more stable income, so the transfer's marginal "
            "consumption impact is smaller. Evidence is weak (p=0.26)."
        ),
    }
    for k, lbl in zip(mod_keys, mod_labels):
        row  = primary_res.loc[f"T_x_{k}"]
        sd   = float(df_c[k].std())
        sig  = ("p<0.01" if row["p_value"] < 0.01
                else ("p<0.05" if row["p_value"] < 0.05
                      else f"p={row['p_value']:.3f}, n.s."))
        direction = "more" if row["coef"] > 0 else "less"
        beta = row["coef"]
        print(f"  [{sig}] {lbl}")
        print(f"    β={beta:.3f}  β/SD={beta*sd:.2f} GH₵/month"
              f"  →  a 1-SD increase in this modifier makes the treatment effect"
              f" {direction} by {abs(beta*sd):.1f} GH₵/month.")
        print(f"    {interp_notes[k]}")
        print()


if __name__ == "__main__":
    main()
