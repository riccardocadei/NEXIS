"""
Feature-image comparison plot for NEMS-selected effect modifiers.

Layout (one figure per model):
  ┌──────────────────────────────────────────────────────────────────┐
  │  #1  SAE_2677  [SAE]  p=7.1e-08  │  GATE hi=+0.035  Δ=+0.033*  │  "Water Bodies" [med]  │
  ├─ HIGH ──────────────────────────────────────────────────────────┤
  │  [img] [img] [img] [img] [img] [img] [img] [img]               │
  ├─ LOW ───────────────────────────────────────────────────────────┤
  │  [img] [img] [img] [img] [img] [img] [img] [img]               │
  ├──────────────────────────────────────────────────────────────────┤
  │  #2  lang_6  [W]  p=2.0e-03  │  CATE(0)=+0.030  CATE(1)=-0.003  Δ=-0.033*       │
  ├──────────────────────────────────────────────────────────────────┤
  │  [bar chart: CATE(0) vs CATE(1) with 95% CI]                   │
  └──────────────────────────────────────────────────────────────────┘
  ⚠ Low-activation neurons are flagged and shown with reduced opacity.

Usage
-----
    python scripts/plot_features.py [--embed-model dinov2] [--sae-dim 3072] [--k 8]
    python scripts/plot_features.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import NullLocator

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"
IMG_DIR  = DATA_DIR / "Uganda2000_processed"
sys.path.insert(0, str(ROOT / "src"))

from uganda import load_image, geo_label


# ── Palette ───────────────────────────────────────────────────────────────────

C = dict(
    bg          = "#FAFAF8",
    hdr_sae     = "#1B2631",   # dark navy  — interpretable SAE
    hdr_noint   = "#626567",   # mid-gray   — non-interpretable SAE
    hdr_w       = "#4A235A",   # dark purple — W covariate
    top_accent  = "#922B21",   # deep red   — high activation row
    bot_accent  = "#1B4F72",   # deep blue  — low activation row
    w_accent    = "#7D3C98",   # purple     — W info row
    img_top     = "#E74C3C",   # border high
    img_bot     = "#2980B9",   # border low
    sig_star    = "#F39C12",   # gold star  — significant Δ
    hdr_text    = "white",
    subhdr_text = "white",
    img_title   = "#2C3E50",
    gap_bg      = "#FAFAF8",
)

MIN_ACTIVATION = 0.01   # below this: "not interpretable"
IMG_THUMB_PX   = 160    # thumbnail resolution


# ── Image loading ──────────────────────────────────────────────────────────────

def _thumb(key: int) -> "np.ndarray | None":
    from PIL import Image as PILImage
    arr = load_image(int(key), IMG_DIR)
    if arr is None:
        return None
    img = PILImage.fromarray((arr * 255).astype("uint8"))
    img = img.resize((IMG_THUMB_PX, IMG_THUMB_PX), PILImage.BICUBIC)
    return np.asarray(img) / 255.0


# ── Row-structure builder ──────────────────────────────────────────────────────

def _build_row_plan(selected, site_feats, site_keys, k,
                    gate_map, interp_map, n_sae):
    """Return a list of row descriptors + the gridspec height_ratios."""
    H_HDR   = 0.55   # header relative height
    H_IMG   = 2.20   # image row relative height
    H_INFO  = 1.60   # W covariate info row
    H_GAP   = 0.18   # gap between feature blocks

    rows   = []   # each: dict with 'kind' and payload
    ratios = []

    for entry in selected:
        feat_idx = entry["idx"]
        label    = entry["label"]
        group    = entry.get("group", "SAE")
        pval     = entry["pvalue"]
        gate     = gate_map.get(label, {})
        vlm_lbl  = interp_map.get(feat_idx, "")

        if group == "SAE":
            acts = site_feats[:, feat_idx]
            max_act = float(acts.max())
            interpretable = max_act >= MIN_ACTIVATION

            sorted_idxs = np.argsort(acts)
            top_idxs    = sorted_idxs[::-1][:k]
            bot_idxs    = sorted_idxs[:k]
            top_keys    = site_keys[top_idxs].tolist()
            bot_keys    = site_keys[bot_idxs].tolist()
            top_acts    = acts[top_idxs].tolist()
            bot_acts    = acts[bot_idxs].tolist()

            rows.append(dict(kind="header", entry=entry, group="SAE",
                             interpretable=interpretable,
                             gate=gate, vlm_lbl=vlm_lbl,
                             max_act=max_act))
            ratios.append(H_HDR)

            for row_kind, keys, act_list in [
                ("top_imgs", top_keys, top_acts),
                ("bot_imgs", bot_keys, bot_acts),
            ]:
                rows.append(dict(kind=row_kind, feat_idx=feat_idx,
                                 keys=keys, acts=act_list,
                                 interpretable=interpretable))
                ratios.append(H_IMG)

        else:   # W covariate
            rows.append(dict(kind="header", entry=entry, group="W",
                             interpretable=True,
                             gate=gate, vlm_lbl="", max_act=None))
            ratios.append(H_HDR)

            rows.append(dict(kind="w_info", entry=entry, gate=gate))
            ratios.append(H_INFO)

        # gap after each feature block
        rows.append(dict(kind="gap"))
        ratios.append(H_GAP)

    return rows, ratios


# ── Header axes renderer ───────────────────────────────────────────────────────

def _render_header(ax, entry, group, interpretable, gate, vlm_lbl, max_act, rank):
    if group == "W":
        bg = C["hdr_w"]
    elif not interpretable:
        bg = C["hdr_noint"]
    else:
        bg = C["hdr_sae"]

    ax.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    label  = entry["label"]
    pval   = entry["pvalue"]
    diff   = gate.get("diff", float("nan"))
    gate_lo = gate.get("gate_lo", float("nan"))
    gate_hi = gate.get("gate_hi", float("nan"))
    se_lo   = gate.get("se_lo", float("nan"))
    se_hi   = gate.get("se_hi", float("nan"))

    # Significance star for Δ
    if not (np.isnan(diff) or np.isnan(se_lo) or np.isnan(se_hi)):
        se_diff = np.sqrt(se_hi**2 + se_lo**2)
        lo_d = diff - 1.96 * se_diff
        hi_d = diff + 1.96 * se_diff
        sig = " ★" if lo_d * hi_d > 0 else ""
    else:
        sig = ""

    # ── Rank badge (left) ────────────────────────────────────────────────────
    badge_col = C["sig_star"] if (sig and interpretable) else "#888888"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.003, 0.08), 0.042, 0.84,
        boxstyle="round,pad=0.01", fc=badge_col, ec="none", zorder=3,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.024, 0.50, f"#{rank}", transform=ax.transAxes,
            color="white", fontsize=9, fontweight="bold",
            ha="center", va="center", zorder=4)

    # ── Feature label + group tag ────────────────────────────────────────────
    group_tag = f"[{group}]"
    tag_col   = {"SAE": "#5DADE2", "W": "#C39BD3"}.get(group, "#AAAAAA")
    ax.text(0.057, 0.55, label, transform=ax.transAxes,
            color="white", fontsize=11, fontweight="bold", va="center")
    ax.text(0.057, 0.18, group_tag, transform=ax.transAxes,
            color=tag_col, fontsize=8, fontweight="bold", va="center")

    # ── p-value ──────────────────────────────────────────────────────────────
    ax.text(0.19, 0.50, f"p = {pval:.2e}", transform=ax.transAxes,
            color="#D5DBDB", fontsize=9, va="center")

    # ── GATE / CATE numbers ──────────────────────────────────────────────────
    if not np.isnan(gate_lo) and not np.isnan(gate_hi):
        ftype = gate.get("ftype", "")
        if ftype == "binary":
            lbl_lo, lbl_hi = "CATE(0)", "CATE(1)"
        elif ftype == "sparse":
            lbl_lo, lbl_hi = "GATE(inactive)", "GATE(active)"
        else:
            lbl_lo, lbl_hi = "GATE(low)", "GATE(high)"

        col_lo = "#AED6F1"; col_hi = "#F1948A"
        ax.text(0.310, 0.50,
                f"{lbl_lo} = {gate_lo:+.4f}   {lbl_hi} = {gate_hi:+.4f}",
                transform=ax.transAxes,
                color="#D5DBDB", fontsize=9, va="center")

        diff_col = C["sig_star"] if sig else "#D5DBDB"
        ax.text(0.62, 0.50,
                f"Δ = {diff:+.4f}{sig}",
                transform=ax.transAxes,
                color=diff_col, fontsize=9.5, fontweight="bold", va="center")

    # ── Max-activation badge (SAE only) ─────────────────────────────────────
    if group == "SAE" and max_act is not None:
        act_str  = f"max act = {max_act:.4f}"
        act_col  = "#F0B27A" if interpretable else "#AAB7B8"
        ax.text(0.76, 0.50, act_str, transform=ax.transAxes,
                color=act_col, fontsize=8.5, va="center",
                style="italic")

    # ── Non-interpretable warning ────────────────────────────────────────────
    if group == "SAE" and not interpretable:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.855, 0.10), 0.14, 0.80,
            boxstyle="round,pad=0.02",
            fc="#E74C3C", ec="none", zorder=3,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(0.925, 0.50, "⚠ NOT INTERPRETABLE",
                transform=ax.transAxes,
                color="white", fontsize=7.5, fontweight="bold",
                ha="center", va="center", zorder=4)
        return   # skip VLM label for non-interpretable

    # ── VLM interpretation label ─────────────────────────────────────────────
    if vlm_lbl and group == "SAE":
        ax.text(0.855, 0.55, f'"{vlm_lbl}"',
                transform=ax.transAxes,
                color="#F9E79F", fontsize=9, fontweight="bold",
                va="center", ha="left",
                style="italic")
        conf = gate.get("confidence", "")
        if conf:
            ax.text(0.855, 0.18, f"[{conf}]",
                    transform=ax.transAxes,
                    color="#AAB7B8", fontsize=7.5, va="center", ha="left")


# ── Row-label stripe (HIGH / LOW) ─────────────────────────────────────────────

def _row_label_ax(ax, label, color):
    """Render a narrow vertical stripe with the row label."""
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.50, 0.50, label,
            transform=ax.transAxes,
            color="white", fontsize=8, fontweight="bold",
            ha="center", va="center", rotation=90)


# ── Image axes renderer ────────────────────────────────────────────────────────

def _render_image(ax, key, act, border_color, df_rct, interpretable=True, faded=False):
    img = _thumb(int(key))
    alpha = 0.40 if faded else 1.0

    if img is not None:
        ax.imshow(img, interpolation="bilinear", aspect="equal", alpha=alpha)
        if faded:
            ax.set_facecolor("#888888")
    else:
        ax.set_facecolor("#1a1a1a")
        ax.text(0.5, 0.5, "N/A", color="white", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)

    # Geo label (key + short location)
    loc_row = df_rct[df_rct["geo_long_lat_key"] == int(key)]
    loc_str = ""
    if len(loc_row) and "PNAME_VALUE" in df_rct.columns:
        named = loc_row.dropna(subset=["PNAME_VALUE"])
        if len(named):
            loc_str = named.iloc[0]["PNAME_VALUE"][:16]
    title = f"{act:.4f}\n{loc_str or f'key {int(key)}'}"

    ax.set_title(title, fontsize=5.8, pad=2,
                 color=border_color, fontweight="bold", linespacing=1.3)

    # Colored border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.0 if not faded else 0.8)
    ax.set_xticks([]); ax.set_yticks([])

    # "low activation" stamp
    if faded:
        ax.text(0.5, 0.5, "low\nactivation",
                transform=ax.transAxes,
                color="white", fontsize=6.5, fontweight="bold",
                ha="center", va="center", alpha=0.75,
                bbox=dict(fc="#555555", ec="none", alpha=0.6, pad=2))


# ── W covariate info row ───────────────────────────────────────────────────────

def _render_w_info(ax, entry, gate):
    """Show a simple CATE bar chart for a W covariate."""
    ax.set_facecolor("#F4ECF7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7D3C98")
    ax.spines["bottom"].set_color("#7D3C98")

    gate_lo = gate.get("gate_lo", float("nan"))
    gate_hi = gate.get("gate_hi", float("nan"))
    se_lo   = gate.get("se_lo",   float("nan"))
    se_hi   = gate.get("se_hi",   float("nan"))
    lbl_lo  = gate.get("lbl_lo",  "0")
    lbl_hi  = gate.get("lbl_hi",  "1")
    diff    = gate.get("diff",    float("nan"))

    if np.isnan(gate_lo) or np.isnan(gate_hi):
        ax.text(0.5, 0.5, "No GATE data available",
                transform=ax.transAxes, ha="center", va="center",
                color="#7D3C98", fontsize=11)
        return

    cates = [gate_lo, gate_hi]
    errs  = [1.96 * se_lo, 1.96 * se_hi]
    xlbls = [f"= 0\n{lbl_lo[:18]}", f"= 1\n{lbl_hi[:18]}"]
    colors_bar = ["#1F618D", "#922B21"]

    bars = ax.bar([0, 1], cates, color=colors_bar, alpha=0.80,
                  width=0.40, edgecolor="white", linewidth=1.2,
                  yerr=errs, capsize=7,
                  error_kw=dict(ecolor="#333333", lw=1.5, capthick=1.5))
    ax.axhline(0, color="#555555", lw=0.8, ls="--")

    # Annotate values
    for xi, (cate, err) in enumerate(zip(cates, errs)):
        ax.text(xi, cate + err + 0.002,
                f"{cate:+.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=colors_bar[xi])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"{entry['label']} = 0\n{lbl_lo[:20]}", f"{entry['label']} = 1\n{lbl_hi[:20]}"],
        fontsize=8.5)
    ax.set_ylabel("CATE (log hrs)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y", alpha=0.3)

    # Δ annotation
    se_diff = np.sqrt(se_lo**2 + se_hi**2)
    lo_d = diff - 1.96 * se_diff; hi_d = diff + 1.96 * se_diff
    sig = " ★" if lo_d * hi_d > 0 else ""
    ax.set_title(f"Δ CATE = {diff:+.4f}{sig}   95% CI [{lo_d:+.4f}, {hi_d:+.4f}]",
                 fontsize=9, color="#4A235A", pad=5, fontweight="bold")


# ── Main plot function ─────────────────────────────────────────────────────────

def plot_model_features(embed_model, sae_dim, k, df_rct):
    out_dir   = ROOT / "results" / "uganda" / f"{embed_model}_{sae_dim}"
    nems_path = out_dir / "nems_result.json"
    if not nems_path.exists():
        print(f"  Skipping {embed_model}_{sae_dim}: nems_result.json not found")
        return

    with open(nems_path) as f:
        nems_out = json.load(f)

    # Load supporting data
    gate_map = {}
    if (out_dir / "summary.json").exists():
        with open(out_dir / "summary.json") as f:
            for em in json.load(f).get("effect_modifiers", []):
                gate_map[em["label"]] = em

    interp_map = {}
    if (out_dir / "interpretations.json").exists():
        with open(out_dir / "interpretations.json") as f:
            for entry in json.load(f):
                interp_map[entry["feature"]] = entry.get("label", "")

    site_data  = np.load(out_dir / "site_features.npz")
    site_feats = site_data["site_features"]
    site_keys  = site_data["site_keys"]
    n_sae      = nems_out["feature_meta"]["n_sae_features"]

    all_selected = nems_out["nems"]["selected"]

    # ── Build row plan ────────────────────────────────────────────────────────
    rows, ratios = _build_row_plan(
        all_selected, site_feats, site_keys, k,
        gate_map, interp_map, n_sae,
    )
    n_rows = len(rows)

    # ── Figure & GridSpec ─────────────────────────────────────────────────────
    # Columns: 1 narrow label col + k image cols
    N_LABEL_COLS = 1
    N_COLS = N_LABEL_COLS + k

    IMG_W_INCH   = 1.85
    LABEL_W_INCH = 0.45
    L_MARG       = 0.15
    R_MARG       = 0.15
    fig_w = LABEL_W_INCH + k * IMG_W_INCH + L_MARG + R_MARG

    # Normalise height ratios to inches (using 2.20 per image row as baseline)
    scale = IMG_W_INCH   # 1 image unit ≈ 1 image width
    total_h = sum(r * scale for r in ratios) + 1.0  # 1 inch title
    fig_h = max(8.0, total_h)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C["bg"])
    fig.suptitle(
        f"NEMS effect modifiers — {embed_model}  (SAE dim = {sae_dim})\n"
        f"★ = Δ CATE significant (95% CI excludes 0)  ·  "
        f"⚠ NOT INTERPRETABLE = max activation < {MIN_ACTIVATION}",
        fontsize=11, fontweight="bold",
        y=0.995, va="top",
    )

    gs = gridspec.GridSpec(
        n_rows, N_COLS,
        figure=fig,
        height_ratios=ratios,
        width_ratios=[LABEL_W_INCH] + [IMG_W_INCH] * k,
        hspace=0.0, wspace=0.025,
        left=L_MARG / fig_w,
        right=1.0 - R_MARG / fig_w,
        top=0.965,
        bottom=0.01,
    )

    # ── Render rows ───────────────────────────────────────────────────────────
    rank = 0
    for ri, row in enumerate(rows):
        kind = row["kind"]

        if kind == "gap":
            ax = fig.add_subplot(gs[ri, :])
            ax.set_facecolor(C["gap_bg"]); ax.axis("off")
            continue

        if kind == "header":
            rank += 1
            ax = fig.add_subplot(gs[ri, :])   # span all columns
            _render_header(
                ax,
                entry       = row["entry"],
                group       = row["group"],
                interpretable = row["interpretable"],
                gate        = row["gate"],
                vlm_lbl     = row["vlm_lbl"],
                max_act     = row["max_act"],
                rank        = rank,
            )
            continue

        if kind == "w_info":
            # label column
            ax_lbl = fig.add_subplot(gs[ri, 0])
            _row_label_ax(ax_lbl, "W COV", C["w_accent"])
            # info panel spans all image columns
            ax_info = fig.add_subplot(gs[ri, 1:])
            _render_w_info(ax_info, row["entry"], row["gate"])
            continue

        # SAE image rows (top or bottom)
        is_top       = kind == "top_imgs"
        accent       = C["top_accent"]  if is_top else C["bot_accent"]
        border_col   = C["img_top"]     if is_top else C["img_bot"]
        row_label    = "HIGH ▲"         if is_top else "LOW  ▼"
        faded        = not row["interpretable"]

        # Label column
        ax_lbl = fig.add_subplot(gs[ri, 0])
        _row_label_ax(ax_lbl, row_label, accent)

        # Image columns
        keys = row["keys"]; acts = row["acts"]
        for ci in range(k):
            ax = fig.add_subplot(gs[ri, ci + N_LABEL_COLS])
            if ci < len(keys):
                _render_image(
                    ax, keys[ci], acts[ci],
                    border_color  = border_col,
                    df_rct        = df_rct,
                    interpretable = not faded,
                    faded         = faded,
                )
            else:
                ax.set_facecolor(C["bg"]); ax.axis("off")

    out_path = out_dir / "feature_images.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot top/bottom activation images for NEMS-selected SAE features.")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--sae-dim",     type=int, default=3072)
    p.add_argument("--k",           type=int, default=8,
                   help="Images per group (top / bottom).")
    p.add_argument("--all",         action="store_true",
                   help="Plot all result subdirectories.")
    return p.parse_args()


def main():
    args   = parse_args()
    df_rct = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)

    if args.all:
        dirs = sorted((ROOT / "results" / "uganda").glob("*/nems_result.json"))
        pairs = []
        for d in dirs:
            name   = d.parent.name
            parts  = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                pairs.append((parts[0], int(parts[1])))
    else:
        if args.embed_model is None:
            print("Provide --embed-model or use --all"); sys.exit(1)
        pairs = [(args.embed_model, args.sae_dim)]

    for model, dim in pairs:
        print(f"Plotting {model}_{dim} …")
        plot_model_features(model, dim, args.k, df_rct)


if __name__ == "__main__":
    main()
