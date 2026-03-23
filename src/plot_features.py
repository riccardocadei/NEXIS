"""
Feature-image comparison plot for NEMS-selected effect modifiers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"
IMG_DIR  = DATA_DIR / "Uganda2000_processed"
sys.path.insert(0, str(ROOT / "src"))

from uganda import load_image, resolve_outcome, load_basemap, draw_base, w_display


# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    bg           = "#FFFFFF",
    box_valid    = "#52BE80",  hdr_valid  = "#F0FBF4",  stripe_valid = "#27AE60",
    box_ns       = "#F4D03F",  hdr_ns     = "#FEFDF0",  stripe_ns    = "#D4AC0D",
    box_noint    = "#999999",  hdr_noint  = "#EEEEEE",  stripe_noint = "#777777",
    bar_lo       = "#AED6F1",  bar_lo_edge = "#2471A3",
    bar_hi       = "#FAD7A0",  bar_hi_edge = "#CA6F1E",
    zdist_ctrl   = "#AED6F1",
    zdist_trt    = "#F1948A",
    row_hi       = "#922B21",
    row_lo       = "#1A5276",
    img_hi       = "#E74C3C",
    img_lo       = "#2980B9",
    map_all      = "#CCCCCC",   # all trial sites
    map_act      = "#90EE90",   # activated sites (pastel green)
    map_act_edge = "#3CB371",
)

MIN_ACTIVATION = 0.01
IMG_THUMB_PX   = 140
N_IMG          = 6   # example images per row (+ 1 mini-map column)

H_HDR   = 0.62   # header (tall enough for 2-line stats)
H_CHART = 1.48   # GATE/CATE + distribution charts
H_TG    = 0.18   # tick-label gap
H_ILBL  = 0.12   # image-group label banner (reduced)
H_IMG   = 1.35   # image row
H_GAP   = 0.65   # gap between feature boxes



# ── Helpers ────────────────────────────────────────────────────────────────────

def _thumb(key):
    from PIL import Image as PILImage
    arr = load_image(int(key), IMG_DIR)
    if arr is None:
        return None
    img = PILImage.fromarray((arr * 255).astype("uint8"))
    img = img.resize((IMG_THUMB_PX, IMG_THUMB_PX), PILImage.BICUBIC)
    return np.asarray(img) / 255.0


def _is_sig(gate):
    d, sl, sh = (gate.get(k, float("nan")) for k in ("diff", "se_lo", "se_hi"))
    if any(np.isnan(v) for v in (d, sl, sh)):
        return False
    se_d = np.sqrt(sl**2 + sh**2)
    return (d - 1.96*se_d) * (d + 1.96*se_d) > 0


def _box_style(interpretable, significant):
    if not interpretable:
        return C["box_noint"], C["hdr_noint"], C["stripe_noint"]
    if significant:
        return C["box_valid"], C["hdr_valid"], C["stripe_valid"]
    return C["box_ns"], C["hdr_ns"], C["stripe_ns"]


def _ci_str(gate):
    d, sl, sh = (gate.get(k, float("nan")) for k in ("diff", "se_lo", "se_hi"))
    if any(np.isnan(v) for v in (d, sl, sh)):
        return ""
    se_d = np.sqrt(sl**2 + sh**2)
    lo, hi = d - 1.96*se_d, d + 1.96*se_d
    return f"Δ = {d:+.3f}   [{lo:+.3f}, {hi:+.3f}]"


def _pval_str(gate):
    import math
    d, sl, sh = (gate.get(k, float("nan")) for k in ("diff", "se_lo", "se_hi"))
    if any(np.isnan(v) for v in (d, sl, sh)):
        return ""
    se_d = np.sqrt(sl**2 + sh**2)
    if se_d < 1e-12:
        return "p < 0.0001"
    z = abs(d) / se_d
    pval = math.erfc(z / math.sqrt(2))   # two-sided p-value
    if pval < 0.001:
        return "p < 0.001"
    return f"p = {pval:.3f}"


# ── Row plan ───────────────────────────────────────────────────────────────────

def _build_row_plan(selected, site_feats, site_keys, k, gate_map):
    rows, ratios, spans = [], [], []
    ri = 0
    for entry in selected:
        feat_idx = entry["idx"]
        label    = entry["label"]
        group    = entry.get("group", "SAE")
        gate     = gate_map.get(label, {})
        blk      = ri

        if group == "SAE":
            acts          = site_feats[:, feat_idx]
            max_act       = float(acts.max())
            significant   = _is_sig(gate)
            interp_str    = gate.get("interp", "")
            interpretable = not ("low activation" in (interp_str or "").lower())
            order         = np.argsort(acts)
            top_keys = site_keys[order[::-1][:k]].tolist()
            bot_keys = site_keys[order[:k]].tolist()
            top_acts = acts[order[::-1][:k]].tolist()
            bot_acts = acts[order[:k]].tolist()

            rows.append(dict(kind="header", entry=entry, group="SAE",
                             interpretable=interpretable, significant=significant,
                             gate=gate, vlm_lbl=interp_str, max_act=max_act))
            ratios.append(H_HDR); ri += 1

            rows.append(dict(kind="gate_chart", gate=gate, group="SAE",
                             feat_idx=feat_idx, w_label=None,
                             interpretable=interpretable, significant=significant))
            ratios.append(H_CHART); ri += 1

            rows.append(dict(kind="tick_gap")); ratios.append(H_TG); ri += 1

            for rk, ks, al, is_top in [
                    ("top_imgs", top_keys, top_acts, True),
                    ("bot_imgs", bot_keys, bot_acts, False)]:
                rows.append(dict(kind="img_label", is_top=is_top,
                                 vlm_lbl=interp_str))
                ratios.append(H_ILBL); ri += 1
                rows.append(dict(kind=rk, keys=ks, acts=al,
                                 interpretable=interpretable, feat_idx=feat_idx))
                ratios.append(H_IMG); ri += 1

        else:  # W covariate
            sig_w = _is_sig(gate)
            rows.append(dict(kind="header", entry=entry, group="W",
                             interpretable=True, significant=sig_w,
                             gate=gate, vlm_lbl="", max_act=None))
            ratios.append(H_HDR); ri += 1

            rows.append(dict(kind="gate_chart", gate=gate, group="W",
                             feat_idx=None, w_label=label,
                             interpretable=True, significant=sig_w))
            ratios.append(H_CHART); ri += 1

            rows.append(dict(kind="tick_gap")); ratios.append(H_TG); ri += 1

        spans.append((blk, ri - 1))
        rows.append(dict(kind="feat_gap")); ratios.append(H_GAP); ri += 1

    return rows, ratios, spans


# ── Header subtitle helper ─────────────────────────────────────────────────────

def _subtitle_str(gate, group, interpretable, vlm_lbl, label):
    """One-line summary sentence for the box subtitle (includes comparison group)."""
    diff = gate.get("diff", float("nan"))
    sig  = _is_sig(gate)
    if group == "SAE":
        if not interpretable:
            return "Low neuron activation — feature could not be interpreted"
        name = vlm_lbl or label
        comparison = "active vs inactive sites"
        if sig:
            direction = "higher" if diff > 0 else "lower"
            return (f'Active sites (vs inactive) show {direction} CATE '
                    f'(Δ={diff:+.3f}) — feature: "{name}"')
        return f'No significant CATE difference ({comparison}) — feature: "{name}"'
    else:
        disp, tick_lo, tick_hi = w_display(label)
        comparison = f"{tick_hi} vs {tick_lo}"
        if sig:
            direction = "higher" if diff > 0 else "lower"
            return (f'{tick_hi} (vs {tick_lo}) show {direction} CATE '
                    f'(Δ={diff:+.3f})')
        return f'No significant CATE difference ({comparison})'


# ── Header ─────────────────────────────────────────────────────────────────────

def _render_header(ax, entry, group, interpretable, significant,
                   gate, vlm_lbl, max_act, rank, header_ax_w_in):
    box_col, hdr_bg, stripe_col = _box_style(interpretable, significant)
    ax.set_facecolor(hdr_bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    label = entry["label"]
    if group == "SAE":
        if not interpretable:
            name  = f"not interpretable (neuron {entry['idx']})"
            style = "italic"; txt_col = "#888888"
        elif vlm_lbl:
            name  = vlm_lbl
            style = "normal"; txt_col = "#17202A"
        else:
            name  = f"SAE {label}  —  no VLM interpretation"
            style = "italic"; txt_col = "#888888"
    else:
        name, _, _ = w_display(label)
        style = "normal"; txt_col = "#17202A"

    title = f"Effect modifier {rank}:  {name}"
    ax.text(0.010, 0.72, title,
            transform=ax.transAxes,
            color=txt_col, fontsize=12, fontweight="bold",
            style=style, va="center", ha="left", zorder=3)

    subtitle = _subtitle_str(gate, group, interpretable, vlm_lbl,
                              entry["label"])
    ax.text(0.010, 0.48, subtitle,
            transform=ax.transAxes,
            color="#555555", fontsize=7.5, fontweight="normal",
            style="italic", va="center", ha="left", zorder=3, clip_on=True)

    stat_col = C["box_valid"] if _is_sig(gate) else "#AAAAAA"
    delta = _ci_str(gate)
    pval  = _pval_str(gate)
    if delta:
        ax.text(0.985, 0.64, delta, transform=ax.transAxes,
                color=stat_col, fontsize=9, fontweight="bold",
                va="center", ha="right", clip_on=True)
    if pval:
        ax.text(0.985, 0.46, pval, transform=ax.transAxes,
                color=stat_col, fontsize=8.5, fontweight="bold",
                va="center", ha="right", clip_on=True)


# ── GATE / CATE bar chart ──────────────────────────────────────────────────────

def _render_gate_bars(ax, gate, group, ate_est, bg_color="white"):
    gate_lo = gate.get("gate_lo", float("nan"))
    gate_hi = gate.get("gate_hi", float("nan"))
    se_lo   = gate.get("se_lo",   float("nan"))
    se_hi   = gate.get("se_hi",   float("nan"))
    n_lo    = gate.get("n_lo", None)
    n_hi    = gate.get("n_hi", None)

    if np.isnan(gate_lo) or np.isnan(gate_hi):
        ax.axis("off"); return

    if group == "SAE":
        lo_lbl = f"inactive\n(n={n_lo:,})" if n_lo else "inactive"
        hi_lbl = f"active\n(n={n_hi:,})"   if n_hi else "active"
    else:
        w_label       = gate.get("label", "")
        _, tick_lo, tick_hi = w_display(w_label)
        lo_lbl = f"{tick_lo}\n(n={n_lo:,})" if n_lo else tick_lo
        hi_lbl = f"{tick_hi}\n(n={n_hi:,})" if n_hi else tick_hi

    ax.set_facecolor(bg_color)
    for s in ("top", "right"):   ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#DDDDDD")

    errs = [1.96*se_lo, 1.96*se_hi]
    ax.bar([0, 1], [gate_lo, gate_hi],
           color=[C["bar_lo"], C["bar_hi"]],
           edgecolor=[C["bar_lo_edge"], C["bar_hi_edge"]],
           linewidth=1.2, width=0.44, alpha=0.90, zorder=2,
           yerr=errs, capsize=7,
           error_kw=dict(ecolor="#555", lw=1.4, capthick=1.4, zorder=3))
    ax.axhline(0, color="#CCCCCC", lw=0.8, zorder=0)

    if not np.isnan(ate_est):
        ax.axhline(ate_est, color="#999999", lw=1.0, ls=":", zorder=1)
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        mid = (gate_lo + gate_hi) / 2
        ax.text(0.98, ate_est,
                f" ATE={ate_est:+.3f}",
                transform=trans, ha="right",
                va="bottom" if ate_est >= mid else "top",
                fontsize=7, color="#999999", style="italic")

    for xi, (val, err, col) in enumerate(
            zip([gate_lo, gate_hi], errs,
                [C["bar_lo_edge"], C["bar_hi_edge"]])):
        pad = err + max(abs(val)*0.06, 0.001)
        y   = val + pad if val >= 0 else val - pad
        ax.text(xi, y, f"{val:+.4f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold", color=col)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([lo_lbl, hi_lbl], fontsize=9, linespacing=1.35)
    ax.tick_params(axis="y", left=False, labelleft=True,
                   labelcolor="black", labelsize=7.5, pad=2)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.72, 1.72)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7, color="#CCCCCC", zorder=0)
    ax.set_title("GATE" if group == "SAE" else "CATE",
                 fontsize=9.5, fontweight="bold", color="#333333",
                 loc="left", pad=3)


# ── W-covariate distribution (W on x, ctrl/trt colours) ───────────────────────

def _render_w_balance(ax, w_label, Z_col, T_ind, bg_color="white"):
    """Treatment balance plot with W on the x-axis and ctrl/trt as colours."""
    if Z_col is None or T_ind is None:
        ax.axis("off"); return

    _, tick_lo, tick_hi = w_display(w_label)
    Z_ctrl = Z_col[T_ind == 0]; Z_trt = Z_col[T_ind == 1]
    vals = Z_col[np.isfinite(Z_col)]
    is_binary = set(np.unique(vals)).issubset({0.0, 1.0})

    ax.set_facecolor(bg_color)
    for s in ("top", "right"):   ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#DDDDDD")

    if is_binary:
        # Grouped bars: x = W value, bars = ctrl / trt
        p_ctrl = [(Z_ctrl == 0).mean(), (Z_ctrl == 1).mean()]
        p_trt  = [(Z_trt  == 0).mean(), (Z_trt  == 1).mean()]
        w = 0.38
        xs = np.array([0, 1])
        b_c = ax.bar(xs - w/2, p_ctrl, width=w,
                     color=C["zdist_ctrl"], edgecolor=C["bar_lo_edge"],
                     linewidth=0.8, alpha=0.88, label="Control", zorder=2)
        b_t = ax.bar(xs + w/2, p_trt,  width=w,
                     color=C["zdist_trt"],  edgecolor=C["bar_hi_edge"],
                     linewidth=0.8, alpha=0.88, label="Treated", zorder=2)
        for xi, (vc, vt) in enumerate(zip(p_ctrl, p_trt)):
            ax.text(xi - w/2, vc + 0.01, f"{vc:.0%}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C["bar_lo_edge"])
            ax.text(xi + w/2, vt + 0.01, f"{vt:.0%}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C["bar_hi_edge"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([tick_lo, tick_hi], fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(p_ctrl), max(p_trt)) * 1.45))
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right",
                  bbox_to_anchor=(0.98, 0.98), ncol=2)
        ax.axhline(0, color="#CCCCCC", lw=0.6, zorder=0)
    else:
        bins = np.linspace(vals.min(), np.percentile(vals, 95), 25)
        ax.hist(Z_ctrl[np.isfinite(Z_ctrl)], bins=bins, alpha=0.65,
                color=C["zdist_ctrl"], label="Control",
                density=True, edgecolor="none", zorder=2)
        ax.hist(Z_trt[np.isfinite(Z_trt)],  bins=bins, alpha=0.65,
                color=C["zdist_trt"],  label="Treated",
                density=True, edgecolor="none", zorder=2)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  bbox_to_anchor=(0.98, 0.95))
        disp, _, _ = w_display(w_label)
        ax.set_xlabel(disp, fontsize=8, color="#555555", labelpad=3)

    ax.grid(axis="y", alpha=0.22, linewidth=0.7, color="#CCCCCC", zorder=0)
    ax.set_title("Treatment balance", fontsize=9.5, fontweight="bold",
                 color="#333333", loc="left", pad=3)


# ── Distribution plot ──────────────────────────────────────────────────────────

def _render_z_dist(ax, Z_col, T_ind, is_binary=False, bg_color="white"):
    if Z_col is None or T_ind is None:
        ax.axis("off"); return

    ax.set_facecolor(bg_color)
    for s in ("top", "right"):   ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#DDDDDD")

    Z_ctrl = Z_col[T_ind == 0]; Z_trt = Z_col[T_ind == 1]

    if is_binary:
        p_ctrl = (Z_ctrl == 1).mean(); p_trt = (Z_trt == 1).mean()
        ax.bar([0, 1], [p_ctrl, p_trt],
               color=[C["zdist_ctrl"], C["zdist_trt"]],
               alpha=0.85, width=0.44,
               edgecolor=[C["bar_lo_edge"], C["bar_hi_edge"]], linewidth=1.0)
        for xi, (v, col) in enumerate(
                zip([p_ctrl, p_trt], [C["bar_lo_edge"], C["bar_hi_edge"]])):
            ax.text(xi, v + 0.01, f"{v:.1%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=col)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Treated"], fontsize=9)
        ax.set_ylim(0, min(1.0, max(p_ctrl, p_trt) * 1.35))
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title("Treatment balance", fontsize=9.5, fontweight="bold",
                     color="#333333", loc="left", pad=3)
    else:
        Z_c_pos = Z_ctrl[Z_ctrl > 0]; Z_t_pos = Z_trt[Z_trt > 0]
        if len(Z_c_pos) < 5 and len(Z_t_pos) < 5:
            ax.text(0.5, 0.5, "all inactive", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8,
                    color="#AAAAAA", style="italic")
            ax.axis("off"); return

        all_pos = np.concatenate([Z_c_pos, Z_t_pos])
        q95  = np.percentile(all_pos, 95)
        bins = np.linspace(0, max(q95, 1e-6), 28)

        ax.hist(Z_c_pos, bins=bins, alpha=0.65, color=C["zdist_ctrl"],
                label="Control", density=True, edgecolor="none", zorder=2)
        ax.hist(Z_t_pos, bins=bins, alpha=0.65, color=C["zdist_trt"],
                label="Treated", density=True, edgecolor="none", zorder=2)

        f_c = (Z_ctrl == 0).mean(); f_t = (Z_trt == 0).mean()
        ax.text(0.98, 0.97,
                f"inactive — ctrl: {f_c:.0%}  trt: {f_t:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color="#666666")

        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  bbox_to_anchor=(0.98, 0.95))
        ax.set_xlabel("neural activation", fontsize=8, color="#555555", labelpad=3)
        ax.set_title("Treatment balance", fontsize=9.5, fontweight="bold",
                     color="#333333", loc="left", pad=3)

    ax.grid(axis="y", alpha=0.22, linewidth=0.7, color="#CCCCCC", zorder=0)


# ── Uganda site map ─────────────────────────────────────────────────────────────

def _render_site_map(ax, feat_idx, site_feats, site_keys, df_rct,
                     uganda_gdf, neighbors, lakes_c, map_xlim=None):
    acts           = site_feats[:, feat_idx]
    activated_mask = acts > 0        # any neuron fire

    coord_cols = ['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']
    sites_df   = (df_rct[coord_cols].dropna()
                  .drop_duplicates('geo_long_lat_key'))
    key_to_lon = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_long_center']))
    key_to_lat = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_lat_center']))

    all_lons = [key_to_lon[k] for k in site_keys if k in key_to_lon]
    all_lats = [key_to_lat[k] for k in site_keys if k in key_to_lat]
    act_lons = [key_to_lon[k] for k, a in zip(site_keys, activated_mask)
                if a and k in key_to_lon]
    act_lats = [key_to_lat[k] for k, a in zip(site_keys, activated_mask)
                if a and k in key_to_lat]

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    if map_xlim is not None:
        ax.set_xlim(*map_xlim)
        ax.set_ylim(-1.4, 4.6)   # re-pin after xlim change (aspect='equal' may shift it)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])

    ax.scatter(all_lons, all_lats, s=12, color=C["map_all"],
               zorder=4, linewidths=0, alpha=0.9)
    if act_lons:
        ax.scatter(act_lons, act_lats, s=22, color=C["map_act"],
                   zorder=5, edgecolors=C["map_act_edge"], linewidths=0.6, alpha=0.95)

    n_act = int(activated_mask.sum())
    ax.text(0.02, 0.03, f"{n_act} / {len(site_keys)} active",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color="#444444",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_title("Geographical distribution", fontsize=9.5, fontweight="bold",
                 color="#333333", loc="left", pad=3)


# ── Mini map for image example rows ────────────────────────────────────────────

def _render_mini_map_examples(ax, example_keys, df_rct,
                               uganda_gdf, neighbors, lakes_c,
                               site_keys_all, highlight_col, map_xlim=None):
    """7th slot in top/bot image rows: Uganda map highlighting the k example locations."""
    coord_cols = ['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']
    sites_df   = df_rct[coord_cols].dropna().drop_duplicates('geo_long_lat_key')
    key_to_lon = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_long_center']))
    key_to_lat = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_lat_center']))

    all_lons = [key_to_lon[k] for k in site_keys_all if k in key_to_lon]
    all_lats = [key_to_lat[k] for k in site_keys_all if k in key_to_lat]
    ex_lons  = [key_to_lon[int(k)] for k in example_keys if int(k) in key_to_lon]
    ex_lats  = [key_to_lat[int(k)] for k in example_keys if int(k) in key_to_lat]

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    if map_xlim is not None:
        ax.set_xlim(*map_xlim)
        ax.set_ylim(-1.4, 4.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")

    ax.scatter(all_lons, all_lats, s=8,  color=C["map_all"],
               zorder=4, linewidths=0, alpha=0.7)
    if ex_lons:
        ax.scatter(ex_lons, ex_lats, s=22, color=highlight_col,
                   zorder=5, edgecolors="white", linewidths=0.6, alpha=0.95)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(highlight_col)
        sp.set_linewidth(1.5)


# ── W-covariate site map ────────────────────────────────────────────────────────

def _render_w_site_map(ax, w_label, df_ind_sub, df_rct,
                        site_keys, uganda_gdf, neighbors, lakes_c,
                        n_map_cols=4, map_xlim=None):
    """Map of sites coloured by above-/below-median W value."""
    coord_col = 'geo_long_lat_key'
    if (df_ind_sub is None or w_label not in df_ind_sub.columns
            or coord_col not in df_ind_sub.columns):
        ax.axis("off"); return

    site_w    = df_ind_sub.groupby(coord_col)[w_label].mean()
    threshold = float(site_w.median())

    coord_cols = ['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']
    sites_df   = df_rct[coord_cols].dropna().drop_duplicates('geo_long_lat_key')
    key_to_lon = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_long_center']))
    key_to_lat = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_lat_center']))

    all_lons = [key_to_lon[k] for k in site_keys if k in key_to_lon]
    all_lats = [key_to_lat[k] for k in site_keys if k in key_to_lat]
    hi_lons  = [key_to_lon[k] for k in site_keys
                if k in key_to_lon and site_w.get(k, threshold) > threshold]
    hi_lats  = [key_to_lat[k] for k in site_keys
                if k in key_to_lat and site_w.get(k, threshold) > threshold]

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    if map_xlim is not None:
        ax.set_xlim(*map_xlim)
        ax.set_ylim(-1.4, 4.6)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])

    ax.scatter(all_lons, all_lats, s=12, color=C["map_all"],
               zorder=4, linewidths=0, alpha=0.9)
    if hi_lons:
        ax.scatter(hi_lons, hi_lats, s=22, color=C["map_act"],
                   zorder=5, edgecolors=C["map_act_edge"], linewidths=0.6, alpha=0.95)

    _, tick_lo, tick_hi = w_display(w_label)
    n_hi = len(hi_lons)
    ax.text(0.02, 0.03, f"{n_hi} / {len(site_keys)}  ↑ {tick_hi}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color="#444444",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_title("Geographical distribution", fontsize=9.5, fontweight="bold",
                 color="#333333", loc="left", pad=3)


# ── Row label stripe ───────────────────────────────────────────────────────────

def _row_label(ax, color, text=""):
    ax.set_facecolor(color); ax.axis("off")
    if text:
        ax.text(0.50, 0.50, text, transform=ax.transAxes,
                color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", rotation=90, clip_on=False)


# ── Image ──────────────────────────────────────────────────────────────────────

def _render_image(ax, key, act, border_col, df_rct, faded=False):
    img = _thumb(int(key))
    if img is not None:
        ax.imshow(img, interpolation="bilinear", aspect="equal",
                  alpha=0.35 if faded else 1.0)
        if faded: ax.set_facecolor("#999")
    else:
        ax.set_facecolor("#222")
        ax.text(0.5, 0.5, "N/A", color="white", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)

    loc_row = df_rct[df_rct["geo_long_lat_key"] == int(key)]
    loc_str = ""
    if len(loc_row) and "PNAME_VALUE" in df_rct.columns:
        named = loc_row.dropna(subset=["PNAME_VALUE"])
        if len(named): loc_str = named.iloc[0]["PNAME_VALUE"][:12]

    ax.text(0.04, 0.96, f"{act:.3f}", transform=ax.transAxes,
            color=border_col, fontsize=5.5, fontweight="bold",
            va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.55, pad=1))
    if loc_str:
        ax.text(0.50, 0.03, loc_str, transform=ax.transAxes,
                color="white", fontsize=5.0, ha="center", va="bottom",
                bbox=dict(fc="#000000", ec="none", alpha=0.45, pad=1))

    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(border_col)
        sp.set_linewidth(1.5 if not faded else 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if faded:
        ax.text(0.5, 0.5, "low act.", transform=ax.transAxes,
                color="white", fontsize=5.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(fc="#333", ec="none", alpha=0.55, pad=2))


# ── Feature box borders + light background ────────────────────────────────────

def _draw_boxes(fig, spans, rows, ratios, gs_top, gs_bot, gs_left, gs_right):
    total   = sum(ratios)
    h_scale = (gs_top - gs_bot) / total
    w_pad, h_pad = 0.005, 0.003

    for ri_start, ri_end in spans:
        row = rows[ri_start]
        box_col, hdr_bg, _ = _box_style(row["interpretable"], row["significant"])

        top = gs_top - sum(ratios[:ri_start]) * h_scale + h_pad
        bot = gs_top - sum(ratios[:ri_end+1]) * h_scale - h_pad
        bx  = gs_left - w_pad
        bw  = (gs_right - gs_left) + 2*w_pad
        bh  = top - bot

        fig.add_artist(mpatches.FancyBboxPatch(
            (bx, bot), bw, bh,
            boxstyle="round,pad=0.002",
            fc=hdr_bg, ec=box_col, lw=2.0,
            transform=fig.transFigure, clip_on=False, zorder=0))


# ── Main ───────────────────────────────────────────────────────────────────────

def plot_model_features(embed_model, sae_dim, k, df_rct, outcome="log_skilled_hours"):
    csv_col   = resolve_outcome(outcome)
    model_dir = ROOT / "results" / "uganda" / f"{embed_model}_{sae_dim}"
    out_dir   = model_dir / outcome
    nems_path = out_dir / "nems_result.json"
    if not nems_path.exists():
        print(f"  Skipping {embed_model}_{sae_dim}/{outcome}: nems_result.json not found")
        return

    with open(nems_path) as f:
        nems_out = json.load(f)

    gate_map, ate_est = {}, float("nan")
    if (out_dir / "summary.json").exists():
        with open(out_dir / "summary.json") as f:
            summ = json.load(f)
        for em in summ.get("effect_modifiers", []):
            gate_map[em["label"]] = em
        ate_est = summ.get("ate", {}).get("estimate", float("nan"))

    site_data  = np.load(model_dir / "site_features.npz")
    site_feats = site_data["site_features"]
    site_keys  = site_data["site_keys"]

    T_ind = Z_ind_all = None; df_ind_sub = None
    ind_path = model_dir / "individual_features.npz"
    if ind_path.exists():
        ind_feats  = np.load(ind_path)["features"]
        df_ind     = df_rct.rename(columns={"Wobs": "T", csv_col: "Y"}).copy()
        has_feat   = np.isfinite(ind_feats[:, 0])
        mask_ind   = df_ind["Y"].notna() & has_feat
        T_ind      = df_ind.loc[mask_ind, "T"].values.astype(float)
        Z_ind_all  = ind_feats[mask_ind]
        df_ind_sub = df_ind[mask_ind].reset_index(drop=True)

    # Load Uganda basemap once (cached to disk after first download)
    uganda_map = None
    try:
        uganda_map = load_basemap(DATA_DIR / "map_data")
    except Exception as e:
        print(f"  Warning: could not load basemap: {e}")

    rows, ratios, spans = _build_row_plan(
        nems_out["nems"]["selected"], site_feats, site_keys, k, gate_map)

    if not rows:
        print(f"  No features selected by NEMS — skipping plot for "
              f"{embed_model}_{sae_dim}/{outcome}")
        return

    IMG_W  = 1.75; LABEL_W = 0.36; L_MARG = 0.55; R_MARG = 0.30
    n_cols = N_IMG + 1            # N_IMG example images + 1 mini-map column
    fig_w  = LABEL_W + n_cols * IMG_W + L_MARG + R_MARG

    # Shared map xlim: based on gate_chart panel aspect so all maps look identical
    _UG_YLIM    = (-1.4, 4.6)
    _lat_span   = _UG_YLIM[1] - _UG_YLIM[0]
    _lon_center = (29.1 + 35.4) / 2
    _map_asp    = (n_cols / 3.0) / H_CHART   # 1/3 of image columns, H_CHART tall
    map_xlim    = (_lon_center - _lat_span * _map_asp / 2,
                   _lon_center + _lat_span * _map_asp / 2)
    fig_h  = max(8.0, sum(r * IMG_W for r in ratios) + 0.38)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C["bg"])

    # ── Figure title (two rows) ──────────────────────────────────────────────
    outcome_clean = outcome.replace("_", " ").replace("log ", "log(").rstrip()
    if "log(" in outcome_clean:
        outcome_clean += ")"
    ate_str = f"   ·   ATE = {ate_est:+.4f}" if not np.isnan(ate_est) else ""
    fig.text(0.50, 0.999,
             f"Uganda YOP RCT  ·  Outcome: {outcome_clean}{ate_str}",
             ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")
    fig.text(0.50, 0.988,
             f"Encoder: {embed_model}  ·  SAE dim: {sae_dim}",
             ha="center", va="top", fontsize=9, fontweight="normal", color="#777777")

    GS_TOP = 0.938; GS_BOT = 0.004
    GS_L   = L_MARG / fig_w; GS_R = 1.0 - R_MARG / fig_w
    header_ax_w_in = fig_w * (GS_R - GS_L)

    gs = gridspec.GridSpec(
        len(rows), 1 + n_cols, figure=fig,
        height_ratios=ratios,
        width_ratios=[LABEL_W] + [IMG_W] * n_cols,
        hspace=0.0, wspace=0.014,
        left=GS_L, right=GS_R, top=GS_TOP, bottom=GS_BOT)

    rank = 0
    for ri, row in enumerate(rows):
        kind = row["kind"]

        if kind in ("feat_gap", "tick_gap"):
            ax = fig.add_subplot(gs[ri, :])
            ax.set_facecolor(C["bg"]); ax.axis("off")

        elif kind == "img_label":
            ax = fig.add_subplot(gs[ri, :])
            is_top  = row["is_top"]
            vlm_lbl_row = row.get("vlm_lbl", "")
            bg_col  = "#FDF2F2" if is_top else "#EBF5FB"
            ax.set_facecolor(bg_col); ax.axis("off")
            main_lbl = "Most Activated" if is_top else "Least Activated"
            ax.text(0.012, 0.5, main_lbl,
                    transform=ax.transAxes, color="black",
                    fontsize=8.0, fontweight="bold", va="center", ha="left")
            if vlm_lbl_row:
                # position description after the bold label (empirical x offset)
                x_desc = 0.012 + len(main_lbl) * 0.0058 + 0.005
                ax.text(x_desc, 0.5, f"({vlm_lbl_row})",
                        transform=ax.transAxes, color="#666666",
                        fontsize=7.5, fontweight="normal",
                        va="center", ha="left")

        elif kind == "header":
            rank += 1
            ax = fig.add_subplot(gs[ri, :])
            _render_header(ax, row["entry"], row["group"],
                           row["interpretable"], row["significant"],
                           row["gate"], row["vlm_lbl"], row["max_act"],
                           rank, header_ax_w_in)

        elif kind == "gate_chart":
            group    = row["group"]
            feat_idx = row.get("feat_idx")
            w_label  = row.get("w_label")
            _, bg_col, stripe_col = _box_style(row["interpretable"], row["significant"])

            _row_label(fig.add_subplot(gs[ri, 0]), stripe_col,
                       "GATE" if group == "SAE" else "CATE")

            # Three equal-width panels for both SAE and W features
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=gs[ri, 1:], wspace=0.14)

            ax_bars = fig.add_subplot(inner[0, 0])
            _render_gate_bars(ax_bars, row["gate"], group, ate_est, bg_color=bg_col)

            if group == "SAE":
                if uganda_map is not None:
                    _render_site_map(fig.add_subplot(inner[0, 1]),
                                     feat_idx, site_feats, site_keys, df_rct,
                                     *uganda_map, map_xlim=map_xlim)
                else:
                    fig.add_subplot(inner[0, 1]).axis("off")

                Z_col = Z_ind_all[:, feat_idx].astype(float) if Z_ind_all is not None else None
                _render_z_dist(fig.add_subplot(inner[0, 2]), Z_col, T_ind,
                               False, bg_color=bg_col)

            else:  # W covariate — geo distribution + treatment balance
                if uganda_map is not None:
                    _render_w_site_map(fig.add_subplot(inner[0, 1]),
                                       w_label, df_ind_sub, df_rct,
                                       site_keys, *uganda_map,
                                       map_xlim=map_xlim)
                else:
                    fig.add_subplot(inner[0, 1]).axis("off")

                if df_ind_sub is not None and w_label and w_label in df_ind_sub.columns:
                    Z_col_w = df_ind_sub[w_label].values.astype(float)
                    _render_w_balance(fig.add_subplot(inner[0, 2]),
                                      w_label, Z_col_w, T_ind,
                                      bg_color=bg_col)
                else:
                    fig.add_subplot(inner[0, 2]).axis("off")

        elif kind in ("top_imgs", "bot_imgs"):
            is_top   = kind == "top_imgs"
            bdr      = C["img_hi"] if is_top else C["img_lo"]
            faded    = not row["interpretable"]
            _row_label(fig.add_subplot(gs[ri, 0]),
                       C["row_hi"] if is_top else C["row_lo"])
            for ci in range(N_IMG):
                ax = fig.add_subplot(gs[ri, ci + 1])
                if ci < len(row["keys"]):
                    _render_image(ax, row["keys"][ci], row["acts"][ci],
                                  bdr, df_rct, faded)
                else:
                    ax.set_facecolor(C["bg"]); ax.axis("off")
            # 7th column: mini Uganda map with example locations highlighted
            ax_mm = fig.add_subplot(gs[ri, N_IMG + 1])
            feat_idx_mm = row.get("feat_idx")
            if uganda_map is not None and feat_idx_mm is not None:
                _render_mini_map_examples(
                    ax_mm, row["keys"], df_rct, *uganda_map,
                    site_keys, bdr, map_xlim=map_xlim)
            else:
                ax_mm.set_facecolor(C["bg"]); ax_mm.axis("off")

    _draw_boxes(fig, spans, rows, ratios, GS_TOP, GS_BOT, GS_L, GS_R)

    out_path = out_dir / "summary_illustration.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default=None)
    p.add_argument("--sae-dim", type=int, default=3072)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--outcome", default="log_skilled_hours")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    df_rct = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    if args.all:
        triples = []
        for d in sorted((ROOT / "results" / "uganda").glob("*/*/nems_result.json")):
            outcome_name = d.parent.name
            model_dir    = d.parent.parent.name
            parts = model_dir.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                triples.append((parts[0], int(parts[1]), outcome_name))
    else:
        if not args.embed_model:
            print("Provide --embed-model or --all"); sys.exit(1)
        triples = [(args.embed_model, args.sae_dim, args.outcome)]
    for model, dim, outcome in triples:
        print(f"Plotting {model}_{dim}/{outcome} …")
        plot_model_features(model, dim, args.k, df_rct, outcome)

if __name__ == "__main__":
    main()
