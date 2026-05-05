"""
Feature-image comparison plot for NEXIS-selected effect modifiers.
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
from matplotlib.colors import LinearSegmentedColormap

ROOT     = Path(__file__).parent.parent.parent.parent   # repo root
DATA_DIR = ROOT / "data" / "uganda"
IMG_DIR  = DATA_DIR / "Uganda2000_processed"

from apps.uganda.data import load_image, resolve_outcome, load_basemap, draw_base, w_display, outcome_display


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
IMG_THUMB_PX   = 300

# Gray (#AAAAAA) → green (#27AE60) colormap for site maps
_CMAP_GG = LinearSegmentedColormap.from_list("gray_green", ["#AAAAAA", "#27AE60"])
N_IMG          = 6   # example images per row (+ 1 mini-map column)

H_HDR   = 0.62   # header (tall enough for 2-line stats)
H_CHART = 1.48   # GATE + distribution charts
H_TG    = 0.38   # gap between charts and first examples set
H_ILBL  = 0.12   # image-group label banner (reduced)
H_IMG   = 1.35   # image row
H_GAP   = 0.10   # gap between feature boxes (~3.5 mm, fixed in absolute inches)



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

def _build_row_plan(selected, site_feats, site_keys, k, gate_map, interp_full_map=None):
    if interp_full_map is None:
        interp_full_map = {}
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

            # Prefer structured concepts from interp_full_map; fall back to gate["interp"]
            full_interp         = interp_full_map.get(feat_idx, {})
            activated_concept   = (full_interp.get("activated_concept", "")
                                   or gate.get("interp", ""))
            not_activated_concept = full_interp.get("not_activated_concept", "")
            vlm_model           = full_interp.get("vlm_model", "")
            # Short 2-6 word label for the box title; fall back to full concept
            vlm_lbl             = gate.get("vlm_label", "") or activated_concept

            interpretable = not ("low activation" in (activated_concept or "").lower())
            order         = np.argsort(acts)
            top_keys = site_keys[order[::-1][:k]].tolist()
            bot_keys = site_keys[order[:k]].tolist()
            top_acts = acts[order[::-1][:k]].tolist()
            bot_acts = acts[order[:k]].tolist()

            rows.append(dict(kind="header", entry=entry, group="SAE",
                             interpretable=interpretable, significant=significant,
                             gate=gate, vlm_lbl=vlm_lbl, max_act=max_act,
                             activated_concept=activated_concept,
                             not_activated_concept=not_activated_concept,
                             vlm_model=vlm_model))
            ratios.append(H_HDR); ri += 1

            rows.append(dict(kind="gate_chart", gate=gate, group="SAE",
                             feat_idx=feat_idx, w_label=None,
                             interpretable=interpretable, significant=significant))
            ratios.append(H_CHART); ri += 1

            rows.append(dict(kind="tick_gap")); ratios.append(H_TG); ri += 1

            for rk, ks, al, is_top in [
                    ("top_imgs", top_keys, top_acts, True),
                    ("bot_imgs", bot_keys, bot_acts, False)]:
                # Each row shows the concept relevant to its group
                concept = activated_concept if is_top else not_activated_concept
                rows.append(dict(kind="img_label", is_top=is_top,
                                 vlm_lbl=vlm_lbl, concept=concept))
                ratios.append(H_ILBL); ri += 1
                rows.append(dict(kind=rk, keys=ks, acts=al,
                                 interpretable=interpretable, feat_idx=feat_idx))
                ratios.append(H_IMG); ri += 1

        else:  # W covariate
            sig_w = _is_sig(gate)
            rows.append(dict(kind="header", entry=entry, group="W",
                             interpretable=True, significant=sig_w,
                             gate=gate, vlm_lbl="", max_act=None,
                             activated_concept="", not_activated_concept="",
                             vlm_model=""))
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

def _subtitle_str(gate, group, interpretable, vlm_lbl, label,
                  activated_concept=None, not_activated_concept=None):
    """One-line subtitle: what sites look like when the neuron is active vs. inactive.

    Framing mirrors the GATE chart axes ("active" / "inactive") and the image-example
    row labels below, so the reader can follow a consistent thread top-to-bottom.
    """
    diff = gate.get("diff", float("nan"))
    sig  = _is_sig(gate)
    if group == "SAE":
        if not interpretable:
            return "Low neuron activation — feature could not be interpreted"
        name = vlm_lbl or label
        if sig:
            if diff > 0:
                hi_desc = activated_concept or "active"
                lo_desc = not_activated_concept or "inactive"
            else:
                hi_desc = not_activated_concept or "inactive"
                lo_desc = activated_concept or "active"
            return (f'Individuals located in {hi_desc} sites are characterized by higher GATE '
                    f'than individuals in {lo_desc} sites (Δ={diff:+.3f})')
        return f'No significant GATE difference between active and inactive sites — "{name}"'
    else:
        _, tick_lo, tick_hi = w_display(label)
        if sig:
            hi_grp, lo_grp = (tick_hi, tick_lo) if diff > 0 else (tick_lo, tick_hi)
            return (f'{hi_grp} individuals are characterized by higher GATE '
                    f'than {lo_grp} individuals (Δ={diff:+.3f})')
        return f'No significant GATE difference between {tick_hi} and {tick_lo} individuals'


# ── Header ─────────────────────────────────────────────────────────────────────

def _render_header(ax, entry, group, interpretable, significant,
                   gate, vlm_lbl, max_act, rank, header_ax_w_in,
                   activated_concept=None, not_activated_concept=None, vlm_model=""):
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

    subtitle = _subtitle_str(gate, group, interpretable, vlm_lbl, entry["label"],
                              activated_concept=activated_concept,
                              not_activated_concept=not_activated_concept)
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

    conf = gate.get("vlm_confidence", "") if gate else ""
    if conf and group == "SAE":
        conf_col = {"high": "#1E8449", "medium": "#D4AC0D", "low": "#C0392B"}.get(conf, "#999999")
        model_tag = f"  (model: {vlm_model})" if vlm_model else ""
        ax.text(0.010, 0.30, f"Neural interpretation confidence: {conf}{model_tag}",
                transform=ax.transAxes,
                color=conf_col, fontsize=7, fontweight="bold",
                va="center", ha="left", zorder=3)


# ── GATE bar chart ─────────────────────────────────────────────────────────────

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
        ax.text(0.98, ate_est,
                f" ATE={ate_est:+.3f}",
                transform=trans, ha="right",
                va="bottom",
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
    ax.set_title("GATE",
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
                     uganda_gdf, neighbors, lakes_c, regions_gdf=None,
                     map_xlim=None, Z_ind_all=None, df_ind_sub=None):
    """Map coloured by per-site fraction of active individuals (0=gray → 1=green)."""
    coord_cols = ['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']
    sites_df   = (df_rct[coord_cols].dropna()
                  .drop_duplicates('geo_long_lat_key'))
    key_to_lon = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_long_center']))
    key_to_lat = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_lat_center']))

    # Per-site fraction of active individuals (active = activation > MIN_ACTIVATION)
    if (Z_ind_all is not None and df_ind_sub is not None
            and 'geo_long_lat_key' in df_ind_sub.columns):
        ind_binary = (Z_ind_all[:, feat_idx] > MIN_ACTIVATION).astype(float)
        site_mean  = (pd.Series(ind_binary,
                                index=df_ind_sub['geo_long_lat_key'].values)
                      .groupby(level=0).mean())
    else:
        site_mean = pd.Series(
            (site_feats[:, feat_idx] > MIN_ACTIVATION).astype(float),
            index=site_keys)

    lons   = [key_to_lon[k] for k in site_keys if k in key_to_lon]
    lats   = [key_to_lat[k] for k in site_keys if k in key_to_lat]
    values = [float(site_mean.get(k, 0.0)) for k in site_keys if k in key_to_lon]

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    if map_xlim is not None:
        ax.set_xlim(*map_xlim)
        ax.set_ylim(-1.4, 4.6)
    # map_xlim is pre-computed to match the panel's width/height ratio, so
    # aspect='auto' fills the full cell without geographic distortion and
    # keeps the title aligned with neighbouring plots in the same row.
    ax.set_aspect('auto')
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])

    if lons:
        scatter = ax.scatter(lons, lats, s=18, c=values, cmap=_CMAP_GG,
                             vmin=0, vmax=1, zorder=5, linewidths=0.3,
                             edgecolors="#555555", alpha=0.95)
        import matplotlib.pyplot as plt
        cbar = plt.colorbar(scatter, ax=ax,
                           fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)

    ax.set_title("Geographical distribution", fontsize=9.5, fontweight="bold",
                 color="#333333", loc="left", pad=3)


# ── Mini map for image example rows ────────────────────────────────────────────

def _render_mini_map_examples(ax, example_keys, df_rct,
                               uganda_gdf, neighbors, lakes_c, regions_gdf=None,
                               site_keys_all=None, highlight_col=None, map_xlim=None):
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
                        site_keys, uganda_gdf, neighbors, lakes_c, regions_gdf=None,
                        n_map_cols=4, map_xlim=None):
    """Map of sites coloured by mean W per site (gray=0/min → green=1/max)."""
    coord_col = 'geo_long_lat_key'
    if (df_ind_sub is None or w_label not in df_ind_sub.columns
            or coord_col not in df_ind_sub.columns):
        ax.axis("off"); return

    site_w = df_ind_sub.groupby(coord_col)[w_label].mean()

    # Determine colormap range: binary → [0,1]; continuous → [0, max]
    vals_all  = df_ind_sub[w_label].dropna().values
    finite    = vals_all[np.isfinite(vals_all)]
    is_binary = set(np.unique(finite)).issubset({0.0, 1.0})
    vmin = 0.0
    vmax = 1.0 if is_binary else float(site_w.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    coord_cols = ['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']
    sites_df   = df_rct[coord_cols].dropna().drop_duplicates('geo_long_lat_key')
    key_to_lon = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_long_center']))
    key_to_lat = dict(zip(sites_df['geo_long_lat_key'], sites_df['geo_lat_center']))

    lons   = [key_to_lon[k] for k in site_keys if k in key_to_lon]
    lats   = [key_to_lat[k] for k in site_keys if k in key_to_lat]
    values = [float(site_w.get(k, vmin)) for k in site_keys if k in key_to_lon]

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    if map_xlim is not None:
        ax.set_xlim(*map_xlim)
        ax.set_ylim(-1.4, 4.6)
    # map_xlim is pre-computed to match the panel's width/height ratio, so
    # aspect='auto' fills the full cell without geographic distortion and
    # keeps the title aligned with neighbouring plots in the same row.
    ax.set_aspect('auto')
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])

    if lons:
        scatter = ax.scatter(lons, lats, s=18, c=values, cmap=_CMAP_GG,
                             vmin=vmin, vmax=vmax, zorder=5, linewidths=0.3,
                             edgecolors="#555555", alpha=0.95)
        # Add colorbar to show the covariate value scale
        import matplotlib.pyplot as plt
        cbar = plt.colorbar(scatter, ax=ax,
                           fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)

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

def _draw_boxes(fig, spans, rows, ratios, gs_top, gs_bot, gs_left, gs_right, fig_h=None):
    total   = sum(ratios)
    h_scale = (gs_top - gs_bot) / total
    # keep inter-box gap constant at ~4 pt regardless of figure height
    _fig_h  = fig_h if fig_h is not None else 10.0
    w_pad   = 0.005
    h_pad   = 1 / (72 * _fig_h)   # 1 pt in figure-fraction units

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

def plot_model_features(embed_model, sae_dim, k, df_rct, outcome="log_skilled_hours",
                        pipeline="qwen7b"):
    csv_col   = resolve_outcome(outcome)
    model_dir = ROOT / "results" / "uganda" / f"{embed_model}_{sae_dim}"
    out_dir   = model_dir / outcome
    nexis_path = out_dir / "nexis_result.json"
    if not nexis_path.exists():
        print(f"  Skipping {embed_model}_{sae_dim}/{outcome}: nexis_result.json not found")
        return

    with open(nexis_path) as f:
        nexis_out = json.load(f)

    pipeline_dir = out_dir / pipeline
    gate_map, ate_est = {}, float("nan")
    if (pipeline_dir / "summary.json").exists():
        with open(pipeline_dir / "summary.json") as f:
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
        # Add derived W columns that are built during analysis but absent from df_rct
        # (e.g. lang dummies from lang_group, district dummies from district)
        if "lang_group" in df_ind_sub.columns:
            lang_dum = pd.get_dummies(df_ind_sub["lang_group"], prefix="lang", dtype=float)
            for col in lang_dum.columns:
                if col not in df_ind_sub.columns:
                    df_ind_sub[col] = lang_dum[col].values
        if "district" in df_ind_sub.columns:
            dist_dum = pd.get_dummies(df_ind_sub["district"], prefix="district", dtype=float)
            for col in dist_dum.columns:
                if col not in df_ind_sub.columns:
                    df_ind_sub[col] = dist_dum[col].values

    # Load structured VLM interpretations (activated / not_activated concepts + model name)
    interp_full_map = {}   # feat_idx (int) -> {activated_concept, not_activated_concept, vlm_model, ...}
    interp_path = pipeline_dir / "interpretations.json"
    if interp_path.exists():
        with open(interp_path) as f:
            for entry in json.load(f):
                interp_full_map[entry["feature"]] = {
                    "activated_concept":     (entry.get("activated_concept", "")
                                              or entry.get("description", "")),
                    "not_activated_concept": (entry.get("not_activated_concept", "")
                                              or entry.get("contrast", "")),
                    "vlm_model":             entry.get("vlm_model", ""),
                    "label":                 entry.get("label", ""),
                    "confidence":            entry.get("confidence", ""),
                }

    # Load Uganda basemap once (cached to disk after first download)
    uganda_map = None
    try:
        uganda_map = load_basemap(DATA_DIR / "map_data")
    except Exception as e:
        print(f"  Warning: could not load basemap: {e}")

    rows, ratios, spans = _build_row_plan(
        nexis_out["nexis"]["selected"], site_feats, site_keys, k, gate_map,
        interp_full_map=interp_full_map)

    if not rows:
        print(f"  No features selected by NEXIS — skipping plot for "
              f"{embed_model}_{sae_dim}/{outcome}")
        return

    IMG_W  = 1.75; LABEL_W = 0.36; L_MARG = 0.55; R_MARG = 0.30
    n_cols = N_IMG + 1            # N_IMG example images + 1 mini-map column
    fig_w  = LABEL_W + n_cols * IMG_W + L_MARG + R_MARG

    # Map xlim constants
    _UG_YLIM    = (-1.4, 4.6)
    _lat_span   = _UG_YLIM[1] - _UG_YLIM[0]
    _lon_center = (29.1 + 35.4) / 2
    # gate_chart map: 1/3 of image columns wide, H_CHART tall
    _map_asp    = (n_cols / 3.0) / H_CHART
    map_xlim    = (_lon_center - _lat_span * _map_asp / 2,
                   _lon_center + _lat_span * _map_asp / 2)
    # mini-map (example rows): square data (lon_span = lat_span) so that
    # aspect='equal' produces a square axes box matching the satellite images
    # (which also render as squares, width-limited in their portrait cells)
    mini_map_xlim = (_lon_center - _lat_span / 2,
                     _lon_center + _lat_span / 2)
    # Keep enough headroom for the 2-line title, but avoid unnecessary blank space.
    fig_h  = max(8.0, sum(r * IMG_W for r in ratios) + 0.30)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C["bg"])

    # ── Figure title (two rows) ──────────────────────────────────────────────
    outcome_clean = outcome_display(outcome)
    ate_str = f"   ·   ATE = {ate_est:+.4f}" if not np.isnan(ate_est) else ""
    # All vertical spacing expressed in absolute inches so layout is independent
    # of figure height.
    _TOP_MARGIN  = 0.05   # inches from top of figure to title baseline
    _LINE_GAP    = 0.04   # inches between title baseline and subtitle top
    _SUB_TO_BOX  = 0.10   # inches between subtitle baseline and first box top

    title_y = 1.0 - _TOP_MARGIN / fig_h
    sub_y   = title_y - (11 / 72 + _LINE_GAP) / fig_h   # 11 pt title font

    fig.text(0.50, title_y,
             f"Uganda YOP RCT  ·  Outcome: {outcome_clean}{ate_str}",
             ha="center", va="top", fontsize=11, fontweight="bold", color="#222222")
    fig.text(0.50, sub_y,
             f"Encoder: {embed_model}  ·  SAE dim: {sae_dim}",
             ha="center", va="top", fontsize=9, fontweight="normal", color="#777777")

    # GS_TOP sits _SUB_TO_BOX inches below the subtitle baseline (9 pt font).
    GS_TOP = sub_y - (9 / 72 + _SUB_TO_BOX) / fig_h; GS_BOT = 0.004
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
            concept = row.get("concept", "")   # activated_concept or not_activated_concept
            bg_col  = "#FDF2F2" if is_top else "#EBF5FB"
            ax.set_facecolor(bg_col); ax.axis("off")
            main_lbl = "Most Activated" if is_top else "Least Activated"
            # Bold label left-anchored; concept right-anchored to avoid overlap
            ax.text(0.012, 0.5, main_lbl,
                    transform=ax.transAxes, color="black",
                    fontsize=8.0, fontweight="bold", va="center", ha="left")
            if concept:
                ax.text(0.988, 0.5, concept,
                        transform=ax.transAxes, color="#555555",
                        fontsize=7.5, fontweight="normal",
                        style="italic", va="center", ha="right", clip_on=True)

        elif kind == "header":
            rank += 1
            ax = fig.add_subplot(gs[ri, :])
            _render_header(ax, row["entry"], row["group"],
                           row["interpretable"], row["significant"],
                           row["gate"], row["vlm_lbl"], row["max_act"],
                           rank, header_ax_w_in,
                           activated_concept=row.get("activated_concept"),
                           not_activated_concept=row.get("not_activated_concept"),
                           vlm_model=row.get("vlm_model", ""))

        elif kind == "gate_chart":
            group    = row["group"]
            feat_idx = row.get("feat_idx")
            w_label  = row.get("w_label")
            _, bg_col, stripe_col = _box_style(row["interpretable"], row["significant"])

            _row_label(fig.add_subplot(gs[ri, 0]), stripe_col, "GATE")

            # Three equal-width panels for both SAE and W features
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=gs[ri, 1:], wspace=0.14)

            ax_bars = fig.add_subplot(inner[0, 0])
            _render_gate_bars(ax_bars, row["gate"], group, ate_est, bg_color=bg_col)

            if group == "SAE":
                if uganda_map is not None:
                    _render_site_map(fig.add_subplot(inner[0, 1]),
                                     feat_idx, site_feats, site_keys, df_rct,
                                     *uganda_map, map_xlim=map_xlim,
                                     Z_ind_all=Z_ind_all, df_ind_sub=df_ind_sub)
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
            # small gap between last image column and map
            _p = ax_mm.get_position()
            ax_mm.set_position([_p.x0 + 0.004, _p.y0, _p.width, _p.height])
            feat_idx_mm = row.get("feat_idx")
            if uganda_map is not None and feat_idx_mm is not None:
                _render_mini_map_examples(
                    ax_mm, row["keys"], df_rct, *uganda_map,
                    site_keys, bdr, map_xlim=mini_map_xlim)
            else:
                ax_mm.set_facecolor(C["bg"]); ax_mm.axis("off")

    _draw_boxes(fig, spans, rows, ratios, GS_TOP, GS_BOT, GS_L, GS_R, fig_h=fig_h)

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    out_path = pipeline_dir / "summary_illustration.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight",
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
    p.add_argument("--pipeline", default="qwen7b", choices=["qwen7b", "qwen72b", "points", "geochat"],
                   help="Which interpretation pipeline's output to use.")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    df_rct = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    if args.all:
        triples = []
        for d in sorted((ROOT / "results" / "uganda").glob("*/*/nexis_result.json")):
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
        plot_model_features(model, dim, args.k, df_rct, outcome, pipeline=args.pipeline)

if __name__ == "__main__":
    main()
