#!/usr/bin/env python3
"""
Published NEXIS vs the terminal-backward-step default (NEXIS-v2) on the CelebA figures.

Reads the published sweeps (results/celeba/experiment/) and the v2 sweeps
(results/celeba/experiment_v2/, scripts/celeba/submit_experiment_v2.sh) and writes to
results/celeba/figures_v2/:

  comparison.md          per figure: NEXIS line(s) old vs new, precision / recall / IoU
                         macro-averaged over the figure's cells (one cell = one grid point
                         of one row, mean over 50 seeds), first grid value where each
                         metric reaches 0.95 per row, and cells where a v2 line is worse
                         than its published counterpart (paired over seeds, t < -2).
                         Also checks that the rows rerun under the published
                         configuration reproduce the published parquet exactly.
  side_by_side/<fig>.png the published figure (paper/NeurIPS'26/figures/celeba/, read
                         only) next to the v2 one.

Run figure_main.py --nexis-key NEXIS-v2 and figure_appendix.py --variant v2 first.

Usage
-----
    python src/apps/celeba/compare_v2.py
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OLD = ROOT / "results/celeba/experiment"
NEW = ROOT / "results/celeba/experiment_v2"
PAPER_FIGS = ROOT / "paper/NeurIPS'26/figures/celeba"
OUT = ROOT / "results/celeba/figures_v2"

# Rows of the appendix figures: (sweep file, fixed column, fixed value, x column)
ROWS_APPENDIX = [("n", "fixed_effect", 5.0, "n"), ("n", "fixed_effect", 2.0, "n"),
                 ("effect", "fixed_n", 2000, "effect_scale"),
                 ("effect", "fixed_n", 500, "effect_scale")]
ROWS_MAIN = [ROWS_APPENDIX[0], ROWS_APPENDIX[2]]

# figure -> (tree, rows, [(label, published key or None, v2 key or None)])
PAIR_DEFAULT = [("NEXIS", "NEXIS", "NEXIS-v2")]
FIGURES = {
    "figure_main":   ("k20/sae", ROWS_MAIN, PAIR_DEFAULT),
    "dgp":           ("k20/sae", ROWS_APPENDIX, PAIR_DEFAULT),
    "model_k5":      ("k5/sae", ROWS_APPENDIX, PAIR_DEFAULT),
    "model_precode": ("k20/sae_precode", ROWS_APPENDIX, PAIR_DEFAULT),
    "method_test": ("k20/sae", ROWS_APPENDIX, [
        ("linear (default)", "NEXIS", "NEXIS-v2"),
        ("GCM: quadratic", "NEXIS (test=GCM: quadratic)", "NEXIS-v2 (test=GCM: quadratic)"),
        ("GCM: lgbm", "NEXIS (test=GCM: lgbm)", "NEXIS-v2 (test=GCM: lgbm)"),
        ("PCM: quadratic", "NEXIS (test=PCM: quadratic)", "NEXIS-v2 (test=PCM: quadratic)"),
        ("PCM: lgbm", "NEXIS (test=PCM: lgbm)", "NEXIS-v2 (test=PCM: lgbm)"),
    ]),
    "method_adjust": ("k20/sae", ROWS_APPENDIX, [
        ("None", "NEXIS (adjust=None)", "NEXIS-v2 (adjust=None)"),
        ("FDR", "NEXIS (adjust=FDR)", "NEXIS-v2 (adjust=FDR)"),
        ("FWER (default)", "NEXIS", "NEXIS-v2"),
    ]),
    "method_rho": ("k20/sae", ROWS_APPENDIX, [
        ("rho=0", "NEXIS (rho=0)", "NEXIS-v2 (rho=0)"),
        ("rho=0.2", "NEXIS (rho=0.2)", "NEXIS-v2 (rho=0.2)"),
        ("rho=0.5 (default)", "NEXIS", "NEXIS-v2"),
        ("rho=0.8", "NEXIS (rho=0.8)", "NEXIS-v2 (rho=0.8)"),
    ]),
    # Old figure: True (default) = interleaved, False = forward only.  New figure: four
    # lines; the published default is the reference for every v2 line.
    "method_backward": ("k20/sae", ROWS_APPENDIX, [
        ("forward only (old: False)", "NEXIS (backward=False)", "NEXIS-v2 (terminal=False)"),
        ("fwd + interleaved (old: True, default)", "NEXIS",
         "NEXIS-v2 (interleaved=True, terminal=False)"),
        ("fwd + interleaved + backward step", "NEXIS", "NEXIS-v2 (interleaved=True)"),
        ("fwd + backward step (new default)", "NEXIS", "NEXIS-v2"),
    ]),
}
METRICS = ["precision", "recall", "iou"]
KEY = ["seed"]

# Published configuration rerun under a v2 name: must reproduce the published rows.
REPRO = [("Marginal Testing", "Marginal Testing"),
         ("Marginal Testing (FWER)", "Marginal Testing (FWER)"),
         ("Marginal Testing (FDR)", "Marginal Testing (FDR)"),
         ("NEXIS", "NEXIS-v2 (interleaved=True, terminal=False)"),
         ("NEXIS (backward=False)", "NEXIS-v2 (terminal=False)")]


def load(base: Path, tree: str) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(base / tree / f"{s}_sweep.parquet") for s in ("effect", "n")}


def row_frame(dfs, row, method):
    sweep, fcol, fval, xcol = row
    d = dfs[sweep]
    return d[(d[fcol] == fval) & (d["method"] == method)]


def cell_means(dfs, rows, method) -> pd.DataFrame:
    """One line per (row, x) cell: mean of each metric over seeds."""
    out = []
    for i, row in enumerate(rows):
        d = row_frame(dfs, row, method)
        if d.empty:
            return pd.DataFrame()
        g = d.groupby(row[3])[METRICS].mean().reset_index().rename(columns={row[3]: "x"})
        g["row"] = i
        out.append(g)
    return pd.concat(out, ignore_index=True)


def first_hit(dfs, row, method, metric, thr=0.95):
    d = row_frame(dfs, row, method)
    if d.empty:
        return "n/a"
    m = d.groupby(row[3])[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return f"{v:g}"


def row_name(row):
    sweep, fcol, fval, _ = row
    return f"n-sweep eta={fval:g}" if sweep == "n" else f"eta-sweep n={fval:g}"


def worse_cells(old, new, rows, m_old, m_new):
    """Cells where the v2 line is worse than the published one on IoU/precision/recall,
    paired over seeds (same simulated data), mean diff < 0 and t < -2."""
    flags = []
    for row in rows:
        a = row_frame(old, row, m_old).set_index([row[3], "seed"])[METRICS]
        b = row_frame(new, row, m_new).set_index([row[3], "seed"])[METRICS]
        if a.empty or b.empty:
            continue
        j = a.join(b, lsuffix="_old", rsuffix="_new", how="inner")
        for x, g in j.groupby(level=0):
            for met in METRICS:
                d = g[f"{met}_new"] - g[f"{met}_old"]
                mu = d.mean()
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                if mu < 0 and se > 0 and mu / se < -2:
                    flags.append(f"{row_name(row)}, x={x:g}: {met} "
                                 f"{g[f'{met}_old'].mean():.3f} -> {g[f'{met}_new'].mean():.3f} "
                                 f"(paired diff {mu:+.3f}, t={mu / se:.1f})")
    return flags


def repro_check(trees) -> list[str]:
    lines = []
    cols = ["iou", "recall", "precision", "tp", "fp", "n_selected"]
    for tree in trees:
        old, new = load(OLD, tree), load(NEW, tree)
        for s in ("effect", "n"):
            fcol = "fixed_n" if s == "effect" else "fixed_effect"
            xcol = "effect_scale" if s == "effect" else "n"
            for m_old, m_new in REPRO:
                b = new[s][new[s]["method"] == m_new]
                if b.empty:
                    continue
                a = old[s][old[s]["method"] == m_old].set_index([fcol, xcol, "seed"])[cols]
                b = b.set_index([fcol, xcol, "seed"])[cols]
                j = a.join(b, lsuffix="_o", rsuffix="_n", how="outer")
                same = all(np.array_equal(j[f"{c}_o"].values, j[f"{c}_n"].values)
                           for c in cols)
                lines.append(f"| {tree} | {s} | {m_old} | {m_new} | {len(a)} / {len(b)} | "
                             f"{'identical' if same else 'DIFFERENT'} |")
    return lines


def side_by_side(name: str, old_pdf: Path, new_pdf: Path, out_png: Path, dpi=90):
    with tempfile.TemporaryDirectory() as tmp:
        imgs = []
        for tag, pdf in (("old", old_pdf), ("new", new_pdf)):
            stem = Path(tmp) / tag
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                            str(pdf), str(stem)], check=True)
            imgs.append(Image.open(f"{stem}.png").convert("RGB"))
    head = 40
    w = sum(i.width for i in imgs) + 30
    h = max(i.height for i in imgs) + head
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for img, title in zip(imgs, (f"{name}: published (paper)",
                                 f"{name}: NEXIS-v2 (terminal backward step)")):
        canvas.paste(img, (x, head))
        draw.text((x + 10, 8), title, fill="black", font=font)
        x += img.width + 30
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def fmt(v):
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    out = args.out_dir

    md = ["# CelebA figures: published NEXIS vs NEXIS-v2 (terminal backward step)", "",
          "NEXIS-v2 = forward step (Bonferroni gate alpha/|remaining|, rho=0.5), no "
          "interleaved backward step, then the terminal backward step (keep j only if "
          "p_j(A) <= alpha/m for every A in S~ minus j; m = 9,216 columns). Ablation "
          "variants deviate from it along one axis; for adjust=None/FDR only the forward "
          "gate changes, the terminal gate stays alpha/m.", "",
          "Macro = mean over the figure's cells of the per-cell mean over 50 seeds "
          "(Figure 3: 21 cells; appendix figures: 42 cells). 'first >= 0.95' = smallest "
          "grid value where the seed-mean metric reaches 0.95 (the grid is "
          "n in {50,...,10000}, eta in {1,...,10}). 'Worse cells' are paired over seeds "
          "(same simulated data), flagged when the mean paired difference is negative "
          "with t < -2.", ""]

    cache = {}
    for fig, (tree, rows, pairs) in FIGURES.items():
        if tree not in cache:
            cache[tree] = (load(OLD, tree), load(NEW, tree))
        old, new = cache[tree]
        md += [f"## {fig} ({tree})", "",
               "| line | precision old | precision new | recall old | recall new | "
               "IoU old | IoU new |", "|---|---|---|---|---|---|---|"]
        for label, m_old, m_new in pairs:
            a, b = cell_means(old, rows, m_old), cell_means(new, rows, m_new)
            if a.empty or b.empty:
                md.append(f"| {label} | missing ({'old' if a.empty else 'new'}) |||||| ")
                continue
            md.append(f"| {label} | " + " | ".join(
                f"{fmt(a[m].mean())} | {fmt(b[m].mean())}" for m in METRICS) + " |")
        md += ["", "First grid value where the seed-mean metric reaches 0.95 "
               "(old -> new):", "",
               "| line | row | precision | recall | IoU |", "|---|---|---|---|---|"]
        for label, m_old, m_new in pairs:
            for row in rows:
                cells = [f"{first_hit(old, row, m_old, m)} -> {first_hit(new, row, m_new, m)}"
                         for m in METRICS]
                md.append(f"| {label} | {row_name(row)} | " + " | ".join(cells) + " |")
        md += ["", "Worse cells (v2 line below its published counterpart):", ""]
        any_flag = False
        for label, m_old, m_new in pairs:
            for f in worse_cells(old, new, rows, m_old, m_new):
                md.append(f"- {label}: {f}")
                any_flag = True
        if not any_flag:
            md.append("- none")
        md.append("")

    md += ["## Reproduction check (published configuration rerun in experiment_v2)", "",
           "| tree | sweep | published method | v2 rerun | rows old / new | result |",
           "|---|---|---|---|---|---|"]
    md += repro_check(["k20/sae", "k20/sae_precode", "k5/sae"])
    md.append("")

    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text("\n".join(md))
    print(f"Saved → {out / 'comparison.md'}")

    if not args.no_images:
        for fig in FIGURES:
            old_pdf, new_pdf = PAPER_FIGS / f"{fig}.pdf", out / f"{fig}.pdf"
            if old_pdf.exists() and new_pdf.exists():
                side_by_side(fig, old_pdf, new_pdf, out / "side_by_side" / f"{fig}.png")
                print(f"Saved → {out / 'side_by_side' / f'{fig}.png'}")
            else:
                print(f"skip side-by-side for {fig}: missing {old_pdf if not old_pdf.exists() else new_pdf}")


if __name__ == "__main__":
    main()
