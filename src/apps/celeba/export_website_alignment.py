#!/usr/bin/env python3
"""
Export the "Check Principal Alignment" panels of the project website (docs/index.html).

For every dictionary the CelebA section shows (TopK SAE with k = 20 on sparse codes Z,
k = 20 on dense pre-activations Z_pre, k = 5 on Z, and the independently retrained k = 20
replica SAE of paper Figure replica_k20, all encoding the same 19,867 pool images) and
every concept that is a direct modifier in one of its experiments (Wearing_Hat,
Eyeglasses, and Sideburns for the r = 3 DGP ablation on the main dictionary), this writes

  * the alignment spectrum: the sign-free AUC, max(AUC, 1 - AUC), of every coordinate's
    activation as a score for the concept label over the 19,867 pool images, sorted
    (as paper Figure pa_spectrum, alignment_appendix.py); the top ranks and the
    principal coordinate are exact, the tail is decimated;
  * the N_IMG most and N_IMG least activating images of the principal coordinate (the
    top row as pa_top_images), as one downscaled JPEG sprite (row 0: most activating,
    row 1: least activating), and whether each image carries the label. Ties at the
    bottom (sparse codes are zero on most images) are broken by a fixed random
    permutation, so the least activating row is a random draw among the lowest.

The principal coordinates are read from the ground_truth.json of the experiments behind
the site's curves (results/celeba/experiment_v2/<k>/<view>/ and, for Sideburns,
experiment_v2_r3/k20/sae/, experiment_v2_resample_b1/k20/sae/ for the replica); on the
main dictionary they are 5348, 5537 and 1683.

Outputs (docs/assets/):
  celeba_alignment.json
      {"<dict>": {"label", "codes", "m",
                  "concepts": {"<attr>": {"principal", "rank", "auc", "runner_up",
                               "auc_runner_up", "gap", "prevalence", "top100_purity",
                               "spectrum": [[rank, auc], ...],
                               "images": {"sprite", "size", "n", "top", "bottom",
                                          "top_attr", "bottom_attr"}}}}}
  celeba_alignment/<dict>_<attr>.jpg   2 x N_IMG sprite: most / least activating images

Usage (CPU, a few minutes):
    PYTHONPATH=src python src/apps/celeba/export_website_alignment.py
    PYTHONPATH=src python src/apps/celeba/export_website_alignment.py \
        --data-root /path/to/data --results-dir /path/to/results/celeba
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from apps.celeba.alignment_appendix import TOP_M, column_auc, top_purity

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# website key -> (label, codes file under data/, experiment dir under results/celeba)
DICTS = {
    "k20_sae":     ("TopK SAE, k = 20, sparse codes Z", "celeba/embeddings/sae_k20.npy",
                    "experiment_v2/k20/sae"),
    "k20_precode": ("TopK SAE, k = 20, pre-activations Z_pre",
                    "celeba/embeddings/sae_precode_k20.npy", "experiment_v2/k20/sae_precode"),
    "k5_sae":      ("TopK SAE, k = 5, sparse codes Z", "celeba/embeddings/sae_k5.npy",
                    "experiment_v2/k5/sae"),
    "k20_sae_rep": ("Replica TopK SAE, k = 20, sparse codes Z",
                    "celeba_resample_b1/eval/embeddings/sae_k20.npy",
                    "experiment_v2_resample_b1/k20/sae"),
}
R3_GT = "experiment_v2_r3/k20/sae/ground_truth.json"   # adds Sideburns on the main dictionary
PAPER_MAIN = {"Wearing_Hat": 5348, "Eyeglasses": 5537, "Sideburns": 1683}

N_EXACT = 100        # spectrum ranks kept exactly
STRIDE = 25          # then one rank in STRIDE
N_IMG = 16           # most / least activating images per principal (site shows 4, 8, 16)
THUMB = 72           # thumbnail side (px); the source thumbnails are 128 x 128
JPEG_Q = 72


def principals(res: Path, sub: str, with_sideburns: bool) -> dict[str, int]:
    gt = json.loads((res / sub / "ground_truth.json").read_text())
    out = {gt["w1_attr"]: int(gt["w1_neurons"][0]), gt["w2_attr"]: int(gt["w2_neurons"][0])}
    if with_sideburns:
        g3 = json.loads((res / R3_GT).read_text())
        for attr, (j,) in zip(g3["w_attrs"], g3["neurons_per_attr"]):
            assert out.get(attr, j) == j, (attr, j, out)
            out[attr] = int(j)
    return out


def decimated_spectrum(auc: np.ndarray, principal: int) -> tuple[list, int]:
    order = np.argsort(-auc, kind="stable")
    s = auc[order]
    rank_p = int(np.flatnonzero(order == principal)[0]) + 1
    keep = set(range(1, min(N_EXACT, len(s)) + 1))
    keep |= set(range(N_EXACT + STRIDE, len(s) + 1, STRIDE)) | {len(s), rank_p}
    return [[r, round(float(s[r - 1]), 3)] for r in sorted(keep)], rank_p


def extremes(z: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Most and least activating images; ties broken by one fixed random permutation."""
    perm = np.random.default_rng(seed).permutation(len(z))
    order = perm[np.argsort(-z[perm], kind="stable")]
    return order[:N_IMG], order[::-1][:N_IMG]


def sprite(imgs: np.ndarray, rows: list[np.ndarray], path: Path) -> None:
    sheet = Image.new("RGB", (THUMB * N_IMG, THUMB * len(rows)))
    for r, idx in enumerate(rows):
        for k, i in enumerate(idx):
            im = Image.fromarray(np.asarray(imgs[i])).resize((THUMB, THUMB), Image.LANCZOS)
            sheet.paste(im, (k * THUMB, r * THUMB))
    sheet.save(path, "JPEG", quality=JPEG_Q, optimize=True, progressive=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/celeba"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = ap.parse_args()
    root, res, out = (p if p.is_absolute() else ROOT / p
                      for p in (args.data_root, args.results_dir, args.out_dir))
    data = root / "celeba"
    (out / "celeba_alignment").mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(data / "labels.parquet")
    imgs = np.load(data / "images.npy", mmap_mode="r")
    assert len(labels) == len(imgs)

    result = {}
    for key, (label, codes, sub) in DICTS.items():
        pr = principals(res, sub, with_sideburns=(key == "k20_sae"))
        if key == "k20_sae":
            assert pr == PAPER_MAIN, (pr, PAPER_MAIN)
        Z = np.ascontiguousarray(np.load(root / codes, mmap_mode="r"), dtype=np.float32)
        assert len(Z) == len(labels)
        entry = {"label": label, "codes": f"data/{codes}",
                 "m": int(Z.shape[1]), "concepts": {}}
        for attr, j in pr.items():
            w = labels[attr].values.astype(np.float64)
            auc_raw = column_auc(Z, w)
            auc = np.maximum(auc_raw, 1 - auc_raw)
            others = np.arange(Z.shape[1]) != j
            runner = int(np.flatnonzero(others)[np.argmax(auc[others])])
            spec, rank_p = decimated_spectrum(auc, j)
            top, bot = extremes(Z[:, j])
            fname = f"celeba_alignment/{key}_{attr}.jpg"
            sprite(imgs, [top, bot], out / fname)
            entry["concepts"][attr] = {
                "principal": j, "rank": rank_p, "auc": round(float(auc[j]), 3),
                "runner_up": runner, "auc_runner_up": round(float(auc[runner]), 3),
                "gap": round(float(auc[j] - auc[runner]), 3),
                "prevalence": round(float(w.mean()), 4),
                "top100_purity": round(top_purity(Z[:, j], w)["purity"], 2),
                "spectrum": spec,
                "images": {"sprite": fname, "size": THUMB, "n": N_IMG,
                           "top": [int(i) for i in top], "bottom": [int(i) for i in bot],
                           "top_attr": [int(w[i]) for i in top],
                           "bottom_attr": [int(w[i]) for i in bot]},
            }
            print(f"{key:12s} {attr:12s} j={j:5d} rank {rank_p:4d} AUC {auc[j]:.3f} | "
                  f"runner-up {runner} {auc[runner]:.3f} (gap {auc[j] - auc[runner]:+.3f}) | "
                  f"top-{TOP_M} purity {entry['concepts'][attr]['top100_purity']:.2f} | "
                  f"top-{N_IMG} labelled {int(w[top].sum())}, bottom-{N_IMG} labelled "
                  f"{int(w[bot].sum())}", flush=True)
        result[key] = entry
        del Z

    path = out / "celeba_alignment.json"
    path.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
