#!/usr/bin/env python3
"""Attribute-to-coordinate F1 matrix of a CelebA dictionary.

For every CelebA attribute, computes the best-threshold F1 of each coordinate
(experiment.compute_f1_scores, the ground-truth rule of the paper) and saves the full
(n_attrs, m) F1 matrix.  third_modifier_candidates.py reads f1_sae_precode_k20.npz to
rank the candidates for the third direct modifier of the r = 3 DGP.  (Written for the
interleaved-backward experiment of the NeurIPS rebuttal, hence the output path; that
experiment is at git tag neurips-rebuttal-final.)

Usage: python src/apps/celeba/interleaved_backward_align.py <dict> [<dict> ...]
  dict in {sae_k20, sae_k5, sae_precode_k20, sae_precode_k5, sae_dinov2_k20,
           sae_dinov2_k5, sae_precode_dinov2_k20}
Writes results/celeba/interleaved_backward/align/f1_<dict>.npz  (f1, attrs)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from apps.celeba.experiment import compute_f1_scores  # noqa: E402

OUT = ROOT / "results/celeba/interleaved_backward/align"
OUT.mkdir(parents=True, exist_ok=True)
labels = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
attrs = [c for c in labels.columns if c != "celeb_id"]
nj = int(os.environ.get("NJOBS", "8"))

for d in sys.argv[1:]:
    out = OUT / f"f1_{d}.npz"
    if out.exists():
        print("exists", out); continue
    F = np.load(ROOT / "data/celeba/embeddings" / f"{d}.npy", mmap_mode="r")
    f1 = Parallel(n_jobs=nj)(delayed(compute_f1_scores)(
        np.asarray(F), labels[a].values.astype(float)) for a in attrs)
    np.savez(out, f1=np.stack(f1), attrs=np.array(attrs))
    print("wrote", out, flush=True)
