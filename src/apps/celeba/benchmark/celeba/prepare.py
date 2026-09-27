"""Data preparation: CelebA -> SigLIP 2 tokens -> TopK SAEs -> encoded pool -> S*.

Steps (each skipped when its outputs exist, unless overwrite=True):

  embed         download the CelebA validation split (19,867 aligned-and-cropped images),
                save the 40 binary attributes and 128 x 128 thumbnails, and embed every
                image with SigLIP 2 ViT-B/16 at 224 px: 196 final-layer patch tokens
                (768-d, no attention pooling head), saved per patch (float16) and
                mean-pooled (float32).  [GPU]
  sae           train the TopK SAEs (m = 9,216, k in {5, 20}) on the patch tokens.  [GPU]
  encode        encode the mean-pooled pool embeddings: z (sparse) and z_pre (dense).
  ground-truth  principal coordinate of each attribute: argmax_j of the best-threshold
                F1 of z_pre^j against the attribute over the whole pool.
  replica       the same pipeline for the replica dictionary: embed a random sample of
                19,867 train-split images (seed 1), train the k = 20 SAE on it, encode
                the validation pool with it, and compute its principal coordinates. [GPU]
"""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as C

STEPS = ("embed", "sae", "encode", "ground-truth", "replica")


# ── CelebA and SigLIP 2 ───────────────────────────────────────────────────────

def load_split(split: str, sample_n: Optional[int] = None, sample_seed: int = 0):
    """Streamed split, or a sorted random subset of sample_n rows (non-streaming)."""
    from datasets import load_dataset

    if sample_n is None:
        return load_dataset(C.HF_DATASET, split=split, streaming=True,
                            revision=C.HF_DATASET_REVISION)
    ds = load_dataset(C.HF_DATASET, split=split, revision=C.HF_DATASET_REVISION)
    rng = np.random.default_rng(sample_seed)
    idx = np.sort(rng.choice(len(ds), size=sample_n, replace=False))
    return ds.select(idx)


def collect_labels(rows) -> pd.DataFrame:
    """Every non-image field; attributes as {0, 1} int8."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "image"} for r in rows])
    for col in df.columns:
        if col == "celeb_id":
            df[col] = df[col].astype(np.int32)
        elif df[col].dtype.kind in ("i", "f") and df[col].min() < 0:
            df[col] = ((df[col] + 1) // 2).astype(np.int8)
        else:
            df[col] = df[col].astype(np.int8)
    return df


def embed_images(rows, n_images: int, patches_out: Path, device: str,
                 batch_size: int = C.EMBED_BATCH, limit: Optional[int] = None,
                 thumbs_out: Optional[Path] = None) -> np.ndarray:
    """Mean-pooled SigLIP 2 patch tokens (returned), per-patch tokens (to patches_out) and,
    optionally, 128 x 128 thumbnails (to thumbs_out)."""
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)), transforms.ToTensor(),
                             transforms.Normalize(mean=list(C.IMG_MEAN), std=list(C.IMG_STD))])
    model = timm.create_model(C.BACKBONE, pretrained=True, num_classes=0, img_size=C.IMG_SIZE)
    model.eval().to(device)
    n_prefix = getattr(model, "num_prefix_tokens", 0)
    n_total = n_images if limit is None else min(limit, n_images)
    thumb = transforms.Compose([
        transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(128)])
    patches_out.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(patches_out, dtype=np.float16, mode="w+",
                     shape=(n_total, C.N_PATCHES, C.EMBED_DIM))
    pooled, thumbs, buf, offset = [], [], [], 0

    @torch.no_grad()
    def flush(batch):
        nonlocal offset
        x = torch.stack(batch).to(device)
        tok = model.forward_features(x)[:, n_prefix:]          # (B, 196, 768)
        pooled.append(tok.mean(dim=1).cpu().float().numpy())
        mmap[offset:offset + len(batch)] = tok.cpu().to(torch.float16).numpy()
        offset += len(batch)

    for i, row in enumerate(rows):
        if i >= n_total:
            break
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        buf.append(tf(img))
        if thumbs_out is not None:
            thumbs.append(np.asarray(thumb(img), dtype=np.uint8))
        if len(buf) == batch_size:
            flush(buf)
            buf = []
            print(f"  embedded {offset:,}/{n_total:,}", end="\r", flush=True)
    if buf:
        flush(buf)
    mmap.flush()
    print(f"  embedded {offset:,} images")
    if thumbs_out is not None:
        np.save(thumbs_out, np.stack(thumbs, axis=0))
    return np.concatenate(pooled, axis=0)


def step_embed(corpus_dir: Path, split: str, n_images: int, sample_n=None, sample_seed=0,
               overwrite=False, device=None, limit=None, thumbs_out=None) -> None:
    emb, pat = C.embeddings_path(corpus_dir), C.patches_path(corpus_dir)
    labels = corpus_dir / "labels.parquet"
    done = [emb, pat, labels] + ([thumbs_out] if thumbs_out is not None else [])
    if all(f.exists() for f in done) and not overwrite:
        print(f"skip (exists): {emb}")
        return
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CelebA [{split}] -> {corpus_dir} (device={device})", flush=True)
    df = collect_labels(islice(load_split(split, sample_n, sample_seed), limit))
    corpus_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(labels, index=False)
    pooled = embed_images(load_split(split, sample_n, sample_seed), len(df), pat, device,
                          limit=limit, thumbs_out=thumbs_out)
    np.save(emb, pooled)
    print(f"  labels {df.shape} -> {labels}\n  embeddings {pooled.shape} -> {emb}")


# ── SAE, encoding, ground truth ───────────────────────────────────────────────

def step_sae(k: int, replica: bool, overwrite=False) -> None:
    from celeba.sae import open_patches, save_sae, train_topk_sae

    ckpt = C.sae_checkpoint(k, replica)
    if ckpt.exists() and not overwrite:
        print(f"skip (exists): {ckpt}")
        return
    corpus = C.REPLICA_CORPUS_DIR if replica else C.POOL_DIR
    n = len(np.load(C.embeddings_path(corpus), mmap_mode="r"))
    patches = open_patches(C.patches_path(corpus), n, C.EMBED_DIM)
    sae = train_topk_sae(patches, hidden_dim=C.SAE["hidden_dim"], top_k=k,
                         epochs=C.SAE["epochs"], batch_images=C.SAE["batch_images"],
                         lr=C.SAE["lr"], seed=C.SAE["seed"])
    save_sae(sae, ckpt)
    print(f"  -> {ckpt}")


def step_encode(k: int, replica: bool, overwrite=False) -> None:
    from celeba.sae import encode, load_sae

    out_z, out_pre = C.codes_path(k, replica, "z"), C.codes_path(k, replica, "z_pre")
    if out_z.exists() and out_pre.exists() and not overwrite:
        print(f"skip (exists): {out_z}")
        return
    z, z_pre = encode(load_sae(C.sae_checkpoint(k, replica)),
                      np.load(C.embeddings_path(C.POOL_DIR)))
    out_z.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_z, z)
    np.save(out_pre, z_pre)
    print(f"  z {z.shape} (mean L0 {float((z > 0).sum(axis=1).mean()):.1f}) -> {out_z}")


def best_threshold_f1(features: np.ndarray, labels: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Per coordinate, the best F1 over all thresholds of the coordinate for the label."""
    N, D = features.shape
    y = labels.astype(np.int32)
    total_pos = int(y.sum())
    if total_pos in (0, N):
        return np.zeros(D, dtype=np.float32)
    best = np.zeros(D, dtype=np.float32)
    for start in range(0, D, chunk):
        end = min(start + chunk, D)
        S = features[:, start:end].astype(np.float32)
        y_sorted = y[np.argsort(-S, axis=0)]
        TP = np.cumsum(y_sorted, axis=0, dtype=np.int32)
        FP = np.cumsum(1 - y_sorted, axis=0, dtype=np.int32)
        FN = total_pos - TP
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(TP + FP > 0, TP / (TP + FP).astype(np.float32), 0.0)
            R = np.where(TP + FN > 0, TP / (TP + FN).astype(np.float32), 0.0)
            F = np.where(P + R > 0, 2 * P * R / (P + R), 0.0)
        best[start:end] = F.max(axis=0).astype(np.float32)
    return best


def step_ground_truth(k: int, replica: bool, overwrite=False) -> None:
    """Principal coordinate (F1 argmax on z_pre) of every attribute in config.GT_ATTRS."""
    path = C.ground_truth_path(k, replica)
    gt = json.loads(path.read_text()) if path.exists() and not overwrite else \
        {"dictionary": C.dictionary_name(k, replica), "attributes": {}, "f1": {}}
    todo = [a for a in C.GT_ATTRS if a not in gt["attributes"]]
    if not todo:
        print(f"skip (exists): {path}")
        return
    z_pre = np.load(C.codes_path(k, replica, "z_pre"))
    labels = pd.read_parquet(C.LABELS)
    for a in todo:
        f1 = best_threshold_f1(z_pre, labels[a].values.astype(float))
        order = np.argsort(f1)
        j, j2 = int(order[-1]), int(order[-2])
        gt["attributes"][a] = {"coordinate": j, "f1": float(f1[j]),
                               "runner_up": j2, "runner_up_f1": float(f1[j2])}
        gt["f1"][a] = f1.tolist()
        print(f"  {C.dictionary_name(k, replica)} {a:12s}: coordinate {j:5d}  F1={f1[j]:.3f}"
              f"  runner-up {j2} F1={f1[j2]:.3f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gt))


def run(steps=STEPS, overwrite=False, embed_limit: Optional[int] = None) -> None:
    for step in steps:
        print(f"\n== prepare: {step} ==", flush=True)
        if step == "embed":
            step_embed(C.POOL_DIR, C.POOL_SPLIT, C.POOL_SIZE, overwrite=overwrite,
                       limit=embed_limit, thumbs_out=C.IMAGES)
        elif step == "sae":
            for k in C.SAE_KS:
                step_sae(k, False, overwrite)
        elif step == "encode":
            for k in C.SAE_KS:
                step_encode(k, False, overwrite)
        elif step == "ground-truth":
            for k in C.SAE_KS:
                step_ground_truth(k, False, overwrite)
        elif step == "replica":
            R = C.REPLICA
            step_embed(C.REPLICA_CORPUS_DIR, R["split"], R["sample_n"], sample_n=R["sample_n"],
                       sample_seed=R["sample_seed"], overwrite=overwrite, limit=embed_limit)
            for k in R["ks"]:
                step_sae(k, True, overwrite)
                step_encode(k, True, overwrite)
                step_ground_truth(k, True, overwrite)
        else:
            raise ValueError(f"unknown step {step!r}; choose from {STEPS}")
