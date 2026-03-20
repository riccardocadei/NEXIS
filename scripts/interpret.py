"""
Interpret NEMS-selected SAE features using Claude's vision API.

For each feature Claude receives three labeled groups of images:
  - TOP-K    highest activations  (feature fires strongly here)
  - BOTTOM-K lowest activations   (feature is silent here)
  - RANDOM-K random sample        (baseline / context)

The contrastive framing lets Claude identify what distinguishes
activating sites from non-activating ones.

Usage
-----
    python scripts/interpret.py [--k 6] [--model claude-opus-4-6]

Requires:  pip install anthropic
           ANTHROPIC_API_KEY set in environment.
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).parent.parent
IMG_DIR = ROOT / "data" / "uganda" / "Uganda2000_processed"
OUT_DIR = ROOT / "results" / "uganda"
sys.path.insert(0, str(ROOT / "src"))

from uganda import load_image   # (H, W, 3) float32 in [0,1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=6,
                   help="Images per group (top / bottom / random)")
    p.add_argument("--model", default="claude-opus-4-6")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_site_image(key: int, size: int = 224) -> Image.Image | None:
    arr = load_image(key, IMG_DIR)
    if arr is None:
        return None
    img = Image.fromarray((arr * 255).astype("uint8"))
    return img.resize((size, size), Image.BICUBIC)


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def load_group(keys, activations) -> tuple[list[Image.Image], list[int], list[float]]:
    """Load images for a group, skipping missing files."""
    imgs, valid_keys, valid_acts = [], [], []
    for key, act in zip(keys, activations):
        img = load_site_image(int(key))
        if img is not None:
            imgs.append(img)
            valid_keys.append(int(key))
            valid_acts.append(float(act))
    return imgs, valid_keys, valid_acts


# ── Claude API ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in remote sensing and satellite image analysis.
You are helping interpret features of a Sparse Autoencoder (SAE) trained on
Landsat satellite imagery from northern Uganda (~year 2000).
The images are false-color composites: NIR->Red, Green->Green, SWIR->Blue.
In this color scheme: healthy vegetation appears bright red/pink, bare soil
appears tan/brown, water appears dark blue, and settlements appear grey/cyan.
"""


def _img_block(img: Image.Image) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": image_to_base64(img)}}


def build_user_message(feature_idx: int,
                       top_imgs, top_acts,
                       bottom_imgs, bottom_acts,
                       rand_imgs, rand_acts) -> list[dict]:
    content = []

    content.append({"type": "text", "text": (
        f"I need you to interpret SAE feature #{feature_idx}.\n"
        "I will show you three groups of satellite images.\n\n"
        f"GROUP A - TOP activations {[f'{a:.2f}' for a in top_acts]}: "
        "sites where this feature fires most strongly."
    )})
    for img in top_imgs:
        content.append(_img_block(img))

    content.append({"type": "text", "text": (
        f"GROUP B - BOTTOM activations {[f'{a:.2f}' for a in bottom_acts]}: "
        "sites where this feature is essentially silent."
    )})
    for img in bottom_imgs:
        content.append(_img_block(img))

    content.append({"type": "text", "text": (
        f"GROUP C - RANDOM sample {[f'{a:.2f}' for a in rand_acts]}: "
        "randomly selected sites for context."
    )})
    for img in rand_imgs:
        content.append(_img_block(img))

    content.append({"type": "text", "text": (
        "\nBased on the contrast between Group A (high activation) and "
        "Group B (low activation), please:\n"
        "1. Describe in one sentence what landscape/land-cover type "
        "Group A has that Group B lacks.\n"
        "2. Give a short label (2-5 words) for this visual concept.\n"
        "3. Rate your confidence (low / medium / high).\n\n"
        "Format:\n"
        "Description: <sentence>\n"
        "Label: <short label>\n"
        "Confidence: <low|medium|high>"
    )})
    return content


def interpret_feature(client, feature_idx: int,
                      top_keys, top_acts,
                      bottom_keys, bottom_acts,
                      rand_keys, rand_acts,
                      model: str) -> dict:
    top_imgs,    top_keys,    top_acts    = load_group(top_keys,    top_acts)
    bottom_imgs, bottom_keys, bottom_acts = load_group(bottom_keys, bottom_acts)
    rand_imgs,   rand_keys,   rand_acts   = load_group(rand_keys,   rand_acts)

    if not top_imgs:
        return {"feature": feature_idx, "label": "N/A",
                "description": "No images found", "confidence": "low"}

    response = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(
            feature_idx,
            top_imgs, top_acts,
            bottom_imgs, bottom_acts,
            rand_imgs, rand_acts,
        )}],
    )
    raw = response.content[0].text.strip()

    result = {"feature": feature_idx, "raw": raw,
              "top_keys": top_keys, "bottom_keys": bottom_keys, "rand_keys": rand_keys}
    for line in raw.splitlines():
        for k in ("Description", "Label", "Confidence"):
            if line.startswith(f"{k}:"):
                result[k.lower()] = line.split(":", 1)[1].strip()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    nems_path = OUT_DIR / "nems_result.json"
    if not nems_path.exists():
        print(f"ERROR: {nems_path} not found -- run analyze.py first.")
        sys.exit(1)

    with open(nems_path) as f:
        nems_output = json.load(f)

    selected = nems_output["nems"]["selected"]
    if not selected:
        print("No features selected by NEMS -- nothing to interpret.")
        return

    site_data  = np.load(OUT_DIR / "site_features.npz")
    site_feats = site_data["site_features"]   # (N_exp, hidden_dim)
    site_keys  = site_data["site_keys"]        # (N_exp,)
    n_sites    = len(site_keys)

    print(f"Interpreting {len(selected)} feature(s) with {args.model}  "
          f"(k={args.k} per group: top / bottom / random)\n")

    interpretations = []
    for feat_idx in selected:
        acts = site_feats[:, feat_idx]

        sorted_idxs = np.argsort(acts)
        top_idxs    = sorted_idxs[::-1][:args.k]
        bottom_idxs = sorted_idxs[:args.k]
        # random: exclude top and bottom indices
        exclude = set(top_idxs) | set(bottom_idxs)
        pool    = [i for i in range(n_sites) if i not in exclude]
        rand_idxs = rng.choice(pool, size=min(args.k, len(pool)), replace=False)

        print(f"Feature {feat_idx:4d}  "
              f"top={acts[top_idxs[0]]:.2f}..{acts[top_idxs[-1]]:.2f}  "
              f"bottom={acts[bottom_idxs[0]]:.2f}..{acts[bottom_idxs[-1]]:.2f}")

        interp = interpret_feature(
            client, feat_idx,
            site_keys[top_idxs].tolist(),    acts[top_idxs].tolist(),
            site_keys[bottom_idxs].tolist(), acts[bottom_idxs].tolist(),
            site_keys[rand_idxs].tolist(),   acts[rand_idxs].tolist(),
            args.model,
        )
        interpretations.append(interp)
        print(f"  [{interp.get('confidence','?')}] {interp.get('label','?')}: "
              f"{interp.get('description', interp.get('raw',''))[:120]}\n")

    out_path = OUT_DIR / "feature_interpretations.json"
    with open(out_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"Saved -> {out_path}")

    print(f"\n{'Feature':>8}  {'Label':<35}  {'Confidence':>10}")
    print("-" * 58)
    for r in interpretations:
        print(f"{r['feature']:>8}  {r.get('label','?'):<35}  "
              f"{r.get('confidence','?'):>10}")


if __name__ == "__main__":
    main()
