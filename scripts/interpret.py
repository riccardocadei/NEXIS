"""
Direct-contrast VLM interpretation of NEMS-selected SAE features.

For each selected feature the VLM receives the top-k and bottom-k activation
images TOGETHER in a single call and is asked to identify what VISUALLY
DISTINGUISHES the two groups.  This avoids the generic per-image description
problem (all images described as "~60% vegetation") and forces the model to
find the actual difference between high- and low-activation sites.

Usage
-----
    python scripts/interpret.py [options]
    python scripts/interpret.py --vlm-model Qwen/Qwen2-VL-7B-Instruct --quantize

Requires:  pip install transformers accelerate bitsandbytes torch pillow
           (no external API key needed – model runs locally on GPU)
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT    = Path(__file__).parent.parent
IMG_DIR = ROOT / "data" / "uganda" / "Uganda2000_processed"

# ── Image loading ──────────────────────────────────────────────────────────────

_FILL = 5000
_PCT_LO, _PCT_HI = 2, 98


def load_image(key: int, img_dir: Path) -> "np.ndarray | None":
    """Load false-color (NIR→R, Green→G, SWIR→B) array (H,W,3) float32 in [0,1]."""
    import pandas as _pd
    raw = {}
    for b in [1, 2, 3]:
        p = Path(img_dir) / f"GeoKey{key}_BAND{b}.csv"
        if not p.exists():
            return None
        arr = _pd.read_csv(p, header=None).values.astype(np.float32)
        arr[arr >= _FILL] = np.nan
        raw[b] = arr
    out = []
    for b in [2, 1, 3]:          # NIR→R, Green→G, SWIR→B
        ch = raw[b]
        valid = ch[~np.isnan(ch)]
        lo = np.percentile(valid, _PCT_LO)
        hi = np.percentile(valid, _PCT_HI)
        s = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        s[np.isnan(ch)] = 0.0
        out.append(s)
    return np.stack(out, axis=-1)


def load_site_image(key: int, size: int = 224) -> "Image.Image | None":
    arr = load_image(key, IMG_DIR)
    if arr is None:
        return None
    img = Image.fromarray((arr * 255).astype("uint8"))
    return img.resize((size, size), Image.BICUBIC)


def load_group(keys, activations):
    imgs, valid_keys, valid_acts = [], [], []
    for key, act in zip(keys, activations):
        img = load_site_image(int(key))
        if img is not None:
            imgs.append(img)
            valid_keys.append(int(key))
            valid_acts.append(float(act))
    return imgs, valid_keys, valid_acts


# ── Model loading ──────────────────────────────────────────────────────────────

def load_vlm(model_name: str, quantize: bool = False):
    """Load Qwen2-VL (or compatible) VLM model + processor."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"  Loading VLM: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)

    kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("dtype", None)
        print("  (4-bit quantization enabled)")

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, processor


def unload_model(model):
    """Free GPU memory after the VLM stage."""
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ── VLM stage: direct multi-image contrast ────────────────────────────────────
#
# Instead of describing each image individually (which produces generic
# "vegetation dominates" text for every image), we show the top-k and bottom-k
# images TOGETHER in a single VLM call and ask what DISTINGUISHES the two groups.
# This forces the model to find differences rather than a generic description,
# reduces calls from O(k * n_features) to O(n_features), and leverages the
# VLM's ability to do in-context visual comparison.

STUDY_CONTEXT = """\
STUDY CONTEXT
You are helping interpret a feature neuron from a Sparse Autoencoder (SAE) \
trained on DINOv2 patch embeddings of Landsat satellite imagery.

The images come from the Youth Opportunities Programme (YOP) evaluation — \
a randomised controlled trial in northern Uganda in which small teams of \
young adults competed for one-time business grants (~USD 7,500) to pursue \
skilled trades. The PRIMARY OUTCOME is skilled-labour hours two years after \
the grant. The satellite images were captured circa 2000, EIGHT YEARS BEFORE \
the intervention, and represent the PRE-TREATMENT landscape of the sites.

We are running NEMS (Neural Effect Modifier Selection): a forward stepwise \
procedure that tests which SAE features statistically modify the programme's \
average treatment effect. The neuron you are interpreting has been selected \
because its activation pattern correlates with heterogeneous treatment effects \
across sites — meaning the programme works DIFFERENTLY depending on the \
landscape this neuron detects.

IMAGE FORMAT
False-colour Landsat ETM+ composites, ~5 × 5 km footprint per tile.
· Bright red / pink  = healthy dense vegetation (high NIR reflectance)
· Tan / brown        = bare or degraded soil, low vegetation, dry season crops
· Dark blue / black  = open water (lakes, rivers, wetlands)
· Grey / cyan        = settlements, roads, built-up impervious surfaces
Resolution: ~28 m/pixel. Images are 224 × 224 px (≈ 6 km²).

ECONOMICALLY RELEVANT VISUAL PROPERTIES TO CONSIDER
The following landscape features have known links to rural programme outcomes:
· Water access (rivers, lakes, wetlands) — irrigation, livestock, fishing
· Road / path networks (linear grey features) — market access, transport
· Settlement clusters — population density, social capital, service access
· Agricultural field geometry (regular cleared patches) — commercial farming
· Forest / dense vegetation cover — timber, charcoal, non-farm income
· Bare soil / degraded land — land pressure, soil quality, erosion
· Karamoja region signature (sparse scrub, semi-arid) — agro-pastoral zone\
"""

CONTRAST_PROMPT_TEMPLATE = STUDY_CONTEXT + """

TASK
I will show you two groups of satellite images for SAE feature #{feat_idx}.
The neuron activates STRONGLY on Group A (activations {act_hi}) and is \
NEARLY SILENT on Group B (activations {act_lo}).

Group A — HIGH activation ({n_top} images):
{group_a_images}

Group B — LOW activation ({n_bot} images):
{group_b_images}

Compare the two groups carefully. Identify the ONE visual property of the \
landscape that best explains why the neuron fires on Group A but not Group B.
· Focus exclusively on DIFFERENCES — not on what both groups have in common.
· "Both have vegetation" is NOT a useful answer; vegetation is everywhere.
· Think about texture, spatial pattern, specific land-use signatures, \
  infrastructure, water features, or settlement indicators.
· If the images look identical and no difference is visible, say so.

Answer in this EXACT format (three lines, no extra text):
Description: <one sentence naming the distinguishing visual property>
Label: <2–5 word label for this concept>
Confidence: <low|medium|high>"""


def contrast_groups_vlm(
    model, processor,
    feat_idx: int,
    top_imgs: list, top_keys: list, top_acts: list,
    bottom_imgs: list, bottom_keys: list, bottom_acts: list,
) -> dict:
    """Single VLM call: show top and bottom images together, ask for contrast."""
    import torch

    # Build interleaved content: text header → images A1…Ak → text header → images B1…Bk
    content: list = []

    # Inline image placeholders will be filled by processor; we just embed
    # {"type": "image", "image": img} entries in order.
    def _act_range(acts):
        return f"{max(acts):.3f}…{min(acts):.3f}"

    group_a_placeholder = "".join(f"[A{i+1}] " for i in range(len(top_imgs)))
    group_b_placeholder = "".join(f"[B{i+1}] " for i in range(len(bottom_imgs)))

    intro = CONTRAST_PROMPT_TEMPLATE.format(
        feat_idx=feat_idx,
        n_top=len(top_imgs), act_hi=_act_range(top_acts),
        n_bot=len(bottom_imgs), act_lo=_act_range(bottom_acts),
        group_a_images=group_a_placeholder,
        group_b_images=group_b_placeholder,
    )

    # Reconstruct with actual image tokens interleaved
    content = [{"type": "text", "text":
        f"I am analysing SAE feature #{feat_idx} extracted from a vision model trained "
        f"on false-colour Landsat satellite imagery of northern Uganda (~year 2000).\n\n"
        f"Colour key: bright red/pink = healthy vegetation · tan/brown = bare soil "
        f"· dark blue/black = water · grey/cyan = settlements.\n\n"
        f"Group A — HIGH activation ({len(top_imgs)} images, "
        f"activations {_act_range(top_acts)}):\n"
    }]
    for i, (img, act) in enumerate(zip(top_imgs, top_acts)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[A{i+1} act={act:.3f}] "})

    content.append({"type": "text", "text":
        f"\n\nGroup B — LOW activation ({len(bottom_imgs)} images, "
        f"activations {_act_range(bottom_acts)}):\n"
    })
    for i, (img, act) in enumerate(zip(bottom_imgs, bottom_acts)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[B{i+1} act={act:.3f}] "})

    content.append({"type": "text", "text":
        "\n\nYour task: identify what SPECIFIC visual property causes the feature "
        "to fire on Group A but not Group B. Focus exclusively on DIFFERENCES. "
        "Ignore properties common to both groups. Consider: water bodies, "
        "settlement clusters, road/path networks, agricultural patterns, "
        "soil exposure, vegetation density or fragmentation, burned areas, "
        "or distinctive texture.\n\n"
        "Answer in this EXACT format (no extra text):\n"
        "Description: <one sentence on what distinguishes Group A from Group B>\n"
        "Label: <2–5 word label for this visual concept>\n"
        "Confidence: <low|medium|high>"
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    all_imgs = list(top_imgs) + list(bottom_imgs)
    inputs = processor(
        text=[text], images=all_imgs, return_tensors="pt", padding=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=True).strip()

    result = {
        "feature":     feat_idx,
        "raw":         raw,
        "top_keys":    list(top_keys),
        "bottom_keys": list(bottom_keys),
        "top_acts":    list(top_acts),
        "bottom_acts": list(bottom_acts),
    }
    for line in raw.splitlines():
        for k in ("Description", "Label", "Confidence"):
            if line.strip().startswith(f"{k}:"):
                result[k.lower()] = line.split(":", 1)[1].strip()
    if "label" not in result:
        result["label"] = "unknown"
    if "confidence" not in result:
        result["confidence"] = "low"
    return result


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage VLM→LLM interpretation of NEMS-selected SAE features."
    )
    p.add_argument("--embed-model",  default="dinov2_vitb14",
                   help="Vision backbone name (determines results subdir).")
    p.add_argument("--sae-dim",      type=int, default=3072,
                   help="SAE hidden dimension (determines results subdir).")
    p.add_argument("--k",            type=int, default=6,
                   help="Images per group shown to VLM (top / bottom). "
                        "6 gives good signal diversity; reduce if GPU OOMs.")
    p.add_argument("--min-activation", type=float, default=0.01,
                   help="Skip VLM interpretation if max top-activation is below "
                        "this threshold (feature barely fires, signal unreliable).")
    p.add_argument("--vlm-model",    default="Qwen/Qwen2-VL-7B-Instruct",
                   help="HuggingFace model ID for the vision-language model "
                        "(stage 1: image description).")
    p.add_argument("--quantize", action="store_true",
                   help="Load models in 4-bit (requires bitsandbytes). "
                        "Reduces VRAM from ~14 GB to ~4 GB for a 7B model.")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    OUT_DIR = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"

    nems_path = OUT_DIR / "nems_result.json"
    if not nems_path.exists():
        print(f"ERROR: {nems_path} not found -- run analyze.py first.")
        sys.exit(1)

    with open(nems_path) as f:
        nems_output = json.load(f)

    n_sae = nems_output.get("feature_meta", {}).get("n_sae_features", args.sae_dim)
    selected_entries = nems_output["nems"]["selected"]
    if selected_entries and isinstance(selected_entries[0], dict):
        sae_entries  = [e for e in selected_entries if e["group"] != "W"]
        skip_entries = [e for e in selected_entries if e["group"] == "W"]
        selected     = [e["idx"] for e in sae_entries]
        if skip_entries:
            print(f"Note: W-covariate features selected (not image features, "
                  f"skipping VLM stage): {[e['label'] for e in skip_entries]}")
    else:
        selected = [i for i in selected_entries if i < n_sae]
        skipped  = [i for i in selected_entries if i >= n_sae]
        if skipped:
            print(f"Skipping W-covariate features (indices): {skipped}")

    if not selected:
        print("No SAE features selected by NEMS -- nothing to interpret.")
        return

    site_data  = np.load(OUT_DIR / "site_features.npz")
    site_feats = site_data["site_features"]   # (N_sites, hidden_dim)
    site_keys  = site_data["site_keys"]        # (N_sites,)
    n_sites    = len(site_keys)

    print(f"Interpreting {len(selected)} SAE feature(s) "
          f"(k={args.k} per group: top / bottom)")
    print(f"  VLM : {args.vlm_model}  (direct top-vs-bottom contrast)")
    print()

    # ── Collect image groups per feature ──────────────────────────────────────
    feature_groups = []
    for feat_idx in selected:
        acts = site_feats[:, feat_idx]
        sorted_idxs = np.argsort(acts)
        top_idxs    = sorted_idxs[::-1][:args.k]
        bottom_idxs = sorted_idxs[:args.k]
        feature_groups.append({
            "feat_idx":    feat_idx,
            "top_keys":    site_keys[top_idxs].tolist(),
            "top_acts":    acts[top_idxs].tolist(),
            "bottom_keys": site_keys[bottom_idxs].tolist(),
            "bottom_acts": acts[bottom_idxs].tolist(),
        })

    # ── VLM direct contrast (one call per feature) ────────────────────────────
    print("── VLM direct contrast ──────────────────────────────────────────────")
    vlm_model, vlm_processor = load_vlm(args.vlm_model, quantize=args.quantize)

    interpretations = []
    for fg in feature_groups:
        feat_idx = fg["feat_idx"]
        print(f"  Feature {feat_idx:4d}  "
              f"top={fg['top_acts'][0]:.3f}..{fg['top_acts'][-1]:.3f}  "
              f"bottom={fg['bottom_acts'][0]:.3f}..{fg['bottom_acts'][-1]:.3f}")

        top_imgs, top_keys, top_acts       = load_group(fg["top_keys"],   fg["top_acts"])
        bottom_imgs, bottom_keys, bot_acts = load_group(fg["bottom_keys"], fg["bottom_acts"])

        if not top_imgs or not bottom_imgs:
            print("    (skipped: could not load images)")
            continue

        max_act = fg["top_acts"][0]
        if max_act < args.min_activation:
            print(f"    (skipped: max activation {max_act:.4f} < "
                  f"threshold {args.min_activation} — signal too weak for VLM)")
            interpretations.append({
                "feature": feat_idx,
                "label": "low activation — uninterpretable",
                "description": f"Max activation {max_act:.4f} is below threshold; "
                               f"top and bottom images are visually indistinguishable.",
                "confidence": "low",
                "raw": "",
                "top_keys": fg["top_keys"], "bottom_keys": fg["bottom_keys"],
                "top_acts": fg["top_acts"], "bottom_acts": fg["bottom_acts"],
            })
            continue

        result = contrast_groups_vlm(
            vlm_model, vlm_processor,
            feat_idx,
            top_imgs, top_keys, top_acts,
            bottom_imgs, bottom_keys, bot_acts,
        )
        interpretations.append(result)
        print(f"    [{result.get('confidence','?')}] {result.get('label','?')}: "
              f"{result.get('description', result.get('raw',''))[:160]}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "interpretations.json"
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
