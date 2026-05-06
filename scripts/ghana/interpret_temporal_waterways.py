"""Temporal VLM interpretation for neuron 3821 (ephemeral waterways).

For each of the 6 LEAP communities activated by neuron 3821, shows the VLM
a side-by-side pair (2015 vs 2017 Landsat 8) and asks how the landscape changed
over the two years of the LEAP 1000 cash transfer programme.

Usage:
    python interpret_temporal_waterways.py [--quantize] [--vlm-model MODEL]

Output:
    results/ghana/temporal/neuron_3821_temporal.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT    = Path(__file__).resolve().parents[2]
TIF_DIR_2015 = ROOT / "data" / "ghana" / "satellite" / "tif"
TIF_DIR_2017 = ROOT / "data" / "ghana" / "satellite" / "tif_2017"
OUT_PATH = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_temporal.json"

sys.path.insert(0, str(ROOT))

# Neuron 3821 active communities sorted by activation strength
COMMUNITIES = [
    {"comm_id": 951,  "activation": 2.8516},
    {"comm_id": 675,  "activation": 2.3856},
    {"comm_id": 395,  "activation": 2.0504},
    {"comm_id": 1265, "activation": 1.9522},
    {"comm_id": 655,  "activation": 0.7231},
    {"comm_id": 624,  "activation": 0.6847},
]

TEMPORAL_PROMPT = """\
STUDY CONTEXT
You are interpreting a pair of Landsat 8 satellite images of the same ~5×5 km \
community in rural northern Ghana, taken at two different times during the LEAP 1000 \
cash transfer programme:

  • LEFT IMAGE  — 2015 (baseline, before transfers began)
  • RIGHT IMAGE — 2017 (endline, after two years of transfers)

Each image is a false-colour composite (NIR→Red / Green→Green / SWIR2→Blue):
  • Bright red/magenta = dense healthy vegetation (high NIR)
  • Dark red/maroon    = burn scar or post-fire regrowth
  • Tan/pink           = bare soil, dry fallow, degraded land
  • Dark blue/black    = open water (river, seasonal stream, reservoir)
  • Dark olive         = wetland, seasonally flooded valley bottom
  • Grey/white         = settlement, roads, compacted surfaces

This community was identified by SAE neuron 3821 as having EPHEMERAL WATERWAYS — \
narrow seasonal streams that carry water only during the rainy season. \
These waterways shape agricultural potential: communities near seasonal streams \
can sustain dry-season farming, and cash transfers may amplify this capacity.

TASK: Compare the 2015 image (left) and 2017 image (right) for this community.

Work through these questions in order:

  1. WATERWAY VISIBILITY: Are the seasonal streams/wetland corridors visible in both \
years? Do they appear more or less prominent in 2017? (dark blue/olive features)

  2. AGRICULTURAL EXPANSION: Is there more bare/tan cropland in 2017 vs 2015? \
Are new fields visible near the waterway corridors?

  3. VEGETATION CHANGE: Does the vegetation cover appear denser, sparser, or \
similarly distributed? Focus on areas adjacent to waterways.

  4. SETTLEMENT CHANGE: Any visible growth in grey/white settlement patches?

  5. OVERALL TRAJECTORY: Based on visible changes, does this community appear to \
have expanded agricultural activity near waterways between 2015 and 2017?

Answer in EXACTLY this format (five lines, no extra text):

Waterway change: <one sentence — more/less/similarly visible in 2017>
Agricultural change: <one sentence — expansion/contraction/stable>
Vegetation change: <one sentence>
Settlement change: <one sentence>
Overall: <1-2 sentences summary of whether land use near waterways intensified>
Confidence: <low|medium|high>
"""


def _load_image(tif_path, size=224):
    import rasterio
    from PIL import Image as PILImage
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)

    def norm(b):
        valid = b[b > 0]
        if valid.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(valid, [2, 98])
        out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        out[b <= 0] = 0
        return out

    fc = (np.stack([norm(nir), norm(green), norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return PILImage.fromarray(fc).resize((size, size), PILImage.BICUBIC)


def _load_side_by_side(comm_id, size=336):
    from PIL import Image as PILImage
    name = f"ghana_comm{comm_id:04d}.tif"
    img_2015 = _load_image(TIF_DIR_2015 / name, size)
    img_2017 = _load_image(TIF_DIR_2017 / name, size)
    if img_2015 is None or img_2017 is None:
        return None, img_2015 is not None, img_2017 is not None
    combined = PILImage.new("RGB", (size * 2, size))
    combined.paste(img_2015, (0, 0))
    combined.paste(img_2017, (size, 0))
    return combined, True, True


def _run_vlm(model, processor, image, comm_id, activation):
    content = [
        {"type": "text",  "text": TEMPORAL_PROMPT + f"\n\nCommunity ID: {comm_id}  |  Neuron 3821 activation: {activation:.4f}\n"},
        {"type": "image", "image": image},
    ]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    raw = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    return raw


def _parse_response(raw, comm_id, activation):
    result = {"comm_id": comm_id, "activation": activation, "raw": raw}
    for prefix, field in [
        ("Waterway change",    "waterway_change"),
        ("Agricultural change","agricultural_change"),
        ("Vegetation change",  "vegetation_change"),
        ("Settlement change",  "settlement_change"),
        ("Overall",            "overall"),
        ("Confidence",         "confidence"),
    ]:
        for line in raw.splitlines():
            if line.strip().startswith(f"{prefix}:"):
                result[field] = line.split(":", 1)[1].strip()
                break
    result.setdefault("confidence", "low")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--quantize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check 2017 images exist
    missing = [c["comm_id"] for c in COMMUNITIES
               if not (TIF_DIR_2017 / f"ghana_comm{c['comm_id']:04d}.tif").exists()]
    if missing:
        print(f"ERROR: missing 2017 TIFs for communities: {missing}")
        print(f"Run scripts/ghana/download_waterway_2017.py first.")
        sys.exit(1)

    print(f"Loading VLM: {args.vlm_model} ...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.vlm_model)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype")
        print("  (4-bit quantization)")
    model = AutoModelForImageTextToText.from_pretrained(args.vlm_model, **kwargs)
    model.eval()

    results = []
    for c in COMMUNITIES:
        comm_id, activation = c["comm_id"], c["activation"]
        print(f"\ncomm {comm_id}  (activation={activation:.4f}) ...")
        image, has_2015, has_2017 = _load_side_by_side(comm_id)
        if image is None:
            print(f"  SKIP — 2015={has_2015}, 2017={has_2017}")
            results.append({"comm_id": comm_id, "activation": activation,
                            "error": f"missing 2015={not has_2015} 2017={not has_2017}"})
            continue
        raw = _run_vlm(model, processor, image, comm_id, activation)
        result = _parse_response(raw, comm_id, activation)
        results.append(result)
        print(f"  [{result.get('confidence','?')}] {result.get('overall','')[:120]}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            continue
        print(f"\ncomm {r['comm_id']}  act={r['activation']:.4f}  [{r.get('confidence','?')}]")
        print(f"  Waterway:  {r.get('waterway_change','')}")
        print(f"  Agric:     {r.get('agricultural_change','')}")
        print(f"  Overall:   {r.get('overall','')}")


if __name__ == "__main__":
    main()
