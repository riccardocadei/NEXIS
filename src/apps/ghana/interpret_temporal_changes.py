"""
VLM interpretation of 2015→2017 landscape changes for the 4 most-activated
neuron-3821 communities. Outputs results/ghana/temporal_changes.json.

Format
------
{
  "<comm_id>": {
    "changes": [{"symbol": "+", "label": "cropland", "color": "#c0392b"}, ...],
    "raw": "<VLM output>",
    "model_tag": "..."
  },
  ...
}

Usage:
    python src/apps/ghana/interpret_temporal_changes.py --quantize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
TIF_2015 = SAT_DIR / "tif"
TIF_2017 = SAT_DIR / "tif_2017"
RES_DIR  = ROOT / "results" / "ghana"

import sys
sys.path.insert(0, str(ROOT))
from src.apps.ghana.interpret import load_vlm, _load_composite

# Top-4 communities by neuron-3821 activation (derived from sae_activations.npy)
NEURON = 3821
N_TOP  = 4

CHANGE_COLORS = {
    "cropland":   "#c0392b",
    "vegetation": "#27ae60",
    "bare soil":  "#e67e22",
    "settlement": "#8e44ad",
    "water":      "#2980b9",
    "burn scar":  "#7f8c8d",
}

PROMPT = """\
You are comparing two false-colour Landsat 8 satellite images of the SAME community, \
taken in 2015 (LEFT) and 2017 (RIGHT), approximately 2 years apart.

Image format: false-colour composite — NIR→Red, Green→Green, SWIR2→Blue.
  Bright red/magenta = dense healthy vegetation or crops
  Dark red/maroon    = burn scar or sparse regrowth
  Tan/brown          = bare soil or dry fallow
  Dark blue/black    = open water
  Grey/white         = settlement, roads, compacted ground

TASK: Identify ALL land-cover changes visible between 2015 and 2017.
Focus only on CHANGES — ignore features that look the same in both images.

For each change, state:
  - Direction: "+" (increase / appeared) or "-" (decrease / disappeared)
  - Category: one of [cropland, vegetation, bare soil, settlement, water, burn scar]

Answer in EXACTLY this format (one change per line, then a raw description):
Changes:
+ <category>
- <category>
(list all that apply; omit categories with no change)

Description: <1-2 sentences describing the most salient landscape change>
"""


def load_pair(comm_id: int):
    img_2015 = _load_composite(TIF_2015 / f"ghana_comm{int(comm_id):04d}.tif",
                               size=224, mode="fc")
    img_2017 = _load_composite(TIF_2017 / f"ghana_comm{int(comm_id):04d}.tif",
                               size=224, mode="fc")
    return img_2015, img_2017


def run_vlm_change(model, processor, comm_id: int, img_2015, img_2017) -> dict:
    from PIL import Image as PILImage
    combined = PILImage.new("RGB", (448, 224))
    combined.paste(img_2015, (0, 0))
    combined.paste(img_2017, (224, 0))

    content = [
        {"type": "image", "image": combined},
        {"type": "text",  "text": PROMPT},
    ]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[combined],
        return_tensors="pt", padding=True
    ).to(model.device)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    raw = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

    changes = []
    description = ""
    in_changes = False
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("changes:"):
            in_changes = True
            continue
        if line.lower().startswith("description:"):
            in_changes = False
            description = line.split(":", 1)[1].strip()
            continue
        if in_changes and line and line[0] in ("+", "-"):
            symbol = line[0]
            label  = line[1:].strip().lower().rstrip(".")
            color  = CHANGE_COLORS.get(label, "#444444")
            changes.append({"symbol": symbol, "label": label, "color": color})

    return {"changes": changes, "description": description, "raw": raw}


def get_top_communities(n: int = N_TOP):
    sae_acts = np.load(SAT_DIR / "sae_activations.npy")
    comm_ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    acts     = sae_acts[:, NEURON]
    sorted_idx   = np.argsort(acts)[::-1]
    nonzero_idx  = sorted_idx[acts[sorted_idx] > 0]
    top_idx      = nonzero_idx[:n]
    return [(int(comm_ids[i]), float(acts[i])) for i in top_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_path = RES_DIR / "temporal_changes.json"
    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path} (use --overwrite to redo)")
        return

    communities = get_top_communities()
    print(f"Top-{N_TOP} communities for neuron {NEURON}:")
    for cid, act in communities:
        print(f"  comm {cid:4d}  z={act:.4f}")

    print(f"\nLoading VLM: {args.vlm_model} ...")
    model, processor = load_vlm(args.vlm_model, quantize=args.quantize)

    results = {}
    for comm_id, act in communities:
        img_2015, img_2017 = load_pair(comm_id)
        if img_2015 is None or img_2017 is None:
            print(f"  comm {comm_id}: missing tile(s), skipping")
            continue
        print(f"\n  comm {comm_id} (z={act:.4f}) ...")
        r = run_vlm_change(model, processor, comm_id, img_2015, img_2017)
        r["activation"] = act
        r["model_tag"]  = args.vlm_model
        results[str(comm_id)] = r
        print(f"    changes: {r['changes']}")
        print(f"    {r['description']}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
