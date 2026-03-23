"""
Direct-contrast VLM interpretation of NEMS-selected SAE features.

For each selected feature the VLM receives the top-k and bottom-k activation
images TOGETHER in a single call and is asked to identify what VISUALLY
DISTINGUISHES the two groups.  This avoids the generic per-image description
problem (all images described as "~60% vegetation") and forces the model to
find the actual difference between high- and low-activation sites.

Usage
-----
    python src/interpret.py [options]
    python src/interpret.py --vlm-model Qwen/Qwen2-VL-7B-Instruct --quantize

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

sys.path.insert(0, str(Path(__file__).parent))

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
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"  Loading VLM: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)

    kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("dtype", None)
        print("  (4-bit quantization enabled)")

    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
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
    prior_concepts: list = (),
    max_act: float = 1.0,
) -> dict:
    """Single VLM call: show top and bottom images together, ask for contrast.

    prior_concepts : list of dicts (previous interpretation results), each with
        'feature', 'label', and 'description' keys.  When non-empty, the prompt
        explicitly tells the VLM which visual concepts are already accounted for,
        mirroring the conditional nature of NEMS selection.
    """
    import torch

    content = [{"type": "text", "text":
        f"You are analysing false-colour Landsat satellite imagery of northern Uganda "
        f"(year ~2000, NIR→Red / Green→Green / SWIR→Blue composite).  "
        f"Each tile covers ~5×5 km at ~28 m/pixel.\n\n"
        f"Colour guide for THIS composite:\n"
        f"  • Bright red / magenta   = dense healthy vegetation (high NIR) — "
        f"woodland, gallery forest, vigorous crops\n"
        f"  • Dark red / maroon      = burn scar or sparse regrowth after fire "
        f"(NIR destroyed, some SWIR from char)\n"
        f"  • Tan / pinkish-brown    = bare soil, dry fallow, degraded land\n"
        f"  • Smooth bounded pale patches = cultivated fields (wet-season crops "
        f"slightly greener; dry-season fallow tan)\n"
        f"  • Dark blue / near-black = open water (river, lake, reservoir)\n"
        f"  • Dark olive / very dark = wetland, papyrus swamp, seasonally flooded "
        f"valley bottom (dambo) — distinct from open water by slight olive tone\n"
        f"  • Grey / white           = settlement, metal roofs, bare compacted ground, roads\n"
        f"  • Bright pale mauve/white outcrops = laterite hardpan (infertile)\n\n"
        f"IMPORTANT — nearly every tile in this savanna region contains red/pink "
        f"vegetation.  Overall redness is NOT a useful distinguishing feature.  "
        f"Focus on STRUCTURAL and LAND-COVER differences below.\n\n"
        f"Group A — HIGH activation ({len(top_imgs)} images):\n"
    }]
    for i, img in enumerate(top_imgs):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[A{i+1}] "})

    content.append({"type": "text", "text":
        f"\n\nGroup B — LOW activation ({len(bottom_imgs)} images):\n"
    })
    for i, img in enumerate(bottom_imgs):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[B{i+1}] "})

    # Build the task text, injecting prior-concept exclusions when applicable.
    if prior_concepts:
        prior_lines = "\n".join(
            f"  • Feature {p['feature']} — \"{p.get('label', 'unknown')}\""
            + (f": {p['description']}" if p.get('description') else "")
            for p in prior_concepts
        )
        exclusion_block = (
            f"\n\nBANNED CONCEPTS (already identified in earlier features — "
            f"do NOT repeat or paraphrase these):\n"
            f"{prior_lines}\n\n"
            f"By statistical construction this feature carries information that the "
            f"banned concepts above do NOT explain.  Your label and description MUST "
            f"name something structurally different from every banned concept.  "
            f"If the only visible difference matches a banned concept, write "
            f"\"indistinguishable from prior concept\" rather than inventing a synonym."
        )
    else:
        exclusion_block = ""

    weak_signal_note = (
        f"\nNOTE: This feature has WEAK activation (max={max_act:.4f}).  "
        f"The between-group contrast will be subtle.  Use this two-step strategy:\n"
        f"  Step 1 — Study Group A images ONLY.  What do they share among themselves?  "
        f"Look for any recurring micro-pattern, texture, or land-cover element that "
        f"appears in most Group A tiles, however faint.\n"
        f"  Step 2 — Check whether Group B tiles consistently LACK that element.  "
        f"If yes, that is the concept.  If Group B also has it, move to the next cue.\n"
        f"Set Confidence: low if nothing distinguishes the groups after this search.\n"
    ) if max_act < 0.05 else ""

    content.append({"type": "text", "text":
        f"{exclusion_block}\n\n"
        f"{weak_signal_note}"
        f"TASK: Identify what makes Group A systematically different from Group B.\n"
        f"The difference may be PRESENCE of something in A that B lacks, OR ABSENCE "
        f"of something in A that B has — both are valid interpretations.\n\n"
        f"Work through these land-cover cues in order; stop at the FIRST clear asymmetry:\n\n"
        f"  1. OPEN WATER / WETLAND\n"
        f"     Dark blue-black = river, lake, reservoir.  Dark olive = dambo wetland, "
        f"papyrus swamp, seasonally flooded valley bottom.\n"
        f"     Economic link: water access drives irrigation, fishing, domestic welfare; "
        f"wetland plots (kibanja) are key post-conflict income buffers in Acholi/Lango.\n\n"
        f"  2. BURN SCARS\n"
        f"     Dark red/maroon patches (NOT open water).  Sharp-edged = recent human-set "
        f"fire; diffuse = older scar.\n"
        f"     Economic link: burning signals active land reoccupation and seasonal field "
        f"preparation — a key indicator of post-conflict return stage in northern Uganda.\n\n"
        f"  3. BARE SOIL / CROPLAND EXTENT AND GEOMETRY\n"
        f"     Tan-pink patches.  Are they geometrically regular (large rectangular "
        f"fields = commercial/mechanised) or small and irregular (subsistence smallholder)?  "
        f"More bare soil in one group?\n"
        f"     Economic link: field size and regularity proxy capital, tenure security, "
        f"and market integration.\n\n"
        f"  4. SETTLEMENT AND ROAD INFRASTRUCTURE\n"
        f"     Grey/white compact clusters = villages, trading centres.  Pale linear "
        f"features = roads and tracks.  Denser or more connected in one group?\n"
        f"     Economic link: road proximity is the strongest infrastructure predictor "
        f"of poverty reduction in rural sub-Saharan Africa.\n\n"
        f"  5. DEGRADED / OVERGRAZED LAND\n"
        f"     Persistently pale tan — no seasonal green pulse — near former IDP camp "
        f"zones or peri-urban areas.\n"
        f"     Economic link: degradation proxies displaced population pressure and "
        f"lower long-run agricultural productivity.\n\n"
        f"  6. VEGETATION SPATIAL STRUCTURE (only if 1–5 are symmetric)\n"
        f"     Continuous dense canopy (vivid solid red) vs. fragmented mosaic vs. "
        f"fine-grained speckled patchwork.  Texture and fragmentation matter; "
        f"overall redness does NOT.\n\n"
        f"  7. If nothing structurally differs, state that explicitly.\n\n"
        f"Answer in this EXACT format (four lines, no extra text):\n"
        f"Group A concept: <what Group A HAS or LACKS — cite the cue number and be specific>\n"
        f"Group B contrast: <one sentence from Group B's perspective>\n"
        f"Label: <2–6 words naming the concept, e.g. 'riparian wetland corridor' or "
        f"'no perennial water source' — NEVER the bare word 'absent' alone>\n"
        f"Confidence: <low|medium|high>"
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
    key_map = {
        "Group A concept": "activated_concept",    # what the neuron fires on
        "Group B contrast": "not_activated_concept",  # what is present when the neuron is silent
        "Label": "label",
        "Confidence": "confidence",
    }
    for line in raw.splitlines():
        for prefix, field in key_map.items():
            if line.strip().startswith(f"{prefix}:"):
                result[field] = line.split(":", 1)[1].strip()
    if "label" not in result:
        result["label"] = "unknown"
    if "confidence" not in result:
        result["confidence"] = "low"
    return result


# ── Per-outcome interpretation ─────────────────────────────────────────────────

def interpret_outcome(
    outcome: str,
    model_dir: Path,
    site_feats: "np.ndarray",
    site_keys: "np.ndarray",
    vlm_model, vlm_processor,
    k: int,
    min_activation: float,
    sae_dim: int,
    vlm_model_name: str = "",
) -> bool:
    """Interpret all NEMS-selected SAE features for one outcome.

    Returns True if interpretations were written, False if skipped.
    VLM is already loaded by the caller — this function does not load or unload it.
    """
    out_dir   = model_dir / outcome
    out_dir.mkdir(parents=True, exist_ok=True)
    nems_path = out_dir / "nems_result.json"

    if not nems_path.exists():
        print(f"  [{outcome}] SKIP — nems_result.json not found (run analyze.py first)")
        return False

    with open(nems_path) as f:
        nems_output = json.load(f)

    n_sae            = nems_output.get("feature_meta", {}).get("n_sae_features", sae_dim)
    selected_entries = nems_output["nems"]["selected"]
    if selected_entries and isinstance(selected_entries[0], dict):
        sae_entries  = [e for e in selected_entries if e["group"] != "W"]
        skip_entries = [e for e in selected_entries if e["group"] == "W"]
        selected     = [e["idx"] for e in sae_entries]
        if skip_entries:
            print(f"  [{outcome}] Note: W-covariate features skipped (not image features): "
                  f"{[e['label'] for e in skip_entries]}")
    else:
        selected = [i for i in selected_entries if i < n_sae]

    if not selected:
        print(f"  [{outcome}] No SAE features selected — nothing to interpret.")
        return False

    print(f"  [{outcome}] {len(selected)} feature(s) to interpret")

    # Build image groups for each selected feature
    feature_groups = []
    for feat_idx in selected:
        acts        = site_feats[:, feat_idx]
        sorted_idxs = np.argsort(acts)
        top_idxs    = sorted_idxs[::-1][:k]
        bottom_idxs = sorted_idxs[:k]
        feature_groups.append({
            "feat_idx":    feat_idx,
            "top_keys":    site_keys[top_idxs].tolist(),
            "top_acts":    acts[top_idxs].tolist(),
            "bottom_keys": site_keys[bottom_idxs].tolist(),
            "bottom_acts": acts[bottom_idxs].tolist(),
        })

    # VLM contrast — prior_concepts accumulate within this outcome only
    interpretations = []
    for fg in feature_groups:
        feat_idx = fg["feat_idx"]
        print(f"    Feature {feat_idx:4d}  "
              f"top={fg['top_acts'][0]:.3f}..{fg['top_acts'][-1]:.3f}  "
              f"bottom={fg['bottom_acts'][0]:.3f}..{fg['bottom_acts'][-1]:.3f}")

        top_imgs, top_keys, top_acts       = load_group(fg["top_keys"],   fg["top_acts"])
        bottom_imgs, bottom_keys, bot_acts = load_group(fg["bottom_keys"], fg["bottom_acts"])

        if not top_imgs or not bottom_imgs:
            print("      (skipped: could not load images)")
            continue

        max_act = fg["top_acts"][0]
        if max_act < min_activation:
            print(f"      (skipped: max activation {max_act:.4f} < threshold {min_activation})")
            interpretations.append({
                "feature": feat_idx,
                "label": "low activation — uninterpretable",
                "activated_concept": f"Max activation {max_act:.4f} below threshold.",
                "not_activated_concept": "",
                "confidence": "low",
                "vlm_model": vlm_model_name,
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
            prior_concepts=interpretations,
            max_act=max_act,
        )
        result["vlm_model"] = vlm_model_name
        interpretations.append(result)
        print(f"      [{result.get('confidence','?')}] {result.get('label','?')}:\n"
              f"        Fires on: {result.get('activated_concept', result.get('raw',''))[:140]}"
              + (f"\n        Absent:   {result['not_activated_concept'][:120]}"
                 if result.get('not_activated_concept') else "")
              + "\n")

    out_path = out_dir / "interpretations.json"
    with open(out_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"  [{outcome}] Saved -> {out_path}")

    print(f"  {'Feature':>8}  {'Label':<35}  {'Confidence':>10}")
    print("  " + "-" * 58)
    for r in interpretations:
        print(f"  {r['feature']:>8}  {r.get('label','?'):<35}  {r.get('confidence','?'):>10}")
    print()
    return True


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VLM interpretation of NEMS-selected SAE features. "
                    "The VLM is loaded once and reused across all requested outcomes."
    )
    p.add_argument("--embed-model",  default="dinov2_vitb14",
                   help="Vision backbone name (determines results subdir).")
    p.add_argument("--sae-dim",      type=int, default=3072,
                   help="SAE hidden dimension (determines results subdir).")
    p.add_argument("--k",            type=int, default=10,
                   help="Images per group shown to VLM (top / bottom).")
    p.add_argument("--min-activation", type=float, default=0.001,
                   help="Skip features whose max activation is below this threshold.")
    p.add_argument("--vlm-model",    default="Qwen/Qwen2-VL-7B-Instruct",
                   help="HuggingFace model ID for the vision-language model.")
    p.add_argument("--quantize", action="store_true",
                   help="Load model in 4-bit (requires bitsandbytes). ~4 GB vs ~14 GB VRAM.")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--outcome",      default=None,
                   help="Single outcome alias (legacy; use --outcomes for multiple).")
    p.add_argument("--outcomes",     default=None,
                   help="Comma-separated outcome aliases to process in one VLM session.")
    p.add_argument("--overwrite",    action="store_true",
                   help="Re-run even if interpretations.json already exists.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.default_rng(args.seed)

    MODEL_DIR = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"

    # Resolve outcome list (--outcomes takes priority; --outcome kept for compat)
    if args.outcomes:
        outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    elif args.outcome:
        outcomes = [args.outcome]
    else:
        print("ERROR: specify --outcomes=o1,o2,... or --outcome=o"); sys.exit(1)
    # Keep alias names — directories are named after aliases, not CSV column names

    # Skip outcomes whose output already exists unless --overwrite
    to_run = []
    for outcome in outcomes:
        out_path = MODEL_DIR / outcome / "interpretations.json"
        if not args.overwrite and out_path.exists():
            print(f"  [{outcome}] Skipping (interpretations.json exists; use --overwrite to redo)")
        else:
            to_run.append(outcome)

    if not to_run:
        print("All outcomes already interpreted.")
        return

    # Load shared site features (same file for all outcomes under this model)
    site_data  = np.load(MODEL_DIR / "site_features.npz")
    site_feats = site_data["site_features"]
    site_keys  = site_data["site_keys"]

    # Load VLM once for all outcomes
    print(f"\nLoading VLM: {args.vlm_model}  (will be reused for {len(to_run)} outcome(s))")
    vlm_model, vlm_processor = load_vlm(args.vlm_model, quantize=args.quantize)
    print()

    for outcome in to_run:
        print(f"── {outcome} {'─' * max(0, 55 - len(outcome))}")
        interpret_outcome(
            outcome, MODEL_DIR, site_feats, site_keys,
            vlm_model, vlm_processor,
            k=args.k, min_activation=args.min_activation, sae_dim=args.sae_dim,
            vlm_model_name=args.vlm_model,
        )

    unload_model(vlm_model)


if __name__ == "__main__":
    main()
