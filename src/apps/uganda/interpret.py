"""
Direct-contrast VLM interpretation of NEXIS-selected SAE features.

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

ROOT    = Path(__file__).parent.parent.parent.parent   # repo root
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

def load_vlm(model_name: str, quantize: bool = False, trust_remote_code: bool = False):
    """Load Qwen2-VL (or compatible) VLM model + processor."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"  Loading VLM: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    kwargs = dict(dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code)
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


# ── GeoChat pipeline — Stage 1: per-image RS captioning ───────────────────────

_DESCRIPTIONS_DIR = ROOT / "data" / "uganda"

GEOCHAT_DESCRIBE_PROMPT = (
    "This is a false-colour Landsat ETM+ pan-sharpened satellite image of northern Uganda "
    "(year ~2000). Band mapping: NIR→Red channel, Green→Green channel, SWIR→Blue channel. "
    "The tile covers approximately 5×5 km at ~14 m/pixel native resolution.\n\n"
    "Colour guide:\n"
    "• Bright red/magenta = dense healthy vegetation (woodland, vigorous crops)\n"
    "• Dark red/maroon = burn scar or sparse post-fire regrowth\n"
    "• Tan/pinkish-brown = bare soil, dry fallow, degraded land\n"
    "• Smooth bounded pale patches = cultivated fields\n"
    "• Dark blue/near-black = open water (river, lake, reservoir)\n"
    "• Dark olive = wetland, papyrus swamp, seasonally flooded valley bottom\n"
    "• Grey/white = settlement, roads, compacted ground\n"
    "• Pale mauve/white outcrops = laterite hardpan (infertile)\n\n"
    "Describe the land cover in this image in 2–4 sentences. Focus on: "
    "water bodies, vegetation density and pattern, settlements, roads, "
    "bare or degraded soil, burn scars, and crop field geometry. Be specific and factual."
)


def _desc_cache_path(geochat_model_name: str) -> Path:
    slug = geochat_model_name.replace("/", "_").replace("-", "_")
    return _DESCRIPTIONS_DIR / f"site_descriptions_{slug}.json"


def _save_desc_cache(cache_path: Path, model_name: str, descriptions: dict) -> None:
    with open(cache_path, "w") as f:
        json.dump({"model": model_name, "descriptions": descriptions}, f)


def _remap_geochat_state_dict(state_dict: dict) -> dict:
    """
    Remap GeoChat (old LLaVA 1.5) weight keys to LlavaForConditionalGeneration format.

    GeoChat was trained with the original LLaVA codebase (transformers 4.31) whose
    weight layout differs from the transformers-integrated LlavaForConditionalGeneration:

      model.embed_tokens.*            -> model.language_model.embed_tokens.*
      model.layers.*                  -> model.language_model.layers.*
      model.norm.*                    -> model.language_model.norm.*
      lm_head.*                       -> lm_head.*  (unchanged)
      model.mm_projector.0.*          -> model.multi_modal_projector.linear_1.*
      model.mm_projector.2.*          -> model.multi_modal_projector.linear_2.*
      model.vision_tower.vision_tower.* -> model.vision_tower.*
      model.*.rotary_emb.inv_freq     -> dropped (computed on-the-fly in modern transformers)
    """
    new_sd = {}
    for key, val in state_dict.items():
        if "rotary_emb.inv_freq" in key:
            continue  # not present in modern transformers LLaMA
        if key.startswith("model.mm_projector."):
            rest = key[len("model.mm_projector."):]
            idx_str, _, param = rest.partition(".")
            linear_n = int(idx_str) // 2 + 1  # 0 -> linear_1, 2 -> linear_2
            new_key = f"model.multi_modal_projector.linear_{linear_n}.{param}"
        elif key.startswith("model.vision_tower.vision_tower."):
            new_key = "model.vision_tower." + key[len("model.vision_tower.vision_tower."):]
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model."):]
        else:
            new_key = key  # lm_head.* unchanged
        new_sd[new_key] = val
    return new_sd


def _get_geochat_remapped_dir(model_name: str) -> Path:
    """
    Return a persistent local directory holding LlavaForConditionalGeneration-compatible
    weights remapped from GeoChat.  Builds it on first call (downloads ~13 GB once).
    """
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import LlavaConfig, LlamaConfig, CLIPVisionConfig

    # Cache alongside this script so we only remap once
    cache_dir = ROOT / "models" / "geochat_remapped"
    sentinel = cache_dir / "config.json"
    if sentinel.exists():
        return cache_dir

    print("  Remapping GeoChat weights to LlavaForConditionalGeneration format "
          "(one-time setup, ~13 GB download) ...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- load original config ---
    cfg_path = hf_hub_download(model_name, "config.json")
    with open(cfg_path) as f:
        gc = json.load(f)

    vision_tower = gc.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")

    # --- build LlavaConfig from flat GeoChat config ---
    text_cfg = LlamaConfig(
        hidden_size=gc["hidden_size"],
        num_hidden_layers=gc["num_hidden_layers"],
        num_attention_heads=gc["num_attention_heads"],
        num_key_value_heads=gc.get("num_key_value_heads", gc["num_attention_heads"]),
        intermediate_size=gc["intermediate_size"],
        rms_norm_eps=gc.get("rms_norm_eps", 1e-5),
        vocab_size=gc["vocab_size"],
        max_position_embeddings=gc.get("max_position_embeddings", 4096),
        pretraining_tp=gc.get("pretraining_tp", 1),
    )
    vision_cfg = CLIPVisionConfig.from_pretrained(vision_tower)
    llava_config = LlavaConfig(text_config=text_cfg, vision_config=vision_cfg)
    llava_config.save_pretrained(str(cache_dir))

    # --- download + remap weights ---
    index_path = hf_hub_download(model_name, "pytorch_model.bin.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_files = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for i, fname in enumerate(weight_files, 1):
        print(f"    Downloading shard {i}/{len(weight_files)}: {fname}")
        fpath = hf_hub_download(model_name, fname)
        shard = torch.load(fpath, map_location="cpu", weights_only=True)
        state_dict.update(shard)

    state_dict = _remap_geochat_state_dict(state_dict)

    # --- validate: every expected key should be present ---
    with torch.device("meta"):
        dummy = __import__("transformers").LlavaForConditionalGeneration(llava_config)
    expected = set(dummy.state_dict().keys())
    missing = expected - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - expected
    if missing:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(
            f"GeoChat weight remapping failed: {len(missing)} keys missing from "
            f"remapped dict. Examples: {sorted(missing)[:5]}\n"
            f"Unexpected keys: {sorted(unexpected)[:5]}"
        )

    print(f"  Saving remapped weights ({len(state_dict)} keys) ...")
    from safetensors.torch import save_file
    save_file(state_dict, str(cache_dir / "model.safetensors"))
    return cache_dir


def load_geochat(model_name: str = "MBZUAI/geochat-7b", quantize: bool = False):
    """Load GeoChat (LLaVA-based) for single-image remote-sensing description."""
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import (LlavaProcessor, LlavaForConditionalGeneration,
                              CLIPImageProcessor, AutoTokenizer)
    print(f"  Loading GeoChat: {model_name} ...")

    # Build processor from parts (GeoChat has no preprocessor_config.json)
    cfg_path = hf_hub_download(model_name, "config.json")
    with open(cfg_path) as f:
        vision_tower = json.load(f).get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # patch_size must be set explicitly — GeoChat has no preprocessor_config.json
    # so LlavaProcessor defaults to None, causing "int // NoneType" crash at inference.
    patch_size = getattr(image_processor, "patch_size", None) or 14
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer,
                               patch_size=patch_size)

    # Load remapped weights (downloaded + remapped once, then cached locally)
    remapped_dir = _get_geochat_remapped_dir(model_name)

    kwargs = dict(dtype=torch.float16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)
        print("  (4-bit quantization enabled)")

    model = LlavaForConditionalGeneration.from_pretrained(str(remapped_dir), **kwargs)
    model.eval()
    return model, processor


def describe_image_geochat(model, processor, img: "Image.Image") -> str:
    """Return a 2–4 sentence land-cover description for one RS image via GeoChat."""
    import torch
    prompt = f"USER: <image>\n{GEOCHAT_DESCRIBE_PROMPT}\nASSISTANT:"
    # Build inputs manually to bypass the newer transformers image-token count
    # validation, which mismatches GeoChat's single-<image>-token convention.
    pixel_values = processor.image_processor(
        images=img, return_tensors="pt"
    )["pixel_values"].to(model.device)
    text_inputs = processor.tokenizer(
        prompt, return_tensors="pt", truncation=False
    ).to(model.device)
    inputs = {**text_inputs, "pixel_values": pixel_values}
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    generated = out_ids[0][text_inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_or_build_descriptions(
    geochat_model, geochat_processor,
    all_keys: list,
    model_name: str,
    force_rebuild: bool = False,
) -> dict:
    """Return {str(key): description} for all keys, building missing entries with GeoChat."""
    cache_path = _desc_cache_path(model_name)
    cache: dict = {}
    if cache_path.exists() and not force_rebuild:
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("model") == model_name:
            cache = data.get("descriptions", {})
            print(f"  Loaded {len(cache)} cached site descriptions from {cache_path.name}")

    missing = [k for k in all_keys if str(k) not in cache]
    if missing:
        print(f"  Running GeoChat on {len(missing)} new site(s) ...")
        for i, key in enumerate(missing):
            img = load_site_image(int(key))
            cache[str(key)] = describe_image_geochat(geochat_model, geochat_processor, img) \
                              if img is not None else ""
            if (i + 1) % 50 == 0 or (i + 1) == len(missing):
                print(f"    {i + 1}/{len(missing)}")
                _save_desc_cache(cache_path, model_name, cache)
        print(f"  Descriptions saved → {cache_path.name}")
    return cache


# ── GeoChat pipeline — Stage 2: text-LLM contrastive aggregation ──────────────

def load_text_model(model_name: str, quantize: bool = False):
    """Load a text-only instruction-tuned LLM for description aggregation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading text model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)
        print("  (4-bit quantization enabled)")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def aggregate_descriptions_llm(
    model, tokenizer,
    feat_idx: int,
    top_descriptions:    "list[tuple[int, float, str]]",   # (key, act, desc)
    bottom_descriptions: "list[tuple[int, float, str]]",
    prior_concepts: list = (),
    max_act: float = 1.0,
    n_total_sites: int = None,
    n_nonzero: int = None,
    max_act_percentile: int = None,
) -> dict:
    """Stage-2: text LLM finds the distinguishing concept from GeoChat descriptions."""
    import torch

    group_a = "\n".join(
        f"  A{i+1} [{act:.4f} ({100*act/max_act:.0f}% of max)]: {desc or '(no description)'}"
        for i, (_, act, desc) in enumerate(top_descriptions)
    )
    group_b = "\n".join(
        f"  B{i+1} [{act:.4f}]: {desc or '(no description)'}"
        for i, (_, act, desc) in enumerate(bottom_descriptions)
    )

    if prior_concepts:
        prior_lines = "\n".join(
            f"  • Feature {p['feature']} — \"{p.get('label', 'unknown')}\""
            + (f": {d}" if (d := (p.get('activated_concept') or p.get('description', ''))) else "")
            for p in prior_concepts
        )
        exclusion_block = (
            f"\nBANNED CONCEPTS (already identified — do NOT repeat or paraphrase):\n"
            f"{prior_lines}\n"
        )
    else:
        exclusion_block = ""

    signal_parts = []
    if max_act_percentile is not None and max_act_percentile <= 40:
        signal_parts.append(
            f"ACTIVATION STRENGTH: peak={max_act:.4f}, bottom-{max_act_percentile}% of all "
            f"SAE features — {'extremely' if max_act_percentile <= 20 else 'below-average'} weak."
        )
    if n_total_sites and n_nonzero:
        pct_nz = 100 * n_nonzero / n_total_sites
        if pct_nz < 30:
            signal_parts.append(
                f"SPARSITY: {n_nonzero}/{n_total_sites} sites ({pct_nz:.1f}%) non-zero "
                f"— {'rare niche' if pct_nz < 10 else 'moderately sparse'} feature."
            )
    signal_note = ("\n" + "\n".join(signal_parts) + "\n") if signal_parts else ""

    prompt = (
        STUDY_CONTEXT + "\n\n"
        "The following land-cover descriptions were generated by GeoChat (a remote-sensing "
        f"vision model) for two groups of satellite image sites (SAE feature #{feat_idx}).\n\n"
        f"Group A — HIGH activation ({len(top_descriptions)} sites, strongest→weakest):\n"
        f"{group_a}\n\n"
        f"Group B — LOW activation ({len(bottom_descriptions)} sites):\n"
        f"{group_b}\n"
        f"{exclusion_block}"
        f"{signal_note}\n"
        "TASK: Based solely on these descriptions, identify the ONE land-cover or landscape "
        "property that systematically distinguishes Group A from Group B.\n"
        "Work through these cues in order; stop at the first clear asymmetry:\n"
        "  1. Open water / wetland\n"
        "  2. Burn scars\n"
        "  3. Bare soil / cropland geometry\n"
        "  4. Settlement and road infrastructure\n"
        "  5. Vegetation spatial structure (only if 1–4 are symmetric)\n\n"
        "Answer in this EXACT format (four lines, no extra text):\n"
        "Active description: <5–15 words for what HIGH-activation sites have>\n"
        "Inactive description: <5–15 words for what LOW-activation sites have or lack>\n"
        "Label: <2–6 word label for the distinguishing concept>\n"
        "Confidence: <low|medium|high>"
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    result = {
        "feature":             feat_idx,
        "raw":                 raw,
        "top_keys":            [k for k, _, _ in top_descriptions],
        "bottom_keys":         [k for k, _, _ in bottom_descriptions],
        "top_acts":            [a for _, a, _ in top_descriptions],
        "bottom_acts":         [a for _, a, _ in bottom_descriptions],
        "top_descriptions":    [d for _, _, d in top_descriptions],
        "bottom_descriptions": [d for _, _, d in bottom_descriptions],
    }
    key_map = {
        "Active description": "activated_concept",
        "Inactive description": "not_activated_concept",
        "Label": "label",
        "Confidence": "confidence",
    }
    for line in raw.splitlines():
        for prefix, field in key_map.items():
            if line.strip().startswith(f"{prefix}:"):
                value = line.split(":", 1)[1].strip()
                if field in {"activated_concept", "not_activated_concept"}:
                    value = value.lower().rstrip(". ")
                result[field] = value
    result.setdefault("label", "unknown")
    result.setdefault("confidence", "low")
    return result


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

We are running NEXIS (Neural Effect Modifier Selection): a forward stepwise \
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
Native resolution: ~14 m/pixel (ETM+ pan-sharpened); presented to model at 224 × 224 px (~22 m/pixel effective, ≈ 25 km²).

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
    n_total_sites: int = None,
    n_nonzero: int = None,
    max_act_percentile: int = None,
) -> dict:
    """Single VLM call: show top and bottom images together, ask for contrast.

    prior_concepts : list of dicts (previous interpretation results), each with
        'feature', 'label', and 'description' keys.  When non-empty, the prompt
        explicitly tells the VLM which visual concepts are already accounted for,
        mirroring the conditional nature of NEXIS selection.
    """
    import torch

    content = [{"type": "text", "text": STUDY_CONTEXT + "\n\n"},
               {"type": "text", "text":
        f"Detailed colour guide for THIS composite (NIR→Red / Green→Green / SWIR→Blue):\n"
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
        f"Group A — HIGH activation ({len(top_imgs)} images, ordered strongest→weakest):\n"
    }]
    for i, (img, act) in enumerate(zip(top_imgs, top_acts)):
        content.append({"type": "image", "image": img})
        pct = 100 * act / max_act if max_act > 0 else 0
        content.append({"type": "text", "text": f"[A{i+1} — {act:.4f} ({pct:.0f}% of max)]\n"})

    content.append({"type": "text", "text":
        f"\n\nGroup B — LOW activation ({len(bottom_imgs)} images):\n"
    })
    for i, (img, act) in enumerate(zip(bottom_imgs, bottom_acts)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[B{i+1} — {act:.4f}]\n"})

    # Build the task text, injecting prior-concept exclusions when applicable.
    if prior_concepts:
        prior_lines = "\n".join(
            f"  • Feature {p['feature']} — \"{p.get('label', 'unknown')}\""
            + (f": {desc}" if (desc := (p.get('activated_concept') or p.get('description', ''))) else "")
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

    # ── Signal quality notes (items 2, 3, 4) ──────────────────────────────────
    signal_parts = []

    # Item 3: activation magnitude expressed as percentile rank
    if max_act_percentile is not None:
        if max_act_percentile <= 20:
            signal_parts.append(
                f"ACTIVATION STRENGTH: peak activation = {max_act:.4f}, "
                f"bottom {max_act_percentile}% of all SAE features — an extremely weak signal.  "
                f"Use the two-step strategy: (a) study Group A alone for any recurring "
                f"micro-pattern, however faint; (b) confirm Group B consistently lacks it."
            )
        elif max_act_percentile <= 40:
            signal_parts.append(
                f"ACTIVATION STRENGTH: peak activation = {max_act:.4f}, "
                f"bottom {max_act_percentile}% of all SAE features — a below-average signal.  "
                f"Expect a subtle but real contrast."
            )
    elif max_act < 0.05:
        signal_parts.append(
            f"ACTIVATION STRENGTH: WEAK (max={max_act:.4f}).  "
            f"Use the two-step strategy: (a) study Group A alone for any recurring "
            f"micro-pattern; (b) confirm Group B consistently lacks it."
        )

    # Item 2: sparsity
    if n_total_sites is not None and n_nonzero is not None:
        pct_nz = 100 * n_nonzero / n_total_sites
        if pct_nz < 10:
            signal_parts.append(
                f"SPARSITY: only {n_nonzero}/{n_total_sites} sites ({pct_nz:.1f}%) "
                f"have non-zero activation — Group A are rare outliers.  "
                f"Look for a niche land-cover signature present in very few landscapes, "
                f"not a broad regional pattern."
            )
        elif pct_nz < 30:
            signal_parts.append(
                f"SPARSITY: {n_nonzero}/{n_total_sites} sites ({pct_nz:.1f}%) "
                f"have non-zero activation — a moderately sparse feature."
            )

    # Item 4: gradient instruction
    signal_parts.append(
        f"GRADIENT: images within Group A are ordered by decreasing activation "
        f"(A1 = strongest, percentages shown).  If a pattern is vivid in A1–A2 and "
        f"fades toward A{len(top_imgs)}, that gradient IS the concept — describe what "
        f"diminishes with activation strength."
    )

    weak_signal_note = ("\n" + "\n".join(signal_parts) + "\n") if signal_parts else ""

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
        f"NOTE ON PRIOR CONCEPTS: if this feature appears to detect the SAME landscape "
        f"type as a banned concept but at finer spatial scale, rarer occurrence, or in "
        f"a different seasonal/spectral state (e.g. 'narrow seasonal stream' vs a banned "
        f"'perennial wetland corridor'), that IS a valid and distinct answer — describe "
        f"the specific variant precisely.  Only fall back to "
        f"'indistinguishable from prior concept' when you genuinely cannot find ANY "
        f"meaningful distinction after working through all seven cues above.\n\n"
        f"Answer in this EXACT format (four lines, no extra text).\n"
        f"IMPORTANT: never write 'Group A', 'Group B', 'the first group', etc. — "
        f"use only self-contained noun phrases that describe the landscape itself.\n\n"
        f"Active description: <5–15 words describing what HIGH-activation sites HAVE, "
        f"e.g. 'open water and wetland present' or 'dense burn scars near settlements'>\n"
        f"Inactive description: <5–15 words describing what LOW-activation sites HAVE or LACK, "
        f"e.g. 'no perennial water source, dry cropland only' or 'continuous dense vegetation cover'>\n"
        f"Label: <2–6 words naming the distinguishing concept, e.g. 'riparian wetland corridor' or "
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

    torch.cuda.empty_cache()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                    temperature=None, top_p=None, top_k=None)

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
        "Active description": "activated_concept",
        "Inactive description": "not_activated_concept",
        "Label": "label",
        "Confidence": "confidence",
    }
    # Fields that should be normalised: lowercase, stripped of trailing punctuation
    _normalise_fields = {"activated_concept", "not_activated_concept"}
    for line in raw.splitlines():
        for prefix, field in key_map.items():
            if line.strip().startswith(f"{prefix}:"):
                value = line.split(":", 1)[1].strip()
                if field in _normalise_fields:
                    value = value.lower().rstrip(". ")
                result[field] = value
    if "label" not in result:
        result["label"] = "unknown"
    if "confidence" not in result:
        result["confidence"] = "low"
    return result


# ── Duplicate-label guard ──────────────────────────────────────────────────────

def _label_is_duplicate(new_label: str, prior_concepts: list, threshold: float = 0.8) -> dict | None:
    """Return the first prior concept whose label is too similar to new_label, or None."""
    from difflib import SequenceMatcher
    new = new_label.lower().strip(" .")
    for p in prior_concepts:
        old = p.get("label", "").lower().strip(" .")
        if old and SequenceMatcher(None, new, old).ratio() >= threshold:
            return p
    return None


# ── Per-outcome interpretation ─────────────────────────────────────────────────

def interpret_outcome(
    outcome: str,
    model_dir: Path,
    site_feats: "np.ndarray",
    site_keys: "np.ndarray",
    pipeline: str,                                      # "qwen7b" | "qwen72b" | "points" | "geochat"
    vlm_model=None,         vlm_processor=None,         # qwen* / points
    site_descriptions: dict = None,                     # geochat: {str(key): desc}
    text_model=None,        text_tokenizer=None,        # geochat
    geochat_model_name: str = "",                       # geochat
    text_model_name: str = "",                          # geochat
    k: int = 10,
    min_activation: float = 0.001,
    sae_dim: int = 3072,
    vlm_model_name: str = "",                           # qwen* / points
) -> bool:
    """Interpret all NEXIS-selected SAE features for one outcome.

    Returns True if interpretations were written, False if skipped.
    VLM is already loaded by the caller — this function does not load or unload it.
    """
    out_dir   = model_dir / outcome
    out_dir.mkdir(parents=True, exist_ok=True)
    nexis_path = out_dir / "nexis_result.json"

    if not nexis_path.exists():
        print(f"  [{outcome}] SKIP — nexis_result.json not found (run analyze.py first)")
        return False

    with open(nexis_path) as f:
        nexis_output = json.load(f)

    n_sae            = nexis_output.get("feature_meta", {}).get("n_sae_features", sae_dim)
    selected_entries = nexis_output["nexis"]["selected"]
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

    # Precompute max activation per feature for percentile ranking (item 3)
    all_max_acts = site_feats.max(axis=0)   # shape: (n_features,)
    n_total_sites = site_feats.shape[0]

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

        if pipeline in ("qwen7b", "qwen72b", "points"):
            top_imgs, top_keys, top_acts       = load_group(fg["top_keys"],   fg["top_acts"])
            bottom_imgs, bottom_keys, bot_acts = load_group(fg["bottom_keys"], fg["bottom_acts"])
            if not top_imgs or not bottom_imgs:
                print("      (skipped: could not load images)")
                continue
        else:  # geochat — descriptions already cached; no image loading needed
            top_imgs, bottom_imgs = [], []
            top_keys,  top_acts   = fg["top_keys"],    fg["top_acts"]
            bottom_keys, bot_acts = fg["bottom_keys"], fg["bottom_acts"]

        max_act           = fg["top_acts"][0]
        feat_acts         = site_feats[:, feat_idx]
        n_nonzero         = int(np.sum(feat_acts > 0))
        max_act_pct       = int(np.mean(all_max_acts <= max_act) * 100)

        model_label = vlm_model_name if pipeline in ("qwen7b", "qwen72b") \
                      else f"{geochat_model_name}+{text_model_name}"

        if max_act < min_activation:
            print(f"      (skipped: max activation {max_act:.4f} < threshold {min_activation})")
            interpretations.append({
                "feature": feat_idx,
                "label": "low activation — uninterpretable",
                "activated_concept": f"Max activation {max_act:.4f} below threshold.",
                "not_activated_concept": "",
                "confidence": "low",
                "vlm_model": model_label,
                "pipeline": pipeline,
                "raw": "",
                "top_keys": fg["top_keys"], "bottom_keys": fg["bottom_keys"],
                "top_acts": fg["top_acts"], "bottom_acts": fg["bottom_acts"],
            })
            continue

        if pipeline in ("qwen7b", "qwen72b", "points"):
            result = contrast_groups_vlm(
                vlm_model, vlm_processor,
                feat_idx,
                top_imgs, top_keys, top_acts,
                bottom_imgs, bottom_keys, bot_acts,
                prior_concepts=interpretations,
                max_act=max_act,
                n_total_sites=n_total_sites,
                n_nonzero=n_nonzero,
                max_act_percentile=max_act_pct,
            )
        else:  # geochat
            top_descs    = [(k, a, site_descriptions.get(str(k), ""))
                            for k, a in zip(top_keys, top_acts)]
            bottom_descs = [(k, a, site_descriptions.get(str(k), ""))
                            for k, a in zip(bottom_keys, bot_acts)]
            result = aggregate_descriptions_llm(
                text_model, text_tokenizer,
                feat_idx, top_descs, bottom_descs,
                prior_concepts=interpretations,
                max_act=max_act,
                n_total_sites=n_total_sites,
                n_nonzero=n_nonzero,
                max_act_percentile=max_act_pct,
            )
        result["vlm_model"] = model_label
        result["pipeline"]  = pipeline

        dup = _label_is_duplicate(result.get("label", ""), interpretations)
        if dup:
            print(f"      [!] Label matches feature {dup['feature']} — marking indistinguishable")
            result["label"]             = f"indistinguishable from feature {dup['feature']}"
            result["activated_concept"] = f"Could not distinguish from: {dup.get('label', '')}"
            result["confidence"]        = "low"

        interpretations.append(result)
        print(f"      [{result.get('confidence','?')}] {result.get('label','?')}:\n"
              f"        Fires on: {result.get('activated_concept', result.get('raw',''))[:140]}"
              + (f"\n        Absent:   {result['not_activated_concept'][:120]}"
                 if result.get('not_activated_concept') else "")
              + "\n")

    pipeline_dir = out_dir / pipeline
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    out_path = pipeline_dir / "interpretations.json"
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
        description="VLM interpretation of NEXIS-selected SAE features. "
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
    p.add_argument("--pipeline", choices=["qwen7b", "qwen72b", "points", "geochat"], default="qwen7b",
                   help="Interpretation pipeline: 'qwen7b' (Qwen2-VL-7B direct contrast), "
                        "'qwen72b' (Qwen2-VL-72B direct contrast), "
                        "'points' (POINTS1.5-7B RS-specialist direct contrast), or "
                        "'geochat' (GeoChat per-image captions + text LLM aggregation).")
    p.add_argument("--vlm-model",     default="Qwen/Qwen2-VL-7B-Instruct",
                   help="[qwen7b/qwen72b] HuggingFace VLM model ID.")
    p.add_argument("--points-model",  default="WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat",
                   help="[points] HuggingFace model ID for the POINTS1.5 RS-specialist VLM.")
    p.add_argument("--geochat-model", default="MBZUAI/geochat-7b",
                   help="[geochat] GeoChat model for stage-1 image captioning.")
    p.add_argument("--text-model",    default="Qwen/Qwen2.5-72B-Instruct",
                   help="[geochat] Text LLM for stage-2 contrastive aggregation.")
    p.add_argument("--quantize", action="store_true",
                   help="Load model(s) in 4-bit (requires bitsandbytes).")
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
        out_path = MODEL_DIR / outcome / args.pipeline / "interpretations.json"
        if not args.overwrite and out_path.exists():
            print(f"  [{outcome}] Skipping ({args.pipeline}/interpretations.json exists; "
                  f"use --overwrite to redo)")
        else:
            to_run.append(outcome)

    if not to_run:
        print("All outcomes already interpreted.")
        return

    # Load shared site features (same file for all outcomes under this model)
    site_data  = np.load(MODEL_DIR / "site_features.npz")
    site_feats = site_data["site_features"]
    site_keys  = site_data["site_keys"]

    # Load model(s) based on chosen pipeline
    print(f"\nPipeline: {args.pipeline}  ({len(to_run)} outcome(s))")
    if args.pipeline in ("qwen7b", "qwen72b"):
        print(f"  VLM: {args.vlm_model}")
        vlm_model, vlm_processor = load_vlm(args.vlm_model, quantize=args.quantize)
        site_descriptions = text_model = text_tokenizer = None
    elif args.pipeline == "points":
        print(f"  VLM: {args.points_model}")
        vlm_model, vlm_processor = load_vlm(args.points_model, quantize=args.quantize,
                                             trust_remote_code=True)
        site_descriptions = text_model = text_tokenizer = None
    else:  # geochat
        print(f"  Stage 1 (GeoChat): {args.geochat_model}")
        geochat_model, geochat_processor = load_geochat(args.geochat_model,
                                                         quantize=args.quantize)
        # Build / update description cache for every site in the dataset
        all_keys = site_keys.tolist()
        site_descriptions = load_or_build_descriptions(
            geochat_model, geochat_processor, all_keys,
            model_name=args.geochat_model,
            force_rebuild=args.overwrite,
        )
        unload_model(geochat_model)
        del geochat_processor
        gc.collect()

        print(f"  Stage 2 (text LLM): {args.text_model}")
        text_model, text_tokenizer = load_text_model(args.text_model, quantize=args.quantize)
        vlm_model = vlm_processor = None
    print()

    for outcome in to_run:
        print(f"── {outcome} {'─' * max(0, 55 - len(outcome))}")
        interpret_outcome(
            outcome, MODEL_DIR, site_feats, site_keys,
            pipeline=args.pipeline,
            vlm_model=vlm_model,            vlm_processor=vlm_processor,
            site_descriptions=site_descriptions,
            text_model=text_model,          text_tokenizer=text_tokenizer,
            geochat_model_name=args.geochat_model,
            text_model_name=args.text_model,
            k=args.k, min_activation=args.min_activation, sae_dim=args.sae_dim,
            vlm_model_name=(args.points_model if args.pipeline == "points"
                            else args.vlm_model),
        )

    unload_model(vlm_model if args.pipeline in ("qwen7b", "qwen72b", "points") else text_model)


if __name__ == "__main__":
    main()
