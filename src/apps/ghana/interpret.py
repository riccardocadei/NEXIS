"""
Ghana LEAP 1000 — NEXIS feature search + VLM interpretation.

Methods (run for each representation)
--------------------------------------
  nexis_no_adj  — NEXIS on Z with W as controls, no multiple-testing correction
  nexis_fdr     — NEXIS on Z with W as controls, BH-FDR correction
  marginal_w    — marginal test on W only (no Z), no correction  [run once]

Representations
---------------
  codes     — post-top-k sparse activations  (k=25, ~131 live neurons)
  pre_codes — post-ReLU pre-top-k activations (~272 live neurons)

VLM interpretation uses the national training pool (9,592 tiles) for top/bottom
community selection — much larger contrast pool than the 162 LEAP communities.

Outputs
-------
  results/ghana/{rep}/nexis_no_adj/result.json
  results/ghana/{rep}/nexis_no_adj/{pipeline}/interpretations.json
  results/ghana/{rep}/nexis_fdr/result.json
  results/ghana/{rep}/nexis_fdr/{pipeline}/interpretations.json
  results/ghana/marginal_w/result.json

Pipelines
---------
  qwen72b      — Qwen2-VL-72B sees all 16 images at once and produces the contrast
                 (default; --vlm-model to override)
  geochat_llm  — GeoChat-7B describes each image individually; a text LLM
                 (--synthesis-model, default Qwen/Qwen2.5-7B-Instruct) synthesises
                 the contrast from the 16 text descriptions.  Much lighter: ~14 GB
                 VRAM total (two 7B models loaded sequentially).

Usage
-----
  python src/apps/ghana/interpret.py --quantize
  python src/apps/ghana/interpret.py --mode codes --no-interpret
  python src/apps/ghana/interpret.py --overwrite   # redo VLM even if files exist
  python src/apps/ghana/interpret.py --pipeline geochat_llm --quantize
  python src/apps/ghana/interpret.py --pipeline geochat_llm \\
      --geochat-model MBZUAI/geochat-7B \\
      --synthesis-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
TIF_DIR  = SAT_DIR / "tif"
TIF_NAT  = SAT_DIR / "tif_national"
RES_DIR  = ROOT / "results" / "ghana"

sys.path.insert(0, str(ROOT))
from src.apps.ghana.data import load_data, W_ALL, W_LABELS, COMMUNITY_Z
from src.method.nexis    import nexis, marginal_select, SelectionResult


# ── Image loading ──────────────────────────────────────────────────────────────

def _load_composite(tif_path: Path, size: int = 224, mode: str = "both"):
    """Load Landsat 8 tile as a PIL image.

    TIF band order from GEE: B4(Red), B3(Green), B2(Blue), B5(NIR), B6(SWIR1), B7(SWIR2).
    Rasterio indices (1-based): 1, 2, 3, 4, 5, 6.

    mode="both" — side-by-side 2*size × size: true-colour (B4/B3/B2) | false-colour (B5/B3/B7)
    mode="fc"   — false-colour only (B5/B3/B7), size × size
    """
    import rasterio
    from PIL import Image as PILImage
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        green = src.read(2).astype(np.float32)   # B3
        nir   = src.read(4).astype(np.float32)   # B5
        swir2 = src.read(6).astype(np.float32)   # B7
        if mode == "both":
            red  = src.read(1).astype(np.float32)  # B4
            blue = src.read(3).astype(np.float32)  # B2

    def norm(b):
        valid = b[b > 0]
        if valid.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(valid, [2, 98])
        out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        out[b <= 0] = 0
        return out

    fc_arr = (np.stack([norm(nir), norm(green), norm(swir2)], axis=-1) * 255).astype(np.uint8)
    fc_img = PILImage.fromarray(fc_arr).resize((size, size), PILImage.BICUBIC)

    if mode == "fc":
        return fc_img

    tc_arr = (np.stack([norm(red), norm(green), norm(blue)], axis=-1) * 255).astype(np.uint8)
    tc_img = PILImage.fromarray(tc_arr).resize((size, size), PILImage.BICUBIC)
    combined = PILImage.new("RGB", (size * 2, size))
    combined.paste(tc_img, (0, 0))
    combined.paste(fc_img, (size, 0))
    return combined


def load_national_image(grid_id: int, size: int = 224, mode: str = "both"):
    return _load_composite(TIF_NAT / f"ghana_grid{int(grid_id):06d}.tif", size, mode)


def load_group_national(grid_ids, activations, size: int = 224, mode: str = "both"):
    imgs, valid_ids, valid_acts = [], [], []
    for gid, act in zip(grid_ids, activations):
        img = load_national_image(int(gid), size, mode)
        if img is not None:
            imgs.append(img)
            valid_ids.append(int(gid))
            valid_acts.append(float(act))
    return imgs, valid_ids, valid_acts


def load_community_image(comm_id: int, size: int = 224, mode: str = "both"):
    return _load_composite(TIF_DIR / f"ghana_comm{int(comm_id):04d}.tif", size, mode)


def load_group_community(comm_ids, activations, size: int = 224, mode: str = "both"):
    imgs, valid_ids, valid_acts = [], [], []
    for cid, act in zip(comm_ids, activations):
        img = load_community_image(int(cid), size, mode)
        if img is not None:
            imgs.append(img)
            valid_ids.append(int(cid))
            valid_acts.append(float(act))
    return imgs, valid_ids, valid_acts


# ── VLM loading ────────────────────────────────────────────────────────────────

def load_vlm(model_name: str, quantize: bool = False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"  Loading VLM: {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)
        print("  (4-bit quantization enabled)")
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, processor


# ── Prompt ─────────────────────────────────────────────────────────────────────

STUDY_CONTEXT = """\
STUDY CONTEXT
You are helping interpret a feature neuron from a Sparse Autoencoder (SAE) \
trained on Prithvi patch embeddings of Landsat satellite imagery.

The images come from the LEAP 1000 evaluation — a randomised controlled trial \
in Ghana in which poor households received unconditional cash transfers (~GHS 8,400 \
over two years). The PRIMARY OUTCOME is household consumption two years after \
the transfer. The satellite images represent the landscape around each community.

We are running NEXIS (Neural Effect Modifier Selection): a forward stepwise \
procedure that tests which SAE features statistically modify the programme's \
average treatment effect. The neuron you are interpreting has been selected \
because its activation pattern correlates with heterogeneous treatment effects \
across communities — meaning the programme works DIFFERENTLY depending on the \
landscape this neuron detects.

IMAGE FORMAT
Each tile is shown as a SIDE-BY-SIDE pair from Landsat 8, ~5 × 5 km footprint.
Available bands: B2(Blue), B3(Green), B4(Red), B5(NIR), B6(SWIR1), B7(SWIR2).
Native resolution: ~30 m/pixel; each panel presented at 224 × 224 px (~22 m/pixel effective, ≈ 25 km²).

LEFT PANEL — True-colour (B4→Red / B3→Green / B2→Blue):
  • Natural colours: green vegetation, brown soil, blue water, grey/white settlements
  • Best for: roads, settlements, field boundaries, general land cover

RIGHT PANEL — False-colour (B5/NIR→Red / B3→Green / B7/SWIR2→Blue):
  • Bright red / magenta        = dense healthy vegetation (high NIR) — forest, woodland, vigorous crops
  • Dark red / maroon           = burn scar or sparse post-fire regrowth (NIR destroyed)
  • Tan / pinkish-brown         = bare soil, dry fallow, degraded land
  • Smooth bounded pale patches = cultivated fields (crops slightly greener; dry-season fallow tan)
  • Dark blue / near-black      = open water (river, lake, reservoir)
  • Dark olive / very dark      = wetland, seasonally flooded valley bottom, riparian fringe
  • Grey / white patches        = settlement, metal roofs, compacted ground, roads
  • Pale laterite patches       = infertile hardpan (highly reflective bare surface)
  • Best for: vegetation health, water bodies, burn scars, moisture

IMPORTANT — nearly every tile in southern/central Ghana shows green (left) and red (right) \
vegetation. Overall redness or greenness is NOT a useful distinguishing feature. \
Focus on STRUCTURAL and LAND-COVER differences visible in either panel.

ECONOMICALLY RELEVANT VISUAL PROPERTIES TO CONSIDER
  · Water access (rivers, lakes, wetlands) — irrigation potential, domestic water, fishing
  · Road / path networks (linear grey features) — market access, input delivery, labour mobility
  · Settlement clusters (grey/white) — population density, service access, transfer spending patterns
  · Agricultural field geometry (regular cleared patches) — commercial vs subsistence farming
  · Forest / dense vegetation cover — timber, NTFP income, shade crops (cocoa, oil palm)
  · Burn scars — land clearing, charcoal production, seasonal field prep
  · Bare / degraded land — soil quality, land pressure, erosion\
"""


def contrast_groups_vlm(
    model, processor,
    feat_idx: int,
    top_imgs: list, top_ids: list, top_acts: list,
    bot_imgs: list, bot_ids: list, bot_acts: list,
    prior_concepts: list = (),
    max_act: float = 1.0,
    n_total: int = None,
    n_nonzero: int = None,
    max_act_pct: int = None,
    **_ignored,
) -> dict:
    content = [
        {"type": "text", "text": STUDY_CONTEXT + "\n\n"},
        {"type": "text", "text":
            f"Group A — HIGH activation ({len(top_imgs)} tiles, strongest → weakest):\n"},
    ]
    for i, (img, act) in enumerate(zip(top_imgs, top_acts)):
        content.append({"type": "image", "image": img})
        pct = 100 * act / max_act if max_act > 0 else 0
        content.append({"type": "text", "text": f"[A{i+1} — {act:.4f} ({pct:.0f}% of max)]\n"})

    content.append({"type": "text", "text":
        f"\nGroup B — LOW / ZERO activation ({len(bot_imgs)} tiles):\n"})
    for i, (img, act) in enumerate(zip(bot_imgs, bot_acts)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[B{i+1} — {act:.4f}]\n"})

    # Prior-concept exclusion (mirrors NEXIS conditioning)
    if prior_concepts:
        lines = "\n".join(
            f"  • Feature {p['feature']} — \"{p.get('label','?')}\""
            + (f": {p['activated_concept']}" if p.get('activated_concept') else "")
            for p in prior_concepts
        )
        excl = (
            f"\n\nBANNED CONCEPTS (already identified in earlier features — "
            f"do NOT repeat or paraphrase these):\n{lines}\n\n"
            f"By statistical construction this feature carries information that the "
            f"banned concepts above do NOT explain.  Your label and description MUST "
            f"name something structurally different from every banned concept.  "
            f"If the only visible difference matches a banned concept, write "
            f"\"indistinguishable from prior concept\" rather than inventing a synonym."
        )
    else:
        excl = ""

    # Signal notes
    signal_parts = []
    if n_total and n_nonzero:
        pct_nz = 100 * n_nonzero / n_total
        if pct_nz < 10:
            signal_parts.append(
                f"SPARSITY: only {n_nonzero}/{n_total} tiles ({pct_nz:.1f}%) have non-zero "
                f"activation — Group A are rare outliers.  Look for a niche land-cover "
                f"signature present in very few landscapes, not a broad regional pattern."
            )
        elif pct_nz < 30:
            signal_parts.append(
                f"SPARSITY: {n_nonzero}/{n_total} tiles ({pct_nz:.1f}%) have non-zero "
                f"activation — a moderately sparse feature."
            )
    if max_act_pct is not None:
        if max_act_pct <= 20:
            signal_parts.append(
                f"ACTIVATION STRENGTH: bottom {max_act_pct}% of all SAE features — "
                f"an extremely weak signal.  Use the two-step strategy: "
                f"(a) study Group A alone for any recurring micro-pattern, however faint; "
                f"(b) confirm Group B consistently lacks it."
            )
        elif max_act_pct <= 40:
            signal_parts.append(
                f"ACTIVATION STRENGTH: bottom {max_act_pct}% of all SAE features — "
                f"a below-average signal.  Expect a subtle but real contrast."
            )
    signal_parts.append(
        f"GRADIENT: images within Group A are ordered by decreasing activation "
        f"(A1 = strongest, percentages shown).  If a pattern is vivid in A1–A2 and "
        f"fades toward A{len(top_imgs)}, that gradient IS the concept — describe what "
        f"diminishes with activation strength."
    )
    weak_signal_note = ("\n" + "\n".join(signal_parts) + "\n") if signal_parts else ""

    content.append({"type": "text", "text":
        f"{excl}\n\n"
        f"{weak_signal_note}"
        f"TASK: Identify what makes Group A systematically different from Group B.\n"
        f"The difference may be PRESENCE of something in A that B lacks, OR ABSENCE "
        f"of something in A that B has — both are valid interpretations.\n\n"
        f"Work through these land-cover cues in order; stop at the FIRST clear asymmetry:\n\n"
        f"  1. OPEN WATER / WETLAND\n"
        f"     Dark blue-black = river, lake, reservoir.  Dark olive = riparian wetland, "
        f"seasonally flooded valley bottom.\n"
        f"     Economic link: water access enables irrigation and dry-season farming; "
        f"proximity to water shapes how cash transfers are spent (inputs vs food).\n\n"
        f"  2. BURN SCARS\n"
        f"     Dark red/maroon patches (not open water).  Sharp-edged = recent; diffuse = older.\n"
        f"     Economic link: burning signals active land clearing, charcoal production, "
        f"or seasonal field preparation — proxies for land-use intensity.\n\n"
        f"  3. BARE SOIL / CROPLAND EXTENT AND GEOMETRY\n"
        f"     Tan-pink patches.  Regular large rectangles = commercial/mechanised farming; "
        f"small irregular patches = subsistence smallholder.  More bare soil in one group?\n"
        f"     Economic link: field size and geometry proxy capital, tenure security, "
        f"and integration with output markets.\n\n"
        f"  4. SETTLEMENT AND ROAD INFRASTRUCTURE\n"
        f"     Grey/white clusters = villages, market centres.  Pale linear features = "
        f"roads and tracks.  Denser or more connected in one group?\n"
        f"     Economic link: road proximity strongly predicts market access and the "
        f"degree to which cash transfers translate into consumption gains.\n\n"
        f"  5. DEGRADED / BARE LAND\n"
        f"     Persistently pale tan with no seasonal green pulse — near peri-urban fringes "
        f"or heavily farmed zones.\n"
        f"     Economic link: degradation proxies soil quality loss and land pressure, "
        f"which shape the marginal returns to additional income.\n\n"
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
        f"Active description: <5–15 words describing what HIGH-activation tiles HAVE, "
        f"e.g. 'open water or riparian wetland present' or 'dense burn scars near settlements'>\n"
        f"Inactive description: <5–15 words describing what LOW-activation tiles HAVE or LACK, "
        f"e.g. 'no perennial water, dry cropland only' or 'continuous closed-canopy forest cover'>\n"
        f"Label: <2–6 words naming the distinguishing concept, e.g. 'riparian wetland corridor' or "
        f"'commercial field geometry' — NEVER the bare word 'absent' alone>\n"
        f"Confidence: <low|medium|high>"
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=list(top_imgs) + list(bot_imgs),
        return_tensors="pt", padding=True
    ).to(model.device)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 temperature=None, top_p=None, top_k=None)
    raw = processor.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

    return _parse_structured_output(raw, feat_idx, top_ids, bot_ids, top_acts, bot_acts)


def _label_is_duplicate(new_label: str, prior: list, thr: float = 0.8):
    from difflib import SequenceMatcher
    new = new_label.lower().strip(" .")
    for p in prior:
        old = p.get("label", "").lower().strip(" .")
        if old and SequenceMatcher(None, new, old).ratio() >= thr:
            return p
    return None


def _parse_structured_output(raw: str, feat_idx: int,
                              top_ids, bot_ids, top_acts, bot_acts,
                              extra: dict = None) -> dict:
    """Parse the 4-line structured VLM/LLM output into a result dict."""
    result = {
        "feature": feat_idx, "raw": raw,
        "top_ids": list(top_ids), "bot_ids": list(bot_ids),
        "top_acts": list(top_acts), "bot_acts": list(bot_acts),
        **(extra or {}),
    }
    for line in raw.splitlines():
        for prefix, field in [
            ("Active description",   "activated_concept"),
            ("Inactive description", "not_activated_concept"),
            ("Label",                "label"),
            ("Confidence",           "confidence"),
        ]:
            if line.strip().startswith(f"{prefix}:"):
                val = line.split(":", 1)[1].strip()
                if field in {"activated_concept", "not_activated_concept"}:
                    val = val.lower().rstrip(". ")
                result[field] = val
    result.setdefault("label", "unknown")
    result.setdefault("confidence", "low")
    return result


# ── GeoChat + text-LLM pipeline ────────────────────────────────────────────────

_GEOCHAT_QUERY = (
    "Describe the land cover in this false-colour Landsat 8 satellite image "
    "(NIR→Red channel, Green→Green channel, SWIR2→Blue channel). "
    "Bright red/magenta = vegetation, tan/brown = bare soil or fallow, "
    "dark blue/black = open water, grey/white = settlement or roads. "
    "List the dominant land-cover types, their approximate fraction of the tile, "
    "and any notable spatial patterns (e.g. linear features, field geometry, "
    "settlement clusters, water bodies). Be specific and factual. 3-5 sentences."
)

_SYNTHESIS_SYSTEM = (
    "You are an expert in remote sensing and land-cover analysis. "
    "You will receive text descriptions of two groups of satellite image tiles. "
    "Your task: identify the ONE visual property that most clearly distinguishes "
    "Group A (high SAE neuron activation) from Group B (low / zero activation)."
)

_SYNTHESIS_TASK = (
    "TASK: Based on these descriptions, identify the ONE visual property that best "
    "explains why the SAE neuron fires on Group A but not Group B.\n\n"
    "Work through these cues in order — stop at the FIRST clear asymmetry:\n"
    "  1. OPEN WATER / WETLAND\n"
    "  2. BURN SCARS\n"
    "  3. BARE SOIL / CROPLAND EXTENT AND GEOMETRY\n"
    "  4. SETTLEMENT AND ROAD DENSITY\n"
    "  5. VEGETATION TEXTURE AND FRAGMENTATION\n"
    "  6. TOPOGRAPHIC / SPATIAL PATTERN\n"
    "  7. If NOTHING structurally differs, state that explicitly.\n\n"
    "STRICT RULES:\n"
    "  - Label what makes Group A *different* from Group B, not what they share.\n"
    "  - Do NOT hypothesise about economic causes.\n"
    "  - Only describe what is visually observable.\n\n"
    "Answer in EXACTLY this format (4 lines, no extra text):\n"
    "Active description: <5–15 words — what HIGH-activation tiles SHOW that low ones lack>\n"
    "Inactive description: <5–15 words — what LOW-activation tiles SHOW instead>\n"
    "Label: <2–6 words naming the distinguishing visual concept>\n"
    "Confidence: <low|medium|high>"
)


def _remap_geochat_state_dict(state_dict: dict) -> dict:
    new_sd = {}
    for key, val in state_dict.items():
        if "rotary_emb.inv_freq" in key:
            continue
        if key.startswith("model.mm_projector."):
            rest = key[len("model.mm_projector."):]
            idx_str, _, param = rest.partition(".")
            linear_n = int(idx_str) // 2 + 1
            new_key = f"model.multi_modal_projector.linear_{linear_n}.{param}"
        elif key.startswith("model.vision_tower.vision_tower."):
            new_key = "model.vision_tower." + key[len("model.vision_tower.vision_tower."):]
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model."):]
        else:
            new_key = key
        new_sd[new_key] = val
    return new_sd


def _get_geochat_remapped_dir(model_name: str) -> Path:
    """Return path to LlavaForConditionalGeneration-compatible remapped weights, building once."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from transformers import LlavaConfig, LlamaConfig, CLIPVisionConfig

    cache_dir = ROOT / "models" / "geochat_remapped"
    if (cache_dir / "config.json").exists():
        return cache_dir

    print("  Remapping GeoChat weights (one-time setup) ...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = hf_hub_download(model_name, "config.json")
    with open(cfg_path) as f:
        gc = _json.load(f)
    vision_tower = gc.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")

    text_cfg = LlamaConfig(
        hidden_size=gc["hidden_size"], num_hidden_layers=gc["num_hidden_layers"],
        num_attention_heads=gc["num_attention_heads"],
        num_key_value_heads=gc.get("num_key_value_heads", gc["num_attention_heads"]),
        intermediate_size=gc["intermediate_size"], rms_norm_eps=gc.get("rms_norm_eps", 1e-5),
        vocab_size=gc["vocab_size"],
        max_position_embeddings=gc.get("max_position_embeddings", 4096),
        pretraining_tp=gc.get("pretraining_tp", 1),
    )
    vision_cfg = CLIPVisionConfig.from_pretrained(vision_tower)
    LlavaConfig(text_config=text_cfg, vision_config=vision_cfg).save_pretrained(str(cache_dir))

    index_path = hf_hub_download(model_name, "pytorch_model.bin.index.json")
    with open(index_path) as f:
        weight_files = sorted(set(_json.load(f)["weight_map"].values()))

    state_dict = {}
    for i, fname in enumerate(weight_files, 1):
        print(f"    shard {i}/{len(weight_files)}: {fname}")
        shard = torch.load(hf_hub_download(model_name, fname), map_location="cpu",
                           weights_only=True)
        state_dict.update(shard)

    state_dict = _remap_geochat_state_dict(state_dict)
    from safetensors.torch import save_file
    save_file(state_dict, str(cache_dir / "model.safetensors"))
    return cache_dir


def load_geochat(model_name: str = "MBZUAI/geochat-7B", quantize: bool = False):
    """Load GeoChat via remapped LlavaForConditionalGeneration weights."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from transformers import (LlavaProcessor, LlavaForConditionalGeneration,
                              CLIPImageProcessor, AutoTokenizer)
    print(f"  Loading GeoChat: {model_name} ...")

    cfg_path = hf_hub_download(model_name, "config.json")
    with open(cfg_path) as f:
        vision_tower = _json.load(f).get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_size = getattr(image_processor, "patch_size", None) or 14
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer,
                               patch_size=patch_size)

    remapped_dir = _get_geochat_remapped_dir(model_name)
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)
        print("  (4-bit quantization enabled)")
    model = LlavaForConditionalGeneration.from_pretrained(str(remapped_dir), **kwargs)

    # GeoChat's <image> token sits at index 32000 in the tokenizer but the remapped
    # safetensors only has 32000 embedding rows (0-31999).  resize_token_embeddings is
    # unreliable with device_map="auto" + 4-bit quantization.
    # Solution: remap image_token_index to unk_token_id (typically 0), which is always
    # in-bounds and never produced by normal tokenization of ASCII prompts.
    # LLaVA's forward detects image positions via `input_ids == image_token_index`, so
    # as long as config and input_ids agree on the same ID, visual features are injected.
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
    if img_tok_id == unk_id or img_tok_id >= 32000:
        # <image> not a recognized in-vocab token; use unk as the placeholder
        img_tok_id = unk_id
    model.config.image_token_index = img_tok_id
    print(f"  image_token_index set to {img_tok_id}")

    model.eval()
    return model, processor


def geochat_describe_image(model, processor, img, query: str = _GEOCHAT_QUERY) -> str:
    """Run GeoChat on one image and return a land-cover description."""
    tok = processor.tokenizer
    img_tok_id = model.config.image_token_index  # e.g. 32000

    # Tokenize prefix and suffix separately, then splice in the image token ID manually.
    # Calling tok("USER: <image>\n...") leaves tokens: 0 in newer transformers because
    # <image> is not recognized as a special token during plain text tokenization.
    prefix_ids = tok("USER: ", add_special_tokens=True,
                     return_tensors="pt")["input_ids"]           # includes BOS
    suffix_ids = tok(f"\n{query}\nASSISTANT:", add_special_tokens=False,
                     return_tensors="pt")["input_ids"]
    img_id_t   = torch.tensor([[img_tok_id]], dtype=torch.long)
    input_ids  = torch.cat([prefix_ids, img_id_t, suffix_ids], dim=1).to(model.device)
    attn_mask  = torch.ones_like(input_ids)

    pixel_values = processor.image_processor(
        images=img, return_tensors="pt"
    )["pixel_values"].to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values,
            max_new_tokens=200, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    return tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def load_text_llm(model_name: str, quantize: bool = False):
    """Load a text-only instruction-tuned LLM for contrast synthesis."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading synthesis LLM: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs.pop("torch_dtype", None)
        print("  (4-bit quantization enabled)")
    lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    lm.eval()
    return lm, tokenizer


def synthesize_contrast_llm(
    lm, tokenizer,
    feat_idx: int,
    top_descs: list, top_ids: list, top_acts: list,
    bot_descs: list, bot_ids: list, bot_acts: list,
    prior_concepts: list = (),
    n_total: int = None,
    n_nonzero: int = None,
) -> dict:
    """Build a text prompt from per-image descriptions and run the synthesis LLM."""
    top_block = "\n".join(
        f"  A{i+1} (activation={a:.4f}): {d}"
        for i, (d, a) in enumerate(zip(top_descs, top_acts))
    )
    bot_block = "\n".join(
        f"  B{i+1} (activation={a:.4f}): {d}"
        for i, (d, a) in enumerate(zip(bot_descs, bot_acts))
    )

    excl = ""
    if prior_concepts:
        lines = "\n".join(
            f"  - Neuron {p['feature']}: \"{p.get('label', '?')}\" "
            f"— {p.get('activated_concept', '')}"
            for p in prior_concepts
        )
        excl = (
            f"\nALREADY IDENTIFIED (do NOT repeat or paraphrase):\n{lines}\n\n"
            "This neuron was selected conditional on the above — it must capture "
            "something visually different.\n"
        )

    sparsity_note = ""
    if n_total and n_nonzero:
        pct = 100 * n_nonzero / n_total
        qualifier = (
            "Group A tiles are rare outliers — look for a niche land-cover signature."
            if pct < 10 else "Moderately sparse feature."
        )
        sparsity_note = (
            f"\nSPARSITY: {n_nonzero}/{n_total} tiles ({pct:.1f}%) have non-zero "
            f"activation. {qualifier}\n"
        )

    user_msg = (
        f"{COLOUR_GUIDE}\n\n"
        f"{sparsity_note}"
        f"{excl}\n"
        "GROUP A — HIGH activation tiles (strongest → weakest):\n"
        f"{top_block}\n\n"
        "GROUP B — LOW / ZERO activation tiles:\n"
        f"{bot_block}\n\n"
        f"{_SYNTHESIS_TASK}"
    )

    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out_ids = lm.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    raw = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    return _parse_structured_output(
        raw, feat_idx,
        top_ids, bot_ids, top_acts, bot_acts,
        extra={"top_descriptions": top_descs, "bot_descriptions": bot_descs},
    )


def contrast_groups_geochat_llm(
    geo_model, geo_processor,
    lm, tokenizer,
    feat_idx: int,
    top_imgs: list, top_ids: list, top_acts: list,
    bot_imgs: list, bot_ids: list, bot_acts: list,
    prior_concepts: list = (),
    n_total: int = None,
    n_nonzero: int = None,
    _desc_cache: dict = None,
    _cache_key: tuple = None,
    **_ignored,
) -> dict:
    """GeoChat per-image description → text-LLM contrast synthesis.

    In two-pass mode (geo_model is None), descriptions are read from _desc_cache[_cache_key].
    In single-pass mode both models are provided and descriptions are generated on-the-fly.
    """
    if geo_model is None:
        # Two-pass mode: descriptions already collected in pass 1
        top_descs, bot_descs = _desc_cache.get(_cache_key, ([], []))
    else:
        print(f"    GeoChat: describing {len(top_imgs)} top images ...")
        top_descs = [geochat_describe_image(geo_model, geo_processor, img) for img in top_imgs]
        print(f"    GeoChat: describing {len(bot_imgs)} bottom images ...")
    bot_descs = [geochat_describe_image(geo_model, geo_processor, img) for img in bot_imgs]

    return synthesize_contrast_llm(
        lm, tokenizer,
        feat_idx=feat_idx,
        top_descs=top_descs, top_ids=top_ids, top_acts=top_acts,
        bot_descs=bot_descs, bot_ids=bot_ids, bot_acts=bot_acts,
        prior_concepts=prior_concepts,
        n_total=n_total, n_nonzero=n_nonzero,
    )


# ── Data / SAE helpers ─────────────────────────────────────────────────────────

def _compute_sae_activations(embeddings: np.ndarray, ckpt: dict,
                              wh_mean: np.ndarray, wh_std: np.ndarray,
                              pre_codes: bool = False) -> np.ndarray:
    """Apply SAE encoder to whitened embeddings. pre_codes=True → skip top-k."""
    x = torch.from_numpy(((embeddings - wh_mean) / wh_std)).float()
    W_enc_w = ckpt["W_enc.weight"]
    W_enc_b = ckpt["W_enc.bias"]
    b_dec   = ckpt["b_dec"]
    with torch.no_grad():
        acts = F.relu((x - b_dec) @ W_enc_w.T + W_enc_b)
        if not pre_codes:
            k = 25
            topk_vals, _ = torch.topk(acts, k, dim=-1)
            threshold = topk_vals[:, -1:].detach()
            acts = acts * (acts >= threshold)
    return acts.numpy()


def _load_nexis_inputs(min_activations: int = 10):
    """Load all data needed for Ghana NEXIS experiments."""
    import pandas as pd

    df_full = load_data(DATA_DIR)
    hh_both = df_full.groupby("hhid")["wave"].nunique()
    df = df_full[df_full["hhid"].isin(hh_both[hh_both == 2].index)].copy()
    df0 = df[df["wave"] == 0]
    df1 = df[df["wave"] == 1]

    sp = pd.read_csv(SAT_DIR / "spectral_indices.csv").rename(columns={"comm_id": "comm"})
    # Use mean columns only; strip _mean suffix for clean z_{name} labels
    SPECTRAL_COLS  = [c for c in sp.columns if c.endswith("_mean")]
    SPECTRAL_NAMES = [c[:-5] for c in SPECTRAL_COLS]   # ndvi_mean → ndvi

    merged = (
        df0.set_index("hhid")[["T", "comm"] + W_ALL + COMMUNITY_Z + ["Y"]]
           .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
    )
    merged["dY"] = merged["Y1"] - merged["Y"]
    merged = merged.reset_index().merge(sp, on="comm", how="left").set_index("hhid")

    y       = merged["dY"].values.astype(float)
    t       = merged["T"].values.astype(float)
    W       = merged[W_ALL].values.astype(float)
    W_NAMES = [W_LABELS.get(c, c) for c in W_ALL]
    cluster      = merged["comm"].values                             # community IDs for CRVE
    spectral_hh  = merged[SPECTRAL_COLS].values.astype(float)       # (n_hh, n_spectral)
    community_hh = merged[COMMUNITY_Z].values.astype(float)         # (n_hh, 2) community-level

    # Load SAE checkpoint + whitening
    ckpt    = torch.load(SAT_DIR / "sae_model.pt", map_location="cpu", weights_only=False)
    wh_mean = np.load(SAT_DIR / "sae_whiten_mean.npy")
    wh_std  = np.load(SAT_DIR / "sae_whiten_std.npy")

    # LEAP community activations (codes + pre-codes)
    leap_embs = np.load(SAT_DIR / "prithvi_embeddings.npy")
    leap_ids  = np.load(SAT_DIR / "prithvi_comm_ids.npy")

    leap_codes     = np.load(SAT_DIR / "sae_activations.npy")
    leap_pre_codes = _compute_sae_activations(leap_embs, ckpt, wh_mean, wh_std, pre_codes=True)

    # National grid activations (for VLM pool)
    nat_embs = np.load(SAT_DIR / "national" / "prithvi_embeddings.npy")
    nat_ids  = np.load(SAT_DIR / "national" / "prithvi_comm_ids.npy")

    print("  Computing national grid activations (codes) ...")
    nat_codes     = _compute_sae_activations(nat_embs, ckpt, wh_mean, wh_std, pre_codes=False)
    print("  Computing national grid activations (pre-codes) ...")
    nat_pre_codes = _compute_sae_activations(nat_embs, ckpt, wh_mean, wh_std, pre_codes=True)

    # Build household-level Z matrices
    comm_to_idx = dict(zip(leap_ids, range(len(leap_ids))))
    hh_idx = merged["comm"].map(comm_to_idx).values

    def make_filtered(comm_acts, hh_idx_arr):
        live_mask = (comm_acts > 0).sum(axis=0) >= min_activations
        live_idx  = np.where(live_mask)[0]
        return comm_acts[:, live_mask], live_idx, comm_acts[:, live_mask][hh_idx_arr]

    leap_codes_comm, live_idx_codes, Z_codes_hh = make_filtered(leap_codes,     hh_idx)
    leap_pre_comm,   live_idx_pre,   Z_pre_hh   = make_filtered(leap_pre_codes, hh_idx)

    nat_codes_filt = nat_codes[:,     np.where((leap_codes     > 0).sum(axis=0) >= min_activations)[0]]
    nat_pre_filt   = nat_pre_codes[:, np.where((leap_pre_codes > 0).sum(axis=0) >= min_activations)[0]]

    # Append spectral indices + community-level variables to Z
    COMMUNITY_NAMES = [W_LABELS.get(c, c) for c in COMMUNITY_Z]
    Z_codes_hh = np.concatenate([Z_codes_hh, spectral_hh, community_hh], axis=1)
    Z_pre_hh   = np.concatenate([Z_pre_hh,   spectral_hh, community_hh], axis=1)
    z_names_codes = [None] * len(live_idx_codes) + SPECTRAL_NAMES + COMMUNITY_NAMES
    z_names_pre   = [None] * len(live_idx_pre)   + SPECTRAL_NAMES + COMMUNITY_NAMES

    return dict(
        y=y, t=t, W=W, W_NAMES=W_NAMES, cluster=cluster,
        spectral_names=SPECTRAL_NAMES,
        codes=dict(Z_hh=Z_codes_hh, Z_comm=leap_codes_comm,
                   comm_ids=leap_ids, live_idx=live_idx_codes,
                   z_names=z_names_codes),
        pre_codes=dict(Z_hh=Z_pre_hh, Z_comm=leap_pre_comm,
                       comm_ids=leap_ids, live_idx=live_idx_pre,
                       z_names=z_names_pre),
        nat_codes=dict(Z_nat=nat_codes_filt, nat_ids=nat_ids),
        nat_pre=dict(  Z_nat=nat_pre_filt,   nat_ids=nat_ids),
    )


# ── NEXIS / marginal runners ───────────────────────────────────────────────────

def run_nexis(rep_mode: str, method_name: str, data: dict,
              alpha: float = 0.05, adjust=None) -> SelectionResult:
    cfg = data[rep_mode]
    y, t, W, W_NAMES = data["y"], data["t"], data["W"], data["W_NAMES"]
    print(f"\n{'='*60}")
    print(f"NEXIS  rep={rep_mode}  method={method_name}  Z={cfg['Z_hh'].shape}  adjust={adjust}")
    print(f"{'='*60}")
    res = nexis(y, t, cfg["Z_hh"], w=W, w_names=W_NAMES, z_names=cfg["z_names"],
               alpha=alpha, adjust=adjust, cluster=data["cluster"], verbose=True)
    print(f"\nSelected: {len(res.selected)}")
    for i in res.selected:
        name = res.feature_names[i]
        if name.startswith("z_") and name[2:].isdigit():
            suffix = f"  [neuron {cfg['live_idx'][int(name[2:])]}]"
        else:
            suffix = ""
        print(f"  {name + suffix:55s}  p = {res.pvalues[i]:.4f}")
    _save_result(rep_mode, method_name, res, cfg["live_idx"], adjust)
    return res


def run_marginal_w(data: dict, alpha: float = 0.05, adjust=None) -> SelectionResult:
    y, t, W, W_NAMES = data["y"], data["t"], data["W"], data["W_NAMES"]
    print(f"\n{'='*60}")
    print(f"Marginal  on W  shape={W.shape}  adjust={adjust}")
    print(f"{'='*60}")
    res = marginal_select(y, t, W, alpha=alpha, adjust=adjust)
    print(f"\nSelected: {len(res.selected)}")
    for j in res.selected:
        print(f"  {W_NAMES[j]:40s}  p = {res.pvalues[j]:.4f}")
    out_dir = RES_DIR / "marginal_w"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "marginal_w", "adjust": str(adjust),
        "selected": [
            {"idx": int(j), "name": W_NAMES[j], "pvalue": float(res.pvalues[j])}
            for j in res.selected
        ],
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_dir / 'result.json'}")
    return res


def run_marginal_grouped(rep_mode: str, data: dict, alpha: float = 0.05) -> SelectionResult:
    """Marginal FWER with per-group correction: survey / SAE neurons / spectral."""
    y, t, W, W_NAMES = data["y"], data["t"], data["W"], data["W_NAMES"]
    cfg = data[rep_mode]
    Z_hh = cfg["Z_hh"]
    live_idx = cfg["live_idx"]
    spectral_names = data["spectral_names"]
    n_w   = W.shape[1]
    n_sae = len(live_idx)
    n_sp  = len(spectral_names)

    WZ = np.concatenate([W, Z_hh], axis=1)
    groups = {
        "survey":   list(range(n_w)),
        "neurons":  list(range(n_w, n_w + n_sae)),
        "spectral": list(range(n_w + n_sae, n_w + n_sae + n_sp)),
    }

    print(f"\n{'='*60}")
    print(f"Marginal grouped FWER  rep={rep_mode}  groups: "
          f"survey={n_w}, neurons={n_sae}, spectral={n_sp}")
    print(f"{'='*60}")
    res = marginal_select(y, t, WZ, alpha=alpha, adjust="FWER", groups=groups)

    # Build feature name lookup
    def _feat_name(idx):
        if idx < n_w:
            return f"w_{W_NAMES[idx]}"
        elif idx < n_w + n_sae:
            j = idx - n_w
            return f"z_{j}"
        else:
            return f"z_{spectral_names[idx - n_w - n_sae]}"

    print(f"\nSelected: {len(res.selected)}")
    for i in res.selected:
        name = _feat_name(i)
        if name.startswith("z_") and name[2:].isdigit():
            suffix = f"  [neuron {live_idx[int(name[2:])]}]"
        else:
            suffix = ""
        print(f"  {name + suffix:55s}  p = {res.pvalues[i]:.4f}")

    out_dir = RES_DIR / rep_mode / "marginal_grouped"
    out_dir.mkdir(parents=True, exist_ok=True)
    z_entries, w_entries = [], []
    for i in res.selected:
        name = _feat_name(i)
        if name.startswith("w_"):
            w_entries.append({"label": name[2:], "pvalue": float(res.pvalues[i])})
        else:
            suffix = name[2:]
            if suffix.isdigit():
                j = int(suffix)
                z_entries.append({"filtered_idx": j, "neuron_idx": int(live_idx[j]),
                                   "name": name, "pvalue": float(res.pvalues[i])})
            else:
                z_entries.append({"name": name, "pvalue": float(res.pvalues[i])})
    payload = {
        "rep_mode": rep_mode, "method": "marginal_grouped", "adjust": "FWER_grouped",
        "groups": {k: len(v) for k, v in groups.items()},
        "selected_z": z_entries, "selected_w": w_entries,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {out_dir / 'result.json'}")
    return res


def _save_result(rep_mode: str, method_name: str, res: SelectionResult,
                 live_idx: np.ndarray, adjust) -> None:
    out_dir = RES_DIR / rep_mode / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    z_entries, w_entries = [], []
    for i in res.selected:
        name = res.feature_names[i] if res.feature_names else f"z_{i}"
        if name.startswith("z_"):
            suffix = name[2:]
            if suffix.isdigit():
                j = int(suffix)
                z_entries.append({"filtered_idx": j, "neuron_idx": int(live_idx[j]),
                                   "name": name, "pvalue": float(res.pvalues[i])})
            else:
                z_entries.append({"name": name, "pvalue": float(res.pvalues[i])})
        else:
            w_entries.append({"label": name, "pvalue": float(res.pvalues[i])})
    with open(out_dir / "result.json", "w") as f:
        json.dump({"rep_mode": rep_mode, "method": method_name, "adjust": str(adjust),
                   "selected_z": z_entries, "selected_w": w_entries}, f, indent=2)
    print(f"Saved → {out_dir / 'result.json'}")


# ── VLM interpretation ─────────────────────────────────────────────────────────

def interpret_nexis(
    rep_mode: str,
    method_name: str,
    res: SelectionResult,
    data: dict,
    contrast_fn,          # callable(feat_idx, top_imgs, top_ids, top_acts,
                          #           bot_imgs, bot_ids, bot_acts,
                          #           prior_concepts, max_act, n_total, n_nonzero,
                          #           max_act_pct) -> dict
    pipeline: str,
    model_tag: str = "",  # stored in output for provenance
    k: int = 8,
    min_activation: float = 0.001,
    overwrite: bool = False,
    neuron_filter: set = None,   # if set, only interpret these neuron_idx values
    pool: str = "national",      # "national" or "leap"
) -> None:
    out_dir = RES_DIR / rep_mode / method_name / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    interp_path = out_dir / "interpretations.json"
    tag = f"{rep_mode}/{method_name}"

    if not overwrite and interp_path.exists():
        print(f"  [{tag}] Skipping — {interp_path} exists (use --overwrite)")
        return

    cfg      = data[rep_mode]
    live_idx = cfg["live_idx"]

    if pool == "leap":
        Z_pool    = cfg["Z_comm"]        # (n_comm, n_live)
        pool_ids  = cfg["comm_ids"]
        load_group_fn = load_group_community
        pool_label = f"{len(pool_ids)} LEAP community tiles"
    else:
        nat_key   = "nat_codes" if rep_mode == "codes" else "nat_pre"
        nat_cfg   = data[nat_key]
        Z_pool    = nat_cfg["Z_nat"]
        pool_ids  = nat_cfg["nat_ids"]
        load_group_fn = load_group_national
        pool_label = f"{len(pool_ids)} national tiles"

    # SAE neurons only — spectral z features (non-numeric suffix) have no TIF
    z_feats = [
        (int(res.feature_names[i][2:]), float(res.pvalues[i]))
        for i in res.selected
        if res.feature_names[i].startswith("z_") and res.feature_names[i][2:].isdigit()
        and (neuron_filter is None or int(live_idx[int(res.feature_names[i][2:])]) in neuron_filter)
    ]
    if not z_feats:
        print(f"  [{tag}] No SAE neuron features selected — nothing to interpret.")
        return

    print(f"\n── Interpreting {tag}: {len(z_feats)} neuron(s)  (pool: {pool_label}) ──")
    all_max_acts = Z_pool.max(axis=0)
    n_total = Z_pool.shape[0]
    interpretations = []

    for j, pval in z_feats:
        neuron = int(live_idx[j])
        acts   = Z_pool[:, j]

        top_k_idx = np.argsort(acts)[::-1][:k]
        bot_k_idx = np.argsort(acts)[:k]

        top_ids  = pool_ids[top_k_idx].tolist()
        top_acts = acts[top_k_idx].tolist()
        bot_ids  = pool_ids[bot_k_idx].tolist()
        bot_acts = acts[bot_k_idx].tolist()

        max_act   = float(top_acts[0])
        n_nonzero = int(np.sum(acts > 0))
        act_pct   = int(np.mean(all_max_acts <= max_act) * 100)

        print(f"  neuron {neuron}  (z_{j})  max={max_act:.3f}  "
              f"nonzero={n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%)  p={pval:.4f}")

        if max_act < min_activation:
            print(f"    (skipped: max={max_act:.4f} < threshold)")
            interpretations.append({
                "feature": j, "neuron_idx": neuron, "pvalue": pval,
                "label": "low activation — uninterpretable", "confidence": "low",
                "activated_concept": "", "not_activated_concept": "",
                "model_tag": model_tag, "pipeline": pipeline, "raw": "",
                "top_ids": top_ids, "bot_ids": bot_ids,
            })
            continue

        img_mode = "fc" if pipeline == "geochat_llm" else "both"
        top_imgs, top_valid, top_valid_acts = load_group_fn(top_ids, top_acts, mode=img_mode)
        bot_imgs, bot_valid, bot_valid_acts = load_group_fn(bot_ids, bot_acts, mode=img_mode)
        if not top_imgs or not bot_imgs:
            print("    (skipped: could not load images)")
            continue

        res_c = contrast_fn(
            feat_idx=j,
            top_imgs=top_imgs, top_ids=top_valid, top_acts=top_valid_acts,
            bot_imgs=bot_imgs, bot_ids=bot_valid,  bot_acts=bot_valid_acts,
            prior_concepts=interpretations,
            max_act=max_act, n_total=n_total, n_nonzero=n_nonzero, max_act_pct=act_pct,
            _cache_key=(rep_mode, method_name, j),
        )
        res_c.update({"neuron_idx": neuron, "pvalue": pval,
                      "model_tag": model_tag, "pipeline": pipeline})

        dup = _label_is_duplicate(res_c.get("label", ""), interpretations)
        if dup:
            print(f"    [!] Duplicate label — indistinguishable from neuron "
                  f"{dup.get('neuron_idx','?')}")
            res_c["label"]             = f"indistinguishable from neuron {dup.get('neuron_idx','?')}"
            res_c["activated_concept"] = f"Could not distinguish from: {dup.get('label','')}"
            res_c["confidence"]        = "low"

        interpretations.append(res_c)
        print(f"    [{res_c.get('confidence','?')}] {res_c.get('label','?')}")
        print(f"      Active:   {res_c.get('activated_concept','')[:120]}")
        if res_c.get("not_activated_concept"):
            print(f"      Inactive: {res_c['not_activated_concept'][:120]}")
        print()

    with open(interp_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"  [{tag}] Saved → {interp_path}")
    print(f"\n  {'Neuron':>8}  {'Label':<40}  Conf")
    print("  " + "-" * 58)
    for r in interpretations:
        print(f"  {r.get('neuron_idx','?'):>8}  {r.get('label','?'):<40}  "
              f"{r.get('confidence','?')}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_result_from_json(rep_mode: str, method_name: str):
    """Reconstruct a minimal SelectionResult from a saved result.json."""
    path = RES_DIR / rep_mode / method_name / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved result at {path} — run stats first.")
    with open(path) as f:
        d = json.load(f)

    class _MockResult:
        pass

    res = _MockResult()
    # Rebuild selected indices, feature_names, pvalues as parallel lists
    # Use a dict so pvalues[i] works by integer index
    feature_names = {}
    pvalues = {}
    selected = []
    idx = 0
    for entry in d.get("selected_w", []):
        feature_names[idx] = entry["label"]
        pvalues[idx] = entry["pvalue"]
        selected.append(idx)
        idx += 1
    for entry in d.get("selected_z", []):
        feature_names[idx] = f"z_{entry['filtered_idx']}"
        pvalues[idx] = entry["pvalue"]
        selected.append(idx)
        idx += 1
    res.selected = selected
    res.feature_names = feature_names
    res.pvalues = pvalues
    return res


def _collect_geochat_descriptions(
    rep_mode, method_name, res, data,
    geo_model, geo_processor,
    desc_cache: dict, k: int, neuron_filter=None,
):
    """Pass-1 helper: run GeoChat on all images for this result, store in desc_cache."""
    cfg     = data[rep_mode]
    nat_key = "nat_codes" if rep_mode == "codes" else "nat_pre"
    Z_nat   = data[nat_key]["Z_nat"]
    nat_ids = data[nat_key]["nat_ids"]
    live_idx = cfg["live_idx"]

    z_feats = [
        (int(res.feature_names[i][2:]), float(res.pvalues[i]))
        for i in res.selected
        if res.feature_names[i].startswith("z_") and res.feature_names[i][2:].isdigit()
    ]
    for j, _ in z_feats:
        neuron = int(live_idx[j])
        if neuron_filter and neuron not in neuron_filter:
            continue
        key = (rep_mode, method_name, j)
        if key in desc_cache:
            continue
        acts = Z_nat[:, j]
        top_ids = nat_ids[np.argsort(acts)[::-1][:k]].tolist()
        bot_ids = nat_ids[np.argsort(acts)[:k]].tolist()
        top_acts = acts[np.argsort(acts)[::-1][:k]].tolist()
        bot_acts = acts[np.argsort(acts)[:k]].tolist()
        top_imgs, _, _ = load_group_national(top_ids, top_acts, mode="fc")
        bot_imgs, _, _ = load_group_national(bot_ids, bot_acts, mode="fc")
        if not top_imgs or not bot_imgs:
            continue
        print(f"  GeoChat [{rep_mode}/{method_name}] neuron {neuron}: "
              f"describing {len(top_imgs)}+{len(bot_imgs)} images ...")
        top_descs = [geochat_describe_image(geo_model, geo_processor, img) for img in top_imgs]
        bot_descs = [geochat_describe_image(geo_model, geo_processor, img) for img in bot_imgs]
        desc_cache[key] = (top_descs, bot_descs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["codes", "pre_codes", "both"], default="both")
    p.add_argument("--pipeline", default="qwen72b",
                   help="'qwen72b' (default) or 'geochat_llm'.")
    # qwen72b options
    p.add_argument("--vlm-model", default="Qwen/Qwen2-VL-72B-Instruct",
                   help="HuggingFace model id for the qwen72b pipeline.")
    # geochat_llm options
    p.add_argument("--geochat-model", default="MBZUAI/geochat-7B",
                   help="GeoChat model id for per-image descriptions.")
    p.add_argument("--synthesis-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Text LLM model id for contrast synthesis.")
    # shared
    p.add_argument("--quantize", action="store_true",
                   help="Load model(s) in 4-bit via BitsAndBytes.")
    p.add_argument("--k", type=int, default=8,
                   help="Top/bottom tiles shown per neuron.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-interpret", action="store_true",
                   help="Run only NEXIS/marginal stats, skip interpretation.")
    p.add_argument("--interpret-only", action="store_true",
                   help="Skip stats, load saved results and run interpretation only.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run interpretation even if interpretations.json exists.")
    p.add_argument("--neurons", default=None,
                   help="Comma-separated neuron_idx values to interpret, e.g. 1777,3821. "
                        "If omitted all selected neurons are interpreted.")
    p.add_argument("--method", default=None,
                   help="Single method name to run (e.g. nexis_no_adj_hc1). "
                        "Overrides the default [nexis_no_adj, nexis_fdr] list. "
                        "Requires a saved result.json at results/ghana/{rep}/{method}/.")
    p.add_argument("--pool", default="national", choices=["national", "leap"],
                   help="Image pool for VLM contrast: 'national' (default, 9592 tiles) "
                        "or 'leap' (162 RCT community tiles).")
    p.add_argument("--min-activations", type=int, default=10,
                   help="Min community activations for a neuron to enter Z (default 10).")
    return p.parse_args()


def main():
    import functools, gc
    args = parse_args()
    global RES_DIR
    RES_DIR = ROOT / "results" / "ghana" / f"mact{args.min_activations}"
    rep_modes = ["codes", "pre_codes"] if args.mode == "both" else [args.mode]
    neuron_filter = (
        {int(x) for x in args.neurons.split(",")} if args.neurons else None
    )
    if neuron_filter:
        print(f"Neuron filter: {sorted(neuron_filter)}")

    nexis_methods = (
        [(args.method, None)]
        if args.method
        else [("nexis_no_adj", None), ("nexis_fdr", "FDR"), ("nexis_fwer", "FWER")]
    )

    if args.interpret_only:
        print("Loading data for image pool ...")
        data = _load_nexis_inputs(args.min_activations)
        nexis_results = {}
        for rep_mode in rep_modes:
            nexis_results[rep_mode] = {}
            for method_name, _ in nexis_methods:
                nexis_results[rep_mode][method_name] = _load_result_from_json(
                    rep_mode, method_name
                )
                z_count = sum(
                    1 for i in nexis_results[rep_mode][method_name].selected
                    if nexis_results[rep_mode][method_name].feature_names[i].startswith("z_")
                )
                print(f"  Loaded {rep_mode}/{method_name}: "
                      f"{len(nexis_results[rep_mode][method_name].selected)} selected "
                      f"({z_count} Z neurons)")
    else:
        print("Loading data and computing activations ...")
        data = _load_nexis_inputs(args.min_activations)

        nexis_results = {}
        for rep_mode in rep_modes:
            nexis_results[rep_mode] = {}
            for method_name, adjust in nexis_methods:
                nexis_results[rep_mode][method_name] = run_nexis(
                    rep_mode, method_name, data, alpha=args.alpha, adjust=adjust
                )

        run_marginal_w(data, alpha=args.alpha, adjust=None)
        for rep_mode in rep_modes:
            run_marginal_grouped(rep_mode, data, alpha=args.alpha)

        if args.no_interpret:
            return

    # ── Build contrast_fn based on selected pipeline ───────────────────────────
    if args.pipeline == "geochat_llm":
        # Two-pass to avoid simultaneous VRAM usage of two 7B models:
        # Pass 1 — GeoChat describes every image; results cached in desc_cache.
        # Pass 2 — Synthesis LLM reads cached descriptions and produces labels.
        print(f"\nPipeline: geochat_llm  (two-pass, sequential model loading)")
        model_tag = f"geochat={args.geochat_model} | synth={args.synthesis_model}"

        # ── Pass 1: GeoChat descriptions ──────────────────────────────────────
        print(f"\nPass 1: loading GeoChat for image descriptions ...")
        geo_model, geo_processor = load_geochat(args.geochat_model, quantize=args.quantize)
        desc_cache = {}  # (rep_mode, method_name, feat_idx) -> (top_descs, bot_descs)
        for rep_mode in rep_modes:
            for method_name, _ in nexis_methods:
                _collect_geochat_descriptions(
                    rep_mode, method_name,
                    nexis_results[rep_mode][method_name],
                    data, geo_model, geo_processor,
                    desc_cache, k=args.k, neuron_filter=neuron_filter,
                )
        del geo_model, geo_processor
        gc.collect(); torch.cuda.empty_cache()
        print("\nGeoChat unloaded.")

        # ── Pass 2: synthesis LLM ─────────────────────────────────────────────
        print(f"\nPass 2: loading synthesis LLM ...")
        lm, tokenizer = load_text_llm(args.synthesis_model, quantize=args.quantize)
        contrast_fn = functools.partial(
            contrast_groups_geochat_llm, None, None, lm, tokenizer,
            _desc_cache=desc_cache,
        )
        models_to_free = [lm]
    else:
        # default: qwen72b
        print(f"\nPipeline: {args.pipeline}  model: {args.vlm_model}")
        vlm_model, vlm_processor = load_vlm(args.vlm_model, quantize=args.quantize)
        model_tag = args.vlm_model
        contrast_fn = functools.partial(
            contrast_groups_vlm, vlm_model, vlm_processor
        )
        models_to_free = [vlm_model]

    for rep_mode in rep_modes:
        for method_name, _ in nexis_methods:
            interpret_nexis(
                rep_mode=rep_mode,
                method_name=method_name,
                res=nexis_results[rep_mode][method_name],
                data=data,
                contrast_fn=contrast_fn,
                pipeline=args.pipeline,
                model_tag=model_tag,
                k=args.k,
                overwrite=args.overwrite,
                neuron_filter=neuron_filter,
                pool=args.pool,
            )

    for m in models_to_free:
        del m
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
