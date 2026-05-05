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

Usage
-----
  python src/apps/ghana/interpret.py --quantize
  python src/apps/ghana/interpret.py --mode codes --no-interpret
  python src/apps/ghana/interpret.py --overwrite   # redo VLM even if files exist
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
from src.apps.ghana.data import load_data, W_ALL, W_LABELS
from src.method.nexis    import nexis, marginal_select, SelectionResult


# ── Image loading ──────────────────────────────────────────────────────────────

def _false_colour(tif_path: Path, size: int = 224):
    """Load NIR→R / Green→G / SWIR2→B false-colour PIL image from a Landsat 8 TIF."""
    import rasterio
    from PIL import Image as PILImage
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        nir   = src.read(4).astype(np.float32)   # B5
        green = src.read(2).astype(np.float32)   # B3
        swir2 = src.read(6).astype(np.float32)   # B7
    def norm(b):
        valid = b[b > 0]
        if valid.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(valid, [2, 98])
        out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        out[b <= 0] = 0
        return out
    rgb = (np.stack([norm(nir), norm(green), norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return PILImage.fromarray(rgb).resize((size, size), PILImage.BICUBIC)


def load_national_image(grid_id: int, size: int = 224):
    return _false_colour(TIF_NAT / f"ghana_grid{int(grid_id):06d}.tif", size)


def load_group_national(grid_ids, activations, size: int = 224):
    imgs, valid_ids, valid_acts = [], [], []
    for gid, act in zip(grid_ids, activations):
        img = load_national_image(int(gid), size)
        if img is not None:
            imgs.append(img)
            valid_ids.append(int(gid))
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

COLOUR_GUIDE = """\
False-colour Landsat 8 composite — band assignment: NIR(B5)→Red · Green(B3)→Green · SWIR2(B7)→Blue.

Colour key:
  • Bright red / magenta   = dense healthy vegetation (high NIR: woodland, vigorous crops)
  • Tan / pinkish-brown    = bare soil, fallow, degraded or sparse land
  • Dark blue / near-black = open water (river, lake, reservoir)
  • Dark olive / very dark = wetland, seasonally flooded valley bottom
  • Grey / white patches   = settlement, metal roofs, compacted ground, roads
  • Pale laterite patches  = infertile hardpan (highly reflective bare surface)

Note: nearly every tile contains red/pink vegetation.
Overall redness alone is NOT a useful distinguishing feature — focus on structural differences.\
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
) -> dict:
    content = [
        {"type": "text", "text": COLOUR_GUIDE + "\n\n"},
        {"type": "text", "text":
            f"GROUP A — HIGH activation ({len(top_imgs)} tiles, strongest → weakest):\n"},
    ]
    for i, (img, act) in enumerate(zip(top_imgs, top_acts)):
        content.append({"type": "image", "image": img})
        pct = 100 * act / max_act if max_act > 0 else 0
        content.append({"type": "text", "text": f"[A{i+1} — {act:.4f} ({pct:.0f}% of max)]\n"})

    content.append({"type": "text", "text":
        f"\nGROUP B — LOW / ZERO activation ({len(bot_imgs)} tiles):\n"})
    for i, (img, act) in enumerate(zip(bot_imgs, bot_acts)):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[B{i+1} — {act:.4f}]\n"})

    # Prior-concept exclusion (mirrors NEXIS conditioning)
    if prior_concepts:
        lines = "\n".join(
            f"  • Neuron {p['feature']} — \"{p.get('label','?')}\""
            + (f": {p['activated_concept']}" if p.get('activated_concept') else "")
            for p in prior_concepts
        )
        excl = (
            f"\n\nALREADY IDENTIFIED (do NOT repeat or paraphrase):\n{lines}\n\n"
            "This neuron was selected conditional on the above — it captures "
            "something visually different. If you cannot find any distinction, "
            "write 'indistinguishable from prior concept'.\n"
        )
    else:
        excl = ""

    # Signal notes
    notes = []
    if n_total and n_nonzero:
        pct_nz = 100 * n_nonzero / n_total
        notes.append(
            f"SPARSITY: {n_nonzero}/{n_total} tiles ({pct_nz:.1f}%) have non-zero activation. "
            + ("Group A are rare outliers — look for a niche land-cover signature." if pct_nz < 10
               else "Moderately sparse feature.")
        )
    if max_act_pct is not None and max_act_pct <= 30:
        notes.append(
            f"WEAK SIGNAL (bottom {max_act_pct}% of neurons). "
            "Study Group A carefully for any recurring micro-pattern before comparing."
        )
    notes.append(
        f"GRADIENT: A1 = strongest; pattern may fade toward A{len(top_imgs)}. "
        "Describe what diminishes with activation."
    )

    content.append({"type": "text", "text":
        f"{excl}"
        + ("\n" + "\n".join(notes) + "\n\n" if notes else "\n")
        + "TASK: identify the ONE visual property that best explains why the neuron "
        "fires on Group A but not Group B.\n\n"
        "Work through these cues in order — stop at the FIRST clear asymmetry:\n\n"
        "  1. OPEN WATER / WETLAND\n"
        "     Dark blue-black = river, reservoir, lake.  Dark olive = wetland, flooded valley.\n\n"
        "  2. BURN SCARS\n"
        "     Dark red/maroon patches. Sharp-edged = recent; diffuse = older.\n\n"
        "  3. BARE SOIL / CROPLAND EXTENT AND GEOMETRY\n"
        "     Tan-pink patches. Regular rectangles = large-scale farming; "
        "small irregular = subsistence. More or less bare in one group?\n\n"
        "  4. SETTLEMENT AND ROAD DENSITY\n"
        "     Grey/white clusters = villages. Pale linear = roads/tracks. "
        "Denser or more connected in one group?\n\n"
        "  5. VEGETATION TEXTURE AND FRAGMENTATION\n"
        "     Continuous dense canopy vs. fragmented mosaic vs. fine speckled patchwork. "
        "Overall redness does NOT count — only texture and fragmentation.\n\n"
        "  6. If nothing structurally differs, state that explicitly.\n\n"
        "Do NOT hypothesise about economic causes or welfare effects. "
        "Only describe what is visually observable.\n\n"
        "Answer in EXACTLY this format (4 lines, no extra text):\n"
        "Active description: <5–15 words — what HIGH-activation tiles SHOW>\n"
        "Inactive description: <5–15 words — what LOW-activation tiles SHOW>\n"
        "Label: <2–6 words naming the distinguishing visual concept>\n"
        "Confidence: <low|medium|high>"
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

    result = {
        "feature": feat_idx, "raw": raw,
        "top_ids": list(top_ids), "bot_ids": list(bot_ids),
        "top_acts": list(top_acts), "bot_acts": list(bot_acts),
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


def _label_is_duplicate(new_label: str, prior: list, thr: float = 0.8):
    from difflib import SequenceMatcher
    new = new_label.lower().strip(" .")
    for p in prior:
        old = p.get("label", "").lower().strip(" .")
        if old and SequenceMatcher(None, new, old).ratio() >= thr:
            return p
    return None


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


def _load_nexis_inputs():
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
        df0.set_index("hhid")[["T", "comm"] + W_ALL + ["Y"]]
           .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
    )
    merged["dY"] = merged["Y1"] - merged["Y"]
    merged = merged.reset_index().merge(sp, on="comm", how="left").set_index("hhid")

    y = merged["dY"].values.astype(float)
    t = merged["T"].values.astype(float)
    W = merged[W_ALL].values.astype(float)
    W_NAMES = [W_LABELS.get(c, c) for c in W_ALL]
    spectral_hh = merged[SPECTRAL_COLS].values.astype(float)

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
        live_mask = (comm_acts > 0).sum(axis=0) >= 5
        live_idx  = np.where(live_mask)[0]
        return comm_acts[:, live_mask], live_idx, comm_acts[:, live_mask][hh_idx_arr]

    leap_codes_comm, live_idx_codes, Z_codes_hh = make_filtered(leap_codes,     hh_idx)
    leap_pre_comm,   live_idx_pre,   Z_pre_hh   = make_filtered(leap_pre_codes, hh_idx)

    nat_codes_filt = nat_codes[:,     np.where((leap_codes     > 0).sum(axis=0) >= 5)[0]]
    nat_pre_filt   = nat_pre_codes[:, np.where((leap_pre_codes > 0).sum(axis=0) >= 5)[0]]

    # Append spectral indices to Z (named z_{col}, not z_{number})
    Z_codes_hh = np.concatenate([Z_codes_hh, spectral_hh], axis=1)
    Z_pre_hh   = np.concatenate([Z_pre_hh,   spectral_hh], axis=1)
    z_names_codes = [None] * len(live_idx_codes) + SPECTRAL_COLS
    z_names_pre   = [None] * len(live_idx_pre)   + SPECTRAL_COLS

    return dict(
        y=y, t=t, W=W, W_NAMES=W_NAMES,
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
               alpha=alpha, adjust=adjust, verbose=True)
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
    vlm_model, vlm_processor,
    pipeline: str,
    k: int = 8,
    min_activation: float = 0.001,
    vlm_model_name: str = "",
    overwrite: bool = False,
) -> None:
    out_dir = RES_DIR / rep_mode / method_name / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    interp_path = out_dir / "interpretations.json"
    tag = f"{rep_mode}/{method_name}"

    if not overwrite and interp_path.exists():
        print(f"  [{tag}] Skipping — {interp_path} exists (use --overwrite)")
        return

    cfg     = data[rep_mode]
    nat_key = "nat_codes" if rep_mode == "codes" else "nat_pre"
    nat_cfg = data[nat_key]
    Z_nat   = nat_cfg["Z_nat"]
    nat_ids = nat_cfg["nat_ids"]
    live_idx = cfg["live_idx"]

    # SAE neurons only — spectral z features (non-numeric suffix) have no TIF to show
    z_feats = [
        (int(res.feature_names[i][2:]), float(res.pvalues[i]))
        for i in res.selected
        if res.feature_names[i].startswith("z_") and res.feature_names[i][2:].isdigit()
    ]
    if not z_feats:
        print(f"  [{tag}] No SAE neuron features selected — nothing to interpret.")
        return

    print(f"\n── Interpreting {tag}: {len(z_feats)} neuron(s)  (pool: {len(nat_ids)} national tiles) ──")
    all_max_acts = Z_nat.max(axis=0)
    n_total = Z_nat.shape[0]
    interpretations = []

    for j, pval in z_feats:
        neuron = int(live_idx[j])
        acts   = Z_nat[:, j]

        top_k_idx = np.argsort(acts)[::-1][:k]
        bot_k_idx = np.argsort(acts)[:k]

        top_ids  = nat_ids[top_k_idx].tolist()
        top_acts = acts[top_k_idx].tolist()
        bot_ids  = nat_ids[bot_k_idx].tolist()
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
                "vlm_model": vlm_model_name, "pipeline": pipeline, "raw": "",
                "top_ids": top_ids, "bot_ids": bot_ids,
            })
            continue

        top_imgs, top_valid, top_valid_acts = load_group_national(top_ids, top_acts)
        bot_imgs, bot_valid, bot_valid_acts = load_group_national(bot_ids, bot_acts)
        if not top_imgs or not bot_imgs:
            print("    (skipped: could not load images)")
            continue

        res_vlm = contrast_groups_vlm(
            vlm_model, vlm_processor, feat_idx=j,
            top_imgs=top_imgs, top_ids=top_valid, top_acts=top_valid_acts,
            bot_imgs=bot_imgs, bot_ids=bot_valid,  bot_acts=bot_valid_acts,
            prior_concepts=interpretations,
            max_act=max_act, n_total=n_total, n_nonzero=n_nonzero, max_act_pct=act_pct,
        )
        res_vlm.update({"neuron_idx": neuron, "pvalue": pval,
                        "vlm_model": vlm_model_name, "pipeline": pipeline})

        dup = _label_is_duplicate(res_vlm.get("label", ""), interpretations)
        if dup:
            print(f"    [!] Duplicate label — indistinguishable from neuron {dup.get('neuron_idx','?')}")
            res_vlm["label"]             = f"indistinguishable from neuron {dup.get('neuron_idx','?')}"
            res_vlm["activated_concept"] = f"Could not distinguish from: {dup.get('label','')}"
            res_vlm["confidence"]        = "low"

        interpretations.append(res_vlm)
        print(f"    [{res_vlm.get('confidence','?')}] {res_vlm.get('label','?')}")
        print(f"      Active:   {res_vlm.get('activated_concept','')[:120]}")
        if res_vlm.get("not_activated_concept"):
            print(f"      Inactive: {res_vlm['not_activated_concept'][:120]}")
        print()

    with open(interp_path, "w") as f:
        json.dump(interpretations, f, indent=2)
    print(f"  [{tag}] Saved → {interp_path}")
    print(f"\n  {'Neuron':>8}  {'Label':<40}  Conf")
    print("  " + "-" * 58)
    for r in interpretations:
        print(f"  {r.get('neuron_idx','?'):>8}  {r.get('label','?'):<40}  {r.get('confidence','?')}")


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
        feature_names[idx] = entry.get("name") or f"z_{entry['filtered_idx']}"
        pvalues[idx] = entry["pvalue"]
        selected.append(idx)
        idx += 1
    res.selected = selected
    res.feature_names = feature_names
    res.pvalues = pvalues
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["codes", "pre_codes", "both"], default="both")
    p.add_argument("--vlm-model", default="Qwen/Qwen2-VL-72B-Instruct")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--pipeline", default="qwen72b")
    p.add_argument("--k", type=int, default=8,
                   help="Top/bottom tiles shown to VLM per neuron.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-interpret", action="store_true",
                   help="Run only NEXIS/marginal stats, skip VLM.")
    p.add_argument("--interpret-only", action="store_true",
                   help="Skip stats, load saved results and run VLM only.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run VLM even if interpretations.json already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    rep_modes = ["codes", "pre_codes"] if args.mode == "both" else [args.mode]

    nexis_methods = [
        ("nexis_no_adj", None),
        ("nexis_fdr",    "FDR"),
    ]

    if args.interpret_only:
        # Load saved stats results and jump straight to VLM
        print("Loading data for VLM image pool ...")
        data = _load_nexis_inputs()
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
        data = _load_nexis_inputs()

        nexis_results = {}
        for rep_mode in rep_modes:
            nexis_results[rep_mode] = {}
            for method_name, adjust in nexis_methods:
                nexis_results[rep_mode][method_name] = run_nexis(
                    rep_mode, method_name, data, alpha=args.alpha, adjust=adjust
                )

        run_marginal_w(data, alpha=args.alpha, adjust=None)

        if args.no_interpret:
            return

    print(f"\nLoading VLM: {args.vlm_model}")
    vlm_model, vlm_processor = load_vlm(args.vlm_model, quantize=args.quantize)

    for rep_mode in rep_modes:
        for method_name, _ in nexis_methods:
            interpret_nexis(
                rep_mode=rep_mode,
                method_name=method_name,
                res=nexis_results[rep_mode][method_name],
                data=data,
                vlm_model=vlm_model, vlm_processor=vlm_processor,
                pipeline=args.pipeline,
                k=args.k,
                vlm_model_name=args.vlm_model,
                overwrite=args.overwrite,
            )

    import gc
    del vlm_model, vlm_processor
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
