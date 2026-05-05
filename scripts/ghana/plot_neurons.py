"""
Plot top-k / bottom-k national-grid tiles for each SAE neuron selected by NEXIS.

Usage (from repo root):
  python scripts/ghana/plot_neurons.py             # codes/nexis_no_adj, k=8
  python scripts/ghana/plot_neurons.py --k 12 --rep-mode pre_codes
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

ROOT     = Path(__file__).resolve().parents[2]
SAT_DIR  = ROOT / "data" / "ghana" / "satellite"
TIF_NAT  = SAT_DIR / "tif_national"
RES_DIR  = ROOT / "results" / "ghana"

import sys
sys.path.insert(0, str(ROOT))
from src.apps.ghana.visualize import plot_neuron_activation_map
from src.apps.ghana.data import load_data

DATA_DIR = ROOT / "data" / "ghana"


# ── image helpers ──────────────────────────────────────────────────────────────

def false_colour(tif_path: Path, size: int = 256):
    import rasterio
    from PIL import Image
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        nir   = src.read(4).astype(np.float32)
        green = src.read(2).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)
    def norm(b):
        v = b[b > 0]
        if v.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(v, [2, 98])
        out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        out[b <= 0] = 0
        return out
    rgb = (np.stack([norm(nir), norm(green), norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((size, size)))


# ── SAE activations ────────────────────────────────────────────────────────────

def compute_codes(embs, ckpt, wh_mean, wh_std, pre_codes=False):
    x = torch.from_numpy(((embs - wh_mean) / wh_std)).float()
    with torch.no_grad():
        acts = F.relu((x - ckpt["b_dec"]) @ ckpt["W_enc.weight"].T + ckpt["W_enc.bias"])
        if not pre_codes:
            topk = torch.topk(acts, 25, dim=-1).values[:, -1:]
            acts = acts * (acts >= topk)
    return acts.numpy()


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_neuron(neuron_idx, filtered_idx, acts_nat, nat_ids, pval,
                out_dir, k=8, rep_mode="codes"):
    col = acts_nat[:, filtered_idx]
    top_idx = np.argsort(col)[::-1][:k]
    bot_idx = np.argsort(col)[:k]
    max_act = float(col[top_idx[0]])

    fig = plt.figure(figsize=(k * 2.4, 7))
    fig.patch.set_facecolor("#0f0f0f")

    title = (f"Neuron {neuron_idx}  (z_{filtered_idx}, {rep_mode})  "
             f"p = {pval:.4f}   max_act = {max_act:.3f}   "
             f"non-zero = {int((col>0).sum())}/{len(col)}")
    fig.suptitle(title, color="white", fontsize=11, y=0.97)

    gs = gridspec.GridSpec(2, k, figure=fig, hspace=0.08, wspace=0.04,
                           top=0.91, bottom=0.02)

    for row, (indices, label, cmap_edge) in enumerate([
        (top_idx,  "HIGH activation", "#e05050"),
        (bot_idx,  "LOW / zero",      "#5080e0"),
    ]):
        for col_i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col_i])
            gid = int(nat_ids[idx])
            act = float(col[idx])
            img = false_colour(TIF_NAT / f"ghana_grid{gid:06d}.tif")
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "missing", color="white",
                        ha="center", va="center", transform=ax.transAxes, fontsize=6)

            pct = 100 * act / max_act if max_act > 0 else 0
            caption = f"{act:.3f}" if row == 0 else f"{act:.4f}"
            ax.set_title(caption, color="white", fontsize=7, pad=2)
            ax.set_xlabel(f"grid {gid}", color="#aaaaaa", fontsize=6, labelpad=1)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor(cmap_edge)
                spine.set_linewidth(1.6)

        # row label on left
        fig.text(0.005, 0.73 - row * 0.47, label, color=cmap_edge,
                 fontsize=9, fontweight="bold", va="center", rotation=90)

    out_path = out_dir / f"neuron_{neuron_idx}_top_bottom.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rep-mode", default="codes", choices=["codes", "pre_codes"])
    p.add_argument("--method",   default="nexis_no_adj")
    p.add_argument("--k",        type=int, default=8)
    args = p.parse_args()

    result_path = RES_DIR / args.rep_mode / args.method / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"No result at {result_path} — run stats first.")

    with open(result_path) as f:
        result = json.load(f)

    z_neurons = [e for e in result["selected_z"] if "neuron_idx" in e]
    if not z_neurons:
        print("No SAE neurons selected — nothing to plot.")
        return

    print(f"Loading national activations ({args.rep_mode}) ...")
    ckpt    = torch.load(SAT_DIR / "sae_model.pt", map_location="cpu", weights_only=False)
    wh_mean = np.load(SAT_DIR / "sae_whiten_mean.npy")
    wh_std  = np.load(SAT_DIR / "sae_whiten_std.npy")
    nat_embs = np.load(SAT_DIR / "national" / "prithvi_embeddings.npy")
    nat_ids  = np.load(SAT_DIR / "national" / "prithvi_comm_ids.npy")

    pre_codes = (args.rep_mode == "pre_codes")
    acts_nat = compute_codes(nat_embs, ckpt, wh_mean, wh_std, pre_codes=pre_codes)

    # Filter to live neurons (same mask as in interpret.py)
    leap_codes_raw = np.load(SAT_DIR / "sae_activations.npy")
    if pre_codes:
        from src.apps.ghana.interpret import _compute_sae_activations
        leap_embs = np.load(SAT_DIR / "prithvi_embeddings.npy")
        leap_raw = _compute_sae_activations(leap_embs, ckpt, wh_mean, wh_std, pre_codes=True)
    else:
        leap_embs = np.load(SAT_DIR / "prithvi_embeddings.npy")
        leap_raw = leap_codes_raw
    live_mask = (leap_raw > 0).sum(axis=0) >= 5
    acts_filt      = acts_nat[:,  live_mask]   # (9592, n_live) — national pool
    leap_acts_filt = leap_raw[:, live_mask]    # (162,  n_live) — LEAP communities

    leap_ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")

    print("Loading household data for activation map ...")
    df = load_data(DATA_DIR)

    for entry in z_neurons:
        neuron_idx   = entry["neuron_idx"]
        filtered_idx = entry["filtered_idx"]
        pval         = entry["pvalue"]

        out_dir = RES_DIR / "neurons" / f"neuron_{neuron_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(out_dir / "info.json", "w") as f:
            json.dump({
                "neuron_idx": neuron_idx, "filtered_idx": filtered_idx,
                "rep_mode": args.rep_mode, "method": args.method, "pvalue": pval,
            }, f, indent=2)

        print(f"\nNeuron {neuron_idx}  (z_{filtered_idx})  p={pval:.4f}")
        plot_neuron(neuron_idx, filtered_idx, acts_filt, nat_ids, pval,
                    out_dir=out_dir, k=args.k, rep_mode=args.rep_mode)

        # Activation map over the 162 LEAP communities
        comm_acts = leap_acts_filt[:, filtered_idx]
        fig_map, ax_map = plt.subplots(figsize=(4, 5))
        plot_neuron_activation_map(
            data_dir=DATA_DIR,
            df=df,
            comm_ids=leap_ids,
            activations=comm_acts,
            neuron_idx=neuron_idx,
            pvalue=pval,
            ax=ax_map,
        )
        map_path = out_dir / f"neuron_{neuron_idx}_activation_map.png"
        fig_map.savefig(map_path, dpi=150, bbox_inches="tight")
        plt.close(fig_map)
        print(f"  Saved → {map_path}")


if __name__ == "__main__":
    main()
