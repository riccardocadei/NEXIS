"""All paths, grids, seeds and data-generating-process (DGP) parameters, in one place.

Paths default to ./data and ./results next to this file and can be moved with the
environment variables NEXIS_DATA_DIR and NEXIS_RESULTS_DIR (or --data-dir /
--results-dir on the command line).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("NEXIS_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("NEXIS_RESULTS_DIR", ROOT / "results"))

# ── public assets ─────────────────────────────────────────────────────────────
# CelebA (Liu et al., 2015), aligned-and-cropped images with the official splits, from
# the public Hugging Face mirror; "valid" is the official validation split (19,867 images).
HF_DATASET = "flwrlabs/celeba"
HF_DATASET_REVISION = "2d738f56e0e7f925ea36ae7c808ea925264aacec"
POOL_SPLIT, POOL_SIZE = "valid", 19_867
# SigLIP 2 ViT-B/16 (Tschannen et al., 2025) in timm, pretrained tag pinned explicitly.
BACKBONE = "vit_base_patch16_siglip_224.v2_webli"
IMG_SIZE, N_PATCHES, EMBED_DIM = 224, 196, 768
IMG_MEAN = IMG_STD = (0.5, 0.5, 0.5)
EMBED_BATCH = 128

# ── TopK sparse autoencoders (overcomplete library) ───────────────────────────
SAE = dict(hidden_dim=9_216, epochs=20, batch_images=20, lr=5e-4, seed=0)
SAE_KS = (5, 20)
# Replica dictionary: same architecture and schedule, trained on an independent random
# sample of POOL_SIZE train-split images (identity-disjoint from the validation split),
# then used to encode the same validation pool.
REPLICA = dict(split="train", sample_n=POOL_SIZE, sample_seed=1, ks=(20,))


def dictionary_name(k: int, replica: bool = False) -> str:
    return f"{'replica' if replica else 'main'}_k{k}"


# Artefacts (all under DATA_DIR)
POOL_DIR = DATA_DIR / "celeba"                            # the validation pool
REPLICA_CORPUS_DIR = DATA_DIR / "celeba_train_sample"     # the replica's SAE training images
MODELS_DIR = DATA_DIR / "models"
LABELS = POOL_DIR / "labels.parquet"
IMAGES = POOL_DIR / "images.npy"      # (N, 128, 128, 3) uint8 thumbnails, for pa_top_images.pdf


def embeddings_path(corpus_dir: Path) -> Path:            # (N, 768) mean-pooled tokens
    return corpus_dir / "embeddings" / "siglip.npy"


def patches_path(corpus_dir: Path) -> Path:               # (N, 196, 768) float16, raw memmap
    return corpus_dir / "embeddings" / "siglip_patches.f16"


def sae_checkpoint(k: int, replica: bool = False) -> Path:
    return MODELS_DIR / f"sae_{dictionary_name(k, replica)}.pt"


def codes_path(k: int, replica: bool = False, view: str = "z") -> Path:
    """Encoded pool: view "z" (sparse post-TopK codes) or "z_pre" (pre-activations)."""
    stem = {"z": "sae", "z_pre": "sae_precode"}[view]
    return POOL_DIR / "embeddings" / f"{stem}{'_replica' if replica else ''}_k{k}.npy"


def ground_truth_path(k: int, replica: bool = False) -> Path:
    return POOL_DIR / "ground_truth" / f"{dictionary_name(k, replica)}.json"


# ── benchmark design (Appendix C.1) ───────────────────────────────────────────
ALPHA = 0.05
MAX_ROUNDS = 10              # cap on NEXIS rounds
GCM_SPLITS = 3               # cross-fitting folds of the GCM / PCM nuisances
N_SEEDS = 50                 # Monte Carlo replications per (n, eta) cell; seeds 0..N-1
EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
FIXED_N = [500, 2000]        # n of the effect-size sweeps
FIXED_EFFECT = [2.0, 5.0]    # eta of the sample-size sweeps

# Smoke test: few seeds and grid points, every figure row still present.
SMOKE = dict(n_seeds=2, effect_grid=[2.0, 5.0], n_grid=[100, 500])

# DGP (Eq. C.1):  W_k ~ Bernoulli(p_k) independently, T ~ Bernoulli(p_treat), an image is
# drawn without replacement from the CelebA cell matching (W_1, ..., W_r), and
#   Y = sum_k beta_k W_k + T [tau_0 + eta sum_k gamma_k W_k] + N(0, noise_sd^2).
# The direct-modifier set S* is the principal coordinate of every attribute with gamma != 0.
DGP_MAIN = dict(attrs=["Wearing_Hat", "Eyeglasses"], gammas=[1.0, -1.0], betas=[0.3, -0.2],
                tau0=0.5, noise_sd=1.0, p_treat=0.5, effect_form="attr", eta_scales="gamma")

# r = 1: Wearing_Hat modifies the effect; Eyeglasses is sampled as in the main setting but
# is prognostic only.  Same units, T and images as the main setting for every seed.
DGP_R1 = {**DGP_MAIN, "gammas": [1.0, 0.0]}

# r = 3: the attribute set and interactions of the three-modifier DGP (edit here).  The
# third attribute needs one dominant SAE coordinate and enough images in the joint cell
# with the other two (the run fails loudly if a cell is exhausted).
R3_ATTRS = ["Wearing_Hat", "Eyeglasses", "Sideburns"]
R3_GAMMAS = [1.0, -1.0, 1.0]
R3_BETAS = [0.3, -0.2, 0.3]
DGP_R3 = {**DGP_MAIN, "attrs": R3_ATTRS, "gammas": R3_GAMMAS, "betas": R3_BETAS}

# r = 0: no effect modification, S* = {}.  The sweep layout is a choice (edit here):
#   "fixed_beta"  prognostic effects fixed at the main setting's betas; eta multiplies a
#                 zero interaction, so only the n sweep runs (eta = 1 is a placeholder),
#                 with R0_SEEDS seeds per n so P(any false discovery) is read against alpha.
#   "scale_beta"  eta multiplies the prognostic main effects instead (Y = eta sum_k beta_k
#                 W_k + tau_0 T + noise); the four rows of the main grid, N_SEEDS seeds.
R0_MODE = "fixed_beta"
R0_SEEDS = 200
DGP_R0 = {**DGP_MAIN, "gammas": [0.0, 0.0],
          "eta_scales": "gamma" if R0_MODE == "fixed_beta" else "beta"}

# U-shape: tau = tau_0 + eta (gamma_1 g(Z^{j1}) + gamma_2 g(Z^{j2})) on the two principal
# coordinates, g the quadratic orthogonalised against [1, Z^j] on the whole pool (zero
# covariance with Z^j): invisible to the linear test and the GCM, visible to the PCM.
DGP_USHAPE = {**DGP_MAIN, "effect_form": "ortho_quadratic"}

# Controlled violation of Principal Alignment: the main DGP, with the principal coordinate
# j1 of `attr` split in two complementary halves.  U ~ Bernoulli(share) is drawn once per
# pool image (seed `seed`), Z^{j1} <- U Z^{j1}, and a new coordinate Z^{j_new} = (1 - U)
# Z^{j1} is appended.  Neither half screens the modifier off alone, so S* = {j1, j_new, j2}.
# W, T, the images and Y are those of the main setting, which is the no-split control.
DGP_VIOLATION = {**DGP_MAIN, "split": dict(attr="Wearing_Hat", share=0.5, seed=0)}

# ── methods ───────────────────────────────────────────────────────────────────
BASELINES = ["Marginal Testing", "Marginal Testing (FWER)", "Marginal Testing (FDR)"]
NEXIS_DEFAULT = dict(test="linear", adjust="FWER", rho=0.5,
                     interleaved_backward=False, terminal_backward=True)
# Each variant deviates from NEXIS_DEFAULT along one axis.
NEXIS_VARIANTS = {
    "NEXIS": {},
    "NEXIS (test=GCM: quadratic)": dict(test="GCM: quadratic"),
    "NEXIS (test=GCM: lgbm)": dict(test="GCM: lgbm"),
    "NEXIS (test=PCM: quadratic)": dict(test="PCM: quadratic"),
    "NEXIS (test=PCM: lgbm)": dict(test="PCM: lgbm"),
    "NEXIS (adjust=None)": dict(adjust=None),
    "NEXIS (adjust=FDR)": dict(adjust="FDR"),
    "NEXIS (rho=0)": dict(rho=0.0),
    "NEXIS (rho=0.2)": dict(rho=0.2),
    "NEXIS (rho=0.8)": dict(rho=0.8),
    "NEXIS (forward)": dict(terminal_backward=False),
    "NEXIS (forward + interleaved backward)":
        dict(interleaved_backward=True, terminal_backward=False),
    "NEXIS (forward + interleaved and terminal backward)": dict(interleaved_backward=True),
}
TEST_VARIANTS = [m for m in NEXIS_VARIANTS if m.startswith("NEXIS (test=")]
METHOD_VARIANTS = [m for m in NEXIS_VARIANTS if m != "NEXIS"]

# ── experiments ───────────────────────────────────────────────────────────────
# dictionary: SAE (k, replica); view: "z" or "z_pre"; sweeps: which sweeps run and at
# which fixed values; methods: evaluated on every draw.
_GRID = dict(sweeps={"effect": FIXED_N, "n": FIXED_EFFECT}, n_seeds=N_SEEDS)
EXPERIMENTS = {
    "main":          dict(k=20, replica=False, view="z", dgp=DGP_MAIN, **_GRID,
                          methods=BASELINES + ["NEXIS"]),
    "model_k5":      dict(k=5, replica=False, view="z", dgp=DGP_MAIN, **_GRID,
                          methods=BASELINES + ["NEXIS"]),
    "model_precode": dict(k=20, replica=False, view="z_pre", dgp=DGP_MAIN, **_GRID,
                          methods=BASELINES + ["NEXIS"]),
    "method":        dict(k=20, replica=False, view="z", dgp=DGP_MAIN, **_GRID,
                          methods=METHOD_VARIANTS),      # the default line comes from "main"
    # the interleaved-backward arm gives the replica IoU without the terminal step
    "replica":       dict(k=20, replica=True, view="z", dgp=DGP_MAIN, **_GRID,
                          methods=BASELINES + ["NEXIS", "NEXIS (forward + interleaved backward)"]),
    "dgp_r1":        dict(k=20, replica=False, view="z", dgp=DGP_R1, **_GRID,
                          methods=BASELINES + ["NEXIS"]),
    "dgp_r3":        dict(k=20, replica=False, view="z", dgp=DGP_R3, **_GRID,
                          methods=BASELINES + ["NEXIS"]),
    "dgp_r0":        dict(k=20, replica=False, view="z", dgp=DGP_R0,
                          sweeps={"n": [1.0]} if R0_MODE == "fixed_beta"
                          else {"effect": FIXED_N, "n": FIXED_EFFECT},
                          n_seeds=R0_SEEDS if R0_MODE == "fixed_beta" else N_SEEDS,
                          methods=BASELINES + ["NEXIS"]),
    "ushape":        dict(k=20, replica=False, view="z_pre", dgp=DGP_USHAPE,
                          sweeps={"effect": [2000], "n": [5.0]}, n_seeds=N_SEEDS,
                          methods=["Marginal Testing (FWER)", "NEXIS"] + TEST_VARIANTS),
    # rho = 0 shows whether the spectral-gap gate stops the search before a half enters
    "violation":     dict(k=20, replica=False, view="z", dgp=DGP_VIOLATION, **_GRID,
                          methods=BASELINES + ["NEXIS", "NEXIS (rho=0)"]),
}

# Blocks of the command line: experiments to run, then figures to draw.
BLOCKS = {
    "main":             dict(experiments=["main"], figures=["figure_main", "dgp"]),
    "ablations-model":  dict(experiments=["model_k5", "model_precode"],
                             figures=["model_k5", "model_precode"]),
    "ablations-method": dict(experiments=["main", "method"],
                             figures=["method_test", "method_adjust", "method_rho",
                                      "method_backward"]),
    "replica":          dict(experiments=["replica"], figures=["replica_k20"]),
    "dgp-extra":        dict(experiments=["dgp_r0", "dgp_r1", "dgp_r3"],
                             figures=["dgp_r0", "dgp_r1", "dgp_r3"]),
    "ushape":           dict(experiments=["ushape"], figures=["ushape"]),
    "alignment":        dict(experiments=[], figures=[]),     # celeba/alignment.py
    "violation":        dict(experiments=["main", "violation"], figures=["violation"]),
}

# Attributes whose principal coordinate is computed by `prepare` for every dictionary.
GT_ATTRS = sorted({a for e in EXPERIMENTS.values() for a in e["dgp"]["attrs"]})
