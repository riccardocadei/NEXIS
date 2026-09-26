"""Sweeps of the CelebA benchmark: draw datasets, run every method, score against S*.

The unit of work is (experiment, sweep, method).  Each unit writes one parquet file,
    <runs>/<experiment>/<sweep>_sweep/<method>.parquet,
with one row per (fixed value, grid value, seed): the selected coordinates and tp, fp,
precision, recall, IoU, |S_hat| and the method's wall time (for NEXIS also S~, the
selection before the terminal backward step, its size and the number of terminal tests).
Existing files are skipped unless overwrite=True, so interrupted runs resume and units can
be spread over jobs.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from nexis import iou_score, marginal_select, nexis
from celeba.scm import build_buckets, generate_rct, split_principal

SWEEP_PARAM = {"effect": "effect_scale", "n": "n"}
FIXED_COL = {"effect": "fixed_n", "n": "fixed_effect"}


# ── methods ───────────────────────────────────────────────────────────────────

def run_method(name: str, y, t, z, max_rounds: Optional[int] = None):
    """SelectionResult of one method (a key of config.BASELINES / NEXIS_VARIANTS).
    max_rounds caps the NEXIS forward rounds (None: no cap)."""
    if name == "Marginal Testing":
        return marginal_select(y, t, z, alpha=C.ALPHA, adjust=None)
    if name == "Marginal Testing (FWER)":
        return marginal_select(y, t, z, alpha=C.ALPHA, adjust="FWER")
    if name == "Marginal Testing (FDR)":
        return marginal_select(y, t, z, alpha=C.ALPHA, adjust="FDR")
    if name in C.NEXIS_VARIANTS:
        kw = {**C.NEXIS_DEFAULT, **C.NEXIS_VARIANTS[name]}
        return nexis(y, t, z, alpha=C.ALPHA, max_rounds=max_rounds,
                     n_splits=C.GCM_SPLITS, **kw)
    raise KeyError(f"unknown method {name!r}")


def score(selected: Sequence[int], truth: Sequence[int]) -> Dict[str, float]:
    sel, tru = set(int(x) for x in selected), set(int(x) for x in truth)
    tp, fp = float(len(sel & tru)), float(len(sel - tru))
    recall = tp / len(tru) if tru else 1.0
    precision = tp / len(sel) if sel else (1.0 if not tru else 0.0)
    return {"iou": iou_score(sel, tru), "n_selected": float(len(sel)), "tp": tp, "fp": fp,
            "recall": float(recall), "precision": float(precision)}


# ── experiment context ────────────────────────────────────────────────────────

def load_ground_truth(k: int, replica: bool) -> Dict[str, int]:
    """Attribute -> principal coordinate of the dictionary (written by `prepare`)."""
    path = C.ground_truth_path(k, replica)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python run.py prepare` first")
    gt = json.loads(path.read_text())
    return {a: int(v["coordinate"]) for a, v in gt["attributes"].items()}


@dataclass
class Context:
    features: np.ndarray
    labels: pd.DataFrame
    buckets: dict
    truth: List[int]
    dgp: dict
    modifier_cols: Optional[List[int]]
    max_rounds: Dict[str, int]          # method -> cap on the NEXIS forward rounds


_FEATURES: Dict[Path, np.ndarray] = {}


def load_context(exp: dict) -> Context:
    path = C.codes_path(exp["k"], exp["replica"], exp["view"])
    if path not in _FEATURES:
        _FEATURES.clear()
        _FEATURES[path] = np.load(path)
    labels = pd.read_parquet(C.LABELS)
    dgp = exp["dgp"]
    coords = load_ground_truth(exp["k"], exp["replica"])
    missing = [a for a in dgp["attrs"] if a not in coords]
    if missing:
        raise KeyError(f"no principal coordinate for {missing}: rerun `run.py prepare`")
    truth = sorted({coords[a] for a, g in zip(dgp["attrs"], dgp["gammas"]) if g != 0})
    mod_cols = ([coords[a] for a in dgp["attrs"][:2]]
                if dgp["effect_form"] == "ortho_quadratic" else None)
    features = _FEATURES[path]
    if dgp.get("split"):     # controlled violation of Principal Alignment: S* gains j_new
        sp = dgp["split"]
        features, j_new = split_principal(features, coords[sp["attr"]], sp["share"], sp["seed"])
        truth = sorted(truth + [j_new])
    return Context(features, labels, build_buckets(labels, dgp["attrs"]), truth, dgp, mod_cols,
                   exp.get("max_rounds", {}))


def draw(ctx: Context, n: int, eta: float, seed: int):
    d = ctx.dgp
    if d["eta_scales"] == "beta":     # eta multiplies the prognostic main effects
        betas, effect_scale = [eta * b for b in d["betas"]], 1.0
    else:
        betas, effect_scale = d["betas"], eta
    return generate_rct(n=n, features=ctx.features, labels_df=ctx.labels, buckets=ctx.buckets,
                        attrs=d["attrs"], betas=betas, gammas=d["gammas"],
                        effect_scale=effect_scale, seed=seed, tau0=d["tau0"],
                        noise_sd=d["noise_sd"], p_treat=d["p_treat"],
                        effect_form=d["effect_form"], modifier_cols=ctx.modifier_cols)


def run_cell(ctx: Context, method: str, n: int, eta: float, seed: int) -> Optional[dict]:
    """Draw the dataset of (n, eta, seed), run one method and score it (None if the
    sampler runs out of images)."""
    try:
        data = draw(ctx, n, eta, seed)
    except ValueError:
        return None
    t0 = time.perf_counter()
    res = run_method(method, data.Y, data.T, data.Z, max_rounds=ctx.max_rounds.get(method))
    out = {**score(res.selected, ctx.truth), "time_s": time.perf_counter() - t0,
           "selected": [int(j) for j in res.selected]}
    if method.startswith("NEXIS"):       # |S~| before the terminal step, and its test count
        out["n_forward"] = float(len(res.forward_path))
        out["forward"] = [int(j) for j in res.forward_path]
        out["terminal_tests"] = res.metadata["terminal_tests"]
    return out


# ── units ─────────────────────────────────────────────────────────────────────

@dataclass
class Unit:
    experiment: str
    sweep: str        # "effect" or "n"
    method: str


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")


def unit_path(runs_dir: Path, u: Unit) -> Path:
    return runs_dir / u.experiment / f"{u.sweep}_sweep" / f"{slug(u.method)}.parquet"


def units_of(experiments: Sequence[str]) -> List[Unit]:
    out = []
    for e in experiments:
        spec = C.EXPERIMENTS[e]
        for sweep in spec["sweeps"]:
            for m in spec["methods"]:
                out.append(Unit(e, sweep, m))
    return out


def run_unit(u: Unit, runs_dir: Path, n_seeds: int, effect_grid, n_grid,
             jobs: int = -1, overwrite: bool = False) -> Path:
    out = unit_path(runs_dir, u)
    if out.exists() and not overwrite:
        print(f"  skip (exists): {out}")
        return out
    spec = C.EXPERIMENTS[u.experiment]
    ctx = load_context(spec)
    param, fixed_col = SWEEP_PARAM[u.sweep], FIXED_COL[u.sweep]
    grid = effect_grid if u.sweep == "effect" else n_grid
    tasks = [(fv, pv, s) for fv in spec["sweeps"][u.sweep] for pv in grid
             for s in range(n_seeds)]
    print(f"  {u.experiment} | {u.sweep} sweep | {u.method}: {len(tasks)} runs", flush=True)

    def one(fv, pv, s):
        n, eta = (int(fv), pv) if u.sweep == "effect" else (int(pv), fv)
        return fv, pv, s, run_cell(ctx, u.method, n, eta, s)

    t0 = time.time()
    res = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(*task) for task in tasks)
    rows = []
    for fv, pv, s, r in res:
        if r is None:
            print(f"    sampler exhausted: {fixed_col}={fv} {param}={pv} seed={s}")
            continue
        rows.append({param: pv, "seed": s, "method": u.method, **r, fixed_col: fv})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"    -> {out} ({len(rows)} rows, {time.time() - t0:.0f} s)", flush=True)
    return out


def load_results(runs_dir: Path, experiment: str) -> Dict[str, pd.DataFrame]:
    """{"effect": df, "n": df} with every method of the experiment found on disk."""
    out = {}
    for sweep in ("effect", "n"):
        files = sorted((runs_dir / experiment / f"{sweep}_sweep").glob("*.parquet"))
        if files:
            out[sweep] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return out
