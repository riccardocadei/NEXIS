#!/usr/bin/env python3
"""Reproduce the CelebA experiments of the paper (Figure 3 and Appendix C).

    python run.py prepare                 # data, SigLIP 2 embeddings, SAEs, S*   (GPU)
    python run.py main                    # Figure 3 and the reference figure (dgp.pdf)
    python run.py ablations-model         # SAE ablations: k = 5, pre-activations
    python run.py ablations-method        # NEXIS ablations: test, adjust, rho, backward steps
    python run.py replica                 # main experiment on the replica SAE
    python run.py dgp-extra               # r = 0, 1, 3 direct modifiers
    python run.py ushape                  # U-shape DGP: test comparison (Table, panel B)
    python run.py alignment               # supervised Principal Alignment check
    python run.py violation               # controlled violation of Principal Alignment
    python run.py all                     # prepare + every block above
    python run.py figures                 # redraw every figure from the results on disk

Options: --seeds N (default 50; 200 for r = 0), --jobs N (parallel runs, default all
cores), --smoke (2 seeds, 2 grid points per sweep, minutes on a CPU; skips prepare),
--overwrite (recompute existing outputs), --list-units / --unit I (one unit of work, for
job arrays), --data-dir / --results-dir (override config.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BLOCK_NAMES = ["main", "ablations-model", "ablations-method", "replica", "dgp-extra", "ushape",
               "alignment", "violation"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("block", choices=["prepare", *BLOCK_NAMES, "all", "figures"])
    p.add_argument("--seeds", type=int, default=None,
                   help="Monte Carlo seeds per cell (default: config.N_SEEDS, R0_SEEDS at r=0)")
    p.add_argument("--jobs", type=int, default=-1, help="parallel runs (threads); -1 = all cores")
    p.add_argument("--smoke", action="store_true",
                   help="quick end-to-end check: few seeds and grid points, separate outputs")
    p.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    p.add_argument("--no-figures", action="store_true", help="run experiments only")
    p.add_argument("--list-units", action="store_true", help="print the block's units and exit")
    p.add_argument("--unit", type=int, default=None, help="run only this unit of the block")
    p.add_argument("--steps", nargs="+", default=None,
                   help="prepare only: subset of embed sae encode ground-truth replica")
    p.add_argument("--embed-limit", type=int, default=None,
                   help="prepare only, for dry runs: embed only the first N images")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    if a.data_dir is not None:
        os.environ["NEXIS_DATA_DIR"] = str(a.data_dir.resolve())
    if a.results_dir is not None:
        os.environ["NEXIS_RESULTS_DIR"] = str(a.results_dir.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C

    blocks = {"all": BLOCK_NAMES, "figures": BLOCK_NAMES, "prepare": []}.get(a.block, [a.block])

    # output roots: full runs, smoke runs and non-default seed counts never mix
    if a.smoke:
        root = C.RESULTS_DIR / "smoke"
        grids = dict(effect_grid=C.SMOKE["effect_grid"], n_grid=C.SMOKE["n_grid"])
    else:
        root = C.RESULTS_DIR if a.seeds is None else C.RESULTS_DIR / f"seeds{a.seeds}"
        grids = dict(effect_grid=C.EFFECT_GRID, n_grid=C.N_GRID)
    runs_dir, fig_dir = root / "runs", root / "figures"

    from celeba.experiment import run_unit, units_of
    experiments = list(dict.fromkeys(e for b in blocks for e in C.BLOCKS[b]["experiments"]))
    units = units_of(experiments)
    if a.list_units:
        for i, u in enumerate(units):
            print(f"{i:3d}  {u.experiment:14s} {u.sweep:6s} {u.method}")
        return

    if a.block == "prepare" or (a.block == "all" and not a.smoke and a.unit is None):
        from celeba import prepare
        prepare.run(a.steps or prepare.STEPS, overwrite=a.overwrite, embed_limit=a.embed_limit)
        if a.block == "prepare":
            return

    if a.block != "figures":
        todo = units if a.unit is None else [units[a.unit]]
        t0 = time.time()
        for u in todo:
            n_seeds = (C.SMOKE["n_seeds"] if a.smoke
                       else a.seeds or C.EXPERIMENTS[u.experiment]["n_seeds"])
            run_unit(u, runs_dir, n_seeds, jobs=a.jobs, overwrite=a.overwrite, **grids)
        print(f"experiments done in {(time.time() - t0) / 60:.1f} min -> {runs_dir}")
        if a.unit is not None or a.no_figures:
            return

    from celeba.figures import render
    figures = list(dict.fromkeys(f for b in blocks for f in C.BLOCKS[b]["figures"]))
    render(figures, runs_dir, fig_dir)
    if "alignment" in blocks:
        from celeba import alignment
        alignment.run(fig_dir, overwrite=a.overwrite,
                      n_splits=2 if a.smoke else alignment.N_SPLITS)


if __name__ == "__main__":
    main()
