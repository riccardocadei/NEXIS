"""Final set of real-world NEXIS runs behind the paper's application results.

Runs both NEXIS variants on Uganda YOP and Ghana LEAP 1000:

  published   nexis(backward=True,  terminal_filter=False)
  new default nexis(backward=False, terminal_filter=True)

with rho = 0.5, alpha = 0.05, the FWER forward gate and the linear T x Z_j test, each
with the application's published test and with the level-aware clustered test of
src/nexis/multilevel.py (support gate k = 5 clusters per (side of Z_j) x (arm)
cell).  The grids are in src/apps/uganda/final_runs.py (T = grant received and T =
lottery assignment) and src/apps/ghana/final_runs.py (the 155 and 167 pools).  Post
hoc, a design-based table for every published modifier and every new selection, each
conditional on the rest of the set it was selected with (Uganda: CR1S by group with
district FE and randomization inference replaying the group lottery within district;
Ghana: CR1S by community and a restricted wild cluster bootstrap-t by community).  Each
row also carries the number of clusters (at the candidate's L_j) in each (active /
inactive) x (treated / control) cell.

CPU only.  Reads data/ and results/ (local, not tracked); writes
<out-dir>/report.json (report_<app>.json with --only), by default in
results/realworld_final/.

    python src/apps/realworld_final_runs.py [--n-perm 1999] [--n-boot 9999]
        [--only {uganda,ghana}] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from apps.ghana.final_runs import ghana_block  # noqa: E402
from apps.uganda.final_runs import OUTCOMES as UG_OUTCOMES, uganda_block  # noqa: E402
from nexis.multilevel import ALPHA, MIN_SUPPORT, RHO  # noqa: E402

OUT = ROOT / "results" / "realworld_final"


def summary(block):
    print(f"\n  {'configuration':52s} {'m':>4s}  selected   [published kept / dropped]")
    for k, r in block["runs"].items():
        print(f"  {k:52s} {r['m']:4d}  {r['selected']}   [kept {r['kept']} / dropped {r['dropped']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1999)
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--only", choices=["uganda", "ghana"], default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(alpha=ALPHA, rho=RHO, min_support=MIN_SUPPORT, n_perm=args.n_perm, n_boot=args.n_boot)
    if args.only in (None, "uganda"):
        for o in UG_OUTCOMES:
            report[f"uganda/{o}"] = uganda_block(o, args)
    if args.only in (None, "ghana"):
        report["ghana/consumption"] = ghana_block(args)
    print(f"\n{'=' * 100}\nSUMMARY")
    for k, b in report.items():
        if isinstance(b, dict):
            print(f"\n{k}   published: {b['published']}")
            summary(b)
    name = "report.json" if args.only is None else f"report_{args.only}.json"
    (args.out_dir / name).write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {args.out_dir / name}")


if __name__ == "__main__":
    main()
