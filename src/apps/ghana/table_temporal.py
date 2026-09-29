#!/usr/bin/env python3
"""Emit the per-community temporal table straight from the VLM artifact.

Source
------
`results/ghana/temporal/neuron_3821_temporal_neutral.json`, written by
`interpret_temporal_changes.py` (Qwen2.5-VL-72B-Instruct, 4-bit, greedy
decoding). The VLM sees the 2015 and 2017 false-colour composites of one
community side by side, with a colour key, and is asked to list every
land-cover change as "+/- <category>" plus a 1-2 sentence description. It is
not told which feature the community was selected for (neutral prompt).

The artifact covers the six communities where neuron 3821 (ephemeral
waterways) fires, `[951, 675, 395, 1265, 655, 624]` (activation > 0 in
data/ghana/satellite/sae_activations.npy; `--check` verifies this).

An earlier run, `results/ghana/temporal/neuron_3821_temporal.json`
(`interpret_temporal_waterways.py`), told the model that the tile contains
ephemeral waterways and asked about changes near them. That prompt is leading,
so the table no longer uses it.

Paper item: Table tab:ghana_temporal (paper appendix, "Per-community VLM
temporal analysis for the six waterway-active LEAP communities").

Columns: community, cropland change (from the "+/- cropland" line; "not listed"
when the VLM gives no cropland line), and local NDVI
increase (% area): the area of the patches where NDVI rose between the 2015 and
2017 composites by more than 0.05 beyond the tile's median change, as % of the
tile's non-water area (rule in `ndvi_change.py`; the same patches are boxed in
Figure 5). The vegetation calls are printed as a note, not as a column.

Usage
-----
    python src/apps/ghana/table_temporal.py                # markdown + latex to stdout
    python src/apps/ghana/table_temporal.py --check        # verify against the activations, exit 1 on drift
    python src/apps/ghana/table_temporal.py --paper        # write results/ghana/paper_numbers/table_temporal.{md,tex}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.apps.ghana.ndvi_change import increase_area_pct  # noqa: E402

ARTIFACT = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_temporal_neutral.json"
SAT = ROOT / "data" / "ghana" / "satellite"
NEURON = 3821


def change(entry: dict, category: str) -> str:
    """'+', '-' or '' for one category of the VLM's change list."""
    syms = {c["symbol"] for c in entry["changes"] if c["label"] == category}
    if len(syms) > 1:
        return "+/-"
    return syms.pop() if syms else ""


LATEX = {"+": r"$\uparrow$ increase", "-": r"$\downarrow$ decrease",
         "+/-": "mixed", "": "not listed"}
MD = {"+": "↑ increase", "-": "↓ decrease", "+/-": "mixed", "": "not listed"}


def load_rows():
    data = json.loads(ARTIFACT.read_text())
    rows = [dict(v, comm_id=int(k)) for k, v in data.items()]
    rows.sort(key=lambda e: -e["activation"])
    return rows


def check(rows) -> int:
    """The artifact must cover exactly the communities where the neuron is active."""
    acts = np.load(SAT / "sae_activations.npy")[:, NEURON]
    ids = np.load(SAT / "prithvi_comm_ids.npy")
    active = {int(ids[i]) for i in np.flatnonzero(acts > 0)}
    listed = {int(e["comm_id"]) for e in rows}
    ok = active == listed
    print(f"active communities for neuron {NEURON}: {sorted(active)}")
    print(f"communities in the artifact         : {sorted(listed)}")
    if not ok:
        print(f"  MISSING from artifact: {sorted(active - listed)}")
        print(f"  NOT ACTUALLY ACTIVE  : {sorted(listed - active)}")
    print("OK" if ok else "DRIFT")
    return 0 if ok else 1


def render(rows):
    md = ["| Community | Cropland change | Local NDVI increase (% area) |", "|---|---|---|"]
    tex = [r"\toprule",
           r"Community & Cropland change & Local NDVI increase (\% area) \\",
           r"\midrule"]
    for e in rows:
        c = change(e, "cropland")
        cell_md, cell_tex = MD[c], LATEX[c]
        pct = increase_area_pct(e["comm_id"])
        pct_md = "n/a" if pct is None else f"{pct:.1f}"
        pct_tex = "n/a" if pct is None else f"${pct:.1f}$"
        md.append(f"| {e['comm_id']} | {cell_md} | {pct_md} |")
        tex.append(f"{e['comm_id']:<5}& {cell_tex} & {pct_tex} \\\\")
    tex.append(r"\bottomrule")
    n_crop = sum(change(e, "cropland") == "+" for e in rows)
    veg = ", ".join(f"{e['comm_id']}: {MD[change(e, 'vegetation')]}" for e in rows)
    note = (f"cropland increase in {n_crop} of {len(rows)} communities; "
            f"vegetation (not tabulated): {veg}")
    return "\n".join(md), "\n".join(tex), note


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Only verify the artifact against the activations.")
    p.add_argument("--paper", action="store_true",
                   help="Write table_temporal.{md,tex} to results/ghana/paper_numbers/.")
    args = p.parse_args()

    if not ARTIFACT.exists():
        sys.exit(f"missing artifact: {ARTIFACT}\n"
                 f"run: sbatch scripts/ghana/slurm_temporal_changes.sh")
    rows = load_rows()
    if args.check:
        sys.exit(check(rows))

    md, tex, note = render(rows)
    print(md, "\n"); print(tex, "\n"); print(note)
    for e in rows:
        print(f"  {e['comm_id']}: {e['description']}")
    if args.paper:
        out = ROOT / "results" / "ghana" / "paper_numbers"
        out.mkdir(parents=True, exist_ok=True)
        (out / "table_temporal.md").write_text(md + "\n\n" + note + "\n")
        (out / "table_temporal.tex").write_text(tex + "\n")
        print(f"\nwrote {out/'table_temporal.md'} and {out/'table_temporal.tex'}")


if __name__ == "__main__":
    main()
