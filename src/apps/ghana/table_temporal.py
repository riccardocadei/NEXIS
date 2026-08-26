#!/usr/bin/env python3
"""Emit the per-community temporal table straight from the VLM artifact.

Why this exists
---------------
Table `tab:ghana_temporal` was hand-transcribed and drifted from the analysis it
reports.  The artifact covers the six communities where neuron 3821 fires,
`[951, 675, 395, 1265, 655, 624]` (hard-coded in interpret_temporal_waterways.py
and reproducible from data/ghana/satellite/sae_activations.npy).  The printed
table instead listed communities 311 and 1613, whose activation on that neuron is
exactly 0.0 under both the post-TopK codes and the dense pre-activations, and
dropped 395 and 655, which are genuinely active.  Only the three
cropland-expansion rows were right -- those are the three the source brief
enumerates; the other three rows had no source to copy from.

Generating the table removes the transcription step entirely.

Usage
-----
    python src/apps/ghana/table_temporal.py                # markdown + latex to stdout
    python src/apps/ghana/table_temporal.py --write        # also write files next to the artifact
    python src/apps/ghana/table_temporal.py --check        # verify against the activations, exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_temporal.json"
SAT = ROOT / "data" / "ghana" / "satellite"
NEURON = 3821


def classify(text: str) -> str:
    """Map a free-text VLM change description onto the table's three-valued column.

    Negations are tested first: "no significant expansion or contraction" contains
    the word "expansion" and would otherwise be read as an increase.
    """
    t = text.lower()
    if any(k in t for k in ("no significant", "stable", "similarly", "no noticeable")):
        return "no change"
    if "expansion" in t:
        return "increase"
    if "denser" in t or "increase" in t:
        return "denser"
    return "unclear"


LATEX = {"no change": "no change",
         "increase": r"$\uparrow$ expansion",
         "denser": r"$\uparrow$ denser adjacent to waterway",
         "unclear": "unclear"}
MD = {"no change": "no change",
      "increase": "↑ expansion",
      "denser": "↑ denser adjacent to waterway",
      "unclear": "unclear"}


def load_rows():
    rows = json.loads(ARTIFACT.read_text())
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
    md = ["| Community | Cropland change | Vegetation change |",
          "|---|---|---|"]
    tex = [r"\toprule",
           r"Community & Cropland change & Vegetation change \\", r"\midrule"]
    n_crop = 0
    for e in rows:
        a = classify(e["agricultural_change"])
        v = classify(e["vegetation_change"])
        n_crop += a != "no change"
        md.append(f"| {e['comm_id']} | {MD[a]} | {MD[v]} |")
        tex.append(f"    {e['comm_id']:<5}& {LATEX[a]:<30}& {LATEX[v]:<40}\\\\")
    tex.append(r"\bottomrule")
    n_wat = sum("similarly visible" in e["waterway_change"] for e in rows)
    note = (f"cropland expansion in {n_crop} of {len(rows)} communities; "
            f"waterway structure unchanged in {n_wat} of {len(rows)}")
    return "\n".join(md), "\n".join(tex), note


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true",
                   help="Write table_temporal.{md,tex} beside the artifact.")
    p.add_argument("--check", action="store_true",
                   help="Only verify the artifact against the activations.")
    args = p.parse_args()

    if not ARTIFACT.exists():
        sys.exit(f"missing artifact: {ARTIFACT}\n"
                 f"run: python src/apps/ghana/interpret_temporal_waterways.py")
    rows = load_rows()
    if args.check:
        sys.exit(check(rows))

    md, tex, note = render(rows)
    print(md, "\n"); print(tex, "\n"); print(note)
    if args.write:
        out = ARTIFACT.parent
        (out / "table_temporal.md").write_text(md + "\n\n" + note + "\n")
        (out / "table_temporal.tex").write_text(tex + "\n")
        print(f"\nwrote {out/'table_temporal.md'} and {out/'table_temporal.tex'}")


if __name__ == "__main__":
    main()
