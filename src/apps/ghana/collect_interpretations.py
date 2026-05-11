"""
Collect VLM interpretations from all pipeline outputs and write one
`interpretations.json` per neuron into results/ghana/neurons/neuron_{idx}/.

Scans:  results/ghana/{rep_mode}/{method}/{pipeline}/interpretations.json
Writes: results/ghana/neurons/neuron_{idx}/interpretations.json

Each output file is a list of interpretation records, one per
(pipeline, rep_mode, method) combination that produced a result for
that neuron, sorted by p-value.

Usage (from repo root):
  python scripts/ghana/collect_interpretations.py
  python scripts/ghana/collect_interpretations.py --print   # also pretty-print to stdout
"""

import argparse
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "results" / "ghana"
NEU_DIR = RES_DIR / "neurons"

KEEP_FIELDS = (
    "pipeline", "rep_mode", "method",
    "neuron_idx", "feature", "pvalue",
    "label", "activated_concept", "not_activated_concept", "confidence",
    "model_tag", "vlm_model",                     # provenance (whichever is present)
    "top_ids", "bot_ids", "top_acts", "bot_acts",
    "top_descriptions", "bot_descriptions",        # geochat_llm only
)


def collect() -> dict[int, list]:
    """Return {neuron_idx: [records ...]} from all interpretations.json files."""
    by_neuron: dict[int, list] = {}

    for interp_path in sorted(RES_DIR.rglob("*/interpretations.json")):
        # path shape: results/ghana/{rep_mode}/{method}/{pipeline}/interpretations.json
        parts = interp_path.relative_to(RES_DIR).parts
        if len(parts) != 4:
            continue
        rep_mode, method, pipeline, _ = parts

        with open(interp_path) as f:
            entries = json.load(f)

        for entry in entries:
            nidx = entry.get("neuron_idx")
            if nidx is None:
                continue
            nidx = int(nidx)

            record = {"rep_mode": rep_mode, "method": method}
            for k in KEEP_FIELDS:
                if k in entry:
                    record[k] = entry[k]

            by_neuron.setdefault(nidx, []).append(record)

    # Sort each neuron's records by p-value ascending
    for nidx in by_neuron:
        by_neuron[nidx].sort(key=lambda r: r.get("pvalue", 1.0))

    return by_neuron


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true", dest="do_print",
                   help="Pretty-print a summary to stdout after writing.")
    args = p.parse_args()

    by_neuron = collect()
    if not by_neuron:
        print("No interpretations found — run a pipeline first.")
        return

    for nidx, records in sorted(by_neuron.items()):
        out_dir = NEU_DIR / f"neuron_{nidx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interpretations.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  neuron {nidx:>5}  {len(records)} interpretation(s) → {out_path}")

    if args.do_print:
        print()
        _print_summary(by_neuron)


def _print_summary(by_neuron: dict):
    w_label = 38
    w_conf  = 8
    header  = (f"{'Neuron':>6}  {'p-value':>8}  {'Pipeline':<20}  "
               f"{'rep':>9}  {'Label':<{w_label}}  {'Conf':<{w_conf}}")
    print(header)
    print("-" * len(header))
    for nidx in sorted(by_neuron):
        for r in by_neuron[nidx]:
            print(
                f"{nidx:>6}  {r.get('pvalue', float('nan')):>8.4f}  "
                f"{r.get('pipeline','?'):<20}  "
                f"{r.get('rep_mode','?'):>9}  "
                f"{str(r.get('label','?'))[:w_label]:<{w_label}}  "
                f"{r.get('confidence','?'):<{w_conf}}"
            )
        print()


if __name__ == "__main__":
    main()
