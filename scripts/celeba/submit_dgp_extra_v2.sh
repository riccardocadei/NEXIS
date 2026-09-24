#!/usr/bin/env bash
#SBATCH --job-name=xdgp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#
# Appendix C extras for the NEXIS-v2 default (CPU only, existing SAE features only):
#
#   rep  reproducibility: the main experiment on the independently trained k=20 SAE
#        (data/celeba_resample_b1/eval, docs/celeba_sae_resampling.md), ground truth
#        from results/celeba/experiment_resample_b1/k20/sae/ground_truth.json
#   r3   three direct modifiers: Wearing_Hat (+1), Eyeglasses (-1), Sideburns (+1),
#        the convention of results/celeba/experiment_r3 (run_experiment_rsweep.sh)
#   r1   one direct modifier: Wearing_Hat (+1); Eyeglasses sampled as in the main
#        setting but prognostic only (gamma = 0)
#   r0   no modifier: tau = tau_0 = 0.5 constant, S* = {}; eta scales the prognostic
#        main effects 0.3 W_hat - 0.2 W_glasses instead (see run_experiment_dgp.py);
#        eta grid includes 0.  Also runs the published NEXIS for contrast.
#
# Same grid, seeds and baselines as the main setting: eta in 1..10 at n in {500, 2000},
# n in {50..10000} at eta in {2, 5}, 50 seeds, alpha 0.05, 10 steps.
# Results: results/celeba/experiment_v2_{resample_b1,r3,r1,r0}/k20/sae/.
#
#   bash   scripts/celeba/submit_dgp_extra_v2.sh [--overwrite]   # submit all 8 jobs
#   sbatch scripts/celeba/submit_dgp_extra_v2.sh r0 effect        # one job
#
# Memory: the n sweep peaks at ~170G with 40 threads (n = 10,000 x 9,216 per task);
# the effect sweep (n <= 2,000) needs ~40G.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

out_of() {
  case "$1" in
    rep) echo results/celeba/experiment_v2_resample_b1 ;;
    *)   echo "results/celeba/experiment_v2_$1" ;;
  esac
}
pq() { [[ "$2" == effect ]] && echo "$(out_of "$1")/k20/sae/effect_sweep.parquet" \
                            || echo "$(out_of "$1")/k20/sae/n_sweep.parquet"; }

# ── submitter ────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  OVERWRITE=""
  [[ "${1:-}" == "--overwrite" ]] && OVERWRITE="--overwrite"
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for set in rep r3 r1 r0; do
    for sweep in effect n; do
      if [[ -z "$OVERWRITE" && -f "$(pq "$set" "$sweep")" ]]; then
        echo "skip (exists): $(pq "$set" "$sweep")"; continue
      fi
      mem=200G; [[ "$sweep" == effect ]] && mem=100G
      id=$(sbatch --parsable --job-name="xdgp-$set-$sweep" --mem="$mem" \
                  "$SELF" "$set" "$sweep" $OVERWRITE)
      echo "xdgp-$set-$sweep: $id"
    done
  done
  exit 0
fi

# ── worker ───────────────────────────────────────────────────────────────────
SET="$1"; SWEEP="$2"; OVERWRITE="${3:-}"
FORCE=(); [[ -n "$OVERWRITE" ]] && FORCE=(--force)
BASELINES=("Marginal Testing" "Marginal Testing (FWER)" "Marginal Testing (FDR)")
GRID=(--n-seeds 50 --alpha 0.05 --max-steps 10 --fixed-n 500 2000
      --fixed-effect 2.0 5.0 --gcm-splits 3 --sweep "$SWEEP")
OUT="$(out_of "$SET")"
echo "set=$SET sweep=$SWEEP host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} out=$OUT"

case "$SET" in
  rep)
    exec "$PYTHON" src/apps/celeba/run_experiment.py --sae-top-k 20 \
        --data-dir data/celeba_resample_b1/eval --out-dir "$OUT" \
        --gt-json results/celeba/experiment_resample_b1/k20/sae/ground_truth.json \
        --w1-attr Wearing_Hat --w2-attr Eyeglasses --top-k 1 "${GRID[@]}" \
        --methods "${BASELINES[@]}" NEXIS-v2 "${FORCE[@]}" ;;
  r3)
    exec "$PYTHON" src/apps/celeba/run_experiment.py --sae-top-k 20 \
        --data-dir data/celeba --out-dir "$OUT" \
        --gt-json results/celeba/experiment_r3/k20/sae/ground_truth.json \
        --w-attrs Wearing_Hat Eyeglasses Sideburns --top-k 1 "${GRID[@]}" \
        --methods "${BASELINES[@]}" NEXIS-v2 "${FORCE[@]}" ;;
  r1)
    exec "$PYTHON" src/apps/celeba/run_experiment_dgp.py --out-dir "$OUT" \
        --gammas 1 0 "${GRID[@]}" --methods "${BASELINES[@]}" NEXIS-v2 "${FORCE[@]}" ;;
  r0)
    exec "$PYTHON" src/apps/celeba/run_experiment_dgp.py --out-dir "$OUT" \
        --gammas 0 0 --scale beta --effect-grid 0 1 2 3 4 5 6 7 8 9 10 "${GRID[@]}" \
        --methods "${BASELINES[@]}" NEXIS-v2 NEXIS "${FORCE[@]}" ;;
  *) echo "unknown set: $SET" >&2; exit 1 ;;
esac
