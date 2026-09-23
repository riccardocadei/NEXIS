#!/usr/bin/env bash
#SBATCH --job-name=celeba-v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#
# CelebA sweeps for the terminal-backward-step NEXIS default ("NEXIS-v2", see
# V2_METHODS in src/apps/celeba/experiment.py), for Figure 3 and the Appendix C
# ablations.  Same grid, seeds, SCM and ground truth as the published runs
# (ground_truth.json reused from results/celeba/experiment/); CPU only, existing
# SAE features only.  Results go to results/celeba/experiment_v2/<tree>/.
#
#   bash   scripts/celeba/submit_experiment_v2.sh [--overwrite]   # submit everything
#   sbatch scripts/celeba/submit_experiment_v2.sh k20/sae n fast   # one job
#
# Per (tree, sweep): a "fast" job (baselines + linear/GCM-quadratic v2 variants) and,
# on k20/sae only, a "slow" job (GCM: lgbm, PCM: quadratic, PCM: lgbm) that waits for
# the fast one and merges into its parquet.  Jobs whose output exists are skipped.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
OUT=results/celeba/experiment_v2

BASELINES=("Marginal Testing" "Marginal Testing (FWER)" "Marginal Testing (FDR)")
# "NEXIS-v2 (interleaved=True, terminal=False)" is the published NEXIS configuration,
# rerun under a v2 name as a reproduction check and for the backward-step ablation.
FAST_K20=("${BASELINES[@]}"
    "NEXIS-v2"
    "NEXIS-v2 (test=GCM: quadratic)"
    "NEXIS-v2 (adjust=None)" "NEXIS-v2 (adjust=FDR)"
    "NEXIS-v2 (rho=0)" "NEXIS-v2 (rho=0.2)" "NEXIS-v2 (rho=0.8)"
    "NEXIS-v2 (terminal=False)"
    "NEXIS-v2 (interleaved=True, terminal=False)"
    "NEXIS-v2 (interleaved=True)")
FAST_OTHER=("${BASELINES[@]}" "NEXIS-v2" "NEXIS-v2 (interleaved=True, terminal=False)")
SLOW_K20=("NEXIS-v2 (test=GCM: lgbm)" "NEXIS-v2 (test=PCM: quadratic)"
          "NEXIS-v2 (test=PCM: lgbm)")

pq() { [[ "$2" == effect ]] && echo "$OUT/$1/effect_sweep.parquet" || echo "$OUT/$1/n_sweep.parquet"; }

# has_methods <parquet> <method>... : true when the parquet holds rows for every method
has_methods() {
  local f="$1"; shift
  [[ -f "$f" ]] || return 1
  "$PYTHON" - "$f" "$@" <<'EOF'
import sys, pandas as pd
have = set(pd.read_parquet(sys.argv[1], columns=["method"])["method"])
sys.exit(0 if set(sys.argv[2:]) <= have else 1)
EOF
}

# ── submitter ────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  OVERWRITE=""
  [[ "${1:-}" == "--overwrite" ]] && OVERWRITE="--overwrite"
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for tree in k20/sae k20/sae_precode k5/sae; do
    for sweep in effect n; do
      name="v2-${tree//\//_}-$sweep"
      fast_id=""
      if [[ -z "$OVERWRITE" && -f "$(pq "$tree" "$sweep")" ]]; then
        echo "skip (exists): $(pq "$tree" "$sweep")"
      else
        fast_id=$(sbatch --parsable --job-name="$name" "$SELF" "$tree" "$sweep" fast $OVERWRITE)
        echo "$name fast: $fast_id"
      fi
      if [[ "$tree" == k20/sae ]]; then
        if [[ -z "$OVERWRITE" ]] && has_methods "$(pq "$tree" "$sweep")" "${SLOW_K20[@]}"; then
          echo "skip (exists): slow methods in $(pq "$tree" "$sweep")"
          continue
        fi
        dep=${fast_id:+--dependency=afterok:$fast_id}
        slow_id=$(sbatch --parsable --job-name="$name-slow" $dep "$SELF" "$tree" "$sweep" slow $OVERWRITE)
        echo "$name slow: $slow_id ${dep}"
      fi
    done
  done
  exit 0
fi

# ── worker ───────────────────────────────────────────────────────────────────
TREE="$1"; SWEEP="$2"; KIND="$3"; OVERWRITE="${4:-}"
kdir="${TREE%%/*}"; ftype="${TREE##*/}"
FEAT=(--sae-top-k "${kdir#k}")
[[ "$ftype" == sae_precode ]] && FEAT+=(--precode)

COMMON=(
    --data-dir     data/celeba
    --out-dir      "$OUT"
    --gt-json      "results/celeba/experiment/$TREE/ground_truth.json"
    --w1-attr      Wearing_Hat
    --w2-attr      Eyeglasses
    --top-k        1
    --n-seeds      50
    --alpha        0.05
    --max-steps    10
    --fixed-n      500 2000
    --fixed-effect 2.0 5.0
    --gcm-splits   3
    --sweep        "$SWEEP"
)

if [[ "$KIND" == fast ]]; then
  if [[ "$TREE" == k20/sae ]]; then METHODS=("${FAST_K20[@]}"); else METHODS=("${FAST_OTHER[@]}"); fi
  EXTRA=(); [[ -n "$OVERWRITE" ]] && EXTRA=(--force)
else
  if [[ -z "$OVERWRITE" ]] && has_methods "$(pq "$TREE" "$SWEEP")" "${SLOW_K20[@]}"; then
    echo "slow methods already in $(pq "$TREE" "$SWEEP"), skipping"; exit 0
  fi
  export OMP_NUM_THREADS=1   # as the published PCM runs (run_experiment_pcm.sh)
  METHODS=("${SLOW_K20[@]}")
  EXTRA=(--merge)
fi

echo "tree=$TREE sweep=$SWEEP kind=$KIND host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
exec "$PYTHON" src/apps/celeba/run_experiment.py "${FEAT[@]}" "${COMMON[@]}" \
    --methods "${METHODS[@]}" "${EXTRA[@]}"
