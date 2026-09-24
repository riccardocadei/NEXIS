#!/usr/bin/env bash
#SBATCH --job-name=v2-ushape
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#
# GCM-blind U-shape DGP (rebuttal panel B, --effect-form ortho_quadratic) rerun under the
# NEXIS-v2 default (rho=0.5, no interleaved backward step, terminal filter) and the
# calibrated PCM combination rule (2·min over the two split directions, the nexis()
# default since dd18f32).  The rebuttal run (results/celeba/experiment_ushape/,
# scripts/celeba/run_experiment_ushape.sh) used the published NEXIS and the invalid
# "crossfit" PCM rule.  Same DGP, ground truth, grids and 50 seeds: eta in 1..10 at
# n = 2000, n in 50..10000 at eta = 5, continuous k=20 SAE pre-activations.
# CPU only, existing SAE features only.
#
#   bash   scripts/celeba/submit_experiment_v2_ushape.sh [--overwrite]  # submit all
#   sbatch scripts/celeba/submit_experiment_v2_ushape.sh n pcm_lgbm 0   # one shard
#
# Shards = sweep x method group x seed block of 10, each in its own out-dir under
# results/celeba/experiment_v2_ushape/shards/ so they never race; a merge job (afterok)
# writes results/celeba/experiment_v2_ushape/k20/sae_precode/{effect,n}_sweep.parquet.
# Shards and merged files that exist are skipped unless --overwrite.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
OUT=results/celeba/experiment_v2_ushape
GT=results/celeba/experiment_ushape/k20/sae_precode/ground_truth.json
SEEDS=50; BLOCK=10
GROUPS_ALL=(fast gcm_lgbm pcm_q pcm_lgbm)

methods_of() {
  case "$1" in
    fast)     echo "Marginal Testing (FWER)|NEXIS-v2|NEXIS-v2 (test=GCM: quadratic)" ;;
    gcm_lgbm) echo "NEXIS-v2 (test=GCM: lgbm)" ;;
    pcm_q)    echo "NEXIS-v2 (test=PCM: quadratic)" ;;
    pcm_lgbm) echo "NEXIS-v2 (test=PCM: lgbm)" ;;
    *) echo "unknown group: $1" >&2; exit 1 ;;
  esac
}
shard_pq() { echo "$OUT/shards/$1_$2_s$3/k20/sae_precode/$1_sweep.parquet"; }
final_pq() { echo "$OUT/k20/sae_precode/$1_sweep.parquet"; }

# ── submitter ────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  OVERWRITE=""
  [[ "${1:-}" == "--overwrite" ]] && OVERWRITE="--overwrite"
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for sweep in effect n; do
    if [[ -z "$OVERWRITE" && -f "$(final_pq "$sweep")" ]]; then
      echo "skip (exists): $(final_pq "$sweep")"; continue
    fi
    mem=200G; [[ "$sweep" == effect ]] && mem=100G
    ids=()
    for group in "${GROUPS_ALL[@]}"; do
      for ((off = 0; off < SEEDS; off += BLOCK)); do
        if [[ -z "$OVERWRITE" && -f "$(shard_pq "$sweep" "$group" "$off")" ]]; then
          echo "skip (exists): $(shard_pq "$sweep" "$group" "$off")"; continue
        fi
        id=$(sbatch --parsable --job-name="ush-$sweep-$group-s$off" --mem="$mem" \
                    "$SELF" "$sweep" "$group" "$off" $OVERWRITE)
        ids+=("$id"); echo "ush-$sweep-$group-s$off: $id"
      done
    done
    dep=""; [[ ${#ids[@]} -gt 0 ]] && dep="--dependency=afterok:$(IFS=:; echo "${ids[*]}")"
    mid=$(sbatch --parsable --job-name="ush-$sweep-merge" --cpus-per-task=2 --mem=16G \
                 --time=00:20:00 $dep "$SELF" "$sweep" merge 0 $OVERWRITE)
    echo "ush-$sweep-merge: $mid ${dep}"
  done
  exit 0
fi

# ── worker ───────────────────────────────────────────────────────────────────
SWEEP="$1"; GROUP="$2"; OFF="$3"
echo "sweep=$SWEEP group=$GROUP seed_offset=$OFF host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

if [[ "$GROUP" == merge ]]; then
  exec "$PYTHON" - "$OUT" "$SWEEP" "$SEEDS" "$GT" <<'PY'
import sys, glob, shutil
from pathlib import Path
import pandas as pd
out, sweep, seeds, gt = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
files = sorted(glob.glob(str(out / "shards" / f"{sweep}_*" / "k20/sae_precode" / f"{sweep}_sweep.parquet")))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
x = "n" if sweep == "n" else "effect_scale"
dup = df.duplicated(["method", x, "seed"]).sum()
cnt = df.groupby(["method", x]).seed.nunique()
print(f"{len(files)} shards, {len(df)} rows, duplicates={dup}")
print(cnt.unstack(0).to_string())
if dup or (cnt != seeds).any():
    sys.exit(f"incomplete or overlapping merge (expected {seeds} seeds per cell)")
dest = out / "k20/sae_precode"
dest.mkdir(parents=True, exist_ok=True)
df.to_parquet(dest / f"{sweep}_sweep.parquet", index=False)
shutil.copyfile(gt, dest / "ground_truth.json")
print("->", dest / f"{sweep}_sweep.parquet")
PY
fi

IFS='|' read -r -a METHODS <<< "$(methods_of "$GROUP")"
export OMP_NUM_THREADS=1   # run_sweep parallelises over (grid point x seed)
exec "$PYTHON" src/apps/celeba/run_experiment.py \
    --precode --sae-top-k 20 \
    --data-dir     data/celeba \
    --out-dir      "$OUT/shards/${SWEEP}_${GROUP}_s${OFF}" \
    --backbone     siglip \
    --gt-json      "$GT" \
    --effect-form  ortho_quadratic \
    --w1-attr      Wearing_Hat \
    --w2-attr      Eyeglasses \
    --top-k        1 \
    --n-seeds      "$BLOCK" \
    --seed-offset  "$OFF" \
    --alpha        0.05 \
    --max-steps    10 \
    --fixed-n      2000 \
    --fixed-effect 5.0 \
    --gcm-splits   3 \
    --sweep        "$SWEEP" \
    --methods      "${METHODS[@]}" \
    --force
