#!/usr/bin/env bash
#
# SAE-resampling robustness check, stage 3: rerun *every* CelebA experiment of the paper
# with the resampled-corpus SAEs, on the same experiment data and the same design grid
# (10 effect values × {n=500, 2000}, 11 n values × {η=2, 5}, 50 seeds, 12 methods).
#
# Total work is identical to the paper's run; it is just sharded much finer so it finishes
# in ~1 h instead of ~8 h.  Sharding is over
#   4 configs (k ∈ {5,20} × {sae, sae_precode})
#   × sweep × fixed value × method group × seed block
# and is exact: seeds are the RNG draws, so disjoint seed blocks reproduce the same 50
# datasets per grid point that an unsharded run would use.
#
# Usage:  bash scripts/celeba/submit_resample_experiment.sh [TAG]

set -euo pipefail

TAG="${1:-b1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

W="scripts/celeba/run_resample_shard.sh"
N_JOBS=0
IDS=()

submit() {   # submit <cpus> <mem> <args...>
    local cpus="$1"; shift
    local mem="$1";  shift
    local id
    id=$(sbatch --parsable --cpus-per-task="$cpus" --mem="$mem" "$W" "$@")
    IDS+=("$id")
    N_JOBS=$((N_JOBS + 1))
}

for K in 5 20; do
  for FEATURE in sae sae_precode; do

    # ── effect sweep: n ≤ 2000, cheap and memory-light ────────────────────────
    for FIXED in 500 2000; do
        for GROUP in g1 g2; do
            submit 40 100G "$TAG" "$FEATURE" "$K" effect "$FIXED" "$GROUP" 0 50
        done
        # GCM: lgbm is ~3× slower than the linear test → split the 50 seeds in two
        for OFF in 0 25; do
            submit 40 100G "$TAG" "$FEATURE" "$K" effect "$FIXED" g3 "$OFF" 25
        done
    done

    # ── n sweep: includes n = 10,000 → fewer threads, more memory per thread ──
    for FIXED in 2.0 5.0; do
        for GROUP in g1 g2; do
            for OFF in 0 25; do
                submit 20 150G "$TAG" "$FEATURE" "$K" n "$FIXED" "$GROUP" "$OFF" 25
            done
        done
        # seed blocks 0-12, 13-24, 25-37, 38-49  (13+12+13+12 = 50)
        for BLOCK in "0 13" "13 12" "25 13" "38 12"; do
            set -- $BLOCK
            submit 20 150G "$TAG" "$FEATURE" "$K" n "$FIXED" g3 "$1" "$2"
        done
    done

  done
done

echo "Submitted ${N_JOBS} shard jobs for resample ${TAG}."
echo "${IDS[*]}" > "logs/resample_${TAG}_shard_ids.txt"
echo "Job ids → logs/resample_${TAG}_shard_ids.txt"
echo
echo "When all have finished:"
echo "  python src/apps/celeba/merge_shards.py --tag ${TAG}"
echo "  python src/apps/celeba/sae_agreement.py --tag ${TAG}"
