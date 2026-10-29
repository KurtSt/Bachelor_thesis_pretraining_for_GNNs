#!/usr/bin/env bash
# Experiments after the matrix (meeting of 14.10.2026). Prints the sbatch commands with
# DRY_RUN=1.
#
#   scripts/slurm/submit_extra.sh predict              test predictions of all finished
#                                                      fine-tunings (CPU) + analysis job
#   scripts/slurm/submit_extra.sh datasize <seed ...>  augmented/all fine-tuned from the
#                                                      checkpoints after 1 M and 10 M molecules
#   scripts/slurm/submit_extra.sh chebi25 <seed ...>   ChEBI25: build the dataset (CPU), then
#                                                      augmented/none and augmented/all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm
MODE="$1"; shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
JOB="scripts/slurm/job.sbatch"

run() {  # prints the job id (or a placeholder with DRY_RUN=1)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "sbatch $*" >&2; echo "DRY"
  else
    sbatch --parsable "$@"
  fi
}

case "$MODE" in
  predict)
    LIST="logs/slurm/predict_runs.txt"
    : > "$LIST"
    for d in logs/hpc_* logs/hpc25_*; do
      [ -d "$d" ] || continue
      name="$(basename "$d")"
      case "$name" in *pretrain*|*prepare*) continue ;; esac
      [ -f "$d/.done" ] || continue
      ls "$d"/version_*_predict/test_predictions.pt > /dev/null 2>&1 && continue
      echo "$name" >> "$LIST"
    done
    N="$(wc -l < "$LIST")"
    echo "runs to predict: $N (list: $LIST)"
    if [ "$N" -gt 0 ]; then
      ARRAY="$(run --array=1-"$N"%10 scripts/slurm/predict.sbatch "$LIST")"
      echo "predict: $ARRAY"
      echo "analyse: $(run --dependency=afterany:"$ARRAY" scripts/slurm/analyse.sbatch)"
    fi
    ;;
  datasize)
    for size in 1000000:1m 10000000:10m; do
      molecules="${size%%:*}"; tag="${size##*:}"
      for seed in "${SEEDS[@]}"; do
        name="hpc_augmented_all_${tag}_s${seed}"
        echo "$name: $(run --job-name="$name" --export=ALL,RUN_NAME="$name" "$JOB" finetune augmented "$seed" \
          hpc_pretrain_augmented_all_s0 "molecules_${molecules}.ckpt")"
      done
    done
    ;;
  chebi25)
    PREP="$(run scripts/slurm/prepare_chebi25.sbatch)"
    echo "prepare_chebi25: $PREP"
    for seed in "${SEEDS[@]}"; do
      for strategy in none all; do
        name="hpc25_augmented_${strategy}_s${seed}"
        args=(finetune augmented "$seed")
        [ "$strategy" = "all" ] && args+=(hpc_pretrain_augmented_all_s0 final.ckpt)
        echo "$name: $(run --job-name="$name" --dependency=afterok:"$PREP" \
          --export=ALL,RUN_NAME="$name",DATA_CONFIG=configs/thesis/data/chebi25_augmented.yml "$JOB" "${args[@]}")"
      done
    done
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
