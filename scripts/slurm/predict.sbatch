#!/bin/bash -l
# Tests the best checkpoint of one finished fine-tuning run again and saves its test
# predictions (logs/<run>/version_<n>_predict/test_predictions.pt) for the analyses per
# class and per molecule. CPU only; one array task per line of the run list.
#
#   sbatch --array=1-<N> scripts/slurm/predict.sbatch <file with one run name per line>
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH --time=04:00:00
#SBATCH --job-name=predict
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"

RUN_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
case "$RUN_NAME" in
  *_flat_*) GRAPH=flat ;;
  *_augmented_*) GRAPH=augmented ;;
  *) echo "unknown graph in $RUN_NAME" >&2; exit 1 ;;
esac
case "$RUN_NAME" in
  hpc25_*) export DATA_CONFIG=configs/thesis/data/chebi25_augmented.yml ;;
esac
if ls logs/"$RUN_NAME"/version_*_predict/test_predictions.pt > /dev/null 2>&1; then
  echo "skip $RUN_NAME (predictions exist)"; exit 0
fi
echo "predict $RUN_NAME ($GRAPH) on $(hostname)"
WANDB=0 TEST_ONLY=1 TEST_SUFFIX=_predict NUM_WORKERS=0 RUN_NAME="$RUN_NAME" \
  scripts/finetune.sh "$GRAPH" 0
