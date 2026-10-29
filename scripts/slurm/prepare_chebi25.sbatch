#!/bin/bash -l
# Builds the augmented ChEBI25 dataset (ChEBI v241, vocabularies of ChEBI50 frozen) on a
# CPU node; the fine-tuning jobs then only load it.
#
#   sbatch scripts/slurm/prepare_chebi25.sbatch
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --job-name=prepare_chebi25
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"

if [ -f logs/prepare_chebi25/.done ]; then echo "skip (prepared)"; exit 0; fi
rm -rf logs/prepare_chebi25
WANDB=0 DATA_CONFIG=configs/thesis/data/chebi25_augmented.yml NUM_WORKERS=0 SKIP_TEST=1 \
  RUN_NAME=prepare_chebi25 scripts/finetune.sh augmented 0 --trainer.fast_dev_run=true
mkdir -p logs/prepare_chebi25 && touch logs/prepare_chebi25/.done
