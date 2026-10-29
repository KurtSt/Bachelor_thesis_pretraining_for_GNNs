#!/bin/bash -l
# Check of the installation on a CPU node (partition workq, no waiting for a GPU):
# tests, the preparation of all datasets (the step "prepare" of the matrix) and a tiny
# pretraining + fine-tuning through job.sbatch (logged to W&B after `wandb login`).
#
#   sbatch scripts/slurm/cpu_check.sbatch
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --job-name=cpu_check
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"

python -m unittest discover -s tests/unit -t . -p "test*.py"
python -m unittest tests.integration.testPubChemPretraining tests.integration.testPubChemStream

RUN_NAME=hpc_prepare bash scripts/slurm/job.sbatch prepare

PRETRAIN=cpucheck_pretrain_flat_all
rm -rf "logs/$PRETRAIN" logs/cpucheck_flat_all_s0
STREAM_SIZE=2000 VALIDATION_SIZE=200 NUM_WORKERS=4 RUN_NAME="$PRETRAIN" \
  bash scripts/slurm/job.sbatch pretrain flat all 0
RUN_NAME=cpucheck_flat_all_s0 NUM_WORKERS=0 \
  bash scripts/slurm/job.sbatch finetune flat 0 "$PRETRAIN" final.ckpt \
  --trainer.min_epochs=1 --trainer.max_epochs=1 --trainer.limit_train_batches=20 \
  --trainer.limit_val_batches=10 --trainer.limit_test_batches=10
ls logs/cpucheck_flat_all_s0/version_0_test/metrics.csv
echo CPU_CHECK_OK
