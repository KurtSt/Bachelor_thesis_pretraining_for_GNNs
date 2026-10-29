#!/bin/bash -l
# One step of the experiment matrix on HPC3 (submitted by scripts/slurm/submit_matrix.sh).
#
#   sbatch scripts/slurm/job.sbatch prepare
#   sbatch scripts/slurm/job.sbatch pretrain <graph> <strategy> <seed>
#   sbatch scripts/slurm/job.sbatch finetune <graph> <seed> [<pretraining run> <checkpoint file> [extra args...]]
#
# RUN_NAME is passed with --export. A step whose logs/<RUN_NAME>/.done exists is skipped.
# A pretraining that stopped at the time limit continues from its newest latest.ckpt
# when the job is submitted again (submit_matrix.sh queues several jobs per pretraining).
# With EXTEND=1 a finished fine-tuning continues from its last.ckpt (e.g. 100 -> 200 epochs).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100.80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=256000
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail

source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"
# log to W&B once `wandb login` was done on the cluster
if grep -qs api.wandb.ai ~/.netrc; then export WANDB="${WANDB:-1}"; fi
# the compute nodes reach W&B through a proxy; the default 90 s for wandb.init sometimes
# runs out and stops the job
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
echo "job $SLURM_JOB_ID on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader)"

STEP="$1"; shift
if [ -n "${RUN_NAME:-}" ] && [ -f "logs/$RUN_NAME/.done" ] && [ "${EXTEND:-0}" != "1" ]; then
  echo "skip $RUN_NAME (finished)"; exit 0
fi

case "$STEP" in
  prepare)
    # build all datasets once, parallel jobs must not write the same files
    NUM_WORKERS=0 RUN_NAME=prepare_flat SKIP_TEST=1 \
      scripts/finetune.sh flat 0 --trainer.fast_dev_run=true
    NUM_WORKERS=0 RUN_NAME=prepare_augmented SKIP_TEST=1 \
      scripts/finetune.sh augmented 0 --trainer.fast_dev_run=true
    STREAM_SIZE=1000 VALIDATION_SIZE=100 RUN_NAME=prepare_pubchem \
      scripts/pretrain.sh flat attrmask 0
    ;;
  pretrain)
    LATEST="$(ls logs/"$RUN_NAME"/version_*/checkpoints/latest.ckpt 2>/dev/null | sort -V | tail -n1 || true)"
    if [ -n "$LATEST" ]; then
      echo "resume from $LATEST"
      export RESUME="$LATEST"
    else
      rm -rf "logs/$RUN_NAME"
    fi
    NUM_WORKERS="${NUM_WORKERS:-24}" scripts/pretrain.sh "$@"
    ;;
  finetune)
    GRAPH="$1"; SEED="$2"; PRETRAIN_RUN="${3:-}"; CKPT_FILE="${4:-final.ckpt}"
    shift $(($# < 4 ? $# : 4))
    EXTRA=("$@")
    if [ -n "$PRETRAIN_RUN" ]; then
      # a resumed pretraining writes its later checkpoints to a new version directory
      CKPT="$(ls logs/"$PRETRAIN_RUN"/version_*/checkpoints/"$CKPT_FILE" | sort -V | tail -n1)"
      echo "pretrained encoder: $CKPT"
      EXTRA+=(--model.init_args.pretrained_checkpoint="$CKPT")
    fi
    LAST="$(ls logs/"$RUN_NAME"/version_*/checkpoints/last.ckpt 2>/dev/null | sort -V | tail -n1 || true)"
    if [ "${EXTEND:-0}" = "1" ] && [ -n "$LAST" ]; then
      # continue a finished run up to the epochs of the current trainer config
      echo "continue from $LAST"
      rm -f "logs/$RUN_NAME/.done"
      EXTRA+=(--ckpt_path="$LAST")
    else
      rm -rf "logs/$RUN_NAME"
    fi
    NUM_WORKERS="${NUM_WORKERS:-8}" scripts/finetune.sh "$GRAPH" "$SEED" "${EXTRA[@]}"
    ;;
  *) echo "unknown step: $STEP" >&2; exit 1 ;;
esac

if [ -n "${RUN_NAME:-}" ]; then mkdir -p "logs/$RUN_NAME"; touch "logs/$RUN_NAME/.done"; fi
