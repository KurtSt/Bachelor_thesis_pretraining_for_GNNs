#!/usr/bin/env bash
# Supervised training on ChEBI50 (fine-tuning) on one graph variant.
#
# Usage: scripts/finetune.sh <flat|augmented> <seed> [extra LightningCLI args...]
#   Without pretraining, the encoder starts from random weights; pretrained runs pass
#   --model.init_args.pretrained_checkpoint=<ckpt> as extra argument.
# Environment: RUN_NAME=<name>, extra arguments may include --ckpt_path=<last.ckpt> to continue a run, BATCH_SIZE=<n> (default 128, effective batch size stays 512), SKIP_TEST=1 (no test run),
#   MIN_FREE_MB=<MiB> (memory guard), NUM_WORKERS=<n> (dataloader processes, default 0:
#   each worker holds a copy of the dataset, too much for 32 GB of RAM),
#   WANDB=1 (also log to Weights & Biases, see scripts/logger_args.sh),
#   DATA_CONFIG=<yml> (other dataset, e.g. configs/thesis/data/chebi25_augmented.yml),
#   TEST_ONLY=1 (no training: test the best checkpoint of RUN_NAME again; with
#   TEST_SUFFIX=_predict the test run is logged to version_<n>_predict).
set -euo pipefail

GRAPH="$1"; SEED="$2"; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/configs/thesis"

case "$GRAPH" in
  flat)      MODEL="$CFG/model/gat_flat.yml";      DATA="$CFG/data/chebi50_flat.yml" ;;
  augmented) MODEL="$CFG/model/gat_augmented.yml"; DATA="$CFG/data/chebi50_augmented.yml" ;;
  *) echo "unknown graph variant: $GRAPH" >&2; exit 1 ;;
esac

DATA="${DATA_CONFIG:-$DATA}"

# logs and checkpoints go to logs/<RUN_NAME>/version_<n>/
RUN_NAME="${RUN_NAME:-finetune_${GRAPH}_s${SEED}}"

# optional memory guard, e.g. MIN_FREE_MB=2000 (scripts/memory_guard.sh)
RUN=()
if [ -n "${MIN_FREE_MB:-}" ]; then
  RUN=("$ROOT/scripts/memory_guard.sh" "$MIN_FREE_MB")
fi

# the effective batch size (batch size x gradient accumulation) is the same for all
# runs; a smaller BATCH_SIZE only reduces GPU memory
EFFECTIVE_BATCH_SIZE=512
BATCH_SIZE="${BATCH_SIZE:-128}"
if [ $((EFFECTIVE_BATCH_SIZE % BATCH_SIZE)) -ne 0 ]; then
  echo "BATCH_SIZE must divide $EFFECTIVE_BATCH_SIZE" >&2; exit 1
fi

cd "$ROOT"
source "$ROOT/scripts/logger_args.sh"
COMMON=(
  --seed_everything="$SEED"
  --trainer="$CFG/training/finetune_trainer.yml"
  --model="$MODEL"
  --model.criterion="$CFG/loss/bce_weighted.yml"
  --model.train_metrics="$CFG/metrics/micro-macro-f1.yml"
  --model.val_metrics="$CFG/metrics/micro-macro-f1.yml"
  --model.test_metrics="$CFG/metrics/micro-macro-f1.yml"
  --model.pass_loss_kwargs=false
  --data="$DATA"
  --data.init_args.batch_size="$BATCH_SIZE"
  --trainer.accumulate_grad_batches="$((EFFECTIVE_BATCH_SIZE / BATCH_SIZE))"
  --data.init_args.num_workers="${NUM_WORKERS:-0}"
  --data.init_args.persistent_workers=false
)
if [ "${TEST_ONLY:-0}" != "1" ]; then
  logger_args "$RUN_NAME"
  "${RUN[@]}" python -m chebai fit "${COMMON[@]}" "${LOGGER_ARGS[@]}" "$@"
fi

# evaluate the checkpoint with the best validation macro-F1 on the test split
if [ "${SKIP_TEST:-0}" != "1" ]; then
  VERSION_DIR="$(ls -d logs/"$RUN_NAME"/version_* | grep -v -E '_test|_predict' | sort -V | tail -n1)"
  # a resumed run writes to a new version directory: take the best checkpoint of all
  # versions, ranked by the validation macro-F1 in its file name
  BEST="$(ls logs/"$RUN_NAME"/version_*/checkpoints/best_*.ckpt \
    | sed -E 's/.*val_macro-f1=([0-9.]+).*/\1 &/' | sort -g | tail -n1 | cut -d' ' -f2)"
  echo "Testing $BEST"
  logger_args "$RUN_NAME" "$(basename "$VERSION_DIR")${TEST_SUFFIX:-_test}"
  "${RUN[@]}" python -m chebai test "${COMMON[@]}" "${LOGGER_ARGS[@]}" "$@" \
    --ckpt_path="$BEST"
fi
