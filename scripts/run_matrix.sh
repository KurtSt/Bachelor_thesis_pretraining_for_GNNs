#!/usr/bin/env bash
# Runs the experiment matrix: graph (flat, augmented) x pretraining strategy
# (none, attrmask, atom, global, all), each fine-tuned on ChEBI with every given seed.
#
# Usage: scripts/run_matrix.sh [seed ...]            (default seeds: 0 1 2)
#
# Each (graph, strategy) is pretrained once (seed PRETRAIN_SEED) and transferred to all
# fine-tuning seeds. Finished steps are skipped when the script is started again
# (a step is finished when it completed and left a `.done` marker).
#
# Environment:
#   GRAPHS       subset of graphs (default "flat augmented")
#   STRATEGIES   subset of strategies (default "none attrmask atom global all")
#   CHECKPOINT   pretraining checkpoint to transfer (default final.ckpt; for the data
#                scaling e.g. molecules_10000000.ckpt)
#   PRETRAIN_SEED  seed of the pretraining runs (default 0)
#   RUN_PREFIX   prefix of all run names (default "matrix", "smoke" with SMOKE=1)
#   SMOKE=1      tiny rehearsal: 2000 PubChem molecules, 2 fine-tuning epochs, few batches
#   further variables are passed on to the launchers (BATCH_SIZE, MIN_FREE_MB,
#   STREAM_SIZE, NUM_WORKERS)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
PRETRAIN_SEED="${PRETRAIN_SEED:-0}"
GRAPHS="${GRAPHS:-flat augmented}"
STRATEGIES="${STRATEGIES:-none attrmask atom global all}"
CHECKPOINT="${CHECKPOINT:-final.ckpt}"

FINETUNE_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  PREFIX="${RUN_PREFIX:-smoke}"
  export STREAM_SIZE="${STREAM_SIZE:-2000}" VALIDATION_SIZE="${VALIDATION_SIZE:-200}"
  FINETUNE_ARGS=(--trainer.min_epochs=2 --trainer.max_epochs=2
    --trainer.limit_train_batches=50 --trainer.limit_val_batches=20
    --trainer.limit_test_batches=20)
else
  PREFIX="${RUN_PREFIX:-matrix}"
fi

finished() { [ -f "logs/$1/.done" ]; }
mark_finished() { touch "logs/$1/.done"; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

checkpoint_of() {  # pretraining run name
  local version_dir
  version_dir="$(ls -d logs/"$1"/version_* | grep -v _test | sort -V | tail -n1)"
  echo "$version_dir/checkpoints/$CHECKPOINT"
}

pretrain() {  # graph, strategy -> sets CKPT
  local name="${PREFIX}_pretrain_$1_$2_s${PRETRAIN_SEED}"
  if finished "$name"; then
    log "skip $name (finished)"
  else
    log "start $name"
    rm -rf "logs/$name"  # remove a previous incomplete attempt
    RUN_NAME="$name" scripts/pretrain.sh "$1" "$2" "$PRETRAIN_SEED"
  fi
  CKPT="$(checkpoint_of "$name")"
  [ -f "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 1; }
  mark_finished "$name"
}

finetune() {  # run name, graph, seed, extra args...
  local name="$1" graph="$2" seed="$3"
  shift 3
  if finished "$name"; then
    log "skip $name (finished)"
    return
  fi
  log "start $name"
  rm -rf "logs/$name"
  RUN_NAME="$name" scripts/finetune.sh "$graph" "$seed" "${FINETUNE_ARGS[@]}" "$@"
  mark_finished "$name"
}

for graph in $GRAPHS; do
  for strategy in $STRATEGIES; do
    extra=()
    if [ "$strategy" != "none" ]; then
      pretrain "$graph" "$strategy"
      extra=(--model.init_args.pretrained_checkpoint="$CKPT")
    fi
    for seed in "${SEEDS[@]}"; do
      finetune "${PREFIX}_${graph}_${strategy}_s${seed}" "$graph" "$seed" "${extra[@]}"
    done
  done
done

log "done"
python scripts/collect_results.py logs | grep -E "^run|^${PREFIX}_"
