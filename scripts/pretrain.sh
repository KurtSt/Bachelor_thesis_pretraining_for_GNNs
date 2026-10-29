#!/usr/bin/env bash
# Self-supervised pretraining on the PubChem stream.
#
# Usage: scripts/pretrain.sh <flat|augmented> <strategy> <seed> [extra LightningCLI args...]
#
# Strategies (columns of the experiment matrix):
#   attrmask  atom type of masked atoms only (AttrMask, Hu et al. 2020)
#   atom      all atom-level tasks
#   global    all FG- and molecule-level tasks (no FG task on the flat graph)
#   all       all tasks (hierarchical multi-task)
#   fg        FG task only (augmented graph only)
#   molecule  molecule-level tasks only
#
# Environment:
#   RUN_NAME=<name>        log directory logs/<RUN_NAME>/ (default pretrain_<graph>_<strategy>_s<seed>)
#   STREAM_SIZE=<n>        number of training molecules (default: all of PubChem)
#   VALIDATION_SIZE=<n>    held-out molecules (default 10000)
#   RESUME=<checkpoint>    continue an interrupted run (its checkpoints/latest.ckpt)
#   MIN_FREE_MB=<MiB>      memory guard
#   NUM_WORKERS=<n>        dataloader processes that build the graphs (default 8)
#   WANDB=1                also log to Weights & Biases (see scripts/logger_args.sh)
set -euo pipefail

GRAPH="$1"; STRATEGY="$2"; SEED="$3"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/configs/thesis"

ATOM_TASKS="atom_type,atom_num_hs,bond_type,bond_link"
MOLECULE_TASKS="atom_count,bond_count,hydrogen_count,mol_weight"
case "$GRAPH" in
  flat)      FG_TASKS="" ;;
  augmented) FG_TASKS="fg_type," ;;
  *) echo "unknown graph variant: $GRAPH" >&2; exit 1 ;;
esac
case "$STRATEGY" in
  attrmask) TASKS="atom_type" ;;
  atom)     TASKS="$ATOM_TASKS" ;;
  global)   TASKS="${FG_TASKS}${MOLECULE_TASKS}" ;;
  all)      TASKS="${ATOM_TASKS},${FG_TASKS}${MOLECULE_TASKS}" ;;
  fg)
    [ -n "$FG_TASKS" ] || { echo "the FG task needs the augmented graph" >&2; exit 1; }
    TASKS="fg_type" ;;
  molecule) TASKS="$MOLECULE_TASKS" ;;
  *) echo "unknown strategy: $STRATEGY" >&2; exit 1 ;;
esac

RUN_NAME="${RUN_NAME:-pretrain_${GRAPH}_${STRATEGY}_s${SEED}}"

RUN=()
if [ -n "${MIN_FREE_MB:-}" ]; then
  RUN=("$ROOT/scripts/memory_guard.sh" "$MIN_FREE_MB")
fi
EXTRA=()
if [ -n "${STREAM_SIZE:-}" ]; then EXTRA+=(--data.init_args.stream_size="$STREAM_SIZE"); fi
if [ -n "${VALIDATION_SIZE:-}" ]; then EXTRA+=(--data.init_args.validation_size="$VALIDATION_SIZE"); fi
if [ -n "${RESUME:-}" ]; then EXTRA+=(--ckpt_path="$RESUME"); fi

cd "$ROOT"
source "$ROOT/scripts/logger_args.sh"
logger_args "$RUN_NAME"
"${RUN[@]}" python -m chebai fit \
  --seed_everything="$SEED" \
  --trainer="$CFG/training/pretrain_trainer.yml" \
  --model="$CFG/model/pretrain_${GRAPH}.yml" \
  --model.init_args.tasks="[$TASKS]" \
  --data="$CFG/data/pubchem_stream_${GRAPH}.yml" \
  --data.init_args.batch_size=128 \
  --data.init_args.num_workers="${NUM_WORKERS:-8}" \
  --data.init_args.persistent_workers=false \
  "${LOGGER_ARGS[@]}" \
  "${EXTRA[@]}" \
  "$@"
