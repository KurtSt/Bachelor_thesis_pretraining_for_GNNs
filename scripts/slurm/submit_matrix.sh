#!/usr/bin/env bash
# Submits the experiment matrix to SLURM: one preparation job, then one pretraining job per
# (graph, strategy), and one fine-tuning job per (graph, strategy, seed) that starts when
# its pretraining has finished. Jobs run in parallel as far as GPUs are free.
# A pretraining gets PRETRAIN_JOBS consecutive jobs (default 3): each one continues from
# the checkpoint of the previous one, as a job may run at most 48 hours.
#
# Usage: scripts/slurm/submit_matrix.sh [seed ...]          (default seeds: 0 1 2)
#
# Environment (as for scripts/run_matrix.sh):
#   GRAPHS, STRATEGIES, RUN_PREFIX (default "hpc"), PRETRAIN_SEED, CHECKPOINT,
#   STREAM_SIZE (pretraining molecules), PREPARE_JOB (id of a finished preparation job,
#   or "none" to skip preparing), DRY_RUN=1 (only print the sbatch commands),
#   SBATCH_ARGS (further sbatch arguments, e.g. "--time=12:00:00"), PRETRAIN_JOBS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
PREFIX="${RUN_PREFIX:-hpc}"
PRETRAIN_SEED="${PRETRAIN_SEED:-0}"
PRETRAIN_JOBS="${PRETRAIN_JOBS:-3}"
GRAPHS="${GRAPHS:-flat augmented}"
STRATEGIES="${STRATEGIES:-none attrmask atom global all}"
CHECKPOINT="${CHECKPOINT:-final.ckpt}"
JOB="scripts/slurm/job.sbatch"

submit() {  # name, dependency ("afterok:<id>", "afterany:<id>" or ""), job arguments...
  local name="$1" dep="$2"; shift 2
  local args=(--parsable --job-name="$name" ${SBATCH_ARGS:-})
  if [ -n "$dep" ]; then args+=(--dependency="$dep"); fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "sbatch ${args[*]} --export=ALL,RUN_NAME=$name $JOB $*" >&2
    echo "DRY_$name"
  else
    sbatch "${args[@]}" --export=ALL,RUN_NAME="$name" "$JOB" "$@"
  fi
}

PREPARE="${PREPARE_JOB:-}"
if [ -z "$PREPARE" ]; then
  PREPARE="$(submit "${PREFIX}_prepare" "" prepare)"
  echo "prepare: $PREPARE"
fi
AFTER_PREPARE=""
if [ "$PREPARE" != "none" ]; then AFTER_PREPARE="afterok:$PREPARE"; fi

for graph in $GRAPHS; do
  for strategy in $STRATEGIES; do
    dep="$AFTER_PREPARE"; pretrained=()
    if [ "$strategy" != "none" ]; then
      name="${PREFIX}_pretrain_${graph}_${strategy}_s${PRETRAIN_SEED}"
      job="$(submit "$name" "$AFTER_PREPARE" pretrain "$graph" "$strategy" "$PRETRAIN_SEED")"
      jobs="$job"
      for _ in $(seq 2 "$PRETRAIN_JOBS"); do
        # starts after the previous job, also at its time limit, and resumes from latest.ckpt
        job="$(submit "$name" "afterany:$job" pretrain "$graph" "$strategy" "$PRETRAIN_SEED")"
        jobs="$jobs $job"
      done
      echo "$name: $jobs"
      dep="afterok:$job"; pretrained=("$name" "$CHECKPOINT")
    fi
    for seed in "${SEEDS[@]}"; do
      name="${PREFIX}_${graph}_${strategy}_s${seed}"
      echo "$name: $(submit "$name" "$dep" finetune "$graph" "$seed" "${pretrained[@]}")"
    done
  done
done
