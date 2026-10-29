#!/bin/bash -l
# Analyses the saved test predictions per class and per molecule (scripts/analyse_predictions.py).
#
#   sbatch scripts/slurm/analyse.sbatch
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --job-name=analyse
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"

OBO=data/chebi_v241/raw/chebi.obo
python scripts/analyse_predictions.py --logs logs --runs 'hpc_*' \
  --processed data/chebi_v241/ChEBI50/processed --obo "$OBO" --out logs/analysis/chebi50
# ChEBI25 only once its fine-tunings have written test predictions
if ls logs/hpc25_*/version_*/test_predictions.pt > /dev/null 2>&1; then
  python scripts/analyse_predictions.py --logs logs --runs 'hpc25_*' \
    --processed data/chebi_v241/ChEBI25/processed --obo "$OBO" --out logs/analysis/chebi25
fi
