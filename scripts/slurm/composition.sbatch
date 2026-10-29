#!/bin/bash -l
# Chemical composition of the PubChem pretraining stream and of ChEBI50
# (scripts/pubchem_composition.py).
#
#   sbatch scripts/slurm/composition.sbatch
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --job-name=composition
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"

python scripts/pubchem_composition.py --pubchem data/Pubchem/2026-09-01 \
  --processed data/chebi_v241/ChEBI50/processed --n 200000 --workers "$SLURM_CPUS_PER_TASK" \
  --out logs/analysis/composition
