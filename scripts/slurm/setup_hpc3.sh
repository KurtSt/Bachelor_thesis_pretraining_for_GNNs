#!/bin/bash -l
# One-time setup of the environment on HPC3 (Rocky Linux 8, run on the login node).
# Data, logs and the virtual environment live on /share, the code in ~/ba-pretraining.
set -euo pipefail

S=/share/users/student/k/ksteinkamp/ba
mkdir -p "$S"/{tmp,pipcache,data,logs}
cat > "$S/env.sh" <<ENV
# Environment for the pretraining experiments on HPC3 (source this file).
command -v spack > /dev/null || source /appl/spack/share/spack/setup-env.sh
spack load python@3.10.2
export TMPDIR=$S/tmp
export PIP_CACHE_DIR=$S/pipcache
source $S/venv/bin/activate
ENV

spack load python@3.10.2
export TMPDIR="$S/tmp" PIP_CACHE_DIR="$S/pipcache"
python3 -m venv "$S/venv"
source "$S/venv/bin/activate"
pip install -U pip wheel setuptools
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric==2.6.1
# the prebuilt torch_scatter wheels need glibc >= 2.32 (Rocky 8 has 2.28): build it from
# source; only scatter_add is used, which is plain PyTorch, so the CPU extension suffices
spack load gcc@11.2.0
CC=gcc CXX=g++ FORCE_ONLY_CPU=1 MAX_JOBS=8 \
  pip install --no-build-isolation --no-binary torch-scatter torch-scatter==2.1.2
pip install -e ~/ba-pretraining
pip install pytest wandb

cd ~/ba-pretraining
ln -sfn "$S/data" data
ln -sfn "$S/logs" logs
mkdir -p logs/slurm
