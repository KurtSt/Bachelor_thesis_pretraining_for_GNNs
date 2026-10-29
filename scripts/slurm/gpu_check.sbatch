#!/bin/bash -l
# Short check of a GPU node: CUDA, tests and one training step on the GPU.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|H100.80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:20:00
#SBATCH --job-name=gpu_check
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
source /share/users/student/k/ksteinkamp/ba/env.sh
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
python -m unittest discover -s tests/unit -t . -p "test*.py"
python -m unittest tests.integration.testPubChemPretraining tests.integration.testPubChemStream
python - <<'PY'
import torch
from chebai_graph.models.pretraining import GATGraphPretraining
from tests.unit.pretraining.utils import chain_graph, collate
from tests.unit.pretraining.testPretrainingModel import CONFIG, ALL_TASKS
model = GATGraphPretraining(config=CONFIG, tasks=ALL_TASKS, out_dim=0, input_dim=0, n_shared_node_columns=2).cuda()
batch = collate([chain_graph(n, augmented=True) for n in (12, 9, 15)])
loss = model._execute(batch, 0, prefix="train_", log=False)["loss"]
loss.backward()
print("gpu training step ok, loss", float(loss))
PY
echo GPU_CHECK_OK
