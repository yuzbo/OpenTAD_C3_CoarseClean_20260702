#!/usr/bin/env bash
# Real CUDA gradient gate for BA-FDR before the seed-4407 screen.
#SBATCH --job-name=bafdr-gradient-gate
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err
source /etc/profile
set -euo pipefail
BASE="/data/run01/sczc063/yuzibo"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
EXPECTED_COMMIT="${BAFDR_EXPECTED_COMMIT:?BAFDR_EXPECTED_COMMIT is required}"
cd "${PROJECT_DIR}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "BAFDR_EXPECTED_COMMIT must be a full lowercase SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "BAFDR checkout HEAD mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "BAFDR checkout is not clean" >&2; exit 2; }
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
python -m pytest -q \
  tests/test_bafdr_k16_static_protocol.py \
  tests/test_bafdr_gradient_path.py \
  tests/test_bafdr_execution_count.py \
  tests/test_bafdr_router_endpoint.py
python - <<'PY'
import torch
from torch import nn
assert torch.cuda.is_available()
device = torch.device("cuda")
gamma = nn.Parameter(torch.zeros((), device=device))
q0 = nn.Conv1d(4, 4, 1).to(device)
q1 = nn.Conv1d(4, 4, 1).to(device)
opt = torch.optim.SGD([gamma, *q0.parameters(), *q1.parameters()], lr=0.1)
local = torch.randn(2, 4, 3, device=device)
global_ = torch.randn(2, 4, 3, device=device)
gate = torch.sigmoid(torch.randn(2, 1, 3, device=device))
for step in range(3):
    opt.zero_grad(set_to_none=True)
    residual = gamma * gate * (q0(local) - q1(global_))
    loss = (residual + global_).square().mean()
    loss.backward()
    assert torch.isfinite(loss).item() and torch.isfinite(gamma.grad).item()
    if step == 0:
        assert gamma.grad.abs().item() > 0
    opt.step()
assert gamma.detach().abs().item() > 0
assert q0.weight.grad.abs().sum().item() > 0 and q1.weight.grad.abs().sum().item() > 0
print("BAFDR_GRADIENT_GATE_OK", gamma.item())
PY
