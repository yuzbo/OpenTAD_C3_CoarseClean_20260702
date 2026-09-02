#!/usr/bin/env bash
# Real CUDA gradient gate for BA-FDR before the seed-4407 screen.
#SBATCH --job-name=bafdr-gradient-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err
set -euo pipefail
source /etc/profile
BASE="/data/run01/sczc063/yuzibo"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
cd "${PROJECT_DIR}"
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
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
