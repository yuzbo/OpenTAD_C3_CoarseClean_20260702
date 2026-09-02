#!/usr/bin/env bash
# One real CUDA forward/backward gate for CT-DP before any long campaign.
#SBATCH --job-name=ctdp-cuda-gate
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
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
PRETRAIN="${CT_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "${PRETRAIN}" ]] || { echo "missing CT pretrain: ${PRETRAIN}" >&2; exit 1; }
python -m py_compile opentad/models/bricks/scale_adaptive_conv1d.py opentad/models/backbones/backbone_wrapper.py
"${PYTHON}" - <<'PY'
import torch
from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d
assert torch.cuda.is_available()
layer = ContinuousTimeScaleAdaptiveConv1d(4, 4, reference_spacing_mode="level_nominal").cuda()
x = torch.randn(2, 4, 16, device="cuda", requires_grad=True)
tau = torch.arange(16, device="cuda", dtype=torch.float32).repeat(2, 1)
y = layer(x, temporal_positions=tau)
loss = y.square().mean()
loss.backward()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
assert torch.isfinite(y).all().item()
print("CT_CUDA_GATE_OK", tuple(y.shape))
PY
