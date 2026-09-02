#!/usr/bin/env bash
# Real CUDA pretrain/load/finite-update gate for ET-TRC.
#SBATCH --job-name=et-trc-cuda-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err
source /etc/profile
set -euo pipefail
BASE="/data/run01/sczc063/yuzibo"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
cd "${PROJECT_DIR}"
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
PRETRAIN="${ETTRC_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "${PRETRAIN}" ]] || { echo "missing ET-TRC pretrain: ${PRETRAIN}" >&2; exit 1; }
python -m py_compile opentad/models/backbones/et_trc_videomae.py
python - <<'PY'
import torch
from opentad.models.backbones.et_trc_videomae import ETTRCVisionTransformerAdapter
assert torch.cuda.is_available()
model = ETTRCVisionTransformerAdapter(embed_dims=12, depth=1, num_heads=3, total_frames=16, num_frames=16, jacobian_rank=4).cuda()
x = torch.randn(1, 3, 16, 160, 160, device="cuda")
y = model(x)
loss = y.square().mean()
loss.backward()
assert torch.isfinite(loss).item() and torch.isfinite(y).all().item()
print("ET_TRC_CUDA_GATE_OK", tuple(y.shape))
PY
