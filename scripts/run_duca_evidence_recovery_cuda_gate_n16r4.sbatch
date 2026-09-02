#!/usr/bin/env bash
# Real CUDA forward/backward gate for the corrected Evidence selector.
# This is a gate only; it never writes a formal checkpoint or metric.
#SBATCH --job-name=duca_rec_cuda_gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
source "${BASE}/conda_envs/opentad/bin/activate"
REPO_ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch
from opentad.models.duca.evidence_recovery import DucaEvidenceRecoveryModule

assert torch.cuda.is_available(), "CUDA gate requires a real CUDA device"
device = torch.device("cuda")
torch.manual_seed(8261)
lowres = torch.randn(1, 3, 768, 64, 64, device=device)
mask = torch.ones(1, 768, dtype=torch.bool, device=device)
module = DucaEvidenceRecoveryModule(budget=384, window_size=768).to(device)
out = module.acquire(lowres, valid_mask=mask)
utility = out["scout"]["utility"]
loss = utility.float().mean()
assert torch.isfinite(loss)
loss.backward()
for p in module.parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all()
positions = torch.linspace(0, 767, 384, device=device).round().long().view(1, -1)
h65 = DucaEvidenceRecoveryModule(budget=384, window_size=768, use_h65_selection=True).to(device)
replay = h65.acquire(lowres, valid_mask=mask, h65_positions=positions)
assert replay["selection"]["selected_positions"].shape == (1, 384)
print("[CUDA-GATE] Evidence F/A6 real-shape forward/backward finite")
PY
