#!/usr/bin/env bash
# Exact-SHA CUDA tests and real-video prechecks before replacement training.
#SBATCH --job-name=ctdp-cuda-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err
source /etc/profile
set -euo pipefail
BASE="/data/run01/sczc063/yuzibo"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
cd "${PROJECT_DIR}"
EXPECTED_COMMIT="${CTDP_EXPECTED_COMMIT:?CTDP_EXPECTED_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain)" ]]
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
PRETRAIN="${CT_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "${PRETRAIN}" ]] || { echo "missing CT pretrain: ${PRETRAIN}" >&2; exit 1; }
"${PYTHON}" -m pytest tests/test_ctdp_successful_updates.py \
  tests/test_ctdp_coordinate_contract.py tests/test_truetime_geometry.py \
  tests/test_bata_post_processing_selected_axis.py tests/test_c3_physical_grid_round_trip.py \
  tests/test_train_engine_max_train_iters.py tests/test_duca_ctdp_corrected_geometry.py \
  tests/test_duca_ctdp_coordinate_contract.py tests/test_duca_ct_tubelet_parity.py \
  tests/test_duca_ctdp_launcher_contract.py tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q
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

export REPO_ROOT="${PROJECT_DIR}" CTDP_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
export CTDP_RUN_ROOT="${CTDP_RUN_ROOT:?an external precheck output root is required}"
export PRECHECK_ONLY=1 SEED=3407
for arm in g0 g1 g2 g3; do
  bash scripts/run_duca_ct_dp_revised_thumos_gpu.sh "configs/adatad/thumos/duca_ctdp_geometry_${arm}.py"
done
echo "CTDP_ALL_PRECHECKS_OK"
