#!/bin/bash
#SBATCH --job-name=zt-et-trc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /etc/profile
set -euo pipefail

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_DIR}"

EXPECTED_COMMIT="${ETTRC_CORRECTION_COMMIT:-}"
if [[ -n "${EXPECTED_COMMIT}" && "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  echo "ET-TRC checkout HEAD mismatch: expected ${EXPECTED_COMMIT}, got $(git rev-parse HEAD)" >&2
  exit 2
fi
[[ "$(git branch --show-current)" == "codex/zoomtoken-et-trc-correction-20260902" ]] || {
  echo "ET-TRC submit requires correction branch, got $(git branch --show-current)" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "ET-TRC checkout is not clean" >&2; exit 2; }

SHORT_COMMIT="$(git rev-parse --short HEAD)"
RUN_ROOT="${ETTRC_RUN_ROOT:-${BASE}/projects/zoomtoken_et_trc_fix_${SHORT_COMMIT}}"
mkdir -p "${RUN_ROOT}"

module load cuda/11.8
module load miniforge3/24.11

source "${BASE}/conda_envs/opentad/bin/activate"
export ETTRC_PRETRAIN="${ETTRC_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "${ETTRC_PRETRAIN}" ]] || { echo "missing ET-TRC pretrain: ${ETTRC_PRETRAIN}" >&2; exit 1; }

# 1. Run Pre-flight Diagnostic ZT-DIAG-2025-01
echo "[SLURM] Running ZT-DIAG-2025-01 pre-flight Taylor Residual Diagnostic..."
bash scripts/run_zoomtoken_et_trc_diag_n16r4.sh

# 2. Launch ET-TRC Training
echo "[SLURM] Launching ET-TRC Training on THUMOS14..."
python tools/train.py configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py \
  --seed 4407 --cfg-options model.backbone.backbone.stride_k=4 "work_dir=${RUN_ROOT}"
