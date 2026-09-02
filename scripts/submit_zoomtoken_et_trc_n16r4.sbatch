#!/bin/bash
#SBATCH --job-name=zt-et-trc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/et_trc_videomae_compute/slurm_%j.log
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/et_trc_videomae_compute/slurm_%j.err

source /etc/profile
set -euo pipefail
mkdir -p /data/run01/sczc063/yuzibo/projects/et_trc_videomae_compute

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902}"
cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export ETTRC_PRETRAIN="${ETTRC_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "${ETTRC_PRETRAIN}" ]] || { echo "missing ET-TRC pretrain: ${ETTRC_PRETRAIN}" >&2; exit 1; }

# 1. Run Pre-flight Diagnostic ZT-DIAG-2025-01
echo "[SLURM] Running ZT-DIAG-2025-01 pre-flight Taylor Residual Diagnostic..."
bash scripts/run_zoomtoken_et_trc_diag_n16r4.sh

# 2. Launch ET-TRC Training
echo "[SLURM] Launching ET-TRC Training on THUMOS14..."
python tools/train.py configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py \
  --cfg-options model.backbone.backbone.stride_k=4
