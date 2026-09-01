#!/bin/bash
#SBATCH --job-name=zt-bafdr-k16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

BASE="/data/run01/sczc063/yuzibo"
mkdir -p "${BASE}/slurm_logs"

PROJECT_DIR="${BASE}/projects/zoomtoken_et_trc_v001"
cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

source "${BASE}/conda_envs/opentad/bin/activate"

CONFIG="${1:-configs/adatad/thumos/bafdr_k16_full_seed4407.py}"
echo "[SLURM] Executing BA-FDR K16 cell: ${CONFIG} on job ${SLURM_JOB_ID:-interactive}"
bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh "${CONFIG}"
