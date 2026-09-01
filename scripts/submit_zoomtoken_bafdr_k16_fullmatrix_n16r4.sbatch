#!/bin/bash
#SBATCH --job-name=zt-bafdr-k16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_compute/slurm_%j.log
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_compute/slurm_%j.err

set -euo pipefail

RUN_ROOT="/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_compute"
mkdir -p "${RUN_ROOT}"

cd /data/run01/sczc063/yuzibo/projects/zoomtoken_s2_v3_rpl1_2c8f25fe_src

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

echo "[SLURM] Launching BA-FDR K16 Full-Matrix Orchestration..."
bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh
