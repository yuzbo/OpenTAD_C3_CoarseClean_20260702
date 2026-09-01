#!/bin/bash
#SBATCH -J ct_dp_densevit_s3407
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

BASE="/data/run01/sczc063/yuzibo"
mkdir -p "${BASE}/slurm_logs"

PROJECT_DIR="${BASE}/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901"
CONFIG="configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py"
SEED=3407

cd "${PROJECT_DIR}"
bash "${PROJECT_DIR}/scripts/run_duca_ct_dual_phase_bamod_thumos_gpu.sh" "${CONFIG}"
