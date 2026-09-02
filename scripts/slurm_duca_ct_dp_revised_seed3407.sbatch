#!/bin/bash
#SBATCH -J ct_dp_revised_s3407
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail
source /etc/profile
BASE="/data/run01/sczc063/yuzibo"
PROJECT_DIR="${PROJECT_DIR:-${BASE}/projects/opentad_duca_ct_dp_revised_20260902}"
cd "${PROJECT_DIR}"
REPO_ROOT="${PROJECT_DIR}" BASE="${BASE}" \
  bash "${PROJECT_DIR}/scripts/run_duca_ct_dp_revised_thumos_gpu.sh" \
  "configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos_revised.py"
