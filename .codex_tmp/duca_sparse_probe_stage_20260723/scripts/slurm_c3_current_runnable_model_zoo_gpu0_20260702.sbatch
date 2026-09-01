#!/bin/bash
#SBATCH -J c3_zoo_g0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/slurm_logs/%x_%j.err

set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
mkdir -p "${BASE}/projects/c3_lowres_action_probe/slurm_logs"

export CUDA_VISIBLE_DEVICES=0
export RUN_TAG="${RUN_TAG:-c3_current_runnable_model_zoo_gpu0_${SLURM_JOB_ID}_$(date '+%Y%m%d_%H%M%S_%z')}"
export PROJECT_DIR="${PROJECT_DIR:-${BASE}/OpenTAD_C3_CoarseClean_20260702}"

bash "${PROJECT_DIR}/scripts/run_c3_current_runnable_model_zoo_gpu0_20260702.sh"
