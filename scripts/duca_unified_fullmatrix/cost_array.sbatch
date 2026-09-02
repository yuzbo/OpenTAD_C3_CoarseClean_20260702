#!/bin/bash
#SBATCH -J duca_cost
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH -t 04:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
MATRIX_JSON="${PROJECT_DIR}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${RUN_ROOT}/cost" /data/run01/sczc063/yuzibo/slurm_logs
cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix "$MATRIX_JSON" --run-root "$RUN_ROOT" --cost-index "$SLURM_ARRAY_TASK_ID" --output "${RUN_ROOT}/cost/cost_${SLURM_ARRAY_TASK_ID}.json"
