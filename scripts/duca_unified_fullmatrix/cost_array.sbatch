#!/bin/bash
#SBATCH -J duca_cost
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH -t 04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${RUN_ROOT:-${BASE}/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
MATRIX_JSON="${PROJECT_DIR}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${RUN_ROOT}/cost" "${BASE}/slurm_logs"
cd "${PROJECT_DIR}"

source /etc/profile >/dev/null 2>&1 || true
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix "$MATRIX_JSON" --run-root "$RUN_ROOT" --cost-index "$SLURM_ARRAY_TASK_ID" --output "${RUN_ROOT}/cost/cost_${SLURM_ARRAY_TASK_ID}.json"
