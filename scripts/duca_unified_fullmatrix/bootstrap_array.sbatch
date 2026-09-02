#!/bin/bash
#SBATCH -J duca_bootstrap
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --array=0-15
#SBATCH -t 02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${RUN_ROOT:-${BASE}/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
mkdir -p "${RUN_ROOT}/bootstrap" "${BASE}/slurm_logs"
cd "${PROJECT_DIR}"

source /etc/profile >/dev/null 2>&1 || true
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"

python tools/bata/bootstrap_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards 16 --draws 10000 --output "${RUN_ROOT}/bootstrap/bootstrap_${SLURM_ARRAY_TASK_ID}.json"
