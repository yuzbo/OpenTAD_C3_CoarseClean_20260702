#!/bin/bash
#SBATCH -J duca_bootstrap
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --array=0-15
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
mkdir -p "${RUN_ROOT}/bootstrap" /data/run01/sczc063/yuzibo/slurm_logs
cd "${PROJECT_DIR}"

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/bootstrap_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --shard "$SLURM_ARRAY_TASK_ID" --num-shards 16 --draws 10000 --output "${RUN_ROOT}/bootstrap/bootstrap_${SLURM_ARRAY_TASK_ID}.json"
