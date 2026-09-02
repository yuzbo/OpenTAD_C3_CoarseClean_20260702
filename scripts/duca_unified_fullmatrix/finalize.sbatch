#!/bin/bash
#SBATCH -J duca_finalize
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
cd "${PROJECT_DIR}"

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python tools/bata/aggregate_duca_unified_fullmatrix.py --matrix scripts/duca_unified_fullmatrix/matrix.json --run-root "$RUN_ROOT" --bootstrap-dir "${RUN_ROOT}/bootstrap" --output "${RUN_ROOT}/duca_unified_fullmatrix_summary.json"
