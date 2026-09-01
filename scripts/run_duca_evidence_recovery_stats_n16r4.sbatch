#!/usr/bin/env bash
#SBATCH --job-name=duca_rec_stats
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err

source /etc/profile
set -euo pipefail

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required}"
OUTPUT_PATH="${RUN_ROOT}/statistical_analysis.json"

echo "[INFO] Running Hierarchical Paired Bootstrap on RunRoot=${RUN_ROOT} -> ${OUTPUT_PATH}"

python tools/bata/analyze_duca_evidence_recovery.py \
  --run-root "${RUN_ROOT}" \
  --output "${OUTPUT_PATH}"

echo "[DONE] Statistical analysis completed."
