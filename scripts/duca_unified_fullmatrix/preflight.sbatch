#!/bin/bash
#SBATCH -J duca_preflight
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
mkdir -p "${RUN_ROOT}" /data/run01/sczc063/yuzibo/slurm_logs
cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python -m py_compile tools/train.py tools/test.py tools/bata/generate_duca_unified_fullmatrix.py tools/bata/aggregate_duca_unified_fullmatrix.py tools/bata/bootstrap_duca_unified_fullmatrix.py
python tools/bata/generate_duca_unified_fullmatrix.py --check
python -m pytest tests/test_duca_unified_phase.py tests/test_duca_unified_physical_time.py tests/test_duca_unified_attribution.py tests/test_duca_unified_mod.py tests/test_duca_unified_curriculum.py -q
