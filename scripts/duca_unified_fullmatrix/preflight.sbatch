#!/bin/bash
#SBATCH -J duca_preflight
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${RUN_ROOT:-${BASE}/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
mkdir -p "${RUN_ROOT}" "${BASE}/slurm_logs"
cd "${PROJECT_DIR}"

source /etc/profile >/dev/null 2>&1 || true
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"

python -m py_compile tools/train.py tools/test.py tools/bata/duca_runtime_contract.py tools/bata/generate_duca_unified_fullmatrix.py tools/bata/aggregate_duca_unified_fullmatrix.py tools/bata/bootstrap_duca_unified_fullmatrix.py tools/bata/audit_duca_unified_fullmatrix_slurm.py
python tools/bata/generate_duca_unified_fullmatrix.py --check
python -m pytest tests/test_duca_unified_phase.py tests/test_duca_unified_physical_time.py tests/test_duca_unified_attribution.py tests/test_duca_unified_mod.py tests/test_duca_unified_curriculum.py -q
