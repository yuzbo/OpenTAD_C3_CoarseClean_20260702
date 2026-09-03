#!/bin/bash
#SBATCH -J duca_train_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-40%8
#SBATCH -t 7-00:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${RUN_ROOT:-${BASE}/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
MATRIX_JSON="${PROJECT_DIR}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${RUN_ROOT}/train_eval" "${RUN_ROOT}/runs" "${BASE}/slurm_logs"
cd "${PROJECT_DIR}"

source /etc/profile >/dev/null 2>&1 || true
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"

read -r CONFIG SEED TASK_ID WORK_DIR <<EOF_ROW
$(python - "$MATRIX_JSON" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
row = payload["rows"][int(sys.argv[2])]
print(row["config_path"], row["seed"], row["task_id"], row["work_dir"])
PY
)
EOF_ROW

echo "DUCA train/eval task=${TASK_ID} seed=${SEED} config=${CONFIG}"
TASK_WORK_DIR="${RUN_ROOT}/runs/${TASK_ID}"
if [[ -e "${TASK_WORK_DIR}" ]] && [[ -n "$(find "${TASK_WORK_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to reuse non-empty DUCA task work_dir: ${TASK_WORK_DIR}" >&2
  exit 2
fi
mkdir -p "${TASK_WORK_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node=1 tools/train.py "${CONFIG}" --seed "${SEED}" --id 0 --cfg-options work_dir="${TASK_WORK_DIR}"
python tools/test.py "${CONFIG}" --seed "${SEED}" --id 0 --checkpoint "${TASK_WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth" --cfg-options work_dir="${TASK_WORK_DIR}" inference.load_from_raw_predictions=False inference.save_raw_prediction=True > "${RUN_ROOT}/train_eval/${TASK_ID}.test.log" 2>&1
