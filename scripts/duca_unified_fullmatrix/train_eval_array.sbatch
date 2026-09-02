#!/bin/bash
#SBATCH -J duca_train_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-40%8
#SBATCH -t 7-00:00:00
#SBATCH -o /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
MATRIX_JSON="${PROJECT_DIR}/scripts/duca_unified_fullmatrix/matrix.json"
mkdir -p "${RUN_ROOT}/train_eval" /data/run01/sczc063/yuzibo/slurm_logs
cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

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
torchrun --standalone --nnodes=1 --nproc_per_node=1 tools/train.py "${CONFIG}" --seed "${SEED}" --id 0
python tools/test.py "${CONFIG}" --seed "${SEED}" --id 0 --checkpoint "${WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth" --cfg-options inference.load_from_raw_predictions=False inference.save_raw_prediction=True > "${RUN_ROOT}/train_eval/${TASK_ID}.test.log" 2>&1
