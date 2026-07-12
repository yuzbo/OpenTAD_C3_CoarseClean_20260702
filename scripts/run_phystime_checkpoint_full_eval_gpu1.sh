#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime checkpoint diagnostic] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${PHYSTIME_TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
CONFIG="${PHYSTIME_CONFIG:?PHYSTIME_CONFIG is required}"
CHECKPOINT="${PHYSTIME_DIAGNOSTIC_CHECKPOINT:?PHYSTIME_DIAGNOSTIC_CHECKPOINT is required}"
PRETRAIN="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
OUT_DIR="${PHYSTIME_DIAGNOSTIC_OUT_DIR:?PHYSTIME_DIAGNOSTIC_OUT_DIR is required}"
SEED="${PHYSTIME_SEED:-42}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "checkpoint diagnostics must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
[[ -x "${PYTHON}" ]] || fail "Python environment not found: ${PYTHON}"
[[ -x "${TORCHRUN}" ]] || fail "torchrun not found: ${TORCHRUN}"
[[ -f "${CONFIG}" ]] || fail "config not found: ${CONFIG}"
[[ -f "${CHECKPOINT}" ]] || fail "checkpoint not found: ${CHECKPOINT}"
[[ -f "${PRETRAIN}" ]] || fail "VideoMAE checkpoint not found: ${PRETRAIN}"
: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

MASTER_PORT="$((24000 + (10#${SLURM_JOB_ID} % 16000)))"
START_TIME="$(date +%s)"
set +e
"${TORCHRUN}" --nnodes=1 --nproc_per_node=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT}" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" --seed "${SEED}" --id 0 \
  --cfg-options \
    "model.backbone.custom.pretrain=${PRETRAIN}" \
    "work_dir=${OUT_DIR}/work_dir" \
    "post_processing.save_dict=True" \
  2>&1 | tee "${OUT_DIR}/test.out"
STATUS="${PIPESTATUS[0]}"
set -e
END_TIME="$(date +%s)"

RESULT="${OUT_DIR}/work_dir/gpu1_id0/result_detection.json"
"${PYTHON}" - "${OUT_DIR}" "${STATUS}" "$((END_TIME - START_TIME))" "${RESULT}" <<'PY'
import json
import sys
from pathlib import Path

out_dir, status, wall_time, result = sys.argv[1:]
payload = {
    "schema_version": "phystime_checkpoint_full_eval_v1",
    "exit_code": int(status),
    "wall_time_sec": int(wall_time),
    "result_detection": str(Path(result).resolve()),
    "result_exists": Path(result).is_file(),
}
Path(out_dir, "runtime_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

[[ "${STATUS}" == "0" ]] || fail "tools/test.py failed with exit code ${STATUS}"
[[ -f "${RESULT}" ]] || fail "result_detection.json was not produced"
touch "${OUT_DIR}/CHECKPOINT_DIAGNOSTIC_COMPLETE"
echo "[PhysTime checkpoint diagnostic] complete result=${RESULT}"
