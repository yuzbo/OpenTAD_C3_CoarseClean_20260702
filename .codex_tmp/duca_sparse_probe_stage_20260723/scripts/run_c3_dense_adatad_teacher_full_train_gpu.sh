#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_DENSE_ADATAD_TEACHER][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_DENSE_TEACHER_FULLTRAIN="${ALLOW_C3_DENSE_TEACHER_FULLTRAIN:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_dense_adatad_teacher_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_LOW="${MASTER_PORT_LOW:-30000}"
MASTER_PORT_HIGH="${MASTER_PORT_HIGH:-60999}"
MASTER_PORT_MAX_ATTEMPTS="${MASTER_PORT_MAX_ATTEMPTS:-256}"
CONFIG="${CONFIG:-configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_DENSE_TEACHER_ADATAD_PRETRAIN_PATH:-${BASE}/retrained/${ADATAD_PRETRAIN_FILENAME}}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "${CUDA_VISIBLE_DEVICES}" in
  0|1) ;;
  *) fail "CUDA_VISIBLE_DEVICES must be a single physical GPU id 0 or 1; got ${CUDA_VISIBLE_DEVICES}" ;;
esac

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-${BASE}/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-${BASE}/raw/Test Data/TH14_test_set_mp4}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

pick_master_port() {
  local label="$1"
  if [[ -n "${MASTER_PORT}" ]]; then
    echo "${MASTER_PORT}"
    return 0
  fi
  "${PYTHON}" - "${RUN_TAG}" "${label}" "${MASTER_PORT_LOW}" "${MASTER_PORT_HIGH}" "${MASTER_PORT_MAX_ATTEMPTS}" <<'PY'
import hashlib
import os
import socket
import sys

run_tag, label = sys.argv[1], sys.argv[2]
low, high, max_attempts = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
if not (1024 <= low <= high <= 65535):
    raise SystemExit(f"invalid MASTER_PORT range: {low}-{high}")
span = high - low + 1
seed = "|".join([run_tag, label, os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", ""), str(os.getpid())])
start = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % span
for offset in range(min(max_attempts, span)):
    port = low + ((start + offset) % span)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit(f"no free MASTER_PORT found in {low}-{high} after {max_attempts} attempts")
PY
}

resolve_path() {
  local raw="$1"
  if [[ "${raw}" == /* ]]; then
    echo "${raw}"
  elif [[ -f "${REPO_ROOT}/${raw}" ]]; then
    readlink -f "${REPO_ROOT}/${raw}"
  elif [[ -f "${BASE}/${raw}" ]]; then
    readlink -f "${BASE}/${raw}"
  else
    echo "${REPO_ROOT}/${raw}"
  fi
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

ADATAD_PRETRAIN_PATH="$(resolve_path "${ADATAD_PRETRAIN_PATH}")"
if [[ ! -f "${ADATAD_PRETRAIN_PATH}" && -f "${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}" ]]; then
  ADATAD_PRETRAIN_PATH="$(readlink -f "${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}")"
fi
if [[ ! -f "${ADATAD_PRETRAIN_PATH}" && -f "${REPO_ROOT}/pretrained/${ADATAD_PRETRAIN_FILENAME}" ]]; then
  ADATAD_PRETRAIN_PATH="$(readlink -f "${REPO_ROOT}/pretrained/${ADATAD_PRETRAIN_FILENAME}")"
fi
export C3_DENSE_TEACHER_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"
require_file "${CONFIG}"
require_file "${ADATAD_PRETRAIN_PATH}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
[[ -d "${THUMOS14_TRAIN_DATA_PATH}" ]] || fail "train video directory missing: ${THUMOS14_TRAIN_DATA_PATH}"
[[ -d "${THUMOS14_TEST_DATA_PATH}" ]] || fail "test video directory missing: ${THUMOS14_TEST_DATA_PATH}"

DENSE_TEACHER_ROOT="${DENSE_TEACHER_ROOT:-${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-${DENSE_TEACHER_ROOT}/work_dir}"
mkdir -p "${DENSE_TEACHER_ROOT}" "${WORK_DIR}"
LOG_FILE="${LOG_FILE:-${DENSE_TEACHER_ROOT}/train.out}"
PRECHECK_JSON="${DENSE_TEACHER_ROOT}/dense_teacher_precheck.json"

if [[ "${PRECHECK_ONLY}" != "1" ]]; then
  [[ "${ALLOW_C3_DENSE_TEACHER_FULLTRAIN}" == "1" ]] || fail "set ALLOW_C3_DENSE_TEACHER_FULLTRAIN=1 for full dense teacher training"
  [[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" ]] || fail "full dense teacher training must run inside a Slurm allocation/step"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py

"${PYTHON}" - "${CONFIG}" "${ADATAD_PRETRAIN_PATH}" "${PRECHECK_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.train_schedule import should_eval_epoch

config_path = Path(sys.argv[1])
pretrain = Path(sys.argv[2])
out_path = Path(sys.argv[3])
cfg = Config.fromfile(str(config_path))

assert "frame_selector" not in repr(cfg.model), "dense teacher must not contain sparse selector"
assert int(cfg.dataset.train.pipeline[2]["trunc_len"]) == 768
assert int(cfg.dataset.val.window_size) == 768
assert int(cfg.dataset.test.window_size) == 768
assert int(cfg.model.projection.max_seq_len) == 768
assert int(cfg.model.backbone.backbone.total_frames) == 768
assert int(cfg.workflow.val_start_epoch) == 9
assert int(cfg.workflow.val_eval_interval) == 10
assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
assert int(cfg.workflow.checkpoint_interval) == 10
assert int(cfg.workflow.end_epoch) == 60
assert cfg.dataset.train.ann_file == os.environ["THUMOS14_ANNOTATION_PATH"]
assert cfg.dataset.val.ann_file == os.environ["THUMOS14_ANNOTATION_PATH"]
assert cfg.dataset.test.ann_file == os.environ["THUMOS14_ANNOTATION_PATH"]
assert cfg.dataset.train.class_map == os.environ["THUMOS14_CLASS_MAP"]
assert cfg.dataset.val.class_map == os.environ["THUMOS14_CLASS_MAP"]
assert cfg.dataset.test.class_map == os.environ["THUMOS14_CLASS_MAP"]
assert cfg.dataset.train.data_path == os.environ["THUMOS14_TRAIN_DATA_PATH"]
assert cfg.dataset.val.data_path == os.environ["THUMOS14_TEST_DATA_PATH"]
assert cfg.dataset.test.data_path == os.environ["THUMOS14_TEST_DATA_PATH"]
assert cfg.evaluation.ground_truth_filename == os.environ["THUMOS14_ANNOTATION_PATH"]
eval_epochs = [
    epoch
    for epoch in range(int(cfg.workflow.end_epoch))
    if epoch >= int(cfg.workflow.val_start_epoch) and should_eval_epoch(epoch, cfg.workflow)
]
assert eval_epochs == [9, 19, 29, 39, 49, 59], eval_epochs
assert pretrain.is_file(), f"pretrain missing: {pretrain}"
payload = {
    "decision": "C3_DENSE_ADATAD_TEACHER_PRECHECK_PASS",
    "config": str(config_path),
    "pretrain": str(pretrain),
    "eval_epochs_zero_based": eval_epochs,
    "checkpoint_epochs_zero_based": [9, 19, 29, 39, 49, 59],
    "annotation_path": cfg.dataset.train.ann_file,
    "evaluation_ground_truth_filename": cfg.evaluation.ground_truth_filename,
    "class_map": cfg.dataset.train.class_map,
    "train_data_path": cfg.dataset.train.data_path,
    "test_data_path": cfg.dataset.test.data_path,
    "dense_teacher_axis": "dense_768_frame_axis",
    "selector_free_dense_teacher": True,
    "full_train_requires_slurm": True,
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_DENSE_ADATAD_TEACHER] PRECHECK_ONLY=1 complete: ${PRECHECK_JSON}"
  exit 0
fi

echo "[C3_DENSE_ADATAD_TEACHER] RUN_TAG=${RUN_TAG}"
echo "[C3_DENSE_ADATAD_TEACHER] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[C3_DENSE_ADATAD_TEACHER] WORK_DIR=${WORK_DIR}"
echo "[C3_DENSE_ADATAD_TEACHER] LOG_FILE=${LOG_FILE}"
MASTER_PORT="$(pick_master_port dense_teacher)"
echo "[C3_DENSE_ADATAD_TEACHER] master_port=${MASTER_PORT}"

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" \
  --seed "${SEED}" \
  --id "${RUN_ID}" \
  --cfg-options \
    "work_dir=${WORK_DIR}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${LOG_FILE}"
