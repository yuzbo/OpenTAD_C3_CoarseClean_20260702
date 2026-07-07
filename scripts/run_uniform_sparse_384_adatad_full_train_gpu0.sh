#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_UNIFORM_SPARSE_ADATAD][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN="${ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_uniform_sparse_384_adatad_gpu0_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_LOW="${MASTER_PORT_LOW:-30000}"
MASTER_PORT_HIGH="${MASTER_PORT_HIGH:-60999}"
MASTER_PORT_MAX_ATTEMPTS="${MASTER_PORT_MAX_ATTEMPTS:-256}"
UNIFORM_SPARSE_TARGET_LEN="${UNIFORM_SPARSE_TARGET_LEN:-384}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_UNIFORM_SPARSE_ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  fail "uniform sparse AdaTAD baseline must use GPU0; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
if [[ "${UNIFORM_SPARSE_TARGET_LEN}" != "384" ]]; then
  fail "this launcher is the exact uniform_384 baseline; got UNIFORM_SPARSE_TARGET_LEN=${UNIFORM_SPARSE_TARGET_LEN}"
fi

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-${BASE}/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-${BASE}/raw/Test Data/TH14_test_set_mp4}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

CONFIG="${CONFIG:-configs/adatad/thumos/c3_uniform_sparse_384_ledger_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_uniform_sparse_384_ledger_adatad_full_train_exec.py}"
UNIFORM_LEDGER_GENERATOR="${UNIFORM_LEDGER_GENERATOR:-tools/bata/generate_uniform_sparse_ledger.py}"

UNIFORM_ROOT="${UNIFORM_ROOT:-${BASE}/projects/c3_lowres_action_probe/uniform_sparse_384_adatad/${RUN_TAG}}"
LEDGER_ROOT="${LEDGER_ROOT:-${UNIFORM_ROOT}/ledgers}"
RUN_DIR="${RUN_DIR:-${UNIFORM_ROOT}/logs}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/c3_uniform_sparse_384_ledger_original_adatad_full_train/${RUN_TAG}}"
VALIDATION_DIR="${VALIDATION_DIR:-${UNIFORM_ROOT}/validation}"

C3_UNIFORM_SPARSE_SOURCE_ROOT="${C3_UNIFORM_SPARSE_SOURCE_ROOT:-${C3_PACTION_SOURCE_ROOT:-}}"
if [[ -n "${C3_UNIFORM_SPARSE_SOURCE_ROOT}" ]]; then
  C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL="${C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL:-${C3_UNIFORM_SPARSE_SOURCE_ROOT}/train/samples.jsonl}"
  C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL="${C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL:-${C3_UNIFORM_SPARSE_SOURCE_ROOT}/val/samples.jsonl}"
  C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL="${C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL:-${C3_UNIFORM_SPARSE_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL="${C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL:-}"
  C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL="${C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL:-}"
  C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL="${C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL:-}"
fi

[[ -n "${C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL}" ]] || fail "C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL or C3_UNIFORM_SPARSE_SOURCE_ROOT is required"
[[ -n "${C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL}" ]] || fail "C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL or C3_UNIFORM_SPARSE_SOURCE_ROOT is required"
[[ -n "${C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL}" ]] || fail "C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL or C3_UNIFORM_SPARSE_SOURCE_ROOT is required"

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
export C3_UNIFORM_SPARSE_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"
export C3_UNIFORM_SPARSE_LEDGER_SOURCE="uniform_exact_sparse_384"

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${UNIFORM_LEDGER_GENERATOR}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${ADATAD_PRETRAIN_PATH}"
require_file "${C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL}"
require_file "${C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL}"
require_file "${C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL}"

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

mkdir -p "${LEDGER_ROOT}" "${RUN_DIR}" "${WORK_DIR}" "${VALIDATION_DIR}"

echo "[C3_UNIFORM_SPARSE_ADATAD] repo=${REPO_ROOT}"
echo "[C3_UNIFORM_SPARSE_ADATAD] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[C3_UNIFORM_SPARSE_ADATAD] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[C3_UNIFORM_SPARSE_ADATAD] slurm_step_gpus=${SLURM_STEP_GPUS:-none} slurm_job_gpus=${SLURM_JOB_GPUS:-none}"
echo "[C3_UNIFORM_SPARSE_ADATAD] precheck_only=${PRECHECK_ONLY} unlock=${ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN}"
echo "[C3_UNIFORM_SPARSE_ADATAD] uniform_root=${UNIFORM_ROOT}"
echo "[C3_UNIFORM_SPARSE_ADATAD] adatad_pretrain_path=${C3_UNIFORM_SPARSE_ADATAD_PRETRAIN_PATH}"

if [[ "${PRECHECK_ONLY}" != "1" && -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" && "${ALLOW_NON_SLURM_C3_UNIFORM_SPARSE_FULLTRAIN:-0}" != "1" ]]; then
  fail "formal full train must run inside a Slurm allocation/step; set PRECHECK_ONLY=1 for login-node checks"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile tools/train.py "${UNIFORM_LEDGER_GENERATOR}"
"${PYTHON}" -m pytest tests/test_generate_uniform_sparse_ledger.py tests/test_uniform_sparse_adatad_full_train.py -q

C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH="$("${PYTHON}" - "${UNIFORM_LEDGER_GENERATOR}" "${UNIFORM_SPARSE_TARGET_LEN}" <<'PY'
import hashlib
import json
import sys
path, target_len = sys.argv[1:3]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    digest.update(handle.read())
digest.update(json.dumps({"source": "uniform_exact_sparse_384", "target_len": int(target_len)}, sort_keys=True).encode("utf-8"))
print(digest.hexdigest())
PY
)"
export C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH
echo "[C3_UNIFORM_SPARSE_ADATAD] ledger_config_hash=${C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH}"

ledger_path_for_split() {
  local split="$1"
  echo "${LEDGER_ROOT}/${split}/value_transport_ledger_uniform_sparse_384.jsonl"
}

summary_path_for_split() {
  local split="$1"
  echo "${LEDGER_ROOT}/${split}/uniform_sparse_384.summary.json"
}

source_path_for_split() {
  case "$1" in
    train) echo "${C3_UNIFORM_SPARSE_TRAIN_SOURCE_JSONL}" ;;
    val) echo "${C3_UNIFORM_SPARSE_VAL_SOURCE_JSONL}" ;;
    test) echo "${C3_UNIFORM_SPARSE_TEST_SOURCE_JSONL}" ;;
    *) fail "unknown split: $1" ;;
  esac
}

add_uniform_deploy_metadata() {
  local ledger_jsonl="$1"
  local summary_json="$2"
  "${PYTHON}" - "${ledger_jsonl}" "${summary_json}" "${C3_UNIFORM_SPARSE_LEDGER_SOURCE}" "${C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
source = sys.argv[3]
config_hash = sys.argv[4]
rows = []
for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    row = json.loads(line)
    row["diagnostic_only"] = False
    row["policy_source"] = source
    row["policy_checkpoint_sha256"] = config_hash
    row["uses_uniform_scaffold"] = True
    row["uses_uniform_fill"] = False
    row["uses_gt"] = False
    row["uses_teacher"] = False
    row["uses_oracle"] = False
    row["uses_cache"] = False
    row["uses_prediction_cache"] = False
    row["uses_raw_prediction"] = False
    row["uses_checkpoint"] = False
    row["prediction_uses_gt"] = False
    row["training_only"] = False
    row["deploy_selection_ledger"] = True
    if row.get("selection_family") != "uniform_exact":
        raise ValueError(f"{path}:{line_no}: expected selection_family=uniform_exact")
    if int(row.get("target_len", -1)) != 384 or int(row.get("selected_count", -1)) != 384:
        raise ValueError(f"{path}:{line_no}: expected exact 384 selected positions")
    rows.append(row)
if not rows:
    raise ValueError(f"{path}: no ledger rows")
path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["ledger_sha256"] = digest.hexdigest()
summary["policy_source"] = source
summary["policy_checkpoint_sha256"] = config_hash
summary["uses_uniform_fill"] = False
summary["uses_oracle"] = False
summary["uses_cache"] = False
summary["uses_raw_prediction"] = False
summary["uses_checkpoint"] = False
summary["prediction_uses_gt"] = False
summary["training_only"] = False
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

validate_uniform_ledger() {
  local ledger_jsonl="$1"
  "${PYTHON}" - "${ledger_jsonl}" "${C3_UNIFORM_SPARSE_LEDGER_SOURCE}" "${C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = sys.argv[2]
config_hash = sys.argv[3]
seen = set()
rows = 0
for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    row = json.loads(line)
    rows += 1
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or "|" not in sample_id:
        raise ValueError(f"{path}:{line_no}: invalid sample_id")
    if sample_id in seen:
        raise ValueError(f"{path}:{line_no}: duplicate sample_id={sample_id}")
    seen.add(sample_id)
    if row.get("deploy_selection_ledger") is not True or row.get("diagnostic_only") is True:
        raise ValueError(f"{path}:{line_no}: not a deployable ledger row")
    if row.get("policy_source") != source or row.get("policy_checkpoint_sha256") != config_hash:
        raise ValueError(f"{path}:{line_no}: source/hash mismatch")
    for key in ("uses_gt", "uses_teacher", "uses_oracle", "uses_cache", "uses_prediction_cache", "uses_raw_prediction", "uses_checkpoint", "prediction_uses_gt", "training_only"):
        if row.get(key) is True:
            raise ValueError(f"{path}:{line_no}: forbidden flag {key}=true")
    if row.get("selection_family") != "uniform_exact" or row.get("uses_uniform_scaffold") is not True or row.get("uses_uniform_fill") is True:
        raise ValueError(f"{path}:{line_no}: wrong uniform provenance")
    positions = [int(item) for item in row.get("selected_positions", [])]
    if int(row.get("target_len", -1)) != 384 or int(row.get("selected_count", -1)) != 384 or len(positions) != 384:
        raise ValueError(f"{path}:{line_no}: not exact uniform_384")
    if positions != sorted(set(positions)):
        raise ValueError(f"{path}:{line_no}: positions must be sorted unique")
    if int(row.get("dense_len", -1)) != 768:
        raise ValueError(f"{path}:{line_no}: dense_len must be 768 for matched AdaTAD baseline")
    valid_len = int(row.get("valid_len", -1))
    if valid_len < 384 or positions[0] < 0 or positions[-1] >= valid_len:
        raise ValueError(f"{path}:{line_no}: invalid valid_len/positions")
if rows <= 0:
    raise ValueError(f"{path}: no rows")
PY
}

generate_ledger_for_split() {
  local split="$1"
  local source_jsonl
  local ledger_jsonl
  local summary_json
  source_jsonl="$(source_path_for_split "${split}")"
  ledger_jsonl="$(ledger_path_for_split "${split}")"
  summary_json="$(summary_path_for_split "${split}")"
  mkdir -p "$(dirname "${ledger_jsonl}")"
  "${PYTHON}" "${UNIFORM_LEDGER_GENERATOR}" \
    --input-jsonl "${source_jsonl}" \
    --output-jsonl "${ledger_jsonl}" \
    --summary-json "${summary_json}" \
    --target-len "${UNIFORM_SPARSE_TARGET_LEN}"
  add_uniform_deploy_metadata "${ledger_jsonl}" "${summary_json}"
  validate_uniform_ledger "${ledger_jsonl}"
}

for split in train val test; do
  generate_ledger_for_split "${split}"
done

export C3_UNIFORM_SPARSE_LEDGER_ROOT="${LEDGER_ROOT}"
export C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH
export C3_UNIFORM_SPARSE_VAL_LEDGER_PATH
export C3_UNIFORM_SPARSE_TEST_LEDGER_PATH
C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH="$(ledger_path_for_split train)"
C3_UNIFORM_SPARSE_VAL_LEDGER_PATH="$(ledger_path_for_split val)"
C3_UNIFORM_SPARSE_TEST_LEDGER_PATH="$(ledger_path_for_split test)"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_UNIFORM_SPARSE_ADATAD] PRECHECK_ONLY variant=uniform_sparse_384 target_len=${UNIFORM_SPARSE_TARGET_LEN} complete"
  exit 0
fi

if [[ "${ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN}" != "1" ]]; then
  fail "ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN=1 is required for formal full train"
fi

MASTER_PORT="$(pick_master_port uniform_sparse_384)"
echo "[C3_UNIFORM_SPARSE_ADATAD] train variant=uniform_sparse_384 master_port=${MASTER_PORT} work_dir=${WORK_DIR}"
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${EXEC_CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"
