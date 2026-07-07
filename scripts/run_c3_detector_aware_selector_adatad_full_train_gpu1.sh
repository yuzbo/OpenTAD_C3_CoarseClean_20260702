#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_DETECTOR_AWARE_ADATAD][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN="${ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN:-0}"
ALLOW_C3_DETECTOR_AWARE_GPU0="${ALLOW_C3_DETECTOR_AWARE_GPU0:-0}"
REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY="${REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY:-1}"
ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC="${ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_detector_aware_stage2_adatad_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-}"
MASTER_PORT_LOW="${MASTER_PORT_LOW:-30000}"
MASTER_PORT_HIGH="${MASTER_PORT_HIGH:-60999}"
MASTER_PORT_MAX_ATTEMPTS="${MASTER_PORT_MAX_ATTEMPTS:-256}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_DETECTOR_AWARE_ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == "0" && "${ALLOW_C3_DETECTOR_AWARE_GPU0}" == "1" ]]; then
    echo "[C3_DETECTOR_AWARE_ADATAD] explicit GPU0 override accepted for Stage-2: CUDA_VISIBLE_DEVICES=0"
  else
    fail "Stage-2 detector-aware AdaTAD full train defaults to GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}. Set ALLOW_C3_DETECTOR_AWARE_GPU0=1 only when GPU0 is explicitly assigned to this route."
  fi
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

CONFIG="${CONFIG:-configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train_exec.py}"
CONFIG_VALIDATOR="${CONFIG_VALIDATOR:-tools/bata/validate_c3_detector_aware_adatad_full_train.py}"
TEACHER_UTILITY_EXPORTER="${TEACHER_UTILITY_EXPORTER:-tools/bata/detector_teacher_utility.py}"
TEACHER_POINTS_EXPORTER="${TEACHER_POINTS_EXPORTER:-tools/bata/export_dense_adatad_teacher_points.py}"
RESPONSIBILITY_UTILITY_EXPORTER="${RESPONSIBILITY_UTILITY_EXPORTER:-tools/bata/export_adatad_responsibility_utility.py}"
RESPONSIBILITY_UTILITY_VALIDATOR="${RESPONSIBILITY_UTILITY_VALIDATOR:-tools/bata/validate_adatad_responsibility_utility.py}"
POLICY_TRAINER="${POLICY_TRAINER:-tools/bata/train_detector_aware_acquisition_policy.py}"
POLICY_APPLIER="${POLICY_APPLIER:-tools/bata/apply_detector_aware_acquisition_policy.py}"
LEDGER_PIPELINE="${LEDGER_PIPELINE:-tools/bata/run_detector_aware_ledger_pipeline.py}"
LEDGER_VALIDATOR="${LEDGER_VALIDATOR:-tools/bata/validate_detector_aware_policy_ledger.py}"

ROUTE_ROOT="${ROUTE_ROOT:-${BASE}/projects/c3_lowres_action_probe/detector_aware_stage2/${RUN_TAG}}"
POLICY_DIR="${POLICY_DIR:-${ROUTE_ROOT}/policy}"
LEDGER_ROOT="${LEDGER_ROOT:-${ROUTE_ROOT}/ledgers}"
VALIDATION_DIR="${VALIDATION_DIR:-${ROUTE_ROOT}/validation}"
RUN_DIR_ROOT="${RUN_DIR_ROOT:-${ROUTE_ROOT}/logs}"
WORK_DIR_ROOT="${WORK_DIR_ROOT:-exps/thumos/adatad/c3_detector_aware_stage2_adatad_full_train/${RUN_TAG}}"
DETECTOR_AWARE_POLICY_CHECKPOINT="${DETECTOR_AWARE_POLICY_CHECKPOINT:-${POLICY_DIR}/detector_aware_policy.pth}"

C3_DETECTOR_AWARE_SOURCE_ROOT="${C3_DETECTOR_AWARE_SOURCE_ROOT:-}"
if [[ -n "${C3_DETECTOR_AWARE_SOURCE_ROOT}" ]]; then
  C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL="${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/train/samples.jsonl}"
  C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL="${C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/train/samples_with_teacher_utility.jsonl}"
  C3_DETECTOR_AWARE_VAL_SOURCE_JSONL="${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/val/samples.jsonl}"
  C3_DETECTOR_AWARE_TEST_SOURCE_JSONL="${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL="${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL:-${ROUTE_ROOT}/source/train.samples.jsonl}"
  C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL="${C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL:-${ROUTE_ROOT}/source/train.samples_with_teacher_utility.jsonl}"
  C3_DETECTOR_AWARE_VAL_SOURCE_JSONL="${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL:-${ROUTE_ROOT}/source/val.samples.jsonl}"
  C3_DETECTOR_AWARE_TEST_SOURCE_JSONL="${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL:-${ROUTE_ROOT}/source/test.samples.jsonl}"
fi
C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL="${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL:-}"
C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON="${C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON:-}"
C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL="${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL:-}"
C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON="${C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON:-}"
C3_DETECTOR_AWARE_RESPONSIBILITY_UTILITY_EXPORT_SUMMARY_JSON="${C3_DETECTOR_AWARE_RESPONSIBILITY_UTILITY_EXPORT_SUMMARY_JSON:-}"
C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH="${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH:-}"
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH="${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH:-}"
C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON="${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON:-}"
C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL="${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL:-}"

DETECTOR_AWARE_POLICY_EPOCHS="${DETECTOR_AWARE_POLICY_EPOCHS:-30}"
DETECTOR_AWARE_POLICY_BATCH_SIZE="${DETECTOR_AWARE_POLICY_BATCH_SIZE:-8}"
DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS="${DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS:-128 192 256 320 384 512 768}"
DETECTOR_AWARE_ADATAD_VARIANTS="${DETECTOR_AWARE_ADATAD_VARIANTS:-detector_aware_fixed_384}"
ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384="${ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384:-0}"
DETECTOR_AWARE_MAX_UNSELECTED_HOLE="${DETECTOR_AWARE_MAX_UNSELECTED_HOLE:-96}"
DETECTOR_AWARE_MAX_P95_UNSELECTED_HOLE="${DETECTOR_AWARE_MAX_P95_UNSELECTED_HOLE:-48}"
DETECTOR_AWARE_MAX_UNIFORM_SIMILARITY="${DETECTOR_AWARE_MAX_UNIFORM_SIMILARITY:-0.50}"

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
require_file "${ADATAD_PRETRAIN_PATH}"
export C3_DETECTOR_AWARE_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${CONFIG_VALIDATOR}"
require_file "${TEACHER_UTILITY_EXPORTER}"
require_file "${TEACHER_POINTS_EXPORTER}"
require_file "${RESPONSIBILITY_UTILITY_EXPORTER}"
require_file "${RESPONSIBILITY_UTILITY_VALIDATOR}"
require_file "${POLICY_TRAINER}"
require_file "${POLICY_APPLIER}"
require_file "${LEDGER_PIPELINE}"
require_file "${LEDGER_VALIDATOR}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL}"
require_file "${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL}"
if [[ -n "${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL}" && -n "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}" ]]; then
  fail "set only one of C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL or C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL"
fi
if [[ -z "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}" ]]; then
  if [[ "${REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY}" == "1" ]]; then
    fail "Stage-2 paper-main route requires C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL and C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON. Set REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY=0 and ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC=1 only for proposal-score surrogate diagnostics."
  fi
  if [[ "${ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC}" != "1" ]]; then
    fail "proposal-score surrogate Stage-2 is diagnostic-only; set ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC=1 explicitly"
  fi
fi
if [[ -n "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}" ]]; then
  require_file "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}"
  [[ -n "${C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON}" ]] || fail "C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON is required with responsibility points"
  require_file "${C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON}"
  require_file "${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL}"
elif [[ -n "${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL}" ]]; then
  require_file "${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL}"
  if [[ -z "${C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON}" ]]; then
    dense_points_manifest_candidate="${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL%.jsonl}.manifest.json"
    if [[ -f "${dense_points_manifest_candidate}" ]]; then
      C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON="${dense_points_manifest_candidate}"
    fi
  fi
  [[ -n "${C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON}" ]] || fail "C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON is required with dense teacher points"
  require_file "${C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON}"
  require_file "${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL}"
  [[ -n "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}" ]] || fail "C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH is required with dense teacher points"
  [[ -n "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}" ]] || fail "C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH is required with dense teacher points"
  require_file "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}"
  require_file "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}"
else
  require_file "${C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL}"
  [[ -n "${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON}" ]] || fail "C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON is required when C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL is not provided"
  require_file "${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON}"
fi

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

mkdir -p "${POLICY_DIR}" "${LEDGER_ROOT}" "${VALIDATION_DIR}" "${RUN_DIR_ROOT}" "${WORK_DIR_ROOT}"

pick_master_port() {
  local variant="$1"
  if [[ -n "${MASTER_PORT_BASE}" ]]; then
    local explicit_port="${MASTER_PORT_BASE}"
    MASTER_PORT_BASE="$((MASTER_PORT_BASE + 1))"
    echo "${explicit_port}"
    return 0
  fi
  "${PYTHON}" - "${RUN_TAG}" "${variant}" "${MASTER_PORT_LOW}" "${MASTER_PORT_HIGH}" "${MASTER_PORT_MAX_ATTEMPTS}" <<'PY'
import hashlib
import os
import socket
import sys

run_tag, variant = sys.argv[1], sys.argv[2]
low, high, max_attempts = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
if not (1024 <= low <= high <= 65535):
    raise SystemExit(f"invalid MASTER_PORT range: {low}-{high}")
span = high - low + 1
seed = "|".join(
    [
        run_tag,
        variant,
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("SLURM_STEP_ID", ""),
        str(os.getpid()),
    ]
)
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

if [[ "${PRECHECK_ONLY}" != "1" && -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" && "${ALLOW_NON_SLURM_C3_DETECTOR_AWARE_FULLTRAIN:-0}" != "1" ]]; then
  fail "formal full train must run inside a Slurm allocation/step; set PRECHECK_ONLY=1 for login-node checks"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  "${TEACHER_UTILITY_EXPORTER}" \
  "${TEACHER_POINTS_EXPORTER}" \
  "${RESPONSIBILITY_UTILITY_EXPORTER}" \
  "${RESPONSIBILITY_UTILITY_VALIDATOR}" \
  tools/bata/detector_aware_acquisition_policy.py \
  "${POLICY_TRAINER}" \
  "${POLICY_APPLIER}" \
  "${LEDGER_PIPELINE}" \
  "${LEDGER_VALIDATOR}" \
  "${CONFIG_VALIDATOR}"

TEACHER_UTILITY_SUMMARY_JSON="${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON}"
TEACHER_UTILITY_KIND="proposal_score_surrogate"
if [[ -n "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}" ]]; then
  C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL="${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL:-${POLICY_DIR}/samples_with_responsibility_utility.jsonl}"
  TEACHER_UTILITY_SUMMARY_JSON="${C3_DETECTOR_AWARE_RESPONSIBILITY_UTILITY_EXPORT_SUMMARY_JSON:-${POLICY_DIR}/responsibility_utility_export.summary.json}"
  TEACHER_UTILITY_KIND="point_responsibility"
  "${PYTHON}" "${RESPONSIBILITY_UTILITY_EXPORTER}" \
    --source-jsonl "${C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL}" \
    --base-samples-jsonl "${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL}" \
    --output-jsonl "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}" \
    --summary-json "${TEACHER_UTILITY_SUMMARY_JSON}" \
    --manifest-json "${C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON}"
elif [[ -n "${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL}" ]]; then
  C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL="${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL:-${POLICY_DIR}/samples_with_teacher_utility.jsonl}"
  TEACHER_UTILITY_SUMMARY_JSON="${TEACHER_UTILITY_SUMMARY_JSON:-${POLICY_DIR}/teacher_utility_export.summary.json}"
  teacher_utility_args=(
    --input-jsonl "${C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL}"
    --base-samples-jsonl "${C3_DETECTOR_AWARE_BASE_TRAIN_SOURCE_JSONL}"
    --output-jsonl "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}"
    --summary-json "${TEACHER_UTILITY_SUMMARY_JSON}"
    --teacher-checkpoint-path "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}"
    --teacher-config-path "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}"
    --generator-manifest-json "${C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON}"
    --expected-split training
  )
  "${PYTHON}" "${TEACHER_UTILITY_EXPORTER}" \
    "${teacher_utility_args[@]}"
else
  C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL="${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL:-${C3_DETECTOR_AWARE_TRAIN_SOURCE_JSONL}}"
fi
require_file "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}"
require_file "${TEACHER_UTILITY_SUMMARY_JSON}"
"${PYTHON}" - "${TEACHER_UTILITY_KIND}" "${TEACHER_UTILITY_SUMMARY_JSON}" "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}" <<'PY'
import json
import sys

kind, summary_json, output_jsonl = sys.argv[1:4]
if kind == "point_responsibility":
    from tools.bata.validate_adatad_responsibility_utility import validate_responsibility_utility_export
    evidence = validate_responsibility_utility_export(summary_json, output_jsonl=output_jsonl)
    payload = {
        "decision": evidence["decision"],
        "row_count": evidence["row_count"],
        "utility_semantics": evidence["utility_semantics"],
        "utility_source_type": evidence["utility_source_type"],
        "output_jsonl_sha256": evidence["output_jsonl_sha256"],
    }
else:
    from tools.bata.detector_teacher_utility import validate_teacher_utility_export_evidence
    evidence = validate_teacher_utility_export_evidence(
        summary_json,
        output_jsonl=output_jsonl,
        require_paction=True,
    )
    payload = {
        "decision": evidence["decision"],
        "row_count": evidence["row_count"],
        "teacher_checkpoint_sha256": evidence["teacher_checkpoint_sha256"],
        "output_jsonl_sha256": evidence["output_jsonl_sha256"],
    }
print(json.dumps({
    **payload,
}, sort_keys=True), flush=True)
PY

policy_train_args=(
  --train-jsonl "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}"
  --out-dir "${POLICY_DIR}"
  --checkpoint-path "${DETECTOR_AWARE_POLICY_CHECKPOINT}"
  --summary-json "${POLICY_DIR}/train.summary.json"
  --epochs "${DETECTOR_AWARE_POLICY_EPOCHS}"
  --batch-size "${DETECTOR_AWARE_POLICY_BATCH_SIZE}"
  --dynamic-budget-buckets ${DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS}
  --expected-split training
  --allow-gt-diagnostics-in-training-source
  --allow-teacher-utility-training-artifact
  --device cuda
  --seed "${SEED}"
)
if [[ "${REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY}" == "1" ]]; then
  policy_train_args+=(--require-point-responsibility-utility)
fi
"${PYTHON}" "${POLICY_TRAINER}" "${policy_train_args[@]}"

require_file "${DETECTOR_AWARE_POLICY_CHECKPOINT}"
DETECTOR_AWARE_POLICY_CHECKPOINT_SHA256="$("${PYTHON}" - "${DETECTOR_AWARE_POLICY_CHECKPOINT}" <<'PY'
import hashlib
import sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
export C3_DETECTOR_AWARE_LEDGER_SOURCE="learned_detector_aware_policy_checkpoint"
export C3_DETECTOR_AWARE_LEDGER_CONFIG_HASH="${DETECTOR_AWARE_POLICY_CHECKPOINT_SHA256}"

run_ledger_pipeline_for_split() {
  local split="$1"
  local input_jsonl="$2"
  local out_dir="${LEDGER_ROOT}/${split}"
  mkdir -p "${out_dir}"
  local pipeline_args=(
    --input-jsonl "${input_jsonl}"
    --checkpoint-path "${DETECTOR_AWARE_POLICY_CHECKPOINT}"
    --out-dir "${out_dir}"
    --summary-json "${out_dir}/pipeline.summary.json"
    --fixed-budgets 384 768
    --dynamic-target-len 768
    --dynamic-budget-buckets ${DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS}
    --max-unselected-hole "${DETECTOR_AWARE_MAX_UNSELECTED_HOLE}"
    --max-p95-unselected-hole "${DETECTOR_AWARE_MAX_P95_UNSELECTED_HOLE}"
    --max-uniform-similarity "${DETECTOR_AWARE_MAX_UNIFORM_SIMILARITY}"
    --allow-inferred-paction-positive-provenance
    --device cuda
  )
  if [[ "${REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY}" == "1" ]]; then
    pipeline_args+=(--require-point-responsibility-utility)
  fi
  "${PYTHON}" "${LEDGER_PIPELINE}" "${pipeline_args[@]}"
}

run_ledger_pipeline_for_split train "${C3_DETECTOR_AWARE_POLICY_TRAIN_JSONL}"
run_ledger_pipeline_for_split val "${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL}"
run_ledger_pipeline_for_split test "${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL}"

ledger_path_for_variant() {
  local split="$1"
  local variant="$2"
  case "${variant}" in
    detector_aware_fixed_384) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_detector_aware_fixed_384.jsonl" ;;
    detector_aware_fixed_768) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_detector_aware_fixed_768.jsonl" ;;
    detector_aware_dynamic) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_detector_aware_dynamic.jsonl" ;;
    *) fail "unknown detector-aware AdaTAD variant: ${variant}" ;;
  esac
}

target_len_for_variant() {
  case "$1" in
    detector_aware_fixed_384) echo 384 ;;
    detector_aware_fixed_768|detector_aware_dynamic) echo 768 ;;
    *) fail "unknown detector-aware AdaTAD variant: $1" ;;
  esac
}

strategy_for_variant() {
  case "$1" in
    detector_aware_fixed_384) echo detector_aware_fixed_384 ;;
    detector_aware_fixed_768) echo detector_aware_fixed_768 ;;
    detector_aware_dynamic) echo detector_aware_dynamic ;;
    *) fail "unknown detector-aware AdaTAD variant: $1" ;;
  esac
}

assert_variant_claim_scope() {
  local variant="$1"
  if [[ "${variant}" != "detector_aware_fixed_384" && "${ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384}" != "1" ]]; then
    fail "variant=${variant} exceeds the <=384 main-claim budget. Set ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384=1 only for explicit diagnostic runs."
  fi
}

validate_variant_split() {
  local variant="$1"
  local split="$2"
  local ledger_jsonl
  local target_len
  local strategy
  ledger_jsonl="$(ledger_path_for_variant "${split}" "${variant}")"
  target_len="$(target_len_for_variant "${variant}")"
  strategy="$(strategy_for_variant "${variant}")"
  require_file "${ledger_jsonl}"
  local args=(
    --sample-jsonl "${LEDGER_ROOT}/${split}/samples.detector_aware_all.jsonl"
    --metric-sample-jsonl "${LEDGER_ROOT}/${split}/source.canonical_unique.jsonl"
    --ledger-jsonl "${ledger_jsonl}"
    --strategy "${strategy}"
    --expected-target-len "${target_len}"
    --require-deployable
    --require-policy-source learned_detector_aware_policy_checkpoint
    --require-checkpoint-path "${DETECTOR_AWARE_POLICY_CHECKPOINT}"
    --require-checkpoint-sha256 "${DETECTOR_AWARE_POLICY_CHECKPOINT_SHA256}"
    --require-paction-provenance
    --summary-json "${VALIDATION_DIR}/${split}_${variant}.validation.json"
    --max-unselected-hole "${DETECTOR_AWARE_MAX_UNSELECTED_HOLE}"
    --max-p95-unselected-hole "${DETECTOR_AWARE_MAX_P95_UNSELECTED_HOLE}"
    --max-uniform-similarity "${DETECTOR_AWARE_MAX_UNIFORM_SIMILARITY}"
  )
  if [[ "${variant}" == "detector_aware_fixed_384" ]]; then
    args+=(--require-selected-count 384)
  elif [[ "${variant}" == "detector_aware_fixed_768" ]]; then
    args+=(--require-selected-count 768)
  elif [[ "${variant}" == "detector_aware_dynamic" ]]; then
    args+=(--require-nonconstant-selected-count)
  fi
  "${PYTHON}" "${LEDGER_VALIDATOR}" "${args[@]}"
}

run_adatad_variant() {
  local variant="$1"
  export C3_DETECTOR_AWARE_LEDGER_VARIANT="${variant}"
  export C3_DETECTOR_AWARE_LEDGER_ROOT="${LEDGER_ROOT}"
  export C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH
  export C3_DETECTOR_AWARE_VAL_LEDGER_PATH
  export C3_DETECTOR_AWARE_TEST_LEDGER_PATH
  C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH="$(ledger_path_for_variant train "${variant}")"
  C3_DETECTOR_AWARE_VAL_LEDGER_PATH="$(ledger_path_for_variant val "${variant}")"
  C3_DETECTOR_AWARE_TEST_LEDGER_PATH="$(ledger_path_for_variant test "${variant}")"

  for split in train val test; do
    validate_variant_split "${variant}" "${split}"
  done

  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${CONFIG}" --require-ledger-files
  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${EXEC_CONFIG}" --require-ledger-files --allow-launch-unlocked

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    echo "[C3_DETECTOR_AWARE_ADATAD] PRECHECK_ONLY variant=${variant} complete; detector mAP not run"
    return 0
  fi
  if [[ "${ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN}" != "1" ]]; then
    fail "ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN=1 is required for formal full train"
  fi

  local run_dir="${RUN_DIR_ROOT}/${variant}"
  local work_dir="${WORK_DIR_ROOT}/${variant}"
  local master_port
  mkdir -p "${run_dir}" "${work_dir}"
  master_port="$(pick_master_port "${variant}")"
  echo "[C3_DETECTOR_AWARE_ADATAD] train variant=${variant} work_dir=${work_dir} master_port=${master_port}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${master_port}" \
    tools/train.py \
    "${EXEC_CONFIG}" \
    --id "${RUN_ID}" \
    --seed "${SEED}" \
    --cfg-options "work_dir=${work_dir}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${run_dir}/train.out"
}

"${PYTHON}" -m pytest \
  tests/test_detector_teacher_utility.py \
  tests/test_detector_aware_acquisition_policy.py \
  tests/test_detector_aware_ledger_pipeline.py \
  tests/test_c3_detector_aware_adatad_full_train.py \
  -q

for variant in ${DETECTOR_AWARE_ADATAD_VARIANTS}; do
  assert_variant_claim_scope "${variant}"
  run_adatad_variant "${variant}"
done

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_DETECTOR_AWARE_ADATAD] PRECHECK_ONLY all variants complete; full detector mAP still required for claims"
fi
