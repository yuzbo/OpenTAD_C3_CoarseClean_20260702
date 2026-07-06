#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_GAS_VT_ADATAD][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_GAS_VT_ADATAD_FULLTRAIN="${ALLOW_C3_GAS_VT_ADATAD_FULLTRAIN:-0}"
ALLOW_C3_GAS_VT_GPU0="${ALLOW_C3_GAS_VT_GPU0:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_gas_vt_adatad_full_train_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30410}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_GAS_VT_ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == "0" && "${ALLOW_C3_GAS_VT_GPU0}" == "1" ]]; then
    echo "[C3_GAS_VT_ADATAD] explicit GPU0 override accepted for Stage-0/1: CUDA_VISIBLE_DEVICES=0"
  else
    fail "C3 mainline GAS-VT full train defaults to GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}. Set ALLOW_C3_GAS_VT_GPU0=1 only after explicitly stopping the GPU0 model zoo."
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

CONFIG="${CONFIG:-configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train_exec.py}"
CONFIG_VALIDATOR="${CONFIG_VALIDATOR:-tools/bata/validate_c3_paction_learned_adatad_full_train.py}"
LEDGER_VALIDATOR="${LEDGER_VALIDATOR:-tools/bata/validate_paction_learned_policy_ledger.py}"
POLICY_TRAINER="${POLICY_TRAINER:-tools/bata/train_gap_aware_acquisition_policy.py}"
POLICY_APPLIER="${POLICY_APPLIER:-tools/bata/apply_gap_aware_acquisition_policy.py}"
LEDGER_PIPELINE="${LEDGER_PIPELINE:-tools/bata/run_gap_aware_ledger_pipeline.py}"

GAS_VT_ROOT="${GAS_VT_ROOT:-${BASE}/projects/c3_lowres_action_probe/gas_vt_adatad/${RUN_TAG}}"
POLICY_DIR="${POLICY_DIR:-${GAS_VT_ROOT}/policy}"
LEDGER_ROOT="${LEDGER_ROOT:-${GAS_VT_ROOT}/ledgers}"
VALIDATION_DIR="${VALIDATION_DIR:-${GAS_VT_ROOT}/validation}"
RUN_DIR_ROOT="${RUN_DIR_ROOT:-${GAS_VT_ROOT}/logs}"
WORK_DIR_ROOT="${WORK_DIR_ROOT:-exps/thumos/adatad/c3_gas_vt_ledger_original_adatad_full_train/${RUN_TAG}}"
GAS_VT_POLICY_CHECKPOINT="${GAS_VT_POLICY_CHECKPOINT:-${POLICY_DIR}/gas_vt_policy.pth}"

C3_GAS_VT_SOURCE_ROOT="${C3_GAS_VT_SOURCE_ROOT:-}"
if [[ -n "${C3_GAS_VT_SOURCE_ROOT}" ]]; then
  C3_GAS_VT_TRAIN_SOURCE_JSONL="${C3_GAS_VT_TRAIN_SOURCE_JSONL:-${C3_GAS_VT_SOURCE_ROOT}/train/samples.jsonl}"
  C3_GAS_VT_VAL_SOURCE_JSONL="${C3_GAS_VT_VAL_SOURCE_JSONL:-${C3_GAS_VT_SOURCE_ROOT}/val/samples.jsonl}"
  C3_GAS_VT_TEST_SOURCE_JSONL="${C3_GAS_VT_TEST_SOURCE_JSONL:-${C3_GAS_VT_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_GAS_VT_TRAIN_SOURCE_JSONL="${C3_GAS_VT_TRAIN_SOURCE_JSONL:-${GAS_VT_ROOT}/source/train.samples.jsonl}"
  C3_GAS_VT_VAL_SOURCE_JSONL="${C3_GAS_VT_VAL_SOURCE_JSONL:-${GAS_VT_ROOT}/source/val.samples.jsonl}"
  C3_GAS_VT_TEST_SOURCE_JSONL="${C3_GAS_VT_TEST_SOURCE_JSONL:-${GAS_VT_ROOT}/source/test.samples.jsonl}"
fi

GAS_VT_POLICY_EPOCHS="${GAS_VT_POLICY_EPOCHS:-30}"
GAS_VT_POLICY_BATCH_SIZE="${GAS_VT_POLICY_BATCH_SIZE:-8}"
GAS_VT_DYNAMIC_BUDGET_BUCKETS="${GAS_VT_DYNAMIC_BUDGET_BUCKETS:-128 192 256 320 384 512 768}"
GAS_VT_ADATAD_VARIANTS="${GAS_VT_ADATAD_VARIANTS:-gas_vt_fixed_384 gas_vt_fixed_768 gas_vt_dynamic}"
GAS_VT_MAX_UNSELECTED_HOLE="${GAS_VT_MAX_UNSELECTED_HOLE:-96}"
GAS_VT_MAX_P95_UNSELECTED_HOLE="${GAS_VT_MAX_P95_UNSELECTED_HOLE:-48}"
GAS_VT_MAX_UNIFORM_SIMILARITY="${GAS_VT_MAX_UNIFORM_SIMILARITY:-0.50}"
C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD="${C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD:-0}"

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
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "required file missing: ${ADATAD_PRETRAIN_PATH}"
export C3_GAS_VT_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${CONFIG_VALIDATOR}"
require_file "${LEDGER_VALIDATOR}"
require_file "${POLICY_TRAINER}"
require_file "${POLICY_APPLIER}"
require_file "${LEDGER_PIPELINE}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${C3_GAS_VT_TRAIN_SOURCE_JSONL}"
require_file "${C3_GAS_VT_VAL_SOURCE_JSONL}"
require_file "${C3_GAS_VT_TEST_SOURCE_JSONL}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

mkdir -p "${POLICY_DIR}" "${LEDGER_ROOT}" "${VALIDATION_DIR}" "${RUN_DIR_ROOT}" "${WORK_DIR_ROOT}"

if [[ "${PRECHECK_ONLY}" != "1" && -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" && "${ALLOW_NON_SLURM_C3_GAS_VT_FULLTRAIN:-0}" != "1" ]]; then
  fail "formal full train must run inside a Slurm allocation/step; set PRECHECK_ONLY=1 for login-node checks"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  tools/bata/gas_vt_paction_policy.py \
  "${POLICY_TRAINER}" \
  "${POLICY_APPLIER}" \
  "${LEDGER_PIPELINE}" \
  "${LEDGER_VALIDATOR}" \
  "${CONFIG_VALIDATOR}"

if [[ "${C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD}" == "1" && -f "${GAS_VT_POLICY_CHECKPOINT}" && -f "${POLICY_DIR}/train.summary.json" ]]; then
  echo "[C3_GAS_VT_ADATAD] reusing existing GAS-VT policy checkpoint: ${GAS_VT_POLICY_CHECKPOINT}"
else
  "${PYTHON}" "${POLICY_TRAINER}" \
    --train-jsonl "${C3_GAS_VT_TRAIN_SOURCE_JSONL}" \
    --out-dir "${POLICY_DIR}" \
    --checkpoint-path "${GAS_VT_POLICY_CHECKPOINT}" \
    --summary-json "${POLICY_DIR}/train.summary.json" \
    --epochs "${GAS_VT_POLICY_EPOCHS}" \
    --batch-size "${GAS_VT_POLICY_BATCH_SIZE}" \
    --dynamic-budget-buckets ${GAS_VT_DYNAMIC_BUDGET_BUCKETS} \
    --expected-split training \
    --allow-missing-split-from-source-path \
    --allow-gt-diagnostics-in-training-source \
    --device cuda \
    --seed "${SEED}"
fi

require_file "${GAS_VT_POLICY_CHECKPOINT}"
GAS_VT_POLICY_CHECKPOINT_SHA256="$("${PYTHON}" - "${GAS_VT_POLICY_CHECKPOINT}" <<'PY'
import hashlib
import sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
export C3_GAS_VT_LEDGER_SOURCE="learned_paction_gas_vt_policy_checkpoint"
export C3_GAS_VT_LEDGER_CONFIG_HASH="${GAS_VT_POLICY_CHECKPOINT_SHA256}"

run_ledger_pipeline_for_split() {
  local split="$1"
  local input_jsonl="$2"
  local out_dir="${LEDGER_ROOT}/${split}"
  mkdir -p "${out_dir}"
  "${PYTHON}" "${LEDGER_PIPELINE}" \
    --input-jsonl "${input_jsonl}" \
    --checkpoint-path "${GAS_VT_POLICY_CHECKPOINT}" \
    --out-dir "${out_dir}" \
    --summary-json "${out_dir}/pipeline.summary.json" \
    --fixed-budgets 384 768 \
    --dynamic-target-len 768 \
    --dynamic-budget-buckets ${GAS_VT_DYNAMIC_BUDGET_BUCKETS} \
    --max-unselected-hole "${GAS_VT_MAX_UNSELECTED_HOLE}" \
    --device cuda
}

maybe_run_ledger_pipeline_for_split() {
  local split="$1"
  local input_jsonl="$2"
  local out_dir="${LEDGER_ROOT}/${split}"
  if [[ "${C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD}" == "1" \
      && -f "${out_dir}/pipeline.summary.json" \
      && -f "${out_dir}/samples.gas_vt_all.jsonl" \
      && -f "${out_dir}/value_transport_ledger_gas_vt_fixed_384.jsonl" \
      && -f "${out_dir}/value_transport_ledger_gas_vt_fixed_768.jsonl" \
      && -f "${out_dir}/value_transport_ledger_gas_vt_dynamic.jsonl" ]]; then
    echo "[C3_GAS_VT_ADATAD] reusing existing GAS-VT ledgers for split=${split}: ${out_dir}"
    return 0
  fi
  run_ledger_pipeline_for_split "${split}" "${input_jsonl}"
}

maybe_run_ledger_pipeline_for_split train "${C3_GAS_VT_TRAIN_SOURCE_JSONL}"
maybe_run_ledger_pipeline_for_split val "${C3_GAS_VT_VAL_SOURCE_JSONL}"
maybe_run_ledger_pipeline_for_split test "${C3_GAS_VT_TEST_SOURCE_JSONL}"

ledger_path_for_variant() {
  local split="$1"
  local variant="$2"
  case "${variant}" in
    gas_vt_fixed_384) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_gas_vt_fixed_384.jsonl" ;;
    gas_vt_fixed_768) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_gas_vt_fixed_768.jsonl" ;;
    gas_vt_dynamic) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_gas_vt_dynamic.jsonl" ;;
    *) fail "unknown GAS-VT AdaTAD variant: ${variant}" ;;
  esac
}

target_len_for_variant() {
  case "$1" in
    gas_vt_fixed_384) echo 384 ;;
    gas_vt_fixed_768|gas_vt_dynamic) echo 768 ;;
    *) fail "unknown GAS-VT AdaTAD variant: $1" ;;
  esac
}

strategy_for_variant() {
  case "$1" in
    gas_vt_fixed_384) echo gas_vt_fixed_384 ;;
    gas_vt_fixed_768) echo gas_vt_fixed_768 ;;
    gas_vt_dynamic) echo gas_vt_dynamic ;;
    *) fail "unknown GAS-VT AdaTAD variant: $1" ;;
  esac
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
    --sample-jsonl "${LEDGER_ROOT}/${split}/samples.gas_vt_all.jsonl"
    --metric-sample-jsonl "${LEDGER_ROOT}/${split}/source.canonical_unique.jsonl"
    --ledger-jsonl "${ledger_jsonl}"
    --strategy "${strategy}"
    --expected-target-len "${target_len}"
    --allow-short-valid-ratio-count
    --require-deployable
    --require-policy-source learned_paction_gas_vt_policy_checkpoint
    --require-checkpoint-path "${GAS_VT_POLICY_CHECKPOINT}"
    --require-checkpoint-sha256 "${GAS_VT_POLICY_CHECKPOINT_SHA256}"
    --require-paction-provenance
    --summary-json "${VALIDATION_DIR}/${split}_${variant}.validation.json"
    --max-unselected-hole "${GAS_VT_MAX_UNSELECTED_HOLE}"
    --max-p95-unselected-hole "${GAS_VT_MAX_P95_UNSELECTED_HOLE}"
    --max-uniform-similarity "${GAS_VT_MAX_UNIFORM_SIMILARITY}"
  )
  if [[ "${variant}" == "gas_vt_fixed_384" ]]; then
    args+=(--require-selected-count 384)
  elif [[ "${variant}" == "gas_vt_fixed_768" ]]; then
    args+=(--require-selected-count 768)
  elif [[ "${variant}" == "gas_vt_dynamic" ]]; then
    args+=(--require-nonconstant-selected-count)
  fi
  "${PYTHON}" "${LEDGER_VALIDATOR}" "${args[@]}"
}

run_adatad_variant() {
  local variant="$1"
  export C3_GAS_VT_LEDGER_VARIANT="${variant}"
  export C3_GAS_VT_LEDGER_ROOT="${LEDGER_ROOT}"
  export C3_GAS_VT_TRAIN_LEDGER_PATH
  export C3_GAS_VT_VAL_LEDGER_PATH
  export C3_GAS_VT_TEST_LEDGER_PATH
  C3_GAS_VT_TRAIN_LEDGER_PATH="$(ledger_path_for_variant train "${variant}")"
  C3_GAS_VT_VAL_LEDGER_PATH="$(ledger_path_for_variant val "${variant}")"
  C3_GAS_VT_TEST_LEDGER_PATH="$(ledger_path_for_variant test "${variant}")"

  for split in train val test; do
    validate_variant_split "${variant}" "${split}"
  done

  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${CONFIG}" --require-ledger-files
  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${EXEC_CONFIG}" --require-ledger-files --allow-launch-unlocked

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    echo "[C3_GAS_VT_ADATAD] PRECHECK_ONLY variant=${variant} complete"
    return 0
  fi
  if [[ "${ALLOW_C3_GAS_VT_ADATAD_FULLTRAIN}" != "1" ]]; then
    fail "ALLOW_C3_GAS_VT_ADATAD_FULLTRAIN=1 is required for formal full train"
  fi

  local run_dir="${RUN_DIR_ROOT}/${variant}"
  local work_dir="${WORK_DIR_ROOT}/${variant}"
  mkdir -p "${run_dir}" "${work_dir}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT_BASE}" \
    tools/train.py \
    "${EXEC_CONFIG}" \
    --id "${RUN_ID}" \
    --seed "${SEED}" \
    --cfg-options "work_dir=${work_dir}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${run_dir}/train.out"
}

"${PYTHON}" -m pytest \
  tests/test_gas_vt_paction_policy.py \
  tests/test_gap_aware_acquisition_policy_apply.py \
  tests/test_gap_aware_ledger_pipeline.py \
  tests/test_c3_gas_vt_adatad_full_train.py \
  -q

for variant in ${GAS_VT_ADATAD_VARIANTS}; do
  run_adatad_variant "${variant}"
done

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_GAS_VT_ADATAD] PRECHECK_ONLY all variants complete"
fi
