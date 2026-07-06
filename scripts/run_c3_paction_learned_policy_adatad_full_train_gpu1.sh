#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_PACTION_LEARNED_ADATAD][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN="${ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_paction_learned_adatad_full_train_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30310}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_PACTION_ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  fail "C3 mainline full train must use GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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

CONFIG="${CONFIG:-configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train_exec.py}"
CONFIG_VALIDATOR="${CONFIG_VALIDATOR:-tools/bata/validate_c3_paction_learned_adatad_full_train.py}"
LEDGER_VALIDATOR="${LEDGER_VALIDATOR:-tools/bata/validate_paction_learned_policy_ledger.py}"
POLICY_TRAINER="${POLICY_TRAINER:-tools/bata/train_paction_acquisition_policy.py}"
LEDGER_PIPELINE="${LEDGER_PIPELINE:-tools/bata/run_paction_learned_policy_ledger_pipeline.py}"

LEARNED_ROOT="${LEARNED_ROOT:-${BASE}/projects/c3_lowres_action_probe/paction_learned_adatad/${RUN_TAG}}"
POLICY_DIR="${POLICY_DIR:-${LEARNED_ROOT}/policy}"
LEDGER_ROOT="${LEDGER_ROOT:-${LEARNED_ROOT}/ledgers}"
VALIDATION_DIR="${VALIDATION_DIR:-${LEARNED_ROOT}/validation}"
RUN_DIR_ROOT="${RUN_DIR_ROOT:-${LEARNED_ROOT}/logs}"
WORK_DIR_ROOT="${WORK_DIR_ROOT:-exps/thumos/adatad/c3_paction_learned_ledger_original_adatad_full_train/${RUN_TAG}}"
PACTION_POLICY_CHECKPOINT="${PACTION_POLICY_CHECKPOINT:-${POLICY_DIR}/paction_policy.pth}"

C3_PACTION_SOURCE_ROOT="${C3_PACTION_SOURCE_ROOT:-}"
if [[ -n "${C3_PACTION_SOURCE_ROOT}" ]]; then
  C3_PACTION_TRAIN_SOURCE_JSONL="${C3_PACTION_TRAIN_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/train/samples.jsonl}"
  C3_PACTION_VAL_SOURCE_JSONL="${C3_PACTION_VAL_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/val/samples.jsonl}"
  C3_PACTION_TEST_SOURCE_JSONL="${C3_PACTION_TEST_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_PACTION_TRAIN_SOURCE_JSONL="${C3_PACTION_TRAIN_SOURCE_JSONL:-${LEARNED_ROOT}/source/train.samples.jsonl}"
  C3_PACTION_VAL_SOURCE_JSONL="${C3_PACTION_VAL_SOURCE_JSONL:-${LEARNED_ROOT}/source/val.samples.jsonl}"
  C3_PACTION_TEST_SOURCE_JSONL="${C3_PACTION_TEST_SOURCE_JSONL:-${LEARNED_ROOT}/source/test.samples.jsonl}"
fi

PACTION_POLICY_EPOCHS="${PACTION_POLICY_EPOCHS:-30}"
PACTION_POLICY_BATCH_SIZE="${PACTION_POLICY_BATCH_SIZE:-8}"
PACTION_POLICY_LR="${PACTION_POLICY_LR:-0.001}"
PACTION_POLICY_WEIGHT_DECAY="${PACTION_POLICY_WEIGHT_DECAY:-0.0001}"
PACTION_POLICY_HIDDEN_DIM="${PACTION_POLICY_HIDDEN_DIM:-64}"
PACTION_POLICY_NUM_LAYERS="${PACTION_POLICY_NUM_LAYERS:-3}"
PACTION_POLICY_DROPOUT="${PACTION_POLICY_DROPOUT:-0.10}"
PACTION_POLICY_GAP_MAX="${PACTION_POLICY_GAP_MAX:-3}"
PACTION_POLICY_BUDGET_CE_WEIGHT="${PACTION_POLICY_BUDGET_CE_WEIGHT:-0.25}"
PACTION_DYNAMIC_BUDGET_BUCKETS="${PACTION_DYNAMIC_BUDGET_BUCKETS:-128 192 256 320 384 512 768}"
PACTION_ADATAD_VARIANTS="${PACTION_ADATAD_VARIANTS:-learned_fixed_384 learned_fixed_768 learned_dynamic}"
REQUIRE_DYNAMIC_NONCONSTANT="${REQUIRE_DYNAMIC_NONCONSTANT:-1}"

MIN_BOUNDARY_SUPPORT="${MIN_BOUNDARY_SUPPORT:-}"
MIN_ACTION_COVERAGE="${MIN_ACTION_COVERAGE:-}"
MAX_MAX_GAP="${MAX_MAX_GAP:-}"
MAX_P95_GAP="${MAX_P95_GAP:-}"
MAX_UNSELECTED_HOLE="${MAX_UNSELECTED_HOLE:-}"
MAX_P95_UNSELECTED_HOLE="${MAX_P95_UNSELECTED_HOLE:-}"
MAX_UNIFORM_SIMILARITY="${MAX_UNIFORM_SIMILARITY:-}"

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
export C3_PACTION_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${CONFIG_VALIDATOR}"
require_file "${LEDGER_VALIDATOR}"
require_file "${POLICY_TRAINER}"
require_file "${LEDGER_PIPELINE}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${C3_PACTION_TRAIN_SOURCE_JSONL}"
require_file "${C3_PACTION_VAL_SOURCE_JSONL}"
require_file "${C3_PACTION_TEST_SOURCE_JSONL}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

mkdir -p "${POLICY_DIR}" "${LEDGER_ROOT}" "${VALIDATION_DIR}" "${RUN_DIR_ROOT}" "${WORK_DIR_ROOT}"

echo "[C3_PACTION_LEARNED_ADATAD] repo=${REPO_ROOT}"
echo "[C3_PACTION_LEARNED_ADATAD] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[C3_PACTION_LEARNED_ADATAD] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[C3_PACTION_LEARNED_ADATAD] slurm_step_gpus=${SLURM_STEP_GPUS:-none} slurm_job_gpus=${SLURM_JOB_GPUS:-none}"
echo "[C3_PACTION_LEARNED_ADATAD] precheck_only=${PRECHECK_ONLY} unlock=${ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN}"
echo "[C3_PACTION_LEARNED_ADATAD] learned_root=${LEARNED_ROOT}"
echo "[C3_PACTION_LEARNED_ADATAD] adatad_pretrain_path=${C3_PACTION_ADATAD_PRETRAIN_PATH}"

if [[ "${PRECHECK_ONLY}" != "1" && -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" && "${ALLOW_NON_SLURM_C3_PACTION_FULLTRAIN:-0}" != "1" ]]; then
  fail "formal full train must run inside a Slurm allocation/step; set PRECHECK_ONLY=1 for login-node checks"
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  "${POLICY_TRAINER}" \
  "${LEDGER_PIPELINE}" \
  tools/bata/apply_paction_acquisition_policy.py \
  tools/bata/paction_acquisition_policy.py \
  "${LEDGER_VALIDATOR}" \
  "${CONFIG_VALIDATOR}"

"${PYTHON}" "${POLICY_TRAINER}" \
  --train-jsonl "${C3_PACTION_TRAIN_SOURCE_JSONL}" \
  --out-dir "${POLICY_DIR}" \
  --checkpoint-path "${PACTION_POLICY_CHECKPOINT}" \
  --summary-json "${POLICY_DIR}/train.summary.json" \
  --epochs "${PACTION_POLICY_EPOCHS}" \
  --batch-size "${PACTION_POLICY_BATCH_SIZE}" \
  --lr "${PACTION_POLICY_LR}" \
  --weight-decay "${PACTION_POLICY_WEIGHT_DECAY}" \
  --hidden-dim "${PACTION_POLICY_HIDDEN_DIM}" \
  --num-layers "${PACTION_POLICY_NUM_LAYERS}" \
  --dropout "${PACTION_POLICY_DROPOUT}" \
  --gap-loss-max-gap "${PACTION_POLICY_GAP_MAX}" \
  --budget-ce-loss-weight "${PACTION_POLICY_BUDGET_CE_WEIGHT}" \
  --dynamic-budget-buckets ${PACTION_DYNAMIC_BUDGET_BUCKETS} \
  --expected-split training \
  --device cuda \
  --seed "${SEED}"

require_file "${PACTION_POLICY_CHECKPOINT}"
PACTION_POLICY_CHECKPOINT_SHA256="$("${PYTHON}" - "${PACTION_POLICY_CHECKPOINT}" <<'PY'
import hashlib
import sys
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
echo "[C3_PACTION_LEARNED_ADATAD] policy_checkpoint_sha256=${PACTION_POLICY_CHECKPOINT_SHA256}"

run_ledger_pipeline_for_split() {
  local split="$1"
  local input_jsonl="$2"
  local out_dir="${LEDGER_ROOT}/${split}"
  mkdir -p "${out_dir}"
  local args=(
    --input-jsonl "${input_jsonl}"
    --checkpoint-path "${PACTION_POLICY_CHECKPOINT}"
    --out-dir "${out_dir}"
    --summary-json "${out_dir}/pipeline.summary.json"
    --fixed-budgets 384 768
    --dynamic-target-len 768
    --dynamic-budget-buckets ${PACTION_DYNAMIC_BUDGET_BUCKETS}
    --device cuda
  )
  if [[ "${split}" != "test" ]]; then
    [[ -n "${MIN_BOUNDARY_SUPPORT}" ]] && args+=(--min-boundary-support "${MIN_BOUNDARY_SUPPORT}")
    [[ -n "${MIN_ACTION_COVERAGE}" ]] && args+=(--min-action-coverage "${MIN_ACTION_COVERAGE}")
    [[ "${REQUIRE_DYNAMIC_NONCONSTANT}" == "1" ]] && args+=(--require-dynamic-nonconstant-count)
  fi
  [[ -n "${MAX_MAX_GAP}" ]] && args+=(--max-max-gap "${MAX_MAX_GAP}")
  [[ -n "${MAX_P95_GAP}" ]] && args+=(--max-p95-gap "${MAX_P95_GAP}")
  [[ -n "${MAX_UNSELECTED_HOLE}" ]] && args+=(--max-unselected-hole "${MAX_UNSELECTED_HOLE}")
  [[ -n "${MAX_P95_UNSELECTED_HOLE}" ]] && args+=(--max-p95-unselected-hole "${MAX_P95_UNSELECTED_HOLE}")
  [[ -n "${MAX_UNIFORM_SIMILARITY}" ]] && args+=(--max-uniform-similarity "${MAX_UNIFORM_SIMILARITY}")
  "${PYTHON}" "${LEDGER_PIPELINE}" "${args[@]}"
}

run_ledger_pipeline_for_split train "${C3_PACTION_TRAIN_SOURCE_JSONL}"
run_ledger_pipeline_for_split val "${C3_PACTION_VAL_SOURCE_JSONL}"
run_ledger_pipeline_for_split test "${C3_PACTION_TEST_SOURCE_JSONL}"

ledger_path_for_variant() {
  local split="$1"
  local variant="$2"
  case "${variant}" in
    learned_fixed_384) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_learned_fixed_384.jsonl" ;;
    learned_fixed_768) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_learned_fixed_768.jsonl" ;;
    learned_dynamic) echo "${LEDGER_ROOT}/${split}/value_transport_ledger_learned_dynamic.jsonl" ;;
    *) fail "unknown p_action AdaTAD variant: ${variant}" ;;
  esac
}

sample_path_for_variant() {
  local split="$1"
  local variant="$2"
  case "${variant}" in
    learned_fixed_384) echo "${LEDGER_ROOT}/${split}/samples.learned_fixed_384.jsonl" ;;
    learned_fixed_768) echo "${LEDGER_ROOT}/${split}/samples.learned_fixed_768.jsonl" ;;
    learned_dynamic) echo "${LEDGER_ROOT}/${split}/samples.learned_dynamic.jsonl" ;;
    *) fail "unknown p_action AdaTAD variant: ${variant}" ;;
  esac
}

metric_sample_path_for_split() {
  case "$1" in
    train|val|test) echo "${LEDGER_ROOT}/$1/source.canonical_unique.jsonl" ;;
    *) fail "unknown split: $1" ;;
  esac
}

target_len_for_variant() {
  case "$1" in
    learned_fixed_384) echo 384 ;;
    learned_fixed_768|learned_dynamic) echo 768 ;;
    *) fail "unknown p_action AdaTAD variant: $1" ;;
  esac
}

strategy_for_variant() {
  case "$1" in
    learned_fixed_384|learned_fixed_768) echo learned_paction_gap_loss_value ;;
    learned_dynamic) echo learned_paction_gap_loss_dynamic_budget ;;
    *) fail "unknown p_action AdaTAD variant: $1" ;;
  esac
}

validate_variant_split() {
  local variant="$1"
  local split="$2"
  local sample_jsonl
  local ledger_jsonl
  local metric_jsonl
  local target_len
  local strategy
  sample_jsonl="$(sample_path_for_variant "${split}" "${variant}")"
  ledger_jsonl="$(ledger_path_for_variant "${split}" "${variant}")"
  metric_jsonl="$(metric_sample_path_for_split "${split}")"
  target_len="$(target_len_for_variant "${variant}")"
  strategy="$(strategy_for_variant "${variant}")"
  require_file "${sample_jsonl}"
  require_file "${ledger_jsonl}"
  require_file "${metric_jsonl}"
  local args=(
    --sample-jsonl "${sample_jsonl}"
    --metric-sample-jsonl "${metric_jsonl}"
    --ledger-jsonl "${ledger_jsonl}"
    --strategy "${strategy}"
    --expected-target-len "${target_len}"
    --allow-short-valid-ratio-count
    --require-deployable
    --require-policy-source learned_paction_gap_loss_policy_checkpoint
    --require-checkpoint-path "${PACTION_POLICY_CHECKPOINT}"
    --require-checkpoint-sha256 "${PACTION_POLICY_CHECKPOINT_SHA256}"
    --require-paction-provenance
    --summary-json "${VALIDATION_DIR}/${split}_${variant}.validation.json"
  )
  if [[ "${variant}" == "learned_fixed_384" ]]; then
    args+=(--require-selected-count 384)
  elif [[ "${variant}" == "learned_fixed_768" ]]; then
    args+=(--require-selected-count 768)
  fi
  if [[ "${variant}" == "learned_dynamic" && "${split}" != "test" && "${REQUIRE_DYNAMIC_NONCONSTANT}" == "1" ]]; then
    args+=(--require-nonconstant-selected-count)
  fi
  if [[ "${split}" != "test" ]]; then
    [[ -n "${MIN_BOUNDARY_SUPPORT}" ]] && args+=(--min-boundary-support "${MIN_BOUNDARY_SUPPORT}")
    [[ -n "${MIN_ACTION_COVERAGE}" ]] && args+=(--min-action-coverage "${MIN_ACTION_COVERAGE}")
  fi
  [[ -n "${MAX_MAX_GAP}" ]] && args+=(--max-max-gap "${MAX_MAX_GAP}")
  [[ -n "${MAX_P95_GAP}" ]] && args+=(--max-p95-gap "${MAX_P95_GAP}")
  [[ -n "${MAX_UNSELECTED_HOLE}" ]] && args+=(--max-unselected-hole "${MAX_UNSELECTED_HOLE}")
  [[ -n "${MAX_P95_UNSELECTED_HOLE}" ]] && args+=(--max-p95-unselected-hole "${MAX_P95_UNSELECTED_HOLE}")
  [[ -n "${MAX_UNIFORM_SIMILARITY}" ]] && args+=(--max-uniform-similarity "${MAX_UNIFORM_SIMILARITY}")
  "${PYTHON}" "${LEDGER_VALIDATOR}" "${args[@]}"
}

run_adatad_variant() {
  local variant="$1"
  local target_len
  target_len="$(target_len_for_variant "${variant}")"
  export C3_PACTION_LEDGER_VARIANT="${variant}"
  export C3_PACTION_LEARNED_LEDGER_ROOT="${LEDGER_ROOT}"
  export C3_PACTION_TRAIN_LEDGER_PATH
  export C3_PACTION_VAL_LEDGER_PATH
  export C3_PACTION_TEST_LEDGER_PATH
  C3_PACTION_TRAIN_LEDGER_PATH="$(ledger_path_for_variant train "${variant}")"
  C3_PACTION_VAL_LEDGER_PATH="$(ledger_path_for_variant val "${variant}")"
  C3_PACTION_TEST_LEDGER_PATH="$(ledger_path_for_variant test "${variant}")"
  export C3_PACTION_LEDGER_SOURCE="learned_paction_gap_loss_policy_checkpoint"
  export C3_PACTION_LEDGER_CONFIG_HASH="${PACTION_POLICY_CHECKPOINT_SHA256}"

  for split in train val test; do
    validate_variant_split "${variant}" "${split}"
  done

  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${CONFIG}" --require-ledger-files
  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${EXEC_CONFIG}" --require-ledger-files --allow-launch-unlocked

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    echo "[C3_PACTION_LEARNED_ADATAD] PRECHECK_ONLY variant=${variant} target_len=${target_len} complete"
    return 0
  fi

  if [[ "${ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN}" != "1" ]]; then
    fail "ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN=1 is required for formal full train"
  fi

  local run_dir="${RUN_DIR_ROOT}/${variant}"
  local work_dir="${WORK_DIR_ROOT}/${variant}"
  mkdir -p "${run_dir}" "${work_dir}"
  local variant_index=0
  case "${variant}" in
    learned_fixed_384) variant_index=0 ;;
    learned_fixed_768) variant_index=1 ;;
    learned_dynamic) variant_index=2 ;;
  esac
  echo "[C3_PACTION_LEARNED_ADATAD] train variant=${variant} target_len=${target_len} work_dir=${work_dir}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="$((MASTER_PORT_BASE + variant_index))" \
    tools/train.py \
    "${EXEC_CONFIG}" \
    --id "${RUN_ID}" \
    --seed "${SEED}" \
    --cfg-options "work_dir=${work_dir}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${run_dir}/train.out"
}

"${PYTHON}" -m pytest \
  tests/test_paction_acquisition_policy.py \
  tests/test_apply_paction_acquisition_policy.py \
  tests/test_paction_learned_ledger_pipeline.py \
  tests/test_train_paction_acquisition_policy.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  -q

for variant in ${PACTION_ADATAD_VARIANTS}; do
  run_adatad_variant "${variant}"
done

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_PACTION_LEARNED_ADATAD] PRECHECK_ONLY all variants complete"
fi
