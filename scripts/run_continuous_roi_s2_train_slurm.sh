#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[CONTINUOUS_ROI_S2_TRAIN][FAIL] %s\n' "$*" >&2
  exit 2
}

require_control_free_value() {
  local name="$1"
  local value="$2"
  local character
  local ordinal
  local index
  [[ -n "${value}" ]] || fail "${name} must not be empty"
  [[ "${value}" != [[:space:]]* && "${value}" != *[[:space:]] ]] || \
    fail "${name} contains leading or trailing whitespace"
  [[ "${value}" != *,* ]] || fail "${name} contains a comma"
  for ((index = 0; index < ${#value}; index++)); do
    character="${value:index:1}"
    printf -v ordinal '%d' "'${character}"
    if ((ordinal < 32 || ordinal == 127)); then
      fail "${name} contains an ASCII control character"
    fi
  done
}

ROOT="${CONTINUOUS_ROI_S2_SOURCE_ROOT:?set CONTINUOUS_ROI_S2_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${CONTINUOUS_ROI_S2_RUN_ROOT:?set CONTINUOUS_ROI_S2_RUN_ROOT}"
MANIFEST="${CONTINUOUS_ROI_S2_MANIFEST:?set CONTINUOUS_ROI_S2_MANIFEST}"
ANNOTATION="${CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION:?set CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION}"
CLASS_MAP="${CONTINUOUS_ROI_S2_CLASS_MAP:?set CONTINUOUS_ROI_S2_CLASS_MAP}"
VIDEO_ROOT="${CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT:?set CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT}"
PRETRAINED="${CONTINUOUS_ROI_S2_PRETRAINED:?set CONTINUOUS_ROI_S2_PRETRAINED}"
FULL_MODEL_GATE="${CONTINUOUS_ROI_S2_FULL_MODEL_GATE:?set CONTINUOUS_ROI_S2_FULL_MODEL_GATE}"
TRAINING_RUNTIME_PRECHECK="${CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK:?set CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK}"
RUNTIME_AUTHORIZATION="${CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION:?set CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION}"
EXPECTED_COMMIT="${CONTINUOUS_ROI_S2_EXPECTED_COMMIT:?set CONTINUOUS_ROI_S2_EXPECTED_COMMIT}"
FAMILY="${CONTINUOUS_ROI_S2_FAMILY:?set CONTINUOUS_ROI_S2_FAMILY}"
SEED="${CONTINUOUS_ROI_S2_SEED:?set CONTINUOUS_ROI_S2_SEED}"

for variable_name in \
  ROOT BASE RUN_ROOT MANIFEST ANNOTATION CLASS_MAP VIDEO_ROOT PRETRAINED \
  FULL_MODEL_GATE TRAINING_RUNTIME_PRECHECK RUNTIME_AUTHORIZATION \
  EXPECTED_COMMIT FAMILY SEED; do
  require_control_free_value "${variable_name}" "${!variable_name}"
done
[[ "${BASE}" == "/data/run01/sczc063/yuzibo" ]] || \
  fail "YUZIBO_ROOT must be /data/run01/sczc063/yuzibo"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal training requires Slurm"
if [[ "${CONTINUOUS_ROI_S2_SINGLE_GPU_STEP:-0}" != "1" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a JOB_GPU_ARRAY <<< "${SLURM_JOB_GPUS:-}"
  if [[ "${#JOB_GPU_ARRAY[@]}" -gt 1 ]]; then
    export CONTINUOUS_ROI_S2_SINGLE_GPU_STEP=1
    exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
      bash "${ROOT}/scripts/run_continuous_roi_s2_train_slurm.sh"
  fi
fi

[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "formal training requires one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "inner step must expose one GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "inner step must expose five CPUs"
case "${FAMILY}" in
  D160|G96|U128) ;;
  *) fail "family must be D160, G96, or U128" ;;
esac
case "${SEED}" in
  3407|3408|3409) ;;
  *) fail "seed must be 3407, 3408, or 3409" ;;
esac
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail "run root must stay inside the remote write boundary" ;;
esac
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be full"
[[ -d "${ROOT}/.git" ]] || fail "source root is not a Git checkout"
for path in \
  "${MANIFEST}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" \
  "${FULL_MODEL_GATE}" "${TRAINING_RUNTIME_PRECHECK}" \
  "${RUNTIME_AUTHORIZATION}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "development video root does not exist"

cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"
[[ -f "${CONDA_ACTIVATE}" ]] || fail "Conda activation script is missing: ${CONDA_ACTIVATE}"
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"

python -c 'from tools.bata.spatial_zoom_s1_training import require_slurm_memory_limit_mb; print(require_slurm_memory_limit_mb(minimum_mb=90000))'
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__; print(numpy.__file__)'
python -c "from tools.bata.continuous_roi_s2_training import require_clean_git_checkout; require_clean_git_checkout(expected_commit='${EXPECTED_COMMIT}')"

FAMILY_LOWER="$(printf '%s' "${FAMILY}" | tr '[:upper:]' '[:lower:]')"
SOURCE_CONFIG="${ROOT}/configs/adatad/thumos/continuous_roi_s2_${FAMILY_LOWER}_videomae_s_768x1_adapter.py"
CONTROL_DIR="${RUN_ROOT}/control"
WORK_DIR="${RUN_ROOT}/${FAMILY_LOWER}/seed${SEED}"
BOUND_CONFIG="${CONTROL_DIR}/${FAMILY_LOWER}_seed${SEED}.py"
COMPLETION="${WORK_DIR}/training_completion.json"
CHECKPOINT="${WORK_DIR}/checkpoint/epoch_59.pth"
[[ -f "${SOURCE_CONFIG}" ]] || fail "canonical source config is missing"
[[ ! -e "${WORK_DIR}" ]] || fail "formal work directory already exists"
[[ ! -e "${BOUND_CONFIG}" ]] || fail "bound config already exists"
mkdir -p "${CONTROL_DIR}"

python tools/bata/build_continuous_roi_s2_training_config.py \
  "${SOURCE_CONFIG}" \
  --family "${FAMILY}" \
  --seed "${SEED}" \
  --work-dir "${WORK_DIR}" \
  --manifest "${MANIFEST}" \
  --development-annotation "${ANNOTATION}" \
  --class-map "${CLASS_MAP}" \
  --development-video-root "${VIDEO_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --full-model-gate "${FULL_MODEL_GATE}" \
  --training-runtime-precheck "${TRAINING_RUNTIME_PRECHECK}" \
  --runtime-authorization "${RUNTIME_AUTHORIZATION}" \
  --output "${BOUND_CONFIG}"

torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="continuous-roi-s2-${SLURM_JOB_ID}-${FAMILY_LOWER}-${SEED}" \
  tools/train.py "${BOUND_CONFIG}" --seed "${SEED}" --id 0

python tools/bata/finalize_continuous_roi_s2_training.py \
  --config "${BOUND_CONFIG}" \
  --seed "${SEED}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${COMPLETION}"

printf '[CONTINUOUS_ROI_S2_TRAIN] PASS family=%s seed=%s completion=%s\n' \
  "${FAMILY}" "${SEED}" "${COMPLETION}"
