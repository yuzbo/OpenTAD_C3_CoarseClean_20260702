#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[CONTINUOUS_ROI_S2_GATE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${CONTINUOUS_ROI_S2_SOURCE_ROOT:?set CONTINUOUS_ROI_S2_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
OUT_ROOT="${CONTINUOUS_ROI_S2_GATE_ROOT:?set CONTINUOUS_ROI_S2_GATE_ROOT}"
EXPECTED_COMMIT="${CONTINUOUS_ROI_S2_EXPECTED_COMMIT:?set CONTINUOUS_ROI_S2_EXPECTED_COMMIT}"
CHECKPOINT="${CONTINUOUS_ROI_S2_PRETRAINED:?set CONTINUOUS_ROI_S2_PRETRAINED}"
MANIFEST="${CONTINUOUS_ROI_S2_MANIFEST:?set CONTINUOUS_ROI_S2_MANIFEST}"
ANNOTATION="${CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION:?set CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION}"
CLASS_MAP="${CONTINUOUS_ROI_S2_CLASS_MAP:?set CONTINUOUS_ROI_S2_CLASS_MAP}"
VIDEO_ROOT="${CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT:?set CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT}"
CONFIG="${ROOT}/configs/adatad/thumos/continuous_roi_s2_u128_videomae_s_768x1_adapter.py"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the formal Gate requires Slurm"
[[ -d "${ROOT}/.git" ]] || fail "bound source root is not a Git checkout"
if [[ "${CONTINUOUS_ROI_S2_SINGLE_GPU_STEP:-0}" != "1" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a JOB_GPU_ARRAY <<< "${SLURM_JOB_GPUS:-}"
  if [[ "${#JOB_GPU_ARRAY[@]}" -gt 1 ]]; then
    export CONTINUOUS_ROI_S2_SINGLE_GPU_STEP=1
    exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
      bash "${ROOT}/scripts/run_continuous_roi_s2_cuda_gate_slurm.sh"
  fi
fi

[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "the exact Gate step requires one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "the exact Gate step requires one GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "the exact Gate step requires five CPUs"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be full"
for path in "${CHECKPOINT}" "${MANIFEST}" "${ANNOTATION}" "${CLASS_MAP}"; do
  [[ -f "${path}" ]] || fail "required Gate file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "development video root does not exist"
case "${OUT_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail "Gate output must stay inside the remote write boundary" ;;
esac
[[ ! -e "${OUT_ROOT}" ]] || fail "Gate namespace already exists and is immutable"

ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || \
  fail "snapshot commit ${ACTUAL_COMMIT} != expected ${EXPECTED_COMMIT}"
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "formal Gate snapshot is not completely clean"

cd "${ROOT}"
mkdir -p "${OUT_ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

python -c 'from tools.bata.spatial_zoom_s1_training import require_slurm_memory_limit_mb, require_slurm_single_gpu_allocation; require_slurm_single_gpu_allocation(); print(require_slurm_memory_limit_mb(minimum_mb=90000))'
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__; print(numpy.__file__)'

python -m py_compile \
  tools/train.py \
  opentad/models/backbones/continuous_roi_geometry.py \
  opentad/models/backbones/continuous_roi_sampler.py \
  opentad/models/backbones/continuous_roi_wrapper.py \
  tools/bata/build_continuous_roi_s2_training_config.py \
  tools/bata/build_continuous_roi_s2_runtime_gate_config.py \
  tools/bata/authorize_continuous_roi_s2_training_runtime.py \
  tools/bata/continuous_roi_s2_contract.py \
  tools/bata/continuous_roi_s2_training.py \
  tools/bata/deploy_continuous_roi_s2_training_matrix.py \
  tools/bata/finalize_continuous_roi_s2_training.py \
  tools/bata/finalize_continuous_roi_s2_runtime_gate.py \
  tools/bata/continuous_roi_s2_runtime_gate.py \
  tools/bata/precheck_continuous_roi_s2_training_runtime.py \
  tools/bata/validate_continuous_roi_s2_implementation.py \
  tools/bata/run_continuous_roi_s2_one_step_gate.py

python -m pytest -p no:cacheprovider \
  tests/test_continuous_roi_s2_protocol.py \
  tests/test_continuous_roi_s2_implementation_static.py \
  tests/test_continuous_roi_geometry_sampler.py \
  tests/test_continuous_roi_representation.py \
  tests/test_continuous_roi_source_views.py \
  tests/test_continuous_roi_s2_one_step_gate.py \
  tests/test_continuous_roi_s2_training.py \
  tests/test_continuous_roi_s2_deployment_posix.py \
  tests/test_train_engine_max_train_iters.py \
  -q

python tools/bata/run_continuous_roi_s2_one_step_gate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --device cuda:0 \
  --amp \
  --output "${OUT_ROOT}/full_model_one_step_gate.json"

python tools/bata/precheck_continuous_roi_s2_training_runtime.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --manifest "${MANIFEST}" \
  --development-annotation "${ANNOTATION}" \
  --class-map "${CLASS_MAP}" \
  --development-video-root "${VIDEO_ROOT}" \
  --pretrained "${CHECKPOINT}" \
  --full-model-gate "${OUT_ROOT}/full_model_one_step_gate.json" \
  --output "${OUT_ROOT}/training_runtime_precheck.json"

for FAMILY in D160 G96 U128; do
  FAMILY_LOWER="$(printf '%s' "${FAMILY}" | tr '[:upper:]' '[:lower:]')"
  SOURCE_CONFIG="${ROOT}/configs/adatad/thumos/continuous_roi_s2_${FAMILY_LOWER}_videomae_s_768x1_adapter.py"
  RUNTIME_CONFIG="${OUT_ROOT}/training_runtime_gate/${FAMILY_LOWER}.py"
  RUNTIME_WORK_DIR="${OUT_ROOT}/training_runtime_gate/${FAMILY_LOWER}"
  RUNTIME_COMPLETION="${OUT_ROOT}/training_runtime_gate/${FAMILY_LOWER}_completion.json"

  python tools/bata/build_continuous_roi_s2_runtime_gate_config.py \
    "${SOURCE_CONFIG}" \
    --family "${FAMILY}" \
    --work-dir "${RUNTIME_WORK_DIR}" \
    --manifest "${MANIFEST}" \
    --development-annotation "${ANNOTATION}" \
    --class-map "${CLASS_MAP}" \
    --development-video-root "${VIDEO_ROOT}" \
    --pretrained "${CHECKPOINT}" \
    --full-model-gate "${OUT_ROOT}/full_model_one_step_gate.json" \
    --training-runtime-precheck "${OUT_ROOT}/training_runtime_precheck.json" \
    --output "${RUNTIME_CONFIG}"

  torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
    --rdzv_id="continuous-roi-s2-runtime-${SLURM_JOB_ID}-${FAMILY_LOWER}" \
    tools/train.py "${RUNTIME_CONFIG}" --seed 3407 --id 0

  python tools/bata/finalize_continuous_roi_s2_runtime_gate.py \
    --config "${RUNTIME_CONFIG}" \
    --seed 3407 \
    --checkpoint "${RUNTIME_WORK_DIR}/checkpoint/epoch_0.pth" \
    --output "${RUNTIME_COMPLETION}"
done

python tools/bata/authorize_continuous_roi_s2_training_runtime.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --full-model-gate "${OUT_ROOT}/full_model_one_step_gate.json" \
  --training-runtime-precheck "${OUT_ROOT}/training_runtime_precheck.json" \
  --completion "${OUT_ROOT}/training_runtime_gate/d160_completion.json" \
  --completion "${OUT_ROOT}/training_runtime_gate/g96_completion.json" \
  --completion "${OUT_ROOT}/training_runtime_gate/u128_completion.json" \
  --output "${OUT_ROOT}/training_runtime_authorization.json"

printf '[CONTINUOUS_ROI_S2_GATE] PASS output=%s\n' "${OUT_ROOT}"
