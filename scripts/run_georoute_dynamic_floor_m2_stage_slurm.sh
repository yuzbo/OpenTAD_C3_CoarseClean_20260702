#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00

set -euo pipefail

fail() {
  printf '[DYNAMIC_FLOOR_M2_STAGE] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT:?set GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT}"
ARM="${GEOROUTE_DYNAMIC_FLOOR_M2_ARM:?set GEOROUTE_DYNAMIC_FLOOR_M2_ARM}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
SOURCE_CONFIG="${GEOROUTE_DYNAMIC_FLOOR_M2_SOURCE_CONFIG:-${ROOT}/configs/adatad/thumos/georoute_dynamic_scnr_stage1_base.py}"
MANIFEST="${GEOROUTE_DEVELOPMENT_MANIFEST:?set GEOROUTE_DEVELOPMENT_MANIFEST}"
ANNOTATION="${GEOROUTE_DEVELOPMENT_ANNOTATION:?set GEOROUTE_DEVELOPMENT_ANNOTATION}"
CLASS_MAP="${GEOROUTE_CLASS_MAP:?set GEOROUTE_CLASS_MAP}"
VIDEO_ROOT="${GEOROUTE_DEVELOPMENT_VIDEO_ROOT:?set GEOROUTE_DEVELOPMENT_VIDEO_ROOT}"
PRETRAINED="${GEOROUTE_PRETRAINED:?set GEOROUTE_PRETRAINED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

case "${ARM}" in
  native_1cell_main|native_2cell_sensitivity) ;;
  *) fail "unsupported arm ${ARM}" ;;
esac
case "${PRECHECK_ONLY}" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be 0 or 1" ;;
esac
for path in "${SOURCE_CONFIG}" "${MANIFEST}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "development video root is missing"
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail "source root is not a Git checkout"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

COMMON_ARGS=(
  --arm "${ARM}"
  --run-root "${RUN_ROOT}"
  --source-config "${SOURCE_CONFIG}"
  --manifest "${MANIFEST}"
  --development-annotation "${ANNOTATION}"
  --class-map "${CLASS_MAP}"
  --development-video-root "${VIDEO_ROOT}"
  --pretrained "${PRETRAINED}"
  --expected-commit "${EXPECTED_COMMIT}"
)

cd "${ROOT}"
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  python tools/bata/georoute_dynamic_floor_m2_stage_runner.py \
    "${COMMON_ARGS[@]}" --precheck-only
  printf '[DYNAMIC_FLOOR_M2_STAGE] PRECHECK PASS arm=%s\n' "${ARM}"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "training must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "training requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-1}" == "1" ]] || fail "Slurm must expose one GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "stage requires five CPUs"
ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
[[ "$(awk -F, '{print NF}' <<<"${ALLOCATED_CPUS}")" == "5" ]] || \
  fail "Slurm affinity does not expose five CPUs"
command -v taskset >/dev/null 2>&1 || fail "taskset is required"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c "${ALLOCATED_CPUS}" \
  python tools/bata/georoute_dynamic_floor_m2_stage_runner.py "${COMMON_ARGS[@]}"
