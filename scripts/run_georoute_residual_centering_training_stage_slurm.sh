#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00

source /etc/profile
set -euo pipefail

fail() {
  printf '[SCNR_RESIDUAL_CENTERING_TRAINING] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT:?set SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT}"
VARIANT="${SCNR_RESIDUAL_CENTERING_TRAINING_VARIANT:?set SCNR_RESIDUAL_CENTERING_TRAINING_VARIANT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
SOURCE_CONFIG="${SCNR_RESIDUAL_CENTERING_TRAINING_SOURCE_CONFIG:-${ROOT}/configs/adatad/thumos/georoute_dynamic_scnr_stage1_base.py}"
MANIFEST="${GEOROUTE_DEVELOPMENT_MANIFEST:?set GEOROUTE_DEVELOPMENT_MANIFEST}"
ANNOTATION="${GEOROUTE_DEVELOPMENT_ANNOTATION:?set GEOROUTE_DEVELOPMENT_ANNOTATION}"
CLASS_MAP="${GEOROUTE_CLASS_MAP:?set GEOROUTE_CLASS_MAP}"
VIDEO_ROOT="${GEOROUTE_DEVELOPMENT_VIDEO_ROOT:?set GEOROUTE_DEVELOPMENT_VIDEO_ROOT}"
PRETRAINED="${GEOROUTE_PRETRAINED:?set GEOROUTE_PRETRAINED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

case "${VARIANT}" in
  none_control|residual_window_center) ;;
  *) fail "unsupported variant ${VARIANT}" ;;
esac
case "${PRECHECK_ONLY}" in
  0|1) ;;
  *) fail 'PRECHECK_ONLY must be 0 or 1' ;;
esac
for path in "${SOURCE_CONFIG}" "${MANIFEST}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail 'development video root is missing'
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail 'source root is not a Git checkout'

module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__'

COMMON_ARGS=(
  --variant "${VARIANT}"
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
  python tools/bata/run_georoute_residual_centering_training.py \
    "${COMMON_ARGS[@]}" --precheck-only
  printf '[SCNR_RESIDUAL_CENTERING_TRAINING] PRECHECK PASS variant=%s\n' "${VARIANT}"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'training must run inside Slurm'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'training requires exactly one Slurm-visible GPU'
[[ "${SLURM_GPUS_ON_NODE:-1}" == "1" ]] || fail 'Slurm must expose one GPU'
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail 'stage requires five CPUs'
ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
[[ "$(awk -F, '{print NF}' <<<"${ALLOCATED_CPUS}")" == "5" ]] || \
  fail 'Slurm affinity does not expose five CPUs'
command -v taskset >/dev/null 2>&1 || fail 'taskset is required'

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c "${ALLOCATED_CPUS}" \
  python tools/bata/run_georoute_residual_centering_training.py "${COMMON_ARGS[@]}"
