#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00

source /etc/profile
set -euo pipefail

fail() {
  printf '[SCNR_RESIDUAL_CENTERING_COST] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${SCNR_RESIDUAL_CENTERING_COST_RUN_ROOT:?set SCNR_RESIDUAL_CENTERING_COST_RUN_ROOT}"
TRAINING_ROOT="${SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT:?set SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT}"
MODEL_RUNTIME_COMMIT="${SCNR_RESIDUAL_CENTERING_MODEL_RUNTIME_COMMIT:?set SCNR_RESIDUAL_CENTERING_MODEL_RUNTIME_COMMIT}"
EXECUTION_COMMIT="${SCNR_RESIDUAL_CENTERING_COST_EXECUTION_COMMIT:?set SCNR_RESIDUAL_CENTERING_COST_EXECUTION_COMMIT}"
DEPLOYMENT="${SCNR_RESIDUAL_CENTERING_COST_DEPLOYMENT:-${RUN_ROOT}/control/deployment.json}"
POWER_SCRATCH_ROOT="${SCNR_RESIDUAL_CENTERING_COST_POWER_SCRATCH_ROOT:-/tmp/scnr_residual_centering_cost_power}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

case "${PRECHECK_ONLY}" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be 0 or 1" ;;
esac
case "${POWER_SCRATCH_ROOT}" in
  /tmp/*|/var/tmp/*) ;;
  *) fail "power scratch must be node-local /tmp or /var/tmp" ;;
esac
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail "source root is not a Git checkout"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${ROOT}"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  python -m py_compile \
    tools/bata/georoute_residual_centering_cost_contract.py \
    tools/bata/profile_georoute_residual_centering_cost.py \
    tools/bata/deploy_georoute_residual_centering_cost.py \
    tools/bata/finalize_georoute_residual_centering_cost.py
  python - "${TRAINING_ROOT}" "${MODEL_RUNTIME_COMMIT}" <<'PY'
import sys
from tools.bata.georoute_residual_centering_cost_contract import (
    RESIDUAL_CENTERING_COST_ORDER,
    RESIDUAL_CENTERING_COST_PAIRS,
    validate_frozen_residual_centering_cost_contract,
    validate_residual_centering_cost_source,
)
validate_frozen_residual_centering_cost_contract()
assert RESIDUAL_CENTERING_COST_ORDER[:4] == (
    "none_control",
    "residual_window_center",
    "residual_window_center",
    "none_control",
)
assert RESIDUAL_CENTERING_COST_ORDER[4:] == (
    "residual_window_center",
    "none_control",
    "none_control",
    "residual_window_center",
)
assert RESIDUAL_CENTERING_COST_PAIRS == ((0, 1), (3, 2), (5, 4), (6, 7))
validate_residual_centering_cost_source(
    sys.argv[1], expected_model_runtime_commit=sys.argv[2]
)
print("PASS_RESIDUAL_CENTERING_PAIRED_COST_STATIC_PRECHECK")
PY
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "paired cost must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "paired cost requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-1}" == "1" ]] || fail "Slurm must expose one GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "paired cost requires five CPUs"
[[ -f "${DEPLOYMENT}" ]] || fail "immutable deployment receipt is missing"
command -v taskset >/dev/null 2>&1 || fail "paired cost requires taskset"

ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
IFS=',' read -r -a CPU_ARRAY <<< "${ALLOCATED_CPUS}"
[[ "${#CPU_ARRAY[@]}" == "5" ]] || fail "Slurm affinity does not expose five CPUs"
DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]},${CPU_ARRAY[2]},${CPU_ARRAY[3]}"
SIDECAR_CPU="${CPU_ARRAY[4]}"
mkdir -p "${POWER_SCRATCH_ROOT}"

set +e
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c "${DETECTOR_CPUS}" \
  python -m torch.distributed.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint=127.0.0.1:0 \
    --rdzv-id="${SLURM_JOB_ID}" \
    --nproc_per_node=1 \
    tools/bata/profile_georoute_residual_centering_cost.py \
    --run-root "${RUN_ROOT}" \
    --training-run-root "${TRAINING_ROOT}" \
    --deployment "${DEPLOYMENT}" \
    --model-runtime-commit "${MODEL_RUNTIME_COMMIT}" \
    --execution-commit "${EXECUTION_COMMIT}" \
    --allocated-cpus "${ALLOCATED_CPUS}" \
    --detector-cpus "${DETECTOR_CPUS}" \
    --sidecar-cpu "${SIDECAR_CPU}" \
    --power-scratch-root "${POWER_SCRATCH_ROOT}"
PROFILE_STATUS=$?
set -e

set +e
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python tools/bata/finalize_georoute_residual_centering_cost.py \
    --run-root "${RUN_ROOT}" \
    --training-run-root "${TRAINING_ROOT}" \
    --deployment "${DEPLOYMENT}" \
    --model-runtime-commit "${MODEL_RUNTIME_COMMIT}" \
    --execution-commit "${EXECUTION_COMMIT}"
FINALIZER_STATUS=$?
set -e

if [[ "${PROFILE_STATUS}" -ne 0 ]]; then
  fail "paired cost profiler failed with status ${PROFILE_STATUS}; failure finalizer status=${FINALIZER_STATUS}"
fi
if [[ "${FINALIZER_STATUS}" -ne 0 ]]; then
  fail "paired cost finalizer failed with status ${FINALIZER_STATUS}"
fi
