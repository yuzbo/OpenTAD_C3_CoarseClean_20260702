#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00

set -euo pipefail

fail() {
  printf '[DYNAMIC_FLOOR_M2_COST] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT:?set GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
POWER_SCRATCH_ROOT="${GEOROUTE_DYNAMIC_FLOOR_M2_POWER_SCRATCH_ROOT:-/tmp/scnr_dynamic_floor_m2_power}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
G1_RESULT="${RUN_ROOT}/development/g1_native_1cell_main/seed3407/stage_result.json"
G2_RESULT="${RUN_ROOT}/development/g2_native_2cell_sensitivity/seed3407/stage_result.json"

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
  python - <<'PY'
from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_COST_ORDER,
    validate_frozen_dynamic_floor_m2_contract,
)
validate_frozen_dynamic_floor_m2_contract()
assert DYNAMIC_FLOOR_M2_COST_ORDER[0] == DYNAMIC_FLOOR_M2_COST_ORDER[-1]
assert DYNAMIC_FLOOR_M2_COST_ORDER[1] == DYNAMIC_FLOOR_M2_COST_ORDER[2]
print("PASS_DYNAMIC_FLOOR_M2_COST_STATIC_PRECHECK")
PY
  exit 0
fi

for path in "${G1_RESULT}" "${G2_RESULT}"; do
  [[ -f "${path}" ]] || fail "stage result is missing: ${path}"
done
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "cost replay must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "cost replay requires exactly one Slurm-visible GPU"
[[ "${SLURM_GPUS_ON_NODE:-1}" == "1" ]] || fail "Slurm must expose one GPU"
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail "cost replay requires five CPUs"
command -v taskset >/dev/null 2>&1 || fail "cost replay requires taskset"

ALLOCATED_CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
IFS=',' read -r -a CPU_ARRAY <<< "${ALLOCATED_CPUS}"
[[ "${#CPU_ARRAY[@]}" == "5" ]] || fail "Slurm affinity does not expose five CPUs"
DETECTOR_CPUS="${CPU_ARRAY[0]},${CPU_ARRAY[1]},${CPU_ARRAY[2]},${CPU_ARRAY[3]}"
SIDECAR_CPU="${CPU_ARRAY[4]}"
mkdir -p "${POWER_SCRATCH_ROOT}"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c "${DETECTOR_CPUS}" \
  python -m torch.distributed.run --standalone --nproc_per_node=1 \
    tools/bata/profile_georoute_dynamic_floor_m2.py \
    --run-root "${RUN_ROOT}" \
    --stage-result-g1 "${G1_RESULT}" \
    --stage-result-g2 "${G2_RESULT}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --allocated-cpus "${ALLOCATED_CPUS}" \
    --detector-cpus "${DETECTOR_CPUS}" \
    --sidecar-cpu "${SIDECAR_CPU}" \
    --power-scratch-root "${POWER_SCRATCH_ROOT}"
