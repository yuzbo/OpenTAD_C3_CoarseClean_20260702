#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() {
  printf '[GEOROUTE_OFFICIAL_DEVELOPMENT][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT:?set GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
MODE="${GEOROUTE_OFFICIAL_DEVELOPMENT_MODE:-formal}"
TASK="${GEOROUTE_OFFICIAL_DEVELOPMENT_TASK:-accuracy}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'formal development stage requires Slurm'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${#visible_gpus[@]}" -ne 2 ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=2 --cpus-per-task=10 \
    bash "${ROOT}/scripts/run_georoute_official_development_stage_slurm.sh"
fi
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'formal development requires two Slurm-visible GPUs'
[[ -e "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'run root leaves remote write boundary' ;;
esac

if [[ "${MODE}" == "p1" ]]; then
  IMAGE="${GEOROUTE_P1_RUNTIME_CONTAINER_IMAGE:?set GEOROUTE_P1_RUNTIME_CONTAINER_IMAGE}"
  LOCK="${GEOROUTE_P1_RUNTIME_DEPENDENCY_LOCK:?set GEOROUTE_P1_RUNTIME_DEPENDENCY_LOCK}"
  ACTIVE_CONTAINER="${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}"
  if [[ -z "${ACTIVE_CONTAINER}" ]]; then
    if command -v module >/dev/null 2>&1; then
      module load apptainer/1.2.4
    fi
    command -v apptainer >/dev/null 2>&1 || fail 'P1 requires Apptainer runtime entry'
    exec apptainer exec --nv --bind "${BASE}:${BASE}" "${IMAGE}" \
      bash "${ROOT}/scripts/run_georoute_official_development_stage_slurm.sh"
  fi
  [[ "$(readlink -f "${ACTIVE_CONTAINER}")" == "$(readlink -f "${IMAGE}")" ]] || \
    fail 'active immutable container differs from P1 deployment'
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

if [[ "${MODE}" == "p1" ]]; then
  PREFLIGHT="${GEOROUTE_P1_RUNTIME_PREFLIGHT:?set GEOROUTE_P1_RUNTIME_PREFLIGHT}"
  if [[ "${TASK}" == "preflight" ]]; then
    python -m tools.bata.georoute_p1_runtime_attestor \
      --phase preflight \
      --container-image "${IMAGE}" \
      --dependency-lock "${LOCK}" \
      --expected-visible-gpu-count 2 \
      --output "${PREFLIGHT}"
    exit 0
  fi
  ATTESTATION="${GEOROUTE_P1_RUNTIME_ATTESTATION:?set GEOROUTE_P1_RUNTIME_ATTESTATION}"
  python -m tools.bata.georoute_p1_runtime_attestor \
    --phase leaf \
    --container-image "${IMAGE}" \
    --dependency-lock "${LOCK}" \
    --expected-visible-gpu-count 2 \
    --reference "${PREFLIGHT}" \
    --output "${ATTESTATION}"
fi

python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__'

if [[ "${MODE}" == "p1" && "${TASK}" == "cost" ]]; then
  LEAF_ID="${GEOROUTE_P1_COST_LEAF_ID:?set GEOROUTE_P1_COST_LEAF_ID}"
  torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m tools.bata.georoute_official_development_stage_runner \
    --task cost \
    --leaf-id "${LEAF_ID}" \
    --seed "${GEOROUTE_OFFICIAL_DEVELOPMENT_SEED}" \
    --run-root "${RUN_ROOT}" \
    --expected-commit "${EXPECTED_COMMIT}"
else
  ARM="${GEOROUTE_OFFICIAL_DEVELOPMENT_ARM:?set GEOROUTE_OFFICIAL_DEVELOPMENT_ARM}"
  SEED="${GEOROUTE_OFFICIAL_DEVELOPMENT_SEED:?set GEOROUTE_OFFICIAL_DEVELOPMENT_SEED}"
  python -m tools.bata.georoute_official_development_stage_runner \
    --task accuracy \
    --arm "${ARM}" \
    --seed "${SEED}" \
    --run-root "${RUN_ROOT}" \
    --source-config "${GEOROUTE_SOURCE_CONFIG}" \
    --manifest "${GEOROUTE_MANIFEST}" \
    --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
    --class-map "${GEOROUTE_CLASS_MAP}" \
    --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
    --pretrained "${GEOROUTE_PRETRAINED}" \
    --expected-commit "${EXPECTED_COMMIT}"
fi
