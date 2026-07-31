#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_OFFICIAL_WORLD2_KAT][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_OFFICIAL_PREFLIGHT_RUN_ROOT:?set GEOROUTE_OFFICIAL_PREFLIGHT_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'world2 KAT requires Slurm'
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=2 --cpus-per-task=10 --mem=32000M \
    bash "${ROOT}/scripts/run_georoute_official_world2_ddp_kat_slurm.sh"
fi

VISIBLE_COUNT="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-}")"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${VISIBLE_COUNT}" == "2" ]] || \
  fail 'world2 KAT requires exactly two Slurm-visible GPUs'
[[ -e "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'run root leaves remote write boundary' ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__'

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="georoute-formal-world2-${SLURM_JOB_ID}" \
  -m tools.bata.run_georoute_official_world2_ddp_kat \
  --run-root "${RUN_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}"
