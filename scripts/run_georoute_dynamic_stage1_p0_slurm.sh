#!/usr/bin/env bash
#SBATCH --job-name=scnr_dyn_p0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00

set -euo pipefail

fail() {
  printf '[DYNAMIC_STAGE1_P0][FAIL] %s\n' "$*" >&2
  exit 2
}

SOURCE_ROOT="${DYNAMIC_STAGE1_SOURCE_ROOT:?set DYNAMIC_STAGE1_SOURCE_ROOT}"
EXPECTED_COMMIT="${DYNAMIC_STAGE1_EXPECTED_COMMIT:?set DYNAMIC_STAGE1_EXPECTED_COMMIT}"
OUTPUT="${DYNAMIC_STAGE1_P0_OUTPUT:?set DYNAMIC_STAGE1_P0_OUTPUT}"
PRETRAINED="${DYNAMIC_STAGE1_PRETRAINED:?set DYNAMIC_STAGE1_PRETRAINED}"
CONFIG="${DYNAMIC_STAGE1_CONFIG:-${SOURCE_ROOT}/configs/adatad/thumos/georoute_dynamic_scnr_stage1_base.py}"
SEED="${DYNAMIC_STAGE1_SEED:-3407}"
RUNTIME_BASE="${DYNAMIC_STAGE1_RUNTIME_BASE:-/data/run01/sczc063/yuzibo}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'CUDA P0 requires a Slurm allocation'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'CUDA P0 requires exactly one Slurm-visible GPU'
[[ -d "${SOURCE_ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full lowercase SHA'
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source HEAD differs from expected commit'
[[ "$(git -C "${SOURCE_ROOT}" rev-parse refs/remotes/origin/codex/spatial-zoom-s1-audit-fix-20260715)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'origin tracking ref differs from expected commit'
[[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source tree is not clean'
[[ -f "${CONFIG}" ]] || fail 'dynamic Stage-1 config is missing'
[[ -f "${PRETRAINED}" ]] || fail 'VideoMAE pretrained checkpoint is missing'
[[ ! -e "${OUTPUT}" && ! -e "${OUTPUT}.tmp" ]] || fail 'P0 output namespace already exists'
case "${OUTPUT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'P0 output must stay inside the remote write boundary' ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${RUNTIME_BASE}/conda_envs/opentad/bin/activate"
cd "${SOURCE_ROOT}"

python -m py_compile \
  opentad/models/backbones/georoute_routing.py \
  opentad/models/backbones/georoute_wrapper.py \
  opentad/models/backbones/vit_adapter.py \
  tools/bata/run_georoute_dynamic_stage1_p0.py

python -m tools.bata.run_georoute_dynamic_stage1_p0 \
  --config "${CONFIG}" \
  --pretrained "${PRETRAINED}" \
  --output "${OUTPUT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --device cuda:0 \
  --seed "${SEED}"

printf '[DYNAMIC_STAGE1_P0] PASS report=%s\n' "${OUTPUT}"
