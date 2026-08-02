#!/usr/bin/env bash
#SBATCH --job-name=georoute-dyn-health
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00

set -euo pipefail

fail() {
  printf '[GEOROUTE_DYNAMIC_POLICY_HEALTH][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_DYNAMIC_POLICY_HEALTH_RUN_ROOT:?set GEOROUTE_DYNAMIC_POLICY_HEALTH_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'policy health requires a Slurm allocation'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'policy health requires one Slurm-visible GPU and logical cuda:0'
[[ -e "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'run root must stay inside the remote write boundary' ;;
esac
[[ ! -e "${RUN_ROOT}" ]] || fail 'run root already exists; resume is forbidden'

for required in \
  "${GEOROUTE_SOURCE_CONFIG:?set GEOROUTE_SOURCE_CONFIG}" \
  "${GEOROUTE_MANIFEST:?set GEOROUTE_MANIFEST}" \
  "${GEOROUTE_DEVELOPMENT_ANNOTATION:?set GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  "${GEOROUTE_CLASS_MAP:?set GEOROUTE_CLASS_MAP}" \
  "${GEOROUTE_PRETRAINED:?set GEOROUTE_PRETRAINED}"; do
  [[ -f "${required}" ]] || fail "required file is missing: ${required}"
done
[[ -d "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT:?set GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" ]] || \
  fail 'development video root is missing'

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

python -m py_compile \
  tools/bata/georoute_dynamic_policy_health.py \
  tools/bata/run_georoute_dynamic_policy_health.py
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  python -m pytest tests/test_georoute_dynamic_policy_health.py -q
  printf '[GEOROUTE_DYNAMIC_POLICY_HEALTH][PASS] precheck only\n'
  exit 0
fi

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m tools.bata.run_georoute_dynamic_policy_health \
  --run-root "${RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --seed 4423
