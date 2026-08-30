#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_GRIDFUSE32_L6][FAIL] invoke with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_GRIDFUSE32_L6][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
EXPECTED_COMMIT="${ZOOMTOKEN_GRIDFUSE_EXPECTED_COMMIT:?set the reviewed clean commit}"
ROOT="${ZOOMTOKEN_GRIDFUSE_SOURCE_ROOT:?set the reviewed clean checkout}"
RESULT_ROOT="${ZOOMTOKEN_GRIDFUSE_RESULT_ROOT:?set the immutable task result root}"
PHASE="${ZOOMTOKEN_GRIDFUSE_PHASE:-G0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
BRANCH="codex/zoomtoken-gridfuse32-l6-v001"
REMOTE_REF="refs/remotes/origin/${BRANCH}"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_gridfuse32_l6_prebackbone_seed42_v001.py"
R1_CHECKPOINT="${ZOOMTOKEN_GRIDFUSE_R1_CHECKPOINT:-${BASE}/projects/zoomtoken_official_prebackbone_r1_9e25c6d3_seed42_20260822T080108Z/cells/r1_strict_rect8x8_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth}"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full lowercase SHA'
[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'GridFuse32-L6 actions require a Slurm allocation'
case "${RESULT_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves the remote write boundary' ;;
esac
[[ "${RESULT_ROOT}/" != "${ROOT}/"* ]] || fail 'result root must be outside the source checkout'
for path in "${CONFIG}" "${R1_CHECKPOINT}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source checkout is not clean'
[[ "$(git -C "${ROOT}" rev-parse "${REMOTE_REF}")" == "${EXPECTED_COMMIT}" ]] || \
  fail 'prefetched GitHub remote-tracking ref does not resolve to the reviewed candidate'

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/profile
  set -u
fi
command -v module >/dev/null 2>&1 || fail 'environment-modules is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${PRECHECK_ONLY}" == "0" ]] || \
  fail 'the final atomic task forbids a standalone PRECHECK_ONLY scheduler job'
[[ "${PHASE}" == "G0" ]] || \
  fail 'G1 and G2 remain closed pending a fresh Pro decision'
[[ "${#visible_gpus[@]}" -eq 1 ]] || fail 'atomic G0 requires exactly one visible GPU'
[[ "${SLURM_CPUS_PER_TASK:-}" == "4" ]] || fail 'atomic G0 requires --cpus-per-task=4'
G0_ROOT="${RESULT_ROOT}/g0"
[[ ! -e "${G0_ROOT}" ]] || fail 'exclusive atomic G0 result root already exists'
set +e
python tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py \
  --config "${CONFIG}" \
  --checkpoint "${R1_CHECKPOINT}" \
  --run-root "${G0_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --warmup 100 \
  --iterations 500
status=$?
set -e
exit "${status}"
