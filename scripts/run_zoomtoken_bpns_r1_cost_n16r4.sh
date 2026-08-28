#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_BPNS_COST_V003][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_BPNS_COST_V003][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
EXPECTED_COMMIT="${ZOOMTOKEN_BPNS_COST_EXPECTED_COMMIT:?set the reviewed clean commit}"
ROOT="${ZOOMTOKEN_BPNS_COST_SOURCE_ROOT:?set the reviewed clean checkout}"
RESULT_ROOT="${ZOOMTOKEN_BPNS_COST_RESULT_ROOT:?set the unique formal result root}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
K100_CHECKPOINT="${BASE}/projects/zoomtoken_official_prebackbone_b_70dcbe10_seed42_20260821T205840Z/cells/b_alltoken_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth"
R1_CHECKPOINT="${BASE}/projects/zoomtoken_official_prebackbone_r1_9e25c6d3_seed42_20260822T080108Z/cells/r1_strict_rect8x8_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'BPNS cost replay requires a Slurm allocation'
[[ "${SLURM_JOB_NAME:-}" == *v003* ]] || fail 'formal Slurm JobName must use a fresh v003 namespace'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 1 ]] || \
  fail 'BPNS cost replay requires exactly one Slurm-visible GPU'
[[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || \
  fail 'BPNS cost replay requires --cpus-per-task=5'
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full SHA'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source checkout is not clean'
for path in "${ANNOTATION}" "${CLASS_MAP}" "${K100_CHECKPOINT}" "${R1_CHECKPOINT}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical video root does not exist: ${VIDEO_ROOT}"
[[ "$(find -L "${VIDEO_ROOT}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 411 ]] || \
  fail 'canonical THUMOS14 inventory must contain 411 readable MP4 targets'
case "${RESULT_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves /data/run01/sczc063/yuzibo' ;;
esac
[[ ! -e "${RESULT_ROOT}" ]] || fail 'result root already exists; duplicate replay is forbidden'
[[ "$(basename "${RESULT_ROOT}")" == *v003* ]] || fail 'formal result root must use a fresh v003 namespace'

if [[ -r /etc/profile ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/profile
  set -u
fi
command -v module >/dev/null 2>&1 || fail 'environment-modules command is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

common_args=(
  --expected-commit "${EXPECTED_COMMIT}"
  --result-root "${RESULT_ROOT}"
  --annotation "${ANNOTATION}"
  --class-map "${CLASS_MAP}"
  --video-root "${VIDEO_ROOT}"
  --k100-checkpoint "${K100_CHECKPOINT}"
  --r1-checkpoint "${R1_CHECKPOINT}"
)

cd "${ROOT}"
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  exec python tools/bata/profile_zoomtoken_bpns_r1_cost.py precheck "${common_args[@]}"
fi
[[ "${PRECHECK_ONLY}" == "0" ]] || fail 'PRECHECK_ONLY must be 0 or 1'
exec torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-bpns-r1-cost-v003-${SLURM_JOB_ID}" \
  tools/bata/profile_zoomtoken_bpns_r1_cost.py profile "${common_args[@]}"
