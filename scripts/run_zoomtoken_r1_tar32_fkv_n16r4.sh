#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_R1_TAR32_FKV][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_R1_TAR32_FKV][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
LINEAGE_BASE="2d945e64bdccd09ae2e2916524562e3f388c5a2a"
EXPECTED_COMMIT="${ZOOMTOKEN_TAR32_EXPECTED_COMMIT:?set the reviewed clean TAR32-FKV commit}"
ROOT="${ZOOMTOKEN_TAR32_SOURCE_ROOT:?set the reviewed clean TAR32-FKV checkout}"
RUN_ROOT="${ZOOMTOKEN_TAR32_RUN_ROOT:?set the immutable TAR32-FKV result root}"
RESUME="${ZOOMTOKEN_TAR32_RESUME:-}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py"
CELL_ROOT="${RUN_ROOT}/cells/r1_tar32_fkv_prebackbone_sparse_adapter/seed42"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'TAR32-FKV requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'TAR32-FKV requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || \
  fail 'TAR32-FKV requires --cpus-per-task=8'
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
  fail 'expected commit must be a full SHA'
git -C "${ROOT}" merge-base --is-ancestor "${LINEAGE_BASE}" "${EXPECTED_COMMIT}" || \
  fail 'source does not descend from the Pro-frozen base'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || \
  fail 'result root must be outside the source checkout'

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  [[ -z "${RESUME}" ]] || fail 'PRECHECK_ONLY cannot resume training'
  printf '[ZOOMTOKEN_R1_TAR32_FKV][PRECHECK_READY] commit=%s source=%s run_root=%s\n' \
    "${EXPECTED_COMMIT}" "${ROOT}" "${RUN_ROOT}"
  exit 0
fi
[[ "${PRECHECK_ONLY}" == "0" ]] || fail 'PRECHECK_ONLY must be 0 or 1'

resume_args=()
if [[ -n "${RESUME}" ]]; then
  [[ -d "${CELL_ROOT}" ]] || fail 'recovery requires its existing same cell'
  [[ ! -e "${CELL_ROOT}/.zoomtoken_cell_sealed" ]] || fail 'sealed cells are not resumable'
  case "${RESUME}" in
    "${CELL_ROOT}"/checkpoint/recovery_epoch_*.pth) ;;
    *) fail 'recovery requires same-cell checkpoint/recovery_epoch_<N>.pth' ;;
  esac
  [[ -f "${RESUME}" ]] || fail "recovery checkpoint does not exist: ${RESUME}"
  resume_args=(--resume "${RESUME}")
else
  [[ ! -e "${CELL_ROOT}" ]] || fail 'TAR32-FKV cell already exists; duplicate cells are forbidden'
fi

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  # shellcheck disable=SC1091
  set +u
  source /etc/profile
  set -u
fi
command -v module >/dev/null 2>&1 || fail 'the N16R4 environment-modules command is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${ROOT}"
exec torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-r1-tar32-fkv-${SLURM_JOB_ID}-seed42" \
  tools/train.py "${CONFIG}" --seed 42 --id 0 "${resume_args[@]}" \
  --cfg-options \
  "work_dir=${CELL_ROOT}" \
  "zoomtoken_p1_config.source_commit=${EXPECTED_COMMIT}" \
  "dataset.train.ann_file=${ANNOTATION}" \
  "dataset.train.class_map=${CLASS_MAP}" \
  "dataset.train.data_path=${VIDEO_ROOT}" \
  "dataset.train.subset_name=training" \
  "dataset.val.ann_file=${ANNOTATION}" \
  "dataset.val.class_map=${CLASS_MAP}" \
  "dataset.val.data_path=${VIDEO_ROOT}" \
  "dataset.val.subset_name=validation" \
  "dataset.test.ann_file=${ANNOTATION}" \
  "dataset.test.class_map=${CLASS_MAP}" \
  "dataset.test.data_path=${VIDEO_ROOT}" \
  "dataset.test.subset_name=validation" \
  "evaluation.ground_truth_filename=${ANNOTATION}" \
  "model.backbone.custom.pretrain=${PRETRAINED}"
