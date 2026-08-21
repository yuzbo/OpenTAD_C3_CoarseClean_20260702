#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_OFFICIAL_BC][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_OFFICIAL_BC][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
LINEAGE_BASE="01c58b9f2370e914150cf94d392208a4e211c053"
EXPECTED_COMMIT="${ZOOMTOKEN_OFFICIAL_BC_EXPECTED_COMMIT:?set ZOOMTOKEN_OFFICIAL_BC_EXPECTED_COMMIT to the reviewed clean commit}"
ROOT="${ZOOMTOKEN_OFFICIAL_BC_SOURCE_ROOT:?set ZOOMTOKEN_OFFICIAL_BC_SOURCE_ROOT to the reviewed clean checkout}"
RUN_ROOT="${ZOOMTOKEN_OFFICIAL_BC_RUN_ROOT:?set ZOOMTOKEN_OFFICIAL_BC_RUN_ROOT to the immutable paired result root}"
ARM="${ZOOMTOKEN_OFFICIAL_BC_ARM:?set ZOOMTOKEN_OFFICIAL_BC_ARM to B or C}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

case "${ARM}" in
  B)
    CONFIG_NAME="georoute_official_b_alltoken_postbackbone_seed42_v001.py"
    ARM_DIR="b_alltoken_sparse_adapter"
    ;;
  C)
    CONFIG_NAME="georoute_official_c_roi_postbackbone_seed42_v001.py"
    ARM_DIR="c_roi_k64_sparse_adapter"
    ;;
  *) fail 'official post-backbone matrix permits only B or C; A is completed job 1245842' ;;
esac
CONFIG="${ROOT}/configs/adatad/thumos/${CONFIG_NAME}"
CELL_ROOT="${RUN_ROOT}/cells/${ARM_DIR}/seed42"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'official post-backbone B/C requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'official post-backbone B/C requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || \
  fail 'official post-backbone B/C requires --cpus-per-task=8'

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
  fail 'official post-backbone expected commit must be a full SHA'
git -C "${ROOT}" merge-base --is-ancestor "${LINEAGE_BASE}" "${EXPECTED_COMMIT}" || \
  fail 'official post-backbone source does not descend from upstream AdaTAD 01c58b9'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'official post-backbone source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'official post-backbone source snapshot is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'official post-backbone result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || \
  fail 'official post-backbone result root must be outside the source checkout'
[[ ! -e "${CELL_ROOT}" ]] || \
  fail 'official post-backbone B/C cell already exists; A/B/C duplicates are forbidden'

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
command -v module >/dev/null 2>&1 || \
  fail 'the N16R4 environment-modules command is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${ROOT}"
exec torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-official-bc-${SLURM_JOB_ID}-${ARM_DIR}-seed42" \
  tools/train.py "${CONFIG}" --seed 42 --id 0 \
  --cfg-options \
  "work_dir=${CELL_ROOT}" \
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
