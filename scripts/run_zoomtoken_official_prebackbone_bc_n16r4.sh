#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_PREBACKBONE_BC][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_PREBACKBONE_BC][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
LINEAGE_BASE="01c58b9f2370e914150cf94d392208a4e211c053"
EXPECTED_COMMIT="${ZOOMTOKEN_PREBACKBONE_BC_EXPECTED_COMMIT:?set ZOOMTOKEN_PREBACKBONE_BC_EXPECTED_COMMIT to the reviewed clean commit}"
ROOT="${ZOOMTOKEN_PREBACKBONE_BC_SOURCE_ROOT:?set ZOOMTOKEN_PREBACKBONE_BC_SOURCE_ROOT to the reviewed clean checkout}"
RUN_ROOT="${ZOOMTOKEN_PREBACKBONE_BC_RUN_ROOT:?set ZOOMTOKEN_PREBACKBONE_BC_RUN_ROOT to the immutable paired result root}"
ARM="${ZOOMTOKEN_PREBACKBONE_BC_ARM:?set ZOOMTOKEN_PREBACKBONE_BC_ARM to one frozen official arm}"
RESUME="${ZOOMTOKEN_PREBACKBONE_BC_RESUME:-}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

case "${ARM}" in
  B)
    CONFIG_NAME="georoute_official_b_alltoken_prebackbone_seed42_v001.py"
    ARM_DIR="b_alltoken_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=0
    ;;
  C)
    CONFIG_NAME="georoute_official_c_roi_k64_prebackbone_seed42_v001.py"
    ARM_DIR="c_roi_k64_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=0
    ;;
  R1)
    CONFIG_NAME="georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"
    ARM_DIR="r1_strict_rect8x8_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R1-DROP32)
    CONFIG_NAME="georoute_official_r1_drop32_prebackbone_seed42_v001.py"
    ARM_DIR="r1_drop32_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R1-MOD32-KV)
    CONFIG_NAME="georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py"
    ARM_DIR="r1_mod32_kv_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R1-RC32-KV)
    CONFIG_NAME="georoute_official_r1_rc32_kv_prebackbone_seed42_v001.py"
    ARM_DIR="r1_rc32_kv_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R1-DSR6-KV)
    CONFIG_NAME="georoute_official_r1_dsr6_kv_prebackbone_seed42_v001.py"
    ARM_DIR="r1_dsr6_kv_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R2)
    CONFIG_NAME="georoute_official_r2_strict_rect8x8_q48_prebackbone_seed42_v001.py"
    ARM_DIR="r2_strict_rect8x8_q48_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R2-SHUF48)
    CONFIG_NAME="georoute_official_r2_shuf48_prebackbone_seed42_v001.py"
    ARM_DIR="r2_shuf48_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  Q48-GLOBAL)
    CONFIG_NAME="georoute_official_q48_global_prebackbone_seed42_v001.py"
    ARM_DIR="q48_global_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R3)
    CONFIG_NAME="georoute_official_r3_continuous_rect_prebackbone_seed42_v001.py"
    ARM_DIR="r3_continuous_rect_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R3-AREA-SHIFT)
    CONFIG_NAME="georoute_official_r3_area_shift97_prebackbone_seed42_v001.py"
    ARM_DIR="r3_area_shift97_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R4)
    CONFIG_NAME="georoute_official_r4_core49_q15_prebackbone_seed42_v001.py"
    ARM_DIR="r4_core49_q15_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  R4-SHUF15)
    CONFIG_NAME="georoute_official_r4_shuf15_prebackbone_seed42_v001.py"
    ARM_DIR="r4_shuf15_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  Q64-GLOBAL)
    CONFIG_NAME="georoute_official_q64_global_prebackbone_seed42_v001.py"
    ARM_DIR="q64_global_prebackbone_sparse_adapter"
    RECOVERY_ENABLED=1
    ;;
  *) fail 'unknown frozen ZoomToken official pre-backbone arm' ;;
esac
CONFIG="${ROOT}/configs/adatad/thumos/${CONFIG_NAME}"
CELL_ROOT="${RUN_ROOT}/cells/${ARM_DIR}/seed42"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'official pre-backbone B/C requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'official pre-backbone B/C requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || \
  fail 'official pre-backbone B/C requires --cpus-per-task=8'

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
  fail 'official pre-backbone expected commit must be a full SHA'
git -C "${ROOT}" merge-base --is-ancestor "${LINEAGE_BASE}" "${EXPECTED_COMMIT}" || \
  fail 'official pre-backbone source does not descend from upstream AdaTAD 01c58b9'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'official pre-backbone source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'official pre-backbone source snapshot is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'official pre-backbone result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || \
  fail 'official pre-backbone result root must be outside the source checkout'
resume_args=()
if [[ -n "${RESUME}" ]]; then
  [[ "${RECOVERY_ENABLED}" == "1" ]] || fail 'same-cell recovery is disabled for B/C'
  [[ -d "${CELL_ROOT}" ]] || fail 'route recovery requires its existing same cell'
  [[ ! -e "${CELL_ROOT}/.zoomtoken_cell_sealed" ]] || \
    fail 'sealed route cells are not resumable'
  case "${RESUME}" in
    "${CELL_ROOT}"/checkpoint/recovery_epoch_*.pth) ;;
    *) fail 'route recovery requires same-cell checkpoint/recovery_epoch_<N>.pth' ;;
  esac
  [[ -f "${RESUME}" ]] || fail "route recovery checkpoint does not exist: ${RESUME}"
  resume_args=(--resume "${RESUME}")
else
  [[ ! -e "${CELL_ROOT}" ]] || \
    fail 'official pre-backbone cell already exists; duplicate cells are forbidden'
fi

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
  --rdzv_id="zoomtoken-prebackbone-bc-${SLURM_JOB_ID}-${ARM_DIR}-seed42" \
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
