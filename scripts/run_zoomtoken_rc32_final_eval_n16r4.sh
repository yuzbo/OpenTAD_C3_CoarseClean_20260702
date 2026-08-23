#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_RC32_FINAL_EVAL][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
ROOT="${ZOOMTOKEN_RC32_EVAL_SOURCE_ROOT:?set the reviewed clean source root}"
RUN_ROOT="${ZOOMTOKEN_RC32_EVAL_RUN_ROOT:?set the completed RC32 matrix root}"
EXPECTED_COMMIT="${ZOOMTOKEN_RC32_EVAL_EXPECTED_COMMIT:?set the evaluator source commit}"
ARM="${ZOOMTOKEN_RC32_EVAL_ARM:?set FULL64, DROP32, MOD32-KV, or RC32-KV}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

case "${ARM}" in
  FULL64)
    CONFIG_NAME="georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"
    ARM_DIR="r1_strict_rect8x8_prebackbone_sparse_adapter"
    ;;
  DROP32)
    CONFIG_NAME="georoute_official_r1_drop32_prebackbone_seed42_v001.py"
    ARM_DIR="r1_drop32_prebackbone_sparse_adapter"
    ;;
  MOD32-KV)
    CONFIG_NAME="georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py"
    ARM_DIR="r1_mod32_kv_prebackbone_sparse_adapter"
    ;;
  RC32-KV)
    CONFIG_NAME="georoute_official_r1_rc32_kv_prebackbone_seed42_v001.py"
    ARM_DIR="r1_rc32_kv_prebackbone_sparse_adapter"
    ;;
  *) fail 'unknown ZoomToken final-evaluation arm' ;;
esac

CONFIG="${ROOT}/configs/adatad/thumos/${CONFIG_NAME}"
CELL_ROOT="${RUN_ROOT}/cells/${ARM_DIR}/seed42"
WORK_DIR="${CELL_ROOT}/gpu2_id0"
CHECKPOINT="${WORK_DIR}/checkpoint/epoch_59.pth"
RESULT="${WORK_DIR}/result_detection.json"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'final evaluation requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'final evaluation requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || fail 'final evaluation requires 8 CPUs'
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full SHA'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'evaluator source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'evaluator source is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" \
  "${CONDA_ACTIVATE}" "${CHECKPOINT}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail 'canonical THUMOS14 video root is missing'
[[ ! -e "${RESULT}" ]] || fail 'result_detection.json already exists'

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${ROOT}"
exec torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-rc32-final-eval-${SLURM_JOB_ID}-${ARM_DIR}" \
  tools/test.py "${CONFIG}" --checkpoint "${CHECKPOINT}" --seed 42 --id 0 \
  --cfg-options \
  "work_dir=${CELL_ROOT}" \
  "dataset.test.ann_file=${ANNOTATION}" \
  "dataset.test.class_map=${CLASS_MAP}" \
  "dataset.test.data_path=${VIDEO_ROOT}" \
  "dataset.test.subset_name=validation" \
  "evaluation.ground_truth_filename=${ANNOTATION}" \
  "post_processing.save_dict=True" \
  "model.backbone.custom.pretrain=${PRETRAINED}"
