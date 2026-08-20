#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_ROI60][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_ROI60][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
LINEAGE_BASE="bae6462754a3f1bc52da572c3c97444abd96e092"
EXPECTED_COMMIT="${ZOOMTOKEN_ROI60_EXPECTED_COMMIT:?set ZOOMTOKEN_ROI60_EXPECTED_COMMIT to the reviewed clean successor}"
ROOT="${ZOOMTOKEN_ROI60_SOURCE_ROOT:?set ZOOMTOKEN_ROI60_SOURCE_ROOT to the clean ROI60 checkout}"
RUN_ROOT="${ZOOMTOKEN_ROI60_RUN_ROOT:?set ZOOMTOKEN_ROI60_RUN_ROOT to the immutable paired result root}"
ARM="${ZOOMTOKEN_ROI60_ARM:?set ZOOMTOKEN_ROI60_ARM to DN or G}"
RESUME="${ZOOMTOKEN_ROI60_RESUME:-}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

case "${ARM}" in
  DN)
    CONFIG_NAME="georoute_p1_dn_seed3407_v001.py"
    ARM_DIR="dn"
    ;;
  G)
    CONFIG_NAME="georoute_p1_g_seed3407_v001.py"
    ARM_DIR="g"
    ;;
  *) fail 'ROI60 first screen permits only DN or ROI-only G' ;;
esac
CONFIG="${ROOT}/configs/adatad/thumos/${CONFIG_NAME}"
CELL_ROOT="${RUN_ROOT}/cells/${ARM_DIR}/seed3407"
WORK_DIR="${CELL_ROOT}/gpu1_id0"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'ROI60 training requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 1 ]] || \
  fail 'ROI60 training requires exactly one Slurm-visible GPU'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || fail 'ROI60 training requires --cpus-per-task=8'

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'ROI60 expected commit must be a full SHA'
git -C "${ROOT}" merge-base --is-ancestor "${LINEAGE_BASE}" "${EXPECTED_COMMIT}" || \
  fail 'ROI60 source does not descend from the frozen bae646 lineage'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'ROI60 source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'ROI60 source snapshot is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'ROI60 result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || fail 'ROI60 result root must be outside the source checkout'

resume_args=()
if [[ -n "${RESUME}" ]]; then
  [[ "$(dirname "${RESUME}")" == "${WORK_DIR}/checkpoint" ]] || \
    fail 'resume checkpoint is not from this exact DN/G cell'
  [[ "$(basename "${RESUME}")" =~ ^recovery_epoch_[0-9]+\.pth$ ]] || \
    fail 'resume requires recovery_epoch_<N>.pth'
  [[ -f "${RESUME}" ]] || fail "resume checkpoint does not exist: ${RESUME}"
  [[ ! -e "${WORK_DIR}/.zoomtoken_cell_sealed" ]] || fail 'sealed ROI60 cells cannot resume'
  resume_args=(--resume "${RESUME}")
else
  [[ ! -e "${CELL_ROOT}" ]] || fail 'fresh ROI60 cell root must not already exist'
fi

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
command -v module >/dev/null 2>&1 || fail 'the N16R4 environment-modules command is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${ROOT}"
exec torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-roi60-${SLURM_JOB_ID}-${ARM_DIR}-seed3407" \
  tools/train.py "${CONFIG}" --seed 3407 --id 0 --work-dir "${CELL_ROOT}" \
  "${resume_args[@]}" \
  --cfg-options \
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
  "evaluation.type=mAP" \
  "evaluation.subset=validation" \
  "evaluation.tiou_thresholds=[0.3,0.4,0.5,0.6,0.7]" \
  "evaluation.ground_truth_filename=${ANNOTATION}" \
  "post_processing.nms.use_soft_nms=True" \
  "post_processing.nms.sigma=0.7" \
  "post_processing.nms.max_seg_num=2000" \
  "post_processing.nms.multiclass=True" \
  "post_processing.nms.voting_thresh=0.7" \
  "post_processing.save_dict=False" \
  "model.backbone.custom.pretrain=${PRETRAINED}" \
  "workflow.end_epoch=60" \
  "workflow.checkpoint_interval=5" \
  "workflow.checkpoint_policy=recovery_latest3_plus_final" \
  "workflow.val_loss_interval=-1" \
  "workflow.val_eval_interval=60" \
  "workflow.val_start_epoch=59" \
  "zoomtoken_p1_config.split=training_to_official_validation" \
  "zoomtoken_p1_config.official_test_open_allowed=False" \
  "zoomtoken_p1_config.gt_for_route_allowed=False" \
  "zoomtoken_p1_config.teacher_for_route_allowed=False" \
  "zoomtoken_p1_config.oracle_for_route_allowed=False" \
  "zoomtoken_p1_config.raw_prediction_cache_allowed=False"
