#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_K100_TAR50][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_K100_TAR50][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
TASK_BASE="2d945e64bdccd09ae2e2916524562e3f388c5a2a"
EXPECTED_COMMIT="${ZOOMTOKEN_K100_TAR50_EXPECTED_COMMIT:?set the reviewed clean commit}"
ROOT="${ZOOMTOKEN_K100_TAR50_SOURCE_ROOT:?set the reviewed clean checkout}"
RUN_ROOT="${ZOOMTOKEN_K100_TAR50_RUN_ROOT:?set the fresh result root}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
REMOTE_REF="refs/remotes/origin/codex/zoomtoken-k100-tar50-interaction-v001"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_amod50_prebackbone_seed42_v001.py"
CELL_ROOT="${RUN_ROOT}/cells/k100_tar50_interaction/seed42"
EVAL_ROOT="${RUN_ROOT}/final_ema_eval"
CHECKPOINT="${CELL_ROOT}/gpu2_id0/checkpoint/epoch_59.pth"
RESULT_PATH="${EVAL_ROOT}/gpu2_id0/result_detection.json"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"
REFERENCE_CHECKPOINT="${BASE}/projects/official_adatad_reproduction_run_01c58b9_seed42_commandfix_20260821/gpu2_id0/checkpoint/epoch_59.pth"
REFERENCE_PREDICTION="${BASE}/projects/zoomtoken_amod50_capacity1_parity_2d945e64_20260825/gpu2_id0/gpu2_id0/result_detection.json"
REFERENCE_CHECKPOINT_SHA="3aca10bc3593e301b7d7e77271419b8bb557d8f8b29bead195fa2aa350e34ddd"
REFERENCE_PREDICTION_SHA="0d09e3fec839449923db1158a18ead631e813b9d00cdab051328cb2b407485f3"
REFERENCE_CONFIG_SHA="81c805838502639d4fb0e6fcdd0848c53ccbd8eeccf7d1501562af2e84d9ac87"
EXPECTED_ANNOTATION_SHA="ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad"
EXPECTED_CLASS_MAP_SHA="a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31"
EXPECTED_PRETRAINED_SHA="4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
LAUNCH_RECEIPT="${RUN_ROOT}/launch_receipt.tsv"
TERMINAL_RECEIPT="${RUN_ROOT}/terminal_receipt.tsv"
TRAIN_LOG="${RUN_ROOT}/training.log"
EVALUATION_LOG="${RUN_ROOT}/evaluation.log"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'precheck and formal execution require a Slurm allocation'
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  [[ "${SLURM_JOB_NAME:-}" == "zt-k100-tar50-pre" ]] || fail 'precheck JobName must be zt-k100-tar50-pre'
else
  [[ "${SLURM_JOB_NAME:-}" == "zt-k100-tar50-s42" ]] || fail 'formal JobName must be zt-k100-tar50-s42'
fi
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || fail 'requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || fail 'requires --cpus-per-task=8'
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full SHA'
git -C "${ROOT}" merge-base --is-ancestor "${TASK_BASE}" "${EXPECTED_COMMIT}" || fail 'candidate is not a descendant of the frozen task base'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source snapshot is not clean'
[[ "$(git -C "${ROOT}" rev-parse "${REMOTE_REF}")" == "${EXPECTED_COMMIT}" ]] || fail 'candidate is not bound to the pushed remote-tracking ref'

for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}" "${REFERENCE_CHECKPOINT}" "${REFERENCE_PREDICTION}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || fail 'result root must be outside the source checkout'
[[ ! -e "${RUN_ROOT}" ]] || fail 'fresh result root already exists'

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "SHA-256 mismatch: ${path}"
}
check_sha "${REFERENCE_CHECKPOINT_SHA}" "${REFERENCE_CHECKPOINT}"
check_sha "${REFERENCE_PREDICTION_SHA}" "${REFERENCE_PREDICTION}"
check_sha "${REFERENCE_CONFIG_SHA}" "${CONFIG}"
check_sha "${EXPECTED_ANNOTATION_SHA}" "${ANNOTATION}"
check_sha "${EXPECTED_CLASS_MAP_SHA}" "${CLASS_MAP}"
check_sha "${EXPECTED_PRETRAINED_SHA}" "${PRETRAINED}"
[[ "$(find -L "${VIDEO_ROOT}" -type f -name '*.mp4' | wc -l | tr -d ' ')" == "411" ]] || fail 'canonical video inventory is not 411 MP4 files'

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf '[ZOOMTOKEN_K100_TAR50][PRECHECK_READY] commit=%s reference_job=1254040 route=K100,K50x6\n' "${EXPECTED_COMMIT}"
  exit 0
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
mkdir -p "${RUN_ROOT}"
launch_tmp="${LAUNCH_RECEIPT}.tmp.$$"
{
  printf 'schema_version\tzoomtoken_k100_tar50_interaction_launch_v001\n'
  printf 'created_at\t%s\n' "$(date -Iseconds)"
  printf 'task\tZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001\n'
  printf 'commit\t%s\nbase_commit\t%s\n' "${EXPECTED_COMMIT}" "${TASK_BASE}"
  printf 'config\t%s\nconfig_sha256\t%s\n' "${CONFIG}" "${REFERENCE_CONFIG_SHA}"
  printf 'reference_job\t1254040\nreference_checkpoint_sha256\t%s\nreference_prediction_sha256\t%s\n' "${REFERENCE_CHECKPOINT_SHA}" "${REFERENCE_PREDICTION_SHA}"
  printf 'reference_official_vector\t68.73/61.59/47.20\nreference_population\t211 videos / 792 ordered windows / 411 MP4\n'
  printf 'slurm_job_id\t%s\nslurm_job_name\t%s\nseed\t42\nrank_count\t2\n' "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}"
  printf 'route\tnative K100; [K100,K50]x6; flattened 800/400; full K/V; full Adapter; identity bypass\n'
  printf 'formal_submission_limit\t1\nresume_allowed\tfalse\nreplacement_allowed\tfalse\ncost_authorized\tfalse\n'
} > "${launch_tmp}"
mv "${launch_tmp}" "${LAUNCH_RECEIPT}"

set +e
torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-k100-tar50-train-${SLURM_JOB_ID}-seed42" \
  tools/train.py "${CONFIG}" --seed 42 --id 0 \
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
  "model.backbone.custom.pretrain=${PRETRAINED}" \
  2>&1 | tee "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}

eval_status=125
if [[ "${train_status}" -eq 0 ]]; then
  checkpoint_status=125
  if [[ -s "${CHECKPOINT}" ]]; then
    python - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
if checkpoint.get("epoch") != 59:
    raise SystemExit("epoch_59 checkpoint does not report epoch=59")
if not isinstance(checkpoint.get("state_dict_ema"), dict) or not checkpoint["state_dict_ema"]:
    raise SystemExit("epoch_59 checkpoint lacks non-empty state_dict_ema")
PY
    checkpoint_status=$?
  fi
  if [[ "${checkpoint_status}" -eq 0 ]]; then
    torchrun --nnodes=1 --nproc_per_node=2 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="zoomtoken-k100-tar50-eval-${SLURM_JOB_ID}-seed42" \
      tools/test.py "${CONFIG}" --checkpoint "${CHECKPOINT}" --seed 42 --id 0 \
      --cfg-options \
      "work_dir=${EVAL_ROOT}" \
      "zoomtoken_p1_config.source_commit=${EXPECTED_COMMIT}" \
      "dataset.test.ann_file=${ANNOTATION}" \
      "dataset.test.class_map=${CLASS_MAP}" \
      "dataset.test.data_path=${VIDEO_ROOT}" \
      "dataset.test.subset_name=validation" \
      "evaluation.ground_truth_filename=${ANNOTATION}" \
      "model.backbone.custom.pretrain=${PRETRAINED}" \
      "post_processing.save_dict=True" \
      2>&1 | tee "${EVALUATION_LOG}"
    eval_status=${PIPESTATUS[0]}
  else
    eval_status="${checkpoint_status}"
  fi
fi
set -e

if [[ "${eval_status}" -eq 0 ]]; then
  if [[ ! -s "${RESULT_PATH}" ]]; then
    eval_status=124
  fi
fi
terminal_tmp="${TERMINAL_RECEIPT}.tmp.$$"
{
  printf 'schema_version\tzoomtoken_k100_tar50_interaction_terminal_v001\n'
  printf 'finished_at\t%s\ncommit\t%s\n' "$(date -Iseconds)" "${EXPECTED_COMMIT}"
  printf 'slurm_job_id\t%s\nslurm_job_name\t%s\n' "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}"
  printf 'train_exit_code\t%s\nfinal_ema_eval_exit_code\t%s\n' "${train_status}" "${eval_status}"
  printf 'primary_checkpoint\t%s\ncheckpoint_state\tepoch_59 state_dict_ema\n' "${CHECKPOINT}"
  printf 'primary_checkpoint_present\t%s\n' "$([[ -s "${CHECKPOINT}" ]] && printf true || printf false)"
  if [[ -s "${CHECKPOINT}" ]]; then
    printf 'primary_checkpoint_sha256\t%s\n' "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  fi
  printf 'official_result\t%s\n' "${RESULT_PATH}"
  printf 'official_result_present\t%s\n' "$([[ -s "${RESULT_PATH}" ]] && printf true || printf false)"
  if [[ -s "${RESULT_PATH}" ]]; then
    printf 'official_result_sha256\t%s\n' "$(sha256sum "${RESULT_PATH}" | awk '{print $1}')"
  fi
  printf 'training_log\t%s\nevaluation_log\t%s\n' "${TRAIN_LOG}" "${EVALUATION_LOG}"
  printf 'retry_resume_replacement\tfalse\ncost_measurement\tfalse\n'
} > "${terminal_tmp}"
mv "${terminal_tmp}" "${TERMINAL_RECEIPT}"

if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi
exit "${eval_status}"
