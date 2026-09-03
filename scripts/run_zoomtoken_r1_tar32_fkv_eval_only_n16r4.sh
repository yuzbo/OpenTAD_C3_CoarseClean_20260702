#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_R1_TAR32_EVAL][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_R1_TAR32_EVAL][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
EXPECTED_COMMIT="b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7"
EXPECTED_CHECKPOINT_SHA="fc70557ef00788f8e788d59464d8c392943638c446d949d586fefc68c6d9390b"
EXPECTED_CONFIG_SHA="b372d759c402bd82dbc758faa4b69e89351d757e57c8f76d1369f5fee7edc8ec"
EXPECTED_ANNOTATION_SHA="ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad"
EXPECTED_CLASS_MAP_SHA="a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31"
EXPECTED_PRETRAINED_SHA="4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
EXPECTED_JOB_NAME="zt-r1-tar32-eval-b0a1"

ROOT="${ZOOMTOKEN_TAR32_SOURCE_ROOT:-${BASE}/projects/zoomtoken_r1_tar32_fkv_src_b0a1ca11}"
TRAIN_ROOT="${ZOOMTOKEN_TAR32_TRAIN_ROOT:-${BASE}/projects/zoomtoken_r1_tar32_fkv_v001_seed42_20260830}"
RUN_ROOT="${ZOOMTOKEN_TAR32_EVAL_ROOT:-${BASE}/projects/zoomtoken_r1_tar32_fkv_eval_only_b0a1ca11_seed42_20260829}"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py"
CHECKPOINT="${TRAIN_ROOT}/cells/r1_tar32_fkv_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth"
CELL_ROOT="${RUN_ROOT}/gpu2_id0"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"
LAUNCH_RECEIPT="${RUN_ROOT}/launch_receipt.tsv"
TERMINAL_RECEIPT="${RUN_ROOT}/terminal_receipt.tsv"
EVALUATION_LOG="${RUN_ROOT}/evaluation.log"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'evaluation-only completion requires a Slurm allocation'
[[ "${SLURM_JOB_NAME:-}" == "${EXPECTED_JOB_NAME}" ]] || fail "formal JobName must be ${EXPECTED_JOB_NAME}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'evaluation-only completion requires exactly two Slurm-visible GPUs'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || fail 'evaluation-only completion requires --cpus-per-task=8'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source snapshot is not clean'

for path in "${CONFIG}" "${CHECKPOINT}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || fail 'result root must be outside the source checkout'
[[ ! -e "${RUN_ROOT}" ]] || fail 'fresh evaluation-only result root already exists'

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "SHA-256 mismatch: ${path}"
}
check_sha "${EXPECTED_CHECKPOINT_SHA}" "${CHECKPOINT}"
check_sha "${EXPECTED_CONFIG_SHA}" "${CONFIG}"
check_sha "${EXPECTED_ANNOTATION_SHA}" "${ANNOTATION}"
check_sha "${EXPECTED_CLASS_MAP_SHA}" "${CLASS_MAP}"
check_sha "${EXPECTED_PRETRAINED_SHA}" "${PRETRAINED}"
[[ "$(find -L "${VIDEO_ROOT}" -type f -name '*.mp4' | wc -l | tr -d ' ')" == "411" ]] || \
  fail 'canonical video inventory is not 411 MP4 files'

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
  printf 'schema_version\tzoomtoken_r1_tar32_fkv_eval_only_launch_v001\n'
  printf 'created_at\t%s\n' "$(date -Iseconds)"
  printf 'commit\t%s\n' "${EXPECTED_COMMIT}"
  printf 'config\t%s\n' "${CONFIG}"
  printf 'config_sha256\t%s\n' "${EXPECTED_CONFIG_SHA}"
  printf 'checkpoint\t%s\n' "${CHECKPOINT}"
  printf 'checkpoint_sha256\t%s\n' "${EXPECTED_CHECKPOINT_SHA}"
  printf 'checkpoint_state\tepoch_59 state_dict_ema\n'
  printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_job_name\t%s\n' "${SLURM_JOB_NAME}"
  printf 'seed\t42\nrank_count\t2\ntraining_or_resume\tfalse\nparameter_update\tfalse\n'
  printf 'population\tcanonical validation: 211 videos / 792 ordered windows\n'
  printf 'evaluator\tofficial evaluator plus configured Soft-NMS\n'
  printf 'route\tR1-TAR32-FKV [64,32]x6; K/V and Adapter remain K64\n'
} > "${launch_tmp}"
mv "${launch_tmp}" "${LAUNCH_RECEIPT}"

set +e
torchrun --nnodes=1 --nproc_per_node=2 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  --rdzv_id="zoomtoken-r1-tar32-eval-${SLURM_JOB_ID}-seed42" \
  tools/test.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" --seed 42 --id 0 \
  --cfg-options \
  "work_dir=${RUN_ROOT}" \
  "zoomtoken_p1_config.source_commit=${EXPECTED_COMMIT}" \
  "dataset.test.ann_file=${ANNOTATION}" \
  "dataset.test.class_map=${CLASS_MAP}" \
  "dataset.test.data_path=${VIDEO_ROOT}" \
  "dataset.test.subset_name=validation" \
  "evaluation.ground_truth_filename=${ANNOTATION}" \
  "model.backbone.custom.pretrain=${PRETRAINED}" \
  "post_processing.save_dict=True" \
  2>&1 | tee "${EVALUATION_LOG}"
torchrun_status=${PIPESTATUS[0]}
set -e

result_path="${CELL_ROOT}/result_detection.json"
if [[ "${torchrun_status}" -eq 0 ]]; then
  [[ -s "${result_path}" ]] || fail 'official evaluation completed without result_detection.json'
fi
terminal_tmp="${TERMINAL_RECEIPT}.tmp.$$"
{
  printf 'schema_version\tzoomtoken_r1_tar32_fkv_eval_only_terminal_v001\n'
  printf 'finished_at\t%s\n' "$(date -Iseconds)"
  printf 'commit\t%s\n' "${EXPECTED_COMMIT}"
  printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'slurm_job_name\t%s\n' "${SLURM_JOB_NAME}"
  printf 'torchrun_exit_code\t%s\n' "${torchrun_status}"
  printf 'checkpoint_sha256\t%s\n' "${EXPECTED_CHECKPOINT_SHA}"
  printf 'training_or_resume\tfalse\nparameter_update\tfalse\n'
  printf 'official_result\t%s\n' "${result_path}"
  if [[ -s "${result_path}" ]]; then
    printf 'official_result_sha256\t%s\n' "$(sha256sum "${result_path}" | awk '{print $1}')"
  fi
  printf 'evaluation_log\t%s\n' "${EVALUATION_LOG}"
  printf 'evaluation_log_sha256\t%s\n' "$(sha256sum "${EVALUATION_LOG}" | awk '{print $1}')"
  printf 'runtime_route_contract\tall official evaluation forwards completed under candidate in-forward route assertions\n'
} > "${terminal_tmp}"
mv "${terminal_tmp}" "${TERMINAL_RECEIPT}"
exit "${torchrun_status}"
