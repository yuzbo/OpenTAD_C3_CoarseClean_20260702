#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ADATAD_OFFICIAL_REPRODUCTION][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ADATAD_OFFICIAL_REPRODUCTION][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
EXPECTED_COMMIT="01c58b9f2370e914150cf94d392208a4e211c053"
ROOT="${ADATAD_OFFICIAL_SOURCE_ROOT:-${BASE}/projects/official_adatad_reproduction_01c58b9}"
RUN_ROOT="${ADATAD_OFFICIAL_RUN_ROOT:-${BASE}/projects/official_adatad_reproduction_run_01c58b9_seed42}"
CONFIG="${ROOT}/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'the official reproduction requires a Slurm allocation'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 1 ]] || \
  fail 'the official reproduction requires exactly one Slurm-visible GPU'
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || \
  fail 'the official reproduction requires --cpus-per-task=8'

[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'untouched official source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'untouched official source is not clean'
for path in "${CONFIG}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file does not exist: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail "canonical THUMOS14 video root does not exist: ${VIDEO_ROOT}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'official result root leaves the remote write boundary' ;;
esac
[[ "${RUN_ROOT}/" != "${ROOT}/"* ]] || fail 'official result root must be outside the source checkout'
if [[ -e "${RUN_ROOT}" ]]; then
  [[ -d "${RUN_ROOT}" ]] || fail 'official result root exists and is not a directory'
  [[ -z "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    fail 'official result root must be empty before the untouched reproduction'
else
  mkdir -p "${RUN_ROOT}"
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
  --rdzv_id="adatad-official-${SLURM_JOB_ID}-seed42" \
  tools/train.py "${CONFIG}" --seed 42 --id 0 \
  --cfg-options \
  "work_dir=${RUN_ROOT}" \
  "dataset.train.ann_file=${ANNOTATION}" \
  "dataset.train.class_map=${CLASS_MAP}" \
  "dataset.train.data_path=${VIDEO_ROOT}" \
  "dataset.val.ann_file=${ANNOTATION}" \
  "dataset.val.class_map=${CLASS_MAP}" \
  "dataset.val.data_path=${VIDEO_ROOT}" \
  "dataset.test.ann_file=${ANNOTATION}" \
  "dataset.test.class_map=${CLASS_MAP}" \
  "dataset.test.data_path=${VIDEO_ROOT}" \
  "evaluation.ground_truth_filename=${ANNOTATION}" \
  "model.backbone.custom.pretrain=${PRETRAINED}"
