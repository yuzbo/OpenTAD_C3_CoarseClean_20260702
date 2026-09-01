#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ -n "${YUZIBO_ROOT:-}" ]]; then
  BASE="${YUZIBO_ROOT}"
elif [[ -d "/data/run01/sczc063/yuzibo" ]]; then
  BASE="/data/run01/sczc063/yuzibo"
else
  BASE="${ROOT}/tmp/bafdr_local"
fi
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_compute}"
MANIFEST_DIR="${RUN_ROOT}/manifest"
WORK_DIR_ROOT="${RUN_ROOT}/work_dirs"
PRED_DIR="${RUN_ROOT}/predictions"
EVAL_DIR="${RUN_ROOT}/evaluation"
PROFILE_DIR="${RUN_ROOT}/profile"

mkdir -p "${MANIFEST_DIR}" "${WORK_DIR_ROOT}" "${PRED_DIR}" "${EVAL_DIR}" "${PROFILE_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

CONDA_ENV="${BASE}/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mode="${1:-validate}"
if [[ -f "${mode}" || "${mode}" == *.py ]]; then
  CONFIG="${mode}"
  mode="train"
else
  CONFIG="${2:-}"
fi

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  mode="precheck"
  if [[ -z "${CONFIG}" ]]; then
    CONFIG="${PRECHECK_CONFIG:-configs/adatad/thumos/bafdr_k16_d160_seed4407.py}"
  fi
fi

allow_dirty_flag=()
if [[ "${BAFDR_ALLOW_DIRTY:-0}" == "1" ]]; then
  allow_dirty_flag=(--allow-dirty)
fi

cell_work_dir() {
  local config="$1"
  local stem
  stem="$(basename "${config}" .py)"
  printf '%s/%s' "${WORK_DIR_ROOT}" "${stem}"
}

run_torch_cell() {
  local config="$1"
  shift
  local work_dir
  work_dir="$(cell_work_dir "${config}")"
  mkdir -p "${work_dir}"
  if [[ "${BAFDR_ALLOW_SINGLE_PROCESS:-0}" == "1" ]]; then
    python tools/bata/bafdr_k16_fullmatrix_train.py "${config}" \
      --work-dir "${work_dir}" \
      --allow-single-process \
      "$@"
  else
    torchrun --standalone --nproc_per_node="${BAFDR_NPROC_PER_NODE:-2}" \
      tools/bata/bafdr_k16_fullmatrix_train.py "${config}" \
      --work-dir "${work_dir}" \
      "$@"
  fi
}

case "${mode}" in
  validate)
    python tools/bata/bafdr_k16_fullmatrix.py \
      --repo-root "${ROOT}" \
      --output "${MANIFEST_DIR}/submission_receipt.json" \
      "${allow_dirty_flag[@]}"
    ;;
  precheck)
    if [[ -z "${CONFIG}" ]]; then
      echo "precheck mode requires a config path" >&2
      exit 2
    fi
    run_torch_cell "${CONFIG}" --precheck-only
    ;;
  train)
    if [[ -z "${CONFIG}" ]]; then
      echo "train mode requires a config path" >&2
      exit 2
    fi
    run_torch_cell "${CONFIG}"
    ;;
  eval)
    if [[ -z "${CONFIG}" ]]; then
      echo "eval mode requires a config path" >&2
      exit 2
    fi
    work_dir="$(cell_work_dir "${CONFIG}")"
    run_torch_cell "${CONFIG}" \
      --eval-only \
      --prediction-only \
      --checkpoint "${work_dir}/checkpoint/epoch_59.pth"
    ;;
  metrics)
    python tools/bata/bafdr_k16_fullmatrix.py \
      --repo-root "${ROOT}" \
      --seal-predictions \
      --work-dir-root "${WORK_DIR_ROOT}" \
      --output "${MANIFEST_DIR}/prediction_seal_receipt.json"
    for idx in $(seq 0 20); do
      config="$(python tools/bata/bafdr_k16_fullmatrix.py --repo-root "${ROOT}" --array-idx "${idx}")"
      work_dir="$(cell_work_dir "${config}")"
      run_torch_cell "${config}" \
        --eval-only \
        --open-metrics \
        --checkpoint "${work_dir}/checkpoint/epoch_59.pth"
    done
    ;;
  c_exec)
    python tools/bata/bafdr_k16_fullmatrix_c_exec.py \
      --output "${PROFILE_DIR}/c_exec_summary.json"
    ;;
  summary)
    python tools/bata/bafdr_k16_fullmatrix.py \
      --repo-root "${ROOT}" \
      --summary \
      --work-dir-root "${WORK_DIR_ROOT}" \
      --summary-output "${EVAL_DIR}/matrix_summary.json" \
      "${allow_dirty_flag[@]}"
    ;;
  summary-strict)
    python tools/bata/bafdr_k16_fullmatrix.py \
      --repo-root "${ROOT}" \
      --summary \
      --work-dir-root "${WORK_DIR_ROOT}" \
      --summary-output "${EVAL_DIR}/matrix_summary.json" \
      --require-complete \
      "${allow_dirty_flag[@]}"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    echo "Usage: $0 [validate|precheck|train|eval|metrics|c_exec|summary|summary-strict] [config.py]" >&2
    exit 2
    ;;
esac
