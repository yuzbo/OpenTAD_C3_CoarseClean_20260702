#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  ROOT="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/tools/bata/bafdr_k16_fullmatrix_train.py" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
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

CONDA_ENV="${BASE}/conda_envs/opentad/bin/activate"
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

formal_mode=0
case "${mode}" in
  precheck|train|eval|metrics) formal_mode=1 ;;
esac
if (( formal_mode )); then
  local_single_process="${BAFDR_ALLOW_SINGLE_PROCESS:-0}"
  if [[ "${local_single_process}" != "1" ]]; then
    if ! command -v module >/dev/null 2>&1; then
      echo "BA-FDR formal mode requires the cluster module command" >&2
      exit 2
    fi
    module load cuda/11.8
    module load miniforge3/24.11
    if [[ ! -f "${CONDA_ENV}" ]]; then
      echo "BA-FDR formal mode requires the OpenTAD conda environment: ${CONDA_ENV}" >&2
      exit 2
    fi
    # shellcheck disable=SC1091
    source "${CONDA_ENV}"
    command -v torchrun >/dev/null 2>&1 || {
      echo "BA-FDR formal mode requires torchrun after environment activation" >&2
      exit 2
    }
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("BA-FDR formal mode requires CUDA after environment activation")
PY
  elif [[ -f "${CONDA_ENV}" ]]; then
    # Local smoke/precheck may intentionally use the current Python environment.
    # shellcheck disable=SC1091
    source "${CONDA_ENV}"
  fi
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

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
    local master_port="${BAFDR_MASTER_PORT:-}"
    if [[ -z "${master_port}" && -n "${SLURM_JOB_ID:-}" ]]; then
      master_port="$((29500 + (SLURM_JOB_ID % 1000)))"
    fi
    master_port="${master_port:-29500}"
    torchrun --standalone --nproc_per_node="${BAFDR_NPROC_PER_NODE:-2}" \
      --master_port="${master_port}" \
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
