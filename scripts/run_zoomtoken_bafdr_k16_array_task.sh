#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?phase is required (train or eval)}"
ARRAY_IDX="${2:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${ARRAY_IDX}" ]]; then
  echo "array task index is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_compute}"
WORK_DIR_ROOT="${RUN_ROOT}/work_dirs"

if [[ "${PHASE}" != "train" && "${PHASE}" != "eval" ]]; then
  echo "unsupported phase: ${PHASE}" >&2
  exit 2
fi

cell_json="$(python tools/bata/bafdr_k16_fullmatrix.py \
  --repo-root "${ROOT}" --array-idx "${ARRAY_IDX}" --array-cell-json)"
config_path="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["config_path"])' "${cell_json}")"
arm="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["arm"])' "${cell_json}")"
seed="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["seed"])' "${cell_json}")"
stem="$(basename "${config_path}" .py)"
work_dir="${WORK_DIR_ROOT}/${stem}"
mkdir -p "${work_dir}"

if [[ "${PHASE}" == "train" && "${arm}" == "BAFDR-K16-FULL" ]]; then
  teacher_ckpt="${WORK_DIR_ROOT}/bafdr_k16_d160_seed${seed}/checkpoint/epoch_59.pth"
  for _ in $(seq 1 4320); do
    [[ -f "${teacher_ckpt}" ]] && break
    sleep 60
  done
  if [[ ! -f "${teacher_ckpt}" ]]; then
    echo "D160 teacher checkpoint did not appear for seed ${seed}: ${teacher_ckpt}" >&2
    exit 1
  fi
fi

if [[ "${PHASE}" == "train" ]]; then
  exec bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train "${config_path}"
fi

exec bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh eval "${config_path}"
