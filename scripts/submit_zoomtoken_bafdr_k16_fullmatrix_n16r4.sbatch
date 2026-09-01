#!/usr/bin/env bash
#SBATCH --job-name=zt-bafdr-k16-dag
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -n "${YUZIBO_ROOT:-}" ]]; then
  BASE="${YUZIBO_ROOT}"
elif [[ -d "/data/run01/sczc063/yuzibo" ]]; then
  BASE="/data/run01/sczc063/yuzibo"
else
  BASE="${PROJECT_DIR}/tmp/bafdr_local"
fi
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_compute}"
export ZOOMTOKEN_RUN_ROOT="${RUN_ROOT}"
LOG_DIR="${BASE}/slurm_logs"
MANIFEST_DIR="${RUN_ROOT}/manifest"
mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

CONDA_ENV="${BASE}/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export BAFDR_ALLOW_DIRTY="${BAFDR_ALLOW_DIRTY:-0}"

bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh validate

if [[ $# -ge 1 && -n "$1" ]]; then
  seed_list=("$@")
elif [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  seed_list=(${SEEDS})
else
  seed_list=(4407 4408 4409)
fi
if [[ "${seed_list[*]}" != "4407 4408 4409" ]]; then
  echo "complete BA-FDR deployment requires seeds 4407 4408 4409" >&2
  exit 2
fi

submit_job() {
  local name="$1"
  local dependency="$2"
  local gres="$3"
  local cpus="$4"
  local time_limit="$5"
  local command="$6"
  local array_spec="${7:-}"
  local dep_args=()
  local gres_args=()
  local array_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args=(--dependency="${dependency}")
  fi
  if [[ -n "${gres}" && "${gres}" != "none" ]]; then
    if [[ "${gres}" == *"2"* ]]; then
      gres_args=(--gpus=2)
    else
      gres_args=(--gpus=1)
    fi
  else
    # N16R4 rejects jobs without an explicit GPU request, even for ledger-only stages.
    gres_args=(--gpus=1)
  fi
  if [[ -n "${array_spec}" ]]; then
    array_args=("${array_spec}")
  fi
  sbatch --parsable \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    "${gres_args[@]}" \
    --cpus-per-task="${cpus}" \
    --time="${time_limit}" \
    --job-name="${name}" \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" \
    "${dep_args[@]}" \
    "${array_args[@]}" \
    --wrap "cd '${PROJECT_DIR}' && ${command}"
}

train_jobs_tsv="${MANIFEST_DIR}/slurm_train_jobs.tsv"
eval_jobs_tsv="${MANIFEST_DIR}/slurm_eval_jobs.tsv"
train_job="$(submit_job "zt-bafdr-train-e60b4399" "" "gpu:2" 8 "72:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_array_task.sh train \"\${SLURM_ARRAY_TASK_ID}\"" --array=0-20)"
eval_job="$(submit_job "zt-bafdr-eval-e60b4399" "afterok:${train_job}" "gpu:2" 8 "24:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_array_task.sh eval \"\${SLURM_ARRAY_TASK_ID}\"" --array=0-20)"
c_exec_job="$(submit_job "zt-bafdr-cexec" "afterok:${eval_job}" "none" 1 "02:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh c_exec")"
metrics_job="$(submit_job "zt-bafdr-metrics" "afterok:${eval_job}:${c_exec_job}" "gpu:2" 8 "24:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh metrics")"
summary_job="$(submit_job "zt-bafdr-summary" "afterok:${metrics_job}" "none" 1 "02:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh summary-strict")"

printf 'all\t%s\n' "${train_job}" > "${train_jobs_tsv}"
printf 'all\t%s\n' "${eval_job}" > "${eval_jobs_tsv}"

python - <<PY
import json
import time
from pathlib import Path

def read_tsv(path):
    data = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        data[key] = value
    return dict(sorted(data.items()))

receipt = {
    "schema_version": "ZOOMTOKEN-BA-FDR-K16-SLURM-DAG-v001",
    "protocol_id": "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    "project_dir": ${PROJECT_DIR@Q},
    "run_root": ${RUN_ROOT@Q},
    "train_array_job": ${train_job@Q},
    "eval_array_job": ${eval_job@Q},
    "train_jobs": read_tsv(${train_jobs_tsv@Q}),
    "eval_jobs": read_tsv(${eval_jobs_tsv@Q}),
    "c_exec_job": ${c_exec_job@Q},
    "metrics_job": ${metrics_job@Q},
    "summary_job": ${summary_job@Q},
    "submitted_at": time.time(),
}
path = Path(${MANIFEST_DIR@Q}) / "slurm_dag_receipt.json"
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[BA-FDR] Slurm DAG receipt written to {path}")
print("::bafdr-slurm-dag{submission_receipt=\"" + str(path) + "\", summary_job=\"" + ${summary_job@Q} + "\"}")
PY
