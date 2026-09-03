#!/usr/bin/env bash
#SBATCH --job-name=zt-bafdr-k16-dag
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/slurm_logs/%x_%j.err

source /etc/profile
set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_compute}"
export ZOOMTOKEN_RUN_ROOT="${RUN_ROOT}"
LOG_DIR="${BASE}/slurm_logs"
MANIFEST_DIR="${RUN_ROOT}/manifest"
mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"
cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
CONDA_ENV="${BASE}/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}"
fi
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export BAFDR_ALLOW_DIRTY="${BAFDR_ALLOW_DIRTY:-0}"

bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh validate

RELEASE_TAG="$(git rev-parse --short HEAD)"

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
  local gpus="$3"
  local cpus="$4"
  local time_limit="$5"
  local command="$6"
  local dep_args=()
  [[ -n "${dependency}" ]] && dep_args=(--dependency="${dependency}")
  sbatch --parsable \
    --partition=gpu \
    --nodes=1 \
    --ntasks=1 \
    --gpus="${gpus}" \
    --cpus-per-task="${cpus}" \
    --time="${time_limit}" \
    --job-name="${name}" \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" \
    "${dep_args[@]}" \
    --wrap "cd '${PROJECT_DIR}' && export ZOOMTOKEN_RUN_ROOT='${RUN_ROOT}' && ${command}"
}

declare -A train_jobs
arm_list=(D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL)
for arm in "${arm_list[@]}"; do
  dep=""
  [[ "${arm}" == "BAFDR-K16-FULL" ]] && dep="afterok:${train_jobs[D160]}"
  safe_arm="${arm//_/-}"
  train_jobs["${arm}"]="$(submit_job "zt-bafdr-train-${safe_arm}-${RELEASE_TAG}" "${dep}" 2 8 "168:00:00" \
    "bash scripts/run_zoomtoken_bafdr_k16_arm_batch.sh '${arm}' train")"
done

train_deps=()
for arm in "${arm_list[@]}"; do train_deps+=("${train_jobs[${arm}]}"); done
joined_train_deps="$(IFS=:; echo "${train_deps[*]}")"
eval_job="$(submit_job "zt-bafdr-eval-all-${RELEASE_TAG}" "afterok:${joined_train_deps}" 2 8 "168:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_arm_batch.sh ALL eval-all")"
c_exec_job="$(submit_job "zt-bafdr-cexec-${RELEASE_TAG}" "afterok:${eval_job}" 1 1 "02:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh c_exec")"
metrics_job="$(submit_job "zt-bafdr-metrics-${RELEASE_TAG}" "afterok:${eval_job}:${c_exec_job}" 2 8 "168:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh metrics")"
summary_job="$(submit_job "zt-bafdr-summary-${RELEASE_TAG}" "afterok:${metrics_job}" 1 1 "02:00:00" \
  "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh summary-strict")"

train_jobs_tsv="${MANIFEST_DIR}/slurm_train_jobs.tsv"
eval_jobs_tsv="${MANIFEST_DIR}/slurm_eval_jobs.tsv"
{
  for arm in "${arm_list[@]}"; do printf '%s\t%s\n' "${arm}" "${train_jobs[${arm}]}"; done
} > "${train_jobs_tsv}"
printf 'all\t%s\n' "${eval_job}" > "${eval_jobs_tsv}"

python - <<PY
import json
import time
from pathlib import Path

def read_tsv(path):
    return dict(sorted(line.split("\t", 1) for line in Path(path).read_text().splitlines()))

receipt = {
    "schema_version": "ZOOMTOKEN-BA-FDR-K16-SLURM-DAG-v002",
    "protocol_id": "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    "project_dir": ${PROJECT_DIR@Q},
    "run_root": ${RUN_ROOT@Q},
    "train_jobs": read_tsv(${train_jobs_tsv@Q}),
    "eval_jobs": read_tsv(${eval_jobs_tsv@Q}),
    "eval_job": ${eval_job@Q},
    "c_exec_job": ${c_exec_job@Q},
    "metrics_job": ${metrics_job@Q},
    "summary_job": ${summary_job@Q},
    "submitted_at": time.time(),
}
path = Path(${MANIFEST_DIR@Q}) / "slurm_dag_receipt.json"
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[BA-FDR] Slurm DAG receipt written to {path}")
PY
