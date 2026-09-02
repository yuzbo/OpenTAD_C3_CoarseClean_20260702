#!/usr/bin/env bash
#SBATCH --job-name=zt-bafdr-k16-dag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
LOG_DIR="${BASE}/slurm_logs"
MANIFEST_DIR="${RUN_ROOT}/manifest"
mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"

cd "${PROJECT_DIR}"

source /etc/profile

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

seed_list=(4407 4408 4409)
arm_list=(
  "D160"
  "G96"
  "U128-ALL48-A0"
  "U16-UNIFORM-A0"
  "BAFDR-K16-LATE"
  "BAFDR-K16-NOKD"
  "BAFDR-K16-FULL"
)

slug_for_arm() {
  case "$1" in
    D160) echo "d160" ;;
    G96) echo "g96" ;;
    U128-ALL48-A0) echo "u128_all48_a0" ;;
    U16-UNIFORM-A0) echo "u16_uniform_a0" ;;
    BAFDR-K16-LATE) echo "late" ;;
    BAFDR-K16-NOKD) echo "nokd" ;;
    BAFDR-K16-FULL) echo "full" ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

submit_job() {
  local name="$1"
  local dependency="$2"
  local gres="$3"
  local cpus="$4"
  local time_limit="$5"
  local command="$6"
  local dep_args=()
  local gres_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args=(--dependency="${dependency}")
  fi
  if [[ -n "${gres}" && "${gres}" != "none" ]]; then
    gres_args=(--gres="${gres}")
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
    --wrap "cd '${PROJECT_DIR}' && ${command}"
}

declare -A train_jobs
declare -A eval_jobs
all_terminal_deps=()

for seed in "${seed_list[@]}"; do
  d160_cfg="configs/adatad/thumos/bafdr_k16_d160_seed${seed}.py"
  d160_job="$(submit_job "zt-bafdr-train-d160-${seed}" "" "gpu:2" 8 "72:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train '${d160_cfg}'")"
  train_jobs["D160:${seed}"]="${d160_job}"
done

for arm in "${arm_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    if [[ "${arm}" == "D160" ]]; then
      continue
    fi
    slug="$(slug_for_arm "${arm}")"
    cfg="configs/adatad/thumos/bafdr_k16_${slug}_seed${seed}.py"
    dep=""
    if [[ "${arm}" == "BAFDR-K16-FULL" ]]; then
      dep="afterok:${train_jobs["D160:${seed}"]}"
    fi
    safe_arm="${slug//_/-}"
    job="$(submit_job "zt-bafdr-train-${safe_arm}-${seed}" "${dep}" "gpu:2" 8 "72:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train '${cfg}'")"
    train_jobs["${arm}:${seed}"]="${job}"
  done
done

for arm in "${arm_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    slug="$(slug_for_arm "${arm}")"
    cfg="configs/adatad/thumos/bafdr_k16_${slug}_seed${seed}.py"
    train_job="${train_jobs["${arm}:${seed}"]}"
    safe_arm="${slug//_/-}"
    eval_job="$(submit_job "zt-bafdr-eval-${safe_arm}-${seed}" "afterok:${train_job}" "gpu:2" 8 "24:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh eval '${cfg}'")"
    eval_jobs["${arm}:${seed}"]="${eval_job}"
    all_terminal_deps+=("${eval_job}")
  done
done

joined_eval_deps="$(IFS=:; echo "${all_terminal_deps[*]}")"
c_exec_job="$(submit_job "zt-bafdr-cexec" "afterok:${joined_eval_deps}" "none" 1 "02:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh c_exec")"
metrics_job="$(submit_job "zt-bafdr-metrics" "afterok:${joined_eval_deps}:${c_exec_job}" "none" 1 "08:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh metrics")"
summary_job="$(submit_job "zt-bafdr-summary" "afterok:${metrics_job}" "none" 1 "02:00:00" "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh summary-strict")"

train_jobs_tsv="${MANIFEST_DIR}/slurm_train_jobs.tsv"
eval_jobs_tsv="${MANIFEST_DIR}/slurm_eval_jobs.tsv"
: > "${train_jobs_tsv}"
: > "${eval_jobs_tsv}"
for key in "${!train_jobs[@]}"; do
  printf '%s\t%s\n' "${key}" "${train_jobs[${key}]}" >> "${train_jobs_tsv}"
done
for key in "${!eval_jobs[@]}"; do
  printf '%s\t%s\n' "${key}" "${eval_jobs[${key}]}" >> "${eval_jobs_tsv}"
done

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
