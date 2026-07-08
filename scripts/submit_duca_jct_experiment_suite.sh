#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_JCT_SUITE][FAIL] $*" >&2
  exit 1
}

log() {
  echo "[DUCA_JCT_SUITE] $*" >&2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
BRANCH="${BRANCH:-codex/gas-vt-stage23-detector-aware-20260706}"
SYNC_CODE="${SYNC_CODE:-1}"
SUBMIT_JOBS="${SUBMIT_JOBS:-1}"
SUBMIT_RETRY_SECONDS="${SUBMIT_RETRY_SECONDS:-300}"
SUBMIT_MAX_ATTEMPTS="${SUBMIT_MAX_ATTEMPTS:-288}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
RUN_TAG="${RUN_TAG:-duca_jct_suite_${COMMIT}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${RUN_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/${RUN_TAG}}"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
SCRIPT_ROOT="${RUN_ROOT}/sbatch"
FORMAL_X3D_ACTIONNESS_JSONL="${DUCA_X3D_ACTIONNESS_JSONL:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/trainfree_frozen_actionness/best_x3d_actionness.jsonl}"
FORMAL_X3D_MATERIALIZATION_SUMMARY="${FORMAL_X3D_MATERIALIZATION_SUMMARY:-${FORMAL_X3D_ACTIONNESS_JSONL%.jsonl}.materialization.json}"
DUCA_X3D_FORMAL_PROVIDER="${DUCA_X3D_FORMAL_PROVIDER:-x3d_xs}"
DUCA_X3D_FORMAL_FRAME_INTERVAL="${DUCA_X3D_FORMAL_FRAME_INTERVAL:-2}"
DUCA_X3D_FORMAL_CLIP_FRAMES="${DUCA_X3D_FORMAL_CLIP_FRAMES:-}"
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
if [[ ! -x "${PYTHON}" ]] && ! command -v "${PYTHON}" >/dev/null 2>&1; then
  if [[ -n "${PYTHON_FALLBACK:-}" ]]; then
    PYTHON="${PYTHON_FALLBACK}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${SCRIPT_ROOT}"

if [[ "${SYNC_CODE}" == "1" ]]; then
  log "syncing branch=${BRANCH} with git pull --ff-only"
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
  COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
fi

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

for path in \
  scripts/run_duca_online_official_adatad_backend_gpu1.sh \
  scripts/run_duca_must_dynamic_official_adatad_backend_gpu1.sh \
  scripts/run_duca_trainfree_x3d_interval_grid_gpu0.sh \
  scripts/run_duca_x3d_official_adatad_backend_gpu1.sh \
  scripts/run_duca_must_dynamic_x3d_official_adatad_backend_gpu1.sh \
  tools/bata/run_duca_jct_one_step_grad_proof.py \
  tools/bata/materialize_trainfree_x3d_actionness.py \
  tests/test_duca_joint_training_contract.py \
  tests/test_duca_jct_one_step_grad_proof.py \
  tests/test_trainfree_x3d_actionness_materialize.py; do
  require_file "${path}"
done

write_sbatch() {
  local path="$1"
  local job_name="$2"
  local body="$3"
  cat > "${path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${LOG_ROOT}/${job_name}_%j.out
#SBATCH --error=${LOG_ROOT}/${job_name}_%j.err

set -euo pipefail
cd "${REPO_ROOT}"
module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
export BASE="${BASE}"
export YUZIBO_ROOT="${YUZIBO_ROOT}"
export RUN_ROOT="${RUN_ROOT}"
export HOME="\${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="\${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export PYTHON="${PYTHON}"
mkdir -p "\${HOME}" "\${XDG_CACHE_HOME}" "\${XDG_CONFIG_HOME}"
${body}
EOF
  chmod +x "${path}"
}

submit_with_retry() {
  local name="$1"
  shift
  local attempt=1
  local output=""
  if [[ "${SUBMIT_JOBS}" != "1" ]]; then
    log "DRY submit ${name}: sbatch $*"
    echo ""
    return 0
  fi
  while (( attempt <= SUBMIT_MAX_ATTEMPTS )); do
    set +e
    output="$(sbatch "$@" 2>&1)"
    local status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
      log "submitted ${name}: ${output}"
      echo "${output}" | awk '{print $NF}'
      return 0
    fi
    if echo "${output}" | grep -q "AssocMaxSubmitJobLimit"; then
      log "submit limit for ${name}; attempt=${attempt}/${SUBMIT_MAX_ATTEMPTS}; sleeping ${SUBMIT_RETRY_SECONDS}s"
      sleep "${SUBMIT_RETRY_SECONDS}"
      attempt=$((attempt + 1))
      continue
    fi
    echo "${output}" >&2
    fail "sbatch failed for ${name}"
  done
  fail "submit_with_retry exhausted for ${name}"
}

tests_script="${SCRIPT_ROOT}/duca_jct_focused_tests.sbatch"
write_sbatch "${tests_script}" "duca_jct_tests" '
"${PYTHON}" -m py_compile \
  opentad/models/selectors/duca_online_frame_selector.py \
  opentad/models/detectors/single_stage.py \
  opentad/models/detectors/actionformer.py \
  tools/bata/materialize_trainfree_x3d_actionness.py \
  tools/bata/run_duca_jct_one_step_grad_proof.py \
  tools/bata/monitor_duca_jct_experiment_suite.py \
  tools/bata/collect_duca_jct_paper_evidence.py \
  tools/bata/validate_duca_official_adatad_backend.py \
  tools/bata/validate_duca_must_dynamic_official_adatad_backend.py
"${PYTHON}" tools/bata/validate_duca_official_adatad_backend.py --config configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py --max-budget 384
"${PYTHON}" tools/bata/validate_duca_must_dynamic_official_adatad_backend.py --config configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py --max-budget 384
"${PYTHON}" tools/bata/run_duca_jct_one_step_grad_proof.py \
  --fixed-config configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py \
  --must-config configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py \
  --output-json "${RUN_ROOT}/duca_jct_one_step_grad_proof.json" \
  --proof-temporal-len 16 \
  --proof-budget 16 \
  --proof-budget-min 4 \
  --proof-budget-target 8 \
  --proof-budget-multiple 4 \
  --proof-spatial-size 16 \
  --proof-hidden-dim 16
"${PYTHON}" -m pytest \
  tests/test_duca_joint_training_contract.py \
  tests/test_duca_jct_one_step_grad_proof.py \
  tests/test_duca_online_coarse_probe_actionness.py \
  tests/test_duca_online_precheck_config.py \
  tests/test_duca_jct_suite_monitor.py \
  tests/test_duca_jct_paper_evidence.py \
  tests/test_trainfree_x3d_actionness_materialize.py \
  -q
'

fixed_script="${SCRIPT_ROOT}/duca_jct_fixed384_fulltrain.sbatch"
write_sbatch "${fixed_script}" "duca_jct_384" "
export PRECHECK_ONLY=0
export FULLTRAIN_CANDIDATE=1
export RUN_TAG=${RUN_TAG}_duca384
export RUN_DIR=${RUN_ROOT}/duca384_jct/logs
export WORK_DIR=${RUN_ROOT}/duca384_jct/work_dir
export MASTER_PORT=30301
bash scripts/run_duca_online_official_adatad_backend_gpu1.sh
"

must_script="${SCRIPT_ROOT}/duca_jct_must_dynamic_fulltrain.sbatch"
write_sbatch "${must_script}" "duca_jct_must" "
export PRECHECK_ONLY=0
export FULLTRAIN_CANDIDATE=1
export RUN_TAG=${RUN_TAG}_duca_must
export RUN_DIR=${RUN_ROOT}/duca_must_jct/logs
export WORK_DIR=${RUN_ROOT}/duca_must_jct/work_dir
export MASTER_PORT=30311
bash scripts/run_duca_must_dynamic_official_adatad_backend_gpu1.sh
"

x3d_grid_script="${SCRIPT_ROOT}/duca_x3d_interval_grid.sbatch"
write_sbatch "${x3d_grid_script}" "duca_x3d_grid" "
export RUN_TAG=${RUN_TAG}_x3d_grid
export GRID_ROOT=${RUN_ROOT}/x3d_grid
export DUCA_X3D_ACTIONNESS_JSONL=${FORMAL_X3D_ACTIONNESS_JSONL}
export FORMAL_X3D_MATERIALIZATION_SUMMARY=${FORMAL_X3D_MATERIALIZATION_SUMMARY}
export DUCA_X3D_FORMAL_PROVIDER=${DUCA_X3D_FORMAL_PROVIDER}
export DUCA_X3D_FORMAL_FRAME_INTERVAL=${DUCA_X3D_FORMAL_FRAME_INTERVAL}
export DUCA_X3D_FORMAL_CLIP_FRAMES=${DUCA_X3D_FORMAL_CLIP_FRAMES}
export MATERIALIZE_FORMAL_JSONL=1
bash scripts/run_duca_trainfree_x3d_interval_grid_gpu0.sh
"

x3d_fixed_script="${SCRIPT_ROOT}/duca_x3d_fixed384_fulltrain.sbatch"
write_sbatch "${x3d_fixed_script}" "duca_x3d_384" "
export PRECHECK_ONLY=0
export FULLTRAIN_CANDIDATE=1
export RUN_TAG=${RUN_TAG}_x3d_duca384
export RUN_DIR=${RUN_ROOT}/x3d_duca384/logs
export WORK_DIR=${RUN_ROOT}/x3d_duca384/work_dir
export MASTER_PORT=30321
export DUCA_X3D_ACTIONNESS_JSONL=${FORMAL_X3D_ACTIONNESS_JSONL}
bash scripts/run_duca_x3d_official_adatad_backend_gpu1.sh
"

x3d_must_script="${SCRIPT_ROOT}/duca_x3d_must_dynamic_fulltrain.sbatch"
write_sbatch "${x3d_must_script}" "duca_x3d_must" "
export PRECHECK_ONLY=0
export FULLTRAIN_CANDIDATE=1
export RUN_TAG=${RUN_TAG}_x3d_must
export RUN_DIR=${RUN_ROOT}/x3d_must/logs
export WORK_DIR=${RUN_ROOT}/x3d_must/work_dir
export MASTER_PORT=30331
export DUCA_X3D_ACTIONNESS_JSONL=${FORMAL_X3D_ACTIONNESS_JSONL}
bash scripts/run_duca_must_dynamic_x3d_official_adatad_backend_gpu1.sh
"

tests_job="$(submit_with_retry duca_jct_tests "${tests_script}")"
dep_args=()
if [[ -n "${tests_job}" ]]; then
  dep_args=(--dependency=afterok:${tests_job})
fi
fixed_job="$(submit_with_retry duca_jct_384 "${dep_args[@]}" "${fixed_script}")"
must_job="$(submit_with_retry duca_jct_must "${dep_args[@]}" "${must_script}")"
x3d_grid_job="$(submit_with_retry duca_x3d_grid "${dep_args[@]}" "${x3d_grid_script}")"
x3d_dep_args=()
if [[ -n "${x3d_grid_job}" ]]; then
  x3d_dep_args=(--dependency=afterok:${x3d_grid_job})
fi
x3d_fixed_job="$(submit_with_retry duca_x3d_384 "${x3d_dep_args[@]}" "${x3d_fixed_script}")"
x3d_must_job="$(submit_with_retry duca_x3d_must "${x3d_dep_args[@]}" "${x3d_must_script}")"

SUMMARY_JSON="${RUN_ROOT}/deployment_summary.json"
"${PYTHON}" - "${SUMMARY_JSON}" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "duca_jct_experiment_suite_deployment_v1",
    "commit": "${COMMIT}",
    "branch": "${BRANCH}",
    "repo": "${REPO_ROOT}",
    "run_root": "${RUN_ROOT}",
    "duca_jct_one_step_grad_proof": "${RUN_ROOT}/duca_jct_one_step_grad_proof.json",
    "formal_x3d_actionness_jsonl": "${FORMAL_X3D_ACTIONNESS_JSONL}",
    "formal_x3d_materialization_summary": "${FORMAL_X3D_MATERIALIZATION_SUMMARY}",
    "duca_jct_tests_job": "${tests_job}",
    "duca384_job": "${fixed_job}",
    "duca_must_job": "${must_job}",
    "x3d_grid_job": "${x3d_grid_job}",
    "x3d_duca384_job": "${x3d_fixed_job}",
    "x3d_must_job": "${x3d_must_job}",
    "x3d_downstream_dependency": "afterok:${x3d_grid_job}",
    "selection_policy": "pre_registered",
    "x3d_formal_provider": "${DUCA_X3D_FORMAL_PROVIDER}",
    "x3d_formal_frame_interval": "${DUCA_X3D_FORMAL_FRAME_INTERVAL}",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

log "deployment summary: ${SUMMARY_JSON}"
