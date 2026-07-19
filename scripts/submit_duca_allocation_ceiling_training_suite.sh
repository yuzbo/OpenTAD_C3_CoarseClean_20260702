#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_SUBMIT][FAIL] $*" >&2
  exit 1
}

resolve_target_cluster() {
  local cluster="${DUCA_TARGET_CLUSTER:-${SLURM_CLUSTER_NAME:-}}"
  if [[ -z "${cluster}" ]]; then
    command -v scontrol >/dev/null 2>&1 || fail \
      "set DUCA_TARGET_CLUSTER or run where scontrol is available"
    cluster="$(
      scontrol show config | awk -F= \
        '/^[[:space:]]*ClusterName[[:space:]]*=/ {
          gsub(/[[:space:]]/, "", $2)
          print $2
          exit
        }'
    )"
  fi
  [[ "${cluster}" =~ ^[A-Za-z0-9._-]+$ ]] || fail \
    "target cluster is invalid: ${cluster}"
  printf '%s\n' "${cluster}"
}

require_generated_script_safe() {
  local label="$1"
  local value="$2"
  [[ "${value}" != *"'"* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || fail "${label} cannot be represented safely in generated jobs"
}

normalize_job_id() {
  local raw="$1"
  raw="${raw%%$'\n'*}"
  local job_id="${raw%%;*}"
  local cluster=""
  if [[ "${raw}" == *";"* ]]; then
    cluster="${raw#*;}"
    [[ "${cluster}" == "${TARGET_CLUSTER}" ]] \
      || fail "sbatch returned unexpected cluster identity: ${raw}"
  fi
  [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: ${raw}"
  printf '%s\n' "${job_id}"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source scripts/duca_cellcf_path_contract.sh
export BASE
export DUCA_CELLCF_TRAINING_PROFILE="exposure132"
source scripts/duca_cellcf_canonical_env.sh
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"
TARGET_CLUSTER="$(resolve_target_cluster)"
RUN_ROOT="${DUCA_ALLOCATION_RUN_ROOT:-}"
CHECKPOINT="${DUCA_ALLOCATION_CHECKPOINT:-}"
PRETRAIN="${ADATAD_PRETRAIN_PATH:-}"
EXPECTED_EPOCH="${DUCA_ALLOCATION_CHECKPOINT_EPOCH:-131}"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit is invalid"
[[ "${TARGET_CLUSTER}" == "n16r4" ]] || fail \
  "allocation evidence is preregistered for cluster n16r4, got ${TARGET_CLUSTER}"
[[ "${EXPECTED_EPOCH}" =~ ^[0-9]+$ ]] || fail "checkpoint epoch is invalid"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "submission requires a clean tree"
[[ -f "${CHECKPOINT}" ]] || fail "DUCA_ALLOCATION_CHECKPOINT is missing"
[[ -f "${PRETRAIN}" ]] || fail "ADATAD_PRETRAIN_PATH is missing"
[[ -f "${THUMOS14_ANNOTATION_PATH}" ]] || fail "THUMOS14 annotation is missing"
[[ -f "${THUMOS14_CLASS_MAP}" ]] || fail "THUMOS14 class map is missing"
[[ -d "${THUMOS14_TRAIN_DATA_PATH}" ]] || fail "THUMOS14 train data is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "DUCA_ALLOCATION_RUN_ROOT" "${REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "invalid allocation run root"
[[ ! -e "${RUN_ROOT}" ]] || fail "refusing to overwrite RUN_ROOT"
for binding_name in \
  REPO_ROOT BASE EXPECTED_COMMIT TARGET_CLUSTER RUN_ROOT CHECKPOINT PRETRAIN; do
  require_generated_script_safe "${binding_name}" "${!binding_name}"
done
mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs"
CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
PRETRAIN_SHA256="$(sha256sum "${PRETRAIN}" | awk '{print $1}')"

GATE_ROOT="${RUN_ROOT}/gate"
GATE_JSON="${GATE_ROOT}/allocation_ceiling_real_gate.json"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"

cat > "${RUN_ROOT}/suite_manifest.json" <<EOF
{
  "schema_version": "duca_allocation_training_suite_manifest_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "task": "offline_temporal_action_detection",
  "checkpoint": "${CHECKPOINT}",
  "checkpoint_sha256": "${CHECKPOINT_SHA256}",
  "checkpoint_epoch": ${EXPECTED_EPOCH},
  "checkpoint_state_key": "state_dict_ema",
  "pretrain": "${PRETRAIN}",
  "pretrain_sha256": "${PRETRAIN_SHA256}",
  "annotation": "${THUMOS14_ANNOTATION_PATH}",
  "annotation_sha256": "$(sha256sum "${THUMOS14_ANNOTATION_PATH}" | awk '{print $1}')",
  "class_map": "${THUMOS14_CLASS_MAP}",
  "class_map_sha256": "$(sha256sum "${THUMOS14_CLASS_MAP}" | awk '{print $1}')",
  "train_data_path": "${THUMOS14_TRAIN_DATA_PATH}",
  "training_config": "${REPO_ROOT}/configs/adatad/thumos/duca_allocation_ceiling_training_windows.py",
  "training_config_sha256": "$(sha256sum "${REPO_ROOT}/configs/adatad/thumos/duca_allocation_ceiling_training_windows.py" | awk '{print $1}')",
  "target_cluster": "${TARGET_CLUSTER}",
  "validation_subset_consumed": false,
  "selector_training_authorized": false
}
EOF
MANIFEST_SHA256="$(sha256sum "${RUN_ROOT}/suite_manifest.json" | awk '{print $1}')"

write_header() {
  local path="$1"
  local name="$2"
  local gpu="$3"
  local cpus="$4"
  local time_limit="$5"
  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --clusters=${TARGET_CLUSTER}"
    echo '#SBATCH --nodes=1'
    echo '#SBATCH --ntasks=1'
    if [[ "${gpu}" == "1" ]]; then
      echo '#SBATCH --gres=gpu:1'
    fi
    echo "#SBATCH --cpus-per-task=${cpus}"
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --output=${RUN_ROOT}/logs/${name}-%j.out"
    echo "#SBATCH --error=${RUN_ROOT}/logs/${name}-%j.err"
    echo 'set -euo pipefail'
    echo "cd '${REPO_ROOT}'"
    echo "module load cuda/11.8"
    echo "module load miniforge3/24.11"
    echo "[[ \"\${SLURM_CLUSTER_NAME:-}\" == 'n16r4' ]]"
    echo "[[ \"\$(git rev-parse HEAD)\" == '${EXPECTED_COMMIT}' ]]"
    echo "[[ -z \"\$(git status --porcelain --untracked-files=normal)\" ]]"
    echo "[[ \"\$(sha256sum '${RUN_ROOT}/suite_manifest.json' | awk '{print \$1}')\" == '${MANIFEST_SHA256}' ]]"
    echo "[[ \"\$(sha256sum '${CHECKPOINT}' | awk '{print \$1}')\" == '${CHECKPOINT_SHA256}' ]]"
    echo "[[ \"\$(sha256sum '${PRETRAIN}' | awk '{print \$1}')\" == '${PRETRAIN_SHA256}' ]]"
    echo "export BASE='${BASE}'"
    echo "export DUCA_CELLCF_TRAINING_PROFILE='exposure132'"
    echo "source scripts/duca_cellcf_canonical_env.sh"
    echo "export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'"
    echo "export DUCA_ALLOCATION_CHECKPOINT='${CHECKPOINT}'"
    echo "export DUCA_ALLOCATION_CHECKPOINT_EPOCH='${EXPECTED_EPOCH}'"
    echo "export DUCA_ALLOCATION_CHECKPOINT_SHA256='${CHECKPOINT_SHA256}'"
    echo "export ADATAD_PRETRAIN_PATH='${PRETRAIN}'"
    echo "export ADATAD_PRETRAIN_SHA256='${PRETRAIN_SHA256}'"
    echo "export DUCA_ALLOCATION_SUITE_MANIFEST='${RUN_ROOT}/suite_manifest.json'"
    echo "export DUCA_ALLOCATION_SUITE_MANIFEST_SHA256='${MANIFEST_SHA256}'"
    echo "export DUCA_ALLOCATION_RUN_ROOT='${RUN_ROOT}'"
    echo "export DUCA_ALLOCATION_GATE_JSON='${GATE_JSON}'"
  } > "${path}"
}

GATE_JOB="${RUN_ROOT}/jobs/gate.sbatch"
write_header "${GATE_JOB}" "dac-gate-${SHORT_COMMIT}" 1 4 "08:00:00"
{
  echo "export DUCA_ALLOCATION_GATE_ROOT='${GATE_ROOT}'"
  echo "bash scripts/run_duca_allocation_ceiling_real_gate.sh"
} >> "${GATE_JOB}"

EXPORT_JOB="${RUN_ROOT}/jobs/export.sbatch"
write_header "${EXPORT_JOB}" "dac-export-${SHORT_COMMIT}" 1 4 "1-00:00:00"
echo "bash scripts/run_duca_allocation_training_export.sh" >> "${EXPORT_JOB}"

DIAGNOSTIC_JOB="${RUN_ROOT}/jobs/diagnostics.sbatch"
# BSCC-N16R4 rejects one-node jobs without a generic GPU request, including
# CPU-only aggregation/solver stages. These stages still use CPU code only.
write_header "${DIAGNOSTIC_JOB}" "dac-diag-${SHORT_COMMIT}" 1 8 "4-00:00:00"
echo "bash scripts/run_duca_allocation_training_diagnostics.sh" >> "${DIAGNOSTIC_JOB}"

CANDIDATE_JOB="${RUN_ROOT}/jobs/candidate.sbatch"
write_header "${CANDIDATE_JOB}" "dac-cand-${SHORT_COMMIT}" 1 4 "1-00:00:00"
echo "bash scripts/run_duca_allocation_training_candidate_loss.sh" >> "${CANDIDATE_JOB}"

COMPLETION_JOB="${RUN_ROOT}/jobs/completion.sbatch"
write_header "${COMPLETION_JOB}" "dac-done-${SHORT_COMMIT}" 1 2 "1-00:00:00"
cat >> "${COMPLETION_JOB}" <<EOF
'${BASE}/conda_envs/opentad/bin/python' -m tools.bata.finalize_duca_allocation_training_suite \
  --gate-json '${GATE_JSON}' \
  --full-input-jsonl '${RUN_ROOT}/training_inputs.jsonl' \
  --full-ceiling-jsonl '${RUN_ROOT}/training_recoverability.jsonl' \
  --full-ceiling-summary-json '${RUN_ROOT}/training_recoverability.summary.json' \
  --full-ceiling-validation-json '${RUN_ROOT}/training_recoverability.validation.json' \
  --gt-input-jsonl '${RUN_ROOT}/training_gt32_inputs.jsonl' \
  --gt-ceiling-jsonl '${RUN_ROOT}/training_gt32_ceiling.jsonl' \
  --gt-ceiling-summary-json '${RUN_ROOT}/training_gt32_ceiling.summary.json' \
  --gt-ceiling-validation-json '${RUN_ROOT}/training_gt32_ceiling.validation.json' \
  --candidate-jsonl '${RUN_ROOT}/training_gt32_candidate_loss.jsonl' \
  --candidate-summary-json '${RUN_ROOT}/training_gt32_candidate_loss.summary.json' \
  --solver-cost-samples-jsonl '${RUN_ROOT}/training_solver_cost.samples.jsonl' \
  --solver-cost-summary-json '${RUN_ROOT}/training_solver_cost.summary.json' \
  --suite-manifest-json '${RUN_ROOT}/suite_manifest.json' \
  --suite-manifest-sha256 '${MANIFEST_SHA256}' \
  --output-json '${RUN_ROOT}/training_suite_evidence.json'
EOF

chmod 0755 "${RUN_ROOT}"/jobs/*.sbatch
"${PYTHON}" - \
  "${RUN_ROOT}/suite_manifest.json" \
  "${MANIFEST_SHA256}" \
  "${EXPECTED_COMMIT}" \
  "${CHECKPOINT}" \
  "${CHECKPOINT_SHA256}" \
  "${EXPECTED_EPOCH}" \
  "${PRETRAIN}" \
  "${PRETRAIN_SHA256}" <<'PY'
import pathlib
import sys

from tools.bata.finalize_duca_allocation_ceiling_gate import (
    _validate_suite_manifest,
)

_validate_suite_manifest(
    pathlib.Path(sys.argv[1]),
    expected_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
    expected_checkpoint=pathlib.Path(sys.argv[4]).resolve(),
    expected_checkpoint_sha256=sys.argv[5],
    expected_checkpoint_epoch=int(sys.argv[6]),
    expected_pretrain=pathlib.Path(sys.argv[7]).resolve(),
    expected_pretrain_sha256=sys.argv[8],
)
PY
bash -n "${RUN_ROOT}"/jobs/*.sbatch
cat > "${RUN_ROOT}/submission_intent.json" <<EOF
{
  "schema_version": "duca_allocation_training_suite_submission_intent_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "target_cluster": "${TARGET_CLUSTER}",
  "run_root": "${RUN_ROOT}",
  "manifest_sha256": "${MANIFEST_SHA256}",
  "job_files": {
    "gate": "$(sha256sum "${GATE_JOB}" | awk '{print $1}')",
    "export": "$(sha256sum "${EXPORT_JOB}" | awk '{print $1}')",
    "diagnostics": "$(sha256sum "${DIAGNOSTIC_JOB}" | awk '{print $1}')",
    "candidate": "$(sha256sum "${CANDIDATE_JOB}" | awk '{print $1}')",
    "completion": "$(sha256sum "${COMPLETION_JOB}" | awk '{print $1}')"
  },
  "mode": "$([[ "${PRECHECK_ONLY:-0}" == "1" ]] && printf precheck || printf submit)"
}
EOF
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_ALLOCATION_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v scancel >/dev/null 2>&1 || fail "scancel is unavailable"
SUBMITTED_JOB_IDS=()
rollback_partial_submission() {
  local exit_code=$?
  trap - ERR
  if [[ ${#SUBMITTED_JOB_IDS[@]} -gt 0 ]]; then
    echo \
      "[DUCA_ALLOCATION_SUBMIT] rolling back partial DAG: ${SUBMITTED_JOB_IDS[*]}" \
      >&2
    scancel --clusters="${TARGET_CLUSTER}" "${SUBMITTED_JOB_IDS[@]}" || true
  fi
  exit "${exit_code}"
}
trap rollback_partial_submission ERR

GATE_RAW="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${GATE_JOB}")"
GATE_ID="$(normalize_job_id "${GATE_RAW}")"
SUBMITTED_JOB_IDS+=("${GATE_ID}")
EXPORT_RAW="$(
  sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${GATE_ID}" "${EXPORT_JOB}"
)"
EXPORT_ID="$(normalize_job_id "${EXPORT_RAW}")"
SUBMITTED_JOB_IDS+=("${EXPORT_ID}")
DIAGNOSTIC_RAW="$(
  sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${EXPORT_ID}" "${DIAGNOSTIC_JOB}"
)"
DIAGNOSTIC_ID="$(normalize_job_id "${DIAGNOSTIC_RAW}")"
SUBMITTED_JOB_IDS+=("${DIAGNOSTIC_ID}")
CANDIDATE_RAW="$(
  sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${DIAGNOSTIC_ID}" "${CANDIDATE_JOB}"
)"
CANDIDATE_ID="$(normalize_job_id "${CANDIDATE_RAW}")"
SUBMITTED_JOB_IDS+=("${CANDIDATE_ID}")
COMPLETION_RAW="$(
  sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${CANDIDATE_ID}" "${COMPLETION_JOB}"
)"
COMPLETION_ID="$(normalize_job_id "${COMPLETION_RAW}")"
SUBMITTED_JOB_IDS+=("${COMPLETION_ID}")

cat > "${RUN_ROOT}/jobs.tsv" <<EOF
role	job_id	cluster	dependency	job_file
gate	${GATE_ID}	${TARGET_CLUSTER}	none	${GATE_JOB}
export	${EXPORT_ID}	${TARGET_CLUSTER}	afterok:${GATE_ID}	${EXPORT_JOB}
diagnostics	${DIAGNOSTIC_ID}	${TARGET_CLUSTER}	afterok:${EXPORT_ID}	${DIAGNOSTIC_JOB}
candidate	${CANDIDATE_ID}	${TARGET_CLUSTER}	afterok:${DIAGNOSTIC_ID}	${CANDIDATE_JOB}
completion	${COMPLETION_ID}	${TARGET_CLUSTER}	afterok:${CANDIDATE_ID}	${COMPLETION_JOB}
EOF

cat > "${RUN_ROOT}/submission.json" <<EOF
{
  "schema_version": "duca_allocation_training_suite_submission_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "manifest_sha256": "${MANIFEST_SHA256}",
  "submission_intent_sha256": "$(sha256sum "${RUN_ROOT}/submission_intent.json" | awk '{print $1}')",
  "run_root": "${RUN_ROOT}",
  "target_cluster": "${TARGET_CLUSTER}",
  "jobs": {
    "gate": "${GATE_ID}",
    "export": "${EXPORT_ID}",
    "diagnostics": "${DIAGNOSTIC_ID}",
    "candidate": "${CANDIDATE_ID}",
    "completion": "${COMPLETION_ID}"
  }
}
EOF
echo "[DUCA_ALLOCATION_SUBMIT] SUBMITTED ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
trap - ERR
