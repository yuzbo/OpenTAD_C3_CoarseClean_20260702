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
  [[ "${value}" != *"'"* && "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || fail "${label} cannot be represented safely in generated jobs"
}

register_submitted_job() {
  local raw="$1"
  raw="${raw%%$'\n'*}"
  local job_id="${raw%%;*}"
  if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
    SUBMITTED_JOB_IDS+=("${job_id}")
  fi
  [[ "${raw}" =~ ^[0-9]+\;n16r4$ ]] \
    || fail "sbatch did not return full jobid;n16r4 identity: ${raw}"
  NORMALIZED_JOB_ID="${job_id}"
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
GT_SOLVER_TOTAL_DEADLINE_SECONDS=300
GT_RUNTIME_MAX_ALLOWED_SECONDS=43200

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
  "schema_version": "duca_allocation_training_suite_manifest_v2",
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
  "gt_solver_total_deadline_seconds": ${GT_SOLVER_TOTAL_DEADLINE_SECONDS},
  "gt_runtime_max_allowed_seconds": ${GT_RUNTIME_MAX_ALLOWED_SECONDS},
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
    echo 'if [[ -f /etc/profile ]]; then source /etc/profile; fi'
    echo 'if ! type module >/dev/null 2>&1; then'
    echo '  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do'
    echo '    if [[ -f "${init}" ]]; then source "${init}"; break; fi'
    echo '  done'
    echo 'fi'
    echo 'type module >/dev/null 2>&1 || { echo "[DUCA_ALLOCATION_JOB][FAIL] environment modules unavailable" >&2; exit 1; }'
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
    echo "export DUCA_ALLOCATION_GT_TIME_LIMIT_SECONDS='${GT_SOLVER_TOTAL_DEADLINE_SECONDS}'"
    echo "export DUCA_ALLOCATION_MAX_GT32_SECONDS='${GT_RUNTIME_MAX_ALLOWED_SECONDS}'"
    echo "export DUCA_ALLOCATION_RUN_ROOT='${RUN_ROOT}'"
    echo "export DUCA_ALLOCATION_GATE_JSON='${GATE_JSON}'"
    echo "export DUCA_ALLOCATION_SCHEDULER_RECEIPT='${RUN_ROOT}/scheduler_receipt.json'"
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
  --submission-json '${RUN_ROOT}/submission.json' \
  --submission-token "\${DUCA_ALLOCATION_SUBMISSION_TOKEN}" \
  --current-job-id "\${SLURM_JOB_ID}" \
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
SUBMISSION_TOKEN="$(
  sha256sum "${RUN_ROOT}/submission_intent.json" | awk '{print $1}'
)"
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_ALLOCATION_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v scancel >/dev/null 2>&1 || fail "scancel is unavailable"
command -v scontrol >/dev/null 2>&1 || fail "scontrol is unavailable"
command -v squeue >/dev/null 2>&1 || fail "squeue is unavailable"
SUBMITTED_JOB_IDS=()
SUBMISSION_COMPLETE=0
cleanup_partial_submission() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "${SUBMISSION_COMPLETE}" != "1" && ${#SUBMITTED_JOB_IDS[@]} -gt 0 ]]; then
    echo \
      "[DUCA_ALLOCATION_SUBMIT] rolling back partial DAG: ${SUBMITTED_JOB_IDS[*]}" \
      >&2
    if ! scancel --clusters="${TARGET_CLUSTER}" "${SUBMITTED_JOB_IDS[@]}"; then
      echo "[DUCA_ALLOCATION_SUBMIT][FAIL] scancel returned failure" >&2
      exit_code=1
    fi
    local ids_csv
    local active=""
    local query_ok=0
    ids_csv="$(IFS=,; printf '%s' "${SUBMITTED_JOB_IDS[*]}")"
    for _attempt in 1 2 3 4 5 6 7 8 9 10; do
      if active="$(
        squeue --clusters="${TARGET_CLUSTER}" \
          --jobs="${ids_csv}" --noheader --format='%i %T'
      )"; then
        query_ok=1
      else
        query_ok=0
        echo \
          "[DUCA_ALLOCATION_SUBMIT][FAIL] could not confirm rollback with squeue" \
          >&2
        sleep 1
        continue
      fi
      [[ -z "${active}" ]] && break
      sleep 1
    done
    if [[ "${query_ok}" != "1" || -n "${active}" ]]; then
      echo \
        "[DUCA_ALLOCATION_SUBMIT][FAIL] jobs remain visible after rollback: ${active}" \
        >&2
      exit_code=1
    fi
  fi
  exit "${exit_code}"
}
capture_scheduler_job() {
  local phase="$1"
  local role="$2"
  local job_id="$3"
  local output="${RUN_ROOT}/scheduler/${phase}.${role}.scontrol.txt"
  local raw=""
  for _attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if raw="$(
      scontrol --clusters="${TARGET_CLUSTER}" show job -o "${job_id}" 2>/dev/null
    )" && [[ "${raw}" == JobId="${job_id} "* ]]; then
      if [[ "${phase}" == "pre_release" && "${raw}" != *" Reason=JobHeldUser "* ]]; then
        sleep 1
        continue
      fi
      if [[ "${phase}" == "post_release" && "${raw}" == *" Reason=JobHeldUser "* ]]; then
        sleep 1
        continue
      fi
      printf '%s\n' "${raw}" > "${output}"
      return 0
    fi
    sleep 1
  done
  fail "could not capture scheduler state for ${role} Job ${job_id}"
}
trap cleanup_partial_submission EXIT
trap 'exit 130' INT TERM
SBATCH_EXPORT="ALL,DUCA_ALLOCATION_SUBMISSION_TOKEN=${SUBMISSION_TOKEN},DUCA_ALLOCATION_SUBMISSION_JSON=${RUN_ROOT}/submission.json"

GATE_RAW="$(
  sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
    --export="${SBATCH_EXPORT}" "${GATE_JOB}"
)"
register_submitted_job "${GATE_RAW}"
GATE_ID="${NORMALIZED_JOB_ID}"
EXPORT_RAW="$(
  sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
    --export="${SBATCH_EXPORT}" \
    --dependency="afterok:${GATE_ID}" "${EXPORT_JOB}"
)"
register_submitted_job "${EXPORT_RAW}"
EXPORT_ID="${NORMALIZED_JOB_ID}"
DIAGNOSTIC_RAW="$(
  sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
    --export="${SBATCH_EXPORT}" \
    --dependency="afterok:${EXPORT_ID}" "${DIAGNOSTIC_JOB}"
)"
register_submitted_job "${DIAGNOSTIC_RAW}"
DIAGNOSTIC_ID="${NORMALIZED_JOB_ID}"
CANDIDATE_RAW="$(
  sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
    --export="${SBATCH_EXPORT}" \
    --dependency="afterok:${DIAGNOSTIC_ID}" "${CANDIDATE_JOB}"
)"
register_submitted_job "${CANDIDATE_RAW}"
CANDIDATE_ID="${NORMALIZED_JOB_ID}"
COMPLETION_RAW="$(
  sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
    --export="${SBATCH_EXPORT}" \
    --dependency="afterok:${CANDIDATE_ID}" "${COMPLETION_JOB}"
)"
register_submitted_job "${COMPLETION_RAW}"
COMPLETION_ID="${NORMALIZED_JOB_ID}"

cat > "${RUN_ROOT}/jobs.tsv" <<EOF
role	job_id	cluster	dependency	job_name	job_file
gate	${GATE_ID}	${TARGET_CLUSTER}	none	dac-gate-${SHORT_COMMIT}	${GATE_JOB}
export	${EXPORT_ID}	${TARGET_CLUSTER}	afterok:${GATE_ID}	dac-export-${SHORT_COMMIT}	${EXPORT_JOB}
diagnostics	${DIAGNOSTIC_ID}	${TARGET_CLUSTER}	afterok:${EXPORT_ID}	dac-diag-${SHORT_COMMIT}	${DIAGNOSTIC_JOB}
candidate	${CANDIDATE_ID}	${TARGET_CLUSTER}	afterok:${DIAGNOSTIC_ID}	dac-cand-${SHORT_COMMIT}	${CANDIDATE_JOB}
completion	${COMPLETION_ID}	${TARGET_CLUSTER}	afterok:${CANDIDATE_ID}	dac-done-${SHORT_COMMIT}	${COMPLETION_JOB}
EOF

cat > "${RUN_ROOT}/submission.json" <<EOF
{
  "schema_version": "duca_allocation_training_suite_submission_v3",
  "git_commit": "${EXPECTED_COMMIT}",
  "submission_token": "${SUBMISSION_TOKEN}",
  "submission_intent_json": "${RUN_ROOT}/submission_intent.json",
  "submission_intent_sha256": "${SUBMISSION_TOKEN}",
  "suite_manifest_json": "${RUN_ROOT}/suite_manifest.json",
  "suite_manifest_sha256": "${MANIFEST_SHA256}",
  "run_root": "${RUN_ROOT}",
  "target_cluster": "${TARGET_CLUSTER}",
  "jobs_tsv": "${RUN_ROOT}/jobs.tsv",
  "jobs_tsv_sha256": "$(sha256sum "${RUN_ROOT}/jobs.tsv" | awk '{print $1}')",
  "jobs": {
    "gate": {"job_id": "${GATE_ID}", "cluster": "${TARGET_CLUSTER}", "dependency": null, "job_name": "dac-gate-${SHORT_COMMIT}", "job_file": "${GATE_JOB}"},
    "export": {"job_id": "${EXPORT_ID}", "cluster": "${TARGET_CLUSTER}", "dependency": "afterok:${GATE_ID}", "job_name": "dac-export-${SHORT_COMMIT}", "job_file": "${EXPORT_JOB}"},
    "diagnostics": {"job_id": "${DIAGNOSTIC_ID}", "cluster": "${TARGET_CLUSTER}", "dependency": "afterok:${EXPORT_ID}", "job_name": "dac-diag-${SHORT_COMMIT}", "job_file": "${DIAGNOSTIC_JOB}"},
    "candidate": {"job_id": "${CANDIDATE_ID}", "cluster": "${TARGET_CLUSTER}", "dependency": "afterok:${DIAGNOSTIC_ID}", "job_name": "dac-cand-${SHORT_COMMIT}", "job_file": "${CANDIDATE_JOB}"},
    "completion": {"job_id": "${COMPLETION_ID}", "cluster": "${TARGET_CLUSTER}", "dependency": "afterok:${CANDIDATE_ID}", "job_name": "dac-done-${SHORT_COMMIT}", "job_file": "${COMPLETION_JOB}"}
  }
}
EOF
"${PYTHON}" -m tools.bata.validate_duca_allocation_submission_receipt \
  --submission-json "${RUN_ROOT}/submission.json" \
  --submission-token "${SUBMISSION_TOKEN}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-manifest-json "${RUN_ROOT}/suite_manifest.json" \
  --suite-manifest-sha256 "${MANIFEST_SHA256}" \
  --role gate \
  --current-job-id "${GATE_ID}"
mkdir "${RUN_ROOT}/scheduler"
for phase_role_id in \
  "gate:${GATE_ID}" \
  "export:${EXPORT_ID}" \
  "diagnostics:${DIAGNOSTIC_ID}" \
  "candidate:${CANDIDATE_ID}" \
  "completion:${COMPLETION_ID}"; do
  role="${phase_role_id%%:*}"
  job_id="${phase_role_id#*:}"
  capture_scheduler_job pre_release "${role}" "${job_id}"
done
"${PYTHON}" -m tools.bata.validate_duca_allocation_scheduler_receipt capture \
  --submission-json "${RUN_ROOT}/submission.json" \
  --submission-token "${SUBMISSION_TOKEN}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-manifest-json "${RUN_ROOT}/suite_manifest.json" \
  --suite-manifest-sha256 "${MANIFEST_SHA256}" \
  --phase pre_release \
  --raw "gate=${RUN_ROOT}/scheduler/pre_release.gate.scontrol.txt" \
  --raw "export=${RUN_ROOT}/scheduler/pre_release.export.scontrol.txt" \
  --raw "diagnostics=${RUN_ROOT}/scheduler/pre_release.diagnostics.scontrol.txt" \
  --raw "candidate=${RUN_ROOT}/scheduler/pre_release.candidate.scontrol.txt" \
  --raw "completion=${RUN_ROOT}/scheduler/pre_release.completion.scontrol.txt" \
  --output-json "${RUN_ROOT}/scheduler/pre_release.snapshot.json"
for job_id in \
  "${COMPLETION_ID}" \
  "${CANDIDATE_ID}" \
  "${DIAGNOSTIC_ID}" \
  "${EXPORT_ID}" \
  "${GATE_ID}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${job_id}"
done
for phase_role_id in \
  "gate:${GATE_ID}" \
  "export:${EXPORT_ID}" \
  "diagnostics:${DIAGNOSTIC_ID}" \
  "candidate:${CANDIDATE_ID}" \
  "completion:${COMPLETION_ID}"; do
  role="${phase_role_id%%:*}"
  job_id="${phase_role_id#*:}"
  capture_scheduler_job post_release "${role}" "${job_id}"
done
"${PYTHON}" -m tools.bata.validate_duca_allocation_scheduler_receipt capture \
  --submission-json "${RUN_ROOT}/submission.json" \
  --submission-token "${SUBMISSION_TOKEN}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-manifest-json "${RUN_ROOT}/suite_manifest.json" \
  --suite-manifest-sha256 "${MANIFEST_SHA256}" \
  --phase post_release \
  --raw "gate=${RUN_ROOT}/scheduler/post_release.gate.scontrol.txt" \
  --raw "export=${RUN_ROOT}/scheduler/post_release.export.scontrol.txt" \
  --raw "diagnostics=${RUN_ROOT}/scheduler/post_release.diagnostics.scontrol.txt" \
  --raw "candidate=${RUN_ROOT}/scheduler/post_release.candidate.scontrol.txt" \
  --raw "completion=${RUN_ROOT}/scheduler/post_release.completion.scontrol.txt" \
  --pre-release-snapshot-json "${RUN_ROOT}/scheduler/pre_release.snapshot.json" \
  --output-json "${RUN_ROOT}/scheduler_receipt.json"
"${PYTHON}" -m tools.bata.validate_duca_allocation_scheduler_receipt validate \
  --scheduler-receipt-json "${RUN_ROOT}/scheduler_receipt.json" \
  --submission-json "${RUN_ROOT}/submission.json" \
  --submission-token "${SUBMISSION_TOKEN}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-manifest-json "${RUN_ROOT}/suite_manifest.json" \
  --suite-manifest-sha256 "${MANIFEST_SHA256}"
SUBMISSION_COMPLETE=1
trap - EXIT INT TERM
echo "[DUCA_ALLOCATION_SUBMIT] SUBMITTED ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
