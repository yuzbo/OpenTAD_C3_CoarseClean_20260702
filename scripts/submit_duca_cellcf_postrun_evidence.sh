#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH

fail() {
  echo "[DUCA_CELLCF_POSTRUN_SUBMIT][FAIL] $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

EVIDENCE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EVIDENCE_REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${EVIDENCE_REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
RUN_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-}"
TRAINED_REPO_ROOT="${DUCA_CELLCF_TRAINED_REPO_ROOT:-}"
TRAINED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
EVIDENCE_COMMIT="$(git rev-parse HEAD)"
EXPECTED_EVIDENCE_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:-}"
RECOVERY_MANIFEST="${DUCA_CELLCF_COST_RECOVERY_MANIFEST:-}"
RECOVERY_MANIFEST_SHA256="${DUCA_CELLCF_COST_RECOVERY_MANIFEST_SHA256:-}"
AGGREGATE="${RUN_ROOT}/aggregate_suite_evidence.json"
FINAL_SUITE="${RUN_ROOT}/final_suite_evidence.json"
LEDGER="${RUN_ROOT}/jobs.submitted.tsv"
AGGREGATE_SHA256="${DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256:-}"
SUPPORTED_TRAINED_COMMIT="1642f265e48391418a7c8a4a087e33e2b7bf6899"

fsync_file() {
  "${PYTHON}" - "$1" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
with path.open("rb") as handle:
    os.fsync(handle.fileno())
directory_fd = os.open(
    path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -d "${RUN_ROOT}" ]] || fail "formal run root is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${EVIDENCE_REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "formal run root violates the path contract"
[[ -d "${TRAINED_REPO_ROOT}" ]] || fail "trained repository is missing"
[[ "${TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "trained commit is invalid"
[[ "${TRAINED_COMMIT}" == "${SUPPORTED_TRAINED_COMMIT}" ]] \
  || fail "unsupported trained commit for this frozen post-run protocol"
[[ "${EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "evidence commit is invalid"
[[ "${EXPECTED_EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "DUCA_EVIDENCE_EXPECTED_COMMIT is required"
[[ "${EVIDENCE_COMMIT}" != "${TRAINED_COMMIT}" ]] \
  || fail "trained and evidence commits must be distinct"
[[ "${EXPECTED_EVIDENCE_COMMIT}" == "${EVIDENCE_COMMIT}" ]] \
  || fail "checked-out evidence commit differs from the requested commit"
[[ "${RECOVERY_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_COST_RECOVERY_MANIFEST_SHA256 is required"
[[ -f "${RECOVERY_MANIFEST}" ]] || fail "cost recovery manifest is missing"
[[ "$(sha256_file "${RECOVERY_MANIFEST}")" == "${RECOVERY_MANIFEST_SHA256}" ]] \
  || fail "cost recovery manifest hash mismatch"
[[ "$(git -C "${TRAINED_REPO_ROOT}" rev-parse HEAD)" == "${TRAINED_COMMIT}" ]] \
  || fail "trained repository commit drift"
[[ -z "$(git -C "${TRAINED_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || fail "trained repository is dirty"
IGNORED_TRAINED_PYTHON_SOURCES="$(
  git -C "${TRAINED_REPO_ROOT}" ls-files --others --ignored \
    --exclude-standard -- '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py'
)"
[[ -z "${IGNORED_TRAINED_PYTHON_SOURCES}" ]] \
  || fail "ignored Python sources could shadow the trained repository"
unset IGNORED_TRAINED_PYTHON_SOURCES
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "evidence repository is dirty"
IGNORED_EVIDENCE_PYTHON_SOURCES="$(
  git ls-files --others --ignored --exclude-standard -- \
    '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py'
)"
[[ -z "${IGNORED_EVIDENCE_PYTHON_SOURCES}" ]] \
  || fail "ignored Python sources could shadow the evidence repository"
unset IGNORED_EVIDENCE_PYTHON_SOURCES
for path in "${AGGREGATE}" "${FINAL_SUITE}" "${LEDGER}"; do
  [[ -f "${path}" ]] || fail "required terminal evidence is missing: ${path}"
done
[[ "${AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256 is required"
[[ "$(sha256_file "${AGGREGATE}")" == "${AGGREGATE_SHA256}" ]] \
  || fail "aggregate evidence hash mismatch"
[[ ! -e "${RUN_ROOT}/convergence" ]] || fail "convergence output already exists"
[[ ! -e "${RUN_ROOT}/training_cost" ]] || fail "training-cost output already exists"

readarray -t binding < <(
  env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "${PYTHON}" - "${RUN_ROOT}" "${AGGREGATE}" "${FINAL_SUITE}" \
      "${LEDGER}" "${TRAINED_COMMIT}" "${TRAINED_REPO_ROOT}" \
      "${EVIDENCE_REPO_ROOT}" "${EVIDENCE_COMMIT}" \
      "${RECOVERY_MANIFEST}" "${RECOVERY_MANIFEST_SHA256}" <<'PY'
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.bata.duca_cellcf_protocol import LEGACY_EXPOSURE132_COMMITS
from tools.bata.duca_cellcf_suite_binding import load_suite_aggregate_binding
from tools.bata.finalize_duca_cellcf_postrun_evidence import (
    revalidate_trained_suite_exact,
)

run_root = Path(sys.argv[1]).resolve()
aggregate_path = Path(sys.argv[2]).resolve()
final_path = Path(sys.argv[3]).resolve()
ledger_path = Path(sys.argv[4]).resolve()
commit = sys.argv[5]
trained_repo_root = Path(sys.argv[6]).resolve()
evidence_repo_root = Path(sys.argv[7]).resolve()
evidence_commit = sys.argv[8]
recovery_manifest_path = Path(sys.argv[9]).resolve()
recovery_manifest_sha = sys.argv[10]

for path in (aggregate_path, final_path, ledger_path):
    if run_root not in path.parents:
        raise SystemExit(f"terminal evidence escaped the formal root: {path}")

aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
if (
    aggregate.get("schema") != "duca_cellcf_suite_manifest_v1"
    or aggregate.get("ok") is not True
    or aggregate.get("status") != "runs_complete_cost_pending"
    or aggregate.get("git_commit") != commit
):
    raise SystemExit("aggregate evidence identity/status mismatch")
profile = aggregate.get("training_profile")
if profile is None and commit in LEGACY_EXPOSURE132_COMMITS:
    profile = "exposure132"
if profile != "exposure132":
    raise SystemExit("post-run trajectory accepts only exposure132")
completed = aggregate.get("completed_runs")
if not isinstance(completed, dict) or set(completed) != {
    "uniform",
    "transition_beta0",
    "cellcf",
}:
    raise SystemExit("aggregate evidence does not bind exactly three arms")
for variant, record in completed.items():
    path = Path(str(record.get("path") or "")).resolve()
    expected = run_root / "logs" / variant / "post_run_evidence.json"
    if path != expected or not path.is_file():
        raise SystemExit(f"{variant} post-run path mismatch")
    if record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit(f"{variant} post-run hash mismatch")
post_run_paths = {
    variant: run_root / "logs" / variant / "post_run_evidence.json"
    for variant in ("uniform", "transition_beta0", "cellcf")
}
aggregate_binding = load_suite_aggregate_binding(
    aggregate_path,
    hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
    expected_commit=commit,
    expected_profile="exposure132",
    post_run_paths=post_run_paths,
)

final = json.loads(final_path.read_text(encoding="utf-8"))
if (
    final.get("schema") != "duca_cellcf_suite_manifest_v1"
    or final.get("ok") is not True
    or final.get("status") != "complete"
    or final.get("task") != "offline_temporal_action_detection"
    or final.get("git_commit") != commit
    or final.get("seed") != aggregate_binding["seed"]
    or final.get("cost_evidence_required") is not True
):
    raise SystemExit("final suite evidence identity/status mismatch")
final_profile = final.get("training_profile")
if final_profile is None and commit in LEGACY_EXPOSURE132_COMMITS:
    final_profile = "exposure132"
if final_profile != profile:
    raise SystemExit("final suite evidence profile mismatch")
cost_record = final.get("cost_evidence")
if not isinstance(cost_record, dict) or cost_record.get("validated") is not True:
    raise SystemExit("final suite evidence lacks validated cost evidence")
os.environ["DUCA_CELLCF_TRAINING_PROFILE"] = "exposure132"
regenerated_final = revalidate_trained_suite_exact(
    repo_root=trained_repo_root,
    evidence_repo_root=evidence_repo_root,
    expected_evidence_commit=evidence_commit,
    seed=aggregate_binding["seed"],
    expected_commit=commit,
    require_clean=True,
    gate_json=aggregate_binding["real_loader_gate"]["path"],
    pilot_json=aggregate_binding["ddp_pilot"]["path"],
    post_run_evidence=post_run_paths,
    cost_evidence=cost_record["path"],
    require_cost_evidence=True,
)
if regenerated_final != final:
    raise SystemExit("final suite evidence is not independently reproducible")

with ledger_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
required = {
    "uniform",
    "transition_beta0",
    "cellcf",
    "aggregate",
    "cost",
    "completion",
}
if {row.get("job_key") for row in rows} != required or len(rows) != len(required):
    raise SystemExit("submitted ledger does not bind the exact six-job formal DAG")
clusters = {row.get("cluster") for row in rows}
if len(clusters) != 1 or not next(iter(clusters)):
    raise SystemExit("formal jobs do not share one target cluster")
cluster = next(iter(clusters))
formal_by_key = {str(row["job_key"]): row for row in rows}
completion = formal_by_key["completion"]
job_id = str(completion.get("job_id") or "")
if not job_id.isdigit() or not completion.get("job_name"):
    raise SystemExit("original completion job identity is incomplete")
if (
    hashlib.sha256(recovery_manifest_path.read_bytes()).hexdigest()
    != recovery_manifest_sha
):
    raise SystemExit("cost recovery manifest hash drift")
recovery = json.loads(
    recovery_manifest_path.read_text(encoding="utf-8")
)
if (
    recovery.get("schema")
    != "duca_cellcf_cost_recovery_submission_v1"
    or recovery.get("ok") is not True
    or recovery.get("status") != "SUBMITTED_HELD_VERIFIED"
    or recovery.get("trained_git_commit") != commit
    or recovery.get("cost_producer_evidence_commit")
    != evidence_commit
    or recovery.get("final_suite_evidence_path")
    != str(final_path)
):
    raise SystemExit("cost recovery manifest identity/status mismatch")
recovery_jobs = recovery.get("jobs")
if (
    not isinstance(recovery_jobs, list)
    or [record.get("job_key") for record in recovery_jobs]
    != ["cost", "completion"]
):
    raise SystemExit(
        "cost recovery manifest does not bind two ordered jobs"
    )

def query(row):
    current_id = str(row.get("job_id") or "")
    current_name = str(row.get("job_name") or "")
    raw = subprocess.check_output(
        [
            "sacct", "-X", "-M", cluster, "-j", current_id,
            "-n", "-P", "-o",
            "JobIDRaw,JobName%128,Cluster,State,ExitCode",
        ],
        text=True,
    )
    matches = []
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and fields[0] == current_id:
            matches.append(fields[:5])
    if (
        len(matches) != 1
        or matches[0][1] != current_name
        or matches[0][2] != cluster
    ):
        raise SystemExit(
            f"scheduler identity mismatch for {current_name}"
        )
    return matches[0][3], matches[0][4]

old_cost_state, old_cost_exit = query(formal_by_key["cost"])
old_completion_state, _old_completion_exit = query(completion)
if old_cost_state != "FAILED" or old_cost_exit != "1:0":
    raise SystemExit("original cost job is not FAILED/1:0")
if not old_completion_state.startswith("CANCELLED"):
    raise SystemExit("original completion job is not cancelled")
for record in recovery_jobs:
    state, exit_code = query(record)
    if state != "COMPLETED" or exit_code != "0:0":
        raise SystemExit(
            f"recovery {record.get('job_key')} is not COMPLETED/0:0"
        )

print(aggregate["seed"])
print(cluster)
print(job_id)
print(hashlib.sha256(final_path.read_bytes()).hexdigest())
PY
)
[[ "${#binding[@]}" == "4" ]] || fail "terminal suite binding is incomplete"
SEED="${binding[0]}"
TARGET_CLUSTER="${binding[1]}"
FORMAL_COMPLETION_JOB_ID="${binding[2]}"
FINAL_SUITE_SHA256="${binding[3]}"
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "seed is invalid"
[[ "${TARGET_CLUSTER}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "cluster is invalid"
[[ "${FORMAL_COMPLETION_JOB_ID}" =~ ^[0-9]+$ ]] || fail "completion job id is invalid"
[[ "${FINAL_SUITE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "final suite hash is invalid"

CONTROL_ROOT="${RUN_ROOT}/postrun_submission_${EVIDENCE_COMMIT:0:7}_v1"
POSTRUN_OUTPUT_ROOT="${CONTROL_ROOT}/artifacts"
[[ ! -e "${CONTROL_ROOT}" ]] || fail "post-run submission root already exists"
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_CELLCF_POSTRUN_SUBMIT] PRECHECK PASS trained=${TRAINED_COMMIT} evidence=${EVIDENCE_COMMIT}"
  exit 0
fi

for command in sbatch scancel flock seq sha256sum sleep sacct scontrol squeue; do
  command -v "${command}" >/dev/null 2>&1 || fail "${command} is unavailable"
done
SLURM_QUERY_USER="${USER:-$(id -un)}"
[[ -n "${SLURM_QUERY_USER}" ]] || fail "Slurm query user is unavailable"
umask 077
mkdir -p "${CONTROL_ROOT}/jobs" "${CONTROL_ROOT}/logs" \
  "${CONTROL_ROOT}/receipts"
exec 9>"${CONTROL_ROOT}/submit.lock"
flock -n 9 || fail "another post-run submitter holds the lock"
SUBMITTED_JOB_IDS=()
SUBMITTED_JOB_KEYS=()
SUBMITTED_JOB_NAMES=()
SUBMITTED_JOB_DEPENDENCIES=()
SUBMITTED_JOB_TOKENS=()
SUBMITTED_JOB_FILES=()
SUBMITTED_JOB_FILE_SHA256S=()
SUBMISSION_COMMITTED=0
LAST_JOB_ID=""
UNRESOLVED_SUBMISSION_TOKEN=""
UNRESOLVED_JOB_NAME=""
UNRESOLVED_JOB_FILE=""

append_unique_job_id() {
  local candidate="$1"
  local existing
  for existing in "${SUBMITTED_JOB_IDS[@]}"; do
    [[ "${existing}" != "${candidate}" ]] || return 0
  done
  SUBMITTED_JOB_IDS+=("${candidate}")
}

terminate_on_signal() {
  local exit_code="$1"
  trap - HUP INT TERM
  exit "${exit_code}"
}

rollback_submitted_jobs() {
  local status="$?"
  if [[ "${status}" -ne 0 && "${SUBMISSION_COMMITTED}" -ne 1 \
    && ( "${#SUBMITTED_JOB_IDS[@]}" -gt 0 \
      || -n "${UNRESOLVED_SUBMISSION_TOKEN}" ) ]]; then
    local rollback_status="ROLLBACK_REQUESTED"
    local scancel_exit_code="not_run"
    local cancellation_verified=0
    write_rollback_record "${rollback_status}" \
      "${scancel_exit_code}" "${cancellation_verified}" || true
    if [[ -n "${UNRESOLVED_SUBMISSION_TOKEN}" ]]; then
      local recovered_job_id=""
      if recovered_job_id="$(
        recover_job_id_by_token \
          "${UNRESOLVED_SUBMISSION_TOKEN}" \
          "${UNRESOLVED_JOB_NAME}" \
          "${UNRESOLVED_JOB_FILE}"
      )"; then
        append_unique_job_id "${recovered_job_id}"
        UNRESOLVED_SUBMISSION_TOKEN=""
        UNRESOLVED_JOB_NAME=""
        UNRESOLVED_JOB_FILE=""
      fi
    fi
    if [[ "${#SUBMITTED_JOB_IDS[@]}" -gt 0 ]]; then
      if scancel --clusters="${TARGET_CLUSTER}" \
        "${SUBMITTED_JOB_IDS[@]}" >/dev/null 2>&1; then
        scancel_exit_code=0
      else
        scancel_exit_code="$?"
      fi
    fi
    if [[ "${scancel_exit_code}" == "0" \
      && -z "${UNRESOLVED_SUBMISSION_TOKEN}" ]]; then
      local verify_attempt
      for verify_attempt in $(seq 1 10); do
        if verify_cancelled_jobs; then
          cancellation_verified=1
          break
        fi
        sleep 2
      done
    fi
    if [[ "${cancellation_verified}" -eq 1 \
      && -z "${UNRESOLVED_SUBMISSION_TOKEN}" ]]; then
      rollback_status="ROLLED_BACK"
    else
      rollback_status="ROLLBACK_INCOMPLETE"
    fi
    write_rollback_record "${rollback_status}" \
      "${scancel_exit_code}" "${cancellation_verified}" || true
    if [[ "${rollback_status}" != "ROLLED_BACK" ]]; then
      echo "[DUCA_CELLCF_POSTRUN_SUBMIT][FAIL] rollback is incomplete" >&2
    fi
  fi
  return "${status}"
}
trap rollback_submitted_jobs EXIT
trap 'terminate_on_signal 129' HUP
trap 'terminate_on_signal 130' INT
trap 'terminate_on_signal 143' TERM

common_header() {
  local job_name="$1"
  local output_name="$2"
  cat <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=${CONTROL_ROOT}/logs/${output_name}-%j.out
#SBATCH --error=${CONTROL_ROOT}/logs/${output_name}-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${TRAINED_COMMIT}'
export DUCA_EVIDENCE_EXPECTED_COMMIT='${EVIDENCE_COMMIT}'
export DUCA_CELLCF_TRAINED_REPO_ROOT='${TRAINED_REPO_ROOT}'
export DUCA_CELLCF_FORMAL_RUN_ROOT='${RUN_ROOT}'
export DUCA_CELLCF_POSTRUN_OUTPUT_ROOT='${POSTRUN_OUTPUT_ROOT}'
export DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256='${AGGREGATE_SHA256}'
export DUCA_CELLCF_TRAINING_PROFILE='exposure132'
[[ "\$(git -C '${EVIDENCE_REPO_ROOT}' rev-parse HEAD)" == \
  '${EVIDENCE_COMMIT}' ]] || { echo '[DUCA_CELLCF_POSTRUN][FAIL] evidence commit drift' >&2; exit 1; }
[[ -z "\$(git -C '${EVIDENCE_REPO_ROOT}' status --porcelain --untracked-files=normal)" ]] \
  || { echo '[DUCA_CELLCF_POSTRUN][FAIL] evidence repository is dirty' >&2; exit 1; }
[[ -z "\$(git -C '${EVIDENCE_REPO_ROOT}' ls-files --others --ignored --exclude-standard -- '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py')" ]] \
  || { echo '[DUCA_CELLCF_POSTRUN][FAIL] ignored evidence Python source detected' >&2; exit 1; }
[[ "\$(git -C '${TRAINED_REPO_ROOT}' rev-parse HEAD)" == \
  '${TRAINED_COMMIT}' ]] || { echo '[DUCA_CELLCF_POSTRUN][FAIL] trained commit drift' >&2; exit 1; }
[[ -z "\$(git -C '${TRAINED_REPO_ROOT}' status --porcelain --untracked-files=normal)" ]] \
  || { echo '[DUCA_CELLCF_POSTRUN][FAIL] trained repository is dirty' >&2; exit 1; }
[[ -z "\$(git -C '${TRAINED_REPO_ROOT}' ls-files --others --ignored --exclude-standard -- '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py')" ]] \
  || { echo '[DUCA_CELLCF_POSTRUN][FAIL] ignored trained Python source detected' >&2; exit 1; }
cd '${EVIDENCE_REPO_ROOT}'
EOF
}

for variant in uniform transition_beta0 cellcf; do
  job="${CONTROL_ROOT}/jobs/convergence_${variant}.sbatch"
  {
    common_header "cellcf-conv-${variant}-${EVIDENCE_COMMIT:0:7}" \
      "convergence-${variant}"
    cat <<EOF
export DUCA_CELLCF_VARIANT='${variant}'
export SEED='${SEED}'
exec bash '${EVIDENCE_REPO_ROOT}/scripts/run_duca_cellcf_convergence_variant.sh'
EOF
  } > "${job}"
done

{
  common_header "cellcf-conv-summary-${EVIDENCE_COMMIT:0:7}" \
    "convergence-summary"
  cat <<EOF
exec bash '${EVIDENCE_REPO_ROOT}/scripts/summarize_duca_cellcf_convergence.sh'
EOF
} > "${CONTROL_ROOT}/jobs/convergence_summary.sbatch"

{
  common_header "cellcf-training-cost-${EVIDENCE_COMMIT:0:7}" \
    "training-cost"
  cat <<EOF
exec bash '${EVIDENCE_REPO_ROOT}/scripts/summarize_duca_cellcf_training_cost.sh'
EOF
} > "${CONTROL_ROOT}/jobs/training_cost.sbatch"

{
  common_header "cellcf-postrun-complete-${EVIDENCE_COMMIT:0:7}" \
    "postrun-complete"
  cat <<EOF
exec '${PYTHON}' -m tools.bata.finalize_duca_cellcf_postrun_evidence \
  --run-root '${RUN_ROOT}' --control-root '${CONTROL_ROOT}' \
  --trained-repo-root '${TRAINED_REPO_ROOT}' \
  --trained-commit '${TRAINED_COMMIT}' \
  --evidence-repo-root '${EVIDENCE_REPO_ROOT}' \
  --evidence-commit '${EVIDENCE_COMMIT}' \
  --aggregate '${AGGREGATE}' --aggregate-sha256 '${AGGREGATE_SHA256}' \
  --final-suite '${FINAL_SUITE}' \
  --final-suite-sha256 '${FINAL_SUITE_SHA256}' \
  --cost-recovery-manifest '${RECOVERY_MANIFEST}' \
  --cost-recovery-manifest-sha256 '${RECOVERY_MANIFEST_SHA256}' \
  --candidate --output '${CONTROL_ROOT}/postrun_evidence_candidate.json'
EOF
} > "${CONTROL_ROOT}/jobs/completion.sbatch"

for job in "${CONTROL_ROOT}"/jobs/*.sbatch; do
  chmod 0500 "${job}"
  bash -n "${job}" || fail "generated job has invalid syntax: ${job}"
done

INTENT="${CONTROL_ROOT}/submission_intent.json"
"${PYTHON}" - "${INTENT}" "${CONTROL_ROOT}" "${RUN_ROOT}" \
  "${TRAINED_REPO_ROOT}" "${TRAINED_COMMIT}" "${EVIDENCE_REPO_ROOT}" \
  "${EVIDENCE_COMMIT}" "${TARGET_CLUSTER}" "${AGGREGATE}" \
  "${AGGREGATE_SHA256}" "${FINAL_SUITE}" "${FINAL_SUITE_SHA256}" \
  "${POSTRUN_OUTPUT_ROOT}" "${RECOVERY_MANIFEST}" \
  "${RECOVERY_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    output_value,
    control_root_value,
    run_root_value,
    trained_root_value,
    trained_commit,
    evidence_root_value,
    evidence_commit,
    cluster,
    aggregate_value,
    aggregate_sha,
    final_value,
    final_sha,
    postrun_output_root_value,
    recovery_manifest_value,
    recovery_manifest_sha,
) = sys.argv[1:]
output = Path(output_value).resolve()
control_root = Path(control_root_value).resolve()
roles = {
    "convergence_uniform": "none",
    "convergence_transition_beta0": "none",
    "convergence_cellcf": "none",
    "convergence_summary": "afterok_three_convergence_jobs",
    "training_cost": "none",
    "completion": "afterok_summary_and_training_cost",
}
jobs = []
for key, dependency_role in roles.items():
    path = (control_root / "jobs" / f"{key}.sbatch").resolve()
    job_name = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH --job-name="):
            job_name = line.partition("=")[2]
            break
    if not job_name:
        raise SystemExit(f"generated job name is missing: {key}")
    job_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    jobs.append(
        {
            "job_key": key,
            "job_name": job_name,
            "dependency_role": dependency_role,
            "job_file": str(path),
            "job_file_sha256": job_file_sha256,
            "submission_token": (
                f"cellcf-postrun-{evidence_commit[:12]}-{key}-"
                f"{job_file_sha256[:12]}"
            ),
        }
    )
payload = {
    "schema": "duca_cellcf_postrun_submission_intent_v1",
    "status": "INTENT_RECORDED",
    "task": "offline_temporal_action_detection",
    "formal_run_root": str(Path(run_root_value).resolve()),
    "trained_repository": str(Path(trained_root_value).resolve()),
    "trained_git_commit": trained_commit,
    "evidence_repository": str(Path(evidence_root_value).resolve()),
    "evidence_git_commit": evidence_commit,
    "target_cluster": cluster,
    "aggregate_suite_evidence_path": str(Path(aggregate_value).resolve()),
    "aggregate_suite_evidence_sha256": aggregate_sha,
    "final_suite_evidence_path": str(Path(final_value).resolve()),
    "final_suite_evidence_sha256": final_sha,
    "postrun_output_root": str(Path(postrun_output_root_value).resolve()),
    "cost_recovery_manifest_path": str(
        Path(recovery_manifest_value).resolve()
    ),
    "cost_recovery_manifest_sha256": recovery_manifest_sha,
    "jobs": jobs,
}
payload["artifact_sha256"] = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    directory_fd = os.open(
        output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
chmod 0400 "${INTENT}"
INTENT_SHA256="$(sha256_file "${INTENT}")"
[[ "${INTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "submission intent hash is invalid"

LEDGER_OUT="${CONTROL_ROOT}/jobs.submitted.tsv"
printf 'job_key\tjob_id\tjob_name\tcluster\tdependency\tsubmission_token\tjob_file\tjob_file_sha256\tsubmitted_receipt\tsubmitted_receipt_sha256\tverified_receipt\tverified_receipt_sha256\ttrained_commit\tevidence_commit\taggregate_sha256\tsubmission_intent_sha256\n' \
  > "${LEDGER_OUT}"
fsync_file "${LEDGER_OUT}"

write_job_receipt() {
  local output="$1"
  local status="$2"
  local key="$3"
  local job_id="$4"
  local job_name="$5"
  local dependency="$6"
  local token="$7"
  local job_file="$8"
  local job_file_sha256="$9"
  local raw_response="${10}"
  local scheduler_validation="${11:-}"
  local submitted_receipt="${12:-}"
  local submitted_receipt_sha256="${13:-}"
  "${PYTHON}" - "${output}" "${status}" "${key}" "${job_id}" \
    "${job_name}" "${TARGET_CLUSTER}" "${dependency}" "${token}" \
    "${job_file}" "${job_file_sha256}" "${raw_response}" \
    "${scheduler_validation}" "${submitted_receipt}" \
    "${submitted_receipt_sha256}" "${TRAINED_COMMIT}" \
    "${EVIDENCE_COMMIT}" "${AGGREGATE_SHA256}" "${INTENT}" \
    "${INTENT_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    output_value,
    status,
    key,
    job_id,
    job_name,
    cluster,
    dependency,
    token,
    job_file,
    job_file_sha,
    raw_response,
    scheduler_validation,
    submitted_receipt,
    submitted_receipt_sha,
    trained_commit,
    evidence_commit,
    aggregate_sha,
    intent_value,
    intent_sha,
) = sys.argv[1:]
if status not in {"SUBMITTED_UNVERIFIED", "VERIFIED"}:
    raise SystemExit("unsupported post-run receipt status")
validation = json.loads(scheduler_validation) if scheduler_validation else None
payload = {
    "schema": "duca_cellcf_postrun_slurm_receipt_v1",
    "status": status,
    "task": "offline_temporal_action_detection",
    "job_key": key,
    "job_id": int(job_id),
    "job_name": job_name,
    "cluster": cluster,
    "dependency": dependency or None,
    "submission_token": token,
    "job_file": str(Path(job_file).resolve()),
    "job_file_sha256": job_file_sha,
    "raw_sbatch_response": raw_response,
    "scheduler_validation": validation,
    "submitted_receipt": (
        str(Path(submitted_receipt).resolve()) if submitted_receipt else None
    ),
    "submitted_receipt_sha256": submitted_receipt_sha or None,
    "trained_git_commit": trained_commit,
    "evidence_git_commit": evidence_commit,
    "aggregate_suite_evidence_sha256": aggregate_sha,
    "submission_intent": str(Path(intent_value).resolve()),
    "submission_intent_sha256": intent_sha,
}
payload["artifact_sha256"] = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()
output = Path(output_value).resolve()
fd, temporary = tempfile.mkstemp(
    prefix=output.name + ".", suffix=".tmp", dir=output.parent
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    directory_fd = os.open(
        output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

write_rollback_record() {
  local rollback_status="$1"
  local scancel_exit_code="$2"
  local cancellation_verified="$3"
  printf \
    'status=%s\nrollback_requested=true\ncluster=%s\njob_ids=%s\nscancel_exit_code=%s\ncancellation_verified=%s\nunresolved_submission_token=%s\n' \
    "${rollback_status}" "${TARGET_CLUSTER}" "${SUBMITTED_JOB_IDS[*]}" \
    "${scancel_exit_code}" "${cancellation_verified}" \
    "${UNRESOLVED_SUBMISSION_TOKEN}" \
    > "${CONTROL_ROOT}/submission_rollback.txt"
  fsync_file "${CONTROL_ROOT}/submission_rollback.txt"
}

recover_job_id_by_token() {
  local token="$1"
  local job_name="$2"
  local job_file="$3"
  local attempt
  for attempt in $(seq 1 10); do
    if env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
      "${PYTHON}" -m tools.bata.reconcile_duca_cellcf_slurm_submission \
        recover-held-job --token "${token}" --job-name "${job_name}" \
        --cluster "${TARGET_CLUSTER}" --job-file "${job_file}" \
        --user "${SLURM_QUERY_USER}"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

verify_cancelled_jobs() {
  local ids_csv
  ids_csv="$(IFS=,; echo "${SUBMITTED_JOB_IDS[*]}")"
  env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "${PYTHON}" -m tools.bata.reconcile_duca_cellcf_slurm_submission \
      verify-cancelled --job-ids "${ids_csv}" \
      --cluster "${TARGET_CLUSTER}" >/dev/null
}

revalidate_all_current_holds() {
  [[ "${#SUBMITTED_JOB_IDS[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_KEYS[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_NAMES[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_DEPENDENCIES[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_TOKENS[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_FILES[@]}" -eq 6 \
    && "${#SUBMITTED_JOB_FILE_SHA256S[@]}" -eq 6 ]] \
    || fail "held barrier does not bind the exact six-job DAG"
  local index
  for index in "${!SUBMITTED_JOB_IDS[@]}"; do
    local validation=""
    local attempt
    for attempt in $(seq 1 10); do
      if validation="$(
        env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
          "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
            --job-id "${SUBMITTED_JOB_IDS[index]}" \
            --job-name "${SUBMITTED_JOB_NAMES[index]}" \
            --comment "${SUBMITTED_JOB_TOKENS[index]}" \
            --cluster "${TARGET_CLUSTER}" \
            --job-file "${SUBMITTED_JOB_FILES[index]}" \
            --job-file-sha256 "${SUBMITTED_JOB_FILE_SHA256S[index]}" \
            --require-scheduler-script --require-submitted-with-hold \
            --require-current-user-hold \
            --dependency "${SUBMITTED_JOB_DEPENDENCIES[index]}"
      )"; then
        break
      fi
      validation=""
      sleep 2
    done
    [[ -n "${validation}" ]] \
      || fail "release barrier hold validation failed for ${SUBMITTED_JOB_KEYS[index]}"
  done
}

submit_job() {
  local key="$1"
  local dependency="$2"
  local job_file="${CONTROL_ROOT}/jobs/${key}.sbatch"
  local job_name
  job_name="$(awk -F= '/^#SBATCH --job-name=/{print $2; exit}' "${job_file}")"
  local job_file_sha256
  job_file_sha256="$(sha256_file "${job_file}")"
  [[ "$(sha256_file "${INTENT}")" == "${INTENT_SHA256}" ]] \
    || fail "submission intent changed before ${key}"
  [[ "$(sha256_file "${AGGREGATE}")" == "${AGGREGATE_SHA256}" ]] \
    || fail "aggregate evidence changed before ${key}"
  [[ "$(sha256_file "${FINAL_SUITE}")" == "${FINAL_SUITE_SHA256}" ]] \
    || fail "final suite evidence changed before ${key}"
  readarray -t intended < <(
    "${PYTHON}" - "${INTENT}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
matches = [record for record in payload["jobs"] if record.get("job_key") == key]
if len(matches) != 1:
    raise SystemExit(f"submission intent has no unique binding for {key}")
record = matches[0]
print(record["job_name"])
print(Path(record["job_file"]).resolve())
print(record["job_file_sha256"])
print(record["submission_token"])
PY
  )
  [[ "${#intended[@]}" == "4" ]] \
    || fail "submission intent binding is incomplete for ${key}"
  [[ "${intended[0]}" == "${job_name}" ]] \
    || fail "job name changed after submission intent for ${key}"
  [[ "${intended[1]}" == "$(realpath -e -- "${job_file}")" ]] \
    || fail "job path changed after submission intent for ${key}"
  [[ "${intended[2]}" == "${job_file_sha256}" ]] \
    || fail "job file changed after submission intent for ${key}"
  local token="${intended[3]}"
  local output
  UNRESOLVED_SUBMISSION_TOKEN="${token}"
  UNRESOLVED_JOB_NAME="${job_name}"
  UNRESOLVED_JOB_FILE="${job_file}"
  if [[ -n "${dependency}" ]]; then
    if ! output="$(sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
      --job-name="${job_name}" --comment="${token}" \
      --dependency="${dependency}" "${job_file}")"; then
      fail "sbatch failed for ${key}; rollback reconciliation required"
    fi
  else
    if ! output="$(sbatch --parsable --hold --clusters="${TARGET_CLUSTER}" \
      --job-name="${job_name}" --comment="${token}" "${job_file}")"; then
      fail "sbatch failed for ${key}; rollback reconciliation required"
    fi
  fi
  local normalized=""
  if normalized="$("${PYTHON}" - "${output}" "${TARGET_CLUSTER}" <<'PY'
import re
import sys

value, cluster = sys.argv[1:]
match = re.fullmatch(r"([1-9][0-9]*);([A-Za-z0-9._-]+)", value.strip())
if match is None or match.group(2) != cluster:
    raise SystemExit("sbatch did not return exact jobid;cluster")
print(f"{match.group(1)};{match.group(2)}")
PY
)"; then
    :
  else
    local recovered_job_id=""
    if recovered_job_id="$(
      recover_job_id_by_token "${token}" "${job_name}" "${job_file}"
    )"; then
      append_unique_job_id "${recovered_job_id}"
      UNRESOLVED_SUBMISSION_TOKEN=""
      UNRESOLVED_JOB_NAME=""
      UNRESOLVED_JOB_FILE=""
    fi
    fail "invalid sbatch response for ${key}; rollback reconciliation required"
  fi
  local job_id="${normalized%%;*}"
  append_unique_job_id "${job_id}"
  UNRESOLVED_SUBMISSION_TOKEN=""
  UNRESOLVED_JOB_NAME=""
  UNRESOLVED_JOB_FILE=""
  local submitted_receipt="${CONTROL_ROOT}/receipts/${key}.submitted.json"
  write_job_receipt "${submitted_receipt}" "SUBMITTED_UNVERIFIED" \
    "${key}" "${job_id}" "${job_name}" "${dependency}" "${token}" \
    "${job_file}" "${job_file_sha256}" "${output}"
  chmod 0400 "${submitted_receipt}"
  local submitted_receipt_sha256
  submitted_receipt_sha256="$(sha256_file "${submitted_receipt}")"
  local scheduler_validation=""
  local validation_attempt
  for validation_attempt in $(seq 1 10); do
    if scheduler_validation="$(
      env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
        "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
          --job-id "${job_id}" --job-name "${job_name}" \
          --comment "${token}" --cluster "${TARGET_CLUSTER}" \
          --job-file "${job_file}" --job-file-sha256 "${job_file_sha256}" \
          --require-scheduler-script --require-submitted-with-hold \
          --require-current-user-hold \
          --dependency "${dependency}"
    )"; then
      break
    fi
    scheduler_validation=""
    sleep 2
  done
  [[ -n "${scheduler_validation}" ]] \
    || fail "scheduler identity validation failed for ${key}; submitted receipt preserved"
  local verified_receipt="${CONTROL_ROOT}/receipts/${key}.verified.json"
  write_job_receipt "${verified_receipt}" "VERIFIED" \
    "${key}" "${job_id}" "${job_name}" "${dependency}" "${token}" \
    "${job_file}" "${job_file_sha256}" "${output}" \
    "${scheduler_validation}" "${submitted_receipt}" \
    "${submitted_receipt_sha256}"
  chmod 0400 "${verified_receipt}"
  local verified_receipt_sha256
  verified_receipt_sha256="$(sha256_file "${verified_receipt}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${key}" "${job_id}" "${job_name}" "${TARGET_CLUSTER}" \
    "${dependency:-none}" "${token}" "${job_file}" "${job_file_sha256}" \
    "${submitted_receipt}" "${submitted_receipt_sha256}" \
    "${verified_receipt}" "${verified_receipt_sha256}" \
    "${TRAINED_COMMIT}" "${EVIDENCE_COMMIT}" "${AGGREGATE_SHA256}" \
    "${INTENT_SHA256}" \
    >> "${LEDGER_OUT}"
  fsync_file "${LEDGER_OUT}"
  SUBMITTED_JOB_KEYS+=("${key}")
  SUBMITTED_JOB_NAMES+=("${job_name}")
  SUBMITTED_JOB_DEPENDENCIES+=("${dependency}")
  SUBMITTED_JOB_TOKENS+=("${token}")
  SUBMITTED_JOB_FILES+=("${job_file}")
  SUBMITTED_JOB_FILE_SHA256S+=("${job_file_sha256}")
  LAST_JOB_ID="${job_id}"
}

submit_job convergence_uniform ""
uniform_id="${LAST_JOB_ID}"
submit_job convergence_transition_beta0 ""
transition_id="${LAST_JOB_ID}"
submit_job convergence_cellcf ""
cellcf_id="${LAST_JOB_ID}"
summary_dependency="afterok:${uniform_id}:${transition_id}:${cellcf_id}"
submit_job convergence_summary "${summary_dependency}"
summary_id="${LAST_JOB_ID}"
submit_job training_cost ""
training_cost_id="${LAST_JOB_ID}"
completion_dependency="afterok:${summary_id}:${training_cost_id}"
submit_job completion "${completion_dependency}"
completion_id="${LAST_JOB_ID}"
chmod 0400 "${LEDGER_OUT}"

"${PYTHON}" - "${CONTROL_ROOT}/submission_manifest.json" "${LEDGER_OUT}" \
  "${RUN_ROOT}" "${TRAINED_REPO_ROOT}" "${TRAINED_COMMIT}" \
  "${EVIDENCE_REPO_ROOT}" "${EVIDENCE_COMMIT}" "${TARGET_CLUSTER}" \
  "${AGGREGATE}" "${AGGREGATE_SHA256}" "${FINAL_SUITE}" \
  "${FINAL_SUITE_SHA256}" "${FORMAL_COMPLETION_JOB_ID}" \
  "${POSTRUN_OUTPUT_ROOT}" "${INTENT}" "${INTENT_SHA256}" \
  "${RECOVERY_MANIFEST}" "${RECOVERY_MANIFEST_SHA256}" <<'PY'
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    output_value,
    ledger_value,
    run_root_value,
    trained_root_value,
    trained_commit,
    evidence_root_value,
    evidence_commit,
    cluster,
    aggregate_value,
    aggregate_sha,
    final_value,
    final_sha,
    formal_completion_job_id,
    postrun_output_root_value,
    intent_value,
    intent_sha,
    recovery_manifest_value,
    recovery_manifest_sha,
) = sys.argv[1:]
output = Path(output_value).resolve()
ledger = Path(ledger_value).resolve()
with ledger.open(encoding="utf-8", newline="") as handle:
    jobs = list(csv.DictReader(handle, delimiter="\t"))
if [row["job_key"] for row in jobs] != [
    "convergence_uniform",
    "convergence_transition_beta0",
    "convergence_cellcf",
    "convergence_summary",
    "training_cost",
    "completion",
]:
    raise SystemExit("post-run job ledger order mismatch")
payload = {
    "schema": "duca_cellcf_postrun_submission_manifest_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "training_profile": "exposure132",
    "formal_run_root": str(Path(run_root_value).resolve()),
    "trained_repository": str(Path(trained_root_value).resolve()),
    "trained_git_commit": trained_commit,
    "evidence_repository": str(Path(evidence_root_value).resolve()),
    "evidence_git_commit": evidence_commit,
    "target_cluster": cluster,
    "aggregate_suite_evidence_path": str(Path(aggregate_value).resolve()),
    "aggregate_suite_evidence_sha256": aggregate_sha,
    "final_suite_evidence_path": str(Path(final_value).resolve()),
    "final_suite_evidence_sha256": final_sha,
    "postrun_output_root": str(Path(postrun_output_root_value).resolve()),
    "submission_intent_path": str(Path(intent_value).resolve()),
    "submission_intent_sha256": intent_sha,
    "formal_completion_job_id": int(formal_completion_job_id),
    "cost_recovery_manifest_path": str(
        Path(recovery_manifest_value).resolve()
    ),
    "cost_recovery_manifest_sha256": recovery_manifest_sha,
    "completion_mode": "cost_recovery",
    "original_formal_dag_complete": False,
    "jobs_ledger_path": str(ledger),
    "jobs_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "jobs": jobs,
}
payload["artifact_sha256"] = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    directory_fd = os.open(
        output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
chmod 0400 "${CONTROL_ROOT}/submission_manifest.json"

revalidate_all_current_holds
for job_id in "${completion_id}" "${summary_id}" "${training_cost_id}" \
  "${cellcf_id}" "${transition_id}" "${uniform_id}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${job_id}" \
    || fail "failed to release verified held job ${job_id}"
done
SUBMISSION_COMMITTED=1
trap - EXIT

echo "[DUCA_CELLCF_POSTRUN_SUBMIT] submitted convergence=${uniform_id},${transition_id},${cellcf_id} summary=${summary_id} training_cost=${training_cost_id} completion=${completion_id}"
