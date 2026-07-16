#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_SUBMIT][FAIL] $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
RUN_ROOT="${RUN_ROOT:-}"
[[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || fail "RUN_ROOT must name a prepared suite"

MANIFEST="${RUN_ROOT}/suite_manifest.json"
PREPARED_SUBMISSION="${RUN_ROOT}/prepared_submission.json"
PREPARED_SUBMISSION_SHA_FILE="${PREPARED_SUBMISSION}.sha256"
[[ -f "${MANIFEST}" ]] || fail "suite manifest is missing"
[[ -f "${PREPARED_SUBMISSION}" ]] || fail "prepared submission binding is missing"
[[ -f "${PREPARED_SUBMISSION_SHA_FILE}" ]] || fail "prepared submission hash sidecar is missing"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "submission requires a clean tree"
[[ "${DUCA_OFFICIAL_ADATAD_CHECKPOINT_INTERVAL}" == "5" ]] || fail \
  "formal CellCF training must preserve checkpoint-every-5"
command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v flock >/dev/null 2>&1 || fail "flock is unavailable"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is unavailable"
command -v sacct >/dev/null 2>&1 || fail "sacct is unavailable"
command -v squeue >/dev/null 2>&1 || fail "squeue is unavailable"
command -v scontrol >/dev/null 2>&1 || fail "scontrol is unavailable"

RECEIPT_DIR="${RUN_ROOT}/submission_receipts"
mkdir -p "${RECEIPT_DIR}"
exec 9>"${RECEIPT_DIR}/submit.lock"
flock -n 9 || fail "another submission process holds the suite lock"

if ! suite_binding_output="$("${PYTHON}" - \
  "${PREPARED_SUBMISSION}" "${PREPARED_SUBMISSION_SHA_FILE}" \
  "${RUN_ROOT}" "${MANIFEST}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


prepared_path = Path(sys.argv[1]).resolve()
sidecar_path = Path(sys.argv[2]).resolve()
run_root = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4]).resolve()
sidecar_sha = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
if not re.fullmatch(r"[0-9a-f]{64}", sidecar_sha):
    raise SystemExit("prepared submission hash sidecar is invalid")
actual_prepared_sha = sha256(prepared_path)
if actual_prepared_sha != sidecar_sha:
    raise SystemExit("prepared submission binding hash drift")

prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
if prepared.get("schema") != "duca_cellcf_prepared_submission_v1":
    raise SystemExit("invalid prepared submission schema")
commit = prepared.get("git_commit")
seed = prepared.get("seed")
cluster = prepared.get("target_cluster")
if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise SystemExit("prepared submission has an invalid commit")
if not isinstance(seed, int) or seed < 0:
    raise SystemExit("prepared submission has an invalid seed")
if not isinstance(cluster, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", cluster):
    raise SystemExit("prepared submission has an invalid target cluster")
if prepared.get("checkpoint_interval") != 5:
    raise SystemExit("prepared submission changed checkpoint-every-5")
if Path(prepared.get("suite_manifest", "")).resolve() != manifest_path:
    raise SystemExit("prepared submission points at a different suite manifest")
manifest_sha = prepared.get("suite_manifest_sha256")
if not isinstance(manifest_sha, str) or sha256(manifest_path) != manifest_sha:
    raise SystemExit("prepared suite manifest hash drift")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "duca_cellcf_suite_manifest_v1" or manifest.get("ok") is not True:
    raise SystemExit("invalid CellCF suite manifest")
if manifest.get("git_commit") != commit or manifest.get("seed") != seed:
    raise SystemExit("prepared submission and suite manifest disagree")

canonical_env = Path(prepared.get("canonical_env_file", "")).resolve()
if sha256(canonical_env) != prepared.get("canonical_env_sha256"):
    raise SystemExit("prepared canonical environment hash drift")
jobs_tsv = Path(prepared.get("jobs_tsv", "")).resolve()
if sha256(jobs_tsv) != prepared.get("jobs_tsv_sha256"):
    raise SystemExit("prepared jobs ledger hash drift")

expected_order = ["uniform", "transition_beta0", "cellcf", "aggregate", "cost", "completion"]
expected_roles = {
    "uniform": "none",
    "transition_beta0": "none",
    "cellcf": "none",
    "aggregate": "afterok_three_arms",
    "cost": "afterok_aggregate",
    "completion": "afterok_aggregate_and_cost",
}
if prepared.get("job_order") != expected_order:
    raise SystemExit("prepared job order does not describe the formal completion DAG")
jobs = prepared.get("jobs")
if not isinstance(jobs, list) or [job.get("key") for job in jobs] != expected_order:
    raise SystemExit("prepared jobs do not cover the exact formal completion DAG")
for job in jobs:
    key = job["key"]
    expected_path = (run_root / "jobs" / f"{key}.sbatch").resolve()
    if Path(job.get("job_file", "")).resolve() != expected_path:
        raise SystemExit(f"prepared {key} job path mismatch")
    job_sha = job.get("job_file_sha256")
    if not isinstance(job_sha, str) or sha256(expected_path) != job_sha:
        raise SystemExit(f"prepared {key} sbatch hash drift")
    if job.get("dependency_role") != expected_roles[key]:
        raise SystemExit(f"prepared {key} dependency role mismatch")
    expected_name = f"cellcf-{key}-s{seed}-{commit[:7]}"
    if job.get("job_name") != expected_name:
        raise SystemExit(f"prepared {key} job name mismatch")
    script = expected_path.read_text(encoding="utf-8")
    for marker in (
        f"#SBATCH --job-name={expected_name}",
        f"#SBATCH --clusters={cluster}",
        f"# DUCA_CELLCF_DEPENDENCY_ROLE={expected_roles[key]}",
        manifest_sha,
        commit,
    ):
        if marker not in script:
            raise SystemExit(f"prepared {key} sbatch is missing binding marker {marker!r}")

gate = manifest.get("real_loader_gate", {}).get("path")
pilot = manifest.get("ddp_pilot", {}).get("path")
if not isinstance(gate, str) or not isinstance(pilot, str):
    raise SystemExit("suite manifest is missing gate or pilot paths")
print(commit)
print(seed)
print(gate)
print(pilot)
print(cluster)
print(manifest_sha)
print(actual_prepared_sha)
PY
)"; then
  fail "prepared suite binding validation failed"
fi
readarray -t suite_binding <<< "${suite_binding_output}"
[[ "${#suite_binding[@]}" == "7" ]] || fail \
  "prepared suite binding must contain exactly seven fields"
EXPECTED_COMMIT="${suite_binding[0]}"
SEED="${suite_binding[1]}"
GATE_JSON="${suite_binding[2]}"
PILOT_JSON="${suite_binding[3]}"
TARGET_CLUSTER="${suite_binding[4]}"
MANIFEST_SHA256="${suite_binding[5]}"
PREPARED_SUBMISSION_SHA256="${suite_binding[6]}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from suite manifest"
if [[ -n "${DUCA_CELLCF_TARGET_CLUSTER:-}" ]]; then
  [[ "${DUCA_CELLCF_TARGET_CLUSTER}" == "${TARGET_CLUSTER}" ]] || fail \
    "requested cluster differs from the prepared suite"
fi

read_prepared_job() {
  local key="$1"
  local expected_role="$2"
  "${PYTHON}" - "${PREPARED_SUBMISSION}" "${PREPARED_SUBMISSION_SHA256}" \
    "${key}" "${expected_role}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prepared_path, expected_sha, key, expected_role = sys.argv[1:]
path = Path(prepared_path).resolve()
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("prepared submission binding changed during submission")
payload = json.loads(path.read_text(encoding="utf-8"))
matches = [job for job in payload["jobs"] if job.get("key") == key]
if len(matches) != 1:
    raise SystemExit(f"prepared job binding missing or duplicated for {key}")
job = matches[0]
if job.get("dependency_role") != expected_role:
    raise SystemExit(f"prepared dependency role mismatch for {key}")
job_file = Path(job["job_file"]).resolve()
if hashlib.sha256(job_file.read_bytes()).hexdigest() != job.get("job_file_sha256"):
    raise SystemExit(f"prepared sbatch hash drift for {key}")
print(job["job_name"])
print(job_file)
print(job["job_file_sha256"])
print(job["dependency_role"])
PY
}

normalize_job_binding() {
  local raw="$1"
  local expected_cluster="$2"
  raw="${raw//$'\r'/}"
  raw="${raw%%$'\n'*}"
  [[ "${raw}" == *";"* ]] || fail \
    "sbatch response does not preserve jobid;cluster identity: ${raw}"
  local job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail \
    "sbatch response has no canonical positive job id: ${raw}"
  local cluster="${raw#*;}"
  [[ "${cluster}" =~ ^[A-Za-z0-9._-]+$ ]] || fail \
    "cannot bind Slurm cluster identity from: ${raw}"
  [[ "${cluster}" == "${expected_cluster}" ]] || fail \
    "sbatch response belongs to cluster ${cluster}, expected ${expected_cluster}"
  printf '%s\t%s;%s\t%s\n' "${job_id}" "${job_id}" "${cluster}" "${cluster}"
}

write_submission_json() {
  local output="$1"
  local status="$2"
  local job_key="$3"
  local job_name="$4"
  local job_file="$5"
  local job_file_sha256="$6"
  local dependency_role="$7"
  local dependency="$8"
  local cluster="$9"
  local token="${10}"
  local raw_response="${11:-}"
  local job_id="${12:-}"
  local job_ref="${13:-}"
  local intent_sha256="${14:-}"
  "${PYTHON}" - "${output}" "${status}" "${job_key}" "${job_name}" \
    "${job_file}" "${job_file_sha256}" "${dependency_role}" "${dependency}" \
    "${cluster}" "${token}" "${raw_response}" "${job_id}" "${job_ref}" \
    "${intent_sha256}" "${MANIFEST}" "${MANIFEST_SHA256}" \
    "${PREPARED_SUBMISSION}" "${PREPARED_SUBMISSION_SHA256}" \
    "${EXPECTED_COMMIT}" "${SEED}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

(
    output,
    status,
    job_key,
    job_name,
    job_file,
    job_file_sha256,
    dependency_role,
    dependency,
    cluster,
    token,
    raw_response,
    job_id,
    job_ref,
    intent_sha256,
    manifest,
    manifest_sha256,
    prepared_submission,
    prepared_submission_sha256,
    commit,
    seed,
) = sys.argv[1:]
payload = {
    "schema_version": "duca_cellcf_slurm_submission_v2",
    "status": status,
    "job_key": job_key,
    "job_name": job_name,
    "job_file": str(Path(job_file).resolve()),
    "job_file_sha256": job_file_sha256,
    "dependency_role": dependency_role,
    "dependency": dependency or None,
    "cluster": cluster,
    "submission_token": token,
    "raw_sbatch_response": raw_response or None,
    "job_id": int(job_id) if job_id else None,
    "job_ref": job_ref or None,
    "intent_sha256": intent_sha256 or None,
    "suite_manifest": str(Path(manifest).resolve()),
    "suite_manifest_sha256": manifest_sha256,
    "prepared_submission": str(Path(prepared_submission).resolve()),
    "prepared_submission_sha256": prepared_submission_sha256,
    "git_commit": commit,
    "seed": int(seed),
}
target = Path(output)
target.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

read_receipt_binding() {
  local intent="$1"
  local receipt="$2"
  local job_key="$3"
  local job_name="$4"
  local job_file="$5"
  local job_file_sha256="$6"
  local dependency_role="$7"
  local dependency="$8"
  local cluster="$9"
  local token="${10}"
  "${PYTHON}" - "${intent}" "${receipt}" "${job_key}" "${job_name}" \
    "${job_file}" "${job_file_sha256}" "${dependency_role}" "${dependency}" \
    "${cluster}" "${token}" "${MANIFEST}" "${MANIFEST_SHA256}" \
    "${PREPARED_SUBMISSION}" "${PREPARED_SUBMISSION_SHA256}" \
    "${EXPECTED_COMMIT}" "${SEED}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    intent_path,
    receipt_path,
    job_key,
    job_name,
    job_file,
    job_file_sha256,
    dependency_role,
    dependency,
    cluster,
    token,
    manifest,
    manifest_sha256,
    prepared_submission,
    prepared_submission_sha256,
    commit,
    seed,
) = sys.argv[1:]
intent_file = Path(intent_path)
receipt_file = Path(receipt_path)
intent = json.loads(intent_file.read_text(encoding="utf-8"))
receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
common = {
    "schema_version": "duca_cellcf_slurm_submission_v2",
    "job_key": job_key,
    "job_name": job_name,
    "job_file": str(Path(job_file).resolve()),
    "job_file_sha256": job_file_sha256,
    "dependency_role": dependency_role,
    "dependency": dependency or None,
    "cluster": cluster,
    "submission_token": token,
    "suite_manifest": str(Path(manifest).resolve()),
    "suite_manifest_sha256": manifest_sha256,
    "prepared_submission": str(Path(prepared_submission).resolve()),
    "prepared_submission_sha256": prepared_submission_sha256,
    "git_commit": commit,
    "seed": int(seed),
}
for label, payload, status in (
    ("intent", intent, "INTENT_RECORDED"),
    ("receipt", receipt, "SUBMITTED"),
):
    expected = {**common, "status": status}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"{label} {key} mismatch: expected {value!r}, got {payload.get(key)!r}"
            )
if any(intent.get(key) is not None for key in ("raw_sbatch_response", "job_id", "job_ref", "intent_sha256")):
    raise SystemExit("submission intent unexpectedly contains a completed job binding")
actual_intent_sha = hashlib.sha256(intent_file.read_bytes()).hexdigest()
if receipt.get("intent_sha256") != actual_intent_sha:
    raise SystemExit("receipt does not bind the exact submission intent")
job_id = receipt.get("job_id")
job_ref = receipt.get("job_ref")
if not isinstance(job_id, int) or job_id <= 0:
    raise SystemExit("receipt has no valid job id")
if job_ref != f"{job_id};{cluster}":
    raise SystemExit("receipt job_ref does not preserve jobid;cluster identity")
raw = receipt.get("raw_sbatch_response")
if not isinstance(raw, str) or not raw.strip():
    raise SystemExit("receipt has no raw sbatch response")
raw_first = raw.replace("\r", "").splitlines()[0]
raw_job_id, separator, raw_cluster = raw_first.partition(";")
if raw_job_id != str(job_id) or separator != ";" or raw_cluster != cluster:
    raise SystemExit("receipt raw sbatch response does not match its job binding")
print(f"{job_id}\t{job_ref}\t{cluster}")
PY
}

submit_once() {
  local job_key="$1"
  local job_name="$2"
  local job_file="$3"
  local job_file_sha256="$4"
  local dependency_role="$5"
  local dependency="${6:-}"
  local target_cluster="$7"
  local token="cellcf-${EXPECTED_COMMIT:0:12}-s${SEED}-${job_key}-${job_file_sha256:0:12}"
  local intent="${RECEIPT_DIR}/${job_key}.intent.json"
  local receipt="${RECEIPT_DIR}/${job_key}.receipt.json"
  local legacy_receipt="${RECEIPT_DIR}/${job_key}.json"
  [[ ! -e "${legacy_receipt}" ]] || fail \
    "legacy ${job_key} receipt is ambiguous; reconcile Slurm manually"
  if [[ -f "${receipt}" ]]; then
    [[ -f "${intent}" ]] || fail "${job_key} receipt exists without its bound intent"
    local existing_binding existing_job_id existing_job_ref existing_cluster
    if ! existing_binding="$(read_receipt_binding "${intent}" "${receipt}" \
      "${job_key}" "${job_name}" "${job_file}" "${job_file_sha256}" \
      "${dependency_role}" "${dependency}" "${target_cluster}" "${token}")"; then
      fail "${job_key} receipt binding validation failed"
    fi
    IFS=$'\t' read -r existing_job_id existing_job_ref existing_cluster <<< "${existing_binding}"
    [[ "${existing_job_id}" =~ ^[1-9][0-9]*$ ]] || fail \
      "${job_key} receipt returned an invalid job id"
    [[ "${existing_job_ref}" == "${existing_job_id};${target_cluster}" ]] || fail \
      "${job_key} receipt returned an invalid job reference"
    [[ "${existing_cluster}" == "${target_cluster}" ]] || fail \
      "${job_key} receipt returned an invalid cluster"
    if ! "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
      --job-id "${existing_job_id}" --job-name "${job_name}" \
      --comment "${token}" --cluster "${existing_cluster}" \
      --job-file "${job_file}" \
      --job-file-sha256 "${job_file_sha256}" \
      --dependency "${dependency}" >/dev/null; then
      fail "${job_key} existing Slurm receipt could not be reopened"
    fi
    printf '%s\t%s\t%s\n' "${existing_job_id}" "${existing_job_ref}" "${existing_cluster}"
    return
  fi
  if [[ -f "${intent}" ]]; then
    fail "ambiguous prior ${job_key} submission: intent exists without receipt; reconcile Slurm manually"
  fi
  [[ "$(sha256_file "${MANIFEST}")" == "${MANIFEST_SHA256}" ]] || fail \
    "suite manifest changed before ${job_key} submission"
  [[ "$(sha256_file "${PREPARED_SUBMISSION}")" == "${PREPARED_SUBMISSION_SHA256}" ]] || fail \
    "prepared submission binding changed before ${job_key} submission"
  [[ "$(sha256_file "${job_file}")" == "${job_file_sha256}" ]] || fail \
    "prepared ${job_key} sbatch changed before submission"
  if ! write_submission_json "${intent}" "INTENT_RECORDED" "${job_key}" "${job_name}" \
    "${job_file}" "${job_file_sha256}" "${dependency_role}" "${dependency}" \
    "${target_cluster}" "${token}"; then
    fail "failed to persist ${job_key} submission intent"
  fi
  local intent_sha256
  if ! intent_sha256="$(sha256_file "${intent}")"; then
    fail "failed to hash persisted ${job_key} submission intent"
  fi
  [[ "${intent_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail \
    "persisted ${job_key} submission intent has an invalid hash"
  local sbatch_args=(
    --parsable
    "--clusters=${target_cluster}"
    "--job-name=${job_name}"
    "--comment=${token}"
  )
  if [[ -n "${dependency}" ]]; then
    sbatch_args+=("--dependency=${dependency}")
  fi
  local raw_response binding job_id job_ref cluster
  if ! raw_response="$(sbatch "${sbatch_args[@]}" "${job_file}")"; then
    fail "sbatch failed for ${job_key}; reconcile the recorded intent before retrying"
  fi
  if ! binding="$(normalize_job_binding "${raw_response}" "${target_cluster}")"; then
    fail "sbatch returned no valid job binding for ${job_key}; reconcile the recorded intent before retrying"
  fi
  IFS=$'\t' read -r job_id job_ref cluster <<< "${binding}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail \
    "${job_key} parsed an invalid job id"
  [[ "${job_ref}" == "${job_id};${target_cluster}" ]] || fail \
    "${job_key} parsed an invalid job reference"
  [[ "${cluster}" == "${target_cluster}" ]] || fail \
    "${job_key} parsed an invalid cluster"
  if ! "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
    --job-id "${job_id}" --job-name "${job_name}" \
    --comment "${token}" --cluster "${cluster}" \
    --job-file "${job_file}" \
    --job-file-sha256 "${job_file_sha256}" \
    --require-scheduler-script \
    --dependency "${dependency}" >/dev/null; then
    fail "new ${job_key} Slurm binding could not be verified"
  fi
  if ! write_submission_json "${receipt}" "SUBMITTED" "${job_key}" "${job_name}" \
    "${job_file}" "${job_file_sha256}" "${dependency_role}" "${dependency}" \
    "${cluster}" "${token}" "${raw_response}" "${job_id}" "${job_ref}" \
    "${intent_sha256}"; then
    fail "failed to persist ${job_key} submission receipt"
  fi
  local persisted_binding persisted_job_id persisted_job_ref persisted_cluster
  if ! persisted_binding="$(read_receipt_binding "${intent}" "${receipt}" \
    "${job_key}" "${job_name}" "${job_file}" "${job_file_sha256}" \
    "${dependency_role}" "${dependency}" "${target_cluster}" "${token}")"; then
    fail "new ${job_key} submission receipt could not be reopened"
  fi
  IFS=$'\t' read -r persisted_job_id persisted_job_ref persisted_cluster <<< \
    "${persisted_binding}"
  [[ "${persisted_job_id}" == "${job_id}" ]] || fail \
    "new ${job_key} receipt changed its job id"
  [[ "${persisted_job_ref}" == "${job_ref}" ]] || fail \
    "new ${job_key} receipt changed its job reference"
  [[ "${persisted_cluster}" == "${cluster}" ]] || fail \
    "new ${job_key} receipt changed its cluster"
  if ! "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
    --job-id "${persisted_job_id}" --job-name "${job_name}" \
    --comment "${token}" --cluster "${persisted_cluster}" \
    --job-file "${job_file}" \
    --job-file-sha256 "${job_file_sha256}" \
    --dependency "${dependency}" >/dev/null; then
    fail "new ${job_key} Slurm receipt could not be reopened"
  fi
  printf '%s\t%s\t%s\n' "${job_id}" "${job_ref}" "${cluster}"
}

all_keys=()
all_job_names=()
all_job_files=()
all_job_hashes=()
all_dependencies=()
all_job_ids=()
all_job_refs=()
all_clusters=()
all_statuses=()

record_binding() {
  all_keys+=("$1")
  all_job_names+=("$2")
  all_job_files+=("$3")
  all_job_hashes+=("$4")
  all_dependencies+=("${5:-}")
  all_job_ids+=("$6")
  all_job_refs+=("$7")
  all_clusters+=("$8")
  all_statuses+=("$9")
}

variants=(uniform transition_beta0 cellcf)
arm_ids=()
arm_refs=()
for variant in "${variants[@]}"; do
  if ! prepared_job_output="$(read_prepared_job "${variant}" "none")"; then
    fail "failed to load prepared ${variant} job binding"
  fi
  readarray -t prepared_job <<< "${prepared_job_output}"
  [[ "${#prepared_job[@]}" == "4" ]] || fail \
    "prepared ${variant} job binding must contain exactly four fields"
  job_name="${prepared_job[0]}"
  job_file="${prepared_job[1]}"
  job_file_sha256="${prepared_job[2]}"
  if ! binding="$(submit_once "${variant}" "${job_name}" "${job_file}" \
    "${job_file_sha256}" "none" "" "${TARGET_CLUSTER}")"; then
    fail "formal ${variant} submission did not produce a valid Slurm binding"
  fi
  IFS=$'\t' read -r job_id job_ref cluster <<< "${binding}"
  arm_ids+=("${job_id}")
  arm_refs+=("${job_ref}")
  record_binding "${variant}" "${job_name}" "${job_file}" "${job_file_sha256}" \
    "" "${job_id}" "${job_ref}" "${cluster}" "SUBMITTED"
done

arm_dependency="afterok:$(IFS=:; echo "${arm_ids[*]}")"
if ! prepared_job_output="$(read_prepared_job "aggregate" "afterok_three_arms")"; then
  fail "failed to load prepared aggregate job binding"
fi
readarray -t prepared_job <<< "${prepared_job_output}"
[[ "${#prepared_job[@]}" == "4" ]] || fail \
  "prepared aggregate job binding must contain exactly four fields"
aggregate_name="${prepared_job[0]}"
aggregate_job="${prepared_job[1]}"
aggregate_sha256="${prepared_job[2]}"
if ! aggregate_binding="$(submit_once "aggregate" "${aggregate_name}" \
  "${aggregate_job}" "${aggregate_sha256}" "afterok_three_arms" \
  "${arm_dependency}" "${TARGET_CLUSTER}")"; then
  fail "formal aggregate submission did not produce a valid Slurm binding"
fi
IFS=$'\t' read -r aggregate_id aggregate_ref aggregate_cluster <<< "${aggregate_binding}"
record_binding "aggregate" "${aggregate_name}" "${aggregate_job}" "${aggregate_sha256}" \
  "${arm_dependency}" "${aggregate_id}" "${aggregate_ref}" "${aggregate_cluster}" \
  "DEPENDENCY_SUBMITTED"

cost_dependency="afterok:${aggregate_id}"
if ! prepared_job_output="$(read_prepared_job "cost" "afterok_aggregate")"; then
  fail "failed to load prepared cost job binding"
fi
readarray -t prepared_job <<< "${prepared_job_output}"
[[ "${#prepared_job[@]}" == "4" ]] || fail \
  "prepared cost job binding must contain exactly four fields"
cost_name="${prepared_job[0]}"
cost_job="${prepared_job[1]}"
cost_sha256="${prepared_job[2]}"
if ! cost_binding="$(submit_once "cost" "${cost_name}" "${cost_job}" \
  "${cost_sha256}" "afterok_aggregate" "${cost_dependency}" \
  "${TARGET_CLUSTER}")"; then
  fail "formal cost submission did not produce a valid Slurm binding"
fi
IFS=$'\t' read -r cost_id cost_ref cost_cluster <<< "${cost_binding}"
record_binding "cost" "${cost_name}" "${cost_job}" "${cost_sha256}" \
  "${cost_dependency}" "${cost_id}" "${cost_ref}" "${cost_cluster}" \
  "DEPENDENCY_SUBMITTED"

completion_dependency="afterok:${aggregate_id}:${cost_id}"
if ! prepared_job_output="$(read_prepared_job "completion" \
  "afterok_aggregate_and_cost")"; then
  fail "failed to load prepared completion job binding"
fi
readarray -t prepared_job <<< "${prepared_job_output}"
[[ "${#prepared_job[@]}" == "4" ]] || fail \
  "prepared completion job binding must contain exactly four fields"
completion_name="${prepared_job[0]}"
completion_job="${prepared_job[1]}"
completion_sha256="${prepared_job[2]}"
if ! completion_binding="$(submit_once "completion" "${completion_name}" \
  "${completion_job}" "${completion_sha256}" \
  "afterok_aggregate_and_cost" "${completion_dependency}" \
  "${TARGET_CLUSTER}")"; then
  fail "formal completion submission did not produce a valid Slurm binding"
fi
IFS=$'\t' read -r completion_id completion_ref completion_cluster <<< "${completion_binding}"
record_binding "completion" "${completion_name}" "${completion_job}" "${completion_sha256}" \
  "${completion_dependency}" "${completion_id}" "${completion_ref}" \
  "${completion_cluster}" "FORMAL_COMPLETION_SUBMITTED"

ledger_tmp="${RUN_ROOT}/jobs.submitted.tsv.tmp.$$"
printf 'job_key\tseed\tcommit\tmanifest_sha256\tsbatch_file\tsbatch_sha256\tjob_name\tdependency\tjob_id\tjob_ref\tcluster\tstatus\n' > "${ledger_tmp}"
for index in "${!all_keys[@]}"; do
  dependency="${all_dependencies[$index]:-none}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${all_keys[$index]}" "${SEED}" "${EXPECTED_COMMIT}" "${MANIFEST_SHA256}" \
    "${all_job_files[$index]}" "${all_job_hashes[$index]}" \
    "${all_job_names[$index]}" "${dependency}" "${all_job_ids[$index]}" \
    "${all_job_refs[$index]}" "${all_clusters[$index]}" "${all_statuses[$index]}" \
    >> "${ledger_tmp}"
done
mv "${ledger_tmp}" "${RUN_ROOT}/jobs.submitted.tsv"

printf '[DUCA_CELLCF_SUBMIT] arms=%s aggregate=%s cost=%s completion=%s root=%s\n' \
  "$(IFS=,; echo "${arm_refs[*]}")" "${aggregate_ref}" "${cost_ref}" \
  "${completion_ref}" "${RUN_ROOT}"
