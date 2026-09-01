#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_SUBMIT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
[[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || fail "RUN_ROOT must name a prepared suite"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
[[ -f "${MANIFEST}" ]] || fail "suite manifest is missing"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "submission requires a clean git tree"
command -v flock >/dev/null 2>&1 || fail "flock is required for idempotent submission"

SUBMISSION_DIR="${RUN_ROOT}/submission_receipts"
mkdir -p "${SUBMISSION_DIR}"
exec 9>"${SUBMISSION_DIR}/submit.lock"
flock -n 9 || fail "another suite submission process holds the lock"

readarray -t suite_binding < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p["git_commit"])
print(p["seed"])
print(p["formal_core_gate"]["path"])
print(p["formal_ddp_pilot"]["path"])
PY
)
EXPECTED_COMMIT="${suite_binding[0]}"
SEED="${suite_binding[1]}"
CORE_GATE_JSON="${suite_binding[2]}"
DDP_PILOT_JSON="${suite_binding[3]}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from suite manifest"

normalize_job_binding() {
  local raw="$1"
  raw="${raw%%$'\n'*}"
  local job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: ${raw}"
  local cluster=""
  if [[ "${raw}" == *";"* ]]; then
    cluster="${raw#*;}"
  else
    cluster="${SLURM_CLUSTER_NAME:-}"
    if [[ -z "${cluster}" ]]; then
      cluster="$(scontrol show config | awk -F= '/^ClusterName/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
    fi
  fi
  [[ "${cluster}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "cannot bind Slurm cluster identity from: ${raw}"
  printf '%s\t%s;%s\t%s\n' "${job_id}" "${job_id}" "${cluster}" "${cluster}"
}

write_submission_json() {
  local output="$1"
  local status="$2"
  local variant="$3"
  local job_file="$4"
  local token="$5"
  local raw_response="${6:-}"
  local job_id="${7:-}"
  local job_ref="${8:-}"
  local cluster="${9:-}"
  "${PYTHON}" - "${output}" "${status}" "${variant}" "${job_file}" \
    "${token}" "${raw_response}" "${job_id}" "${job_ref}" "${cluster}" \
    "${EXPECTED_COMMIT}" "${SEED}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

out, status, variant, job_file, token, raw, job_id, job_ref, cluster, commit, seed = sys.argv[1:]
payload = {
    "schema_version": "duca_p0_slurm_submission_v2",
    "status": status,
    "variant": variant,
    "job_file": str(Path(job_file).resolve()),
    "submission_token": token,
    "raw_sbatch_response": raw or None,
    "job_id": int(job_id) if job_id else None,
    "job_ref": job_ref or None,
    "cluster": cluster or None,
    "git_commit": commit,
    "seed": int(seed),
}
target = Path(out)
target.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY
}

read_receipt_binding() {
  local receipt="$1"
  local variant="$2"
  local job_file="$3"
  "${PYTHON}" - "${receipt}" "${variant}" "${job_file}" "${EXPECTED_COMMIT}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

receipt, variant, job_file, commit, seed = sys.argv[1:]
p = json.load(open(receipt, encoding="utf-8"))
expected = {
    "schema_version": "duca_p0_slurm_submission_v2",
    "status": "SUBMITTED",
    "variant": variant,
    "job_file": str(Path(job_file).resolve()),
    "git_commit": commit,
    "seed": int(seed),
}
for key, value in expected.items():
    if p.get(key) != value:
        raise SystemExit(f"receipt {key} mismatch: expected {value!r}, got {p.get(key)!r}")
job_id = p.get("job_id")
if not isinstance(job_id, int) or job_id <= 0:
    raise SystemExit("receipt has no valid job id")
cluster = p.get("cluster")
job_ref = p.get("job_ref")
if not isinstance(cluster, str) or not cluster:
    raise SystemExit("receipt has no Slurm cluster identity")
if job_ref != f"{job_id};{cluster}":
    raise SystemExit("receipt job_ref does not preserve jobid;cluster identity")
print(f"{job_id}\t{job_ref}\t{cluster}")
PY
}

submit_once() {
  local variant="$1"
  local job_file="$2"
  local dependency_arg="${3:-}"
  local target_cluster="${4:-}"
  local token="duca-p0-${EXPECTED_COMMIT:0:12}-seed${SEED}-${variant}"
  local intent="${SUBMISSION_DIR}/${variant}.intent.json"
  local receipt="${SUBMISSION_DIR}/${variant}.receipt.json"
  if [[ -f "${receipt}" ]]; then
    read_receipt_binding "${receipt}" "${variant}" "${job_file}"
    return
  fi
  if [[ -f "${intent}" ]]; then
    fail "ambiguous prior ${variant} submission: intent exists without receipt; reconcile Slurm manually"
  fi
  write_submission_json "${intent}" "INTENT_RECORDED" "${variant}" "${job_file}" "${token}"
  local raw_response
  local sbatch_args=(--parsable --comment="${token}")
  if [[ -n "${target_cluster}" ]]; then
    sbatch_args+=(--clusters="${target_cluster}")
  fi
  if [[ -n "${dependency_arg}" ]]; then
    sbatch_args+=(--dependency="${dependency_arg}")
  fi
  raw_response="$(sbatch "${sbatch_args[@]}" "${job_file}")"
  local binding job_id job_ref cluster
  binding="$(normalize_job_binding "${raw_response}")"
  IFS=$'\t' read -r job_id job_ref cluster <<< "${binding}"
  write_submission_json "${receipt}" "SUBMITTED" "${variant}" "${job_file}" \
    "${token}" "${raw_response}" "${job_id}" "${job_ref}" "${cluster}"
  printf '%s\t%s\t%s\n' "${job_id}" "${job_ref}" "${cluster}"
}

variants=(uniform direct transition_beta0 transition_counterfactual)
job_ids=()
job_refs=()
job_clusters=()
for variant in "${variants[@]}"; do
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  [[ -f "${job_file}" ]] || fail "missing job file: ${job_file}"
  target_cluster=""
  if [[ ${#job_clusters[@]} -gt 0 ]]; then
    target_cluster="${job_clusters[0]}"
  fi
  binding="$(submit_once "${variant}" "${job_file}" "" "${target_cluster}")"
  IFS=$'\t' read -r job_id job_ref cluster <<< "${binding}"
  if [[ ${#job_clusters[@]} -gt 0 && "${cluster}" != "${job_clusters[0]}" ]]; then
    fail "formal arm receipt belongs to a different Slurm cluster"
  fi
  job_ids+=("${job_id}")
  job_refs+=("${job_ref}")
  job_clusters+=("${cluster}")
done

ledger_tmp="${RUN_ROOT}/jobs.submitted.tsv.tmp.$$"
printf 'variant\tseed\tcommit\tsbatch_file\tjob_id\tjob_ref\tcluster\tstatus\n' > "${ledger_tmp}"
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${SEED}" "${EXPECTED_COMMIT}" \
    "${RUN_ROOT}/jobs/${variant}.sbatch" "${job_ids[$index]}" \
    "${job_refs[$index]}" "${job_clusters[$index]}" "SUBMITTED" \
    >> "${ledger_tmp}"
done
mv "${ledger_tmp}" "${RUN_ROOT}/jobs.submitted.tsv"

dependency="$(IFS=:; echo "${job_ids[*]}")"
aggregate_job="${RUN_ROOT}/jobs/aggregate.sbatch"
cat > "${aggregate_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca-p0-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=${RUN_ROOT}/logs/aggregate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/aggregate-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
'${PYTHON}' -m tools.bata.validate_duca_transition_only_p0_suite \
  --repo-root '${REPO_ROOT}' \
  --seed '${SEED}' \
  --expected-commit '${EXPECTED_COMMIT}' \
  --require-clean \
  --core-gate-json '${CORE_GATE_JSON}' \
  --ddp-pilot-json '${DDP_PILOT_JSON}' \
  --require-ddp-pilot \
  --post-run-evidence 'uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  --post-run-evidence 'direct=${RUN_ROOT}/logs/direct/post_run_evidence.json' \
  --post-run-evidence 'transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  --post-run-evidence 'transition_counterfactual=${RUN_ROOT}/logs/transition_counterfactual/post_run_evidence.json' \
  --output-json '${RUN_ROOT}/final_suite_evidence.json'
EOF
chmod 0755 "${aggregate_job}"
bash -n "${aggregate_job}" || fail "aggregate job syntax is invalid"
aggregate_binding="$(submit_once "aggregate" "${aggregate_job}" "afterok:${dependency}" "${job_clusters[0]}")"
IFS=$'\t' read -r aggregate_id aggregate_ref aggregate_cluster <<< "${aggregate_binding}"
[[ "${aggregate_cluster}" == "${job_clusters[0]}" ]] \
  || fail "aggregate job was submitted to a different Slurm cluster"
printf 'aggregate\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${SEED}" "${EXPECTED_COMMIT}" "${aggregate_job}" "${aggregate_id}" \
  "${aggregate_ref}" "${aggregate_cluster}" "DEPENDENCY_SUBMITTED" \
  >> "${RUN_ROOT}/jobs.submitted.tsv"

printf '[DUCA_P0_SUBMIT] jobs=%s aggregate=%s run_root=%s\n' \
  "$(IFS=,; echo "${job_refs[*]}")" "${aggregate_ref}" "${RUN_ROOT}"
