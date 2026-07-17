#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH

fail() {
  echo "[DUCA_CELLCF_COST_RECOVERY][FAIL] $*" >&2
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
FORMAL_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-}"
TRAINED_REPO_ROOT="${DUCA_CELLCF_TRAINED_REPO_ROOT:-}"
TRAINED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
EVIDENCE_COMMIT="$(git rev-parse HEAD)"
EXPECTED_EVIDENCE_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:-}"
TARGET_CLUSTER="${DUCA_CELLCF_TARGET_CLUSTER:-n16r4}"
AGGREGATE="${FORMAL_ROOT}/aggregate_suite_evidence.json"
FINAL_SUITE="${FORMAL_ROOT}/final_suite_evidence.json"
ORIGINAL_LEDGER="${FORMAL_ROOT}/jobs.submitted.tsv"
EXPECTED_AGGREGATE_SHA256="${DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256:-}"
EXPECTED_ORIGINAL_LEDGER_SHA256="${DUCA_CELLCF_ORIGINAL_LEDGER_SHA256:-}"
RECOVERY_ROOT="${DUCA_CELLCF_COST_RECOVERY_ROOT:-${FORMAL_ROOT}/cost_recovery_${EVIDENCE_COMMIT:0:7}_v1}"
COST_ROOT="${RECOVERY_ROOT}/cost"
COST_EVIDENCE="${COST_ROOT}/cellcf_vs_bare_uniform.json"
INTENT="${RECOVERY_ROOT}/submission_intent.json"
MANIFEST="${RECOVERY_ROOT}/submission_manifest.json"
LEDGER="${RECOVERY_ROOT}/jobs.submitted.tsv"
ORIGINAL_FAILURE="${RECOVERY_ROOT}/original_failure_receipt.json"
SAMPLES="${DUCA_CELLCF_COST_SAMPLES:-500}"
WARMUP="${DUCA_CELLCF_COST_WARMUP:-20}"
REPEATS="${DUCA_CELLCF_COST_REPEATS:-3}"
SUPPORTED_TRAINED_COMMIT="1642f265e48391418a7c8a4a087e33e2b7bf6899"

for command in git sbatch scancel scontrol sacct sha256sum flock; do
  command -v "${command}" >/dev/null 2>&1 || fail "${command} is unavailable"
done
[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -d "${FORMAL_ROOT}" ]] || fail "formal run root is missing"
FORMAL_ROOT="$(
  duca_cellcf_require_external_path \
    "FORMAL_ROOT" "${EVIDENCE_REPO_ROOT}" "${BASE}" "${FORMAL_ROOT}"
)" || fail "formal run root violates the path contract"
RECOVERY_ROOT="$(
  duca_cellcf_require_external_path \
    "RECOVERY_ROOT" "${EVIDENCE_REPO_ROOT}" "${BASE}" "${RECOVERY_ROOT}"
)" || fail "recovery root violates the path contract"
case "${RECOVERY_ROOT}" in
  "${FORMAL_ROOT}"/*) ;;
  *) fail "recovery root must stay inside the formal run root" ;;
esac
[[ ! -e "${RECOVERY_ROOT}" ]] || fail "refusing to reuse an existing recovery root"
[[ ! -e "${FINAL_SUITE}" ]] || fail "final suite evidence already exists"
[[ -d "${TRAINED_REPO_ROOT}" ]] || fail "trained repository is missing"
[[ "${TRAINED_COMMIT}" == "${SUPPORTED_TRAINED_COMMIT}" ]] \
  || fail "unsupported trained commit"
[[ "${EXPECTED_EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "DUCA_EVIDENCE_EXPECTED_COMMIT is required"
[[ "${EVIDENCE_COMMIT}" == "${EXPECTED_EVIDENCE_COMMIT}" ]] \
  || fail "evidence repository commit drift"
[[ "${EVIDENCE_COMMIT}" != "${TRAINED_COMMIT}" ]] \
  || fail "trained and evidence commits must be distinct"
[[ "$(git -C "${TRAINED_REPO_ROOT}" rev-parse HEAD)" == "${TRAINED_COMMIT}" ]] \
  || fail "trained repository commit drift"
[[ -z "$(git -C "${TRAINED_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || fail "trained repository is dirty"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "evidence repository is dirty"
for repository in "${TRAINED_REPO_ROOT}" "${EVIDENCE_REPO_ROOT}"; do
  ignored="$(
    git -C "${repository}" ls-files --others --ignored --exclude-standard -- \
      '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py'
  )"
  [[ -z "${ignored}" ]] || fail "ignored Python sources shadow ${repository}"
done
[[ -f "${AGGREGATE}" ]] || fail "aggregate evidence is missing"
[[ -f "${ORIGINAL_LEDGER}" ]] || fail "original submitted-job ledger is missing"
[[ "${EXPECTED_AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256 is required"
[[ "${EXPECTED_ORIGINAL_LEDGER_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_ORIGINAL_LEDGER_SHA256 is required"
[[ "$(sha256_file "${AGGREGATE}")" == "${EXPECTED_AGGREGATE_SHA256}" ]] \
  || fail "aggregate evidence hash mismatch"
[[ "$(sha256_file "${ORIGINAL_LEDGER}")" == "${EXPECTED_ORIGINAL_LEDGER_SHA256}" ]] \
  || fail "original submitted-job ledger hash mismatch"
[[ "${SAMPLES}" =~ ^[0-9]+$ && "${SAMPLES}" -ge 500 ]] \
  || fail "at least 500 measured windows are required"
[[ "${WARMUP}" =~ ^[0-9]+$ ]] || fail "warmup count is invalid"
[[ "${REPEATS}" =~ ^[0-9]+$ && "${REPEATS}" -ge 3 ]] \
  || fail "at least three paired repeats are required"

mkdir -p "${RECOVERY_ROOT}/jobs" "${RECOVERY_ROOT}/logs" \
  "${RECOVERY_ROOT}/receipts"
exec 9>"${RECOVERY_ROOT}/submission.lock"
flock -n 9 || fail "another recovery submission owns this root"

readarray -t aggregate_binding < <(
  "${PYTHON}" - "${AGGREGATE}" "${TRAINED_COMMIT}" \
    "${EXPECTED_AGGREGATE_SHA256}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
expected_sha = sys.argv[3]
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("aggregate evidence hash drift")
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("schema") != "duca_cellcf_suite_manifest_v1"
    or payload.get("ok") is not True
    or payload.get("status") != "runs_complete_cost_pending"
    or payload.get("git_commit") != commit
    or payload.get("seed") != 0
):
    raise SystemExit("aggregate evidence identity/status mismatch")
for key in ("real_loader_gate", "ddp_pilot"):
    record = payload.get(key)
    if not isinstance(record, dict):
        raise SystemExit(f"aggregate {key} binding is missing")
    artifact = Path(str(record.get("path") or "")).resolve()
    digest = str(record.get("sha256") or "")
    if (
        not artifact.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
    ):
        raise SystemExit(f"aggregate {key} binding changed")
    print(artifact)
    print(digest)
PY
)
[[ "${#aggregate_binding[@]}" == 4 ]] \
  || fail "failed to reopen aggregate gate/pilot bindings"
GATE_JSON="${aggregate_binding[0]}"
GATE_SHA256="${aggregate_binding[1]}"
PILOT_JSON="${aggregate_binding[2]}"
PILOT_SHA256="${aggregate_binding[3]}"

readarray -t original_binding < <(
  "${PYTHON}" - "${ORIGINAL_LEDGER}" "${TRAINED_COMMIT}" \
    "${EXPECTED_ORIGINAL_LEDGER_SHA256}" <<'PY'
import csv
import hashlib
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
expected_sha = sys.argv[3]
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("original ledger hash drift")
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
keys = ["uniform", "transition_beta0", "cellcf", "aggregate", "cost", "completion"]
if [row.get("job_key") for row in rows] != keys:
    raise SystemExit("original ledger is not the exact six-job DAG")
for row in rows:
    if row.get("commit") != commit:
        raise SystemExit(f"original {row.get('job_key')} commit mismatch")
    if re.fullmatch(r"[1-9][0-9]*", str(row.get("job_id") or "")) is None:
        raise SystemExit(f"original {row.get('job_key')} job id is invalid")
cost = rows[4]
completion = rows[5]
if cost.get("dependency") != f"afterok:{rows[3]['job_id']}":
    raise SystemExit("original cost dependency mismatch")
if completion.get("dependency") != (
    f"afterok:{rows[3]['job_id']}:{cost['job_id']}"
):
    raise SystemExit("original completion dependency mismatch")
if cost.get("cluster") != completion.get("cluster"):
    raise SystemExit("original terminal jobs span multiple clusters")
print(cost["job_id"])
print(cost["job_name"])
print(completion["job_id"])
print(completion["job_name"])
print(cost["cluster"])
PY
)
[[ "${#original_binding[@]}" == 5 ]] \
  || fail "failed to reopen original terminal job binding"
ORIGINAL_COST_ID="${original_binding[0]}"
ORIGINAL_COST_NAME="${original_binding[1]}"
ORIGINAL_COMPLETION_ID="${original_binding[2]}"
ORIGINAL_COMPLETION_NAME="${original_binding[3]}"
ORIGINAL_CLUSTER="${original_binding[4]}"
[[ "${ORIGINAL_CLUSTER}" == "${TARGET_CLUSTER}" ]] \
  || fail "recovery cluster differs from the original DAG"

sacct -X -M "${TARGET_CLUSTER}" \
  -j "${ORIGINAL_COST_ID},${ORIGINAL_COMPLETION_ID}" -n -P \
  -o JobIDRaw,JobName%128,Cluster,State,ExitCode,ElapsedRaw \
  > "${RECOVERY_ROOT}/receipts/original_terminal_jobs.sacct"
"${PYTHON}" - "${RECOVERY_ROOT}/receipts/original_terminal_jobs.sacct" \
  "${ORIGINAL_COST_ID}" "${ORIGINAL_COST_NAME}" \
  "${ORIGINAL_COMPLETION_ID}" "${ORIGINAL_COMPLETION_NAME}" \
  "${TARGET_CLUSTER}" "${ORIGINAL_LEDGER}" \
  "${EXPECTED_ORIGINAL_LEDGER_SHA256}" "${ORIGINAL_FAILURE}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

sacct_path = Path(sys.argv[1]).resolve()
cost_id, cost_name = sys.argv[2], sys.argv[3]
completion_id, completion_name = sys.argv[4], sys.argv[5]
cluster = sys.argv[6]
ledger = Path(sys.argv[7]).resolve()
ledger_sha = sys.argv[8]
output = Path(sys.argv[9]).resolve()
rows = {}
for line in sacct_path.read_text(encoding="utf-8").splitlines():
    fields = line.split("|")
    if len(fields) >= 6 and fields[0] in {cost_id, completion_id}:
        if fields[0] in rows:
            raise SystemExit("duplicate original terminal scheduler row")
        rows[fields[0]] = fields[:6]
if set(rows) != {cost_id, completion_id}:
    raise SystemExit("missing original terminal scheduler row")
cost = rows[cost_id]
completion = rows[completion_id]
if cost[1] != cost_name or cost[2] != cluster:
    raise SystemExit("original cost scheduler identity mismatch")
if cost[3] != "FAILED" or cost[4] != "1:0":
    raise SystemExit("original cost is not uniquely FAILED/1:0")
if completion[1] != completion_name or completion[2] != cluster:
    raise SystemExit("original completion scheduler identity mismatch")
if re.fullmatch(r"CANCELLED(?: by [1-9][0-9]*)?", completion[3]) is None:
    raise SystemExit("original completion has a non-canonical cancelled state")
if completion[4] != "0:0" or int(completion[5]) != 0:
    raise SystemExit("original completion is not uniquely cancelled/0:0/zero-runtime")
payload = {
    "schema": "duca_cellcf_cost_recovery_original_failure_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "original_formal_ledger_path": str(ledger),
    "original_formal_ledger_sha256": ledger_sha,
    "scheduler_query_path": str(sacct_path),
    "scheduler_query_sha256": hashlib.sha256(sacct_path.read_bytes()).hexdigest(),
    "cost": {
        "job_id": int(cost_id),
        "job_name": cost_name,
        "cluster": cluster,
        "state": cost[3],
        "exit_code": cost[4],
        "elapsed_raw_seconds": int(cost[5]),
    },
    "completion": {
        "job_id": int(completion_id),
        "job_name": completion_name,
        "cluster": cluster,
        "state": completion[3],
        "exit_code": completion[4],
        "elapsed_raw_seconds": int(completion[5]),
    },
    "interpretation": (
        "original cost profiles are diagnostic only; recovery reruns fresh paired "
        "profiles and does not rewrite the original six-job ledger"
    ),
}
canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
with output.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
ORIGINAL_FAILURE_SHA256="$(sha256_file "${ORIGINAL_FAILURE}")"

"${PYTHON}" - "${INTENT}" "${FORMAL_ROOT}" "${RECOVERY_ROOT}" \
  "${TRAINED_REPO_ROOT}" "${TRAINED_COMMIT}" "${EVIDENCE_REPO_ROOT}" \
  "${EVIDENCE_COMMIT}" "${TARGET_CLUSTER}" "${AGGREGATE}" \
  "${EXPECTED_AGGREGATE_SHA256}" "${ORIGINAL_LEDGER}" \
  "${EXPECTED_ORIGINAL_LEDGER_SHA256}" "${ORIGINAL_FAILURE}" \
  "${ORIGINAL_FAILURE_SHA256}" "${COST_ROOT}" "${COST_EVIDENCE}" \
  "${FINAL_SUITE}" "${SAMPLES}" "${WARMUP}" "${REPEATS}" \
  "${GATE_JSON}" "${GATE_SHA256}" "${PILOT_JSON}" \
  "${PILOT_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output, formal_root, recovery_root, trained_root, trained_commit,
    evidence_root, evidence_commit, cluster, aggregate, aggregate_sha,
    original_ledger, original_ledger_sha, failure, failure_sha, cost_root,
    cost_evidence, final_suite, samples, warmup, repeats, gate, gate_sha,
    pilot, pilot_sha,
) = sys.argv[1:]
payload = {
    "schema": "duca_cellcf_cost_recovery_intent_v1",
    "status": "INTENT_RECORDED",
    "task": "offline_temporal_action_detection",
    "formal_run_root": str(Path(formal_root).resolve()),
    "recovery_root": str(Path(recovery_root).resolve()),
    "trained_repository": str(Path(trained_root).resolve()),
    "trained_git_commit": trained_commit,
    "cost_producer_repository": str(Path(evidence_root).resolve()),
    "cost_producer_evidence_commit": evidence_commit,
    "target_cluster": cluster,
    "aggregate_evidence_path": str(Path(aggregate).resolve()),
    "aggregate_evidence_sha256": aggregate_sha,
    "real_loader_gate_path": str(Path(gate).resolve()),
    "real_loader_gate_sha256": gate_sha,
    "ddp_pilot_path": str(Path(pilot).resolve()),
    "ddp_pilot_sha256": pilot_sha,
    "original_formal_ledger_path": str(Path(original_ledger).resolve()),
    "original_formal_ledger_sha256": original_ledger_sha,
    "original_failure_receipt_path": str(Path(failure).resolve()),
    "original_failure_receipt_sha256": failure_sha,
    "cost_root": str(Path(cost_root).resolve()),
    "cost_evidence_path": str(Path(cost_evidence).resolve()),
    "final_suite_evidence_path": str(Path(final_suite).resolve()),
    "profiling_protocol": {
        "samples_per_repeat": int(samples),
        "warmup_samples": int(warmup),
        "repeats_per_method": int(repeats),
        "paired_order": "alternating",
        "checkpoint": "terminal_epoch_131_state_dict_ema",
    },
    "recovery_scope": (
        "rerun cost profiling and final suite validation only; do not rerun or "
        "rewrite the three completed 132-epoch training arms"
    ),
}
canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
with Path(output).open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
INTENT_SHA256="$(sha256_file "${INTENT}")"

COST_JOB_NAME="cellcf-cost-r-${EVIDENCE_COMMIT:0:7}"
COMPLETION_JOB_NAME="cellcf-finish-r-${EVIDENCE_COMMIT:0:7}"
COST_JOB="${RECOVERY_ROOT}/jobs/cost.sbatch"
COMPLETION_JOB="${RECOVERY_ROOT}/jobs/completion.sbatch"
cat > "${COST_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${COST_JOB_NAME}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=${RECOVERY_ROOT}/logs/cost-%j.out
#SBATCH --error=${RECOVERY_ROOT}/logs/cost-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
cd '${EVIDENCE_REPO_ROOT}'
[[ "\$(git rev-parse HEAD)" == '${EVIDENCE_COMMIT}' ]] || { echo '[COST_RECOVERY][FAIL] evidence commit drift' >&2; exit 1; }
[[ -z "\$(git status --porcelain --untracked-files=normal)" ]] || { echo '[COST_RECOVERY][FAIL] dirty evidence tree' >&2; exit 1; }
[[ "\$(sha256sum '${INTENT}' | awk '{print \$1}')" == '${INTENT_SHA256}' ]] || { echo '[COST_RECOVERY][FAIL] intent hash drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${COST_JOB_NAME}' ]] || { echo '[COST_RECOVERY][FAIL] job-name drift' >&2; exit 1; }
export BASE='${BASE}'
export DUCA_CELLCF_TRAINING_PROFILE='exposure132'
export DUCA_EXPECTED_COMMIT='${TRAINED_COMMIT}'
export DUCA_EVIDENCE_EXPECTED_COMMIT='${EVIDENCE_COMMIT}'
export DUCA_CELLCF_CHECKPOINT='${FORMAL_ROOT}/work_dirs/cellcf/gpu1_id0/checkpoint/epoch_131.pth'
export DUCA_CELLCF_COST_ROOT='${COST_ROOT}'
export DUCA_CELLCF_AGGREGATE_EVIDENCE='${AGGREGATE}'
export DUCA_CELLCF_COST_EVIDENCE='${COST_EVIDENCE}'
export DUCA_CELLCF_COST_SAMPLES='${SAMPLES}'
export DUCA_CELLCF_COST_WARMUP='${WARMUP}'
export DUCA_CELLCF_COST_REPEATS='${REPEATS}'
bash scripts/run_duca_cellcf_cost_pair.sh \
  --suite-manifest '${INTENT}' \
  --suite-manifest-sha256 '${INTENT_SHA256}' \
  --aggregate-evidence '${AGGREGATE}' \
  --output-json '${COST_EVIDENCE}'
EOF
cat > "${COMPLETION_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${COMPLETION_JOB_NAME}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=${RECOVERY_ROOT}/logs/completion-%j.out
#SBATCH --error=${RECOVERY_ROOT}/logs/completion-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
cd '${EVIDENCE_REPO_ROOT}'
[[ "\$(git rev-parse HEAD)" == '${EVIDENCE_COMMIT}' ]] || { echo '[COST_RECOVERY_COMPLETION][FAIL] evidence commit drift' >&2; exit 1; }
[[ -z "\$(git status --porcelain --untracked-files=normal)" ]] || { echo '[COST_RECOVERY_COMPLETION][FAIL] dirty evidence tree' >&2; exit 1; }
[[ "\$(sha256sum '${INTENT}' | awk '{print \$1}')" == '${INTENT_SHA256}' ]] || { echo '[COST_RECOVERY_COMPLETION][FAIL] intent hash drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${COMPLETION_JOB_NAME}' ]] || { echo '[COST_RECOVERY_COMPLETION][FAIL] job-name drift' >&2; exit 1; }
export BASE='${BASE}'
export DUCA_CELLCF_TRAINING_PROFILE='exposure132'
source scripts/duca_cellcf_canonical_env.sh
'${PYTHON}' -m tools.bata.validate_duca_cellcf_suite \
  --repo-root '${TRAINED_REPO_ROOT}' --seed 0 \
  --expected-commit '${TRAINED_COMMIT}' --require-clean \
  --evidence-repo-root '${EVIDENCE_REPO_ROOT}' \
  --expected-evidence-commit '${EVIDENCE_COMMIT}' \
  --gate-json '${GATE_JSON}' \
  --pilot-json '${PILOT_JSON}' \
  --post-run-evidence 'uniform=${FORMAL_ROOT}/logs/uniform/post_run_evidence.json' \
  --post-run-evidence 'transition_beta0=${FORMAL_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  --post-run-evidence 'cellcf=${FORMAL_ROOT}/logs/cellcf/post_run_evidence.json' \
  --cost-evidence '${COST_EVIDENCE}' --require-cost-evidence \
  --output-json '${FINAL_SUITE}'
EOF
chmod 0755 "${COST_JOB}" "${COMPLETION_JOB}"
bash -n "${COST_JOB}" || fail "generated cost job has invalid syntax"
bash -n "${COMPLETION_JOB}" || fail "generated completion job has invalid syntax"
COST_JOB_SHA256="$(sha256_file "${COST_JOB}")"
COMPLETION_JOB_SHA256="$(sha256_file "${COMPLETION_JOB}")"

"${PYTHON}" -m tools.bata.reconcile_duca_cellcf_slurm_submission \
  fsync-artifacts \
  --file "${INTENT}" --file "${ORIGINAL_FAILURE}" \
  --file "${RECOVERY_ROOT}/receipts/original_terminal_jobs.sacct" \
  --file "${COST_JOB}" --file "${COMPLETION_JOB}" \
  --directory "${FORMAL_ROOT}" \
  --directory "${RECOVERY_ROOT}" \
  --directory "${RECOVERY_ROOT}/jobs" \
  --directory "${RECOVERY_ROOT}/logs" \
  --directory "${RECOVERY_ROOT}/receipts" >/dev/null

token_prefix="duca-cost-recovery-${EVIDENCE_COMMIT:0:12}"
COST_TOKEN="${token_prefix}-cost"
COMPLETION_TOKEN="${token_prefix}-completion"
CURRENT_USER="${SLURM_JOB_USER:-${USER:-}}"
[[ -n "${CURRENT_USER}" ]] || fail "Slurm submission user is unavailable"

submitted_ids=()
released=0
rollback() {
  status=$?
  if [[ "${status}" -ne 0 && "${released}" -eq 0 ]]; then
    if [[ "${#submitted_ids[@]}" -eq 0 ]]; then
      for binding in \
        "${COST_TOKEN}|${COST_JOB_NAME}|${COST_JOB}" \
        "${COMPLETION_TOKEN}|${COMPLETION_JOB_NAME}|${COMPLETION_JOB}"; do
        IFS='|' read -r token name job_file <<< "${binding}"
        recovered="$("${PYTHON}" -m \
          tools.bata.reconcile_duca_cellcf_slurm_submission \
          recover-held-job --token "${token}" --job-name "${name}" \
          --cluster "${TARGET_CLUSTER}" --job-file "${job_file}" \
          --user "${CURRENT_USER}" 2>/dev/null || true)"
        if [[ "${recovered}" =~ ^[1-9][0-9]*$ ]]; then
          submitted_ids+=("${recovered}")
        fi
      done
    fi
    if [[ "${#submitted_ids[@]}" -gt 0 ]]; then
      unique_ids="$(printf '%s\n' "${submitted_ids[@]}" | sort -un | paste -sd, -)"
      "${PYTHON}" -m tools.bata.reconcile_duca_cellcf_slurm_submission \
        cancel-and-verify --job-ids "${unique_ids}" \
        --cluster "${TARGET_CLUSTER}" >/dev/null \
        || echo "[DUCA_CELLCF_COST_RECOVERY][FAIL] rollback verification failed for ${unique_ids}" >&2
    fi
  fi
  exit "${status}"
}
trap rollback EXIT

PAIR_JSON="$("${PYTHON}" -m \
  tools.bata.reconcile_duca_cellcf_slurm_submission submit-held-pair \
  --cost-token "${COST_TOKEN}" --cost-job-name "${COST_JOB_NAME}" \
  --cost-job-file "${COST_JOB}" \
  --completion-token "${COMPLETION_TOKEN}" \
  --completion-job-name "${COMPLETION_JOB_NAME}" \
  --completion-job-file "${COMPLETION_JOB}" \
  --cluster "${TARGET_CLUSTER}" --user "${CURRENT_USER}")"
readarray -t pair_binding < <(
  "${PYTHON}" - "${PAIR_JSON}" "${TARGET_CLUSTER}" <<'PY'
import json
import re
import sys

payload = json.loads(sys.argv[1])
cluster = sys.argv[2]
if set(payload) != {"cost", "completion"}:
    raise SystemExit("held-pair response has an invalid job set")
cost = payload["cost"]
completion = payload["completion"]
for key, record in payload.items():
    job_id = record.get("job_id")
    raw = record.get("raw_sbatch_response")
    if (
        not isinstance(job_id, int)
        or job_id <= 0
        or record.get("cluster") != cluster
        or raw != f"{job_id};{cluster}"
        or record.get("job_ref") != raw
    ):
        raise SystemExit(f"held-pair response is invalid for {key}")
if cost.get("dependency") != "none":
    raise SystemExit("recovery cost unexpectedly has a dependency")
expected_dependency = f"afterok:{cost['job_id']}"
if completion.get("dependency") != expected_dependency:
    raise SystemExit("recovery completion dependency mismatch")
if completion["job_id"] == cost["job_id"]:
    raise SystemExit("duplicate recovery job ids")
print(cost["job_id"])
print(cost["raw_sbatch_response"])
print(completion["job_id"])
print(completion["raw_sbatch_response"])
print(expected_dependency)
PY
)
[[ "${#pair_binding[@]}" == 5 ]] \
  || fail "held-pair transaction returned an incomplete binding"
COST_ID="${pair_binding[0]}"
COST_RAW="${pair_binding[1]}"
COMPLETION_ID="${pair_binding[2]}"
COMPLETION_RAW="${pair_binding[3]}"
COMPLETION_DEPENDENCY="${pair_binding[4]}"
submitted_ids=("${COST_ID}" "${COMPLETION_ID}")

for binding in \
  "cost|${COST_ID}|${COST_JOB_NAME}|${COST_JOB}|${COST_JOB_SHA256}|none|${COST_TOKEN}|${COST_RAW}" \
  "completion|${COMPLETION_ID}|${COMPLETION_JOB_NAME}|${COMPLETION_JOB}|${COMPLETION_JOB_SHA256}|${COMPLETION_DEPENDENCY}|${COMPLETION_TOKEN}|${COMPLETION_RAW}"; do
  IFS='|' read -r key id name job_file job_sha dependency token raw <<< "${binding}"
  receipt="${RECOVERY_ROOT}/receipts/${key}.scheduler.txt"
  scheduler_script="${RECOVERY_ROOT}/receipts/${key}.scheduler.sbatch"
  scontrol -M "${TARGET_CLUSTER}" show job -o "${id}" > "${receipt}"
  scontrol -M "${TARGET_CLUSTER}" write batch_script \
    "${id}" "${scheduler_script}"
  [[ "$(sha256_file "${scheduler_script}")" == "${job_sha}" ]] \
    || fail "scheduler-owned script differs for ${key}"
  "${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt \
    --job-id "${id}" --job-name "${name}" --comment "${token}" \
    --cluster "${TARGET_CLUSTER}" --job-file "${job_file}" \
    --job-file-sha256 "${job_sha}" --dependency \
    "$([[ "${dependency}" == "none" ]] || printf '%s' "${dependency}")" \
    --require-scheduler-script --require-submitted-with-hold \
    --require-current-user-hold >/dev/null
  "${PYTHON}" - "${receipt}" "${id}" "${name}" "${job_file}" \
    "${dependency}" "${token}" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
job_id, name = sys.argv[2], sys.argv[3]
job_file = str(Path(sys.argv[4]).resolve())
dependency, token = sys.argv[5], sys.argv[6]
required = [
    f"JobId={job_id}",
    f"JobName={name}",
    f"Command={job_file}",
    f"Comment={token}",
]
if dependency != "none":
    required.append(f"Dependency={dependency}")
if not all(marker in text for marker in required):
    raise SystemExit("held recovery scheduler binding mismatch")
if "JobState=PENDING" not in text or "Reason=JobHeldUser" not in text:
    raise SystemExit("recovery job was not under JobHeldUser before ledger commit")
PY
done

printf 'job_key\tjob_id\tjob_ref\tjob_name\tcluster\tdependency\tsbatch_file\tsbatch_sha256\traw_sbatch_response\tscheduler_receipt\tscheduler_receipt_sha256\tscheduler_script\tscheduler_script_sha256\ttrained_commit\tcost_producer_evidence_commit\tsubmission_intent_sha256\toriginal_formal_ledger_sha256\n' > "${LEDGER}"
printf 'cost\t%s\t%s;%s\t%s\t%s\tnone\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${COST_ID}" "${COST_ID}" "${TARGET_CLUSTER}" "${COST_JOB_NAME}" \
  "${TARGET_CLUSTER}" "${COST_JOB}" "${COST_JOB_SHA256}" "${COST_RAW}" \
  "${RECOVERY_ROOT}/receipts/cost.scheduler.txt" \
  "$(sha256_file "${RECOVERY_ROOT}/receipts/cost.scheduler.txt")" \
  "${RECOVERY_ROOT}/receipts/cost.scheduler.sbatch" \
  "$(sha256_file "${RECOVERY_ROOT}/receipts/cost.scheduler.sbatch")" \
  "${TRAINED_COMMIT}" "${EVIDENCE_COMMIT}" "${INTENT_SHA256}" \
  "${EXPECTED_ORIGINAL_LEDGER_SHA256}" >> "${LEDGER}"
printf 'completion\t%s\t%s;%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${COMPLETION_ID}" "${COMPLETION_ID}" "${TARGET_CLUSTER}" \
  "${COMPLETION_JOB_NAME}" "${TARGET_CLUSTER}" "${COMPLETION_DEPENDENCY}" \
  "${COMPLETION_JOB}" "${COMPLETION_JOB_SHA256}" "${COMPLETION_RAW}" \
  "${RECOVERY_ROOT}/receipts/completion.scheduler.txt" \
  "$(sha256_file "${RECOVERY_ROOT}/receipts/completion.scheduler.txt")" \
  "${RECOVERY_ROOT}/receipts/completion.scheduler.sbatch" \
  "$(sha256_file "${RECOVERY_ROOT}/receipts/completion.scheduler.sbatch")" \
  "${TRAINED_COMMIT}" "${EVIDENCE_COMMIT}" "${INTENT_SHA256}" \
  "${EXPECTED_ORIGINAL_LEDGER_SHA256}" >> "${LEDGER}"
LEDGER_SHA256="$(sha256_file "${LEDGER}")"

"${PYTHON}" - "${MANIFEST}" "${INTENT}" "${INTENT_SHA256}" \
  "${LEDGER}" "${LEDGER_SHA256}" "${ORIGINAL_FAILURE}" \
  "${ORIGINAL_FAILURE_SHA256}" "${COST_ID}" "${COST_JOB_NAME}" \
  "${COST_JOB}" "${COST_JOB_SHA256}" "${COST_RAW}" \
  "${COMPLETION_ID}" "${COMPLETION_JOB_NAME}" "${COMPLETION_JOB}" \
  "${COMPLETION_JOB_SHA256}" "${COMPLETION_RAW}" \
  "${COMPLETION_DEPENDENCY}" "${TARGET_CLUSTER}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output, intent, intent_sha, ledger, ledger_sha, failure, failure_sha,
    cost_id, cost_name, cost_job, cost_job_sha, cost_raw,
    completion_id, completion_name, completion_job, completion_job_sha,
    completion_raw, completion_dependency, cluster,
) = sys.argv[1:]
intent_payload = json.loads(Path(intent).read_text(encoding="utf-8"))
payload = {
    "schema": "duca_cellcf_cost_recovery_submission_v1",
    "ok": True,
    "status": "SUBMITTED_HELD_VERIFIED",
    "task": "offline_temporal_action_detection",
    "submission_intent_path": str(Path(intent).resolve()),
    "submission_intent_sha256": intent_sha,
    "jobs_ledger_path": str(Path(ledger).resolve()),
    "jobs_ledger_sha256": ledger_sha,
    "original_failure_receipt_path": str(Path(failure).resolve()),
    "original_failure_receipt_sha256": failure_sha,
    "trained_git_commit": intent_payload["trained_git_commit"],
    "cost_producer_evidence_commit": intent_payload[
        "cost_producer_evidence_commit"
    ],
    "aggregate_evidence_path": intent_payload["aggregate_evidence_path"],
    "aggregate_evidence_sha256": intent_payload["aggregate_evidence_sha256"],
    "cost_evidence_path": intent_payload["cost_evidence_path"],
    "final_suite_evidence_path": intent_payload["final_suite_evidence_path"],
    "target_cluster": cluster,
    "jobs": [
        {
            "job_key": "cost",
            "job_id": int(cost_id),
            "job_name": cost_name,
            "cluster": cluster,
            "dependency": "none",
            "sbatch_file": str(Path(cost_job).resolve()),
            "sbatch_sha256": cost_job_sha,
            "raw_sbatch_response": cost_raw,
            "scheduler_script": str(
                Path(cost_job).parents[1]
                / "receipts"
                / "cost.scheduler.sbatch"
            ),
            "scheduler_script_sha256": cost_job_sha,
        },
        {
            "job_key": "completion",
            "job_id": int(completion_id),
            "job_name": completion_name,
            "cluster": cluster,
            "dependency": completion_dependency,
            "sbatch_file": str(Path(completion_job).resolve()),
            "sbatch_sha256": completion_job_sha,
            "raw_sbatch_response": completion_raw,
            "scheduler_script": str(
                Path(completion_job).parents[1]
                / "receipts"
                / "completion.scheduler.sbatch"
            ),
            "scheduler_script_sha256": completion_job_sha,
        },
    ],
}
canonical = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
with Path(output).open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"${PYTHON}" -m tools.bata.reconcile_duca_cellcf_slurm_submission \
  fsync-artifacts \
  --file "${INTENT}" --file "${MANIFEST}" --file "${LEDGER}" \
  --file "${ORIGINAL_FAILURE}" \
  --file "${RECOVERY_ROOT}/receipts/original_terminal_jobs.sacct" \
  --file "${COST_JOB}" --file "${COMPLETION_JOB}" \
  --file "${RECOVERY_ROOT}/receipts/cost.scheduler.txt" \
  --file "${RECOVERY_ROOT}/receipts/completion.scheduler.txt" \
  --file "${RECOVERY_ROOT}/receipts/cost.scheduler.sbatch" \
  --file "${RECOVERY_ROOT}/receipts/completion.scheduler.sbatch" \
  --directory "${FORMAL_ROOT}" \
  --directory "${RECOVERY_ROOT}" \
  --directory "${RECOVERY_ROOT}/jobs" \
  --directory "${RECOVERY_ROOT}/logs" \
  --directory "${RECOVERY_ROOT}/receipts" >/dev/null

scontrol -M "${TARGET_CLUSTER}" release "${COMPLETION_ID}"
scontrol -M "${TARGET_CLUSTER}" release "${COST_ID}"
released=1
trap - EXIT
echo "[DUCA_CELLCF_COST_RECOVERY] cost=${COST_ID};${TARGET_CLUSTER} completion=${COMPLETION_ID};${TARGET_CLUSTER} root=${RECOVERY_ROOT}"
