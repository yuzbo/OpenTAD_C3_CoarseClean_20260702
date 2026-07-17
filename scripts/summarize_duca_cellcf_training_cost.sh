#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_TRAINING_COST][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
BOOTSTRAP_PYTHON="${BASE}/conda_envs/opentad/bin/python"
RUN_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-${RUN_ROOT:-}}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
LEDGER="${RUN_ROOT}/jobs.submitted.tsv"
OUTPUT_ROOT="${RUN_ROOT}/training_cost"
AGGREGATE_EVIDENCE="${RUN_ROOT}/aggregate_suite_evidence.json"
AGGREGATE_SHA256="${DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256:-}"

[[ -x "${BOOTSTRAP_PYTHON}" ]] || fail "Python is missing: ${BOOTSTRAP_PYTHON}"
[[ -d "${RUN_ROOT}" ]] || fail "formal run root is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "RUN_ROOT violates the formal path contract"
LEDGER="${RUN_ROOT}/jobs.submitted.tsv"
OUTPUT_ROOT="${RUN_ROOT}/training_cost"
AGGREGATE_EVIDENCE="${RUN_ROOT}/aggregate_suite_evidence.json"
[[ -f "${LEDGER}" ]] || fail "submitted-job ledger is missing"
[[ -n "${EXPECTED_COMMIT}" ]] || fail "DUCA_EXPECTED_COMMIT is required"
[[ -f "${AGGREGATE_EVIDENCE}" ]] || fail "aggregate suite evidence is missing"
[[ "${AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256 is required"
[[ "$(sha256sum "${AGGREGATE_EVIDENCE}" | awk '{print $1}')" == "${AGGREGATE_SHA256}" ]] \
  || fail "aggregate suite evidence hash mismatch"
AGGREGATE_PROFILE="$(
  env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "${BOOTSTRAP_PYTHON}" - "${AGGREGATE_EVIDENCE}" \
      "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

from tools.bata.duca_cellcf_protocol import LEGACY_EXPOSURE132_COMMITS

path = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("schema") != "duca_cellcf_suite_manifest_v1"
    or payload.get("ok") is not True
    or payload.get("git_commit") != commit
):
    raise SystemExit("aggregate suite evidence identity mismatch")
profile = payload.get("training_profile")
if profile is None and commit in LEGACY_EXPOSURE132_COMMITS:
    profile = "exposure132"
if profile not in {"exposure132", "official60"}:
    raise SystemExit("aggregate suite evidence has no supported training profile")
print(profile)
PY
)" || fail "cannot resolve the hash-bound aggregate training profile"
if [[ -n "${DUCA_CELLCF_TRAINING_PROFILE:-}" \
  && "${DUCA_CELLCF_TRAINING_PROFILE}" != "${AGGREGATE_PROFILE}" ]]; then
  fail "explicit training profile differs from aggregate evidence"
fi
export DUCA_CELLCF_TRAINING_PROFILE="${AGGREGATE_PROFILE}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite training-cost evidence"
mkdir -p "${OUTPUT_ROOT}"

readarray -t bindings < <("${PYTHON}" - "${LEDGER}" "${EXPECTED_COMMIT}" \
  "${DUCA_CELLCF_TRAINING_PROFILE}" <<'PY'
import csv
import sys

from tools.bata.duca_cellcf_protocol import LEGACY_EXPOSURE132_COMMITS

ledger, commit, profile = sys.argv[1:]
with open(ledger, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for key in ("uniform", "transition_beta0", "cellcf"):
    matches = [row for row in rows if row.get("job_key") == key]
    if len(matches) != 1:
        raise SystemExit(
            f"submitted ledger must contain exactly one {key} row, got {len(matches)}"
        )
    row = matches[0]
    if row["commit"] != commit:
        raise SystemExit(f"{key} commit mismatch")
    row_profile = row.get("training_profile")
    if not row_profile and commit in LEGACY_EXPOSURE132_COMMITS:
        row_profile = "exposure132"
    if row_profile != profile:
        raise SystemExit(f"{key} training profile mismatch")
    if not row["job_id"].isdigit() or not row["job_name"] or not row["cluster"]:
        raise SystemExit(f"{key} Slurm identity is incomplete")
    print("\t".join((key, row["job_id"], row["job_name"], row["cluster"])))
PY
)
[[ "${#bindings[@]}" == "3" ]] || fail "expected exactly three arm bindings"

for binding in "${bindings[@]}"; do
  IFS=$'\t' read -r variant job_id job_name cluster <<< "${binding}"
  "${PYTHON}" -m tools.bata.capture_duca_cellcf_slurm_cost \
    --job-id "${job_id}" \
    --expected-job-name "${job_name}" \
    --expected-cluster "${cluster}" \
    --raw-output "${OUTPUT_ROOT}/${variant}.sacct.psv" \
    --output-json "${OUTPUT_ROOT}/${variant}.slurm_cost.json"
done

"${PYTHON}" -m tools.bata.summarize_duca_cellcf_training_cost \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-aggregate "${AGGREGATE_EVIDENCE}" \
  --suite-aggregate-sha256 "${AGGREGATE_SHA256}" \
  --post-run "uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json" \
  --post-run "transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json" \
  --post-run "cellcf=${RUN_ROOT}/logs/cellcf/post_run_evidence.json" \
  --slurm-cost "uniform=${OUTPUT_ROOT}/uniform.slurm_cost.json" \
  --slurm-cost "transition_beta0=${OUTPUT_ROOT}/transition_beta0.slurm_cost.json" \
  --slurm-cost "cellcf=${OUTPUT_ROOT}/cellcf.slurm_cost.json" \
  --output-json "${OUTPUT_ROOT}/training_cost_summary.json" \
  --output-tsv "${OUTPUT_ROOT}/training_cost_summary.tsv"
