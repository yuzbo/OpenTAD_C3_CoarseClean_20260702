#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_CROSSFIT][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS \
  DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256 \
  DUCA_RIME_PHASE2_CROSSFIT_ROOT \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_RISK_THRESHOLD; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-2 cross-fit production must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "cross-fit production requires a complete Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}" ]] \
  || fail "a fresh cross-fit output root is required"
[[ "$(sha256sum "${DUCA_RIME_SPLIT_MANIFEST}" | awk '{print $1}')" == "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" ]] \
  || fail "split manifest SHA-256 drift"
[[ "$(sha256sum "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" | awk '{print $1}')" == "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256}" ]] \
  || fail "counterfactual measurement SHA-256 drift"

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
[[ "${#candidate_budgets[@]}" -ge 3 ]] \
  || fail "at least three candidate budgets are required"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
    "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" \
    "${candidate_budgets[@]}" <<'PY'
import json
import sys

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.produce_duca_rime_crossfit_records import MEASUREMENT_SCHEMA

manifest, manifest_sha, source, *budgets = sys.argv[1:]
validate_rime_splits(manifest, expected_sha256=manifest_sha)
expected = [int(value) for value in budgets]
rows = [
    json.loads(line)
    for line in open(source, encoding="utf-8-sig")
    if line.strip()
]
if not rows or any(
    row.get("schema_version") != MEASUREMENT_SCHEMA
    or row.get("candidate_budgets") != expected
    for row in rows
):
    raise SystemExit("counterfactual source schema/budget panel drift")
PY
  echo "[DUCA_RIME_PHASE2_CROSSFIT] PRECHECK PASS"
  exit 0
fi

python tools/bata/produce_duca_rime_crossfit_records.py \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --measurements-jsonl "${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS}" \
  --output-root "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}" \
  --candidate-budgets "${candidate_budgets[@]}" \
  --ridge "${DUCA_RIME_CROSSFIT_RIDGE:-0.001}" \
  --risk-threshold "${DUCA_RIME_RISK_THRESHOLD}"

for kind in o3 o4 price; do
  python tools/bata/build_duca_rime_gate_records.py "${kind}" \
    --source-jsonl "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/${kind}_source.jsonl" \
    --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
    --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
    --output "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/${kind}_records.jsonl"
done

python tools/bata/build_duca_rime_training_targets.py \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --observations-jsonl "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/budget_observations.jsonl" \
  --hard-utility-jsonl "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/hard_frame_utility.jsonl" \
  --output-jsonl "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/training_targets.jsonl" \
  --candidate-budgets "${candidate_budgets[@]}" \
  > "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/training_targets.summary.json"

printf '%s\n' \
  "schema=duca_rime_phase2_crossfit_production_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "counterfactual_measurements_sha256=${DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256}" \
  "producer_summary_sha256=$(sha256sum "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/producer_summary.json" | awk '{print $1}')" \
  "training_targets_sha256=$(sha256sum "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/training_targets.jsonl" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE2_CROSSFIT_ROOT}/production.receipt"

echo "[DUCA_RIME_PHASE2_CROSSFIT] PASS ${DUCA_RIME_PHASE2_CROSSFIT_ROOT}"
