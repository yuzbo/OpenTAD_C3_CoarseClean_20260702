#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE2_ROOT \
  DUCA_RIME_PHASE1_RECEIPT \
  DUCA_RIME_PHASE1_RECEIPT_SHA256 \
  DUCA_RIME_O1_RECORDS \
  DUCA_RIME_O1_RECORDS_SHA256 \
  DUCA_RIME_O2_RECORDS \
  DUCA_RIME_O2_RECORDS_SHA256 \
  DUCA_RIME_O3_RECORDS \
  DUCA_RIME_O3_RECORDS_SHA256 \
  DUCA_RIME_O4_RECORDS \
  DUCA_RIME_O4_RECORDS_SHA256 \
  DUCA_RIME_PRICE_CALIBRATION \
  DUCA_RIME_PRICE_CALIBRATION_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_CANDIDATE_COSTS \
  DUCA_RIME_TARGET_MEAN_COST \
  DUCA_RIME_DECODER_FAMILY \
  DUCA_RIME_RISK_WEIGHT \
  DUCA_RIME_RISK_THRESHOLD \
  DUCA_RIME_O4_MAX_BRIER \
  DUCA_RIME_O4_MAX_ECE \
  DUCA_RIME_O4_MIN_COVERAGE \
  DUCA_RIME_O4_MAX_LOW_RISK_FAILURE; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-2 gates must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ ! -e "${DUCA_RIME_PHASE2_ROOT}" ]] || fail "a fresh Phase-2 root is required"
[[ -d "${DUCA_RIME_REPO_ROOT}" ]] || fail "repository snapshot is missing"
cd "${DUCA_RIME_REPO_ROOT}"

if [[ -d .git ]]; then
  [[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
    || fail "Git commit drift"
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || fail "Git tree is dirty"
else
  required DUCA_RIME_OVERLAY_SHA256_MANIFEST
  grep -qx "commit=${DUCA_RIME_EXPECTED_COMMIT}" .codex_source_manifest \
    || fail "overlay commit drift"
  sha256sum -c "${DUCA_RIME_OVERLAY_SHA256_MANIFEST}"
fi

check_sha256 \
  "${DUCA_RIME_PHASE1_RECEIPT}" \
  "${DUCA_RIME_PHASE1_RECEIPT_SHA256}" \
  "Phase-1 receipt"
check_sha256 "${DUCA_RIME_O1_RECORDS}" "${DUCA_RIME_O1_RECORDS_SHA256}" "O1 records"
check_sha256 "${DUCA_RIME_O2_RECORDS}" "${DUCA_RIME_O2_RECORDS_SHA256}" "O2 records"
check_sha256 "${DUCA_RIME_O3_RECORDS}" "${DUCA_RIME_O3_RECORDS_SHA256}" "O3 records"
check_sha256 "${DUCA_RIME_O4_RECORDS}" "${DUCA_RIME_O4_RECORDS_SHA256}" "O4 records"
check_sha256 \
  "${DUCA_RIME_PRICE_CALIBRATION}" \
  "${DUCA_RIME_PRICE_CALIBRATION_SHA256}" \
  "price calibration"

readarray -t phase0_thresholds < <(
  python - "${DUCA_RIME_PHASE1_RECEIPT}" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    row.get("schema_version") != "duca_rime_stage_receipt_v1"
    or row.get("phase") != "phase1"
    or row.get("gate_pass") is not True
    or row.get("phase2_authorized") is not True
    or row.get("official_final_subset_consumed") is not False
):
    raise SystemExit("Phase-1 receipt does not authorize Phase-2")
thresholds = row["phase0_thresholds"]
print(float(thresholds["min_o1_headroom"]))
print(float(thresholds["max_o2_decoder_regret"]))
print(float(thresholds["min_o3_spearman"]))
PY
)
[[ "${#phase0_thresholds[@]}" == 3 ]] || fail "Phase-0 thresholds are incomplete"

mkdir -p "${DUCA_RIME_PHASE2_ROOT}"
phase2_tool=(python tools/bata/duca_rime_phase2.py)
"${phase2_tool[@]}" o1 \
  --records-jsonl "${DUCA_RIME_O1_RECORDS}" \
  --output "${DUCA_RIME_PHASE2_ROOT}/o1.json" \
  --target-mean-cost "${DUCA_RIME_TARGET_MEAN_COST}" \
  --min-headroom "${phase0_thresholds[0]}" \
  --bootstrap-samples "${DUCA_RIME_BOOTSTRAP_SAMPLES:-5000}" \
  --shuffles "${DUCA_RIME_SHUFFLES:-5000}" \
  --seed "${DUCA_RIME_PHASE2_SEED:-3407}"
"${phase2_tool[@]}" o2 \
  --records-jsonl "${DUCA_RIME_O2_RECORDS}" \
  --output "${DUCA_RIME_PHASE2_ROOT}/o2.json" \
  --selected-family "${DUCA_RIME_DECODER_FAMILY}" \
  --max-regret "${phase0_thresholds[1]}" \
  --bootstrap-samples "${DUCA_RIME_BOOTSTRAP_SAMPLES:-5000}" \
  --seed "${DUCA_RIME_PHASE2_SEED:-3407}"
"${phase2_tool[@]}" o3 \
  --records-jsonl "${DUCA_RIME_O3_RECORDS}" \
  --output "${DUCA_RIME_PHASE2_ROOT}/o3.json" \
  --min-spearman "${phase0_thresholds[2]}" \
  --null-margin "${DUCA_RIME_O3_NULL_MARGIN:-0.0}" \
  --bootstrap-samples "${DUCA_RIME_BOOTSTRAP_SAMPLES:-5000}" \
  --seed "${DUCA_RIME_PHASE2_SEED:-3407}"
"${phase2_tool[@]}" o4 \
  --records-jsonl "${DUCA_RIME_O4_RECORDS}" \
  --output "${DUCA_RIME_PHASE2_ROOT}/o4.json" \
  --risk-threshold "${DUCA_RIME_RISK_THRESHOLD}" \
  --max-brier "${DUCA_RIME_O4_MAX_BRIER}" \
  --max-ece "${DUCA_RIME_O4_MAX_ECE}" \
  --min-coverage "${DUCA_RIME_O4_MIN_COVERAGE}" \
  --max-low-risk-failure "${DUCA_RIME_O4_MAX_LOW_RISK_FAILURE}" \
  --calibration-bins "${DUCA_RIME_O4_CALIBRATION_BINS:-10}"

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
IFS=',' read -r -a candidate_costs <<<"${DUCA_RIME_CANDIDATE_COSTS}"
"${phase2_tool[@]}" freeze \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o1.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o2.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o3.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o4.json" \
  --calibration-jsonl "${DUCA_RIME_PRICE_CALIBRATION}" \
  --output "${DUCA_RIME_PHASE2_ROOT}/budget_protocol.json" \
  --candidate-budgets "${candidate_budgets[@]}" \
  --candidate-costs "${candidate_costs[@]}" \
  --target-mean-cost "${DUCA_RIME_TARGET_MEAN_COST}" \
  --risk-weight "${DUCA_RIME_RISK_WEIGHT}" \
  --risk-threshold "${DUCA_RIME_RISK_THRESHOLD}" \
  --decoder-family "${DUCA_RIME_DECODER_FAMILY}" \
  --weak-overlap-fraction "${DUCA_RIME_WEAK_OVERLAP_FRACTION:-0.5}"

python tools/bata/duca_rime_stage_contract.py phase2 \
  --phase1-receipt "${DUCA_RIME_PHASE1_RECEIPT}" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o1.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o2.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o3.json" \
  --summary "${DUCA_RIME_PHASE2_ROOT}/o4.json" \
  --budget-protocol "${DUCA_RIME_PHASE2_ROOT}/budget_protocol.json" \
  --output "${DUCA_RIME_PHASE2_ROOT}/phase2_receipt.json"
sha256sum "${DUCA_RIME_PHASE2_ROOT}/phase2_receipt.json" \
  > "${DUCA_RIME_PHASE2_ROOT}/phase2_receipt.sha256"

printf '%s\n' \
  "schema=duca_rime_phase2_slurm_gate_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "phase2_receipt_sha256=$(sha256sum "${DUCA_RIME_PHASE2_ROOT}/phase2_receipt.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE2_ROOT}/gate.receipt"
echo "[DUCA_RIME_PHASE2] PASS ${DUCA_RIME_PHASE2_ROOT}/phase2_receipt.json"
