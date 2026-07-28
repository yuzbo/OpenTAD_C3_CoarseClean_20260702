#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1_SEAL][FAIL] $*" >&2
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
  DUCA_RIME_PHASE1_SEAL_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE1_SPLIT_ROLE \
  DUCA_RIME_CODE_GATE_RECEIPT \
  DUCA_RIME_CODE_GATE_RECEIPT_SHA256 \
  DUCA_RIME_PHASE0_REPLICATE_A_METRICS \
  DUCA_RIME_PHASE0_REPLICATE_A_METRICS_SHA256 \
  DUCA_RIME_PHASE0_REPLICATE_B_METRICS \
  DUCA_RIME_PHASE0_REPLICATE_B_METRICS_SHA256 \
  DUCA_RIME_RELEASED_DENSE_METRICS \
  DUCA_RIME_RELEASED_DENSE_METRICS_SHA256 \
  DUCA_RIME_LOCAL_DENSE_METRICS \
  DUCA_RIME_LOCAL_DENSE_METRICS_SHA256 \
  DUCA_RIME_UNIFORM_K384_METRICS \
  DUCA_RIME_UNIFORM_K384_METRICS_SHA256 \
  DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY \
  DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY_SHA256 \
  DUCA_RIME_UNIFORM_K192_METRICS \
  DUCA_RIME_UNIFORM_K192_METRICS_SHA256 \
  DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY \
  DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY_SHA256 \
  DUCA_RIME_WRAPPER_GATE \
  DUCA_RIME_WRAPPER_GATE_SHA256 \
  DUCA_RIME_NO_PROBE_PROFILE \
  DUCA_RIME_NO_PROBE_PROFILE_SHA256 \
  DUCA_RIME_PROBE_PROFILE \
  DUCA_RIME_PROBE_PROFILE_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-1 sealing must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Phase-1 sealing requires a complete Git worktree"
[[ ! -e "${DUCA_RIME_PHASE1_SEAL_ROOT}" ]] \
  || fail "a fresh Phase-1 seal root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"
check_sha256 \
  "${DUCA_RIME_CODE_GATE_RECEIPT}" \
  "${DUCA_RIME_CODE_GATE_RECEIPT_SHA256}" \
  "code-gate receipt"
for prefix in \
  PHASE0_REPLICATE_A_METRICS \
  PHASE0_REPLICATE_B_METRICS \
  RELEASED_DENSE_METRICS \
  LOCAL_DENSE_METRICS \
  UNIFORM_K384_METRICS \
  UNIFORM_K384_LEDGER_SUMMARY \
  UNIFORM_K192_METRICS \
  UNIFORM_K192_LEDGER_SUMMARY \
  WRAPPER_GATE \
  NO_PROBE_PROFILE \
  PROBE_PROFILE; do
  path_var="DUCA_RIME_${prefix}"
  sha_var="${path_var}_SHA256"
  check_sha256 "${!path_var}" "${!sha_var}" "${prefix}"
done

readarray -t split_values < <(
  python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["assignment_sha256"])
PY
)
[[ "${#split_values[@]}" == 1 ]] \
  || fail "failed to read the split assignment hash"
assignment_sha256="${split_values[0]}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python tools/bata/create_duca_rime_splits.py \
    --validate-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
    --expected-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
    > /dev/null
  echo "[DUCA_RIME_PHASE1_SEAL] PRECHECK PASS"
  exit 0
fi

mkdir -p \
  "${DUCA_RIME_PHASE1_SEAL_ROOT}" \
  "${DUCA_RIME_PHASE1_SEAL_ROOT}/controls"

python tools/bata/audit_duca_rime_phase1_geometry.py \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --split-assignment-sha256 "${assignment_sha256}" \
  --output "${DUCA_RIME_PHASE1_SEAL_ROOT}/geometry_audit.json" \
  > "${DUCA_RIME_PHASE1_SEAL_ROOT}/geometry_audit.stdout.json"

python tools/bata/build_duca_rime_source_manifest.py phase0 \
  --replicate \
  deterministic_reexecution_a \
  deterministic_reexecution \
  "${DUCA_RIME_PHASE0_REPLICATE_A_METRICS}" \
  "${DUCA_RIME_PHASE0_REPLICATE_A_METRICS_SHA256}" \
  --replicate \
  deterministic_reexecution_b \
  deterministic_reexecution \
  "${DUCA_RIME_PHASE0_REPLICATE_B_METRICS}" \
  "${DUCA_RIME_PHASE0_REPLICATE_B_METRICS_SHA256}" \
  --output "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_source_manifest.json" \
  > /dev/null
python tools/bata/build_duca_rime_gate_records.py phase0 \
  --source-manifest \
  "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_source_manifest.json" \
  --primary-metric avg_map \
  --output "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_records.jsonl" \
  > "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_record_build.json"
python tools/bata/duca_rime_phase2.py phase0 \
  --records-jsonl "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_records.jsonl" \
  --primary-metric avg_map \
  --alpha 0.05 \
  --power 0.80 \
  --output "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_summary.json" \
  > /dev/null

python tools/bata/build_duca_rime_phase1_controls.py \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --split-role "${DUCA_RIME_PHASE1_SPLIT_ROLE}" \
  --released-dense-metrics "${DUCA_RIME_RELEASED_DENSE_METRICS}" \
  --local-dense-metrics "${DUCA_RIME_LOCAL_DENSE_METRICS}" \
  --uniform-k384-metrics "${DUCA_RIME_UNIFORM_K384_METRICS}" \
  --uniform-k384-ledger-summary \
  "${DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY}" \
  --uniform-k192-metrics "${DUCA_RIME_UNIFORM_K192_METRICS}" \
  --uniform-k192-ledger-summary \
  "${DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY}" \
  --wrapper-gate "${DUCA_RIME_WRAPPER_GATE}" \
  --geometry-audit "${DUCA_RIME_PHASE1_SEAL_ROOT}/geometry_audit.json" \
  --no-probe-profile "${DUCA_RIME_NO_PROBE_PROFILE}" \
  --probe-profile "${DUCA_RIME_PROBE_PROFILE}" \
  --output-dir "${DUCA_RIME_PHASE1_SEAL_ROOT}/controls" \
  > "${DUCA_RIME_PHASE1_SEAL_ROOT}/control_build.stdout.json"

control_args=()
for control in \
  released_dense \
  local_dense \
  uniform_k384 \
  uniform_k192 \
  wrapper_parity \
  q_to_t_before_nms \
  no_probe_uniform_cost \
  probe_uniform_cost; do
  control_args+=(--control "${DUCA_RIME_PHASE1_SEAL_ROOT}/controls/${control}.json")
done
python tools/bata/duca_rime_stage_contract.py phase1 \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase0-summary "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_summary.json" \
  --code-gate-receipt "${DUCA_RIME_CODE_GATE_RECEIPT}" \
  "${control_args[@]}" \
  --output "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase1_receipt.json" \
  > "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase1_seal.stdout.json"

printf '%s\n' \
  "schema=duca_rime_phase1_seal_run_receipt_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "phase0_summary_sha256=$(sha256sum "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase0_summary.json" | awk '{print $1}')" \
  "control_manifest_sha256=$(sha256sum "${DUCA_RIME_PHASE1_SEAL_ROOT}/controls/control_build_manifest.json" | awk '{print $1}')" \
  "phase1_receipt_sha256=$(sha256sum "${DUCA_RIME_PHASE1_SEAL_ROOT}/phase1_receipt.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE1_SEAL_ROOT}/seal.receipt"
echo "[DUCA_RIME_PHASE1_SEAL] PASS ${DUCA_RIME_PHASE1_SEAL_ROOT}"
