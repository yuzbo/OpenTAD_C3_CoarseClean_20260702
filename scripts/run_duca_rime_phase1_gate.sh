#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1][FAIL] $*" >&2
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
  DUCA_RIME_PHASE1_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE0_SUMMARY \
  DUCA_RIME_PHASE0_SUMMARY_SHA256 \
  DUCA_RIME_CODE_GATE_RECEIPT \
  DUCA_RIME_CODE_GATE_RECEIPT_SHA256 \
  DUCA_RIME_PHASE1_CONTROL_DIR; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-1 sealing must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ ! -e "${DUCA_RIME_PHASE1_ROOT}" ]] || fail "a fresh Phase-1 root is required"
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
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
check_sha256 \
  "${DUCA_RIME_PHASE0_SUMMARY}" \
  "${DUCA_RIME_PHASE0_SUMMARY_SHA256}" \
  "Phase-0 variance/power summary"
check_sha256 \
  "${DUCA_RIME_CODE_GATE_RECEIPT}" \
  "${DUCA_RIME_CODE_GATE_RECEIPT_SHA256}" \
  "code-gate receipt"

control_names=(
  released_dense
  local_dense
  uniform_k384
  uniform_k192
  acquisition_admission
  q_to_t_before_nms
  no_probe_uniform_cost
  probe_uniform_cost
)
control_args=()
for name in "${control_names[@]}"; do
  path="${DUCA_RIME_PHASE1_CONTROL_DIR}/${name}.json"
  [[ -f "${path}" ]] || fail "missing Phase-1 control ${name}: ${path}"
  control_args+=(--control "${path}")
done

mkdir -p "${DUCA_RIME_PHASE1_ROOT}"
python tools/bata/duca_rime_stage_contract.py phase1 \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase0-summary "${DUCA_RIME_PHASE0_SUMMARY}" \
  --code-gate-receipt "${DUCA_RIME_CODE_GATE_RECEIPT}" \
  "${control_args[@]}" \
  --output "${DUCA_RIME_PHASE1_ROOT}/phase1_receipt.json"
sha256sum "${DUCA_RIME_PHASE1_ROOT}/phase1_receipt.json" \
  > "${DUCA_RIME_PHASE1_ROOT}/phase1_receipt.sha256"
printf '%s\n' \
  "schema=duca_rime_phase1_slurm_gate_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "phase1_receipt_sha256=$(sha256sum "${DUCA_RIME_PHASE1_ROOT}/phase1_receipt.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE1_ROOT}/gate.receipt"
echo "[DUCA_RIME_PHASE1] PASS ${DUCA_RIME_PHASE1_ROOT}/phase1_receipt.json"
