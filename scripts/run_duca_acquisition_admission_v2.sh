#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ACQUISITION_V2][FAIL] $*" >&2
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
  DUCA_RIME_EXPECTED_BRANCH \
  DUCA_ACQUISITION_V2_MODE \
  DUCA_ACQUISITION_V2_ROOT \
  DUCA_RIME_CODE_GATE_RECEIPT \
  DUCA_RIME_CODE_GATE_RECEIPT_SHA256 \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_SHA256_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_SHA256_TRIDET \
  DUCA_RIME_TRAIN_BLOCK_LIST \
  DUCA_RIME_DEVELOPMENT_BLOCK_LIST \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_DATA_MANIFEST \
  DUCA_RIME_DATA_MANIFEST_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256; do
  required "${name}"
done

[[ "${DUCA_ACQUISITION_V2_MODE}" == calibrate || \
   "${DUCA_ACQUISITION_V2_MODE}" == admit ]] \
  || fail "DUCA_ACQUISITION_V2_MODE must be calibrate or admit"
[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the runtime gate must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ "${DUCA_RIME_ENABLE_PHASE4:-0}" == 0 ]] \
  || fail "Phase-4 must remain sealed during acquisition-v2 admission"
[[ "${DUCA_RIME_OFFICIAL_FINAL_CONSUMED:-0}" == 0 ]] \
  || fail "official-final consumption is forbidden during acquisition-v2 admission"
[[ ! -e "${DUCA_ACQUISITION_V2_ROOT}" ]] \
  || fail "a fresh acquisition-v2 output root is required"

cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${DUCA_RIME_EXPECTED_BRANCH}" ]] \
  || fail "Git branch drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

for binding in \
  "${DUCA_RIME_CODE_GATE_RECEIPT}|${DUCA_RIME_CODE_GATE_RECEIPT_SHA256}|code gate" \
  "${DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER}|${DUCA_RIME_DENSE_CHECKPOINT_SHA256_ACTIONFORMER}|ActionFormer checkpoint" \
  "${DUCA_RIME_DENSE_CHECKPOINT_TRIDET}|${DUCA_RIME_DENSE_CHECKPOINT_SHA256_TRIDET}|TriDet checkpoint" \
  "${DUCA_RIME_TARGETS_JSONL}|${DUCA_RIME_TARGETS_SHA256}|training targets" \
  "${DUCA_RIME_BUDGET_PROTOCOL_JSON}|${DUCA_RIME_BUDGET_PROTOCOL_SHA256}|budget protocol" \
  "${DUCA_RIME_DATA_MANIFEST}|${DUCA_RIME_DATA_MANIFEST_SHA256}|data manifest" \
  "${DUCA_RIME_SPLIT_MANIFEST}|${DUCA_RIME_SPLIT_MANIFEST_SHA256}|split manifest"; do
  IFS='|' read -r path expected label <<<"${binding}"
  check_sha256 "${path}" "${expected}" "${label}"
done
for path in "${DUCA_RIME_TRAIN_BLOCK_LIST}" "${DUCA_RIME_DEVELOPMENT_BLOCK_LIST}"; do
  [[ -f "${path}" ]] || fail "registered block list is missing: ${path}"
done

selected_actionformer="${DUCA_ACQUISITION_SELECTED_ACTIONFORMER_CONFIG:-configs/adatad/thumos/duca_rime_full_selected_axis_total60.py}"
standard_actionformer="${DUCA_ACQUISITION_STANDARD_ACTIONFORMER_CONFIG:-configs/adatad/thumos/duca_rime_dense_actionformer_total60.py}"
selected_tridet="${DUCA_ACQUISITION_SELECTED_TRIDET_CONFIG:-configs/adatad/thumos/duca_rime_full_tridet_selected_axis_total60.py}"
standard_tridet="${DUCA_ACQUISITION_STANDARD_TRIDET_CONFIG:-configs/adatad/thumos/duca_rime_dense_tridet_total60.py}"
for path in \
  "${selected_actionformer}" \
  "${standard_actionformer}" \
  "${selected_tridet}" \
  "${standard_tridet}"; do
  [[ -f "${path}" ]] || fail "admission config is missing: ${path}"
done

mkdir -p "${DUCA_ACQUISITION_V2_ROOT}"
common_args=(
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}"
  --expected-branch "${DUCA_RIME_EXPECTED_BRANCH}"
  --selected-actionformer-config "${selected_actionformer}"
  --standard-actionformer-config "${standard_actionformer}"
  --actionformer-checkpoint "${DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER}"
  --selected-tridet-config "${selected_tridet}"
  --standard-tridet-config "${standard_tridet}"
  --tridet-checkpoint "${DUCA_RIME_DENSE_CHECKPOINT_TRIDET}"
  --train-block-list "${DUCA_RIME_TRAIN_BLOCK_LIST}"
  --development-block-list "${DUCA_RIME_DEVELOPMENT_BLOCK_LIST}"
  --targets-jsonl "${DUCA_RIME_TARGETS_JSONL}"
  --budget-protocol "${DUCA_RIME_BUDGET_PROTOCOL_JSON}"
  --data-manifest "${DUCA_RIME_DATA_MANIFEST}"
  --split-assignment "${DUCA_RIME_SPLIT_MANIFEST}"
  --code-gate-receipt "${DUCA_RIME_CODE_GATE_RECEIPT}"
)

if [[ "${DUCA_ACQUISITION_V2_MODE}" == calibrate ]]; then
  python -m tools.bata.run_duca_acquisition_runtime_gate_v2 \
    "${common_args[@]}" \
    --calibration-output \
    "${DUCA_ACQUISITION_V2_ROOT}/numeric_calibration.json"
  printf '%s\n' \
    "schema=duca_acquisition_numeric_calibration_receipt_v1" \
    "status=frozen" \
    "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
    "slurm_job_id=${SLURM_JOB_ID}" \
    "uses_official_final=false" \
    "phase4_submission_enabled=false" \
    "calibration_sha256=$(sha256sum "${DUCA_ACQUISITION_V2_ROOT}/numeric_calibration.json" | awk '{print $1}')" \
    > "${DUCA_ACQUISITION_V2_ROOT}/calibration.receipt"
  echo "[DUCA_ACQUISITION_V2] CALIBRATION PASS ${DUCA_ACQUISITION_V2_ROOT}"
  exit 0
fi

for name in \
  DUCA_ACQUISITION_NUMERIC_CALIBRATION \
  DUCA_ACQUISITION_NUMERIC_CALIBRATION_SHA256 \
  DUCA_ACQUISITION_SCIENTIFIC_PROTOCOL \
  DUCA_ACQUISITION_SCIENTIFIC_PROTOCOL_SHA256; do
  required "${name}"
done
check_sha256 \
  "${DUCA_ACQUISITION_NUMERIC_CALIBRATION}" \
  "${DUCA_ACQUISITION_NUMERIC_CALIBRATION_SHA256}" \
  "numeric calibration"
check_sha256 \
  "${DUCA_ACQUISITION_SCIENTIFIC_PROTOCOL}" \
  "${DUCA_ACQUISITION_SCIENTIFIC_PROTOCOL_SHA256}" \
  "scientific protocol"

receipt="${DUCA_ACQUISITION_V2_ROOT}/admission_v2.receipt.json"
python -m tools.bata.run_duca_acquisition_runtime_gate_v2 \
  "${common_args[@]}" \
  --numeric-calibration "${DUCA_ACQUISITION_NUMERIC_CALIBRATION}" \
  --scientific-protocol "${DUCA_ACQUISITION_SCIENTIFIC_PROTOCOL}" \
  --evidence-output "${receipt}"

printf '%s\n' \
  "schema=duca_acquisition_admission_runtime_receipt_v2" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "uses_official_final=false" \
  "phase4_submission_enabled=false" \
  "producer=tools.bata.run_duca_acquisition_runtime_gate_v2" \
  "admission_sha256=$(sha256sum "${receipt}" | awk '{print $1}')" \
  > "${DUCA_ACQUISITION_V2_ROOT}/runtime_gate.receipt"
echo "[DUCA_ACQUISITION_V2] ADMISSION PASS ${receipt}"
