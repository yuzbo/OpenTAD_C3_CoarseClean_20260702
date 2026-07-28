#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1_COST][FAIL] $*" >&2
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
  DUCA_RIME_PHASE1_COST_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE1_SPLIT_ROLE \
  DUCA_RIME_PHASE1_NO_PROBE_CONFIG \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT \
  DUCA_RIME_PHASE1_PROBE_CONFIG \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT \
  DUCA_RIME_ADATAD_PRETRAIN \
  DUCA_RIME_ADATAD_PRETRAIN_SHA256 \
  DUCA_RIME_PHASE1_PROFILE_SESSION_ID \
  DUCA_RIME_PHASE1_PROFILE_PAIR_ID; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-1 cost controls must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ "${DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "no-probe checkpoint trained commit must be exact"
[[ "${DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "probe checkpoint trained commit must be exact"
[[ "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256}" == \
  "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256}" ]] \
  || fail "paired cost controls must use the same checkpoint bytes"
[[ "${DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT}" == \
  "${DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT}" ]] \
  || fail "paired cost controls must use the same trained commit"
[[ "${DUCA_RIME_PHASE1_PROFILE_ORDER:-no_probe_first}" =~ ^(no_probe_first|probe_first)$ ]] \
  || fail "profile order must be no_probe_first or probe_first"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Phase-1 cost controls require a complete Git worktree"
[[ ! -e "${DUCA_RIME_PHASE1_COST_ROOT}" ]] \
  || fail "a fresh Phase-1 cost root is required"
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
  "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT}" \
  "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256}" \
  "no-probe checkpoint"
check_sha256 \
  "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT}" \
  "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256}" \
  "probe checkpoint"
check_sha256 \
  "${DUCA_RIME_ADATAD_PRETRAIN}" \
  "${DUCA_RIME_ADATAD_PRETRAIN_SHA256}" \
  "VideoMAE-S pretrain"

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_PHASE1_SPLIT_ROLE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
if role not in manifest["train_roles"]:
    raise SystemExit("Phase-1 cost split role is not registered")
print(manifest["train_roles"][role]["block_list_path"])
print(manifest["train_roles"][role]["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 2 ]] \
  || fail "failed to resolve the Phase-1 cost split role"
check_sha256 "${split_values[0]}" "${split_values[1]}" "Phase-1 block list"
export DUCA_RIME_PHASE1_EVAL_BLOCK_LIST="${split_values[0]}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_PHASE1_NO_PROBE_CONFIG}" \
    "${DUCA_RIME_PHASE1_PROBE_CONFIG}" <<'PY'
import sys
from mmengine.config import Config

no_probe = Config.fromfile(sys.argv[1])
probe = Config.fromfile(sys.argv[2])
assert (
    no_probe.duca_rime_phase1_cost_contract.contract
    == "duca_rime_phase1_no_probe_uniform_cost_v1"
)
assert no_probe.duca_rime_phase1_cost_contract.coarse_probe_executed is False
assert no_probe.duca_rime_phase1_cost_contract.paired_checkpoint_identity_required
assert (
    probe.duca_rime_phase1_cost_contract.contract
    == "duca_rime_phase1_probe_uniform_cost_v1"
)
assert probe.duca_rime_phase1_cost_contract.coarse_probe_executed is True
assert probe.duca_rime_phase1_cost_contract.probe_output_used_for_selection is False
assert probe.duca_rime_phase1_cost_contract.paired_checkpoint_identity_required
PY
  echo "[DUCA_RIME_PHASE1_COST] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE1_COST_ROOT}"
power_args=()
if [[ "${DUCA_RIME_PHASE1_SAMPLE_POWER:-1}" == 1 ]]; then
  power_args+=(--sample-power)
fi

run_profile() {
  local method="$1"
  local config="$2"
  local checkpoint="$3"
  local trained_commit="$4"
  local output_name="$5"
  local order_position="$6"
  python tools/bata/profile_duca_full_stack_cost.py \
    "${config}" \
    --checkpoint "${checkpoint}" \
    --backbone-pretrain "${DUCA_RIME_ADATAD_PRETRAIN}" \
    --output-prefix "${DUCA_RIME_PHASE1_COST_ROOT}/${output_name}" \
    --method-name "${method}" \
    --config-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --trained-commit "${trained_commit}" \
    --evidence-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --device cuda:0 \
    --samples "${DUCA_RIME_PHASE1_PROFILE_SAMPLES:-30}" \
    --warmup-samples "${DUCA_RIME_PHASE1_PROFILE_WARMUP:-5}" \
    --batch-size 1 \
    --loader-workers 0 \
    --amp \
    --use-ema \
    --profile-session-id "${DUCA_RIME_PHASE1_PROFILE_SESSION_ID}" \
    --profile-pair-id "${DUCA_RIME_PHASE1_PROFILE_PAIR_ID}" \
    --profile-repeat-index 1 \
    --profile-order-position "${order_position}" \
    "${power_args[@]}"
}

if [[ "${DUCA_RIME_PHASE1_PROFILE_ORDER:-no_probe_first}" == no_probe_first ]]; then
  run_profile \
    phase1-no-probe-uniform \
    "${DUCA_RIME_PHASE1_NO_PROBE_CONFIG}" \
    "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT}" \
    "${DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT}" \
    no_probe_uniform \
    1
  run_profile \
    phase1-probe-uniform \
    "${DUCA_RIME_PHASE1_PROBE_CONFIG}" \
    "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT}" \
    "${DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT}" \
    probe_uniform \
    2
else
  run_profile \
    phase1-probe-uniform \
    "${DUCA_RIME_PHASE1_PROBE_CONFIG}" \
    "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT}" \
    "${DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT}" \
    probe_uniform \
    1
  run_profile \
    phase1-no-probe-uniform \
    "${DUCA_RIME_PHASE1_NO_PROBE_CONFIG}" \
    "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT}" \
    "${DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT}" \
    no_probe_uniform \
    2
fi

python - \
  "${DUCA_RIME_PHASE1_COST_ROOT}/no_probe_uniform.summary.json" \
  "${DUCA_RIME_PHASE1_COST_ROOT}/probe_uniform.summary.json" <<'PY'
import json
import sys
from tools.bata.duca_full_stack_cost import validate_and_rebuild_profile_summary

left = json.load(open(sys.argv[1], encoding="utf-8"))
right = json.load(open(sys.argv[2], encoding="utf-8"))
validate_and_rebuild_profile_summary(left)
validate_and_rebuild_profile_summary(right)
for key in (
    "protocol",
    "hardware_fingerprint",
    "host_fingerprint",
    "software_fingerprint",
    "profile_session_id",
    "profile_pair_id",
    "sample_count",
    "loader_workers",
    "amp",
    "uses_ema",
    "checkpoint_sha256",
    "trained_commit",
    "checkpoint_epoch",
    "checkpoint_state_key",
):
    if left.get(key) != right.get(key):
        raise SystemExit(f"paired Phase-1 cost profiles differ on {key}")
if left["phase1_cost_contract"]["coarse_probe_executed"] is not False:
    raise SystemExit("no-probe profile contract drift")
if right["phase1_cost_contract"]["coarse_probe_executed"] is not True:
    raise SystemExit("probe profile contract drift")
if float(left["stages"]["coarse_probe_ms"]["p50"]) != 0.0:
    raise SystemExit("no-probe control executed the coarse probe")
if float(right["stages"]["coarse_probe_ms"]["p50"]) <= 0.0:
    raise SystemExit("probe control did not execute a measurable coarse probe")
PY

printf '%s\n' \
  "schema=duca_rime_phase1_cost_pair_receipt_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "profile_session_id=${DUCA_RIME_PHASE1_PROFILE_SESSION_ID}" \
  "profile_pair_id=${DUCA_RIME_PHASE1_PROFILE_PAIR_ID}" \
  "shared_checkpoint_sha256=${DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256}" \
  "shared_trained_commit=${DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT}" \
  "no_probe_summary_sha256=$(sha256sum "${DUCA_RIME_PHASE1_COST_ROOT}/no_probe_uniform.summary.json" | awk '{print $1}')" \
  "probe_summary_sha256=$(sha256sum "${DUCA_RIME_PHASE1_COST_ROOT}/probe_uniform.summary.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE1_COST_ROOT}/cost_pair.receipt"
echo "[DUCA_RIME_PHASE1_COST] PASS ${DUCA_RIME_PHASE1_COST_ROOT}"
