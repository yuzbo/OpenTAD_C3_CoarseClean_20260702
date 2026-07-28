#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[HRIME_STAGE1_PREPARE][FAIL] $*" >&2
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
  HRIME_STAGE1_PREPARE_ROOT \
  HRIME_STAGE1_SOURCE_CONFIG \
  HRIME_STAGE1_PREREGISTRATION \
  HRIME_STAGE1_PREREGISTRATION_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Stage-1 preparation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ "${DUCA_RIME_TARGET_MEAN_COST:-384}" == 384 ]] \
  || fail "Stage-1 source checkpoint must use the K384 panel"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Stage-1 preparation requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${HRIME_STAGE1_PREPARE_ROOT}" ]] \
  || fail "a fresh Stage-1 preparation root is required"

check_sha256 \
  "${HRIME_STAGE1_PREREGISTRATION}" \
  "${HRIME_STAGE1_PREREGISTRATION_SHA256}" \
  "Stage-1 preregistration"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"
check_sha256 \
  "${DUCA_RIME_TARGETS_JSONL}" \
  "${DUCA_RIME_TARGETS_SHA256}" \
  "RIME targets"
check_sha256 \
  "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
  "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  "RIME budget protocol"
check_sha256 \
  "${DUCA_RIME_TRAINING_RECEIPT}" \
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  "RIME-full training receipt"
check_sha256 \
  "${DUCA_RIME_CHECKPOINT}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" \
  "RIME-full terminal checkpoint"
[[ -f "${HRIME_STAGE1_SOURCE_CONFIG}" ]] \
  || fail "Stage-1 source config is missing"

readarray -t split_values < <(
  python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    binding = manifest["train_roles"][role]
    print(binding["block_list_path"])
    print(binding["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] \
  || fail "failed to resolve Stage-1 split roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "detector-train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"

export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"
export DUCA_RIME_TARGET_MEAN_COST=384
unset DUCA_RIME_REPLAY_JSONL
unset DUCA_RIME_REPLAY_SHA256
unset DUCA_RIME_ALLOW_ORACLE_REPLAY
unset DUCA_RIME_INFERENCE_LEDGER_ROOT

python - \
  "${HRIME_STAGE1_PREREGISTRATION}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    payload.get("schema_version") != "hrime_stage1_preregistration_v1"
    or payload.get("status") != "frozen"
    or payload.get("git_commit") != sys.argv[2]
    or payload.get("split_manifest_sha256") != sys.argv[3]
    or payload.get("development_role") != "certification_development"
    or payload.get("uses_official_final") is not False
):
    raise SystemExit("Stage-1 preregistration contract drift")
PY

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python -m py_compile \
    tools/bata/produce_hrime_stage1_window_options.py \
    tools/bata/hrime_stage1_oracle.py
  echo "[HRIME_STAGE1_PREPARE] PRECHECK PASS"
  exit 0
fi

mkdir -p "${HRIME_STAGE1_PREPARE_ROOT}"
producer_command=(
  python
  -m
  tools.bata.produce_hrime_stage1_window_options
  --config
  "${HRIME_STAGE1_SOURCE_CONFIG}"
  --checkpoint
  "${DUCA_RIME_CHECKPOINT}"
  --checkpoint-sha256
  "${DUCA_RIME_CHECKPOINT_SHA256}"
  --training-receipt
  "${DUCA_RIME_TRAINING_RECEIPT}"
  --training-receipt-sha256
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}"
  --split-manifest
  "${DUCA_RIME_SPLIT_MANIFEST}"
  --split-manifest-sha256
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}"
  --preregistration
  "${HRIME_STAGE1_PREREGISTRATION}"
  --preregistration-sha256
  "${HRIME_STAGE1_PREREGISTRATION_SHA256}"
  --output-jsonl
  "${HRIME_STAGE1_PREPARE_ROOT}/window_options.jsonl"
  --summary-json
  "${HRIME_STAGE1_PREPARE_ROOT}/window_options_summary.json"
  --seed
  3407
  --device
  cuda:0
  --num-workers
  "${HRIME_STAGE1_NUM_WORKERS:-2}"
)
if [[ -n "${DUCA_RIME_BACKBONE_PRETRAIN:-}" ]]; then
  producer_command+=(--backbone-pretrain "${DUCA_RIME_BACKBONE_PRETRAIN}")
fi
"${producer_command[@]}"

window_options_sha="$(
  sha256sum "${HRIME_STAGE1_PREPARE_ROOT}/window_options.jsonl" | awk '{print $1}'
)"
python -m tools.bata.hrime_stage1_oracle plan \
  --preregistration "${HRIME_STAGE1_PREREGISTRATION}" \
  --preregistration-sha256 "${HRIME_STAGE1_PREREGISTRATION_SHA256}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --window-options-jsonl "${HRIME_STAGE1_PREPARE_ROOT}/window_options.jsonl" \
  --window-options-sha256 "${window_options_sha}" \
  --budget-protocol-sha256 "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  --output-root "${HRIME_STAGE1_PREPARE_ROOT}/plan"

printf '%s\n' \
  "schema=hrime_stage1_prepare_receipt_v1" \
  "status=passed" \
  "claim_scope=development_oracle_preparation_not_result" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "uses_official_final=false" \
  "authorizes_stage2_training=false" \
  "window_options_sha256=${window_options_sha}" \
  "window_options_summary_sha256=$(sha256sum "${HRIME_STAGE1_PREPARE_ROOT}/window_options_summary.json" | awk '{print $1}')" \
  "plan_manifest_sha256=$(sha256sum "${HRIME_STAGE1_PREPARE_ROOT}/plan/plan_manifest.json" | awk '{print $1}')" \
  > "${HRIME_STAGE1_PREPARE_ROOT}/prepare.receipt"

echo "[HRIME_STAGE1_PREPARE] PASS ${HRIME_STAGE1_PREPARE_ROOT}/plan/plan_manifest.json"
