#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_MIXED_K_EVAL][FAIL] $*" >&2
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
  DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE2_SPLIT_ROLE \
  DUCA_RIME_EVAL_FIXED_BUDGET \
  DUCA_RIME_EVAL_SEED \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-2 mixed-K evaluation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "mixed-K evaluation requires a complete exact Git worktree"
[[ ! -e "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}" ]] \
  || fail "a fresh mixed-K evaluation root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_TRAINING_RECEIPT}" \
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  "mixed-K training receipt"
check_sha256 \
  "${DUCA_RIME_CHECKPOINT}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" \
  "mixed-K terminal checkpoint"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_PHASE2_SPLIT_ROLE}" \
    "${DUCA_RIME_EVAL_FIXED_BUDGET}" \
    "${DUCA_RIME_TRAINING_RECEIPT}" \
    "${DUCA_RIME_CHECKPOINT_SHA256}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
budget = int(sys.argv[3])
receipt = json.load(open(sys.argv[4], encoding="utf-8"))
if role not in manifest["train_roles"]:
    raise SystemExit("mixed-K evaluation role is not registered")
if budget not in {192, 256, 384, 512}:
    raise SystemExit("mixed-K evaluation budget is outside the registered panel")
if (
    receipt.get("schema_version")
    != "duca_rime_phase2_mixed_k_training_receipt_v1"
    or receipt.get("status") != "passed"
    or receipt.get("git_commit") != sys.argv[6]
    or receipt.get("checkpoint_sha256") != sys.argv[5]
    or receipt.get("detector_training_exposure")
    != "mixed_k_registered_panel"
    or receipt.get("uses_official_final") is not False
):
    raise SystemExit("mixed-K training receipt/checkpoint binding is invalid")
train_role = "detector_selector_train"
if train_role not in manifest["train_roles"]:
    raise SystemExit("mixed-K detector-train role is not registered")
print(manifest["train_roles"][role]["block_list_path"])
print(manifest["train_roles"][role]["block_list_sha256"])
print(manifest["train_roles"][train_role]["block_list_path"])
print(manifest["train_roles"][train_role]["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] \
  || fail "failed to resolve the mixed-K train/evaluation roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "evaluation block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "detector-train block list"
export DUCA_RIME_PHASE2_EVAL_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[2]}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
assert cfg.workflow.formal_protocol == "duca_rime_phase2_mixed_k_baseline_v1"
assert cfg.duca_rime_variant.arm == "U-mixed-K"
assert (
    cfg.duca_rime_variant.detector_training_exposure
    == "mixed_k_registered_panel"
)
assert int(cfg.duca_rime_variant.evaluation_budget) in {192, 256, 384, 512}
assert cfg.duca_rime_contract.pad_to_kmax is False
PY
  echo "[DUCA_RIME_PHASE2_MIXED_K_EVAL] PRECHECK PASS"
  exit 0
fi

mkdir -p \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}" \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/ledger"
export DUCA_RIME_INFERENCE_LEDGER_ROOT="${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/ledger"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}" --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --checkpoint "${DUCA_RIME_CHECKPOINT}" \
  --seed "${DUCA_RIME_EVAL_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/terminal_evaluation.json" \
  --cfg-options \
  "work_dir=${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/runtime"

python -m tools.bata.evaluate_duca_rime_predictions \
  --terminal-evaluation \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/terminal_evaluation.json" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase 2 \
  --split-role "${DUCA_RIME_PHASE2_SPLIT_ROLE}" \
  --short-max-seconds "${DUCA_RIME_SHORT_MAX_SECONDS}" \
  --medium-max-seconds "${DUCA_RIME_MEDIUM_MAX_SECONDS}" \
  --output "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/localization_metrics.json"

ledger_shard="${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/ledger/inference_ledger.rank0000.jsonl"
[[ -f "${ledger_shard}" ]] \
  || fail "mixed-K evaluation did not emit its exact-uniform ledger"
python tools/bata/finalize_duca_rime_inference_ledger.py \
  --shard "${ledger_shard}" \
  --output-jsonl \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/inference_ledger.jsonl" \
  --expected-arm uniform_mixed_k \
  --summary-json \
  "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/inference_ledger_summary.json" \
  > /dev/null

printf '%s\n' \
  "schema=duca_rime_phase2_mixed_k_evaluation_receipt_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "split_role=${DUCA_RIME_PHASE2_SPLIT_ROLE}" \
  "budget=${DUCA_RIME_EVAL_FIXED_BUDGET}" \
  "seed=${DUCA_RIME_EVAL_SEED}" \
  "detector_training_exposure=mixed_k_registered_panel" \
  "checkpoint_sha256=${DUCA_RIME_CHECKPOINT_SHA256}" \
  "terminal_evaluation_sha256=$(sha256sum "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/terminal_evaluation.json" | awk '{print $1}')" \
  "localization_metrics_sha256=$(sha256sum "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/localization_metrics.json" | awk '{print $1}')" \
  "inference_ledger_sha256=$(sha256sum "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/inference_ledger.jsonl" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}/evaluation.receipt"
echo \
  "[DUCA_RIME_PHASE2_MIXED_K_EVAL] PASS ${DUCA_RIME_PHASE2_MIXED_K_EVAL_ROOT}"
