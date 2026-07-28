#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[HRIME_STAGE1_EVAL][FAIL] $*" >&2
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
  HRIME_STAGE1_STRATEGY \
  HRIME_STAGE1_ANCHOR_BUDGET \
  HRIME_STAGE1_PLAN_MANIFEST \
  HRIME_STAGE1_PLAN_MANIFEST_SHA256 \
  HRIME_STAGE1_REPLAY_JSONL \
  HRIME_STAGE1_REPLAY_SHA256 \
  HRIME_STAGE1_EVAL_CONFIG \
  HRIME_STAGE1_EVAL_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256 \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_EVAL_SEED; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Stage-1 evaluation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ "${HRIME_STAGE1_ANCHOR_BUDGET}" =~ ^[0-9]+$ ]] \
  || fail "the Stage-1 anchor must be an integer"
[[ "${DUCA_RIME_EVAL_SEED}" == 3407 ]] \
  || fail "Stage-1 is bound to the Phase-3 development seed 3407"
[[ "${DUCA_RIME_TARGET_MEAN_COST:-384}" == 384 ]] \
  || fail "Stage-1 source checkpoint must be the K384 RIME-full panel"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Stage-1 requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${HRIME_STAGE1_EVAL_ROOT}" ]] \
  || fail "a fresh Stage-1 evaluation root is required"

check_sha256 \
  "${HRIME_STAGE1_PLAN_MANIFEST}" \
  "${HRIME_STAGE1_PLAN_MANIFEST_SHA256}" \
  "Stage-1 plan manifest"
check_sha256 \
  "${HRIME_STAGE1_REPLAY_JSONL}" \
  "${HRIME_STAGE1_REPLAY_SHA256}" \
  "Stage-1 replay"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"
check_sha256 \
  "${DUCA_RIME_TARGETS_JSONL}" \
  "${DUCA_RIME_TARGETS_SHA256}" \
  "RIME train-only targets"
check_sha256 \
  "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
  "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  "RIME frozen budget protocol"
check_sha256 \
  "${DUCA_RIME_TRAINING_RECEIPT}" \
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  "RIME-full training receipt"
check_sha256 \
  "${DUCA_RIME_CHECKPOINT}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" \
  "RIME-full terminal checkpoint"
[[ -f "${HRIME_STAGE1_EVAL_CONFIG}" ]] \
  || fail "Stage-1 evaluation config is missing"

readarray -t plan_values < <(
  python - \
    "${HRIME_STAGE1_PLAN_MANIFEST}" \
    "${HRIME_STAGE1_STRATEGY}" \
    "${HRIME_STAGE1_ANCHOR_BUDGET}" \
    "${HRIME_STAGE1_REPLAY_JSONL}" \
    "${HRIME_STAGE1_REPLAY_SHA256}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import json
import pathlib
import os
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
strategy = sys.argv[2]
anchor = int(sys.argv[3])
replay = pathlib.Path(sys.argv[4]).resolve()
replay_sha = sys.argv[5]
commit = sys.argv[6]
if (
    plan.get("schema_version") != "hrime_stage1_oracle_plan_v1"
    or plan.get("status") != "planned"
    or plan.get("git_commit") != commit
    or plan.get("development_role") != "certification_development"
    or plan.get("uses_official_final") is not False
    or plan.get("authorizes_stage2_training") is not False
):
    raise SystemExit("Stage-1 plan contract drift")
try:
    binding = plan["replay_artifacts"][strategy][str(anchor)]
    contract = plan["strategies"][strategy]
except KeyError as exc:
    raise SystemExit(f"unregistered Stage-1 cell: {exc}") from exc
if pathlib.Path(binding["path"]).resolve() != replay or binding["sha256"] != replay_sha:
    raise SystemExit("Stage-1 replay differs from its plan binding")
role = contract["role"]
position_policy = contract["position_policy"]
if position_policy == "exact_uniform":
    expected_variant = "H-RIME-Stage1-Uniform-Positions"
    expected_ledger_arm = "hrime_stage1_uniform_positions"
elif position_policy == "frozen_rime_selector":
    expected_variant = "H-RIME-Stage1-Learned-Positions"
    expected_ledger_arm = "hrime_stage1_learned_positions"
else:
    raise SystemExit("unsupported Stage-1 position policy")
print(role)
print(expected_variant)
print(expected_ledger_arm)
PY
)
[[ "${#plan_values[@]}" == 3 ]] \
  || fail "failed to resolve the Stage-1 strategy contract"
decision_role="${plan_values[0]}"
expected_variant="${plan_values[1]}"
expected_ledger_arm="${plan_values[2]}"

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
export DUCA_RIME_REPLAY_JSONL="${HRIME_STAGE1_REPLAY_JSONL}"
export DUCA_RIME_REPLAY_SHA256="${HRIME_STAGE1_REPLAY_SHA256}"
export DUCA_RIME_ALLOW_ORACLE_REPLAY=1
export DUCA_RIME_TARGET_MEAN_COST=384
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export HRIME_STAGE1_DECISION_ROLE="${decision_role}"

python - "${HRIME_STAGE1_EVAL_CONFIG}" "${expected_variant}" <<'PY'
import os
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
if (
    str(cfg.duca_rime_variant.arm) != sys.argv[2]
    or str(cfg.evaluation.subset) != "training"
    or not cfg.dataset.test.block_list
    or cfg.duca_rime_contract.official_final_subset_consumed is not False
    or cfg.duca_rime_variant.oracle_only is not True
    or cfg.duca_rime_variant.deployment_candidate is not False
    or cfg.workflow.evaluation_protocol != "hrime_stage1_oracle_execution_v1"
    or cfg.hrime_stage1_execution_contract.decision_role
    != os.environ["HRIME_STAGE1_DECISION_ROLE"]
):
    raise SystemExit("Stage-1 evaluation config contract drift")
PY

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[HRIME_STAGE1_EVAL] PRECHECK PASS ${HRIME_STAGE1_STRATEGY} K${HRIME_STAGE1_ANCHOR_BUDGET}"
  exit 0
fi

mkdir -p "${HRIME_STAGE1_EVAL_ROOT}/ledger"
export DUCA_RIME_INFERENCE_LEDGER_ROOT="${HRIME_STAGE1_EVAL_ROOT}/ledger"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}" --nproc_per_node=1 tools/test.py \
  "${HRIME_STAGE1_EVAL_CONFIG}" \
  --checkpoint "${DUCA_RIME_CHECKPOINT}" \
  --seed "${DUCA_RIME_EVAL_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${HRIME_STAGE1_EVAL_ROOT}/terminal_evaluation.json" \
  --cfg-options "work_dir=${HRIME_STAGE1_EVAL_ROOT}/runtime"

python -m tools.bata.evaluate_duca_rime_predictions \
  --terminal-evaluation "${HRIME_STAGE1_EVAL_ROOT}/terminal_evaluation.json" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase 3 \
  --short-max-seconds "${DUCA_RIME_SHORT_MAX_SECONDS}" \
  --medium-max-seconds "${DUCA_RIME_MEDIUM_MAX_SECONDS}" \
  --output "${HRIME_STAGE1_EVAL_ROOT}/localization_metrics.json"

ledger_shard="${HRIME_STAGE1_EVAL_ROOT}/ledger/inference_ledger.rank0000.jsonl"
[[ -f "${ledger_shard}" ]] \
  || fail "Stage-1 selector did not emit an inference ledger"
python tools/bata/finalize_duca_rime_inference_ledger.py \
  --shard "${ledger_shard}" \
  --output-jsonl "${HRIME_STAGE1_EVAL_ROOT}/inference_ledger.jsonl" \
  --expected-arm "${expected_ledger_arm}" \
  --expected-protocol-sha256 "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
  --require-explicit-budget-truth \
  --allow-oracle-only \
  --expected-decision-role "${decision_role}" \
  --summary-json "${HRIME_STAGE1_EVAL_ROOT}/inference_ledger_summary.json" \
  > "${HRIME_STAGE1_EVAL_ROOT}/inference_ledger_stdout.json"

python -m tools.bata.hrime_stage1_oracle validate-execution \
  --plan-manifest "${HRIME_STAGE1_PLAN_MANIFEST}" \
  --plan-manifest-sha256 "${HRIME_STAGE1_PLAN_MANIFEST_SHA256}" \
  --strategy "${HRIME_STAGE1_STRATEGY}" \
  --anchor-nominal-budget "${HRIME_STAGE1_ANCHOR_BUDGET}" \
  --replay-jsonl "${HRIME_STAGE1_REPLAY_JSONL}" \
  --replay-sha256 "${HRIME_STAGE1_REPLAY_SHA256}" \
  --inference-ledger-jsonl "${HRIME_STAGE1_EVAL_ROOT}/inference_ledger.jsonl" \
  --terminal-evaluation "${HRIME_STAGE1_EVAL_ROOT}/terminal_evaluation.json" \
  --localization-metrics "${HRIME_STAGE1_EVAL_ROOT}/localization_metrics.json" \
  --output-receipt "${HRIME_STAGE1_EVAL_ROOT}/execution_receipt.json"

echo "[HRIME_STAGE1_EVAL] PASS ${HRIME_STAGE1_EVAL_ROOT}/execution_receipt.json"
