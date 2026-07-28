#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_COUNTERFACTUALS][FAIL] $*" >&2
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
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "formal counterfactual measurement must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "a complete exact Git worktree is required"
[[ ! -e "${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}" ]] \
  || fail "a fresh counterfactual output root is required"
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
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_TRAINING_RECEIPT}" \
    "${DUCA_RIME_CHECKPOINT_SHA256}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" \
    "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

from tools.bata.create_duca_rime_splits import validate_rime_splits

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    receipt.get("schema_version")
    != "duca_rime_phase2_mixed_k_training_receipt_v1"
    or receipt.get("status") != "passed"
    or receipt.get("arm") != "U-mixed-K"
    or receipt.get("checkpoint_sha256") != sys.argv[2]
    or receipt.get("git_commit") != sys.argv[3]
    or receipt.get("detector_training_exposure")
    != "mixed_k_registered_panel"
    or int(receipt.get("successful_detector_updates", -1)) != 6000
    or receipt.get("uses_official_final") is not False
):
    raise SystemExit("invalid mixed-K training receipt")
validation = validate_rime_splits(sys.argv[4])
manifest = json.load(open(sys.argv[4], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    print(manifest["train_roles"][role]["block_list_path"])
    print(manifest["train_roles"][role]["block_list_sha256"])
print(validation["assignment_sha256"])
PY
)
[[ "${#split_values[@]}" == 5 ]] \
  || fail "failed to resolve the frozen train-role artifacts"
check_sha256 "${split_values[0]}" "${split_values[1]}" "detector-train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
[[ "${candidate_budgets[*]}" == "192 256 384 512" ]] \
  || fail "formal counterfactual measurement requires K={192,256,384,512}"

export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"
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
assert tuple(cfg.duca_rime_variant.candidate_budgets) == (192, 256, 384, 512)
assert cfg.model.frame_selector.actionness_source_cfg is None
assert cfg.duca_rime_contract.pad_to_kmax is False
PY
  echo "[DUCA_RIME_PHASE2_COUNTERFACTUALS] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}"
all_train_block_list="${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}/all_train_empty_block_list.txt"
: > "${all_train_block_list}"
export DUCA_RIME_TRAIN_BLOCK_LIST="${all_train_block_list}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${all_train_block_list}"

python tools/bata/produce_duca_rime_counterfactual_measurements.py \
  --config "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --checkpoint "${DUCA_RIME_CHECKPOINT}" \
  --checkpoint-sha256 "${DUCA_RIME_CHECKPOINT_SHA256}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --output-jsonl \
  "${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}/counterfactual_measurements.jsonl" \
  --summary-json \
  "${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}/producer_summary.json" \
  --candidate-budgets "${candidate_budgets[@]}" \
  --frame-swap-budget "${DUCA_RIME_FRAME_SWAP_BUDGET:-384}" \
  --max-frame-counterfactuals \
  "${DUCA_RIME_MAX_FRAME_COUNTERFACTUALS:-16}" \
  --relative-pair-tolerance \
  "${DUCA_RIME_RELATIVE_PAIR_TOLERANCE:-0.05}" \
  --absolute-pair-tolerance \
  "${DUCA_RIME_ABSOLUTE_PAIR_TOLERANCE:-0.01}" \
  --seed "${DUCA_RIME_PHASE2_SEED:-3407}" \
  --device cuda:0 \
  --num-workers "${DUCA_RIME_COUNTERFACTUAL_WORKERS:-2}" \
  --backbone-pretrain "${DUCA_RIME_PRETRAIN_PATH}"

python - \
  "${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${split_values[4]}" <<'PY'
import hashlib
import json
import os
import sys

root, commit, job_id, assignment_sha256 = sys.argv[1:]
measurements = os.path.join(root, "counterfactual_measurements.jsonl")
summary = os.path.join(root, "producer_summary.json")
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
payload = {
    "schema_version": "duca_rime_phase2_counterfactual_production_receipt_v1",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": job_id,
    "split_assignment_sha256": assignment_sha256,
    "measurements_path": os.path.abspath(measurements),
    "measurements_sha256": sha(measurements),
    "producer_summary_path": os.path.abspath(summary),
    "producer_summary_sha256": sha(summary),
    "uses_official_final": False,
    "claim_scope": "train_only_detector_loss_counterfactuals_not_tad_map",
}
output = os.path.join(root, "production_receipt.json")
temporary = output + f".partial.{os.getpid()}"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

echo \
  "[DUCA_RIME_PHASE2_COUNTERFACTUALS] PASS ${DUCA_RIME_PHASE2_COUNTERFACTUAL_ROOT}"
