#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_TRAIN][FAIL] $*" >&2
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
  DUCA_RIME_PHASE3_ARM \
  DUCA_RIME_PHASE3_CONFIG \
  DUCA_RIME_PHASE3_ROOT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_TRAIN_BLOCK_LIST \
  DUCA_RIME_DEVELOPMENT_BLOCK_LIST \
  DUCA_RIME_TRAINING_EXPOSURE_JSON \
  DUCA_RIME_TRAINING_EXPOSURE_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-3 training must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "formal training requires a complete exact Git worktree, not an overlay"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE3_ROOT}" ]] || fail "a fresh Phase-3 arm root is required"
[[ -f "${DUCA_RIME_PHASE3_CONFIG}" ]] || fail "Phase-3 config is missing"
check_sha256 \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "Phase-2 receipt"
check_sha256 "${DUCA_RIME_PRETRAIN_PATH}" "${DUCA_RIME_PRETRAIN_SHA256}" "VideoMAE pretrain"
check_sha256 \
  "${DUCA_RIME_TRAINING_EXPOSURE_JSON}" \
  "${DUCA_RIME_TRAINING_EXPOSURE_SHA256}" \
  "shared training exposure"

case "${DUCA_RIME_PHASE3_ARM}" in
  U-fixed)
    ;;
  F-bound|D-no-risk|RIME-full)
    for name in \
      DUCA_RIME_TARGETS_JSONL \
      DUCA_RIME_TARGETS_SHA256 \
      DUCA_RIME_BUDGET_PROTOCOL_JSON \
      DUCA_RIME_BUDGET_PROTOCOL_SHA256; do
      required "${name}"
    done
    check_sha256 \
      "${DUCA_RIME_TARGETS_JSONL}" \
      "${DUCA_RIME_TARGETS_SHA256}" \
      "cross-fitted RIME targets"
    check_sha256 \
      "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
      "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
      "frozen budget protocol"
    ;;
  D-shuffle|AdapTok-TAD)
    for name in \
      DUCA_RIME_TARGETS_JSONL \
      DUCA_RIME_TARGETS_SHA256 \
      DUCA_RIME_BUDGET_PROTOCOL_JSON \
      DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
      DUCA_RIME_REPLAY_JSONL \
      DUCA_RIME_REPLAY_SHA256; do
      required "${name}"
    done
    check_sha256 \
      "${DUCA_RIME_TARGETS_JSONL}" \
      "${DUCA_RIME_TARGETS_SHA256}" \
      "cross-fitted RIME targets"
    check_sha256 \
      "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" \
      "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}" \
      "frozen budget protocol"
    check_sha256 \
      "${DUCA_RIME_REPLAY_JSONL}" \
      "${DUCA_RIME_REPLAY_SHA256}" \
      "arm budget replay"
    ;;
  U-same-K)
    fail "U-same-K is evaluation-only and cannot enter the training launcher"
    ;;
  *)
    fail "unregistered Phase-3 arm: ${DUCA_RIME_PHASE3_ARM}"
    ;;
esac

readarray -t sealed_config_values < <(python - \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE3_CONFIG}" \
  "${DUCA_RIME_PHASE3_ARM}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys

from mmengine.config import Config
from tools.bata.duca_p0_evaluation import evaluation_config_sha256

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    receipt.get("schema_version") != "duca_rime_stage_receipt_v1"
    or receipt.get("phase") != "phase2"
    or receipt.get("gate_pass") is not True
    or receipt.get("phase3_training_authorized") is not True
    or receipt.get("official_final_subset_consumed") is not False
    or receipt.get("git_commit") != sys.argv[4]
):
    raise SystemExit("Phase-2 receipt does not authorize Phase-3")
cfg = Config.fromfile(sys.argv[2])
if cfg.duca_rime_variant.arm != sys.argv[3]:
    raise SystemExit("Phase-3 arm/config mismatch")
if int(cfg.workflow.expected_successful_optimizer_updates) != 6000:
    raise SystemExit("Phase-3 config is not frozen to exactly 6000 updates")
if int(cfg.solver.train.batch_size) != 1:
    raise SystemExit("Phase-3 exact variable-K execution requires batch_size=1")
if sys.argv[3] != "U-fixed":
    if cfg.duca_rime_contract.pad_to_kmax is not False:
        raise SystemExit("Phase-3 config pads heavy execution to Kmax")
canonical = json.dumps(
    cfg.to_dict(),
    sort_keys=True,
    separators=(",", ":"),
    default=str,
)
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
print(
    evaluation_config_sha256(
        cfg.evaluation,
        expected_subset=str(cfg.evaluation.subset),
    )
)
print(cfg.evaluation.ground_truth_filename)
print(cfg.dataset.test.class_map)
PY
)
[[ "${#sealed_config_values[@]}" == 4 ]] || fail "failed to seal the Phase-3 config"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE3_TRAIN] PRECHECK PASS ${DUCA_RIME_PHASE3_ARM}"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE3_ROOT}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_P0_VARIANT="${DUCA_RIME_PHASE3_ARM}"
export DUCA_RESOLVED_CONFIG_SHA256="${sealed_config_values[0]}"
export DUCA_RIME_EVALUATION_CONFIG_SHA256="${sealed_config_values[1]}"
export DUCA_RIME_EVALUATION_ANNOTATION_SHA256="$(
  sha256sum "${sealed_config_values[2]}" | awk '{print $1}'
)"
export DUCA_RIME_EVALUATION_CLASS_MAP_SHA256="$(
  sha256sum "${sealed_config_values[3]}" | awk '{print $1}'
)"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}" --nproc_per_node=1 tools/train.py \
  "${DUCA_RIME_PHASE3_CONFIG}" \
  --seed "${DUCA_RIME_PHASE3_SEED:-3407}" \
  --id 0 \
  --cfg-options \
  "work_dir=${DUCA_RIME_PHASE3_ROOT}/train" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

actual_root="${DUCA_RIME_PHASE3_ROOT}/train/gpu1_id0"
audit="${actual_root}/duca_rime_training_audit.json"
full_checkpoint="${actual_root}/checkpoint/epoch_59.pth"
checkpoint="${actual_root}/checkpoint/terminal_ema.pth"
python tools/bata/compact_duca_rime_checkpoint.py \
  --source "${full_checkpoint}" \
  --output "${checkpoint}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --remove-source
python - \
  "${audit}" \
  "${checkpoint}" \
  "${DUCA_RIME_PHASE3_ARM}" \
  "${DUCA_RIME_PHASE3_SEED:-3407}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_PHASE3_ROOT}/training_receipt.json" <<'PY'
import hashlib
import json
import os
import sys

audit_path, checkpoint_path, arm, seed, commit, output = sys.argv[1:]
compaction_receipt = checkpoint_path + ".receipt.json"
if (
    not os.path.isfile(audit_path)
    or not os.path.isfile(checkpoint_path)
    or not os.path.isfile(compaction_receipt)
):
    raise SystemExit("terminal audit/checkpoint is missing")
audit = json.load(open(audit_path, encoding="utf-8"))
compaction = json.load(open(compaction_receipt, encoding="utf-8"))
updates = audit.get("update_audit", {})
if (
    audit.get("status") != "complete"
    or audit.get("git_commit") != commit
    or audit.get("variant") != arm
    or int(audit.get("seed", -1)) != int(seed)
    or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
    or int(updates.get("successful_optimizer_updates", -1)) != 6000
    or int(updates.get("scheduler_updates", -1)) != 6000
    or int(updates.get("ema_updates", -1)) != 6000
    or compaction.get("schema_version")
    != "duca_rime_compact_checkpoint_receipt_v1"
    or compaction.get("status") != "passed"
    or compaction.get("evaluation_equivalent") is not True
    or compaction.get("training_resume_supported") is not False
):
    raise SystemExit("terminal training audit violates the 6000-update contract")
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
payload = {
    "schema_version": "duca_rime_phase3_training_receipt_v1",
    "status": "passed",
    "arm": arm,
    "seed": int(seed),
    "git_commit": commit,
    "successful_detector_updates": 6000,
    "formal_update_audit_passed": True,
    "training_audit_path": os.path.abspath(audit_path),
    "training_audit_sha256": sha(audit_path),
    "checkpoint_path": os.path.abspath(checkpoint_path),
    "checkpoint_sha256": sha(checkpoint_path),
    "checkpoint_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "checkpoint_compaction_receipt_path": os.path.abspath(compaction_receipt),
    "checkpoint_compaction_receipt_sha256": sha(compaction_receipt),
    "uses_official_final": False,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
sha256sum "${DUCA_RIME_PHASE3_ROOT}/training_receipt.json" \
  > "${DUCA_RIME_PHASE3_ROOT}/training_receipt.sha256"
echo "[DUCA_RIME_PHASE3_TRAIN] PASS ${DUCA_RIME_PHASE3_ROOT}/training_receipt.json"
