#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_MIXED_K_TRAIN][FAIL] $*" >&2
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
  DUCA_RIME_PHASE2_MIXED_K_ROOT \
  DUCA_RIME_PHASE1_RECEIPT \
  DUCA_RIME_PHASE1_RECEIPT_SHA256 \
  DUCA_RIME_TRAIN_BLOCK_LIST \
  DUCA_RIME_DEVELOPMENT_BLOCK_LIST \
  DUCA_RIME_TRAINING_EXPOSURE_JSON \
  DUCA_RIME_TRAINING_EXPOSURE_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-2 mixed-K training must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "mixed-K training requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE2_MIXED_K_ROOT}" ]] \
  || fail "a fresh Phase-2 mixed-K root is required"
[[ -f "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" ]] \
  || fail "mixed-K config is missing"
check_sha256 \
  "${DUCA_RIME_PHASE1_RECEIPT}" \
  "${DUCA_RIME_PHASE1_RECEIPT_SHA256}" \
  "Phase-1 receipt"
check_sha256 \
  "${DUCA_RIME_TRAINING_EXPOSURE_JSON}" \
  "${DUCA_RIME_TRAINING_EXPOSURE_SHA256}" \
  "mixed-K training exposure"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"

unset \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_REPLAY_JSONL \
  DUCA_RIME_REPLAY_SHA256

readarray -t sealed_config_values < <(
  python - \
    "${DUCA_RIME_PHASE1_RECEIPT}" \
    "${DUCA_RIME_TRAINING_EXPOSURE_JSON}" \
    "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys

from mmengine.config import Config
from tools.bata.duca_p0_evaluation import evaluation_config_sha256

phase1 = json.load(open(sys.argv[1], encoding="utf-8"))
exposure = json.load(open(sys.argv[2], encoding="utf-8"))
if (
    phase1.get("schema_version") != "duca_rime_stage_receipt_v1"
    or phase1.get("phase") != "phase1"
    or phase1.get("gate_pass") is not True
    or phase1.get("official_final_subset_consumed") is not False
    or phase1.get("git_commit") != sys.argv[4]
):
    raise SystemExit("Phase-1 receipt does not authorize mixed-K training")
if (
    exposure.get("schema_version")
    != "duca_rime_phase2_mixed_k_training_exposure_v1"
    or exposure.get("git_commit") != sys.argv[4]
    or exposure.get("split_assignment_sha256")
    != phase1.get("split_assignment_sha256")
    or int(exposure.get("seed", -1)) != 3407
    or exposure.get("detector_backend") != "ActionFormer"
    or float(exposure.get("target_mean_cost", -1.0)) != 384.0
    or int(exposure.get("successful_detector_updates", -1)) != 6000
    or exposure.get("official_final_subset_consumed") is not False
):
    raise SystemExit("mixed-K training exposure is invalid")
cfg = Config.fromfile(sys.argv[3])
if (
    cfg.workflow.formal_protocol
    != "duca_rime_phase2_mixed_k_baseline_v1"
    or cfg.duca_rime_variant.arm != "U-mixed-K"
    or tuple(cfg.duca_rime_variant.candidate_budgets)
    != (192, 256, 384, 512)
    or tuple(cfg.duca_rime_variant.training_schedule_counts)
    != (8, 12, 16, 24)
    or float(cfg.duca_rime_variant.training_target_mean_cost) != 384.0
    or cfg.duca_rime_variant.coarse_probe_executed is not False
    or int(cfg.workflow.expected_successful_optimizer_updates) != 6000
    or int(cfg.solver.train.batch_size) != 1
    or cfg.duca_rime_contract.pad_to_kmax is not False
):
    raise SystemExit("mixed-K config contract drift")
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
[[ "${#sealed_config_values[@]}" == 4 ]] \
  || fail "failed to seal the mixed-K config"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE2_MIXED_K_TRAIN] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE2_MIXED_K_ROOT}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_P0_VARIANT="U-mixed-K"
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
  "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --seed 3407 \
  --id 0 \
  --cfg-options \
  "work_dir=${DUCA_RIME_PHASE2_MIXED_K_ROOT}/train" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

actual_root="${DUCA_RIME_PHASE2_MIXED_K_ROOT}/train/gpu1_id0"
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
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_PHASE2_MIXED_K_ROOT}/training_receipt.json" <<'PY'
import hashlib
import json
import os
import sys

audit_path, checkpoint_path, commit, output = sys.argv[1:]
compaction_path = checkpoint_path + ".receipt.json"
if not all(os.path.isfile(path) for path in (audit_path, checkpoint_path, compaction_path)):
    raise SystemExit("mixed-K terminal audit/checkpoint is missing")
audit = json.load(open(audit_path, encoding="utf-8"))
compaction = json.load(open(compaction_path, encoding="utf-8"))
updates = audit.get("update_audit", {})
if (
    audit.get("status") != "complete"
    or audit.get("git_commit") != commit
    or audit.get("variant") != "U-mixed-K"
    or int(audit.get("seed", -1)) != 3407
    or int(audit.get("research_phase", -1)) != 2
    or float(audit.get("formal_budget_panel", -1.0)) != 384.0
    or audit.get("detector_backend") != "ActionFormer"
    or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
    or int(updates.get("successful_optimizer_updates", -1)) != 6000
    or int(updates.get("scheduler_updates", -1)) != 6000
    or int(updates.get("ema_updates", -1)) != 6000
    or int(updates.get("duca_schedule_updates", -1)) != 6000
    or compaction.get("schema_version")
    != "duca_rime_compact_checkpoint_receipt_v1"
    or compaction.get("status") != "passed"
    or compaction.get("evaluation_equivalent") is not True
    or compaction.get("training_resume_supported") is not False
):
    raise SystemExit("mixed-K training audit violates its frozen contract")
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
payload = {
    "schema_version": "duca_rime_phase2_mixed_k_training_receipt_v1",
    "status": "passed",
    "research_phase": 2,
    "arm": "U-mixed-K",
    "seed": 3407,
    "git_commit": commit,
    "detector_backend": "ActionFormer",
    "target_mean_cost": 384.0,
    "candidate_budgets": [192, 256, 384, 512],
    "training_schedule_counts": [8, 12, 16, 24],
    "detector_training_exposure": "mixed_k_registered_panel",
    "successful_detector_updates": 6000,
    "formal_update_audit_passed": True,
    "training_audit_path": os.path.abspath(audit_path),
    "training_audit_sha256": sha(audit_path),
    "checkpoint_path": os.path.abspath(checkpoint_path),
    "checkpoint_sha256": sha(checkpoint_path),
    "checkpoint_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "checkpoint_compaction_receipt_path": os.path.abspath(compaction_path),
    "checkpoint_compaction_receipt_sha256": sha(compaction_path),
    "uses_official_final": False,
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
sha256sum "${DUCA_RIME_PHASE2_MIXED_K_ROOT}/training_receipt.json" \
  > "${DUCA_RIME_PHASE2_MIXED_K_ROOT}/training_receipt.sha256"
echo \
  "[DUCA_RIME_PHASE2_MIXED_K_TRAIN] PASS ${DUCA_RIME_PHASE2_MIXED_K_ROOT}/training_receipt.json"
