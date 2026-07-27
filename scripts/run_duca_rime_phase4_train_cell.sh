#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_TRAIN][FAIL] $*" >&2
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
  DUCA_RIME_PHASE4_ARM \
  DUCA_RIME_PHASE4_CONFIG \
  DUCA_RIME_PHASE4_ROOT \
  DUCA_RIME_PHASE4_SEED \
  DUCA_RIME_TARGET_MEAN_COST \
  DUCA_RIME_PHASE4_AUTHORIZATION \
  DUCA_RIME_PHASE4_AUTHORIZATION_SHA256 \
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

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-4 training must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "formal training requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE4_ROOT}" ]] || fail "a fresh Phase-4 cell root is required"
[[ -f "${DUCA_RIME_PHASE4_CONFIG}" ]] || fail "Phase-4 config is missing"
check_sha256 \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" \
  "Phase-4 authorization"
check_sha256 \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "Phase-2 receipt"
check_sha256 \
  "${DUCA_RIME_TRAINING_EXPOSURE_JSON}" \
  "${DUCA_RIME_TRAINING_EXPOSURE_SHA256}" \
  "Phase-4 training exposure"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE pretrain"

case "${DUCA_RIME_PHASE4_ARM}" in
  RIME-full|RIME-full-TriDet)
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
  U-fixed)
    [[ "${DUCA_RIME_FIXED_BUDGET:-}" == "${DUCA_RIME_TARGET_MEAN_COST}" ]] \
      || fail "ActionFormer fixed control must use the authorized budget panel"
    ;;
  U-fixed-TriDet)
    [[ "${DUCA_RIME_FIXED_BUDGET:-}" == "${DUCA_RIME_TARGET_MEAN_COST}" ]] \
      || fail "TriDet fixed control must use the authorized budget panel"
    ;;
  *)
    fail "unregistered Phase-4 train arm: ${DUCA_RIME_PHASE4_ARM}"
    ;;
esac

readarray -t sealed_config_values < <(python - \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE4_CONFIG}" \
  "${DUCA_RIME_PHASE4_ARM}" \
  "${DUCA_RIME_PHASE4_SEED}" \
  "${DUCA_RIME_TARGET_MEAN_COST}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys

from mmengine.config import Config
from tools.bata.duca_p0_evaluation import evaluation_config_sha256

authorization = json.load(open(sys.argv[1], encoding="utf-8"))
config_path, arm, seed, budget, commit = sys.argv[2:]
seed = int(seed)
budget = float(budget)
backend = "TriDet" if arm.endswith("TriDet") else "ActionFormer"
if (
    authorization.get("schema_version") != "duca_rime_stage_receipt_v1"
    or authorization.get("phase") != "phase4_authorization"
    or authorization.get("status") != "authorized"
    or authorization.get("gate_pass") is not True
    or authorization.get("git_commit") != commit
    or seed not in {int(value) for value in authorization.get("formal_seeds", ())}
    or backend not in authorization.get("required_detectors", ())
    or budget not in {
        float(value) for value in authorization.get("required_budget_panels", ())
    }
    or authorization.get("official_final_subset_consumed") is not False
):
    raise SystemExit("Phase-4 authorization does not cover this cell")
cfg = Config.fromfile(config_path)
if cfg.duca_rime_variant.arm != arm:
    raise SystemExit("Phase-4 arm/config mismatch")
if int(cfg.workflow.expected_successful_optimizer_updates) != 6000:
    raise SystemExit("Phase-4 config is not frozen to exactly 6000 updates")
if int(cfg.solver.train.batch_size) != 1:
    raise SystemExit("Phase-4 exact variable-K execution requires batch_size=1")
if arm.startswith("RIME-full") and cfg.duca_rime_contract.pad_to_kmax is not False:
    raise SystemExit("Phase-4 RIME config pads heavy execution to Kmax")
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
print(backend)
PY
)
[[ "${#sealed_config_values[@]}" == 5 ]] || fail "failed to seal the Phase-4 config"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE4_TRAIN] PRECHECK PASS ${DUCA_RIME_PHASE4_ARM}"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE4_ROOT}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_P0_VARIANT="${DUCA_RIME_PHASE4_ARM}"
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
torchrun --standalone --nproc_per_node=1 tools/train.py \
  "${DUCA_RIME_PHASE4_CONFIG}" \
  --seed "${DUCA_RIME_PHASE4_SEED}" \
  --id 0 \
  --cfg-options \
  "work_dir=${DUCA_RIME_PHASE4_ROOT}/train" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

actual_root="${DUCA_RIME_PHASE4_ROOT}/train/gpu1_id0"
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
  "${DUCA_RIME_PHASE4_ARM}" \
  "${DUCA_RIME_PHASE4_SEED}" \
  "${DUCA_RIME_TARGET_MEAN_COST}" \
  "${sealed_config_values[4]}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" \
  "${DUCA_RIME_PHASE4_ROOT}/training_receipt.json" <<'PY'
import hashlib
import json
import os
import sys

(
    audit_path,
    checkpoint_path,
    arm,
    seed,
    budget,
    backend,
    commit,
    authorization_path,
    authorization_sha,
    output,
) = sys.argv[1:]
compaction_path = checkpoint_path + ".receipt.json"
if not all(os.path.isfile(path) for path in (audit_path, checkpoint_path, compaction_path)):
    raise SystemExit("terminal Phase-4 audit/checkpoint evidence is missing")
audit = json.load(open(audit_path, encoding="utf-8"))
compaction = json.load(open(compaction_path, encoding="utf-8"))
updates = audit.get("update_audit", {})
if (
    audit.get("status") != "complete"
    or audit.get("git_commit") != commit
    or audit.get("variant") != arm
    or int(audit.get("seed", -1)) != int(seed)
    or int(audit.get("research_phase", -1)) != 4
    or audit.get("phase4_authorization_sha256") != authorization_sha
    or float(audit.get("formal_budget_panel", -1)) != float(budget)
    or audit.get("detector_backend") != backend
    or int(updates.get("successful_optimizer_updates", -1)) != 6000
    or int(updates.get("scheduler_updates", -1)) != 6000
    or int(updates.get("ema_updates", -1)) != 6000
    or compaction.get("status") != "passed"
    or compaction.get("evaluation_equivalent") is not True
):
    raise SystemExit("terminal Phase-4 audit violates its frozen cell contract")
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
payload = {
    "schema_version": "duca_rime_phase4_training_receipt_v1",
    "status": "passed",
    "research_phase": 4,
    "arm": arm,
    "seed": int(seed),
    "git_commit": commit,
    "detector_backend": backend,
    "target_mean_cost": float(budget),
    "phase4_authorization_path": os.path.abspath(authorization_path),
    "phase4_authorization_sha256": authorization_sha,
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
sha256sum "${DUCA_RIME_PHASE4_ROOT}/training_receipt.json" \
  > "${DUCA_RIME_PHASE4_ROOT}/training_receipt.sha256"
echo "[DUCA_RIME_PHASE4_TRAIN] PASS ${DUCA_RIME_PHASE4_ROOT}/training_receipt.json"
