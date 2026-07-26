#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_RATE_STAGE2_RECOVERY][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_RATE_STAGE2_RECOVERY_RUN_ROOT:-}"
STAGE1_CHECKPOINT="${DUCA_STAGE1_REUSE_CHECKPOINT:-}"
STAGE1_CHECKPOINT_SHA256="${DUCA_STAGE1_REUSE_CHECKPOINT_SHA256:-}"
STAGE2_CHECKPOINT="${DUCA_STAGE2_RECOVERY_CHECKPOINT:-}"
STAGE2_CHECKPOINT_SHA256="${DUCA_STAGE2_RECOVERY_CHECKPOINT_SHA256:-}"
STAGE2_CHECKPOINT_EPOCH="${DUCA_STAGE2_RECOVERY_CHECKPOINT_EPOCH:-}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh run root is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"
[[ -f "${STAGE1_CHECKPOINT}" ]] || fail "sealed Stage-1 checkpoint is missing"
[[ "${STAGE1_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "sealed Stage-1 SHA256 is required"
[[ "$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')" == "${STAGE1_CHECKPOINT_SHA256}" ]] || fail "sealed Stage-1 SHA256 mismatch"
[[ -f "${STAGE2_CHECKPOINT}" ]] || fail "sealed Stage-2 recovery checkpoint is missing"
[[ "${STAGE2_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "sealed Stage-2 recovery SHA256 is required"
[[ "$(sha256sum "${STAGE2_CHECKPOINT}" | awk '{print $1}')" == "${STAGE2_CHECKPOINT_SHA256}" ]] || fail "sealed Stage-2 recovery SHA256 mismatch"
[[ "${STAGE2_CHECKPOINT_EPOCH}" == "9" ]] || fail "recovery must start from sealed Stage-2 epoch 9"

export DUCA_STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_CHECKPOINT_SHA256}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
export DUCA_STAGE2_RECOVERY_CHECKPOINT="${STAGE2_CHECKPOINT}"
export DUCA_STAGE2_RECOVERY_CHECKPOINT_SHA256="${STAGE2_CHECKPOINT_SHA256}"
export DUCA_STAGE2_RECOVERY_CHECKPOINT_EPOCH="${STAGE2_CHECKPOINT_EPOCH}"
export DUCA_STAGE2_UPDATE_AUDIT_JSON="${RUN_ROOT}/stage2/update_audit.json"

STAGE2_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
STAGE2_WORK="${RUN_ROOT}/stage2/work"
mkdir -p "${RUN_ROOT}/stage2"

export DUCA_RECOVERY_MANIFEST_PATH="${RUN_ROOT}/stage2/recovery_manifest.json"
export DUCA_RECOVERY_SOURCE_JOB="1190528"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch
from mmengine import Config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


stage1 = Path(os.environ["DUCA_STAGE1_CHECKPOINT"]).resolve()
stage2 = Path(os.environ["DUCA_STAGE2_RECOVERY_CHECKPOINT"]).resolve()
checkpoint = torch.load(stage2, map_location="cpu")
if int(checkpoint.get("epoch", -1)) != 9:
    raise RuntimeError("Stage-2 recovery checkpoint epoch drift")
for key in ("state_dict", "state_dict_ema", "optimizer", "scheduler", "grad_scaler"):
    if key not in checkpoint:
        raise RuntimeError(f"Stage-2 recovery checkpoint lacks {key}")
cfg = Config.fromfile("configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py")
if str(cfg.workflow.get("intermediate_validation_role", "")) != "learning_curve_only":
    raise RuntimeError("Stage-2 recovery must keep intermediate validation diagnostic-only")
if cfg.workflow.get("intermediate_validation_selects_checkpoint", None) is not False:
    raise RuntimeError("Stage-2 recovery must forbid intermediate checkpoint selection")
payload = {
    "schema_version": "duca_rate_curriculum_stage2_recovery_v1",
    "task": "offline_temporal_action_detection",
    "source_job": os.environ["DUCA_RECOVERY_SOURCE_JOB"],
    "git_commit": os.environ["DUCA_EXPECTED_COMMIT"],
    "stage1_checkpoint": {"path": str(stage1), "sha256": sha256(stage1), "epoch": 29},
    "stage2_checkpoint": {
        "path": str(stage2),
        "sha256": sha256(stage2),
        "epoch": int(checkpoint["epoch"]),
        "rng_state_present": "rng_state" in checkpoint,
    },
    "recovery_policy": {
        "max_nonfinite_loss_retries": 8,
        "same_batch_replay": True,
        "state_restored_before_replay": True,
        "persistent_nonfinite_fails_closed": True,
        "intermediate_checkpoint_selection": False,
        "intermediate_validation_role": "learning_curve_only",
        "terminal_checkpoint": "epoch_59_state_dict_ema",
    },
}
target = Path(os.environ["DUCA_RECOVERY_MANIFEST_PATH"])
temporary = target.with_name(target.name + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_RATE_STAGE2_RECOVERY] precheck passed: ${RUN_ROOT}"
  exit 0
fi

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate-stage2-recovery-${SLURM_JOB_ID}" \
  tools/train.py "${STAGE2_CONFIG}" --resume "${STAGE2_CHECKPOINT}" --id 0 --seed 3407 --cfg-options \
  "work_dir=${STAGE2_WORK}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/train.out"

[[ ! -e "${STAGE2_WORK}/gpu1_id0/intermediate_validation/best_validation_ema.json" ]] || \
  fail "Stage-2 intermediate mAP selected a checkpoint"
STAGE2_TERMINAL_CHECKPOINT="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_59.pth"
[[ -f "${STAGE2_TERMINAL_CHECKPOINT}" ]] || fail "terminal Stage-2 EMA checkpoint is missing"
[[ -f "${DUCA_STAGE2_UPDATE_AUDIT_JSON}" ]] || fail "Stage-2 update audit is missing"

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate-stage2-recovery-${SLURM_JOB_ID}-eval" \
  tools/test.py "${STAGE2_CONFIG}" --checkpoint "${STAGE2_TERMINAL_CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 \
  --metrics-json "${RUN_ROOT}/stage2/terminal_evaluation.json" --id 0 --seed 3407 \
  --cfg-options "work_dir=${RUN_ROOT}/stage2/eval" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/eval.out"

for epoch_one in $(seq 5 5 60); do
  epoch_zero=$((epoch_one - 1))
  checkpoint="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_${epoch_zero}.pth"
  quality_dir="${RUN_ROOT}/stage2/quality/epoch_${epoch_one}"
  [[ -f "${checkpoint}" ]] || fail "Stage-2 checkpoint is missing: ${checkpoint}"
  mkdir -p "${quality_dir}"
  "${PYTHON}" -m tools.bata.export_duca_selection_quality \
    --config "${STAGE2_CONFIG}" --checkpoint "${checkpoint}" \
    --output-jsonl "${quality_dir}/records.jsonl" \
    --summary-json "${quality_dir}/export.json" --split val --device cuda:0 \
    --use-ema true --seed 3407 \
    2>&1 | tee "${quality_dir}/export.out"
  "${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
    --records-jsonl "${quality_dir}/records.jsonl" --output-dir "${quality_dir}" \
    --bootstrap-samples 200 --random-seed 3407 \
    2>&1 | tee "${quality_dir}/analyze.out"
done

echo "[DUCA_RATE_STAGE2_RECOVERY] completed under ${RUN_ROOT}"
