#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_RATE25_E24_RECOVERY][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_RATE25_RECOVERY_RUN_ROOT:-}"
SOURCE_ROOT="${DUCA_RATE25_SOURCE_RUN_ROOT:-}"
SOURCE_JOB="${DUCA_RATE25_SOURCE_JOB:-}"
STAGE1_CHECKPOINT="${SOURCE_ROOT}/stage1/work/gpu1_id0/checkpoint/epoch_29.pth"
STAGE2_CHECKPOINT="${SOURCE_ROOT}/stage2/work/gpu1_id0/checkpoint/epoch_24.pth"
SOURCE_AUDIT="${SOURCE_ROOT}/stage2/update_audit.json"
SOURCE_TRAIN_LOG="${SOURCE_ROOT}/stage2/train.out"
STAGE1_SHA256="${DUCA_STAGE1_REUSE_CHECKPOINT_SHA256:-}"
STAGE2_SHA256="${DUCA_STAGE2_RECOVERY_CHECKPOINT_SHA256:-}"
SOURCE_AUDIT_SHA256="${DUCA_STAGE2_SOURCE_AUDIT_SHA256:-}"
SOURCE_TRAIN_SHA256="${DUCA_STAGE2_SOURCE_TRAIN_SHA256:-}"
STAGE2_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint192.py"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh recovery root is required"
[[ "${SOURCE_JOB}" == "1193437" ]] || fail "source job drift"
[[ -d "${SOURCE_ROOT}" ]] || fail "source run root is missing"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"
[[ -f "${STAGE1_CHECKPOINT}" ]] || fail "sealed Stage-1 checkpoint is missing"
[[ -f "${STAGE2_CHECKPOINT}" ]] || fail "sealed Stage-2 epoch-24 checkpoint is missing"
[[ -f "${SOURCE_AUDIT}" ]] || fail "source update audit is missing"
[[ -f "${SOURCE_TRAIN_LOG}" ]] || fail "source training log is missing"
[[ "$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')" == "${STAGE1_SHA256}" ]] || fail "Stage-1 checkpoint hash mismatch"
[[ "$(sha256sum "${STAGE2_CHECKPOINT}" | awk '{print $1}')" == "${STAGE2_SHA256}" ]] || fail "Stage-2 checkpoint hash mismatch"
[[ "$(sha256sum "${SOURCE_AUDIT}" | awk '{print $1}')" == "${SOURCE_AUDIT_SHA256}" ]] || fail "source update audit hash mismatch"
[[ "$(sha256sum "${SOURCE_TRAIN_LOG}" | awk '{print $1}')" == "${SOURCE_TRAIN_SHA256}" ]] || fail "source training-log hash mismatch"
[[ ! -e "${SOURCE_ROOT}/stage2/work/gpu1_id0/intermediate_validation/best_validation_ema.json" ]] || fail "source run selected an intermediate checkpoint"

# The source failure is a Decord EOF retry exhaustion during diagnostic
# evaluation. Keep the model/config/checkpoint unchanged and increase only the
# decoder's bounded EOF retry allowance, as suggested by Decord itself.
export DECORD_EOF_RETRY_MAX=20480
export DUCA_STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA256}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
export DUCA_STAGE2_UPDATE_AUDIT_JSON="${RUN_ROOT}/stage2/update_audit.json"

mkdir -p "${RUN_ROOT}/stage2"
export DUCA_RATE25_RECOVERY_MANIFEST="${RUN_ROOT}/recovery_manifest.json"
export DUCA_STAGE2_RECOVERY_CHECKPOINT="${STAGE2_CHECKPOINT}"
export DUCA_STAGE2_RECOVERY_CHECKPOINT_SHA256="${STAGE2_SHA256}"
export DUCA_STAGE2_SOURCE_AUDIT="${SOURCE_AUDIT}"
export DUCA_STAGE2_SOURCE_AUDIT_SHA256="${SOURCE_AUDIT_SHA256}"
export DUCA_STAGE2_SOURCE_TRAIN_LOG="${SOURCE_TRAIN_LOG}"
export DUCA_STAGE2_SOURCE_TRAIN_SHA256="${SOURCE_TRAIN_SHA256}"
export DUCA_RATE25_SOURCE_ROOT="${SOURCE_ROOT}"
export DUCA_RATE25_SOURCE_JOB="${SOURCE_JOB}"

"${PYTHON}" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from mmengine import Config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


stage1 = Path(os.environ["DUCA_STAGE1_CHECKPOINT"]).resolve()
stage2 = Path(os.environ["DUCA_STAGE2_RECOVERY_CHECKPOINT"]).resolve()
source_audit_path = Path(os.environ["DUCA_STAGE2_SOURCE_AUDIT"]).resolve()
checkpoint = torch.load(stage2, map_location="cpu")
required = {
    "epoch",
    "state_dict",
    "state_dict_ema",
    "optimizer",
    "scheduler",
    "grad_scaler",
}
require(required.issubset(checkpoint), "Stage-2 checkpoint lacks resumable state")
require(int(checkpoint["epoch"]) == 24, "recovery checkpoint must be epoch 24")
require(int(checkpoint["scheduler"]["last_epoch"]) == 2500, "scheduler step drift")
require(
    math.isfinite(float(checkpoint["grad_scaler"]["scale"])),
    "GradScaler state is non-finite",
)

selector_keys = [
    key
    for key in checkpoint["state_dict"]
    if key.endswith("frame_selector._loss_weight_schedule_step")
]
require(len(selector_keys) == 1, "selector schedule state is ambiguous")
require(
    int(checkpoint["state_dict"][selector_keys[0]].item()) == 2500,
    "selector schedule step drift",
)

source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
counters = source_audit["update_audit"]
for key in (
    "attempted_batches",
    "successful_optimizer_updates",
    "scheduler_updates",
    "ema_updates",
    "duca_schedule_updates",
):
    require(int(counters[key]) == 2500, f"source audit {key} drift")
require(int(counters["amp_skipped_attempts"]) == 6, "source AMP audit drift")
require(int(counters["max_amp_retries_observed"]) == 2, "source AMP retry bound drift")
require(int(counters["nonfinite_loss_attempts"]) == 0, "source has non-finite loss")
require(int(counters["replay_exhaustions"]) == 0, "source exhausted AMP replay")
require(
    int(counters["nonfinite_loss_replay_exhaustions"]) == 0,
    "source exhausted non-finite replay",
)

cfg = Config.fromfile(
    "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint192.py"
)
selector = cfg.model.frame_selector
schedule = selector.loss_weight_schedule
require(int(selector.budget) == 192, "K192 budget drift")
require(int(selector.dense_window_size) == 768, "dense-window drift")
require(selector.sampling_rate_utility_components == "both", "cls/reg utility drift")
require(selector.detector_contribution_components == "both", "cls/reg target drift")
require(selector.detector_gradient_mode == "density_transport_st", "detector bridge drift")
require(float(schedule.detector_contribution.end) == 1.0, "contribution schedule drift")
require(float(schedule.detector_gradient.end) == 0.25, "detector-gradient schedule drift")
require(float(schedule.asformer_adapt.end) == 1.0, "ASFormer adaptation drift")
require(cfg.workflow.intermediate_validation_selects_checkpoint is False, "checkpoint-selection drift")
require(int(os.environ["DECORD_EOF_RETRY_MAX"]) == 20480, "EOF retry fix drift")

payload = {
    "schema_version": "duca_rate25_epoch24_recovery_v1",
    "task": "offline_temporal_action_detection",
    "git_commit": os.environ["DUCA_EXPECTED_COMMIT"],
    "source_job": os.environ["DUCA_RATE25_SOURCE_JOB"],
    "source_run_root": os.environ["DUCA_RATE25_SOURCE_ROOT"],
    "failure": {
        "phase": "one_based_epoch_25_intermediate_ema_evaluation",
        "processed_batches": 147,
        "total_batches": 396,
        "type": "decord_eof_retry_exhaustion",
        "training_updates_after_checkpoint": 0,
        "train_log_path": os.environ["DUCA_STAGE2_SOURCE_TRAIN_LOG"],
        "train_log_sha256": os.environ["DUCA_STAGE2_SOURCE_TRAIN_SHA256"],
    },
    "decoder_fix": {
        "variable": "DECORD_EOF_RETRY_MAX",
        "old_value": 10240,
        "new_value": 20480,
    },
    "stage1_checkpoint": {
        "path": str(stage1),
        "sha256": sha256(stage1),
        "epoch": 29,
        "state_key": "state_dict_ema",
    },
    "stage2_checkpoint": {
        "path": str(stage2),
        "sha256": sha256(stage2),
        "epoch": 24,
        "state_keys": sorted(required),
        "scheduler_last_epoch": int(checkpoint["scheduler"]["last_epoch"]),
        "selector_schedule_step": int(
            checkpoint["state_dict"][selector_keys[0]].item()
        ),
        "grad_scaler_scale": float(checkpoint["grad_scaler"]["scale"]),
        "global_rng_state_present": "rng_state" in checkpoint,
    },
    "source_update_audit": {
        "path": str(source_audit_path),
        "sha256": sha256(source_audit_path),
        "successful_optimizer_updates": 2500,
        "amp_skipped_attempts": 6,
        "max_amp_retries_observed": 2,
    },
    "recovery_policy": {
        "evaluate_missing_epoch_25_before_new_updates": True,
        "resume_epoch": 25,
        "terminal_epoch": 59,
        "continuation_updates": 3500,
        "combined_updates": 6000,
        "deterministic_resume_seed": 3407,
        "checkpoint_global_rng_limitation_recorded": "rng_state" not in checkpoint,
        "intermediate_checkpoint_selection": False,
        "strict_state_dict_loading": True,
    },
}
target = Path(os.environ["DUCA_RATE25_RECOVERY_MANIFEST"])
temporary = target.with_name(target.name + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_RATE25_E24_RECOVERY] precheck passed: ${RUN_ROOT}"
  exit 0
fi

STAGE2_WORK="${RUN_ROOT}/stage2/work"
mkdir -p "${RUN_ROOT}/stage2/eval_epoch25"

# Recover the missing non-selecting epoch-25 diagnostic before any new update.
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-epoch25-eval" \
  tools/test.py "${STAGE2_CONFIG}" --checkpoint "${STAGE2_CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 24 \
  --metrics-json "${RUN_ROOT}/stage2/epoch_025_ema_evaluation.json" \
  --id 0 --seed 3407 --cfg-options \
  "work_dir=${RUN_ROOT}/stage2/eval_epoch25" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  "post_processing.save_dict=True" \
  2>&1 | tee "${RUN_ROOT}/stage2/eval_epoch25.out"

"${PYTHON}" - <<'PY'
import json
import os

receipt = json.load(
    open(
        os.path.join(
            os.environ["DUCA_RATE25_RECOVERY_RUN_ROOT"],
            "stage2",
            "epoch_025_ema_evaluation.json",
        ),
        encoding="utf-8",
    )
)
if int(receipt["checkpoint_epoch"]) != 24:
    raise RuntimeError("recovered epoch-25 diagnostic checkpoint drift")
if receipt["checkpoint_state_key"] != "state_dict_ema":
    raise RuntimeError("recovered epoch-25 diagnostic did not use EMA")
if receipt["checkpoint_sha256"] != os.environ["DUCA_STAGE2_RECOVERY_CHECKPOINT_SHA256"]:
    raise RuntimeError("recovered epoch-25 diagnostic checkpoint hash drift")
PY

# Resume model, optimizer, scheduler, EMA, and GradScaler from epoch 24.
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-stage2-resume" \
  tools/train.py "${STAGE2_CONFIG}" --resume "${STAGE2_CHECKPOINT}" \
  --id 0 --seed 3407 --cfg-options \
  "work_dir=${STAGE2_WORK}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/train.out"

[[ ! -e "${STAGE2_WORK}/gpu1_id0/intermediate_validation/best_validation_ema.json" ]] || \
  fail "Stage-2 intermediate mAP selected a checkpoint"
STAGE2_TERMINAL_CHECKPOINT="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_59.pth"
[[ -f "${STAGE2_TERMINAL_CHECKPOINT}" ]] || fail "terminal Stage-2 checkpoint is missing"
[[ -f "${DUCA_STAGE2_UPDATE_AUDIT_JSON}" ]] || fail "continuation update audit is missing"

export DUCA_STAGE2_TERMINAL_CHECKPOINT="${STAGE2_TERMINAL_CHECKPOINT}"
export DUCA_STAGE2_COMBINED_AUDIT="${RUN_ROOT}/stage2/combined_update_audit.json"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_path = Path(os.environ["DUCA_STAGE2_SOURCE_AUDIT"])
continuation_path = Path(os.environ["DUCA_STAGE2_UPDATE_AUDIT_JSON"])
source = json.loads(source_path.read_text(encoding="utf-8"))["update_audit"]
continuation = json.loads(continuation_path.read_text(encoding="utf-8"))[
    "update_audit"
]
for key in (
    "attempted_batches",
    "successful_optimizer_updates",
    "scheduler_updates",
    "ema_updates",
    "duca_schedule_updates",
):
    if int(continuation[key]) != 3500:
        raise RuntimeError(f"continuation {key} drift")
if int(continuation["optimizer_attempts"]) != (
    int(continuation["successful_optimizer_updates"])
    + int(continuation["amp_skipped_attempts"])
):
    raise RuntimeError("continuation optimizer-attempt audit drift")
if int(continuation["nonfinite_loss_attempts"]) != 0:
    raise RuntimeError("continuation has non-finite loss")
if int(continuation["replay_exhaustions"]) != 0:
    raise RuntimeError("continuation exhausted AMP replay")
if int(continuation["nonfinite_loss_replay_exhaustions"]) != 0:
    raise RuntimeError("continuation exhausted non-finite replay")

summed_keys = set(source).intersection(continuation)
combined = {
    key: int(source[key]) + int(continuation[key])
    for key in sorted(summed_keys)
    if key != "max_amp_retries_observed"
    and key != "max_nonfinite_loss_retries_observed"
}
combined["max_amp_retries_observed"] = max(
    int(source["max_amp_retries_observed"]),
    int(continuation["max_amp_retries_observed"]),
)
combined["max_nonfinite_loss_retries_observed"] = max(
    int(source["max_nonfinite_loss_retries_observed"]),
    int(continuation["max_nonfinite_loss_retries_observed"]),
)
if int(combined["successful_optimizer_updates"]) != 6000:
    raise RuntimeError("combined successful-update count drift")

payload = {
    "schema_version": "duca_rate25_combined_update_audit_v1",
    "task": "offline_temporal_action_detection",
    "source_job": os.environ["DUCA_RATE25_SOURCE_JOB"],
    "recovery_job": os.environ["SLURM_JOB_ID"],
    "source_audit": {
        "path": str(source_path.resolve()),
        "sha256": sha256(source_path),
        "counters": source,
    },
    "continuation_audit": {
        "path": str(continuation_path.resolve()),
        "sha256": sha256(continuation_path),
        "counters": continuation,
    },
    "combined_counters": combined,
}
target = Path(os.environ["DUCA_STAGE2_COMBINED_AUDIT"])
temporary = target.with_name(target.name + ".tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

# Seal terminal epoch-59 EMA mAP with a prediction-backed receipt.
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-terminal-eval" \
  tools/test.py "${STAGE2_CONFIG}" \
  --checkpoint "${STAGE2_TERMINAL_CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 \
  --metrics-json "${RUN_ROOT}/stage2/terminal_evaluation.json" \
  --id 0 --seed 3407 --cfg-options \
  "work_dir=${RUN_ROOT}/stage2/eval_terminal" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  "post_processing.save_dict=True" \
  2>&1 | tee "${RUN_ROOT}/stage2/eval_terminal.out"

for epoch_one in $(seq 5 5 60); do
  epoch_zero=$((epoch_one - 1))
  if (( epoch_one <= 25 )); then
    checkpoint="${SOURCE_ROOT}/stage2/work/gpu1_id0/checkpoint/epoch_${epoch_zero}.pth"
  else
    checkpoint="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_${epoch_zero}.pth"
  fi
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
    --records-jsonl "${quality_dir}/records.jsonl" \
    --output-dir "${quality_dir}" --bootstrap-samples 200 \
    --random-seed 3407 \
    2>&1 | tee "${quality_dir}/analyze.out"
done

echo "[DUCA_RATE25_E24_RECOVERY] completed under ${RUN_ROOT}"
