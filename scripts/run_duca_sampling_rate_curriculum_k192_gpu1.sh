#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_RATE25_CURRICULUM][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_RATE25_CURRICULUM_RUN_ROOT:-}"
STAGE1_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform192.py"
STAGE2_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint192.py"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh run root is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"

export DUCA_RATE25_CURRICULUM_RUN_ROOT="${RUN_ROOT}"
export DUCA_STAGE2_UPDATE_AUDIT_JSON="${RUN_ROOT}/stage2/update_audit.json"
mkdir -p "${RUN_ROOT}"

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from mmengine import Config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


stage1 = Config.fromfile(
    "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform192.py"
)
stage2 = Config.fromfile(
    "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint192.py"
)
for name, cfg in (("stage1", stage1), ("stage2", stage2)):
    selector = cfg.model.frame_selector
    require(int(selector.dense_window_size) == 768, f"{name}: dense T drift")
    require(int(selector.budget) == 192, f"{name}: budget must be K192")
    require(int(cfg.model.backbone.backbone.total_frames) == 192, f"{name}: VideoMAE frame drift")
    require(int(cfg.model.projection.max_seq_len) == 192, f"{name}: projection length drift")
    require(selector.acquisition_policy == "budget_calibrated_sampling_rate", f"{name}: policy drift")
    require(selector.detector_gradient_mode == "density_transport_st", f"{name}: detector bridge drift")
    require(cfg.workflow.intermediate_validation_selects_checkpoint is False, f"{name}: intermediate checkpoint selection forbidden")

schedule1 = stage1.model.frame_selector.loss_weight_schedule
require(float(schedule1.policy_alpha.end) == 0.0, "stage1 must remain exact uniform")
require(float(schedule1.detector_gradient.end) == 0.0, "stage1 detector gradient must be off")
require(float(schedule1.detector_contribution.end) == 0.0, "stage1 contribution must be off")

selector2 = stage2.model.frame_selector
schedule2 = selector2.loss_weight_schedule
require(selector2.sampling_rate_utility_components == "both", "stage2 cls/reg contribution inputs required")
require(selector2.detector_contribution_components == "both", "stage2 cls/reg contribution targets required")
require(float(selector2.detector_contribution_distillation_weight) == 1.0, "stage2 contribution weight drift")
require(float(schedule2.policy_alpha.end) == 1.0, "stage2 learned rate must be enabled")
require(float(schedule2.detector_gradient.end) == 0.25, "stage2 detector gradient endpoint drift")
require(int(schedule2.detector_gradient.warmup_steps) == 1000, "stage2 detector gradient warmup drift")
require(float(schedule2.detector_contribution.end) == 1.0, "stage2 contribution endpoint drift")
require(int(schedule2.detector_contribution.warmup_steps) == 1000, "stage2 contribution warmup drift")
require(float(schedule2.asformer_adapt.end) == 1.0, "stage2 full ASFormer adaptation required")
require(stage2.workflow.intermediate_validation_role == "learning_curve_only", "stage2 validation must be diagnostic")
require(int(stage2.workflow.primary_checkpoint_epoch) == 59, "stage2 terminal epoch drift")
require(stage2.workflow.primary_checkpoint_state_key == "state_dict_ema", "stage2 terminal state drift")

payload = {
    "schema_version": "duca_rate25_curriculum_v1",
    "task": "offline_temporal_action_detection",
    "git_commit": os.environ["DUCA_EXPECTED_COMMIT"],
    "comparison_anchor": {
        "job": "1191957",
        "commit": "42dba3f90b37243e7965d18b6707e88e81bf7109",
        "budget": 384,
    },
    "single_changed_model_variable": {
        "name": "sampling_budget",
        "dense_window": 768,
        "anchor_value": 384,
        "candidate_value": 192,
        "candidate_fraction": 0.25,
    },
    "fixed_contract": {
        "seed": 3407,
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "contribution_components": "cls_reg",
        "contribution_distillation": True,
        "detector_gradient": "density_transport_st",
        "full_asformer_adaptation": True,
        "intermediate_checkpoint_selection": False,
        "terminal_checkpoint": "epoch_59_state_dict_ema",
    },
}
target = Path(os.environ["DUCA_RATE25_CURRICULUM_RUN_ROOT"]) / "manifest.json"
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_RATE25_CURRICULUM] precheck passed: ${RUN_ROOT}"
  exit 0
fi

STAGE1_WORK="${RUN_ROOT}/stage1/work"
STAGE2_WORK="${RUN_ROOT}/stage2/work"
mkdir -p "${RUN_ROOT}/stage1/quality" "${RUN_ROOT}/stage2"

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-stage1" \
  tools/train.py "${STAGE1_CONFIG}" --id 0 --seed 3407 --cfg-options \
  "work_dir=${STAGE1_WORK}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage1/train.out"

for epoch_one in 5 10 15 20 25 30; do
  epoch_zero=$((epoch_one - 1))
  checkpoint="${STAGE1_WORK}/gpu1_id0/checkpoint/epoch_${epoch_zero}.pth"
  quality_dir="${RUN_ROOT}/stage1/quality/epoch_${epoch_one}"
  [[ -f "${checkpoint}" ]] || fail "stage1 checkpoint is missing: ${checkpoint}"
  mkdir -p "${quality_dir}"
  "${PYTHON}" -m tools.bata.export_duca_selection_quality \
    --config "${STAGE1_CONFIG}" --checkpoint "${checkpoint}" \
    --output-jsonl "${quality_dir}/records.jsonl" \
    --summary-json "${quality_dir}/export.json" --split val --device cuda:0 \
    --use-ema true --seed 3407 \
    2>&1 | tee "${quality_dir}/export.out"
  "${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
    --records-jsonl "${quality_dir}/records.jsonl" --output-dir "${quality_dir}" \
    --bootstrap-samples 200 --random-seed 3407 \
    2>&1 | tee "${quality_dir}/analyze.out"
done

STAGE1_CHECKPOINT="${STAGE1_WORK}/gpu1_id0/checkpoint/epoch_29.pth"
[[ -f "${STAGE1_CHECKPOINT}" ]] || fail "terminal stage1 EMA checkpoint is missing"
export DUCA_STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}"
export DUCA_STAGE1_CHECKPOINT_SHA256="$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-stage2" \
  tools/train.py "${STAGE2_CONFIG}" --id 0 --seed 3407 --cfg-options \
  "work_dir=${STAGE2_WORK}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/train.out"

[[ ! -e "${STAGE2_WORK}/gpu1_id0/intermediate_validation/best_validation_ema.json" ]] || \
  fail "Stage-2 intermediate mAP selected a checkpoint"
STAGE2_CHECKPOINT="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_59.pth"
[[ -f "${STAGE2_CHECKPOINT}" ]] || fail "terminal stage2 EMA checkpoint is missing"
[[ -f "${DUCA_STAGE2_UPDATE_AUDIT_JSON}" ]] || fail "Stage-2 update audit is missing"

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate25-${SLURM_JOB_ID}-stage2-eval" \
  tools/test.py "${STAGE2_CONFIG}" --checkpoint "${STAGE2_CHECKPOINT}" \
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

echo "[DUCA_RATE25_CURRICULUM] completed under ${RUN_ROOT}"
