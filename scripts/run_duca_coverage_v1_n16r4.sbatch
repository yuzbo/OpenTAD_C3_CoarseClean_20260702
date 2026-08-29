#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
#SBATCH --qos=normal
#SBATCH --job-name=duca-coverage-v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
set -euo pipefail

fail() { echo "[DUCA_COVERAGE_V1][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" && -f "$ROOT/tools/train.py" ]] || fail "DUCA_REPO_ROOT must name the deployed source"
ROOT="$(cd -- "$ROOT" && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
PYTHON="${PYTHON:-python}"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export DUCA_THUMOS14_ROOT="${DUCA_THUMOS14_ROOT:-/data/run01/sczc063/yuzibo/thumos14}"
export DUCA_STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth}"
export DUCA_STAGE1_CHECKPOINT_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:-bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3}"
export DUCA_STAGE1_CHECKPOINT_EPOCH="${DUCA_STAGE1_CHECKPOINT_EPOCH:-29}"
PRETRAIN="${DUCA_ADATAD_PRETRAIN_PATH:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
CONTROL_CONFIG="configs/adatad/thumos/duca_coverage_v1_matched_h65_control.py"
CANDIDATE_CONFIG="configs/adatad/thumos/duca_coverage_v1_candidate.py"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "one Slurm GPU allocation is required"
[[ -r "$DUCA_STAGE1_CHECKPOINT" && -r "$PRETRAIN" ]] || fail "checkpoint or VideoMAE pretrain is missing"
[[ "$(sha256sum "$DUCA_STAGE1_CHECKPOINT" | awk '{print $1}')" == "$DUCA_STAGE1_CHECKPOINT_SHA256" ]] || fail "Stage-1 checkpoint identity mismatch"
for path in \
  "$DUCA_THUMOS14_ROOT/raw_data/video" \
  "$DUCA_THUMOS14_ROOT/annotations/thumos_14_anno.json" \
  "$DUCA_THUMOS14_ROOT/annotations/category_idx.txt"; do
  [[ -r "$path" ]] || fail "canonical THUMOS14 resource missing: $path"
done
if [[ -n "${DUCA_EXPECTED_COMMIT:-}" ]]; then
  [[ "$(git rev-parse HEAD)" == "$DUCA_EXPECTED_COMMIT" ]] || fail "commit drift"
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
fi

run_static_checks() {
  "$PYTHON" -m py_compile \
    opentad/models/duca/acquisition.py \
    opentad/models/selectors/duca_online_frame_selector.py \
    tools/bata/validate_duca_coverage_v1_replay.py
  "$PYTHON" -m pytest \
    tests/test_duca_temporal_coverage_selector.py \
    tests/test_duca_sampling_rate_official60.py \
    tests/test_duca_two_stage_curriculum.py \
    tests/test_duca_checkpoint_resume_roundtrip.py -q
  "$PYTHON" - "$CONTROL_CONFIG" "$CANDIDATE_CONFIG" "$PRETRAIN" <<'PY'
import sys
from mmengine.config import Config
import opentad.datasets
from opentad.models import build_detector
from tools.bata.duca_frontend_initialization import initialize_model_from_checkpoint
from tools.bata import duca_p0_training

for path in sys.argv[1:3]:
    cfg = Config.fromfile(path)
    cfg.model.backbone.custom.pretrain = sys.argv[3]
    contract = duca_p0_training.formal_training_contract(cfg)
    assert contract["expected_successful_optimizer_updates"] == 6000
    assert contract["checkpoint_interval"] == 5
    assert contract["primary_checkpoint_epoch"] == 59
    model = build_detector(cfg.model)
    receipt = initialize_model_from_checkpoint(model, cfg.workflow.model_initialization)
    assert receipt["checkpoint_state_key"] == "state_dict_ema"
    assert receipt["checkpoint_epoch"] == 29
PY
}

run_smoke() {
  local arm="$1" config="$2" work="$3"
  DUCA_COVERAGE_PRE_RUN=1 "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=1 \
    tools/train.py "$config" --id 0 --seed 3407 --cfg-options \
      "work_dir=$work" \
      "model.backbone.custom.pretrain=$PRETRAIN" \
      "workflow.end_epoch=1" \
      "workflow.max_train_iters=2" \
      "workflow.val_eval_interval=-1" \
      "workflow.val_loss_interval=-1" \
      "workflow.val_start_epoch=9999" \
      "workflow.checkpoint_interval=1" \
    2>&1 | tee "${work%/work}/${arm}_smoke.log"
  [[ -f "$work/gpu1_id0/checkpoint/epoch_0.pth" ]] || fail "$arm smoke checkpoint missing"
}

if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  RUN_ROOT="${DUCA_RUN_ROOT:-}"
  [[ -n "$RUN_ROOT" && ! -e "$RUN_ROOT" ]] || fail "fresh DUCA_RUN_ROOT is required"
  mkdir -p "$RUN_ROOT"
  run_static_checks
  "$PYTHON" tools/bata/validate_duca_coverage_v1_replay.py \
    --control-config "$CONTROL_CONFIG" \
    --candidate-config "$CANDIDATE_CONFIG" \
    --checkpoint "$DUCA_STAGE1_CHECKPOINT" \
    --output-json "$RUN_ROOT/unlabeled_replay_gate.json" \
    --split train --device cuda:0 --seed 3407
  run_smoke control "$CONTROL_CONFIG" "$RUN_ROOT/control_smoke/work"
  run_smoke coverage "$CANDIDATE_CONFIG" "$RUN_ROOT/coverage_smoke/work"
  echo "[DUCA_COVERAGE_V1] PRE_RUN_PASS $RUN_ROOT"
  exit 0
fi

ARM="${DUCA_ARM:-}"
case "$ARM" in
  control) CONFIG="$CONTROL_CONFIG" ;;
  coverage) CONFIG="$CANDIDATE_CONFIG" ;;
  *) fail "DUCA_ARM must be control or coverage" ;;
esac
RUN_ROOT="${DUCA_RUN_ROOT:-}"
[[ -n "$RUN_ROOT" && ! -e "$RUN_ROOT" ]] || fail "fresh DUCA_RUN_ROOT is required"
mkdir -p "$RUN_ROOT"
export DUCA_STAGE2_UPDATE_AUDIT_JSON="$RUN_ROOT/training_update_audit.json"

"$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  tools/train.py "$CONFIG" --id 0 --seed 3407 --cfg-options \
    "work_dir=$RUN_ROOT/work" \
    "model.backbone.custom.pretrain=$PRETRAIN" \
  2>&1 | tee "$RUN_ROOT/train.log"

CHECKPOINT="$RUN_ROOT/work/gpu1_id0/checkpoint/epoch_59.pth"
[[ -f "$CHECKPOINT" ]] || fail "terminal epoch-59 checkpoint missing"
"$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  tools/test.py "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-state-key state_dict_ema \
    --expected-checkpoint-epoch 59 \
    --metrics-json "$RUN_ROOT/terminal_evaluation.json" \
    --id 0 --seed 3407 --cfg-options \
      "work_dir=$RUN_ROOT/terminal_eval" \
      "model.backbone.custom.pretrain=$PRETRAIN" \
      "post_processing.save_dict=True" \
      "inference.load_from_raw_predictions=False" \
  2>&1 | tee "$RUN_ROOT/terminal_eval.log"

"$PYTHON" - "$RUN_ROOT" "$ARM" "$CHECKPOINT" <<'PY'
import json
import os
import sys
from pathlib import Path

root, arm, checkpoint = sys.argv[1:]
evaluation = Path(root, "terminal_evaluation.json")
if not evaluation.is_file():
    raise SystemExit("terminal evaluation is missing")
payload = {
    "schema": "duca_coverage_v1_completion",
    "arm": arm,
    "seed": 3407,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "evaluation": str(evaluation.resolve()),
}
Path(root, "completion.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
echo "[DUCA_COVERAGE_V1] TRAIN_COMPLETE $RUN_ROOT/completion.json"
