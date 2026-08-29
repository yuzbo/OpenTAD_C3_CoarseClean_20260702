#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-dyn-tubelet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

set -eo pipefail
source /etc/profile
set -u

fail() { echo "[DUCA_DYNAMIC_NATIVE_TUBELET][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "source root is missing tools/train.py: $ROOT"
cd "$ROOT"
export DUCA_REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

EXPECTED_COMMIT="${DUCA_EXACT_COMMIT:-}"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "DUCA_EXACT_COMMIT is required"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "source revision drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean source tree required"

source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
PYTHON="${PYTHON:-python}"
VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
export DUCA_STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth}"
export DUCA_STAGE1_CHECKPOINT_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:-bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
export THUMOS14_ANNOTATION_PATH="$ANNOTATION_PATH"
export THUMOS14_CLASS_MAP="$CATEGORY_PATH"
export THUMOS14_TRAIN_DATA_PATH="$VIDEO_ROOT"
export THUMOS14_TEST_DATA_PATH="$VIDEO_ROOT"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN" "$DUCA_STAGE1_CHECKPOINT"; do
  [[ -r "$path" ]] || fail "required read-only input is missing: $path"
done

CONFIG="configs/adatad/thumos/duca_dynamic_native_tubelet_budget_official60.py"
TABLE_CONFIG="configs/adatad/thumos/duca_native_tubelet_uniform_reconstruct_fixed384_official60.py"
RUN_ROOT="${DUCA_RUN_ROOT:-}"
[[ -n "$RUN_ROOT" ]] || fail "DUCA_RUN_ROOT is required"
[[ "$RUN_ROOT" == /data/run01/sczc063/yuzibo/* ]] || fail "run root must be under the project remote boundary"
[[ ! -e "$RUN_ROOT" ]] || fail "fresh run root required: $RUN_ROOT"

"$PYTHON" -m pytest \
  tests/test_duca_native_tubelet_coreset.py \
  tests/test_duca_physical_time_reconstruction.py -q
"$PYTHON" -m py_compile \
  tools/bata/build_duca_dynamic_native_tubelet_budget_table.py \
  opentad/models/duca/tubelet_coreset.py \
  opentad/models/backbones/backbone_wrapper.py \
  opentad/models/detectors/actionformer.py
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi

mkdir -p "$RUN_ROOT/tables"
export DUCA_DYNAMIC_TRAIN_BUDGET_TABLE="$RUN_ROOT/tables/train.json"
export DUCA_DYNAMIC_VALIDATION_BUDGET_TABLE="$RUN_ROOT/tables/validation.json"
"$PYTHON" tools/bata/build_duca_dynamic_native_tubelet_budget_table.py \
  --config "$TABLE_CONFIG" --checkpoint "$DUCA_STAGE1_CHECKPOINT" \
  --split train --output "$DUCA_DYNAMIC_TRAIN_BUDGET_TABLE" --device cuda:0
"$PYTHON" tools/bata/build_duca_dynamic_native_tubelet_budget_table.py \
  --config "$TABLE_CONFIG" --checkpoint "$DUCA_STAGE1_CHECKPOINT" \
  --split val --output "$DUCA_DYNAMIC_VALIDATION_BUDGET_TABLE" --device cuda:0

common_cfg=(
  data.train.ann_file="$ANNOTATION_PATH"
  data.val.ann_file="$ANNOTATION_PATH"
  data.test.ann_file="$ANNOTATION_PATH"
  data.train.category_file="$CATEGORY_PATH"
  data.val.category_file="$CATEGORY_PATH"
  data.test.category_file="$CATEGORY_PATH"
  model.backbone.custom.pretrain="$ADATAD_PRETRAIN"
)

if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  "$PYTHON" tools/train.py "$CONFIG" --seed 3407 --cfg-options \
    work_dir="$RUN_ROOT/work" "${common_cfg[@]}" \
    workflow.end_epoch=1 workflow.max_train_iters=2 \
    workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 \
    workflow.val_start_epoch=9999 workflow.checkpoint_interval=1
  PRE_RUN_CHECKPOINT="$RUN_ROOT/work/gpu1_id0/checkpoint/epoch_0.pth"
  [[ -r "$PRE_RUN_CHECKPOINT" ]] || fail "PRE_RUN checkpoint is missing"
  "$PYTHON" tools/train.py "$CONFIG" --seed 3407 --resume "$PRE_RUN_CHECKPOINT" \
    --cfg-options work_dir="$RUN_ROOT/resume" "${common_cfg[@]}" \
    workflow.end_epoch=2 workflow.max_train_iters=2 \
    workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 \
    workflow.val_start_epoch=9999 workflow.checkpoint_interval=1
  RESUMED_CHECKPOINT="$RUN_ROOT/resume/gpu1_id0/checkpoint/epoch_1.pth"
  [[ -r "$RESUMED_CHECKPOINT" ]] || fail "resumed PRE_RUN checkpoint is missing"
  "$PYTHON" -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu"); required={"state_dict", "state_dict_ema", "optimizer", "scheduler", "grad_scaler", "epoch", "successful_optimizer_updates", "rng_state"}; missing=sorted(required.difference(c)); assert not missing, missing; assert int(c["epoch"]) == 1; assert int(c["successful_optimizer_updates"]) == 4' "$RESUMED_CHECKPOINT"
  exit 0
fi

"$PYTHON" tools/train.py "$CONFIG" --seed 3407 --cfg-options \
  work_dir="$RUN_ROOT/work" "${common_cfg[@]}"
CHECKPOINT="$RUN_ROOT/work/gpu1_id0/checkpoint/epoch_59.pth"
[[ -r "$CHECKPOINT" ]] || fail "terminal epoch-59 checkpoint is missing"
"$PYTHON" -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu"); required={"state_dict", "state_dict_ema", "optimizer", "scheduler", "grad_scaler", "epoch", "successful_optimizer_updates", "rng_state"}; missing=sorted(required.difference(c)); assert not missing, missing; assert int(c["epoch"]) == 59; assert int(c["successful_optimizer_updates"]) == 6000' "$CHECKPOINT"
"$PYTHON" tools/test.py "$CONFIG" --checkpoint "$CHECKPOINT" --seed 3407 \
  --expected-checkpoint-epoch 59 --checkpoint-state-key state_dict_ema \
  --metrics-json "$RUN_ROOT/metrics_epoch59_ema.json" --cfg-options \
  work_dir="$RUN_ROOT/eval" "${common_cfg[@]}"
