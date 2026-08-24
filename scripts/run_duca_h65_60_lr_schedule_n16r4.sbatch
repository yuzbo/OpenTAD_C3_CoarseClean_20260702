#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-lr60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_LR60][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "source root lacks tools/train.py: $ROOT"
export DUCA_REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
[[ -n "$EXPECTED_COMMIT" ]] || fail "DUCA_EXPECTED_COMMIT is required"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal checkout must be clean"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fail "MASTER_PORT must be 1024-65535"

ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical read-only input missing: $path"
done

STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
STAGE1_SHA="${DUCA_STAGE1_CHECKPOINT_SHA256:-}"
[[ -r "$STAGE1_CHECKPOINT" && -n "$STAGE1_SHA" ]] || fail "Stage-1 checkpoint and SHA-256 are required"
[[ "${DUCA_STAGE1_CHECKPOINT_EPOCH:-}" == 29 ]] || fail "Stage-1 checkpoint epoch must be 29"
ACTUAL_STAGE1_SHA="$(sha256sum "$STAGE1_CHECKPOINT" | awk '{print $1}')"
[[ "${ACTUAL_STAGE1_SHA,,}" == "${STAGE1_SHA,,}" ]] || fail "Stage-1 checkpoint SHA-256 mismatch"
export DUCA_STAGE1_CHECKPOINT="$(realpath "$STAGE1_CHECKPOINT")"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA,,}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

ARM="${ARM:-}"
case "$ARM" in
  am_rpch25)
    CONFIG="configs/adatad/thumos/duca_h65_60_stage2_am_rpch25.py"
    ;;
  longcosine_h6000)
    CONFIG="configs/adatad/thumos/duca_h65_60_stage2_longcosine_h6000.py"
    ;;
  *) fail "ARM must be am_rpch25 or longcosine_h6000" ;;
esac

precheck() {
  "$PYTHON" -m py_compile opentad/cores/scheduler.py "$CONFIG" tools/train.py
  "$PYTHON" -m pytest tests/test_duca_h65_60_lr_schedules.py -q
}
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  precheck
  exit 0
fi

WORK_DIR="${DUCA_STAGE2_WORK_DIR:-}"
[[ -n "$WORK_DIR" ]] || fail "DUCA_STAGE2_WORK_DIR is required"
WORK_DIR="$(realpath -m "$WORK_DIR")"
UPDATE_AUDIT="${DUCA_STAGE2_UPDATE_AUDIT_JSON:-$WORK_DIR/update_audit.json}"
export DUCA_STAGE2_UPDATE_AUDIT_JSON="$UPDATE_AUDIT"

TRAIN_ARGS=(
  "$CONFIG" --seed 3407
  --cfg-options
  "work_dir=$WORK_DIR"
  "data.train.ann_file=$ANNOTATION_PATH"
  "data.val.ann_file=$ANNOTATION_PATH"
  "data.test.ann_file=$ANNOTATION_PATH"
  "data.train.category_file=$CATEGORY_PATH"
  "data.val.category_file=$CATEGORY_PATH"
  "data.test.category_file=$CATEGORY_PATH"
  "model.backbone.custom.pretrain=$ADATAD_PRETRAIN"
)

if [[ "${PRE_RUN_ONLY:-0}" == 1 && "${PRE_RUN_RESUME_ONLY:-0}" == 1 ]]; then
  fail "PRE_RUN_ONLY and PRE_RUN_RESUME_ONLY are mutually exclusive"
fi
if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  precheck
  [[ ! -e "$WORK_DIR" ]] || fail "PRE_RUN output root already exists: $WORK_DIR"
  TRAIN_ARGS+=(
    "workflow.end_epoch=1"
    "workflow.max_train_iters=${DUCA_PRE_RUN_MAX_ITERS:-2}"
    "workflow.val_eval_interval=-1"
    "workflow.val_loss_interval=-1"
    "workflow.val_start_epoch=9999"
    "workflow.checkpoint_interval=1"
  )
elif [[ "${PRE_RUN_RESUME_ONLY:-0}" == 1 ]]; then
  precheck
  [[ -n "${DUCA_RESUME_CHECKPOINT:-}" ]] || fail "PRE_RUN resume checkpoint is required"
  [[ ! -e "$WORK_DIR" ]] || fail "PRE_RUN resume output root already exists: $WORK_DIR"
  TRAIN_ARGS+=(
    "workflow.end_epoch=2"
    "workflow.max_train_iters=${DUCA_PRE_RUN_MAX_ITERS:-2}"
    "workflow.val_eval_interval=-1"
    "workflow.val_loss_interval=-1"
    "workflow.val_start_epoch=9999"
    "workflow.checkpoint_interval=1"
  )
fi

if [[ -n "${DUCA_RESUME_CHECKPOINT:-}" ]]; then
  [[ -r "$DUCA_RESUME_CHECKPOINT" ]] || fail "resume checkpoint is unreadable"
  TRAIN_ARGS+=(--resume "$DUCA_RESUME_CHECKPOINT")
elif [[ "${PRE_RUN_ONLY:-0}" != 1 && "${PRE_RUN_RESUME_ONLY:-0}" != 1 ]]; then
  [[ ! -e "$WORK_DIR" ]] || fail "formal output root already exists: $WORK_DIR"
fi

exec "$PYTHON" tools/train.py "${TRAIN_ARGS[@]}"
