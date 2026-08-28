#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-sem-budget
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_SEMANTIC_BUDGET][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "source root missing tools/train.py: $ROOT"
export DUCA_REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fail "invalid MASTER_PORT"

ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

export DUCA_WINDOW_TABLE_PATH="${DUCA_WINDOW_TABLE_PATH:-}"
export DUCA_STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
export DUCA_STAGE1_CHECKPOINT_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:-}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
[[ -r "$DUCA_STAGE1_CHECKPOINT" ]] || fail "DUCA_STAGE1_CHECKPOINT is missing"
[[ "$DUCA_STAGE1_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || fail "invalid checkpoint SHA256"

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical read-only path missing: $path"
done

if [[ "${BUILD_TABLE_ONLY:-0}" == 1 ]]; then
  [[ -n "$DUCA_WINDOW_TABLE_PATH" ]] || fail "DUCA_WINDOW_TABLE_PATH output is required"
  [[ ! -e "$DUCA_WINDOW_TABLE_PATH" ]] || fail "refusing to overwrite window table: $DUCA_WINDOW_TABLE_PATH"
  exec "$PYTHON" tools/bata/build_duca_semantic_window_budget_table.py \
    --stage1-config configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py \
    --checkpoint "$DUCA_STAGE1_CHECKPOINT" \
    --checkpoint-sha256 "$DUCA_STAGE1_CHECKPOINT_SHA256" \
    --pretrain "$ADATAD_PRETRAIN" \
    --annotation "$ANNOTATION_PATH" --class-map "$CATEGORY_PATH" \
    --data-path "$VIDEO_ROOT" --output "$DUCA_WINDOW_TABLE_PATH" \
    --seed 3407 --device cuda:0
fi

[[ -r "$DUCA_WINDOW_TABLE_PATH" ]] || fail "DUCA_WINDOW_TABLE_PATH is missing"
export DUCA_WINDOW_BUDGET_ARM="${DUCA_WINDOW_BUDGET_ARM:-}"
case "$DUCA_WINDOW_BUDGET_ARM" in
  fixed384) CFG="configs/adatad/thumos/duca_semantic_budget_fixed384.py" ;;
  semantic) CFG="configs/adatad/thumos/duca_semantic_budget_semantic.py" ;;
  permuted_control) CFG="configs/adatad/thumos/duca_semantic_budget_permuted_control.py" ;;
  *) fail "DUCA_WINDOW_BUDGET_ARM must be fixed384, semantic or permuted_control" ;;
esac

precheck() {
  "$PYTHON" tools/bata/validate_duca_semantic_window_budget.py \
    --config "$CFG" --table "$DUCA_WINDOW_TABLE_PATH" \
    --checkpoint "$DUCA_STAGE1_CHECKPOINT" \
    --checkpoint-sha256 "$DUCA_STAGE1_CHECKPOINT_SHA256"
  "$PYTHON" -m pytest tests/test_duca_semantic_window_budget.py -q
  "$PYTHON" -m py_compile \
    tools/train.py tools/bata/build_duca_semantic_window_budget_table.py \
    tools/bata/validate_duca_semantic_window_budget.py
}

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then precheck; exit 0; fi
precheck

WORK_DIR="${DUCA_WORK_DIR:-exps/thumos/adatad/duca_semantic_budget_matched_${DUCA_WINDOW_BUDGET_ARM}}"
EXTRA=()
if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  WORK_DIR="${DUCA_PRE_RUN_WORK_DIR:-${WORK_DIR}_pre_run}"
  EXTRA=(workflow.end_epoch=1 workflow.max_train_iters=1 workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 workflow.val_start_epoch=9999 workflow.checkpoint_interval=1)
fi

exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
  --cfg-options work_dir="$WORK_DIR" \
  dataset.train.ann_file="$ANNOTATION_PATH" dataset.val.ann_file="$ANNOTATION_PATH" \
  dataset.test.ann_file="$ANNOTATION_PATH" dataset.train.class_map="$CATEGORY_PATH" \
  dataset.val.class_map="$CATEGORY_PATH" dataset.test.class_map="$CATEGORY_PATH" \
  dataset.train.data_path="$VIDEO_ROOT" dataset.val.data_path="$VIDEO_ROOT" \
  dataset.test.data_path="$VIDEO_ROOT" model.backbone.custom.pretrain="$ADATAD_PRETRAIN" \
  "${EXTRA[@]}"
