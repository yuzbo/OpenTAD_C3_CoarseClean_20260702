#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_CYCLE4][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$ROOT"
MODE="${MODE:-PRECHECK}"
if [[ "$MODE" == PRECHECK ]]; then MODE=STAGE1; fi
case "$MODE" in STAGE1|STAGE2_OFF|STAGE2_ON) ;; *) fail "MODE must be STAGE1, STAGE2_OFF, or STAGE2_ON" ;; esac
PRECHECK_TARGET="${PRECHECK_TARGET:-$MODE}"
case "$PRECHECK_TARGET" in STAGE1|STAGE2_OFF|STAGE2_ON) ;; *) fail "PRECHECK_TARGET must be STAGE1, STAGE2_OFF, or STAGE2_ON" ;; esac
PYTHON="${PYTHON:-python}"
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do [[ -r "$path" ]] || fail "canonical read-only path missing: $path"; done

STAGE1_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py"
STAGE2_OFF_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
STAGE2_ON_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
STAGE1_SHA="${DUCA_STAGE1_CHECKPOINT_SHA256:-${DUCA_STAGE1_CHECKPOINT_SHA:-}}"
STAGE1_EPOCH="${DUCA_STAGE1_CHECKPOINT_EPOCH:-29}"

precheck() {
  args=(--target "$PRECHECK_TARGET" --video-root "$VIDEO_ROOT" --annotation "$ANNOTATION_PATH" --category "$CATEGORY_PATH" --pretrain "$ADATAD_PRETRAIN" --chunk-dim 0)
  if [[ "$PRECHECK_TARGET" != STAGE1 ]]; then
    [[ "$STAGE1_EPOCH" == 29 ]] || fail "Stage1 epoch must be 29"
    [[ -n "$STAGE1_CHECKPOINT" && -n "$STAGE1_SHA" ]] || fail "DUCA_STAGE1_CHECKPOINT and SHA are required"
    args+=(--stage1 "$STAGE1_CHECKPOINT" --sha256 "$STAGE1_SHA" --epoch 29)
  fi
  "$PYTHON" tools/bata/validate_duca_h65_first_singleclock_cycle4.py "${args[@]}"
  "$PYTHON" -m pytest tests/test_duca_h65_cycle4_singleclock_contract.py -q
  "$PYTHON" -m py_compile tools/train.py tools/bata/validate_duca_h65_first_singleclock_cycle4.py
}
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then precheck; exit 0; fi

# Bounded repository-owned smoke run.  This is deliberately an explicit
# opt-in path; official STAGE1/STAGE2 commands below retain their semantics.
if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  [[ "$MODE" == STAGE1 || "$MODE" == STAGE2_OFF || "$MODE" == STAGE2_ON ]] || fail "PRE_RUN_ONLY requires a formal stage MODE"
  if [[ "$MODE" == STAGE1 ]]; then
    CFG="$STAGE1_CONFIG"
    PRE_RUN_WORK_DIR="${DUCA_PRE_RUN_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle4_pre_run_stage1}"
  elif [[ "$MODE" == STAGE2_OFF ]]; then
    CFG="$STAGE2_OFF_CONFIG"
    PRE_RUN_WORK_DIR="${DUCA_PRE_RUN_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle4_pre_run_stage2_off}"
  else
    CFG="$STAGE2_ON_CONFIG"
    PRE_RUN_WORK_DIR="${DUCA_PRE_RUN_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle4_pre_run_stage2_on}"
  fi
  exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
    --cfg-options work_dir="$PRE_RUN_WORK_DIR" \
    data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
    data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
    data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
    model.backbone.custom.pretrain="$ADATAD_PRETRAIN" \
    workflow.end_epoch=1 workflow.max_train_iters=1 \
    workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 \
    workflow.val_start_epoch=9999 workflow.checkpoint_interval=1
fi

if [[ "$MODE" == STAGE1 ]]; then
  exec "$PYTHON" tools/train.py "$STAGE1_CONFIG" --seed 3407 \
    --cfg-options work_dir="${DUCA_STAGE1_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle4_stage1_uniform384}" \
    data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
    data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
    data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
    model.backbone.custom.pretrain="$ADATAD_PRETRAIN"
fi

PRECHECK_TARGET="$MODE" precheck
export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export DUCA_STAGE1_CHECKPOINT_SHA256="$STAGE1_SHA"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
if [[ "$MODE" == STAGE2_OFF ]]; then CFG="$STAGE2_OFF_CONFIG"; else CFG="$STAGE2_ON_CONFIG"; fi
exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
  --cfg-options work_dir="${DUCA_STAGE2_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle4_${MODE,,}}" \
  data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
  data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
  data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
  model.backbone.custom.pretrain="$ADATAD_PRETRAIN"
