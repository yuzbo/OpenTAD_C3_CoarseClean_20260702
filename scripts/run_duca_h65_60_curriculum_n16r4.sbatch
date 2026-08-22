#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_H65_60][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "exact source root missing tools/train.py: $ROOT"
export DUCA_REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fail "MASTER_PORT must be 1024-65535"
export MASTER_PORT

MODE="${MODE:-STAGE1_20}"
case "$MODE" in STAGE1_20|STAGE2_40) ;; *) fail "MODE must be STAGE1_20 or STAGE2_40" ;; esac
PYTHON="${PYTHON:-python}"
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do [[ -r "$path" ]] || fail "canonical read-only path missing: $path"; done

STAGE1_CONFIG="configs/adatad/thumos/duca_h65_60_stage1_uniform20.py"
STAGE2_CONFIG="configs/adatad/thumos/duca_h65_60_stage2_transition20_joint20.py"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
STAGE1_SHA="${DUCA_STAGE1_CHECKPOINT_SHA256:-${DUCA_STAGE1_CHECKPOINT_SHA:-}}"

precheck() {
  args=(--target "$MODE" --video-root "$VIDEO_ROOT" --annotation "$ANNOTATION_PATH" --category "$CATEGORY_PATH" --pretrain "$ADATAD_PRETRAIN")
  if [[ "$MODE" == STAGE2_40 ]]; then
    [[ -n "$STAGE1_CHECKPOINT" && -n "$STAGE1_SHA" ]] || fail "DUCA_STAGE1_CHECKPOINT and SHA are required"
    args+=(--stage1 "$STAGE1_CHECKPOINT" --sha256 "$STAGE1_SHA")
  fi
  "$PYTHON" tools/bata/validate_duca_h65_60_curriculum.py "${args[@]}"
  "$PYTHON" -m pytest tests/test_duca_h65_60_curriculum_contract.py -q
  "$PYTHON" -m py_compile tools/train.py tools/bata/validate_duca_h65_60_curriculum.py
}

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then precheck; exit 0; fi

precheck

if [[ "$MODE" == STAGE1_20 ]]; then
  CFG="$STAGE1_CONFIG"
  WORK_DIR="${DUCA_STAGE1_WORK_DIR:-exps/thumos/adatad/duca_h65_60_stage1_uniform20}"
else
  STAGE1_CHECKPOINT="$(realpath "$STAGE1_CHECKPOINT")"
  export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
  export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA,,}"
  export DUCA_STAGE1_CHECKPOINT_EPOCH=19
  CFG="$STAGE2_CONFIG"
  WORK_DIR="${DUCA_STAGE2_WORK_DIR:-exps/thumos/adatad/duca_h65_60_stage2_transition20_joint20}"
fi

if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  WORK_DIR="${DUCA_PRE_RUN_WORK_DIR:-${WORK_DIR}_pre_run}"
  PRE_RUN_MAX_ITERS="${DUCA_PRE_RUN_MAX_ITERS:-2}"
  [[ "$PRE_RUN_MAX_ITERS" =~ ^[0-9]+$ && "$PRE_RUN_MAX_ITERS" -ge 2 ]] || fail "DUCA_PRE_RUN_MAX_ITERS must be at least 2"
  EXTRA=(workflow.end_epoch=1 workflow.max_train_iters="$PRE_RUN_MAX_ITERS" workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 workflow.val_start_epoch=9999 workflow.checkpoint_interval=1)
else
  EXTRA=()
fi

exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
  --cfg-options work_dir="$WORK_DIR" \
  data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
  data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
  data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
  model.backbone.custom.pretrain="$ADATAD_PRETRAIN" "${EXTRA[@]}"
