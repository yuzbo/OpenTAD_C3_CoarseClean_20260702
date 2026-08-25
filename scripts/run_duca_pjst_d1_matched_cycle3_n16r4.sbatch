#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
#SBATCH --qos=normal
#SBATCH --job-name=duca-pjst-d1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_PJST_D1][FAIL] $*" >&2; exit 1; }
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
MODE="${MODE:-STAGE2_OFF}"
case "$MODE" in STAGE2_OFF|STAGE2_ON) ;; *) fail "MODE must be STAGE2_OFF or STAGE2_ON" ;; esac
PYTHON="${PYTHON:-python}"
BASE="/data/run01/sczc063/yuzibo"
module load cuda/11.8
module load miniforge3/24.11
ENV_ROOT="$BASE/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"

VIDEO_ROOT="${VIDEO_ROOT:-$BASE/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$BASE/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-$BASE/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do [[ -r "$path" ]] || fail "canonical read-only path missing: $path"; done

STAGE2_OFF_CONFIG="configs/adatad/thumos/duca_pjst_d1_stage2_off.py"
STAGE2_ON_CONFIG="configs/adatad/thumos/duca_pjst_d1_stage2_on.py"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
STAGE1_SHA="${DUCA_STAGE1_CHECKPOINT_SHA256:-${DUCA_STAGE1_CHECKPOINT_SHA:-}}"
STAGE1_EPOCH="${DUCA_STAGE1_CHECKPOINT_EPOCH:-29}"

precheck() {
  [[ "$STAGE1_EPOCH" == 29 ]] || fail "Stage1 epoch must be exactly 29"
  [[ -n "$STAGE1_CHECKPOINT" ]] || fail "DUCA_STAGE1_CHECKPOINT is required"
  [[ -f "$STAGE1_CHECKPOINT" ]] || fail "Stage1 checkpoint is not a readable regular file: $STAGE1_CHECKPOINT"
  [[ -r "$STAGE1_CHECKPOINT" ]] || fail "Stage1 checkpoint is not readable: $STAGE1_CHECKPOINT"
  [[ -n "$STAGE1_SHA" ]] || fail "DUCA_STAGE1_CHECKPOINT_SHA256 is required"
  [[ "$STAGE1_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || fail "Stage1 SHA must be exactly 64 hex chars ([0-9a-fA-F]{64})"
  "$PYTHON" tools/bata/validate_duca_pjst_d1_derivative_only.py \
    --stage1 "$STAGE1_CHECKPOINT" --sha256 "$STAGE1_SHA" --epoch 29
  "$PYTHON" -m pytest tests/test_duca_pjst_d1_derivative_only.py -q
  "$PYTHON" -m py_compile tools/train.py tools/bata/validate_duca_pjst_d1_derivative_only.py
}
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then precheck; exit 0; fi

precheck
export DUCA_STAGE1_CHECKPOINT="$(realpath "$STAGE1_CHECKPOINT")"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA,,}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
if [[ "$MODE" == STAGE2_OFF ]]; then CFG="$STAGE2_OFF_CONFIG"; else CFG="$STAGE2_ON_CONFIG"; fi
exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
  --cfg-options work_dir="${DUCA_STAGE2_WORK_DIR:-exps/thumos/adatad/duca_pjst_d1_${MODE,,}}" \
  data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
  data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
  data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
  model.backbone.custom.pretrain="$ADATAD_PRETRAIN"
