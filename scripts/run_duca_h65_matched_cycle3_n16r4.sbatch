#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-c3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_CYCLE3][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$ROOT"
MODE="${MODE:-PRECHECK}"
case "$MODE" in PRECHECK|STAGE1|STAGE2_OFF|STAGE2_ON) ;; *) fail "MODE must be PRECHECK, STAGE1, STAGE2_OFF, or STAGE2_ON" ;; esac
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then MODE=PRECHECK; fi
PYTHON="${PYTHON:-python}"
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"

THUMOS_ROOT="${THUMOS_ROOT:-/data/run01/sczc063/yuzibo/data/thumos14}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$THUMOS_ROOT/annotations/thumos14.json}"
CATEGORY_PATH="${CATEGORY_PATH:-$THUMOS_ROOT/annotations/category.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/checkpoints/adatad_pretrained.pth}"
for path in "$THUMOS_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do [[ -r "$path" ]] || fail "canonical read-only path missing: $path"; done

STAGE1_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py"
STAGE2_OFF_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
STAGE2_ON_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle3.py"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:-}"
STAGE1_SHA="${DUCA_STAGE1_CHECKPOINT_SHA256:-${DUCA_STAGE1_CHECKPOINT_SHA:-}}"
STAGE1_EPOCH="${DUCA_STAGE1_CHECKPOINT_EPOCH:-29}"

precheck() {
  [[ "$STAGE1_EPOCH" == 29 ]] || fail "Stage1 epoch must be 29"
  [[ -n "$STAGE1_CHECKPOINT" && -n "$STAGE1_SHA" ]] || fail "DUCA_STAGE1_CHECKPOINT and SHA are required"
  "$PYTHON" tools/bata/validate_duca_h65_first_singleclock_cycle3.py \
    --stage1 "$STAGE1_CHECKPOINT" --sha256 "$STAGE1_SHA" --epoch 29 --chunk-dim 0
  "$PYTHON" -m pytest tests/test_duca_h65_cycle3_singleclock_contract.py -q
}
if [[ "$MODE" == PRECHECK ]]; then precheck; exit 0; fi

if [[ "$MODE" == STAGE1 ]]; then
  exec "$PYTHON" tools/train.py "$STAGE1_CONFIG" --seed 3407 \
    --cfg-options work_dir="${DUCA_STAGE1_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle3_stage1_uniform384}" \
    data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
    data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
    data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
    model.backbone.pretrained="$ADATAD_PRETRAIN"
fi

precheck
export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export DUCA_STAGE1_CHECKPOINT_SHA256="$STAGE1_SHA"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
if [[ "$MODE" == STAGE2_OFF ]]; then CFG="$STAGE2_OFF_CONFIG"; else CFG="$STAGE2_ON_CONFIG"; fi
exec "$PYTHON" tools/train.py "$CFG" --seed 3407 \
  --cfg-options work_dir="${DUCA_STAGE2_WORK_DIR:-exps/thumos/adatad/duca_h65_cycle3_${MODE,,}}" \
  data.train.ann_file="$ANNOTATION_PATH" data.val.ann_file="$ANNOTATION_PATH" \
  data.test.ann_file="$ANNOTATION_PATH" data.train.category_file="$CATEGORY_PATH" \
  data.val.category_file="$CATEGORY_PATH" data.test.category_file="$CATEGORY_PATH" \
  model.backbone.pretrained="$ADATAD_PRETRAIN"
