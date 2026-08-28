#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-tubelet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00

set -eo pipefail
source /etc/profile
set -u

fail() { echo "[DUCA_NATIVE_TUBELET][FAIL] $*" >&2; exit 1; }

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
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN" "$DUCA_STAGE1_CHECKPOINT"; do
  [[ -r "$path" ]] || fail "required read-only input is missing: $path"
done

ARM="${DUCA_ARM:-}"
case "$ARM" in
  UNIFORM)
    CONFIG="configs/adatad/thumos/duca_native_tubelet_uniform_reconstruct_fixed384_official60.py"
    ;;
  CORESET)
    CONFIG="configs/adatad/thumos/duca_native_tubelet_coreset_fixed384_official60.py"
    ;;
  *) fail "DUCA_ARM must be UNIFORM or CORESET" ;;
esac

RUN_ROOT="${DUCA_RUN_ROOT:-}"
[[ -n "$RUN_ROOT" ]] || fail "DUCA_RUN_ROOT is required"
[[ "$RUN_ROOT" == /data/run01/sczc063/yuzibo/* ]] || fail "run root must be under the project remote boundary"
[[ ! -e "$RUN_ROOT" ]] || fail "fresh arm run root required: $RUN_ROOT"
mkdir -p "$RUN_ROOT"

common_cfg=(
  data.train.ann_file="$ANNOTATION_PATH"
  data.val.ann_file="$ANNOTATION_PATH"
  data.test.ann_file="$ANNOTATION_PATH"
  data.train.category_file="$CATEGORY_PATH"
  data.val.category_file="$CATEGORY_PATH"
  data.test.category_file="$CATEGORY_PATH"
  model.backbone.custom.pretrain="$ADATAD_PRETRAIN"
)

"$PYTHON" -m pytest \
  tests/test_duca_native_tubelet_coreset.py \
  tests/test_duca_physical_time_reconstruction.py -q
"$PYTHON" -m py_compile \
  opentad/models/duca/tubelet_coreset.py \
  opentad/models/duca/physical_time_reconstruction.py \
  opentad/models/selectors/duca_online_frame_selector.py

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi

if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then
  exec "$PYTHON" tools/train.py "$CONFIG" --seed 3407 --cfg-options \
    work_dir="$RUN_ROOT/work" "${common_cfg[@]}" \
    workflow.end_epoch=1 workflow.max_train_iters=2 \
    workflow.val_eval_interval=-1 workflow.val_loss_interval=-1 \
    workflow.val_start_epoch=9999 workflow.checkpoint_interval=1
fi

"$PYTHON" tools/train.py "$CONFIG" --seed 3407 --cfg-options \
  work_dir="$RUN_ROOT/work" "${common_cfg[@]}"

CHECKPOINT="$RUN_ROOT/work/gpu1_id0/checkpoint/epoch_59.pth"
[[ -r "$CHECKPOINT" ]] || fail "terminal epoch-59 checkpoint is missing"
"$PYTHON" tools/test.py "$CONFIG" --checkpoint "$CHECKPOINT" --seed 3407 \
  --expected-checkpoint-epoch 59 --checkpoint-state-key state_dict_ema \
  --metrics-json "$RUN_ROOT/metrics_epoch59_ema.json" --cfg-options \
  work_dir="$RUN_ROOT/eval" "${common_cfg[@]}"
