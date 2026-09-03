#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=h65pro-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/h65pro-eval-%x-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/h65pro-eval-%x-%j.err
source /etc/profile
set -euo pipefail

fail() { echo "[H65_PRO_EVAL][FAIL] $*" >&2; exit 1; }

CONFIG="${1:-}"
SEED="${2:-}"
EXPERIMENT_ID="${3:-}"
VARIANT="${4:-}"
[[ -n "$CONFIG" && -n "$SEED" && -n "$EXPERIMENT_ID" && -n "$VARIANT" ]] || \
  fail "usage: $0 <config> <seed> <experiment_id> <variant>"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "seed must be numeric: $SEED"

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/test.py" ]] || fail "exact source root missing tools/test.py: $ROOT"
cd "$ROOT"
export DUCA_REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

module load cuda/11.8
module load miniforge3/24.11
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

export DUCA_CELLCF_TRAINING_PROFILE=official60
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  slurm_job_id="${SLURM_JOB_ID:-0}"
  [[ "$slurm_job_id" =~ ^[0-9]+$ ]] || slurm_job_id=0
  MASTER_PORT=$((30000 + (slurm_job_id % 20000)))
fi
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || \
  fail "MASTER_PORT must be 1024-65535"
export MASTER_PORT

export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -r "$CONFIG" ]] || fail "config is not readable: $CONFIG"
for path in "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" "$THUMOS14_TEST_DATA_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical path is not readable: $path"
done

[[ -n "${H65_PRO_EXPECTED_COMMIT:-}" ]] || fail "H65_PRO_EXPECTED_COMMIT is required for H65-Pro evaluation"
EXPECTED_COMMIT="$H65_PRO_EXPECTED_COMMIT"
OBSERVED_COMMIT="$(git rev-parse HEAD)"
[[ "$OBSERVED_COMMIT" == "$EXPECTED_COMMIT" ]] || fail "checkout commit $OBSERVED_COMMIT differs from expected $EXPECTED_COMMIT"
export DUCA_EXPECTED_COMMIT="$EXPECTED_COMMIT"
export DUCA_ADATAD_PRETRAIN_PATH="$(cd "$(dirname "$ADATAD_PRETRAIN")" && pwd -P)/$(basename "$ADATAD_PRETRAIN")"
export DUCA_ADATAD_PRETRAIN_SHA256="$(sha256sum "$ADATAD_PRETRAIN" | awk '{print tolower($1)}')"
export DUCA_SELECTED_OPT_VARIANT="$VARIANT"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  "$PYTHON" tools/bata/validate_h65_pro_fullmatrix.py
  "$PYTHON" -m py_compile tools/test.py tools/bata/duca_selected_axis_training.py
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "$($PYTHON -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || \
    fail "H65-Pro Slurm job requires exactly one scheduler-visible GPU"
else
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] || fail "direct H65-Pro launch requires physical GPU1"
fi
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal evaluation requires a clean exact-commit checkout"

WORK_ROOT="${H65_PRO_WORK_ROOT:-exps/thumos/adatad/h65_pro_fullmatrix_20260902}"
WORK_DIR="$WORK_ROOT/${EXPERIMENT_ID}_seed${SEED}"
RUN_WORK_DIR="$WORK_DIR/gpu1_id0"
CHECKPOINT="${H65_PRO_CHECKPOINT:-$RUN_WORK_DIR/checkpoint/epoch_59.pth}"
METRICS_JSON="${H65_PRO_METRICS_JSON:-$RUN_WORK_DIR/eval/${EXPERIMENT_ID}_seed${SEED}_epoch59_ema_metrics.json}"
[[ -r "$CHECKPOINT" ]] || fail "terminal checkpoint is not readable: $CHECKPOINT"
mkdir -p "$(dirname "$METRICS_JSON")"

exec "$PYTHON" tools/test.py "$CONFIG" --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "$METRICS_JSON" \
  --cfg-options \
  work_dir="$WORK_DIR" \
  model.backbone.custom.pretrain="$DUCA_ADATAD_PRETRAIN_PATH" \
  post_processing.save_dict=True \
  inference.load_from_raw_predictions=False
