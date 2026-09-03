#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=h65pro-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/h65pro-train-%x-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/h65pro-train-%x-%j.err
source /etc/profile
set -euo pipefail

fail() { echo "[H65_PRO_TRAIN][FAIL] $*" >&2; exit 1; }

CONFIG="${1:-}"
SEED="${2:-}"
EXPERIMENT_ID="${3:-}"
VARIANT="${4:-}"
[[ -n "$CONFIG" && -n "$SEED" && -n "$EXPERIMENT_ID" && -n "$VARIANT" ]] || \
  fail "usage: $0 <config> <seed> <experiment_id> <variant>"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "seed must be numeric: $SEED"

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "exact source root missing tools/train.py: $ROOT"
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
MOBILENETV3_WEIGHTS="${DUCA_MOBILENETV3_WEIGHTS_PATH:-$YUZIBO_ROOT/pretrained/mobilenet_v3_small-047dcff4.pth}"
[[ -r "$CONFIG" ]] || fail "config is not readable: $CONFIG"
for path in "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" "$THUMOS14_TRAIN_DATA_PATH" "$THUMOS14_TEST_DATA_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical path is not readable: $path"
done
if [[ "$VARIANT" == "h65_pro_ref_mnv3fc384" ]]; then
  [[ -r "$MOBILENETV3_WEIGHTS" ]] || fail "MobileNetV3 reference weights are not readable: $MOBILENETV3_WEIGHTS"
fi

[[ -n "${H65_PRO_EXPECTED_COMMIT:-}" ]] || fail "H65_PRO_EXPECTED_COMMIT is required for H65-Pro training"
EXPECTED_COMMIT="$H65_PRO_EXPECTED_COMMIT"
OBSERVED_COMMIT="$(git rev-parse HEAD)"
[[ "$OBSERVED_COMMIT" == "$EXPECTED_COMMIT" ]] || fail "checkout commit $OBSERVED_COMMIT differs from expected $EXPECTED_COMMIT"
export DUCA_EXPECTED_COMMIT="$EXPECTED_COMMIT"
export DUCA_ADATAD_PRETRAIN_PATH="$(cd "$(dirname "$ADATAD_PRETRAIN")" && pwd -P)/$(basename "$ADATAD_PRETRAIN")"
export DUCA_ADATAD_PRETRAIN_SHA256="$(sha256sum "$ADATAD_PRETRAIN" | awk '{print tolower($1)}')"
export DUCA_MOBILENETV3_WEIGHTS_PATH="$MOBILENETV3_WEIGHTS"
export DUCA_SELECTED_OPT_VARIANT="$VARIANT"

precheck() {
  "$PYTHON" tools/bata/validate_h65_pro_fullmatrix.py
  "$PYTHON" -m py_compile tools/train.py tools/test.py \
    tools/bata/duca_selected_axis_training.py \
    tools/bata/validate_h65_pro_fullmatrix.py \
    tools/bata/generate_h65_pro_fullmatrix.py
  "$PYTHON" - "$CONFIG" "$EXPERIMENT_ID" "$SEED" "$VARIANT" <<'PY'
import sys
from pathlib import Path
from mmengine.config import Config

root = Path.cwd()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from tools.bata.duca_selected_axis_training import H65_PRO_SEEDS, H65_PRO_VARIANT_CONFIGS

config_path = Path(sys.argv[1])
experiment_id = sys.argv[2]
seed = int(sys.argv[3])
variant = sys.argv[4]
cfg = Config.fromfile(str(config_path))
config_experiment_id = cfg.get("h65_pro_experiment_id")
canonical_prefix = experiment_id.split("-S", 1)[0]
if config_experiment_id != experiment_id:
    if (
        not experiment_id.startswith("C")
        or "-S" not in experiment_id
        or config_experiment_id != canonical_prefix
        or not experiment_id.endswith(f"-S{seed}")
    ):
        raise SystemExit(
            f"config experiment id drift: {config_experiment_id} != {experiment_id}"
        )
if int(cfg.workflow.end_epoch) != 60 or int(cfg.get("max_updates", 6000)) != 6000:
    raise SystemExit("strict60/6000 contract drift")
expected_name = H65_PRO_VARIANT_CONFIGS.get(variant)
if expected_name != config_path.name:
    raise SystemExit(f"H65-Pro variant/config mismatch: {variant} -> {expected_name}, got {config_path.name}")
if seed not in H65_PRO_SEEDS:
    raise SystemExit(f"H65-Pro seed is not pre-registered: {seed}")
if variant == "h65_pro_ref_d768" and (
    experiment_id != "REF-D768" or cfg.model.get("frame_selector", None) is not None
):
    raise SystemExit("h65_pro_ref_d768 is reserved for REF-D768 dense reference")
PY
}

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  precheck
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "$($PYTHON -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || \
    fail "H65-Pro Slurm job requires exactly one scheduler-visible GPU"
else
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] || fail "direct H65-Pro launch requires physical GPU1"
fi
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal training requires a clean exact-commit checkout"
precheck

WORK_ROOT="${H65_PRO_WORK_ROOT:-$YUZIBO_ROOT/experiments/h65_pro_fullmatrix_${EXPECTED_COMMIT}}"
[[ "$WORK_ROOT" = /* ]] || fail "H65_PRO_WORK_ROOT must be an absolute path"
case "$WORK_ROOT/" in
  "$ROOT/"*) fail "H65_PRO_WORK_ROOT must stay outside the clean source checkout" ;;
esac
WORK_DIR="$WORK_ROOT/${EXPERIMENT_ID}_seed${SEED}"
exec "$PYTHON" tools/train.py "$CONFIG" --seed "$SEED" --cfg-options \
  work_dir="$WORK_DIR" \
  model.backbone.custom.pretrain="$DUCA_ADATAD_PRETRAIN_PATH"
