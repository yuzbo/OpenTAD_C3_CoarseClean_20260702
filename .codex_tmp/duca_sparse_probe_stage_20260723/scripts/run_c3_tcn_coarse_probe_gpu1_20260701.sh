#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
OUT_DIR="${OUT_DIR:-/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/outputs/c3_tcn_coarse_probe_gpu1_20260701}"
TCN_VARIANTS="${TCN_VARIANTS:-lite dilated multiscale motion residual gated separable_dilated causal_dilated ms_tcnpp c2f_tcn}"
BATCH_SIZE="${BATCH_SIZE:-3}"
EPOCHS="${EPOCHS:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1.0e-4}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_loss}"
EARLY_STOP_MODE="${EARLY_STOP_MODE:-auto}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must be exactly 1 for C3 mainline GPU1, got '${CUDA_VISIBLE_DEVICES}'." >&2
  exit 44
fi

if [[ -n "${SLURM_STEP_GPUS:-}" && "${SLURM_STEP_GPUS}" != "1" ]]; then
  echo "Refusing to start: SLURM_STEP_GPUS must be GPU1 when set, got '${SLURM_STEP_GPUS}'." >&2
  exit 45
fi

if [[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *"1"* ]]; then
  echo "Refusing to start: SLURM_JOB_GPUS does not include GPU1, got '${SLURM_JOB_GPUS}'." >&2
  exit 46
fi

cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "START $(date -Iseconds)"
echo "HOST $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TCN_VARIANTS=${TCN_VARIANTS}"
echo "EPOCHS=${EPOCHS} EARLY_STOP=${EARLY_STOP_METRIC}/${EARLY_STOP_MODE} patience=${EARLY_STOP_PATIENCE} min_epochs=${EARLY_STOP_MIN_EPOCHS}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Refusing to start: expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES=1.")
PY

python -u tools/bata/train_lowres_action_probe.py \
  --config configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 4 \
  --lr 1.0e-4 \
  --seed 0 \
  --probe-model temporal-tcn \
  --scout-spatial-size 64 \
  --tcn-variants ${TCN_VARIANTS} \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --early-stop-metric "${EARLY_STOP_METRIC}" \
  --early-stop-mode "${EARLY_STOP_MODE}" \
  --log-every-batches 10 \
  --fast-lowres-pipeline \
  --probe-window-size 384 \
  --ann-file /data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --class-map /data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt \
  --train-data-path "/data/run01/sczc063/yuzibo/raw/Validation Data/validation" \
  --val-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --test-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --save-checkpoint
