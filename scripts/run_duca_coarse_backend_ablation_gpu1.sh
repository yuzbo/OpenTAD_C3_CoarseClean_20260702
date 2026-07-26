#!/bin/bash
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
OFFICIAL_REPOS="${C3_OFFICIAL_ACTION_SEG_REPOS:-${BASE}/projects/external_official_action_segmentation_repos_20260702}"
BACKEND="${BACKEND:?BACKEND is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-3407}"
PROBE_WINDOW_SIZE="${PROBE_WINDOW_SIZE:-768}"

case "${BACKEND}" in
  official_ms_tcn2)
    REQUIRED_SOURCE="${OFFICIAL_REPOS}/MS-TCN2/model.py"
    ;;
  official_asformer)
    REQUIRED_SOURCE="${OFFICIAL_REPOS}/ASFormer/model.py"
    ;;
  official_fact)
    REQUIRED_SOURCE="${OFFICIAL_REPOS}/CVPR2024-FACT/models/blocks.py"
    ;;
  official_video_mamba_asformer)
    REQUIRED_SOURCE="${OFFICIAL_REPOS}/video-mamba-suite/video-mamba-suite/temporal-action-segmentation/model.py"
    ;;
  *)
    echo "Unsupported official coarse backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "This training entry must run inside Slurm." >&2; exit 3; }
[[ -f "${REQUIRED_SOURCE}" ]] || { echo "Missing official source: ${REQUIRED_SOURCE}" >&2; exit 4; }

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${PROJECT_DIR}"

export C3_OFFICIAL_ACTION_SEG_REPOS="${OFFICIAL_REPOS}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${OUT_DIR}"

python - "${BACKEND}" <<'PY'
import sys
import torch
from tools.bata import train_lowres_action_probe as probe

backend = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one Slurm-visible CUDA device")
if not probe.official_action_seg_backend_available(backend):
    raise SystemExit(f"official backend unavailable: {backend}")
frames = torch.rand(1, 8, 3, 64, 64, device="cuda")
valid = torch.ones(1, 8, dtype=torch.bool, device="cuda")
model = probe.C3OfficialActionSegmentationProbe(
    backend=backend,
    spatial_size=64,
    hidden_dim=96,
    num_layers=2,
).to("cuda").eval()
with torch.no_grad():
    logits = model(frames, valid)
if tuple(logits.shape) != (1, 8) or not torch.isfinite(logits).all():
    raise SystemExit(f"invalid tiny CUDA output for {backend}: {tuple(logits.shape)}")
print("DUCA_COARSE_BACKEND_CUDA_GATE_OK", backend, flush=True)
PY

python -u tools/bata/train_lowres_action_probe.py \
  --config configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --num-workers 4 \
  --lr 1.0e-4 \
  --seed "${SEED}" \
  --probe-model official-action-seg \
  --scout-spatial-size 64 \
  --official-action-seg-backends "${BACKEND}" \
  --max-train-batches 0 \
  --max-val-batches 0 \
  --val-every-epochs "${EPOCHS}" \
  --early-stop-patience 0 \
  --log-every-batches 10 \
  --fast-lowres-pipeline \
  --probe-window-size "${PROBE_WINDOW_SIZE}" \
  --ann-file "${BASE}/thumos14/annotations/thumos_14_anno.json" \
  --class-map "${BASE}/thumos14/annotations/category_idx.txt" \
  --train-data-path "${BASE}/raw/Validation Data/validation" \
  --val-data-path "${BASE}/raw/Test Data/TH14_test_set_mp4" \
  --test-data-path "${BASE}/raw/Test Data/TH14_test_set_mp4" \
  --save-checkpoint

echo "DUCA_COARSE_BACKEND_COMPLETE backend=${BACKEND} out_dir=${OUT_DIR}"
