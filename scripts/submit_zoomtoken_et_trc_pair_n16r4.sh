#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
SEED=4407
STRIDE_K=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --stride-k) STRIDE_K="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
cd "$PROJECT_DIR"
for mode in off on; do
  if [[ "$mode" == off ]]; then
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed${SEED}.py"
  else
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed${SEED}.py"
  fi
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
  sbatch --parsable --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --job-name="et-trc-${mode}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
    --wrap="source /etc/profile; module load cuda/11.8; module load miniforge3/24.11; source '${BASE}/conda_envs/opentad/bin/activate'; cd '${PROJECT_DIR}'; python tools/train.py '${cfg}' --seed '${SEED}' --cfg-options model.backbone.backbone.stride_k=${STRIDE_K}"
done
