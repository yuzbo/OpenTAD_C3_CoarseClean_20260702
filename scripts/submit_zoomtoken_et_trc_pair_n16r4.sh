#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
ETTRC_PRETRAIN="${ETTRC_PRETRAIN:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
[[ -f "$ETTRC_PRETRAIN" ]] || { echo "missing ET-TRC pretrain: $ETTRC_PRETRAIN" >&2; exit 1; }
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
[[ "$(git branch --show-current)" == "codex/zoomtoken-et-trc-correction-20260902" ]] || {
  echo "ET-TRC pair requires correction branch" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "ET-TRC checkout is not clean" >&2; exit 2; }
EXPECTED_COMMIT="${ETTRC_EXPECTED_COMMIT:?ETTRC_EXPECTED_COMMIT must be the full 40-character target SHA}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "ETTRC_EXPECTED_COMMIT must be a full SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "ET-TRC HEAD mismatch" >&2; exit 2; }
export ETTRC_PRETRAIN
for mode in off on; do
  if [[ "$mode" == off ]]; then
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed${SEED}.py"
  else
    cfg="configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed${SEED}.py"
  fi
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
    sbatch --parsable --export=ALL,ETTRC_PRETRAIN="$ETTRC_PRETRAIN",ETTRC_EXPECTED_COMMIT="$EXPECTED_COMMIT" --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --job-name="et-trc-${mode}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
      --wrap="bash -lc 'source /etc/profile; set -euo pipefail; module load cuda/11.8; module load miniforge3/24.11; source \"${BASE}/conda_envs/opentad/bin/activate\"; export ETTRC_PRETRAIN=\"${ETTRC_PRETRAIN}\" ETTRC_EXPECTED_COMMIT=\"${EXPECTED_COMMIT}\"; cd \"${PROJECT_DIR}\"; test \"\$(git rev-parse HEAD)\" = \"\${ETTRC_EXPECTED_COMMIT}\"; test -z \"\$(git status --porcelain)\"; torchrun --standalone --nproc_per_node=2 tools/train.py \"${cfg}\" --seed \"${SEED}\" --cfg-options model.backbone.backbone.stride_k=${STRIDE_K} model.backbone.custom.pretrain=\"${ETTRC_PRETRAIN}\"'"
done
