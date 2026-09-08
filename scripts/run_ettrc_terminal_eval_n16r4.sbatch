#!/usr/bin/env bash
#SBATCH --job-name=ettrc-terminal-eval
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00

source /etc/profile
set -euo pipefail
cd "${ETTRC_REPO:?}"
[[ "$(git rev-parse HEAD)" == "${ETTRC_EXPECTED_COMMIT:?}" ]]
[[ -z "$(git status --porcelain)" ]]
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export YUZIBO_ROOT=/data/run01/sczc063/yuzibo DECORD_EOF_RETRY_MAX=20480
export THUMOS14_ROOT=/data/run01/sczc063/yuzibo/thumos14
PRETRAIN=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
RUN_ROOT="${ETTRC_EVAL_ROOT:?}"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) ARM=off ;;
  1) ARM=on ;;
  *) exit 1 ;;
esac
CONFIG="configs/adatad/thumos/ettrc_test_guided_${ARM}_seed4407.py"
CHECKPOINT="/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/$ARM/gpu2_id0/checkpoint/epoch_59.pth"
python -m pytest tests/test_ettrc_terminal_receipt.py tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
python - "$CHECKPOINT" "$CONFIG" <<'PY'
import sys
import torch
from pathlib import Path
from mmengine.config import Config
from tools.bata.ettrc_terminal_receipt import source_identity, terminal_counts, training_binding
assert torch.cuda.device_count() == 2
print("EXACT_SHA", source_identity(Path.cwd()))
binding = training_binding(Config.fromfile(sys.argv[2]), sys.argv[1], seed=4407)
print("TRAINING_BINDING", binding["training_protocol_sha256"], len(binding["evaluated_video_ids"]), binding["test_window_count"])
print("TERMINAL", terminal_counts(torch.load(sys.argv[1], map_location="cpu")))
PY
EXTRA=()
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  WORK_DIR="$RUN_ROOT/precheck/$ARM"
  EXTRA+=(--not_eval --max-batches 1)
else
  WORK_DIR="$RUN_ROOT/formal/$ARM"
  EXTRA+=(--ettrc-metrics-json "$WORK_DIR/gpu2_id0/official_eval/metrics.json")
fi
torchrun --standalone --nproc_per_node=2 tools/test.py "$CONFIG" \
  --checkpoint "$CHECKPOINT" --seed 4407 "${EXTRA[@]}" \
  --cfg-options work_dir="$WORK_DIR" model.backbone.custom.pretrain="$PRETRAIN" post_processing.save_dict=True
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo ETTRC_EVAL5_TERMINAL_PRECHECK_OK
fi
