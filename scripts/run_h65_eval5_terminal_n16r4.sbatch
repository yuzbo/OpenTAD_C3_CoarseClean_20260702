#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=h65-eval5-terminal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/h65-eval5-%A_%a.out
#SBATCH --error=/data/run01/sczc063/yuzibo/h65-eval5-%A_%a.err
source /etc/profile
set -euo pipefail

EXPECTED="${1:?expected evaluator SHA required}"
OUTPUT="${2:?absolute output root required}"
ROOT="${SLURM_SUBMIT_DIR:?submit from exact evaluator checkout}"
cd "$ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED" && -z "$(git status --porcelain)" ]]
[[ -n "${SLURM_JOB_ID:-}" && "$OUTPUT" == /data/run01/sczc063/yuzibo/experiments/* ]]
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT" OMP_NUM_THREADS=4
export YUZIBO_ROOT=/data/run01/sczc063/yuzibo
export DUCA_CELLCF_TRAINING_PROFILE=official60
export THUMOS14_ANNOTATION_PATH="$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json"
export THUMOS14_CLASS_MAP="$YUZIBO_ROOT/thumos14/annotations/category_idx.txt"
export THUMOS14_TRAIN_DATA_PATH="$YUZIBO_ROOT/thumos14/train"
export THUMOS14_TEST_DATA_PATH="$YUZIBO_ROOT/thumos14/test"
export DECORD_EOF_RETRY_MAX=20480
export DUCA_EXPECTED_COMMIT="$EXPECTED"
export DUCA_ADATAD_PRETRAIN_PATH="$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
export DUCA_ADATAD_PRETRAIN_SHA256="$(sha256sum "$DUCA_ADATAD_PRETRAIN_PATH" | awk '{print $1}')"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 20000))
[[ "$(python -c 'import torch; print(torch.cuda.device_count())')" == 1 ]]

case "${SLURM_ARRAY_TASK_ID:?array task required}" in
  0) ARM=uniform; ID=TEST-UNIFORM ;;
  1) ARM=phaseoff; ID=TEST-PHASEOFF ;;
  2) ARM=phaseon; ID=TEST-PHASEON ;;
  *) exit 2 ;;
esac
export DUCA_SELECTED_OPT_VARIANT="h65_pro_eval5_$ARM"
TRAIN_SOURCE="$YUZIBO_ROOT/projects/h65_tia_eval5_629162cd"
TRAIN_ROOT="$YUZIBO_ROOT/experiments/h65_tia_eval5_629162cd_seed3407/formal"
CONFIG="$TRAIN_SOURCE/configs/adatad/thumos/h65_pro/h65_pro_eval5_$ARM.py"
CHECKPOINT="$TRAIN_ROOT/${ID}_seed3407/gpu1_id0/checkpoint/epoch_59.pth"
MODE=formal
EXTRA=()
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  MODE=precheck
  EXTRA+=(--precheck-only)
fi
WORK="$OUTPUT/$MODE/$ARM"
exec python tools/test.py "$CONFIG" --seed 3407 --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-epoch 59 --checkpoint-state-key state_dict_ema \
  --metrics-json "$WORK/gpu1_id0/official_eval/metrics.json" "${EXTRA[@]}" \
  --cfg-options work_dir="$WORK" model.backbone.custom.pretrain="$DUCA_ADATAD_PRETRAIN_PATH" \
  post_processing.save_dict=True inference.load_from_raw_predictions=False
