#!/usr/bin/env bash
#SBATCH --job-name=duca-uvt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/data/run01/sczc063/yuzibo/logs/duca_uvt_%A_%a.out
#SBATCH --error=/data/run01/sczc063/yuzibo/logs/duca_uvt_%A_%a.err

source /etc/profile
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo
SOURCE=${DUCA_SOURCE:?set DUCA_SOURCE}
EXPECTED_COMMIT=${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}
RUN_ROOT=${DUCA_RUN_ROOT:?set DUCA_RUN_ROOT}
IMAGE=${DUCA_IMAGE:-$ROOT/containers/zoomtoken_p1_ubi9_20260815.sif}
ENV_ROOT=${DUCA_ENV_ROOT:-$ROOT/conda_envs/opentad}
ANNOTATION=${DUCA_ANNOTATION:-$ROOT/thumos14/annotations/thumos_14_anno.json}
CLASS_MAP=${DUCA_CLASS_MAP:-$ROOT/thumos14/annotations/category_idx.txt}
TRAIN_DATA=${DUCA_TRAIN_DATA:-$ROOT/thumos14/train}
VALIDATION_DATA=${DUCA_VALIDATION_DATA:-$ROOT/thumos14/test}
PRETRAINED=${DUCA_PRETRAINED:-$ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}

if [[ -z "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}" ]]; then
  [[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || exit 2
  [[ -z "$(git -C "$SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || exit 2
  export DUCA_SOURCE_VERIFIED=1
  module load apptainer/1.2.4
  exec apptainer exec --nv --bind "$ROOT:$ROOT" "$IMAGE" bash "$SOURCE/scripts/run_duca_uvt_value_portal_n16r4.sbatch"
fi

[[ "${DUCA_SOURCE_VERIFIED:-0}" == "1" ]] || exit 2
source "$ENV_ROOT/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

modes=(off geo geo_ema)
seed=3407
mode=${modes[$SLURM_ARRAY_TASK_ID]}
cell_root=$RUN_ROOT/$mode/seed_$seed

test -f "$SOURCE/tools/train.py"
test -f "$SOURCE/configs/adatad/thumos/duca_uvt_value_portal_n16r4.py"
test -f "$IMAGE"
test -x "$ENV_ROOT/bin/python"
test -f "$ANNOTATION"
test -f "$CLASS_MAP"
test -d "$TRAIN_DATA"
test -d "$VALIDATION_DATA"
test -f "$PRETRAINED"
mkdir -p "$(dirname "$cell_root")" "$RUN_ROOT/logs" "$ROOT/logs"

export DUCA_UVT_MODE=$mode
export DUCA_UVT_WORK_DIR=$cell_root
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((25000 + SLURM_ARRAY_TASK_ID + (SLURM_ARRAY_JOB_ID % 1000)))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SOURCE"
cfg_options=(--cfg-options \
  work_dir="$cell_root" \
  dataset.train.ann_file="$ANNOTATION" \
  dataset.train.class_map="$CLASS_MAP" \
  dataset.train.data_path="$TRAIN_DATA" \
  dataset.val.ann_file="$ANNOTATION" \
  dataset.val.class_map="$CLASS_MAP" \
  dataset.val.data_path="$VALIDATION_DATA" \
  dataset.test.ann_file="$ANNOTATION" \
  dataset.test.class_map="$CLASS_MAP" \
  dataset.test.data_path="$VALIDATION_DATA" \
  evaluation.ground_truth_filename="$ANNOTATION" \
  model.backbone.custom.pretrain="$PRETRAINED")

torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  tools/train.py configs/adatad/thumos/duca_uvt_value_portal_n16r4.py \
  --seed "$seed" --id 0 "${cfg_options[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/${mode}_seed${seed}_train.log"

checkpoint="$cell_root/gpu1_id0/checkpoint/epoch_59.pth"
test -f "$checkpoint"
torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  tools/test.py configs/adatad/thumos/duca_uvt_value_portal_n16r4.py \
  --checkpoint "$checkpoint" --seed "$seed" --id 0 "${cfg_options[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/${mode}_seed${seed}_test.log"
