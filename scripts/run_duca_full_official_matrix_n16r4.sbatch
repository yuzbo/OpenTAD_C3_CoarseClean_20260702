#!/usr/bin/env bash
#SBATCH --job-name=duca-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=72:00:00
#SBATCH --array=0-14%8
#SBATCH --output=/data/run01/sczc063/yuzibo/logs/duca_full_%A_%a.out
#SBATCH --error=/data/run01/sczc063/yuzibo/logs/duca_full_%A_%a.err

source /etc/profile
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo
SOURCE=${DUCA_SOURCE:?set DUCA_SOURCE to a clean immutable checkout}
EXPECTED_COMMIT=${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}
RUN_ROOT=${DUCA_RUN_ROOT:?set DUCA_RUN_ROOT to an empty run directory}
IMAGE=${DUCA_IMAGE:-$ROOT/containers/zoomtoken_p1_ubi9_20260815.sif}
ENV_ROOT=${DUCA_ENV_ROOT:-$ROOT/conda_envs/opentad}
ANNOTATION=${DUCA_ANNOTATION:-$ROOT/thumos14/annotations/thumos_14_anno.json}
CLASS_MAP=${DUCA_CLASS_MAP:-$ROOT/thumos14/annotations/category_idx.txt}
TRAIN_DATA=${DUCA_TRAIN_DATA:-$ROOT/thumos14/raw/Validation_Data/validation}
VALIDATION_DATA=${DUCA_VALIDATION_DATA:-$ROOT/thumos14/raw/Test_Data/TH14_test_set_mp4}
PRETRAINED=${DUCA_PRETRAINED:-$ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}

if [[ -z "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}" ]]; then
  [[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
    echo "source commit mismatch" >&2
    exit 2
  }
  [[ -z "$(git -C "$SOURCE" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source checkout is not clean" >&2
    exit 2
  }
  export DUCA_SOURCE_VERIFIED=1
  module load apptainer/1.2.4
  exec apptainer exec --nv --bind "$ROOT:$ROOT" "$IMAGE" bash "$SOURCE/scripts/run_duca_full_official_matrix_n16r4.sbatch"
fi

[[ "${DUCA_SOURCE_VERIFIED:-0}" == "1" ]] || {
  echo "source checkout was not verified before entering the container" >&2
  exit 2
}
source "$ENV_ROOT/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

arms=(official_dense uniform_k384 learned_k384 dynamic_k dynamic_k_no_risk)
seeds=(3407 3408 3409)
arm_index=$((SLURM_ARRAY_TASK_ID % 5))
seed_index=$((SLURM_ARRAY_TASK_ID / 5))
arm=${arms[$arm_index]}
seed=${seeds[$seed_index]}
cell_root=$RUN_ROOT/$arm/seed_$seed

test -f "$SOURCE/tools/train.py"
test -f "$SOURCE/configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py"
test -f "$IMAGE"
test -x "$ENV_ROOT/bin/python"
test -f "$ANNOTATION"
test -f "$CLASS_MAP"
test -d "$TRAIN_DATA"
test -d "$VALIDATION_DATA"
test -f "$PRETRAINED"
test ! -e "$cell_root"
mkdir -p "$(dirname "$cell_root")" "$RUN_ROOT/logs" "$ROOT/logs"

export DUCA_ARM=$arm
export DUCA_SEED=$seed
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((24000 + SLURM_ARRAY_TASK_ID + (SLURM_ARRAY_JOB_ID % 1000)))
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
  tools/train.py configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py \
  --seed "$seed" --id 0 "${cfg_options[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/${arm}_seed${seed}_train.log"

checkpoint="$cell_root/checkpoint/epoch_59.pth"
test -f "$checkpoint"
torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  tools/test.py configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py \
  --checkpoint "$checkpoint" --seed "$seed" --id 0 "${cfg_options[@]}" \
  2>&1 | tee "$RUN_ROOT/logs/${arm}_seed${seed}_test.log"
