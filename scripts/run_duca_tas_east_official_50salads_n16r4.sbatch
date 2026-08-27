#!/usr/bin/env bash
#SBATCH -J duca_east_50s
#SBATCH -p gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --array=1-5
#SBATCH -o /data/run01/sczc063/yuzibo/duca_tas_east_official_37c0d08_gpu2_async_20260823T001000Z/slurm/%x_%A_%a.out
#SBATCH -e /data/run01/sczc063/yuzibo/duca_tas_east_official_37c0d08_gpu2_async_20260823T001000Z/slurm/%x_%A_%a.err

source /etc/profile
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

repo=/data/run01/sczc063/yuzibo/projects/duca_east_baseline_37c0d08_20260822
envp=/data/run01/sczc063/yuzibo/conda_envs/opentad
expected_revision=37c0d080a2bce948dc73643578f05b2229934d2c
run_root=/data/run01/sczc063/yuzibo/duca_tas_east_official_37c0d08_gpu2_async_20260823T001000Z
video_dir=/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/data/50salads/raw_data/video_fps2
annotation_dir=/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/data/annotations
pretrain=/data/run01/sczc063/yuzibo/pretrained/vit-giant-p14_videomaev2-hybrid_pt_1200e_k710_ft_my.pth
config=configs/adatad/50salads/e2e_50salads_videomaev2_g_768x1_160_fps2_adapter.py
split_id="${SLURM_ARRAY_TASK_ID}"
split="split${split_id}"
annotation="${annotation_dir}/50salads.fps2.split${split_id}.json"
class_map="${annotation_dir}/50salads_category_idx.txt"
eval_dir="${run_root}/eval_protocol/${split}"
eval_annotation="${eval_dir}/50salads.fps2.split${split_id}.json"

test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_revision}"
test -z "$(git -C "${repo}" status --short)"
test -x "${envp}/bin/python"
test -f "${annotation}"
test -f "${class_map}"
test -f "${pretrain}"
test "$(stat -c %s "${pretrain}")" = 2025314665
test "$(find "${video_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)" = 50

module load cuda/11.8
module load miniforge3/24.11
source "${envp}/bin/activate"
cd "${repo}"

mkdir -p "${run_root}/slurm"
mkdir -p "${eval_dir}"
ln -sfn "${annotation}" "${eval_annotation}"
ln -sfn "${class_map}" "${eval_dir}/50salads_category_idx.txt"

common_cfg=(
  "split=${split}"
  "annotation_path=${eval_annotation}"
  "class_map=${class_map}"
  "data_path=${video_dir}"
  "dataset.train.class_map=${class_map}"
  "dataset.val.class_map=${class_map}"
  "dataset.test.class_map=${class_map}"
  "solver.train.batch_size=2"
  "solver.val.batch_size=2"
  "solver.test.batch_size=2"
  "feature_stride=2"
  "workflow.val_start_epoch=1"
  "workflow.checkpoint_keep_last=3"
  "workflow.checkpoint_milestone_epochs=[59,119,179,239]"
  "work_dir=${run_root}/stage1"
  "model.backbone.backbone.type=VisionTransformerAdapter3"
  "model.backbone.backbone.adapter_drop_rate=0.5"
  "model.backbone.custom.pretrain=${pretrain}"
  "evaluation.save_npz=True"
)

if [[ "${EAST_PRE_RUN_ONLY:-0}" = 1 ]]; then
  common_cfg+=("workflow.end_epoch=2" "workflow.val_start_epoch=1" "work_dir=${run_root}/pre_run")
  torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    tools/train.py "${config}" --seed 42 --cfg-options "${common_cfg[@]}"
  exit 0
fi

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  tools/train.py "${config}" --seed 42 --cfg-options "${common_cfg[@]}"

checkpoint="${run_root}/stage1/${split}/gpu2_id0/checkpoint/epoch_239.pth"
test -f "${checkpoint}"

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  tools/test.py "${config}" --checkpoint "${checkpoint}" --seed 42 \
  --cfg-options "${common_cfg[@]}"
