#!/usr/bin/env bash
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
BASE=/data/run01/sczc063/yuzibo
source "$BASE/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export YUZIBO_ROOT="$BASE" DECORD_EOF_RETRY_MAX=20480
cd "${PROJECT_DIR:?}"
test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT:?}"
test -z "$(git status --porcelain)"
MODE="${MODE:?off or on}"
case "$MODE" in off|on) ;; *) exit 2;; esac
CFG="configs/adatad/thumos/ettrc_test_guided_${MODE}_seed4407.py"
RUN="${RUN_ROOT:?}/$MODE"
OPTIONS=(model.backbone.custom.pretrain="$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth" work_dir="$RUN")
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python -m pytest tests/test_ettrc_anchor_correction.py tests/test_test_guided_eval.py tests/test_ettrc_terminal_receipt.py -q
  OPTIONS+=(test_guided_exploratory=False workflow.end_epoch=1 workflow.max_train_iters=2 workflow.val_start_epoch=4)
fi
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="ettrc-${SLURM_JOB_ID}-${MODE}" tools/train.py "$CFG" --seed 4407 --cfg-options "${OPTIONS[@]}"
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="ettrc-eval-${SLURM_JOB_ID}-${MODE}" tools/test.py "$CFG" \
    --checkpoint "$RUN/gpu2_id0/checkpoint/epoch_0.pth" --seed 4407 --not_eval --max-batches 1 \
    --cfg-options "${OPTIONS[@]}"
  echo ETTRC_TEST_GUIDED_PRECHECK_OK
fi
