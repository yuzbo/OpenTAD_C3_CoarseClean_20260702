#!/usr/bin/env bash
#SBATCH --job-name=sdpq-support-v5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v5/slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v5/slurm-%j.err

set -euo pipefail
umask 027

RUN_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sdpq_support_observability_20260729_v5
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v6
CONFIG_REPO=/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_4a57577_20260716_gtboundaryfix
CONFIG="${CONFIG_REPO}/configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py"
CHECKPOINT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_4a57577_gtboundaryfix_medium20_20260716_190900_0800/sdpq_pilot/work_dir/gpu1_id0/checkpoint/epoch_19.pth
VIDEOMAE=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
OUTPUT_DIR="${RUN_ROOT}/support_online"

test -d "${RUN_ROOT}"
test ! -e "${OUTPUT_DIR}"
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENTAD_THUMOS14_ANNOTATION=/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json
export OPENTAD_THUMOS14_CLASS_MAP=/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt
export OPENTAD_THUMOS14_TRAIN_VIDEOS=/data/run01/sczc063/yuzibo/thumos14/train
export OPENTAD_THUMOS14_TEST_VIDEOS=/data/run01/sczc063/yuzibo/thumos14/test

test "${CUDA_VISIBLE_DEVICES:-}" != ""
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 17a7b55071e3ac9da1f5a6611a84f52b98db3e21
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 7cad5c56392d1f7cf296158688083a4f5e327131
test -z "$(git -C "${RUNTIME}" status --porcelain)"
test "$(git -C "${CONFIG_REPO}" rev-parse HEAD)" = 4a57577193c07cc90ac0867176aa79c76f637c36
test "$(git -C "${CONFIG_REPO}" rev-parse "HEAD^{tree}")" = 2d9ae007b7d9cea179a9ec5e08a82bf01ef4cf4c
test -z "$(git -C "${CONFIG_REPO}" status --porcelain)"
echo "21ba537b007f416050db41bf9cb19a07a145f10d8f4a6e8623d8f405b67ca26e  ${CONFIG}" | sha256sum -c -
echo "40fccfd854a88903aaf795c04b94068af4007663c5d63064201990d70b2c3fc7  ${CHECKPOINT}" | sha256sum -c -
echo "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251  ${VIDEOMAE}" | sha256sum -c -
sha256sum \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${VIDEOMAE}" \
    "${OPENTAD_THUMOS14_ANNOTATION}" \
    "${OPENTAD_THUMOS14_CLASS_MAP}" \
    > "${RUN_ROOT}/input_sha256s.txt"
{
    printf 'runtime_commit=%s\n' "$(git -C "${RUNTIME}" rev-parse HEAD)"
    printf 'runtime_tree=%s\n' "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")"
    printf 'config_commit=%s\n' "$(git -C "${CONFIG_REPO}" rev-parse HEAD)"
    printf 'config_tree=%s\n' "$(git -C "${CONFIG_REPO}" rev-parse "HEAD^{tree}")"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
} > "${RUN_ROOT}/runtime_identity.txt"

cd "${RUNTIME}"
python tools/bata/audit_sdpq_support_observability.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --expected-checkpoint-epoch 19 \
    --weights-source online \
    --videomae-checkpoint "${VIDEOMAE}" \
    --expected-runtime-commit 17a7b55071e3ac9da1f5a6611a84f52b98db3e21 \
    --expected-runtime-tree 7cad5c56392d1f7cf296158688083a4f5e327131 \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num-windows 64 \
    2>&1 | tee "${RUN_ROOT}/support_audit.log"

test -s "${OUTPUT_DIR}/SUPPORT_AUDIT_COMPLETE.json"
test -s "${OUTPUT_DIR}/support_observability_rows.json"
test -s "${OUTPUT_DIR}/SAMPLE_SEAL.json"
test -z "$(git -C "${RUNTIME}" status --porcelain)"
test -z "$(git -C "${CONFIG_REPO}" status --porcelain)"
sha256sum \
    "${RUN_ROOT}/support_audit.log" \
    "${RUN_ROOT}/input_sha256s.txt" \
    "${RUN_ROOT}/runtime_identity.txt" \
    "${OUTPUT_DIR}/SUPPORT_AUDIT_COMPLETE.json" \
    "${OUTPUT_DIR}/support_observability_rows.json" \
    "${OUTPUT_DIR}/SAMPLE_SEAL.json" \
    > "${RUN_ROOT}/SHA256SUMS"
