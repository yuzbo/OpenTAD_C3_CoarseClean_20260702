#!/usr/bin/env bash
#SBATCH --job-name=pt-ind-v6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260729_v6/slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260729_v6/slurm-%j.err

set -euo pipefail
umask 027

RUN_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260729_v6
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v7
V16_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v16
ANNOTATION=/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json
POLICY="${RUNTIME}/configs/adatad/thumos/phystime_decode_cross_independent_policy_v1.json"
OUTPUT_DIR="${RUN_ROOT}/independent_recompute"
OUTPUT="${OUTPUT_DIR}/INDEPENDENT_RECOMPUTE.json"

test -d "${RUN_ROOT}"
test ! -e "${OUTPUT_DIR}"
test "${CUDA_VISIBLE_DEVICES:-}" != ""
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = b4742b5246ff43443e3244b929743eb8f389d4cb
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 7fc8f2d43dba5dfa3f8ef8c66ca064e88c240b1b
test -z "$(git -C "${RUNTIME}" status --porcelain)"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir "${OUTPUT_DIR}"
{
    printf 'runtime_commit=%s\n' "$(git -C "${RUNTIME}" rev-parse HEAD)"
    printf 'runtime_tree=%s\n' "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'gpu_used_by_validator=false\n'
} > "${RUN_ROOT}/runtime_identity.txt"

sha256sum \
    "${ANNOTATION}" \
    "${POLICY}" \
    "${V16_ROOT}/selected_online/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/selected_ema/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/physical_online/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/physical_ema/DECODE_CROSS_COMPLETE.json" \
    > "${RUN_ROOT}/input_sha256s.txt"

cd "${RUNTIME}"
python tools/bata/validate_phystime_independent_recompute.py \
    --completion "${V16_ROOT}/selected_online/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/selected_ema/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/physical_online/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/physical_ema/DECODE_CROSS_COMPLETE.json" \
    --annotation "${ANNOTATION}" \
    --policy "${POLICY}" \
    --output "${OUTPUT}" \
    --proposal-atol 0.0001 \
    --metric-atol 0.0001 \
    2>&1 | tee "${OUTPUT_DIR}/validator.log"

test -s "${OUTPUT}"
test -z "$(git status --porcelain)"
sha256sum \
    "${OUTPUT}" \
    "${OUTPUT_DIR}/validator.log" \
    "${RUN_ROOT}/input_sha256s.txt" \
    "${RUN_ROOT}/runtime_identity.txt" \
    > "${RUN_ROOT}/SHA256SUMS"
