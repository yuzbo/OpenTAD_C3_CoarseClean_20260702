#!/usr/bin/env bash
#SBATCH --job-name=r5-cost-ac-u
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs/r5-cost-ac-u-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs/r5-cost-ac-u-%j.err

set -euo pipefail
: "${DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}"
: "${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}"
: "${ADATAD_PRETRAIN_PATH:?set ADATAD_PRETRAIN_PATH}"
: "${ADATAD_PRETRAIN_SHA256:?set ADATAD_PRETRAIN_SHA256}"
cd "${DUCA_REPO_ROOT}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${DUCA_EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')" == "${ADATAD_PRETRAIN_SHA256}" ]]
mkdir -p 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/logs'
[[ -f E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/actionformer_uniform_k384_s3407/gpu1_id0/checkpoint/epoch_59.pth ]]
mkdir -p E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/cost
export PRECHECK_ONLY=0
export CONFIG=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/configs/actionformer_uniform_k384_s3407.py
export PROFILE_CHECKPOINT=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/actionformer_uniform_k384_s3407/gpu1_id0/checkpoint/epoch_59.pth
export PROFILE_METHOD=actionformer_uniform_k384_s3407
export OUTPUT_PREFIX=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/cost/actionformer_uniform_k384_s3407
export PROFILE_SAMPLES=30 PROFILE_WARMUP_SAMPLES=5 PROFILE_BATCH_SIZE=1
bash scripts/run_duca_full_stack_cost_profile_gpu1.sh
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/cost/actionformer_uniform_k384_s3407.summary.json ]]
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/cost/actionformer_uniform_k384_s3407.samples.jsonl ]]
