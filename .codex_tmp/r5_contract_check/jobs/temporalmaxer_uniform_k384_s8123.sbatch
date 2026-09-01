#!/usr/bin/env bash
#SBATCH --job-name=r5-te-u-k384-s8123
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-te-u-k384-s8123-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-te-u-k384-s8123-%j.err

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
export DUCA_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"
export DUCA_ADATAD_PRETRAIN_SHA256="${ADATAD_PRETRAIN_SHA256}"
mkdir -p 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs'
export R5_MATRIX_SUMMARY=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json
export R5_MECHANISM_GATE_JSON=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/temporalmaxer_one_step.json
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json.sha256 ]]
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/temporalmaxer_one_step.json.sha256 ]]
IFS= read -r R5_MATRIX_SUMMARY_SHA256 < E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/matrix_summary.json.sha256
IFS= read -r R5_MECHANISM_GATE_SHA256 < E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/temporalmaxer_one_step.json.sha256
[[ "${R5_MATRIX_SUMMARY_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${R5_MECHANISM_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${R5_MATRIX_SUMMARY}" | awk '{print $1}')" == "${R5_MATRIX_SUMMARY_SHA256}" ]]
[[ "$(sha256sum "${R5_MECHANISM_GATE_JSON}" | awk '{print $1}')" == "${R5_MECHANISM_GATE_SHA256}" ]]
export R5_MATRIX_SUMMARY_SHA256 R5_MECHANISM_GATE_SHA256
export DUCA_SELECTED_OPT_VARIANT=temporalmaxer_uniform_k384_s8123
mkdir -p E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/results
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="r5-${SLURM_JOB_ID}-r5-te-u-k384-s8123" \
  tools/train.py E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/configs/temporalmaxer_uniform_k384_s8123.py \
  --id 0 --seed 8123 \
  --cfg-options "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}"
[[ -f E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/runs/temporalmaxer_uniform_k384_s8123/gpu1_id0/checkpoint/epoch_59.pth ]]
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --rdzv_id="r5-${SLURM_JOB_ID}-r5-te-u-k384-s8123-eval" \
  tools/test.py E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/configs/temporalmaxer_uniform_k384_s8123.py \
  --checkpoint E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/runs/temporalmaxer_uniform_k384_s8123/gpu1_id0/checkpoint/epoch_59.pth \
  --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 \
  --metrics-json E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/results/temporalmaxer_uniform_k384_s8123.terminal_evaluation.json \
  --id 0 --seed 8123 \
  --cfg-options \
    "work_dir=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/results/temporalmaxer_uniform_k384_s8123_eval" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False"
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/results/temporalmaxer_uniform_k384_s8123.terminal_evaluation.json ]]
