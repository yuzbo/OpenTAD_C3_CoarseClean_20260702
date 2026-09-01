#!/usr/bin/env bash
#SBATCH --job-name=r5-temporalmaxer-one-step
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_verify/logs/r5-temporalmaxer-one-step-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_verify/logs/r5-temporalmaxer-one-step-%j.err

set -euo pipefail
: "${DUCA_REPO_ROOT:?set DUCA_REPO_ROOT}"
: "${DUCA_EXPECTED_COMMIT:?set DUCA_EXPECTED_COMMIT}"
: "${ADATAD_PRETRAIN_PATH:?set ADATAD_PRETRAIN_PATH}"
: "${ADATAD_PRETRAIN_SHA256:?set ADATAD_PRETRAIN_SHA256}"
cd "${DUCA_REPO_ROOT}"
source scripts/duca_cellcf_canonical_env.sh
[[ "$(git rev-parse HEAD)" == "${DUCA_EXPECTED_COMMIT}" ]]
[[ "$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')" == "${ADATAD_PRETRAIN_SHA256}" ]]
mkdir -p 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_verify/logs'
: "${DUCA_FRONTEND_CHECKPOINT:?set DUCA_FRONTEND_CHECKPOINT}"
: "${DUCA_FRONTEND_CHECKPOINT_SHA256:?set DUCA_FRONTEND_CHECKPOINT_SHA256}"
: "${DUCA_FRONTEND_CHECKPOINT_EPOCH:?set DUCA_FRONTEND_CHECKPOINT_EPOCH}"
"${PYTHON}" -m tools.bata.run_duca_temporalmaxer_one_step \
  --config 'E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.codex_tmp\r5_verify\configs\temporalmaxer_learned_k384_s3407.py' \
  --pretrain "${ADATAD_PRETRAIN_PATH}" \
  --pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
  --seed 3407 \
  --output 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_verify/temporalmaxer_one_step.json'
