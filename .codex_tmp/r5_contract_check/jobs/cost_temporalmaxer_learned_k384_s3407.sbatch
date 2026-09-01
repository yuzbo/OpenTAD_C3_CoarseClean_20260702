#!/usr/bin/env bash
#SBATCH --job-name=r5-cost-te-l
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-cost-te-l-%j.out
#SBATCH --error=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/logs/r5-cost-te-l-%j.err

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
: "${R5_FRONTEND_DECISION:?set R5_FRONTEND_DECISION}"
: "${R5_ALIGNMENT_JSON:?set R5_ALIGNMENT_JSON}"
if [[ -z "${R5_FRONTEND_DECISION_SHA256:-}" ]]; then
  : "${R5_FRONTEND_DECISION_SHA256_FILE:?set R5_FRONTEND_DECISION_SHA256_FILE}"
  IFS= read -r R5_FRONTEND_DECISION_SHA256 < "${R5_FRONTEND_DECISION_SHA256_FILE}"
fi
if [[ -z "${R5_ALIGNMENT_SHA256:-}" ]]; then
  : "${R5_ALIGNMENT_SHA256_FILE:?set R5_ALIGNMENT_SHA256_FILE}"
  IFS= read -r R5_ALIGNMENT_SHA256 < "${R5_ALIGNMENT_SHA256_FILE}"
fi
[[ "${R5_FRONTEND_DECISION_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${R5_ALIGNMENT_SHA256}" =~ ^[0-9a-f]{64}$ ]]
export R5_FRONTEND_DECISION_SHA256 R5_ALIGNMENT_SHA256
readarray -t r5_frontend < <("${PYTHON}" -   "${R5_FRONTEND_DECISION}" "${R5_FRONTEND_DECISION_SHA256}"   "${R5_ALIGNMENT_JSON}" "${R5_ALIGNMENT_SHA256}"   "${DUCA_EXPECTED_COMMIT}" boundary_burst_r2q3_g1   E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py <<'PY'
import hashlib
import sys
from pathlib import Path

from tools.bata.duca_boundary_burst_hard_swap_alignment import (
    validate_alignment_artifact,
)
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
)

decision_path, decision_sha, alignment_path, alignment_sha, commit, variant, source = sys.argv[1:]
decision = validate_frontend_decision(
    decision_path=decision_path,
    decision_sha256=decision_sha,
    expected_commit=commit,
)
validate_alignment_artifact(
    path=alignment_path,
    digest=alignment_sha,
    expected_commit=commit,
    expected_variant=variant,
    source_config_path=source,
    source_config_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
)
selected = decision["family_routing"]["selected_p0_variant"]
winner = decision["winners"][selected]
print(winner["checkpoint_path"])
print(winner["checkpoint_sha256"])
print(int(winner["epoch_one_based"]) - 1)
PY
)
[[ "${#r5_frontend[@]}" == 3 ]]
export DUCA_FRONTEND_CHECKPOINT="${r5_frontend[0]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${r5_frontend[1]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${r5_frontend[2]}"
export DUCA_SELECTED_OPT_VARIANT=boundary_burst_r2q3_g1
export DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${R5_ALIGNMENT_JSON}"
export DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${R5_ALIGNMENT_SHA256}"
[[ -f "${DUCA_FRONTEND_CHECKPOINT}" ]]
[[ "$(sha256sum "${DUCA_FRONTEND_CHECKPOINT}" | awk '{print $1}')" ==    "${DUCA_FRONTEND_CHECKPOINT_SHA256}" ]]
export DUCA_SELECTED_OPT_VARIANT=temporalmaxer_learned_k384_s3407
[[ -f E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/runs/temporalmaxer_learned_k384_s3407/gpu1_id0/checkpoint/epoch_59.pth ]]
mkdir -p E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/cost
export PRECHECK_ONLY=0
export CONFIG=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/configs/temporalmaxer_learned_k384_s3407.py
export PROFILE_CHECKPOINT=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/runs/temporalmaxer_learned_k384_s3407/gpu1_id0/checkpoint/epoch_59.pth
export PROFILE_METHOD=temporalmaxer_learned_k384_s3407
export OUTPUT_PREFIX=E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/cost/temporalmaxer_learned_k384_s3407
export PROFILE_SAMPLES=30 PROFILE_WARMUP_SAMPLES=5 PROFILE_BATCH_SIZE=1
bash scripts/run_duca_full_stack_cost_profile_gpu1.sh
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/cost/temporalmaxer_learned_k384_s3407.summary.json ]]
[[ -s E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_contract_check/cost/temporalmaxer_learned_k384_s3407.samples.jsonl ]]
