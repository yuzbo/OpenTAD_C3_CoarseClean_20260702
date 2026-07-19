#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_CANDIDATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_ALLOCATION_RUN_ROOT:-}"
CHECKPOINT="${DUCA_ALLOCATION_CHECKPOINT:-}"
PRETRAIN="${ADATAD_PRETRAIN_PATH:-}"
CONFIG="configs/adatad/thumos/duca_allocation_ceiling_training_windows.py"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "candidate evaluation requires a Slurm GPU"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] || fail "candidate evaluation requires cluster n16r4"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "candidate evaluation requires a clean tree"
[[ -f "${CHECKPOINT}" && -f "${PRETRAIN}" ]] || fail "checkpoint or pretrain is missing"
[[ -f "${RUN_ROOT}/training_gt32_ceiling.jsonl" ]] || fail "GT ceiling artifact is missing"

"${PYTHON}" -m tools.bata.evaluate_duca_allocation_candidates \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --backbone-pretrain "${PRETRAIN}" \
  --input-jsonl "${RUN_ROOT}/training_gt32_inputs.jsonl" \
  --ceiling-jsonl "${RUN_ROOT}/training_gt32_ceiling.jsonl" \
  --ceiling-summary-json "${RUN_ROOT}/training_gt32_ceiling.summary.json" \
  --ceiling-validation-json "${RUN_ROOT}/training_gt32_ceiling.validation.json" \
  --output-jsonl "${RUN_ROOT}/training_gt32_candidate_loss.jsonl" \
  --summary-json "${RUN_ROOT}/training_gt32_candidate_loss.summary.json" \
  --split train \
  --family-keys \
    A_exact_uniform D_deploy_score D_privileged_gt_ceiling E_privileged_unrestricted_gt \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 2

"${PYTHON}" -m tools.bata.validate_duca_allocation_candidate_loss_artifact \
  --ceiling-jsonl "${RUN_ROOT}/training_gt32_ceiling.jsonl" \
  --candidate-jsonl "${RUN_ROOT}/training_gt32_candidate_loss.jsonl" \
  --summary-json "${RUN_ROOT}/training_gt32_candidate_loss.summary.json" \
  --validation-json "${RUN_ROOT}/training_gt32_candidate_loss.validation.json"

echo "[DUCA_ALLOCATION_CANDIDATE] PASS"
