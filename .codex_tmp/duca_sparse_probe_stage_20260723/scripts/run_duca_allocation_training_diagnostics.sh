#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_DIAGNOSTICS][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_ALLOCATION_RUN_ROOT:-}"
INPUT="${RUN_ROOT}/training_inputs.jsonl"
GT_TIME_LIMIT_SECONDS="${DUCA_ALLOCATION_GT_TIME_LIMIT_SECONDS:-300}"
GATE_JSON="${DUCA_ALLOCATION_GATE_JSON:-}"
SUBMISSION_JSON="${DUCA_ALLOCATION_SUBMISSION_JSON:-}"
SUBMISSION_TOKEN="${DUCA_ALLOCATION_SUBMISSION_TOKEN:-}"
SUITE_MANIFEST="${DUCA_ALLOCATION_SUITE_MANIFEST:-}"
SUITE_MANIFEST_SHA256="${DUCA_ALLOCATION_SUITE_MANIFEST_SHA256:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "diagnostics must run inside Slurm"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] || fail "diagnostics require cluster n16r4"
[[ "${GT_TIME_LIMIT_SECONDS}" == "300" ]] \
  || fail "GT total solver deadline must match the registered 300 seconds"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "diagnostics require a clean tree"
[[ "${RUN_ROOT}" == "${BASE}/"* && -f "${INPUT}" ]] || fail "training input artifact is missing"

"${PYTHON}" -m tools.bata.validate_duca_allocation_submission_receipt \
  --submission-json "${SUBMISSION_JSON}" \
  --submission-token "${SUBMISSION_TOKEN}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --suite-manifest-json "${SUITE_MANIFEST}" \
  --suite-manifest-sha256 "${SUITE_MANIFEST_SHA256}" \
  --role diagnostics \
  --current-job-id "${SLURM_JOB_ID}" \
  --gate-json "${GATE_JSON}"

"${PYTHON}" -m tools.bata.diagnose_duca_allocation_family_ceiling \
  --input-jsonl "${INPUT}" \
  --output-jsonl "${RUN_ROOT}/training_recoverability.jsonl" \
  --summary-json "${RUN_ROOT}/training_recoverability.summary.json" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --gt-families none \
  --quantization-scale 1000000

"${PYTHON}" -m tools.bata.validate_duca_allocation_ceiling_artifact \
  --input-jsonl "${INPUT}" \
  --output-jsonl "${RUN_ROOT}/training_recoverability.jsonl" \
  --summary-json "${RUN_ROOT}/training_recoverability.summary.json" \
  --validation-json "${RUN_ROOT}/training_recoverability.validation.json"

"${PYTHON}" -m tools.bata.subset_duca_allocation_inputs \
  --input-jsonl "${INPUT}" \
  --output-jsonl "${RUN_ROOT}/training_gt32_inputs.jsonl" \
  --summary-json "${RUN_ROOT}/training_gt32_inputs.summary.json" \
  --first-n 32 \
  --strategy hash_video_round_robin \
  --seed duca-allocation-ceiling-v1

"${PYTHON}" -m tools.bata.diagnose_duca_allocation_family_ceiling \
  --input-jsonl "${RUN_ROOT}/training_gt32_inputs.jsonl" \
  --output-jsonl "${RUN_ROOT}/training_gt32_ceiling.jsonl" \
  --summary-json "${RUN_ROOT}/training_gt32_ceiling.summary.json" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --gt-families both \
  --lex-block-size 30 \
  --quantization-scale 1000000 \
  --gt-time-limit-seconds "${GT_TIME_LIMIT_SECONDS}"

"${PYTHON}" -m tools.bata.validate_duca_allocation_ceiling_artifact \
  --input-jsonl "${RUN_ROOT}/training_gt32_inputs.jsonl" \
  --output-jsonl "${RUN_ROOT}/training_gt32_ceiling.jsonl" \
  --summary-json "${RUN_ROOT}/training_gt32_ceiling.summary.json" \
  --validation-json "${RUN_ROOT}/training_gt32_ceiling.validation.json"

"${PYTHON}" -m tools.bata.profile_duca_allocation_solver_cost \
  --input-jsonl "${INPUT}" \
  --output-samples-jsonl "${RUN_ROOT}/training_solver_cost.samples.jsonl" \
  --output-summary-json "${RUN_ROOT}/training_solver_cost.summary.json" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --warmup-samples 10 \
  --samples 100

"${PYTHON}" -m tools.bata.validate_duca_allocation_solver_cost_artifact \
  --input-jsonl "${INPUT}" \
  --samples-jsonl "${RUN_ROOT}/training_solver_cost.samples.jsonl" \
  --summary-json "${RUN_ROOT}/training_solver_cost.summary.json" \
  --validation-json "${RUN_ROOT}/training_solver_cost.validation.json"

echo "[DUCA_ALLOCATION_DIAGNOSTICS] PASS"
