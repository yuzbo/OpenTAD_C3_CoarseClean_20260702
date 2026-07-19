#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_ALLOCATION_CHECKPOINT:-}"
PRETRAIN="${ADATAD_PRETRAIN_PATH:-}"
OUTPUT_ROOT="${DUCA_ALLOCATION_GATE_ROOT:-}"
EXPECTED_EPOCH="${DUCA_ALLOCATION_CHECKPOINT_EPOCH:-131}"
CONFIG="configs/adatad/thumos/duca_allocation_ceiling_training_windows.py"
EXPECTED_CHECKPOINT_SHA256="${DUCA_ALLOCATION_CHECKPOINT_SHA256:-}"
EXPECTED_PRETRAIN_SHA256="${ADATAD_PRETRAIN_SHA256:-}"
SUITE_MANIFEST="${DUCA_ALLOCATION_SUITE_MANIFEST:-}"
SUITE_MANIFEST_SHA256="${DUCA_ALLOCATION_SUITE_MANIFEST_SHA256:-}"
GT_TIME_LIMIT_SECONDS="${DUCA_ALLOCATION_GT_TIME_LIMIT_SECONDS:-300}"
MAX_PROJECTED_GT32_SECONDS="${DUCA_ALLOCATION_MAX_PROJECTED_GT32_SECONDS:-259200}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "real gate must run inside Slurm"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] || fail "real gate requires cluster n16r4"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ -x "${PYTHON}" ]] || fail "OpenTAD Python is missing"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "DUCA_EXPECTED_COMMIT is invalid"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "gate requires a clean tree"
[[ -f "${CHECKPOINT}" ]] || fail "DUCA_ALLOCATION_CHECKPOINT is missing"
[[ -f "${PRETRAIN}" ]] || fail "ADATAD_PRETRAIN_PATH is missing"
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CHECKPOINT_SHA256}" ]] \
  || fail "checkpoint hash drift"
[[ "$(sha256sum "${PRETRAIN}" | awk '{print $1}')" == "${EXPECTED_PRETRAIN_SHA256}" ]] \
  || fail "pretrain hash drift"
[[ -f "${SUITE_MANIFEST}" ]] || fail "suite manifest is missing"
[[ "$(sha256sum "${SUITE_MANIFEST}" | awk '{print $1}')" == "${SUITE_MANIFEST_SHA256}" ]] \
  || fail "suite manifest hash drift"
[[ "${GT_TIME_LIMIT_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || fail "GT time limit is invalid"
[[ "${MAX_PROJECTED_GT32_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || fail "GT projected-cost limit is invalid"
[[ -n "${OUTPUT_ROOT}" ]] || fail "DUCA_ALLOCATION_GATE_ROOT is required"
[[ "${OUTPUT_ROOT}" == "${BASE}/"* ]] || fail "gate output must stay under ${BASE}"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite gate output"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "gate requires exactly one Slurm-visible GPU"
mkdir -p "${OUTPUT_ROOT}"

INPUT_JSONL="${OUTPUT_ROOT}/training_input_one.jsonl"
EXPORT_SUMMARY="${OUTPUT_ROOT}/training_input_one.summary.json"
CEILING_JSONL="${OUTPUT_ROOT}/training_ceiling_one.jsonl"
CEILING_SUMMARY="${OUTPUT_ROOT}/training_ceiling_one.summary.json"
VALIDATION_JSON="${OUTPUT_ROOT}/training_ceiling_one.validation.json"
CANDIDATE_JSONL="${OUTPUT_ROOT}/candidate_loss_one.jsonl"
CANDIDATE_SUMMARY="${OUTPUT_ROOT}/candidate_loss_one.summary.json"
CANDIDATE_VALIDATION="${OUTPUT_ROOT}/candidate_loss_one.validation.json"
SOLVER_SAMPLES="${OUTPUT_ROOT}/solver_cost.samples.jsonl"
SOLVER_SUMMARY="${OUTPUT_ROOT}/solver_cost.summary.json"
GATE_JSON="${OUTPUT_ROOT}/allocation_ceiling_real_gate.json"
GT_RUNTIME="${OUTPUT_ROOT}/gt_runtime.json"

"${PYTHON}" -m tools.bata.export_duca_allocation_ceiling_inputs \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-jsonl "${INPUT_JSONL}" \
  --summary-json "${EXPORT_SUMMARY}" \
  --split train \
  --requested-budget 384 \
  --device cuda:0 \
  --use-ema true \
  --batch-size 1 \
  --num-workers 0 \
  --limit-batches 1 \
  --coordinate-tolerance-frames 0

GT_GENERATION_START="$(date +%s.%N)"
"${PYTHON}" -m tools.bata.diagnose_duca_allocation_family_ceiling \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${CEILING_JSONL}" \
  --summary-json "${CEILING_SUMMARY}" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --gt-families both \
  --lex-block-size 30 \
  --quantization-scale 1000000 \
  --gt-time-limit-seconds "${GT_TIME_LIMIT_SECONDS}"
GT_GENERATION_END="$(date +%s.%N)"

GT_VALIDATION_START="$(date +%s.%N)"
"${PYTHON}" -m tools.bata.validate_duca_allocation_ceiling_artifact \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${CEILING_JSONL}" \
  --summary-json "${CEILING_SUMMARY}" \
  --validation-json "${VALIDATION_JSON}"
GT_VALIDATION_END="$(date +%s.%N)"

"${PYTHON}" - \
  "${GT_GENERATION_START}" \
  "${GT_GENERATION_END}" \
  "${GT_VALIDATION_START}" \
  "${GT_VALIDATION_END}" \
  "${MAX_PROJECTED_GT32_SECONDS}" \
  "${GT_RUNTIME}" <<'PY'
import json
import pathlib
import sys

generation = float(sys.argv[2]) - float(sys.argv[1])
validation = float(sys.argv[4]) - float(sys.argv[3])
maximum = float(sys.argv[5])
payload = {
    "schema_version": "duca_allocation_gt_runtime_projection_v1",
    "gt_generation_seconds": generation,
    "gt_validation_seconds": validation,
    "projected_gt32_seconds": (generation + validation) * 32.0,
    "max_projected_gt32_seconds": maximum,
}
target = pathlib.Path(sys.argv[6])
target.open("x", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
PY

"${PYTHON}" -m tools.bata.evaluate_duca_allocation_candidates \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --backbone-pretrain "${PRETRAIN}" \
  --input-jsonl "${INPUT_JSONL}" \
  --ceiling-jsonl "${CEILING_JSONL}" \
  --ceiling-summary-json "${CEILING_SUMMARY}" \
  --ceiling-validation-json "${VALIDATION_JSON}" \
  --output-jsonl "${CANDIDATE_JSONL}" \
  --summary-json "${CANDIDATE_SUMMARY}" \
  --split train \
  --family-keys \
    A_exact_uniform D_deploy_score D_privileged_gt_ceiling E_privileged_unrestricted_gt \
  --device cuda:0 \
  --batch-size 1 \
  --num-workers 0

"${PYTHON}" -m tools.bata.validate_duca_allocation_candidate_loss_artifact \
  --ceiling-jsonl "${CEILING_JSONL}" \
  --candidate-jsonl "${CANDIDATE_JSONL}" \
  --summary-json "${CANDIDATE_SUMMARY}" \
  --validation-json "${CANDIDATE_VALIDATION}"

"${PYTHON}" -m tools.bata.profile_duca_allocation_solver_cost \
  --input-jsonl "${INPUT_JSONL}" \
  --output-samples-jsonl "${SOLVER_SAMPLES}" \
  --output-summary-json "${SOLVER_SUMMARY}" \
  --score-key transition_policy_scores \
  --cap-policy uniform_reference \
  --warmup-samples 1 \
  --samples 5

"${PYTHON}" -m tools.bata.finalize_duca_allocation_ceiling_gate \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-checkpoint-epoch "${EXPECTED_EPOCH}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}" \
  --pretrain "${PRETRAIN}" \
  --expected-pretrain-sha256 "${EXPECTED_PRETRAIN_SHA256}" \
  --suite-manifest-json "${SUITE_MANIFEST}" \
  --suite-manifest-sha256 "${SUITE_MANIFEST_SHA256}" \
  --ceiling-validation-json "${VALIDATION_JSON}" \
  --gt-runtime-json "${GT_RUNTIME}" \
  --max-projected-gt32-seconds "${MAX_PROJECTED_GT32_SECONDS}" \
  --execution-cluster "${SLURM_CLUSTER_NAME}" \
  --input-jsonl "${INPUT_JSONL}" \
  --ceiling-jsonl "${CEILING_JSONL}" \
  --ceiling-summary-json "${CEILING_SUMMARY}" \
  --candidate-jsonl "${CANDIDATE_JSONL}" \
  --candidate-summary-json "${CANDIDATE_SUMMARY}" \
  --solver-cost-samples-jsonl "${SOLVER_SAMPLES}" \
  --solver-cost-summary-json "${SOLVER_SUMMARY}" \
  --output-json "${GATE_JSON}"

echo "[DUCA_ALLOCATION_GATE] PASS ${GATE_JSON}"
