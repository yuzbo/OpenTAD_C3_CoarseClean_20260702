#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_COST][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_COST_PHASE \
  DUCA_RIME_COST_ARM \
  DUCA_RIME_COST_BACKEND \
  DUCA_RIME_COST_TARGET \
  DUCA_RIME_COST_SEED \
  DUCA_RIME_COST_ROOT \
  DUCA_RIME_CANDIDATE_CONFIG \
  DUCA_RIME_CANDIDATE_CHECKPOINT \
  DUCA_RIME_CANDIDATE_CHECKPOINT_SHA256 \
  DUCA_RIME_CANDIDATE_TRAINING_RECEIPT \
  DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_FIXED_CONFIG \
  DUCA_RIME_FIXED_CHECKPOINT \
  DUCA_RIME_FIXED_CHECKPOINT_SHA256 \
  DUCA_RIME_FIXED_TRAINING_RECEIPT \
  DUCA_RIME_FIXED_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_REPLAY_JSONL \
  DUCA_RIME_REPLAY_SHA256 \
  DUCA_RIME_DENSE_CONFIG \
  DUCA_RIME_DENSE_CHECKPOINT \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256 \
  DUCA_RIME_DENSE_TRAINED_COMMIT; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "full-stack cost measurement must run inside Slurm"
[[ "${DUCA_RIME_COST_PHASE}" == 3 || "${DUCA_RIME_COST_PHASE}" == 4 ]] \
  || fail "cost phase must be 3 or 4"
[[ "${DUCA_RIME_COST_BACKEND}" == ActionFormer || "${DUCA_RIME_COST_BACKEND}" == TriDet ]] \
  || fail "unsupported detector backend"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact RIME commit is required"
[[ "${DUCA_RIME_DENSE_TRAINED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact dense trained commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] || fail "RIME repository is not a Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "RIME Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "RIME Git tree is dirty"
[[ ! -e "${DUCA_RIME_COST_ROOT}" ]] || fail "a fresh cost root is required"

check_sha256 \
  "${DUCA_RIME_CANDIDATE_CHECKPOINT}" \
  "${DUCA_RIME_CANDIDATE_CHECKPOINT_SHA256}" \
  "candidate checkpoint"
check_sha256 \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT}" \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256}" \
  "candidate training receipt"
check_sha256 \
  "${DUCA_RIME_FIXED_CHECKPOINT}" \
  "${DUCA_RIME_FIXED_CHECKPOINT_SHA256}" \
  "fixed checkpoint"
check_sha256 \
  "${DUCA_RIME_FIXED_TRAINING_RECEIPT}" \
  "${DUCA_RIME_FIXED_TRAINING_RECEIPT_SHA256}" \
  "matched U-same-K source training receipt"
check_sha256 \
  "${DUCA_RIME_REPLAY_JSONL}" \
  "${DUCA_RIME_REPLAY_SHA256}" \
  "matched U-same-K replay"
check_sha256 \
  "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE}" \
  "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256}" \
  "dense checkpoint evidence"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_COST] PRECHECK PASS ${DUCA_RIME_COST_BACKEND} K${DUCA_RIME_COST_TARGET} seed${DUCA_RIME_COST_SEED}"
  exit 0
fi

mkdir -p "${DUCA_RIME_COST_ROOT}"
session="rime-${DUCA_RIME_COST_PHASE}-${DUCA_RIME_COST_BACKEND}-k${DUCA_RIME_COST_TARGET}-s${DUCA_RIME_COST_SEED}-${SLURM_JOB_ID}"
samples="${DUCA_RIME_COST_SAMPLES:-30}"
warmup="${DUCA_RIME_COST_WARMUP:-5}"
candidate_method="duca-rime-phase${DUCA_RIME_COST_PHASE}-${DUCA_RIME_COST_BACKEND}-${DUCA_RIME_COST_ARM}-k${DUCA_RIME_COST_TARGET}-s${DUCA_RIME_COST_SEED}"
fixed_arm="U-same-K"
[[ "${DUCA_RIME_COST_BACKEND}" == TriDet ]] && fixed_arm="U-same-K-TriDet"
fixed_method="duca-rime-phase${DUCA_RIME_COST_PHASE}-${DUCA_RIME_COST_BACKEND}-${fixed_arm}-k${DUCA_RIME_COST_TARGET}-s${DUCA_RIME_COST_SEED}"
power_args=()
if [[ "${DUCA_RIME_COST_PHASE}" == 4 ]]; then
  power_args+=(--sample-power)
fi

profile_rime() {
  local config="$1" checkpoint="$2" receipt="$3" receipt_sha="$4"
  local arm="$5" method="$6" pair="$7" order="$8" prefix="$9"
  local replay_mode="${10}"
  local env_command=(env)
  if [[ "${replay_mode}" == without_replay ]]; then
    env_command+=(-u DUCA_RIME_REPLAY_JSONL -u DUCA_RIME_REPLAY_SHA256)
  elif [[ "${replay_mode}" != with_replay ]]; then
    fail "unknown cost replay mode: ${replay_mode}"
  fi
  "${env_command[@]}" python tools/bata/profile_duca_full_stack_cost.py \
    "${config}" \
    --checkpoint "${checkpoint}" \
    --output-prefix "${prefix}" \
    --method-name "${method}" \
    --config-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --trained-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --evidence-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --device cuda:0 \
    --samples "${samples}" \
    --warmup-samples "${warmup}" \
    --loader-workers 0 \
    --batch-size 1 \
    --amp \
    --use-ema \
    --rime-training-receipt "${receipt}" \
    --rime-training-receipt-sha256 "${receipt_sha}" \
    --rime-evaluation-arm "${arm}" \
    --rime-seed "${DUCA_RIME_COST_SEED}" \
    --profile-session-id "${session}" \
    --profile-pair-id "${pair}" \
    --profile-repeat-index 1 \
    --profile-order-position "${order}" \
    "${power_args[@]}"
}

profile_rime \
  "${DUCA_RIME_CANDIDATE_CONFIG}" \
  "${DUCA_RIME_CANDIDATE_CHECKPOINT}" \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT}" \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256}" \
  "${DUCA_RIME_COST_ARM}" \
  "${candidate_method}" \
  "${session}-fixed" \
  1 \
  "${DUCA_RIME_COST_ROOT}/candidate_fixed" \
  without_replay
profile_rime \
  "${DUCA_RIME_FIXED_CONFIG}" \
  "${DUCA_RIME_FIXED_CHECKPOINT}" \
  "${DUCA_RIME_FIXED_TRAINING_RECEIPT}" \
  "${DUCA_RIME_FIXED_TRAINING_RECEIPT_SHA256}" \
  "${fixed_arm}" \
  "${fixed_method}" \
  "${session}-fixed" \
  2 \
  "${DUCA_RIME_COST_ROOT}/fixed" \
  with_replay

python tools/bata/profile_duca_full_stack_cost.py \
  "${DUCA_RIME_DENSE_CONFIG}" \
  --checkpoint "${DUCA_RIME_DENSE_CHECKPOINT}" \
  --output-prefix "${DUCA_RIME_COST_ROOT}/dense" \
  --method-name dense-adatad \
  --config-commit "${DUCA_RIME_DENSE_TRAINED_COMMIT}" \
  --trained-commit "${DUCA_RIME_DENSE_TRAINED_COMMIT}" \
  --evidence-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --device cuda:0 \
  --samples "${samples}" \
  --warmup-samples "${warmup}" \
  --loader-workers 0 \
  --batch-size 1 \
  --amp \
  --use-ema \
  --checkpoint-evidence "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE}" \
  --checkpoint-evidence-sha256 "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256}" \
  --profile-session-id "${session}" \
  --profile-pair-id "${session}-dense" \
  --profile-repeat-index 1 \
  --profile-order-position 1 \
  "${power_args[@]}"
profile_rime \
  "${DUCA_RIME_CANDIDATE_CONFIG}" \
  "${DUCA_RIME_CANDIDATE_CHECKPOINT}" \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT}" \
  "${DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256}" \
  "${DUCA_RIME_COST_ARM}" \
  "${candidate_method}" \
  "${session}-dense" \
  2 \
  "${DUCA_RIME_COST_ROOT}/candidate_dense" \
  without_replay

python tools/bata/finalize_duca_rime_cost.py \
  --candidate-fixed-profile "${DUCA_RIME_COST_ROOT}/candidate_fixed.summary.json" \
  --fixed-profile "${DUCA_RIME_COST_ROOT}/fixed.summary.json" \
  --candidate-dense-profile "${DUCA_RIME_COST_ROOT}/candidate_dense.summary.json" \
  --dense-profile "${DUCA_RIME_COST_ROOT}/dense.summary.json" \
  --output "${DUCA_RIME_COST_ROOT}/paired_cost.json" \
  --expected-phase "${DUCA_RIME_COST_PHASE}" \
  --expected-arm "${DUCA_RIME_COST_ARM}" \
  --expected-seed "${DUCA_RIME_COST_SEED}" \
  --expected-backend "${DUCA_RIME_COST_BACKEND}" \
  --expected-target-mean-cost "${DUCA_RIME_COST_TARGET}" \
  --matched-k-tolerance "${DUCA_RIME_MATCHED_K_TOLERANCE:-1.0}"

echo "[DUCA_RIME_COST] PASS ${DUCA_RIME_COST_ROOT}/paired_cost.json"
