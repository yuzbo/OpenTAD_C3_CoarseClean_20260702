#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_ARM_PIPELINE][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE3_BUNDLE_ROOT \
  DUCA_RIME_PHASE3_ARM \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON \
  DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-3 arm pipeline must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

case "${DUCA_RIME_PHASE3_ARM}" in
  U-fixed)
    config="configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py"
    ;;
  F-bound)
    config="configs/adatad/thumos/duca_rime_fixed_bound_total60.py"
    ;;
  D-shuffle)
    config="configs/adatad/thumos/duca_rime_dynamic_shuffle_total60.py"
    required DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL
    required DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256
    ;;
  D-no-risk)
    config="configs/adatad/thumos/duca_rime_dynamic_no_risk_total60.py"
    ;;
  AdapTok-TAD)
    config="configs/adatad/thumos/duca_adaptok_tad_direct_total60.py"
    required DUCA_RIME_ADAPTOK_REPLAY_JSONL
    required DUCA_RIME_ADAPTOK_REPLAY_SHA256
    export DUCA_RIME_REPLAY_JSONL="${DUCA_RIME_ADAPTOK_REPLAY_JSONL}"
    export DUCA_RIME_REPLAY_SHA256="${DUCA_RIME_ADAPTOK_REPLAY_SHA256}"
    ;;
  RIME-full)
    config="configs/adatad/thumos/duca_rime_full_total60.py"
    ;;
  U-same-K)
    fail "U-same-K is evaluation-only and has no training pipeline"
    ;;
  *)
    fail "unregistered Phase-3 arm: ${DUCA_RIME_PHASE3_ARM}"
    ;;
esac

key="$(printf '%s' "${DUCA_RIME_PHASE3_ARM}" | tr '[:lower:]-' '[:upper:]_')"
arm_root="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/${key}"
train_root="${arm_root}/train"
eval_root="${arm_root}/eval"
[[ ! -e "${arm_root}" ]] || fail "a fresh Phase-3 arm root is required"

IFS=',' read -r -a candidate_budgets <<<"${DUCA_RIME_CANDIDATE_BUDGETS}"
if [[ "${DUCA_RIME_PHASE3_ARM}" == D-shuffle ]]; then
  [[ "$(sha256sum "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL}" | awk '{print $1}')" == \
    "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256}" ]] \
    || fail "D-shuffle training replay SHA-256 drift"
  rime_ledger="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/RIME_FULL/eval/inference_ledger.jsonl"
  [[ -f "${rime_ledger}" ]] \
    || fail "D-shuffle requires the completed RIME-full development ledger"
  replay_root="${arm_root}/replay"
  mkdir -p "${replay_root}"
  python tools/bata/build_duca_rime_budget_replay.py \
    --mode shuffle \
    --input-jsonl "${rime_ledger}" \
    --output-jsonl "${replay_root}/development_shuffle.jsonl" \
    --candidate-budgets "${candidate_budgets[@]}" \
    --seed 3407
  python tools/bata/build_duca_rime_budget_replay.py \
    --mode merge \
    --input-jsonl "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL}" \
    --additional-input-jsonl "${replay_root}/development_shuffle.jsonl" \
    --output-jsonl "${replay_root}/combined_replay.jsonl"
  export DUCA_RIME_REPLAY_JSONL="${replay_root}/combined_replay.jsonl"
  export DUCA_RIME_REPLAY_SHA256="$(
    sha256sum "${DUCA_RIME_REPLAY_JSONL}" | awk '{print $1}'
  )"
fi

export DUCA_RIME_PHASE3_CONFIG="${config}"
export DUCA_RIME_PHASE3_ROOT="${train_root}"
export DUCA_RIME_PHASE3_SEED=3407
export DUCA_RIME_TRAINING_EXPOSURE_JSON="${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON}"
export DUCA_RIME_TRAINING_EXPOSURE_SHA256="${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256}"
export DUCA_RIME_FIXED_BUDGET=384
scripts/run_duca_rime_phase3_train_arm.sh

receipt="${train_root}/training_receipt.json"
checkpoint="${train_root}/train/gpu1_id0/checkpoint/terminal_ema.pth"
[[ -f "${receipt}" && -f "${checkpoint}" ]] \
  || fail "Phase-3 arm training did not emit its terminal artifacts"

export DUCA_RIME_EVAL_PHASE=3
export DUCA_RIME_EVAL_ARM="${DUCA_RIME_PHASE3_ARM}"
export DUCA_RIME_EVAL_CONFIG="${config}"
export DUCA_RIME_EVAL_ROOT="${eval_root}"
export DUCA_RIME_EVAL_SEED=3407
export DUCA_RIME_TRAINING_RECEIPT="${receipt}"
export DUCA_RIME_TRAINING_RECEIPT_SHA256="$(sha256sum "${receipt}" | awk '{print $1}')"
export DUCA_RIME_CHECKPOINT="${checkpoint}"
export DUCA_RIME_CHECKPOINT_SHA256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
scripts/run_duca_rime_evaluate_arm.sh

if [[ "${DUCA_RIME_PHASE3_ARM}" == RIME-full ]]; then
  same_root="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/U_SAME_K/eval"
  [[ ! -e "${same_root}" ]] \
    || fail "a fresh shared U-same-K evaluation root is required"
  same_replay_root="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/U_SAME_K/replay"
  [[ ! -e "${same_replay_root}" ]] \
    || fail "a fresh U-same-K replay root is required"
  mkdir -p "${same_replay_root}"
  python tools/bata/build_duca_rime_budget_replay.py \
    --mode paired \
    --input-jsonl "${eval_root}/inference_ledger.jsonl" \
    --output-jsonl "${same_replay_root}/paired_replay.jsonl" \
    --candidate-budgets "${candidate_budgets[@]}"
  export DUCA_RIME_REPLAY_JSONL="${same_replay_root}/paired_replay.jsonl"
  export DUCA_RIME_REPLAY_SHA256="$(
    sha256sum "${DUCA_RIME_REPLAY_JSONL}" | awk '{print $1}'
  )"
  export DUCA_RIME_EVAL_ARM=U-same-K
  export DUCA_RIME_EVAL_CONFIG="configs/adatad/thumos/duca_rime_uniform_same_k_eval.py"
  export DUCA_RIME_EVAL_ROOT="${same_root}"
  scripts/run_duca_rime_evaluate_arm.sh
fi

echo "[DUCA_RIME_PHASE3_ARM_PIPELINE] PASS ${DUCA_RIME_PHASE3_ARM}"
