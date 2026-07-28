#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_COST_PIPELINE][FAIL] $*" >&2
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
  DUCA_RIME_DENSE_CONFIG_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER \
  DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-3 cost pipeline must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

rime_root="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/RIME_FULL/train"
rime_receipt="${rime_root}/training_receipt.json"
rime_checkpoint="${rime_root}/train/gpu1_id0/checkpoint/terminal_ema.pth"
replay="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/U_SAME_K/replay/paired_replay.jsonl"
for path in "${rime_receipt}" "${rime_checkpoint}" "${replay}"; do
  [[ -f "${path}" ]] || fail "required training output is missing: ${path}"
done
export DUCA_RIME_REPLAY_JSONL="${replay}"
export DUCA_RIME_REPLAY_SHA256="$(sha256sum "${replay}" | awk '{print $1}')"

export DUCA_RIME_COST_PHASE=3
export DUCA_RIME_COST_ARM=RIME-full
export DUCA_RIME_COST_BACKEND=ActionFormer
export DUCA_RIME_COST_TARGET=384
export DUCA_RIME_COST_SEED=3407
export DUCA_RIME_COST_ROOT="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/cost"
export DUCA_RIME_CANDIDATE_CONFIG="configs/adatad/thumos/duca_rime_full_total60.py"
export DUCA_RIME_CANDIDATE_CHECKPOINT="${rime_checkpoint}"
export DUCA_RIME_CANDIDATE_CHECKPOINT_SHA256="$(sha256sum "${rime_checkpoint}" | awk '{print $1}')"
export DUCA_RIME_CANDIDATE_TRAINING_RECEIPT="${rime_receipt}"
export DUCA_RIME_CANDIDATE_TRAINING_RECEIPT_SHA256="$(sha256sum "${rime_receipt}" | awk '{print $1}')"
export DUCA_RIME_FIXED_CONFIG="configs/adatad/thumos/duca_rime_uniform_same_k_eval.py"
export DUCA_RIME_FIXED_CHECKPOINT="${rime_checkpoint}"
export DUCA_RIME_FIXED_CHECKPOINT_SHA256="$(sha256sum "${rime_checkpoint}" | awk '{print $1}')"
export DUCA_RIME_FIXED_TRAINING_RECEIPT="${rime_receipt}"
export DUCA_RIME_FIXED_TRAINING_RECEIPT_SHA256="$(sha256sum "${rime_receipt}" | awk '{print $1}')"
export DUCA_RIME_DENSE_CONFIG="${DUCA_RIME_DENSE_CONFIG_ACTIONFORMER}"
export DUCA_RIME_DENSE_CHECKPOINT="${DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER}"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER}"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256="${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER}"
export DUCA_RIME_DENSE_TRAINED_COMMIT="${DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER}"
scripts/run_duca_rime_cost_cell.sh

echo "[DUCA_RIME_PHASE3_COST_PIPELINE] PASS ${DUCA_RIME_COST_ROOT}/paired_cost.json"
