#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_EVAL][FAIL] $*" >&2
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
  DUCA_RIME_EVAL_PHASE \
  DUCA_RIME_EVAL_ARM \
  DUCA_RIME_EVAL_CONFIG \
  DUCA_RIME_EVAL_ROOT \
  DUCA_RIME_EVAL_SEED \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_TRAINING_RECEIPT \
  DUCA_RIME_TRAINING_RECEIPT_SHA256 \
  DUCA_RIME_CHECKPOINT \
  DUCA_RIME_CHECKPOINT_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "RIME evaluation must run inside Slurm"
[[ "${DUCA_RIME_EVAL_PHASE}" == 3 || "${DUCA_RIME_EVAL_PHASE}" == 4 ]] \
  || fail "RIME evaluation phase must be 3 or 4"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "formal evaluation requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_EVAL_ROOT}" ]] || fail "a fresh evaluation root is required"
[[ -f "${DUCA_RIME_EVAL_CONFIG}" ]] || fail "evaluation config is missing"
check_sha256 \
  "${DUCA_RIME_TRAINING_RECEIPT}" \
  "${DUCA_RIME_TRAINING_RECEIPT_SHA256}" \
  "training receipt"
check_sha256 \
  "${DUCA_RIME_CHECKPOINT}" \
  "${DUCA_RIME_CHECKPOINT_SHA256}" \
  "terminal compact checkpoint"

if [[ "${DUCA_RIME_EVAL_PHASE}" == 4 ]]; then
  for name in \
    DUCA_RIME_PHASE4_AUTHORIZATION \
    DUCA_RIME_PHASE4_AUTHORIZATION_SHA256 \
    DUCA_RIME_TARGET_MEAN_COST; do
    required "${name}"
  done
  check_sha256 \
    "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
    "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" \
    "Phase-4 authorization"
fi

readarray -t config_values < <(python - \
  "${DUCA_RIME_EVAL_CONFIG}" \
  "${DUCA_RIME_EVAL_ARM}" \
  "${DUCA_RIME_EVAL_PHASE}" <<'PY'
import sys

from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
if str(cfg.duca_rime_variant.arm) != sys.argv[2]:
    raise SystemExit("RIME evaluation arm/config mismatch")
phase = int(sys.argv[3])
expected_subset = "training" if phase == 3 else "validation"
if str(cfg.evaluation.subset) != expected_subset:
    raise SystemExit("RIME evaluation subset does not match its phase")
if phase == 3 and not cfg.dataset.test.block_list:
    raise SystemExit("Phase-3 evaluation lacks the development block list")
if phase == 4 and cfg.dataset.test.block_list is not None:
    raise SystemExit("Phase-4 evaluation is not the full official validation set")
print(cfg.evaluation.ground_truth_filename)
print(cfg.dataset.test.class_map)
PY
)
[[ "${#config_values[@]}" == 2 ]] || fail "failed to validate evaluation config"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_EVAL] PRECHECK PASS ${DUCA_RIME_EVAL_ARM}"
  exit 0
fi

mkdir -p "${DUCA_RIME_EVAL_ROOT}/ledger"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_RIME_INFERENCE_LEDGER_ROOT="${DUCA_RIME_EVAL_ROOT}/ledger"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
torchrun --standalone --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_EVAL_CONFIG}" \
  --checkpoint "${DUCA_RIME_CHECKPOINT}" \
  --seed "${DUCA_RIME_EVAL_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${DUCA_RIME_EVAL_ROOT}/terminal_evaluation.json" \
  --cfg-options "work_dir=${DUCA_RIME_EVAL_ROOT}/runtime"

python -m tools.bata.evaluate_duca_rime_predictions \
  --terminal-evaluation "${DUCA_RIME_EVAL_ROOT}/terminal_evaluation.json" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase "${DUCA_RIME_EVAL_PHASE}" \
  --short-max-seconds "${DUCA_RIME_SHORT_MAX_SECONDS}" \
  --medium-max-seconds "${DUCA_RIME_MEDIUM_MAX_SECONDS}" \
  --output "${DUCA_RIME_EVAL_ROOT}/localization_metrics.json"

ledger_shard="${DUCA_RIME_EVAL_ROOT}/ledger/inference_ledger.rank0000.jsonl"
if [[ -f "${ledger_shard}" ]]; then
  case "${DUCA_RIME_EVAL_ARM}" in
    F-bound) expected_ledger_arm="fixed_bound" ;;
    D-shuffle) expected_ledger_arm="dynamic_shuffle" ;;
    D-no-risk) expected_ledger_arm="dynamic_no_risk" ;;
    AdapTok-TAD) expected_ledger_arm="adaptok_tad" ;;
    RIME-full|RIME-full-TriDet) expected_ledger_arm="rime_full" ;;
    U-same-K|U-same-K-TriDet) expected_ledger_arm="uniform_same_k" ;;
    *) fail "unexpected RIME ledger for ${DUCA_RIME_EVAL_ARM}" ;;
  esac
  ledger_command=(
    python
    tools/bata/finalize_duca_rime_inference_ledger.py
    --shard
    "${ledger_shard}"
    --output-jsonl
    "${DUCA_RIME_EVAL_ROOT}/inference_ledger.jsonl"
    --expected-arm
    "${expected_ledger_arm}"
  )
  if [[ -n "${DUCA_RIME_BUDGET_PROTOCOL_SHA256:-}" ]]; then
    ledger_command+=(
      --expected-protocol-sha256
      "${DUCA_RIME_BUDGET_PROTOCOL_SHA256}"
    )
  fi
  "${ledger_command[@]}" > "${DUCA_RIME_EVAL_ROOT}/inference_ledger_summary.json"
elif [[ "${DUCA_RIME_EVAL_ARM}" != U-fixed && "${DUCA_RIME_EVAL_ARM}" != U-fixed-TriDet ]]; then
  fail "RIME selector evaluation did not emit an inference ledger"
fi

printf '%s\n' \
  "schema=duca_rime_evaluation_receipt_v1" \
  "status=passed" \
  "phase=${DUCA_RIME_EVAL_PHASE}" \
  "arm=${DUCA_RIME_EVAL_ARM}" \
  "seed=${DUCA_RIME_EVAL_SEED}" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "terminal_evaluation_sha256=$(sha256sum "${DUCA_RIME_EVAL_ROOT}/terminal_evaluation.json" | awk '{print $1}')" \
  "localization_metrics_sha256=$(sha256sum "${DUCA_RIME_EVAL_ROOT}/localization_metrics.json" | awk '{print $1}')" \
  > "${DUCA_RIME_EVAL_ROOT}/evaluation.receipt"
echo "[DUCA_RIME_EVAL] PASS ${DUCA_RIME_EVAL_ROOT}/terminal_evaluation.json"
