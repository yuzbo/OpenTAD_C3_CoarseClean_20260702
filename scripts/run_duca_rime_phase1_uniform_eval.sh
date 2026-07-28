#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1_UNIFORM][FAIL] $*" >&2
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
  DUCA_RIME_PHASE1_UNIFORM_ROOT \
  DUCA_RIME_PHASE1_UNIFORM_CONFIG \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE1_SPLIT_ROLE \
  DUCA_RIME_FIXED_BUDGET \
  DUCA_RIME_EVAL_SEED \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-1 exact-uniform evaluation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ "${DUCA_RIME_FIXED_BUDGET}" =~ ^(192|384)$ ]] \
  || fail "Phase-1 exact-uniform budget must be 192 or 384"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Phase-1 uniform evaluation requires a complete Git worktree"
[[ ! -e "${DUCA_RIME_PHASE1_UNIFORM_ROOT}" ]] \
  || fail "a fresh Phase-1 uniform root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT}" \
  "${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256}" \
  "Phase-1 exact-uniform checkpoint"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_PHASE1_SPLIT_ROLE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
if role not in manifest["train_roles"]:
    raise SystemExit("Phase-1 exact-uniform split role is not registered")
print(manifest["train_roles"][role]["block_list_path"])
print(manifest["train_roles"][role]["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 2 ]] \
  || fail "failed to resolve the Phase-1 split role"
check_sha256 "${split_values[0]}" "${split_values[1]}" "Phase-1 block list"
export DUCA_RIME_PHASE2_EVAL_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_PHASE1_EVAL_BLOCK_LIST="${split_values[0]}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_PHASE1_UNIFORM_CONFIG}" \
    "${DUCA_RIME_FIXED_BUDGET}" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
budget = int(sys.argv[2])
assert cfg.workflow.formal_protocol == "duca_protected_physical_v1"
assert cfg.duca_rime_baseline_contract.phase == 1
assert cfg.duca_rime_baseline_contract.variant == f"uniform_k{budget}"
assert cfg.duca_rime_baseline_contract.position_policy == "exact_uniform"
assert cfg.duca_rime_baseline_contract.target_mean_cost == float(budget)
assert cfg.duca_rime_baseline_contract.padded_to_kmax is False
assert cfg.evaluation.subset == "training"
PY
  echo "[DUCA_RIME_PHASE1_UNIFORM] PRECHECK PASS"
  exit 0
fi

mkdir -p \
  "${DUCA_RIME_PHASE1_UNIFORM_ROOT}" \
  "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/ledger"
export DUCA_RIME_INFERENCE_LEDGER_ROOT="${DUCA_RIME_PHASE1_UNIFORM_ROOT}/ledger"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}" --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_PHASE1_UNIFORM_CONFIG}" \
  --checkpoint "${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT}" \
  --seed "${DUCA_RIME_EVAL_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json \
  "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/terminal_evaluation.json" \
  --cfg-options "work_dir=${DUCA_RIME_PHASE1_UNIFORM_ROOT}/runtime"

python -m tools.bata.evaluate_duca_rime_predictions \
  --terminal-evaluation \
  "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/terminal_evaluation.json" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase 1 \
  --split-role "${DUCA_RIME_PHASE1_SPLIT_ROLE}" \
  --short-max-seconds "${DUCA_RIME_SHORT_MAX_SECONDS}" \
  --medium-max-seconds "${DUCA_RIME_MEDIUM_MAX_SECONDS}" \
  --output "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/localization_metrics.json"

ledger_shard="${DUCA_RIME_PHASE1_UNIFORM_ROOT}/ledger/inference_ledger.rank0000.jsonl"
[[ -f "${ledger_shard}" ]] \
  || fail "Phase-1 uniform evaluation did not emit its inference ledger"
python tools/bata/finalize_duca_rime_inference_ledger.py \
  --shard "${ledger_shard}" \
  --output-jsonl "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/inference_ledger.jsonl" \
  --expected-arm exact_uniform \
  --summary-json \
  "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/inference_ledger_summary.json" \
  > /dev/null

printf '%s\n' \
  "schema=duca_rime_phase1_uniform_evaluation_receipt_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "split_role=${DUCA_RIME_PHASE1_SPLIT_ROLE}" \
  "budget=${DUCA_RIME_FIXED_BUDGET}" \
  "seed=${DUCA_RIME_EVAL_SEED}" \
  "checkpoint_sha256=${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256}" \
  "terminal_evaluation_sha256=$(sha256sum "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/terminal_evaluation.json" | awk '{print $1}')" \
  "localization_metrics_sha256=$(sha256sum "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/localization_metrics.json" | awk '{print $1}')" \
  "inference_ledger_sha256=$(sha256sum "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/inference_ledger.jsonl" | awk '{print $1}')" \
  "inference_ledger_summary_sha256=$(sha256sum "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/inference_ledger_summary.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE1_UNIFORM_ROOT}/evaluation.receipt"
echo "[DUCA_RIME_PHASE1_UNIFORM] PASS ${DUCA_RIME_PHASE1_UNIFORM_ROOT}"
