#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_CELL][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE4_AUTHORIZATION \
  DUCA_RIME_PHASE4_AUTHORIZATION_SHA256 \
  DUCA_RIME_PHASE4_BACKEND \
  DUCA_RIME_PHASE4_TARGET \
  DUCA_RIME_PHASE4_SEED \
  DUCA_RIME_PHASE4_RIME_EVAL_ROOT \
  DUCA_RIME_PHASE4_FIXED_EVAL_ROOT \
  DUCA_RIME_PHASE4_SAME_K_EVAL_ROOT \
  DUCA_RIME_PHASE4_COST_EVIDENCE \
  DUCA_RIME_PHASE4_COST_EVIDENCE_SHA256 \
  DUCA_RIME_PHASE4_CELL_ROOT; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-4 cell sealing must run inside Slurm"
[[ "${DUCA_RIME_PHASE4_BACKEND}" == ActionFormer || "${DUCA_RIME_PHASE4_BACKEND}" == TriDet ]] \
  || fail "invalid detector backend"
[[ "${DUCA_RIME_PHASE4_TARGET}" == 384 || "${DUCA_RIME_PHASE4_TARGET}" == 192 ]] \
  || fail "invalid formal budget panel"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] || fail "repository is not a Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE4_CELL_ROOT}" ]] || fail "a fresh Phase-4 cell root is required"
[[ "$(sha256sum "${DUCA_RIME_PHASE4_AUTHORIZATION}" | awk '{print $1}')" == "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" ]] \
  || fail "Phase-4 authorization SHA-256 drift"
[[ "$(sha256sum "${DUCA_RIME_PHASE4_COST_EVIDENCE}" | awk '{print $1}')" == "${DUCA_RIME_PHASE4_COST_EVIDENCE_SHA256}" ]] \
  || fail "Phase-4 cost SHA-256 drift"
for root in \
  "${DUCA_RIME_PHASE4_RIME_EVAL_ROOT}" \
  "${DUCA_RIME_PHASE4_FIXED_EVAL_ROOT}" \
  "${DUCA_RIME_PHASE4_SAME_K_EVAL_ROOT}"; do
  [[ -f "${root}/localization_metrics.json" ]] \
    || fail "Phase-4 localization metrics are missing under ${root}"
done
[[ -f "${DUCA_RIME_PHASE4_RIME_EVAL_ROOT}/inference_ledger_summary.json" ]] \
  || fail "RIME-full inference ledger summary is missing"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE4_CELL] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE4_CELL_ROOT}"
python tools/bata/bootstrap_duca_rime_phase4.py \
  --rime-metrics "${DUCA_RIME_PHASE4_RIME_EVAL_ROOT}/localization_metrics.json" \
  --fixed-metrics "${DUCA_RIME_PHASE4_FIXED_EVAL_ROOT}/localization_metrics.json" \
  --same-k-metrics "${DUCA_RIME_PHASE4_SAME_K_EVAL_ROOT}/localization_metrics.json" \
  --output "${DUCA_RIME_PHASE4_CELL_ROOT}/comparisons.json" \
  --bootstrap-samples "${DUCA_RIME_PHASE4_BOOTSTRAP_SAMPLES:-1000}" \
  --seed "${DUCA_RIME_PHASE4_SEED}" \
  --workers "${DUCA_RIME_PHASE4_BOOTSTRAP_WORKERS:-1}"

python tools/bata/finalize_duca_rime_phase4_cell.py \
  --authorization-receipt "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  --rime-metrics "${DUCA_RIME_PHASE4_RIME_EVAL_ROOT}/localization_metrics.json" \
  --fixed-metrics "${DUCA_RIME_PHASE4_FIXED_EVAL_ROOT}/localization_metrics.json" \
  --same-k-metrics "${DUCA_RIME_PHASE4_SAME_K_EVAL_ROOT}/localization_metrics.json" \
  --comparisons "${DUCA_RIME_PHASE4_CELL_ROOT}/comparisons.json" \
  --cost-evidence "${DUCA_RIME_PHASE4_COST_EVIDENCE}" \
  --rime-ledger-summary "${DUCA_RIME_PHASE4_RIME_EVAL_ROOT}/inference_ledger_summary.json" \
  --output "${DUCA_RIME_PHASE4_CELL_ROOT}/cell_result.json" \
  --detector-backend "${DUCA_RIME_PHASE4_BACKEND}" \
  --target-mean-cost "${DUCA_RIME_PHASE4_TARGET}" \
  --seed "${DUCA_RIME_PHASE4_SEED}"

echo "[DUCA_RIME_PHASE4_CELL] PASS ${DUCA_RIME_PHASE4_CELL_ROOT}/cell_result.json"
