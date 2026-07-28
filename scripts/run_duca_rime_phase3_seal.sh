#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_SEAL][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_PHASE3_SEAL_ROOT \
  DUCA_RIME_PHASE3_COST_EVIDENCE \
  DUCA_RIME_PHASE3_COST_EVIDENCE_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-3 sealing must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] || fail "repository is not a Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE3_SEAL_ROOT}" ]] || fail "a fresh Phase-3 seal root is required"
[[ "$(sha256sum "${DUCA_RIME_PHASE2_RECEIPT}" | awk '{print $1}')" == "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" ]] \
  || fail "Phase-2 receipt SHA-256 drift"
[[ "$(sha256sum "${DUCA_RIME_PHASE3_COST_EVIDENCE}" | awk '{print $1}')" == "${DUCA_RIME_PHASE3_COST_EVIDENCE_SHA256}" ]] \
  || fail "Phase-3 cost evidence SHA-256 drift"

arms=(U-fixed U-same-K F-bound D-shuffle D-no-risk AdapTok-TAD RIME-full)
for arm in "${arms[@]}"; do
  key="$(printf '%s' "${arm}" | tr '[:lower:]-' '[:upper:]_')"
  eval_var="DUCA_RIME_PHASE3_${key}_EVAL_ROOT"
  required "${eval_var}"
  eval_root="${!eval_var}"
  [[ -f "${eval_root}/localization_metrics.json" ]] \
    || fail "${arm} localization metrics are missing"
  if [[ "${arm}" != U-fixed ]]; then
    [[ -f "${eval_root}/inference_ledger_summary.json" ]] \
      || fail "${arm} inference ledger summary is missing"
  fi
done

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE3_SEAL] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE3_SEAL_ROOT}/arms"
for arm in "${arms[@]}"; do
  key="$(printf '%s' "${arm}" | tr '[:lower:]-' '[:upper:]_')"
  eval_var="DUCA_RIME_PHASE3_${key}_EVAL_ROOT"
  eval_root="${!eval_var}"
  command=(
    python tools/bata/finalize_duca_rime_phase3_arm.py
    --arm "${arm}"
    --seed 3407
    --localization-metrics "${eval_root}/localization_metrics.json"
    --output "${DUCA_RIME_PHASE3_SEAL_ROOT}/arms/${key}.json"
  )
  if [[ "${arm}" != U-fixed ]]; then
    command+=(--ledger-summary "${eval_root}/inference_ledger_summary.json")
  fi
  if [[ "${arm}" == RIME-full ]]; then
    command+=(--cost-evidence "${DUCA_RIME_PHASE3_COST_EVIDENCE}")
  fi
  "${command[@]}"
done

python - "${DUCA_RIME_PHASE3_SEAL_ROOT}" "${arms[@]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arms = sys.argv[2:]
target = root / "phase3_results.jsonl"
with target.open("x", encoding="utf-8") as handle:
    for arm in arms:
        key = arm.upper().replace("-", "_")
        payload = json.loads((root / "arms" / f"{key}.json").read_text(encoding="utf-8"))
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
PY

python tools/bata/duca_rime_stage_contract.py phase3 \
  --phase2-receipt "${DUCA_RIME_PHASE2_RECEIPT}" \
  --results-jsonl "${DUCA_RIME_PHASE3_SEAL_ROOT}/phase3_results.jsonl" \
  --output "${DUCA_RIME_PHASE3_SEAL_ROOT}/phase3_receipt.json" \
  --expected-seed 3407 \
  --bootstrap-samples "${DUCA_RIME_PHASE3_BOOTSTRAP_SAMPLES:-5000}" \
  --cost-tolerance "${DUCA_RIME_MATCHED_K_TOLERANCE:-1.0}"

python tools/bata/duca_rime_stage_contract.py authorize-phase4 \
  --phase3-receipt "${DUCA_RIME_PHASE3_SEAL_ROOT}/phase3_receipt.json" \
  --output "${DUCA_RIME_PHASE3_SEAL_ROOT}/phase4_authorization.json" \
  --formal-seeds 5801 8123 12011

echo "[DUCA_RIME_PHASE3_SEAL] PASS ${DUCA_RIME_PHASE3_SEAL_ROOT}/phase4_authorization.json"
