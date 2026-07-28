#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[HRIME_STAGE1_FINALIZE][FAIL] $*" >&2
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
  HRIME_STAGE1_PLAN_MANIFEST \
  HRIME_STAGE1_PLAN_MANIFEST_SHA256 \
  HRIME_STAGE1_PREREGISTRATION \
  HRIME_STAGE1_PREREGISTRATION_SHA256 \
  HRIME_STAGE1_EVALUATION_ROOT \
  HRIME_STAGE1_ORACLE_RECEIPT; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Stage-1 finalization must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Stage-1 finalization requires a complete exact Git worktree"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${HRIME_STAGE1_ORACLE_RECEIPT}" ]] \
  || fail "Stage-1 oracle receipt is immutable and already exists"
check_sha256 \
  "${HRIME_STAGE1_PLAN_MANIFEST}" \
  "${HRIME_STAGE1_PLAN_MANIFEST_SHA256}" \
  "Stage-1 plan manifest"
check_sha256 \
  "${HRIME_STAGE1_PREREGISTRATION}" \
  "${HRIME_STAGE1_PREREGISTRATION_SHA256}" \
  "Stage-1 preregistration"

mapfile -t expected_cells < <(
  python - "${HRIME_STAGE1_PLAN_MANIFEST}" "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    plan.get("schema_version") != "hrime_stage1_oracle_plan_v1"
    or plan.get("status") != "planned"
    or plan.get("git_commit") != sys.argv[2]
    or plan.get("uses_official_final") is not False
    or plan.get("authorizes_stage2_training") is not False
):
    raise SystemExit("Stage-1 plan contract drift")
for strategy in plan["strategies"]:
    for anchor in plan["anchor_nominal_budgets"]:
        print(f"{strategy}:{int(anchor)}")
PY
)
[[ "${#expected_cells[@]}" -gt 0 ]] || fail "Stage-1 plan has no cells"

finalize_command=(
  python
  -m
  tools.bata.hrime_stage1_oracle
  finalize
  --plan-manifest
  "${HRIME_STAGE1_PLAN_MANIFEST}"
  --plan-manifest-sha256
  "${HRIME_STAGE1_PLAN_MANIFEST_SHA256}"
  --preregistration
  "${HRIME_STAGE1_PREREGISTRATION}"
  --preregistration-sha256
  "${HRIME_STAGE1_PREREGISTRATION_SHA256}"
)
for cell in "${expected_cells[@]}"; do
  strategy="${cell%%:*}"
  anchor="${cell##*:}"
  receipt="${HRIME_STAGE1_EVALUATION_ROOT}/${strategy}/k${anchor}/execution_receipt.json"
  [[ -f "${receipt}" ]] || fail "missing Stage-1 execution receipt: ${receipt}"
  receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
  finalize_command+=(
    --execution
    "${strategy}:${anchor}:${receipt}:${receipt_sha}"
  )
done
finalize_command+=(--output-receipt "${HRIME_STAGE1_ORACLE_RECEIPT}")

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[HRIME_STAGE1_FINALIZE] PRECHECK PASS ${#expected_cells[@]} cells"
  exit 0
fi

"${finalize_command[@]}"
readarray -t terminal_values < <(
  python - "${HRIME_STAGE1_ORACLE_RECEIPT}" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    receipt.get("schema_version") != "hrime_stage1_oracle_receipt_v1"
    or receipt.get("execution_matrix_complete") is not True
    or receipt.get("uses_official_final") is not False
    or receipt.get("claim_scope")
    != "complete_development_oracle_not_paper_result"
):
    raise SystemExit("Stage-1 terminal oracle receipt drift")
print(receipt["status"])
print(str(receipt["authorizes_stage2_training"]).lower())
PY
)
[[ "${#terminal_values[@]}" == 2 ]] || fail "failed to read Stage-1 terminal status"
echo "[HRIME_STAGE1_FINALIZE] ${terminal_values[0]} authorizes_stage2_training=${terminal_values[1]}"
