#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE0][FAIL] $*" >&2
  exit 1
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE0_SOURCE_MANIFEST \
  DUCA_RIME_PHASE0_ROOT; do
  [[ -n "${!name:-}" ]] || fail "${name} is required"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-0 sealing must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE0_ROOT}" ]] || fail "a fresh Phase-0 root is required"
[[ -f "${DUCA_RIME_PHASE0_SOURCE_MANIFEST}" ]] || fail "Phase-0 source manifest is missing"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE0] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE0_ROOT}"
python tools/bata/build_duca_rime_gate_records.py phase0 \
  --source-manifest "${DUCA_RIME_PHASE0_SOURCE_MANIFEST}" \
  --output "${DUCA_RIME_PHASE0_ROOT}/phase0_measurements.jsonl" \
  --primary-metric "${DUCA_RIME_PHASE0_PRIMARY_METRIC:-avg_map}"
python tools/bata/duca_rime_phase2.py phase0 \
  --records-jsonl "${DUCA_RIME_PHASE0_ROOT}/phase0_measurements.jsonl" \
  --output "${DUCA_RIME_PHASE0_ROOT}/phase0_summary.json" \
  --primary-metric "${DUCA_RIME_PHASE0_PRIMARY_METRIC:-avg_map}" \
  --alpha "${DUCA_RIME_PHASE0_ALPHA:-0.05}" \
  --power "${DUCA_RIME_PHASE0_POWER:-0.80}"

echo "[DUCA_RIME_PHASE0] PASS ${DUCA_RIME_PHASE0_ROOT}/phase0_summary.json"
