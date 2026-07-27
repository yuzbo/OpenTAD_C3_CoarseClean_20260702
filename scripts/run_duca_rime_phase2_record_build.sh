#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_RECORDS][FAIL] $*" >&2
  exit 1
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_O1_SOURCE_MANIFEST \
  DUCA_RIME_O2_SOURCE_MANIFEST \
  DUCA_RIME_O3_SOURCE_JSONL \
  DUCA_RIME_O4_SOURCE_JSONL \
  DUCA_RIME_PRICE_SOURCE_JSONL \
  DUCA_RIME_PHASE2_RECORD_ROOT; do
  [[ -n "${!name:-}" ]] || fail "${name} is required"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-2 record sealing must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE2_RECORD_ROOT}" ]] || fail "a fresh Phase-2 record root is required"
for source in \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_O1_SOURCE_MANIFEST}" \
  "${DUCA_RIME_O2_SOURCE_MANIFEST}" \
  "${DUCA_RIME_O3_SOURCE_JSONL}" \
  "${DUCA_RIME_O4_SOURCE_JSONL}" \
  "${DUCA_RIME_PRICE_SOURCE_JSONL}"; do
  [[ -f "${source}" ]] || fail "Phase-2 source is missing: ${source}"
done

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE2_RECORDS] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE2_RECORD_ROOT}"
python tools/bata/build_duca_rime_gate_records.py o1 \
  --source-manifest "${DUCA_RIME_O1_SOURCE_MANIFEST}" \
  --output "${DUCA_RIME_PHASE2_RECORD_ROOT}/o1_records.jsonl"
python tools/bata/build_duca_rime_gate_records.py o2 \
  --source-manifest "${DUCA_RIME_O2_SOURCE_MANIFEST}" \
  --output "${DUCA_RIME_PHASE2_RECORD_ROOT}/o2_records.jsonl"
for kind in o3 o4 price; do
  source_var="DUCA_RIME_${kind^^}_SOURCE_JSONL"
  python tools/bata/build_duca_rime_gate_records.py "${kind}" \
    --source-jsonl "${!source_var}" \
    --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
    --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
    --output "${DUCA_RIME_PHASE2_RECORD_ROOT}/${kind}_records.jsonl"
done

echo "[DUCA_RIME_PHASE2_RECORDS] PASS ${DUCA_RIME_PHASE2_RECORD_ROOT}"
