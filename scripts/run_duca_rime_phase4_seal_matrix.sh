#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_MATRIX][FAIL] $*" >&2
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
  DUCA_RIME_PHASE4_CELLS_ROOT \
  DUCA_RIME_PHASE4_MATRIX_ROOT; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Phase-4 matrix sealing must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE4_MATRIX_ROOT}" ]] || fail "a fresh Phase-4 matrix root is required"
[[ "$(sha256sum "${DUCA_RIME_PHASE4_AUTHORIZATION}" | awk '{print $1}')" == "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" ]] \
  || fail "Phase-4 authorization SHA-256 drift"

cell_paths=()
for backend in ActionFormer TriDet; do
  for target in 384 192; do
    for seed in 5801 8123 12011; do
      cell="${DUCA_RIME_PHASE4_CELLS_ROOT}/${backend}/K${target}/seed${seed}/cell_result.json"
      [[ -f "${cell}" ]] || fail "missing formal cell: ${cell}"
      cell_paths+=("${cell}")
    done
  done
done

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_PHASE4_MATRIX] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE4_MATRIX_ROOT}"
python - "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_results.jsonl" "${cell_paths[@]}" <<'PY'
import json
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
with target.open("x", encoding="utf-8") as handle:
    for source in sys.argv[2:]:
        payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
PY

python tools/bata/duca_rime_stage_contract.py phase4 \
  --authorization-receipt "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  --results-jsonl "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_results.jsonl" \
  --output "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_receipt.json"

echo "[DUCA_RIME_PHASE4_MATRIX] PASS ${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_receipt.json"
