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
  python - "${DUCA_RIME_PHASE4_AUTHORIZATION}" "${cell_paths[@]}" <<'PY'
import json
import pathlib
import sys
import tempfile

from tools.bata.duca_rime_stage_contract import seal_phase4

with tempfile.TemporaryDirectory(prefix="duca-rime-phase4-precheck-") as directory:
    root = pathlib.Path(directory)
    results = root / "phase4_results.jsonl"
    with results.open("x", encoding="utf-8") as handle:
        for source in sys.argv[2:]:
            payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
    seal_phase4(
        authorization_receipt=sys.argv[1],
        results_jsonl=results,
        output=root / "phase4_receipt.json",
    )
PY
  echo "[DUCA_RIME_PHASE4_MATRIX] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE4_MATRIX_ROOT}"
python - "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_results.jsonl" "${cell_paths[@]}" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    for source in sys.argv[2:]:
        payload = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

receipt_tmp="${DUCA_RIME_PHASE4_MATRIX_ROOT}/.phase4_receipt.json.partial.$$"
python tools/bata/duca_rime_stage_contract.py phase4 \
  --authorization-receipt "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  --results-jsonl "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_results.jsonl" \
  --output "${receipt_tmp}"
mv "${receipt_tmp}" "${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_receipt.json"

python - "${DUCA_RIME_PHASE4_MATRIX_ROOT}" "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
job_id = sys.argv[2]
results = root / "phase4_results.jsonl"
receipt = root / "phase4_receipt.json"
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
results_sha = sha(results)
receipt_sha = sha(receipt)

def atomic_write(path, content):
    path = pathlib.Path(path)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

atomic_write(
    root / "phase4_results.jsonl.sha256",
    f"{results_sha}  {results.name}\n",
)
atomic_write(
    root / "phase4_receipt.json.sha256",
    f"{receipt_sha}  {receipt.name}\n",
)
seal = {
    "schema_version": "duca_rime_phase4_matrix_seal_v1",
    "status": "sealed",
    "slurm_job_id": job_id,
    "results_path": str(results.resolve()),
    "results_sha256": results_sha,
    "stage_receipt_path": str(receipt.resolve()),
    "stage_receipt_sha256": receipt_sha,
}
atomic_write(
    root / "matrix_seal.json",
    json.dumps(seal, indent=2, sort_keys=True) + "\n",
)
PY

echo "[DUCA_RIME_PHASE4_MATRIX] PASS ${DUCA_RIME_PHASE4_MATRIX_ROOT}/phase4_receipt.json"
