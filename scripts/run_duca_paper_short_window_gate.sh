#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_SHORT_WINDOW_GATE][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

for name in \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_CODE_GATE_RECEIPT \
  DUCA_PAPER_CODE_GATE_RECEIPT_SHA256 \
  DUCA_PAPER_SHORT_WINDOW_GATE_JSON \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "gate requires a clean tree"
"${PYTHON}" -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
DUCA_PAPER_SHORT_WINDOW_GATE_JSON="$(
  duca_cellcf_require_external_path \
    "DUCA_PAPER_SHORT_WINDOW_GATE_JSON" \
    "${REPO_ROOT}" \
    "${BASE}" \
    "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}"
)" || fail "gate receipt path violates the formal path contract"
[[ ! -e "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" ]] || fail "refusing to overwrite gate evidence"
mkdir -p "$(dirname "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}")"

"${PYTHON}" -m tools.bata.run_duca_paper_short_window_gate \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --code-gate-receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --code-gate-receipt-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  --pretrain "${ADATAD_PRETRAIN_PATH}" \
  --pretrain-sha256 "${DUCA_PAPER_PRETRAIN_SHA256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --annotation-sha256 "${DUCA_PAPER_ANNOTATION_SHA256}" \
  --class-map "${THUMOS14_CLASS_MAP}" \
  --class-map-sha256 "${DUCA_PAPER_CLASS_MAP_SHA256}" \
  --train-data-path "${THUMOS14_TRAIN_DATA_PATH}" \
  --output-json "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}"

receipt_sha256="$(sha256sum "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${receipt_sha256}"
