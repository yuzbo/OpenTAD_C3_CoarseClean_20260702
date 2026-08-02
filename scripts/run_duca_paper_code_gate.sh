#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_CODE_GATE][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_PAPER_REPO_ROOT \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_CODE_GATE_ROOT \
  DUCA_PAPER_PRETRAIN_PATH \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_PATH \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_PATH \
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the paper code gate must run inside Slurm"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ ! -e "${DUCA_PAPER_CODE_GATE_ROOT}" ]] \
  || fail "a fresh code-gate root is required"
cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

for binding in \
  "${DUCA_PAPER_PRETRAIN_PATH}|${DUCA_PAPER_PRETRAIN_SHA256}|VideoMAE initialization" \
  "${DUCA_PAPER_ANNOTATION_PATH}|${DUCA_PAPER_ANNOTATION_SHA256}|THUMOS14 annotation" \
  "${DUCA_PAPER_CLASS_MAP_PATH}|${DUCA_PAPER_CLASS_MAP_SHA256}|THUMOS14 class map"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done

mkdir -p "${DUCA_PAPER_CODE_GATE_ROOT}/logs"
python -m py_compile \
  tools/train.py \
  tools/test.py \
  tools/bata/duca_paper_training.py \
  tools/bata/build_duca_paper_matrix_manifest.py
bash -n \
  scripts/run_duca_paper_code_gate.sh \
  scripts/run_duca_paper_stage_a_cell.sh \
  scripts/run_duca_paper_stage_a_seal.sh \
  scripts/submit_duca_paper_stage_a.sh
python -m pytest \
  tests/test_duca_paper_full200_contract.py \
  tests/test_duca_rime_backbone_mask_contract.py \
  tests/test_duca_protected_e2e_detector_contract.py \
  -q 2>&1 | tee "${DUCA_PAPER_CODE_GATE_ROOT}/logs/pytest.out"

export DUCA_PAPER_MATRIX_MANIFEST="${DUCA_PAPER_CODE_GATE_ROOT}/protocol_manifest.json"
python -m tools.bata.build_duca_paper_matrix_manifest \
  --repo-root "${DUCA_PAPER_REPO_ROOT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --pretrain "${DUCA_PAPER_PRETRAIN_PATH}" \
  --annotation "${DUCA_PAPER_ANNOTATION_PATH}" \
  --class-map "${DUCA_PAPER_CLASS_MAP_PATH}" \
  --output "${DUCA_PAPER_MATRIX_MANIFEST}" \
  > "${DUCA_PAPER_CODE_GATE_ROOT}/logs/manifest.out"
export DUCA_PAPER_MATRIX_MANIFEST_SHA256="$(
  sha256sum "${DUCA_PAPER_MATRIX_MANIFEST}" | awk '{print $1}'
)"

export DUCA_PAPER_ARM=dense
export DUCA_PAPER_CONFIG="${DUCA_PAPER_REPO_ROOT}/configs/adatad/thumos/duca_paper_dense_actionformer_full200.py"
export DUCA_PAPER_CELL_ROOT="${DUCA_PAPER_CODE_GATE_ROOT}/precheck-cell"
export DUCA_PAPER_SEED=5801
PRECHECK_ONLY=1 bash scripts/run_duca_paper_stage_a_cell.sh \
  > "${DUCA_PAPER_CODE_GATE_ROOT}/logs/cell-precheck.out"

printf '%s\n' \
  "schema=duca_paper_code_gate_v1" \
  "status=passed" \
  "commit=${DUCA_PAPER_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "protocol_manifest_sha256=${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" \
  "official_train_video_count=200" \
  "official_evaluation_video_count=211" \
  "stage_a_cell_count=12" \
  > "${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt"
echo "[DUCA_PAPER_CODE_GATE] PASS ${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt"
