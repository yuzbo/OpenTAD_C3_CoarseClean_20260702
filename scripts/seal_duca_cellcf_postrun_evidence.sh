#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
umask 077

fail() {
  echo "[DUCA_CELLCF_POSTRUN_SEAL][FAIL] $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

EVIDENCE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EVIDENCE_REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${EVIDENCE_REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
RUN_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-}"
CONTROL_ROOT="${DUCA_CELLCF_POSTRUN_CONTROL_ROOT:-}"
TRAINED_REPO_ROOT="${DUCA_CELLCF_TRAINED_REPO_ROOT:-}"
TRAINED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
EVIDENCE_COMMIT="$(git rev-parse HEAD)"
EXPECTED_EVIDENCE_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:-}"
EXPECTED_AGGREGATE_SHA256="${DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256:-}"
SUPPORTED_TRAINED_COMMIT="1642f265e48391418a7c8a4a087e33e2b7bf6899"

[[ -z "${SLURM_JOB_ID:-}" ]] \
  || fail "the final seal must be written by an observer outside the post-run DAG"
[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -d "${RUN_ROOT}" ]] || fail "formal run root is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${EVIDENCE_REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "formal run root violates the path contract"
[[ -d "${CONTROL_ROOT}" ]] || fail "post-run control root is missing"
CONTROL_ROOT="$(realpath -e -- "${CONTROL_ROOT}")"
case "${CONTROL_ROOT}/" in
  "${RUN_ROOT}/postrun_submission_"*/) ;;
  *) fail "post-run control root is not a versioned child of the formal run" ;;
esac
[[ -d "${TRAINED_REPO_ROOT}" ]] || fail "trained repository is missing"
[[ "${TRAINED_COMMIT}" == "${SUPPORTED_TRAINED_COMMIT}" ]] \
  || fail "unsupported trained commit for this frozen post-run protocol"
[[ "${EXPECTED_EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "DUCA_EVIDENCE_EXPECTED_COMMIT is required"
[[ "${EVIDENCE_COMMIT}" == "${EXPECTED_EVIDENCE_COMMIT}" ]] \
  || fail "evidence repository commit drift"
[[ "${EVIDENCE_COMMIT}" != "${TRAINED_COMMIT}" ]] \
  || fail "trained and evidence commits must be distinct"
[[ "$(git -C "${TRAINED_REPO_ROOT}" rev-parse HEAD)" == "${TRAINED_COMMIT}" ]] \
  || fail "trained repository commit drift"
[[ -z "$(git -C "${TRAINED_REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
  || fail "trained repository is dirty"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "evidence repository is dirty"

AGGREGATE="${RUN_ROOT}/aggregate_suite_evidence.json"
FINAL_SUITE="${RUN_ROOT}/final_suite_evidence.json"
CANDIDATE="${CONTROL_ROOT}/postrun_evidence_candidate.json"
OUTPUT="${CONTROL_ROOT}/postrun_evidence_complete.json"
for path in "${AGGREGATE}" "${FINAL_SUITE}" "${CANDIDATE}"; do
  [[ -f "${path}" ]] || fail "required evidence is missing: ${path}"
done
[[ ! -e "${OUTPUT}" ]] || fail "final post-run seal already exists"
[[ "${EXPECTED_AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256 is required"
[[ "$(sha256_file "${AGGREGATE}")" == "${EXPECTED_AGGREGATE_SHA256}" ]] \
  || fail "aggregate evidence hash mismatch"
FINAL_SUITE_SHA256="$(sha256_file "${FINAL_SUITE}")"

exec "${PYTHON}" -m tools.bata.finalize_duca_cellcf_postrun_evidence \
  --run-root "${RUN_ROOT}" --control-root "${CONTROL_ROOT}" \
  --trained-repo-root "${TRAINED_REPO_ROOT}" \
  --trained-commit "${TRAINED_COMMIT}" \
  --evidence-repo-root "${EVIDENCE_REPO_ROOT}" \
  --evidence-commit "${EVIDENCE_COMMIT}" \
  --aggregate "${AGGREGATE}" \
  --aggregate-sha256 "${EXPECTED_AGGREGATE_SHA256}" \
  --final-suite "${FINAL_SUITE}" \
  --final-suite-sha256 "${FINAL_SUITE_SHA256}" \
  --output "${OUTPUT}"
