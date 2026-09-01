#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_CONVERGENCE_SUMMARY][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EVIDENCE_COMMIT="$(git rev-parse HEAD)"
EXPECTED_EVIDENCE_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-}"
POSTRUN_OUTPUT_ROOT="${DUCA_CELLCF_POSTRUN_OUTPUT_ROOT:-}"
AGGREGATE_EVIDENCE="${RUN_ROOT}/aggregate_suite_evidence.json"
AGGREGATE_SHA256="${DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256:-}"

[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ "${EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "evidence repository commit is invalid"
[[ "${EXPECTED_EVIDENCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "DUCA_EVIDENCE_EXPECTED_COMMIT is required"
[[ "${EVIDENCE_COMMIT}" == "${EXPECTED_EVIDENCE_COMMIT}" ]] \
  || fail "evidence repository commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "evidence repository is dirty"
[[ -n "${EXPECTED_COMMIT}" ]] || fail "DUCA_EXPECTED_COMMIT is required"
[[ -d "${RUN_ROOT}" ]] || fail "formal run root is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "RUN_ROOT violates the formal path contract"
if [[ -z "${POSTRUN_OUTPUT_ROOT}" ]]; then
  POSTRUN_OUTPUT_ROOT="${RUN_ROOT}"
else
  POSTRUN_OUTPUT_ROOT="$(
    duca_cellcf_require_external_path \
      "POSTRUN_OUTPUT_ROOT" "${REPO_ROOT}" "${BASE}" \
      "${POSTRUN_OUTPUT_ROOT}"
  )" || fail "POSTRUN_OUTPUT_ROOT violates the formal path contract"
  case "${POSTRUN_OUTPUT_ROOT}/" in
    "${RUN_ROOT}/"*) ;;
    *) fail "POSTRUN_OUTPUT_ROOT must stay under RUN_ROOT" ;;
  esac
fi
CONVERGENCE_ROOT="${POSTRUN_OUTPUT_ROOT}/convergence"
[[ -f "${AGGREGATE_EVIDENCE}" ]] || fail "aggregate suite evidence is missing"
[[ "${AGGREGATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256 is required"
for variant in uniform transition_beta0 cellcf; do
  [[ -f "${CONVERGENCE_ROOT}/${variant}/variant_complete.json" ]] \
    || fail "${variant} trajectory receipt is missing"
done
[[ ! -e "${CONVERGENCE_ROOT}/fixed_trajectory.json" ]] \
  || fail "refusing to overwrite fixed trajectory JSON"
[[ ! -e "${CONVERGENCE_ROOT}/fixed_trajectory.tsv" ]] \
  || fail "refusing to overwrite fixed trajectory TSV"

"${PYTHON}" -m tools.bata.summarize_duca_cellcf_convergence \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-evidence-commit "${EVIDENCE_COMMIT}" \
  --suite-aggregate "${AGGREGATE_EVIDENCE}" \
  --suite-aggregate-sha256 "${AGGREGATE_SHA256}" \
  --post-run "uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json" \
  --post-run "transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json" \
  --post-run "cellcf=${RUN_ROOT}/logs/cellcf/post_run_evidence.json" \
  --variant-receipt "uniform=${CONVERGENCE_ROOT}/uniform/variant_complete.json" \
  --variant-receipt "transition_beta0=${CONVERGENCE_ROOT}/transition_beta0/variant_complete.json" \
  --variant-receipt "cellcf=${CONVERGENCE_ROOT}/cellcf/variant_complete.json" \
  --evaluation "uniform:59=${CONVERGENCE_ROOT}/uniform/epoch_59/evaluation.json" \
  --evaluation "uniform:89=${CONVERGENCE_ROOT}/uniform/epoch_89/evaluation.json" \
  --evaluation "uniform:131=${RUN_ROOT}/logs/uniform/terminal_evaluation.json" \
  --evaluation "transition_beta0:59=${CONVERGENCE_ROOT}/transition_beta0/epoch_59/evaluation.json" \
  --evaluation "transition_beta0:89=${CONVERGENCE_ROOT}/transition_beta0/epoch_89/evaluation.json" \
  --evaluation "transition_beta0:131=${RUN_ROOT}/logs/transition_beta0/terminal_evaluation.json" \
  --evaluation "cellcf:59=${CONVERGENCE_ROOT}/cellcf/epoch_59/evaluation.json" \
  --evaluation "cellcf:89=${CONVERGENCE_ROOT}/cellcf/epoch_89/evaluation.json" \
  --evaluation "cellcf:131=${RUN_ROOT}/logs/cellcf/terminal_evaluation.json" \
  --output-json "${CONVERGENCE_ROOT}/fixed_trajectory.json" \
  --output-tsv "${CONVERGENCE_ROOT}/fixed_trajectory.tsv"
