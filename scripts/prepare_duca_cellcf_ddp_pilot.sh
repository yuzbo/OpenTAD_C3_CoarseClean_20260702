#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_DDP_PILOT_PREPARE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
REAL_LOADER_GATE_JSON="${DUCA_CELLCF_GATE_JSON:-}"

[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "pilot preparation requires a clean exact-commit checkout"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -n "${REAL_LOADER_GATE_JSON}" && -f "${REAL_LOADER_GATE_JSON}" ]] \
  || fail "DUCA_CELLCF_GATE_JSON must name the real-loader CUDA gate"

REAL_LOADER_GATE_JSON="$("${PYTHON}" -c \
  'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
  "${REAL_LOADER_GATE_JSON}")"
OBSERVED_GATE_SHA256="$(sha256sum "${REAL_LOADER_GATE_JSON}" | awk '{print $1}')"
EXPECTED_GATE_SHA256="${DUCA_CELLCF_GATE_SHA256:-${DUCA_CELLCF_REAL_LOADER_GATE_SHA256:-${OBSERVED_GATE_SHA256}}}"
[[ "${EXPECTED_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_GATE_SHA256 must be a lowercase SHA256"
[[ "${OBSERVED_GATE_SHA256}" == "${EXPECTED_GATE_SHA256}" ]] \
  || fail "real-loader gate SHA256 differs from the requested binding"

RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_cellcf_ddp_pilot_${CURRENT_HEAD:0:12}_${EXPECTED_GATE_SHA256:0:12}}"
SBATCH_FILE="${SBATCH_FILE:-${RUN_ROOT}.sbatch}"
case "${RUN_ROOT}" in
  "${BASE}"/*) ;;
  *) fail "RUN_ROOT must stay beneath BASE=${BASE}" ;;
esac
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"
[[ ! -e "${SBATCH_FILE}" ]] || fail "SBATCH_FILE already exists: ${SBATCH_FILE}"

PRECHECK_ONLY=1 \
DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}" \
DUCA_CELLCF_GATE_JSON="${REAL_LOADER_GATE_JSON}" \
DUCA_CELLCF_GATE_SHA256="${EXPECTED_GATE_SHA256}" \
DUCA_CELLCF_REAL_LOADER_GATE_SHA256="${EXPECTED_GATE_SHA256}" \
RUN_ROOT="${RUN_ROOT}" \
bash scripts/run_duca_cellcf_ddp_pilot.sh

mkdir -p "$(dirname "${SBATCH_FILE}")"
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' '#SBATCH --job-name=duca-cellcf-ddp-pilot'
  printf '%s\n' '#SBATCH --nodes=1'
  printf '%s\n' '#SBATCH --ntasks=1'
  printf '%s\n' '#SBATCH --gres=gpu:1'
  printf '%s\n' '#SBATCH --cpus-per-task=4'
  printf '#SBATCH --output=%s\n' "${RUN_ROOT}.slurm-%j.out"
  printf '#SBATCH --error=%s\n' "${RUN_ROOT}.slurm-%j.err"
  printf '%s\n' 'set -euo pipefail'
  printf 'cd %q\n' "${REPO_ROOT}"
  printf 'export BASE=%q\n' "${BASE}"
  printf 'export DUCA_EXPECTED_COMMIT=%q\n' "${EXPECTED_COMMIT}"
  printf 'export DUCA_CELLCF_GATE_JSON=%q\n' "${REAL_LOADER_GATE_JSON}"
  printf 'export DUCA_CELLCF_GATE_SHA256=%q\n' "${EXPECTED_GATE_SHA256}"
  printf 'export DUCA_CELLCF_REAL_LOADER_GATE_SHA256=%q\n' "${EXPECTED_GATE_SHA256}"
  printf 'export RUN_ROOT=%q\n' "${RUN_ROOT}"
  printf '%s\n' 'bash scripts/run_duca_cellcf_ddp_pilot.sh'
} >"${SBATCH_FILE}"
chmod 0755 "${SBATCH_FILE}"
bash -n "${SBATCH_FILE}" || fail "generated sbatch file has invalid syntax"

echo "[DUCA_CELLCF_DDP_PILOT_PREPARE] commit=${EXPECTED_COMMIT}"
echo "[DUCA_CELLCF_DDP_PILOT_PREPARE] real_loader_gate_sha256=${EXPECTED_GATE_SHA256}"
echo "[DUCA_CELLCF_DDP_PILOT_PREPARE] sbatch_file=${SBATCH_FILE}"
echo "[DUCA_CELLCF_DDP_PILOT_PREPARE] no Slurm job was submitted"
