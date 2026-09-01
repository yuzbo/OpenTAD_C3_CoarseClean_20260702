#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_DDP_PILOT_PREPARE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"
CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_p0_ddp_pilot_${CURRENT_HEAD:0:7}}"
SBATCH_FILE="${SBATCH_FILE:-${RUN_ROOT}.sbatch}"

[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "pilot preparation requires a clean git tree"
[[ -n "${CORE_GATE_JSON}" && -f "${CORE_GATE_JSON}" ]] || fail "DUCA_CORE_GATE_JSON is missing"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"
[[ ! -e "${SBATCH_FILE}" ]] || fail "SBATCH_FILE already exists: ${SBATCH_FILE}"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca-p0-ddp-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}.slurm-%j.out
#SBATCH --error=${RUN_ROOT}.slurm-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CORE_GATE_JSON='${CORE_GATE_JSON}'
export RUN_ROOT='${RUN_ROOT}'
bash scripts/run_duca_transition_only_p0_ddp_pilot.sh
EOF
chmod 0755 "${SBATCH_FILE}"
bash -n "${SBATCH_FILE}" || fail "generated sbatch file has invalid syntax"

echo "[DUCA_P0_DDP_PILOT_PREPARE] ${SBATCH_FILE}"
echo "[DUCA_P0_DDP_PILOT_PREPARE] no Slurm job was submitted"
