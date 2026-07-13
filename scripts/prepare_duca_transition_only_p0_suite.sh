#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_PREPARE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
SEED="${SEED:-0}"
CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_p0_matched_${CURRENT_HEAD:0:7}_seed${SEED}}"
CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"

[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -n "${CORE_GATE_JSON}" && -f "${CORE_GATE_JSON}" ]] || fail "DUCA_CORE_GATE_JSON must name an existing formal gate"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal suite preparation requires a clean git tree"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
"${PYTHON}" tools/bata/validate_duca_transition_only_p0_suite.py \
  --repo-root "${REPO_ROOT}" \
  --seed "${SEED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --core-gate-json "${CORE_GATE_JSON}" \
  --output-json "${MANIFEST}"

variants=(uniform direct transition_beta0 transition_beta025)
ports=(30511 30512 30513 30514)
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca-p0-${variant}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export DUCA_P0_VARIANT='${variant}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CORE_GATE_JSON='${CORE_GATE_JSON}'
export FULLTRAIN_CANDIDATE=1
export SEED='${SEED}'
export RUN_ID='${index}'
export MASTER_PORT='${ports[$index]}'
export RUN_DIR='${RUN_ROOT}/logs/${variant}'
export WORK_DIR='${RUN_ROOT}/work_dirs/${variant}'
bash scripts/run_duca_transition_only_p0_variant_gpu1.sh
EOF
  chmod 0755 "${job_file}"
done

cat > "${RUN_ROOT}/jobs.tsv" <<EOF
variant\tseed\tcommit\tsbatch_file\tstatus
uniform\t${SEED}\t${EXPECTED_COMMIT}\t${RUN_ROOT}/jobs/uniform.sbatch\tPREPARED_NOT_SUBMITTED
direct\t${SEED}\t${EXPECTED_COMMIT}\t${RUN_ROOT}/jobs/direct.sbatch\tPREPARED_NOT_SUBMITTED
transition_beta0\t${SEED}\t${EXPECTED_COMMIT}\t${RUN_ROOT}/jobs/transition_beta0.sbatch\tPREPARED_NOT_SUBMITTED
transition_beta025\t${SEED}\t${EXPECTED_COMMIT}\t${RUN_ROOT}/jobs/transition_beta025.sbatch\tPREPARED_NOT_SUBMITTED
EOF

echo "[DUCA_P0_PREPARE] prepared four matched jobs under ${RUN_ROOT}"
echo "[DUCA_P0_PREPARE] no Slurm jobs were submitted"
