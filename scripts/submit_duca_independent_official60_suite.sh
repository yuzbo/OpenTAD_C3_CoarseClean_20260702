#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_INDEPENDENT_SUBMIT][FAIL] $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "clean tree required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"
export DUCA_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"
export DUCA_ADATAD_PRETRAIN_SHA256="${ADATAD_PRETRAIN_SHA256}"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/arms"
"${PYTHON}" -m tools.bata.audit_duca_map_protocols \
  --output-json "${RUN_ROOT}/map_protocol_audit.json" \
  > "${RUN_ROOT}/map_protocol_audit.out"

if [[ -n "${DUCA_INDEPENDENT_VARIANTS:-}" ]]; then
  read -r -a variants <<< "${DUCA_INDEPENDENT_VARIANTS}"
else
  variants=(
    two_stage_exact_uniform
    gaussian_matched_g0
    boundary_burst_r2q3_g0
    boundary_burst_r4q5_g0
  )
fi
[[ "${#variants[@]}" -gt 0 ]] || fail "at least one variant is required"
for variant in "${variants[@]}"; do
  sbatch_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${sbatch_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca_${variant}_${EXPECTED_COMMIT:0:7}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_ADATAD_PRETRAIN_PATH='${ADATAD_PRETRAIN_PATH}'
export DUCA_ADATAD_PRETRAIN_SHA256='${ADATAD_PRETRAIN_SHA256}'
export DUCA_INDEPENDENT_VARIANT='${variant}'
export DUCA_INDEPENDENT_ARM_ROOT='${RUN_ROOT}/arms/${variant}'
bash scripts/run_duca_independent_official60_gpu1.sh
EOF
  bash -n "${sbatch_file}"
done

"${PYTHON}" - "${RUN_ROOT}/submission_manifest.json" "${EXPECTED_COMMIT}" \
  "${RUN_ROOT}/map_protocol_audit.json" "${RUN_ROOT}" "${variants[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.bata.duca_selected_axis_training import atomic_write_json

output, commit, audit, root, *variants = sys.argv[1:]
root_path = Path(root).resolve()
audit_path = Path(audit).resolve()
jobs = []
for variant in variants:
    path = root_path / "jobs" / f"{variant}.sbatch"
    jobs.append(
        {
            "variant": variant,
            "dependency": "none",
            "sbatch": str(path),
            "sbatch_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
atomic_write_json(
    Path(output).resolve(),
    {
        "schema": "duca_independent_official60_submission_v1",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "run_root": str(root_path),
        "official_map_audit": str(audit_path),
        "official_map_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "inter_job_dependencies": False,
        "frontend_recipe": "fixed_terminal_epoch_19_full_training_subset",
        "official_recipe": "60_epochs_terminal_epoch_59_ema_full_validation",
        "jobs": jobs,
    },
)
PY

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_INDEPENDENT_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

printf 'variant\tjob_id\tdependency\tsbatch\n' > "${RUN_ROOT}/jobs.tsv"
for variant in "${variants[@]}"; do
  sbatch_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${sbatch_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch id for ${variant}"
  printf '%s\t%s\tnone\t%s\n' "${variant}" "${job_id}" "${sbatch_file}" \
    >> "${RUN_ROOT}/jobs.tsv"
done
cat "${RUN_ROOT}/jobs.tsv"
