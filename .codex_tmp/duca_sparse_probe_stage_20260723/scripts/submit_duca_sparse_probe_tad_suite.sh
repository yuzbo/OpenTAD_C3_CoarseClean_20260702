#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_SPARSE_PROBE_SUITE][FAIL] $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
PRETRAIN="${DUCA_ADATAD_PRETRAIN_PATH:-}"
PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"

[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PRETRAIN}" ]] || fail "AdaTAD pretrain is missing"
[[ "$(sha256sum "${PRETRAIN}" | awk '{print $1}')" == "${PRETRAIN_SHA256}" ]] \
  || fail "AdaTAD pretrain hash drift"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/arms"
SBATCH_FILE="${RUN_ROOT}/jobs/sparse_probe_four_stride.sbatch"
cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca_sparse_probe_${EXPECTED_COMMIT:0:7}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --time=7-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/suite-%j.out
#SBATCH --error=${RUN_ROOT}/logs/suite-%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'

pids=()
for stride in 1 2 3 4; do
  srun --exact --exclusive --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=4 \
    env BASE='${BASE}' \
    DUCA_REPO_ROOT='${REPO_ROOT}' \
    DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}' \
    DUCA_ADATAD_PRETRAIN_PATH='${PRETRAIN}' \
    DUCA_ADATAD_PRETRAIN_SHA256='${PRETRAIN_SHA256}' \
    DUCA_INDEPENDENT_VARIANT="sparse_probe_hidden_linear_d\${stride}" \
    DUCA_SPARSE_PROBE_STRIDE="\${stride}" \
    DUCA_INDEPENDENT_ARM_ROOT='${RUN_ROOT}'/arms/d"\${stride}" \
    bash scripts/run_duca_independent_official60_gpu1.sh \
    > '${RUN_ROOT}'/logs/d"\${stride}".out 2>&1 &
  pids+=("\$!")
done

failed=0
for pid in "\${pids[@]}"; do wait "\${pid}" || failed=1; done
[[ "\${failed}" == 0 ]] || exit 1
python -m tools.bata.aggregate_duca_sparse_probe_tad \
  --run-root '${RUN_ROOT}'/arms \
  --output '${RUN_ROOT}'/sparse_probe_tad_summary.json
EOF
bash -n "${SBATCH_FILE}"

python - "${RUN_ROOT}/deployment_manifest.json" "${EXPECTED_COMMIT}" \
  "${RUN_ROOT}" "${SBATCH_FILE}" "${PRETRAIN_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, commit, run_root, sbatch, pretrain_sha = sys.argv[1:]
sbatch_path = Path(sbatch).resolve()
payload = {
    "schema": "duca_sparse_probe_hidden_linear_deployment_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "run_root": str(Path(run_root).resolve()),
    "strides_dense_candidates": [1, 2, 3, 4],
    "intervals_source_frames": [4, 8, 12, 16],
    "reconstruction": "multidimensional_temporal_hidden_linear_to_768",
    "selector_receives_anchor_metadata": False,
    "hard_budget": 384,
    "backend": "official_derived_adatad_actionformer",
    "seed": 3407,
    "terminal_checkpoint_epoch_zero_based": 59,
    "checkpoint_state_key": "state_dict_ema",
    "sbatch": str(sbatch_path),
    "sbatch_sha256": hashlib.sha256(sbatch_path.read_bytes()).hexdigest(),
    "adatad_pretrain_sha256": pretrain_sha,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_SPARSE_PROBE_SUITE] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${SBATCH_FILE}")"
job_id="${raw%%;*}"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid Slurm job id"
printf 'role\tjob_id\tdependency\tgpus\tsbatch\n' > "${RUN_ROOT}/jobs.tsv"
printf 'sparse_probe_d1_d4\t%s\tnone\t4\t%s\n' "${job_id}" "${SBATCH_FILE}" \
  >> "${RUN_ROOT}/jobs.tsv"
sha256sum "${RUN_ROOT}/jobs.tsv" | awk '{print $1}' > "${RUN_ROOT}/jobs.tsv.sha256"
echo "[DUCA_SPARSE_PROBE_SUITE] submitted job=${job_id} root=${RUN_ROOT}"
