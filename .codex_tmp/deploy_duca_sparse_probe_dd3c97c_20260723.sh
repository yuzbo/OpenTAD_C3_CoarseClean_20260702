#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAPSHOT=${BASE}/projects/opentad_duca_sparse_62efd9c_20260723
COMMIT=dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
PRETRAIN_SHA=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251
OFFICIAL_REPOS=${BASE}/projects/external_official_action_segmentation_repos_20260702
RUN_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_$(date +%Y%m%d_%H%M%S)

[[ "$(git -C "${SNAPSHOT}" rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git -C "${SNAPSHOT}" status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "${PRETRAIN}" | awk '{print $1}')" == "${PRETRAIN_SHA}" ]]
[[ -f "${OFFICIAL_REPOS}/ASFormer/model.py" ]]
mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/gate"

GATE_SBATCH=${RUN_ROOT}/jobs/cuda_gate.sbatch
cat > "${GATE_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca_sparse_gate_dd3c97c
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=${RUN_ROOT}/logs/gate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/gate-%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${SNAPSHOT}'
[[ "\$(git rev-parse HEAD)" == '${COMMIT}' ]]
[[ -z "\$(git status --porcelain --untracked-files=normal)" ]]
export C3_OFFICIAL_ACTION_SEG_REPOS='${OFFICIAL_REPOS}'
python -m pytest tests/test_duca_sparse_probe_interpolation.py -q
python -m tools.bata.run_duca_sparse_probe_cuda_gate \
  --output '${RUN_ROOT}/gate/sparse_probe_cuda_gate.json' \
  --temporal-len 32
EOF
bash -n "${GATE_SBATCH}"

gate_raw="$(sbatch --parsable --clusters=n16r4 "${GATE_SBATCH}")"
gate_job="${gate_raw%%;*}"
[[ "${gate_job}" =~ ^[1-9][0-9]*$ ]]

SUITE_ROOT=${RUN_ROOT}/suite
PRECHECK_ONLY=1 \
RUN_ROOT="${SUITE_ROOT}" \
DUCA_EXPECTED_COMMIT="${COMMIT}" \
DUCA_ADATAD_PRETRAIN_PATH="${PRETRAIN}" \
DUCA_ADATAD_PRETRAIN_SHA256="${PRETRAIN_SHA}" \
DUCA_TARGET_CLUSTER=n16r4 \
bash "${SNAPSHOT}/scripts/submit_duca_sparse_probe_tad_suite.sh"

SUITE_SBATCH=${SUITE_ROOT}/jobs/sparse_probe_four_stride.sbatch
suite_raw="$(sbatch --parsable --clusters=n16r4 \
  --dependency=afterok:${gate_job} "${SUITE_SBATCH}")"
suite_job="${suite_raw%%;*}"
[[ "${suite_job}" =~ ^[1-9][0-9]*$ ]]

printf 'role\tjob_id\tdependency\tgpus\tsbatch\n' > "${RUN_ROOT}/jobs.tsv"
printf 'cuda_gate\t%s\tnone\t1\t%s\n' "${gate_job}" "${GATE_SBATCH}" \
  >> "${RUN_ROOT}/jobs.tsv"
printf 'sparse_probe_d1_d4\t%s\tafterok:%s\t4\t%s\n' \
  "${suite_job}" "${gate_job}" "${SUITE_SBATCH}" >> "${RUN_ROOT}/jobs.tsv"
sha256sum "${RUN_ROOT}/jobs.tsv" | awk '{print $1}' > "${RUN_ROOT}/jobs.tsv.sha256"

python - "${RUN_ROOT}/deployment_receipt.json" "${COMMIT}" "${SNAPSHOT}" \
  "${RUN_ROOT}" "${gate_job}" "${suite_job}" "${GATE_SBATCH}" \
  "${SUITE_SBATCH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, commit, snapshot, root, gate_job, suite_job, gate_sbatch, suite_sbatch = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema": "duca_sparse_probe_deployment_receipt_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "snapshot": snapshot,
    "run_root": root,
    "gate_job": int(gate_job),
    "suite_job": int(suite_job),
    "suite_dependency": f"afterok:{gate_job}",
    "gate_sbatch_sha256": digest(gate_sbatch),
    "suite_sbatch_sha256": digest(suite_sbatch),
    "strides_dense_candidates": [1, 2, 3, 4],
    "intervals_source_frames": [4, 8, 12, 16],
    "terminal_metric": "official_validation_epoch59_state_dict_ema_map",
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "GATE_JOB=${gate_job}"
echo "SUITE_JOB=${suite_job}"
