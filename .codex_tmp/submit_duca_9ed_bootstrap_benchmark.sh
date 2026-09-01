#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_9ed1013_20260722
SOURCE=${BASE}/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map
ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_9ed1013_bootstrap_benchmark_$(date +%Y%m%d_%H%M%S)
mkdir -p ${ROOT}

cat > ${ROOT}/run.sbatch <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate
cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == 9ed10139317c4196072d471ced883eb1dfc31703 ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
python - "${SOURCE}" "${OUTPUT_JSON}" <<'PY'
import json
import sys
import time
from pathlib import Path

from tools.bata.duca_p0_evaluation import (
    bootstrap_official_map_differences,
    canonical_sha256,
    evaluation_video_ids,
)

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
families = (
    "A_exact_uniform",
    "R2Q3_privileged_boundary_burst",
    "R4Q5_privileged_boundary_burst",
    "Z_unrestricted_gt_oracle",
)
metrics = {
    family: json.loads((root / "map" / family / "metrics.json").read_text(encoding="utf-8"))
    for family in families
}
cfg = metrics[families[0]]["evaluation_config"]
predictions = {family: metrics[family]["prediction_path"] for family in families}
videos = evaluation_video_ids(cfg, expected_subset="training")
start = time.perf_counter()
bootstrap = bootstrap_official_map_differences(
    predictions,
    cfg,
    baseline_family=families[0],
    expected_video_ids=videos,
    expected_subset="training",
    samples=100,
    seed=3407,
    confidence=0.95,
    workers=8,
)
elapsed = time.perf_counter() - start
payload = {
    "ok": True,
    "tool_commit": "9ed10139317c4196072d471ced883eb1dfc31703",
    "source_prediction_commit": "e49ef69605e1f98a7217957483f93a8a64bfc348",
    "samples": 100,
    "workers": 8,
    "elapsed_seconds": elapsed,
    "projected_full_1000_seconds": elapsed * 10.0,
    "bootstrap": bootstrap,
}
payload["payload_sha256"] = canonical_sha256(payload)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({key: payload[key] for key in ("ok", "samples", "workers", "elapsed_seconds", "projected_full_1000_seconds", "payload_sha256")}, sort_keys=True))
PY
sha256sum "${OUTPUT_JSON}" | awk '{print $1}' > "${OUTPUT_JSON}.sha256"
SBATCH

JOB_RAW=$(sbatch --parsable --clusters=n16r4 \
  --job-name=duca9ed_bootbench --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gpus=1 --time=01:00:00 \
  --export=ALL,SNAPSHOT=${SNAPSHOT},SOURCE=${SOURCE},OUTPUT_JSON=${ROOT}/result.json \
  --output=${ROOT}/slurm-%j.out --error=${ROOT}/slurm-%j.err \
  ${ROOT}/run.sbatch)
JOB=${JOB_RAW%%;*}
[[ ${JOB} =~ ^[1-9][0-9]*$ ]]
printf 'job_id\t%s\nroot\t%s\nsnapshot\t%s\nsource\t%s\n' \
  ${JOB} ${ROOT} ${SNAPSHOT} ${SOURCE} > ${ROOT}/submission.tsv
sha256sum ${ROOT}/submission.tsv | awk '{print $1}' > ${ROOT}/submission.tsv.sha256
echo "JOB=${JOB}"
echo "ROOT=${ROOT}"
