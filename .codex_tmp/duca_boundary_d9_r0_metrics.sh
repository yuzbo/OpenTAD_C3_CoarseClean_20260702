#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357/r0_holdout_map
set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
echo "DATE=$(date '+%F %T %z')"
python - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
families = (
    "A_exact_uniform",
    "R2Q3_privileged_boundary_burst",
    "R4Q5_privileged_boundary_burst",
    "Z_unrestricted_gt_oracle",
)
for family in families:
    path = root / "map" / family / "metrics.json"
    if not path.is_file():
        print("{}|PENDING".format(family))
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    fields = [family, "avg={:.6f}".format(100.0 * metrics["average_mAP"])]
    for iou in (0.3, 0.4, 0.5, 0.6, 0.7):
        key = "mAP@{:.1f}".format(iou)
        fields.append("{}={:.6f}".format(key, 100.0 * metrics[key]))
    fields.append("sha={}".format(data["evaluation_sha256"]))
    print("|".join(fields))
for name in ("r0_bootstrap.json", "r0_summary.json"):
    path = root / name
    if path.is_file():
        print("=== {} ===".format(name))
        print(path.read_text(encoding="utf-8"))
    else:
        print("{}|PENDING".format(name))
PY

echo "=== JOBS ==="
sacct -j 1179517,1179533 --format=JobID,JobName%30,State,Elapsed,ExitCode,NodeList%20 -P || true
squeue -j 1179517,1179533 -o '%i|%j|%T|%M|%R' || true
echo "=== ERRORS ==="
grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite|\[.*FAIL.*\]|ValueError' \
  "${ROOT}/../logs" 2>/dev/null || true
