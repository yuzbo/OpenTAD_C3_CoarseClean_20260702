#!/usr/bin/env bash
source /etc/profile
set -u
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_63e25eb_serial_20260721_2120
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("lr_control_c25_a50_s100", "lr_coarse50_action100_scorer25", "lr_coarse100_action200_scorer50"):
    completion = root / "p0" / name / "run" / "completion.json"
    print("VARIANT", name, "complete", completion.is_file())
    if not completion.is_file():
        continue
    payload = json.loads(completion.read_text())
    print("winner", json.dumps(payload.get("winner", {}), sort_keys=True))
    print("candidates", len(payload.get("candidates", [])))
    for row in payload.get("candidates", []):
        print(row.get("epoch_one_based"), row.get("summary_path"))
PY
echo '===FATAL-SCAN==='
grep -R -n -E 'Traceback|OutOfMemory|CUDA out of memory|non-finite loss|\[FAIL\]' "$ROOT"/p0/*/run/*.out 2>/dev/null | tail -30 || true
