#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_d9fb398_20260722"
OLD_RUN="$BASE/projects/c3_lowres_action_probe/duca_boundary_f90595d_r0_formal_20260722_0753"
cd "$SNAP"
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

echo 'MARK bash_syntax'
for script in \
  scripts/run_duca_boundary_burst_p0_gpu1.sh \
  scripts/run_duca_boundary_burst_gate_gpu1.sh \
  scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh \
  scripts/submit_duca_boundary_burst_official60_suite.sh \
  scripts/run_duca_frontend_pretrain_variant_gpu1.sh \
  scripts/run_duca_two_stage_curriculum_variant_gpu1.sh; do
  bash -n "$script"
  echo "BASH_OK $script"
done

echo 'MARK old_run_files'
find "$OLD_RUN" -maxdepth 2 -type f -printf '%P|%s\n' | sort | head -80

echo 'MARK runtime_binding_keys'
python - "$OLD_RUN/r0_holdout_map/runtime_bindings.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(sorted(payload))
print(json.dumps(payload.get("split"), indent=2, sort_keys=True))
PY

echo 'MARK probe_done'
