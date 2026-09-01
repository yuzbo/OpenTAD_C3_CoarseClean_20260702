#!/usr/bin/env bash
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
set -euo pipefail

declare -A ARM_ROOTS=(
  [exact_uniform]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048/arms/two_stage_exact_uniform/official60
  [R2Q3]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_26ce86d_recovery_20260723_050516/arms/boundary_burst_r2q3_g0/official60
  [R4Q5]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220/arms/boundary_burst_r4q5_g0/official60
  [soft_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_detached/official60
  [hard_detached]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/hard_detached/official60
  [soft_adapted]=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756/arms/soft_adapted/official60
)

for name in exact_uniform R2Q3 R4Q5 soft_detached hard_detached soft_adapted; do
  root="${ARM_ROOTS[$name]}"
  audit="${root}/work/gpu1_id0/duca_selected_axis_training_audit.json"
  echo "=== ${name} ==="
  source_config="$(python - "${audit}" <<'PY'
import json
import sys
from pathlib import Path
audit = json.loads(Path(sys.argv[1]).read_text())
print(Path(audit["source_config_path"]).resolve())
PY
)"
  repo="${source_config%%/configs/*}"
  export DUCA_FRONTEND_CHECKPOINT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print((d.get("selector_initialization_contract") or {}).get("checkpoint_path", ""))' "${audit}")"
  export DUCA_FRONTEND_CHECKPOINT_SHA256="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print((d.get("selector_initialization_contract") or {}).get("checkpoint_sha256", ""))' "${audit}")"
  export DUCA_FRONTEND_CHECKPOINT_EPOCH="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print((d.get("selector_initialization_contract") or {}).get("checkpoint_epoch", ""))' "${audit}")"
  cd "${repo}"
  echo "source_config ${source_config}"
  python - "${source_config}" <<'PY'
import sys
from mmengine.config import Config
cfg = Config.fromfile(sys.argv[1])
print("formal_protocol", cfg.workflow.formal_protocol)
print("budget", cfg.model.frame_selector.budget)
print("max_hole", cfg.model.frame_selector.max_unselected_hole)
print("frontend_checkpoint", cfg.workflow.get("frontend_checkpoint", "loaded-via-model-config"))
PY
done
