#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ARM="${ARM:-TRUETIME_K384}"
case "$ARM" in
  RANKPACK_K384) CONFIG="configs/adatad/thumos/duca_rankpack_k384_curriculum.py" ;;
  TRUETIME_K384) CONFIG="configs/adatad/thumos/duca_truetime_k384_curriculum.py" ;;
  *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac
export CONFIG_FOR_CHECK="$CONFIG"
python -m py_compile "$CONFIG"
python - <<'PY'
from mmengine.config import Config
import os
c=Config.fromfile(os.environ.get("CONFIG_FOR_CHECK", "configs/adatad/thumos/duca_truetime_k384_curriculum.py"))
assert c.window_size == 384
assert c.experiment_scope.requested_k == c.experiment_scope.effective_k == 384
assert c.experiment_scope.executed_k == 384
assert tuple(c.duca_curriculum.phase_boundaries) == (20,40,60)
assert tuple(c.duca_curriculum.phase_successful_update_boundaries) == (2000,4000,6000)
assert c.model.frame_selector.homotopy_warmup_steps == 2000
assert c.model.frame_selector.homotopy_transition_steps == 2000
assert c.model.frame_selector.homotopy_total_steps == 6000
assert c.workflow.end_epoch == 60 and c.workflow.max_train_iters is None
assert c.workflow.checkpoint_interval == 5
assert c.workflow.primary_checkpoint_state_key == "state_dict_ema"
assert c.dataset.train.type == "DucaStatelessThumosPaddingDataset"
assert c.experiment_scope.repeats_dense_uniform_random is False
PY
if [[ "$PRECHECK_ONLY" == 1 ]]; then
  echo "DUCA_TRUE_TIME_INDIRECT_CURRICULUM_PRECHECK_PASS arm=$ARM config=$CONFIG"
  exit 0
fi
[[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" ]] || { echo 'full training requires Slurm' >&2; exit 3; }
python -m torch.distributed.run --nproc_per_node=1 tools/train.py "$CONFIG" --seed "${SEED:-0}"
