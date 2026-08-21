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
assert c.window_size == c.requested_k == c.effective_k == c.executed_k == 384
assert tuple(c.duca_curriculum.phase_boundaries) == (20,40,60)
assert c.workflow.end_epoch == 60 and c.workflow.max_train_iters == 6000
assert c.workflow.checkpoint_interval == 5
PY
if [[ "$PRECHECK_ONLY" == 1 ]]; then
  echo "DUCA_TRUE_TIME_INDIRECT_CURRICULUM_PRECHECK_PASS arm=$ARM config=$CONFIG"
  exit 0
fi
[[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" ]] || { echo 'full training requires Slurm' >&2; exit 3; }
python -m torch.distributed.run --nproc_per_node=1 tools/train.py "$CONFIG" --seed "${SEED:-0}"
