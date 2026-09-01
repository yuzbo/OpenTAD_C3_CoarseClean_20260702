source /etc/profile
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_d9fb398_20260722"
OLD_RUN="$BASE/projects/c3_lowres_action_probe/duca_boundary_f90595d_r0_formal_20260722_0753"
COMMIT=d9fb398578716d278e818745677a92976bcedf2c

cd "$SNAP"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
source scripts/duca_cellcf_canonical_env.sh

INPUT="$OLD_RUN/r0_holdout_map/holdout_inputs.jsonl"
RUNTIME="$OLD_RUN/r0_holdout_map/runtime_bindings.json"
test -f "$INPUT"
test -f "$RUNTIME"
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST="$(python - "$RUNTIME" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["split"]["holdout_block_list"])
PY
)"
test -f "$DUCA_FRONTEND_HOLDOUT_BLOCK_LIST"

REPLAY_ROOT="$BASE/tmp/duca_r0_determinism_d9fb398_v2_20260722"
test ! -e "$REPLAY_ROOT"
mkdir -p "$REPLAY_ROOT/run1" "$REPLAY_ROOT/run2"

for run in run1 run2; do
  echo "REPLAY_START $run"
  python -m tools.bata.build_duca_r0_boundary_burst_oracles \
    --input-jsonl "$INPUT" \
    --config configs/adatad/thumos/duca_boundary_burst_r0_holdout_export.py \
    --output-jsonl "$REPLAY_ROOT/$run/holdout_families.jsonl" \
    --summary-json "$REPLAY_ROOT/$run/holdout_families.summary.json" \
    --max-unselected-hole 2
  echo "REPLAY_DONE $run"
done

cmp "$REPLAY_ROOT/run1/holdout_families.jsonl" \
  "$REPLAY_ROOT/run2/holdout_families.jsonl"
echo 'REPLAY_JSONL_IDENTICAL'
sha256sum \
  "$REPLAY_ROOT/run1/holdout_families.jsonl" \
  "$REPLAY_ROOT/run1/holdout_families.summary.json" \
  "$REPLAY_ROOT/run2/holdout_families.jsonl" \
  "$REPLAY_ROOT/run2/holdout_families.summary.json"
printf 'REMOTE_D9F_REPLAY_PASS commit=%s replay_root=%s\n' "$COMMIT" "$REPLAY_ROOT"
