#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
SNAP="$BASE/projects/opentad_duca_boundary_4a07a2a_20260722"
RUN_ROOT="$BASE/projects/c3_lowres_action_probe/duca_boundary_4a07a2a_precheck_20260722_0205"
CHECKPOINT="$BASE/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth"
COMMIT=4a07a2af72e68f1330467161cbcac2ffba53d367
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
cd "$SNAP"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
export BASE RUN_ROOT DUCA_EXPECTED_COMMIT="$COMMIT"
export DUCA_R0_CHECKPOINT="$CHECKPOINT" DUCA_R0_CHECKPOINT_EPOCH=131
export PRECHECK_ONLY=1
bash scripts/submit_duca_boundary_burst_official60_suite.sh
test -f "$RUN_ROOT/frontend_split/frontend_split_manifest.json"
for file in "$RUN_ROOT"/submission/*.sbatch; do bash -n "$file"; done
printf 'REMOTE_SUBMIT_PRECHECK_PASS root=%s\n' "$RUN_ROOT"
