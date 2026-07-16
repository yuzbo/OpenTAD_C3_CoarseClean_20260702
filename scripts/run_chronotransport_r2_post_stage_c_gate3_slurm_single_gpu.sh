#!/usr/bin/env bash
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CHRONOTRANSPORT_REGISTRATION_COMMIT:?registration commit R is required}"
: "${CHRONOTRANSPORT_REGISTRATION:?immutable registration JSON is required}"
: "${CHRONOTRANSPORT_GATE1_UNLOCK:?canonical Gate-1 unlock is required}"
: "${CHRONOTRANSPORT_GATES23_REPLAY:?canonical pre-Stage-C replay is required}"
: "${CHRONOTRANSPORT_GATES23_REPORT:?canonical pre-Stage-C PASS report is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3407:?Stage-B marker 3407 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3408:?Stage-B marker 3408 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3409:?Stage-B marker 3409 is required}"

FORMAL_OUTPUT_BASE="/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2"
SHARED_ROOT="$FORMAL_OUTPUT_BASE/$CHRONOTRANSPORT_REGISTRATION_COMMIT/shared"
[[ "$CHRONOTRANSPORT_GATE1_UNLOCK" == "$SHARED_ROOT/gate1/gate1_result.json" ]] || {
  echo "Gate-1 unlock must use the canonical R path" >&2; exit 41;
}
[[ "$CHRONOTRANSPORT_GATES23_REPLAY" == "$SHARED_ROOT/gates23/gates23_replay.json" ]] || {
  echo "pre-Stage-C replay must use the canonical R path" >&2; exit 42;
}
[[ "$CHRONOTRANSPORT_GATES23_REPORT" == "$SHARED_ROOT/gates23/gates23_report.json" ]] || {
  echo "pre-Stage-C report must use the canonical R path" >&2; exit 43;
}

[[ "$(git status --porcelain)" == "" ]] || { echo "dirty worktree" >&2; exit 44; }
[[ "$(git rev-parse HEAD)" == "$CHRONOTRANSPORT_REGISTRATION_COMMIT" ]] || {
  echo "HEAD must equal registration commit R" >&2; exit 45;
}
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
  echo "formal post-Stage-C Gate 3 requires a Slurm allocation and step" >&2; exit 46;
}
# Slurm owns CUDA visibility.  This launcher never assigns or rewrites
# CUDA_VISIBLE_DEVICES; every process uses the single logical cuda:0.

python -m py_compile \
  opentad/models/chronotransport/post_stage_c.py \
  opentad/models/chronotransport/formal_stage_c.py \
  tools/bata/chronotransport_r2_post_stage_c_factory.py \
  tools/bata/run_chronotransport_r2_post_stage_c_gate3.py \
  tools/bata/validate_chronotransport_r2_post_stage_c_gate3.py

COMMON_ARGS=(
  --registration "$CHRONOTRANSPORT_REGISTRATION"
  --gate1-unlock "$CHRONOTRANSPORT_GATE1_UNLOCK"
  --gates23-replay "$CHRONOTRANSPORT_GATES23_REPLAY"
  --gates23-report "$CHRONOTRANSPORT_GATES23_REPORT"
  --phase-marker-3407 "$CHRONOTRANSPORT_PHASE_MARKER_3407"
  --phase-marker-3408 "$CHRONOTRANSPORT_PHASE_MARKER_3408"
  --phase-marker-3409 "$CHRONOTRANSPORT_PHASE_MARKER_3409"
)

python tools/bata/run_chronotransport_r2_post_stage_c_gate3.py \
  "${COMMON_ARGS[@]}" \
  --precheck-only

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "PRECHECK_OK"
  exit 0
fi

python tools/bata/run_chronotransport_r2_post_stage_c_gate3.py \
  "${COMMON_ARGS[@]}"

python tools/bata/validate_chronotransport_r2_post_stage_c_gate3.py \
  "${COMMON_ARGS[@]}"

echo "POST_STAGE_C_GATE3_SUCCESS:$SHARED_ROOT/post_stage_c_gate3/terminal_marker.json"
