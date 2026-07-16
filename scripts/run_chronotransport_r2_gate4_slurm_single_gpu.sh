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
: "${CHRONOTRANSPORT_GATES23_REPORT:?canonical pre-Stage-C report is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3407:?Stage-B marker 3407 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3408:?Stage-B marker 3408 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3409:?Stage-B marker 3409 is required}"
: "${CHRONOTRANSPORT_POST_STAGE_C_REPLAY:?post-Stage-C replay is required}"
: "${CHRONOTRANSPORT_POST_STAGE_C_REPORT:?post-Stage-C Gate-3 report is required}"
: "${CHRONOTRANSPORT_POST_STAGE_C_UNLOCK:?post-Stage-C Gate-3 unlock is required}"
: "${CHRONOTRANSPORT_POST_STAGE_C_TERMINAL:?post-Stage-C terminal is required}"
: "${CHRONOTRANSPORT_GATE4_MODE:?Gate4 mode seed or finalize is required}"

FORMAL_OUTPUT_BASE="/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2"
SHARED_ROOT="$FORMAL_OUTPUT_BASE/$CHRONOTRANSPORT_REGISTRATION_COMMIT/shared"
[[ "$CHRONOTRANSPORT_GATE1_UNLOCK" == "$SHARED_ROOT/gate1/gate1_result.json" ]] || {
  echo "Gate-1 unlock must use canonical R path" >&2; exit 51;
}
[[ "$CHRONOTRANSPORT_GATES23_REPLAY" == "$SHARED_ROOT/gates23/gates23_replay.json" ]] || {
  echo "pre-Stage-C replay must use canonical R path" >&2; exit 52;
}
[[ "$CHRONOTRANSPORT_GATES23_REPORT" == "$SHARED_ROOT/gates23/gates23_report.json" ]] || {
  echo "pre-Stage-C report must use canonical R path" >&2; exit 53;
}
POST_ROOT="$SHARED_ROOT/post_stage_c_gate3"
[[ "$CHRONOTRANSPORT_POST_STAGE_C_REPLAY" == "$POST_ROOT/post_stage_c_replay.json" ]] || exit 54
[[ "$CHRONOTRANSPORT_POST_STAGE_C_REPORT" == "$POST_ROOT/post_stage_c_gate3_report.json" ]] || exit 55
[[ "$CHRONOTRANSPORT_POST_STAGE_C_UNLOCK" == "$POST_ROOT/post_stage_c_gate3_unlock.json" ]] || exit 56
[[ "$CHRONOTRANSPORT_POST_STAGE_C_TERMINAL" == "$POST_ROOT/terminal_marker.json" ]] || exit 57

[[ "$(git status --porcelain)" == "" ]] || { echo "dirty worktree" >&2; exit 58; }
[[ "$(git rev-parse HEAD)" == "$CHRONOTRANSPORT_REGISTRATION_COMMIT" ]] || {
  echo "HEAD must equal registration commit R" >&2; exit 59;
}
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
  echo "formal Gate4 requires a Slurm allocation and step" >&2; exit 60;
}
# Slurm owns CUDA_VISIBLE_DEVICES. This launcher never assigns or rewrites it;
# every seed process proves one visible device and uses logical cuda:0.

python -m py_compile \
  opentad/models/chronotransport/gate4.py \
  opentad/models/chronotransport/gate4_population.py \
  opentad/models/chronotransport/formal_gate4.py \
  opentad/models/chronotransport/profiler.py \
  opentad/models/chronotransport/runtime.py \
  tools/bata/chronotransport_r2_gate4_factory.py \
  tools/bata/run_chronotransport_r2_gate4.py \
  tools/bata/validate_chronotransport_r2_gate4.py

COMMON_ARGS=(
  --registration "$CHRONOTRANSPORT_REGISTRATION"
  --gate1-unlock "$CHRONOTRANSPORT_GATE1_UNLOCK"
  --gates23-replay "$CHRONOTRANSPORT_GATES23_REPLAY"
  --gates23-report "$CHRONOTRANSPORT_GATES23_REPORT"
  --phase-marker-3407 "$CHRONOTRANSPORT_PHASE_MARKER_3407"
  --phase-marker-3408 "$CHRONOTRANSPORT_PHASE_MARKER_3408"
  --phase-marker-3409 "$CHRONOTRANSPORT_PHASE_MARKER_3409"
  --post-stage-c-replay "$CHRONOTRANSPORT_POST_STAGE_C_REPLAY"
  --post-stage-c-report "$CHRONOTRANSPORT_POST_STAGE_C_REPORT"
  --post-stage-c-unlock "$CHRONOTRANSPORT_POST_STAGE_C_UNLOCK"
  --post-stage-c-terminal "$CHRONOTRANSPORT_POST_STAGE_C_TERMINAL"
)

case "$CHRONOTRANSPORT_GATE4_MODE" in
  seed)
    : "${CHRONOTRANSPORT_GATE4_SEED:?Gate4 seed is required}"
    case "$CHRONOTRANSPORT_GATE4_SEED" in 3407|3408|3409) ;; *) exit 61 ;; esac
    python tools/bata/run_chronotransport_r2_gate4.py \
      "${COMMON_ARGS[@]}" --seed "$CHRONOTRANSPORT_GATE4_SEED" --precheck-only
    if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
      echo "PRECHECK_OK"
      exit 0
    fi
    python tools/bata/run_chronotransport_r2_gate4.py \
      "${COMMON_ARGS[@]}" --seed "$CHRONOTRANSPORT_GATE4_SEED"
    echo "GATE4_SEED_SUCCESS:$CHRONOTRANSPORT_GATE4_SEED"
    ;;
  finalize)
    [[ "${PRECHECK_ONLY:-0}" != "1" ]] || {
      echo "Gate4 finalize has no precheck-only mode" >&2; exit 62;
    }
    set +e
    python tools/bata/run_chronotransport_r2_gate4.py \
      "${COMMON_ARGS[@]}" --finalize
    FINALIZE_STATUS=$?
    set -e
    [[ "$FINALIZE_STATUS" == "0" || "$FINALIZE_STATUS" == "2" ]] || {
      echo "Gate4 finalizer failed before a valid scientific terminal" >&2
      exit "$FINALIZE_STATUS"
    }
    python tools/bata/validate_chronotransport_r2_gate4.py \
      "${COMMON_ARGS[@]}" --finalize
    echo "GATE4_FINAL:$SHARED_ROOT/gate4/terminal_marker.json"
    [[ "$FINALIZE_STATUS" == "0" ]] || exit "$FINALIZE_STATUS"
    ;;
  *) echo "Gate4 mode must be seed or finalize" >&2; exit 63 ;;
esac
