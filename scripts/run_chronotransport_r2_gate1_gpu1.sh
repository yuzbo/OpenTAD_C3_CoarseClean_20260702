#!/usr/bin/env bash
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CHRONOTRANSPORT_REGISTRATION_COMMIT:?registration commit R is required}"
: "${CHRONOTRANSPORT_REGISTRATION:?immutable registration JSON is required}"
: "${CHRONOTRANSPORT_GATE1_INPUT:?registered Gate 1 input JSON is required}"
: "${CHRONOTRANSPORT_GATE1_OUTPUT:?Gate 1 output JSON is required}"
: "${CHRONOTRANSPORT_RUN_ROOT:?registered run root is required}"

FORMAL_OUTPUT_BASE="/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2"
EXPECTED_RUN_ROOT="$FORMAL_OUTPUT_BASE/$CHRONOTRANSPORT_REGISTRATION_COMMIT/shared/gate1"
[[ "$CHRONOTRANSPORT_RUN_ROOT" == "$EXPECTED_RUN_ROOT" ]] || {
  echo "run root must equal the fixed formal R-derived Gate 1 root" >&2; exit 19;
}
[[ "$CHRONOTRANSPORT_GATE1_INPUT" == "$EXPECTED_RUN_ROOT/gate1_input.json" ]] || {
  echo "Gate 1 input must use the fixed canonical filename" >&2; exit 26;
}
[[ "$CHRONOTRANSPORT_GATE1_OUTPUT" == "$EXPECTED_RUN_ROOT/gate1_result.json" ]] || {
  echo "Gate 1 result must use the fixed canonical filename" >&2; exit 27;
}
TERMINAL_MARKER="$EXPECTED_RUN_ROOT/gate1_terminal.json"

[[ "$(git status --porcelain)" == "" ]] || { echo "dirty worktree" >&2; exit 20; }
[[ "$(git rev-parse HEAD)" == "$CHRONOTRANSPORT_REGISTRATION_COMMIT" ]] || {
  echo "HEAD must equal registration commit R" >&2; exit 21;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] || { echo "CUDA_VISIBLE_DEVICES must equal 1" >&2; exit 22; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
  echo "formal Gate 1 requires a Slurm allocation and step" >&2; exit 23;
}
PHYSICAL_GPU_IDS="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
[[ "$PHYSICAL_GPU_IDS" == "1" ]] || {
  echo "formal Gate 1 requires one protected physical GPU1 allocation" >&2; exit 28;
}

RUN_LOCK="$EXPECTED_RUN_ROOT/.gate1.run.lock"
LOCK_HELD=0
release_run_lock() {
  if [[ "$LOCK_HELD" == "1" ]]; then
    rmdir -- "$RUN_LOCK" || true
    LOCK_HELD=0
  fi
}
if ! mkdir "$RUN_LOCK"; then
  echo "formal Gate 1 run root is already locked" >&2
  exit 29
fi
LOCK_HELD=1
trap 'release_run_lock' EXIT

python -m py_compile \
  opentad/models/chronotransport/protocol.py \
  opentad/models/chronotransport/adjudication.py \
  opentad/models/chronotransport/controls.py \
  opentad/models/chronotransport/full_stack_profiler.py \
  opentad/models/chronotransport/registration.py \
  opentad/models/chronotransport/runtime.py \
  tools/bata/chronotransport_r2_opentad_profile_backend.py \
  tools/bata/chronotransport_r2_profile_factory.py \
  tools/bata/profile_chronotransport_r2_full_stack.py \
  tools/bata/validate_chronotransport_r2_precheck.py \
  tools/bata/run_chronotransport_r2_gate1.py

python tools/bata/validate_chronotransport_r2_precheck.py \
  --registration "$CHRONOTRANSPORT_REGISTRATION" \
  --repository-root "$ROOT" \
  --registration-commit "$CHRONOTRANSPORT_REGISTRATION_COMMIT" \
  --gate1-input "$CHRONOTRANSPORT_GATE1_INPUT" \
  --gate1-output "$CHRONOTRANSPORT_GATE1_OUTPUT" \
  --terminal-marker "$TERMINAL_MARKER" \
  --output-root "$CHRONOTRANSPORT_RUN_ROOT"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "PRECHECK_OK"
  exit 0
fi

case "$TERMINAL_MARKER" in
  "$CHRONOTRANSPORT_RUN_ROOT"/*) ;;
  *) echo "terminal marker escapes registered run root" >&2; exit 25 ;;
esac
TERMINAL_WRITTEN=0
RUN_STARTED=0

write_terminal_marker() {
  local state="$1"
  case "$state" in
    SUCCESS|FAIL|STOPPED|INVALID_IMPLEMENTATION) ;;
    *) echo "invalid terminal state" >&2; return 97 ;;
  esac
  local output_sha256=""
  if [[ -s "$CHRONOTRANSPORT_GATE1_OUTPUT" ]]; then
    output_sha256="$(sha256sum "$CHRONOTRANSPORT_GATE1_OUTPUT" | awk '{print $1}')"
  fi
  local TEMP_MARKER
  TEMP_MARKER="$(mktemp "${TERMINAL_MARKER}.tmp.XXXXXX")"
  printf '{"output_sha256":"%s","registration_commit":"%s","schema":"chronotransport-r2-gate1-terminal-v1","state":"%s"}\n' \
    "$output_sha256" "$CHRONOTRANSPORT_REGISTRATION_COMMIT" "$state" \
    > "$TEMP_MARKER"
  if ! ln -- "$TEMP_MARKER" "$TERMINAL_MARKER"; then
    rm -f -- "$TEMP_MARKER"
    echo "terminal marker already exists; refusing to overwrite" >&2
    return 98
  fi
  rm -f -- "$TEMP_MARKER"
  TERMINAL_WRITTEN=1
  echo "GATE1_${state}:$TERMINAL_MARKER"
}

handle_exit() {
  local rc="$1"
  if [[ "$RUN_STARTED" == "1" && "$TERMINAL_WRITTEN" != "1" ]]; then
    write_terminal_marker INVALID_IMPLEMENTATION || true
  fi
  release_run_lock
  return "$rc"
}

handle_signal() {
  local signal="$1"
  local rc="$2"
  trap - EXIT INT TERM
  write_terminal_marker STOPPED || true
  echo "Gate 1 stopped by $signal" >&2
  exit "$rc"
}

trap 'handle_exit $?' EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

RUN_STARTED=1
set +e
python tools/bata/run_chronotransport_r2_gate1.py \
  --input "$CHRONOTRANSPORT_GATE1_INPUT" \
  --output "$CHRONOTRANSPORT_GATE1_OUTPUT" \
  --repository-root "$ROOT" \
  --registration-commit "$CHRONOTRANSPORT_REGISTRATION_COMMIT" \
  --registration-relpath "$(realpath --relative-to="$ROOT" "$CHRONOTRANSPORT_REGISTRATION")"
GATE_RC=$?
set -e

case "$GATE_RC" in
  0)
    [[ -s "$CHRONOTRANSPORT_GATE1_OUTPUT" ]] || {
      echo "Gate 1 PASS returned without an exact output artifact" >&2; exit 24;
    }
    write_terminal_marker SUCCESS
    ;;
  2)
    [[ -s "$CHRONOTRANSPORT_GATE1_OUTPUT" ]] || {
      echo "Gate 1 FAIL returned without an exact output artifact" >&2; exit 24;
    }
    write_terminal_marker FAIL
    ;;
  *)
    echo "Gate 1 implementation failed with exit code $GATE_RC" >&2
    exit "$GATE_RC"
    ;;
esac
exit "$GATE_RC"
