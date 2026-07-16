#!/usr/bin/env bash
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CHRONOTRANSPORT_REGISTRATION_COMMIT:?registration commit R is required}"
: "${CHRONOTRANSPORT_REGISTRATION:?immutable registration JSON is required}"
: "${CHRONOTRANSPORT_GATE1_UNLOCK:?Gate-1 unlock is required}"
: "${CHRONOTRANSPORT_GATES23_REPLAY:?Gates-2/3 replay is required}"
: "${CHRONOTRANSPORT_GATES23_REPORT:?Gates-2/3 PASS report is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3407:?Stage-B marker 3407 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3408:?Stage-B marker 3408 is required}"
: "${CHRONOTRANSPORT_PHASE_MARKER_3409:?Stage-B marker 3409 is required}"
: "${CHRONOTRANSPORT_MANIFEST:?registered manifest is required}"
: "${CHRONOTRANSPORT_MEDIA_REGISTRY:?registered media registry is required}"
: "${CHRONOTRANSPORT_CONFIG_IDENTITY:?registered config identity is required}"
: "${CHRONOTRANSPORT_STAGE_C_SEED:?Stage-C seed is required}"
: "${CHRONOTRANSPORT_STAGE_C_RUN_ROOT:?canonical Stage-C run root is required}"

case "$CHRONOTRANSPORT_STAGE_C_SEED" in
  3407|3408|3409) ;;
  *) echo "Stage-C seed must be 3407, 3408, or 3409" >&2; exit 31 ;;
esac

FORMAL_OUTPUT_BASE="/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2"
EXPECTED_ROOT="$FORMAL_OUTPUT_BASE/$CHRONOTRANSPORT_REGISTRATION_COMMIT/$CHRONOTRANSPORT_STAGE_C_SEED/stage_c"
[[ "$CHRONOTRANSPORT_STAGE_C_RUN_ROOT" == "$EXPECTED_ROOT" ]] || {
  echo "Stage-C run root must equal the fixed R/seed/stage_c path" >&2; exit 32;
}
OUTPUT="$EXPECTED_ROOT/stage_c_paired_complete.pth"
LEDGER="$EXPECTED_ROOT/stage_c_paired_ledger.jsonl"
TERMINAL="$EXPECTED_ROOT/stage_c_paired_terminal.json"

[[ "$(git status --porcelain)" == "" ]] || { echo "dirty worktree" >&2; exit 33; }
[[ "$(git rev-parse HEAD)" == "$CHRONOTRANSPORT_REGISTRATION_COMMIT" ]] || {
  echo "HEAD must equal registration commit R" >&2; exit 34;
}
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
  echo "formal Stage C requires a Slurm allocation and step" >&2; exit 35;
}
# Slurm exclusively owns CUDA_VISIBLE_DEVICES.  This launcher never assigns,
# rewrites, normalizes, or appends to it; the formal environment guard proves
# exactly one logical device and the process uses cuda:0.

python -m py_compile \
  opentad/models/chronotransport/formal_stage_c.py \
  opentad/models/chronotransport/stage_c.py \
  opentad/models/chronotransport/runtime.py \
  tools/bata/chronotransport_r2_stage_c_factory.py \
  tools/bata/train_chronotransport_r2_stage_c.py \
  tools/bata/train_chronotransport_r2_matched_dense.py \
  tools/bata/validate_chronotransport_r2_stage_c.py

COMMON_ARGS=(
  --registration "$CHRONOTRANSPORT_REGISTRATION"
  --gate1-unlock "$CHRONOTRANSPORT_GATE1_UNLOCK"
  --gates23-replay "$CHRONOTRANSPORT_GATES23_REPLAY"
  --gates23-report "$CHRONOTRANSPORT_GATES23_REPORT"
  --phase-marker-3407 "$CHRONOTRANSPORT_PHASE_MARKER_3407"
  --phase-marker-3408 "$CHRONOTRANSPORT_PHASE_MARKER_3408"
  --phase-marker-3409 "$CHRONOTRANSPORT_PHASE_MARKER_3409"
  --manifest "$CHRONOTRANSPORT_MANIFEST"
  --media-registry "$CHRONOTRANSPORT_MEDIA_REGISTRY"
  --config-identity "$CHRONOTRANSPORT_CONFIG_IDENTITY"
  --seed "$CHRONOTRANSPORT_STAGE_C_SEED"
  --output "$OUTPUT"
  --ledger "$LEDGER"
  --terminal "$TERMINAL"
)

python tools/bata/train_chronotransport_r2_stage_c.py \
  "${COMMON_ARGS[@]}" \
  --precheck-only

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "PRECHECK_OK"
  exit 0
fi

RESUME_ARGS=()
if [[ -n "${CHRONOTRANSPORT_STAGE_C_RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$CHRONOTRANSPORT_STAGE_C_RESUME")
fi

python tools/bata/train_chronotransport_r2_stage_c.py \
  "${COMMON_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

python tools/bata/validate_chronotransport_r2_stage_c.py \
  "${COMMON_ARGS[@]}"

echo "STAGE_C_SUCCESS:$TERMINAL"
