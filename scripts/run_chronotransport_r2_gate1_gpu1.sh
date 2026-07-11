#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${CHRONOTRANSPORT_REGISTRATION_COMMIT:?registration commit R is required}"
: "${CHRONOTRANSPORT_GATE1_INPUT:?registered Gate 1 input JSON is required}"
: "${CHRONOTRANSPORT_GATE1_OUTPUT:?Gate 1 output JSON is required}"

[[ "$(git status --porcelain)" == "" ]] || { echo "dirty worktree" >&2; exit 20; }
[[ "$(git rev-parse HEAD)" == "$CHRONOTRANSPORT_REGISTRATION_COMMIT" ]] || {
  echo "HEAD must equal registration commit R" >&2; exit 21;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "1" ]] || { echo "CUDA_VISIBLE_DEVICES must equal 1" >&2; exit 22; }

python -m py_compile \
  opentad/models/chronotransport/protocol.py \
  opentad/models/chronotransport/adjudication.py \
  opentad/models/chronotransport/registration.py \
  tools/bata/run_chronotransport_r2_gate1.py

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "PRECHECK_OK"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
  echo "formal Gate 1 requires a Slurm allocation and step" >&2; exit 23;
}

python tools/bata/run_chronotransport_r2_gate1.py \
  --input "$CHRONOTRANSPORT_GATE1_INPUT" \
  --output "$CHRONOTRANSPORT_GATE1_OUTPUT"

