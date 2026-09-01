#!/usr/bin/env bash
set -euo pipefail
date '+%F %T %z'
squeue -u "${USER}" -o '%i|%j|%T|%M|%R'
sacct -j 1179517,1179533,1179602 \
  --format=JobID,JobName%32,State,Elapsed,ExitCode -P || true
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357
find "${ROOT}"/r0_holdout_map -maxdepth 2 -type f \
  \( -name '*summary*.json' -o -name '*decision*.json' -o -name '*bootstrap*.json' \) \
  -printf '%p\n' 2>/dev/null | sort
tail -80 "${ROOT}"/logs/burst_r0_d9fb398-1179517.out 2>/dev/null || true
tail -40 "${ROOT}"/logs/burst_r0_d9fb398-1179517.err 2>/dev/null || true
