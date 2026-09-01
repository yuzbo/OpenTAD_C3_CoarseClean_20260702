#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
echo "DATE=$(date -Is)"
echo ROOTS
mapfile -t roots < <(ls -dt "${BASE}"/duca_boundary_e49ef69_formal_*_r0_r3 2>/dev/null | head -5 || true)
printf '%s\n' "${roots[@]:-}"
R="${roots[0]:-}"
echo "SELECTED_ROOT=${R}"
if [[ -n "${R}" ]]; then
  echo JOBS
  cat "${R}/jobs.tsv"
  echo SEAL
  ls -l "${R}/jobs.complete.json" 2>/dev/null || true
  echo SUBMISSION
  ls -l "${R}/submission"
fi
echo QUEUE
squeue -u sczc063 -o '%i|%j|%T|%M|%R' | grep -E 'duca|burst|JOBID' || true
