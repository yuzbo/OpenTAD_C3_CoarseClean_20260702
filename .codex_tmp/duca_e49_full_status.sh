#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
PREFIX=${BASE}/duca_boundary_e49ef69_formal_20260722_155037
R03=${PREFIX}_r0_r3
R4=${PREFIX}_r4
R5=${PREFIX}_r5

echo "DATE=$(date -Is)"
echo R03_JOBS
cat ${R03}/jobs.tsv
echo R03_SEAL
ls -l ${R03}/jobs.complete.json
echo R4
find ${R4} -maxdepth 2 -type f -printf '%p\n' 2>/dev/null | sort || true
echo R5_COUNTS
wc -l ${R5}/cells.tsv ${R5}/costs.tsv 2>/dev/null || true
echo R5_JOBS
if [[ -f ${R5}/jobs.tsv ]]; then
  cat ${R5}/jobs.tsv
  echo "R5_JOB_ROWS=$(($(wc -l < ${R5}/jobs.tsv)-1))"
else
  echo MISSING
fi
echo RECEIPT
ls -l ${PREFIX}_deployment.tsv* 2>/dev/null || true
echo QUEUE
squeue -u sczc063 -o '%i|%j|%T|%M|%R' | head -100
echo RECENT_SACCT
sacct -S 2026-07-22T15:45:00 -u sczc063 --format=JobID,JobName%48,State,Elapsed,ExitCode -P | tail -100
