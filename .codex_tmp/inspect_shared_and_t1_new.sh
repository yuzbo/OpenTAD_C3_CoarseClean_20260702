#!/usr/bin/env bash
set -u

SHARED=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
T1=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_919aa55_recovery_20260723_053508

date '+%F %T %z'
echo '=== SHARED JOB/STEPS ==='
squeue -j 1180493 -s || true
sstat -j 1180493.batch --format=JobID,AveCPU,MaxRSS,AveRSS,MaxDiskRead,MaxDiskWrite -P || true
echo '=== SHARED RECENT FILES ==='
find "${SHARED}" -type f -mmin -70 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null |
  sort | tail -30
echo '=== SHARED LOG TAILS ==='
find "${SHARED}" -type f \( -name '*.out' -o -name '*.err' -o -name train.out \) -mmin -70 -print0 2>/dev/null |
  while IFS= read -r -d '' log; do
    echo "--- ${log}"
    tail -8 "${log}" 2>/dev/null || true
  done

echo '=== NEW T1 LOG TAILS ==='
find "${T1}" -type f \( -name '*.out' -o -name '*.err' -o -name train.out \) -print0 2>/dev/null |
  while IFS= read -r -d '' log; do
    echo "--- ${log}"
    tail -12 "${log}" 2>/dev/null || true
  done
