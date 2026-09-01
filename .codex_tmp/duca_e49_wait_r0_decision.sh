#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map
SUMMARY=${ROOT}/r0_summary.json
BOOTSTRAP=${ROOT}/r0_bootstrap.json

for _ in $(seq 1 60); do
  state=$(sacct -j 1179795 --format=State -n -X | awk 'NF {print $1; exit}')
  if [[ -f "${SUMMARY}" ]]; then
    echo "EVENT=SUMMARY_READY"
    cat "${SUMMARY}"
    exit 0
  fi
  if [[ -f "${BOOTSTRAP}" ]]; then
    echo "EVENT=BOOTSTRAP_READY"
    cat "${BOOTSTRAP}"
    exit 0
  fi
  if [[ "${state}" != RUNNING* ]]; then
    echo "EVENT=JOB_STATE_${state}"
    sacct -j 1179795,1179796 --format=JobID,JobName%32,State,Elapsed,ExitCode -P -X
    exit 0
  fi
  sleep 5
done

echo "EVENT=TIMEOUT_STILL_RUNNING"
sacct -j 1179795,1179796 --format=JobID,JobName%32,State,Elapsed,ExitCode -P -X
