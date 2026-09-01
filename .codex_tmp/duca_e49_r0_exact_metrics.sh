#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map

echo "DATE=$(date -Is)"
for family in \
  A_exact_uniform \
  R2Q3_privileged_boundary_burst \
  R4Q5_privileged_boundary_burst \
  Z_unrestricted_gt_oracle; do
  path="${ROOT}/map/${family}/metrics.json"
  echo "FAMILY=${family}"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  else
    echo "PENDING"
  fi
done

echo "ROOT_FILES"
find "${ROOT}" -maxdepth 1 -type f -printf '%f\n' | sort

for name in r0_holdout_map.summary.json r0_holdout_map.bootstrap.json r0_holdout_map.decision.json; do
  path="${ROOT}/${name}"
  echo "ARTIFACT=${name}"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  else
    echo "PENDING"
  fi
done

sacct -j 1179795,1179796 --format=JobID,JobName%32,State,Elapsed,ExitCode -P -X || true
