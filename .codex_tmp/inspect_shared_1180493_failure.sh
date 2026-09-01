#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
date '+%F %T %z'
sacct -j 1180493 --format=JobIDRaw,JobName%35,State,Elapsed,ExitCode -P || true
echo '=== RECENT FILES ==='
find "${ROOT}" -type f -mmin -55 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null |
  sort | tail -80
echo '=== TOP-LEVEL LOG TAILS ==='
find "${ROOT}" -maxdepth 3 -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) -print0 2>/dev/null |
  while IFS= read -r -d '' log; do
    echo "--- ${log}"
    tail -80 "${log}" 2>/dev/null || true
  done
echo '=== R0 JSON ==='
find "${ROOT}/bundles/shared_bootstrap/r0_holdout_map" -maxdepth 2 -type f -name '*.json' -print 2>/dev/null | sort
for name in r0_summary.json r0_decision.json decision.json bootstrap_summary.json; do
  path="${ROOT}/bundles/shared_bootstrap/r0_holdout_map/${name}"
  if [[ -f "${path}" ]]; then
    echo "--- ${path}"
    cat "${path}"
  fi
done
echo '=== ERRORS ==='
find "${ROOT}" -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) -print0 2>/dev/null |
  xargs -0 -r grep -Hn -E 'Traceback|ValueError|RuntimeError|\[FAIL\]|ERROR|HOLD|KILL|headroom|selected_weakest' 2>/dev/null |
  tail -100 || true
