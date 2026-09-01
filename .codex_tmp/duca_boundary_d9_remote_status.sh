source /etc/profile
set -u

echo '=== DATE ==='
date '+%Y-%m-%d %H:%M:%S %z'
echo '=== OLD R0 ==='
sacct -j 1179392 --format=JobID,JobName%40,State,Elapsed,ExitCode -P || true
echo '=== CORRECTED R0 ==='
sacct -j 1179517 --format=JobID,JobName%40,State,Elapsed,ExitCode,NodeList%20 -P || true
squeue -j 1179517 -o '%i|%j|%T|%M|%R' || true
RUN_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357
echo '=== CORRECTED R0 FILES ==='
find "${RUN_ROOT}" -maxdepth 3 -type f -printf '%P|%s|%TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort | tail -80 || true
echo '=== CORRECTED R0 LOG TAIL ==='
for log in "${RUN_ROOT}"/logs/*1179517*.out "${RUN_ROOT}"/logs/*1179517*.err; do
  [[ -f "${log}" ]] || continue
  echo "--- ${log}"
  tail -80 "${log}"
done
echo '=== CORRECTED R0 ERROR SCAN ==='
grep -R -n -E 'Traceback|OutOfMemory|OOM|non-finite|ValueError|\[.*FAIL.*\]' \
  "${RUN_ROOT}"/logs "${RUN_ROOT}"/r0_holdout_map 2>/dev/null || true
echo '=== CURRENT QUEUE ==='
squeue -u sczc063 -o '%i|%j|%T|%M|%R' | grep -E 'JOBID|duca|boundary|burst|r0' || true
echo '=== EXACT SNAPSHOT ==='
cd /data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_d9fb398_20260722
git rev-parse HEAD
git status --short
echo '=== REPLAY HASHES ==='
sha256sum \
  /data/run01/sczc063/yuzibo/tmp/duca_r0_determinism_d9fb398_v2_20260722/run1/holdout_families.jsonl \
  /data/run01/sczc063/yuzibo/tmp/duca_r0_determinism_d9fb398_v2_20260722/run2/holdout_families.jsonl
