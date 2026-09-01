#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
PREFIX=${BASE}/duca_boundary_e49ef69_formal_20260722_155037
R03=${PREFIX}_r0_r3
R4=${PREFIX}_r4
R5=${PREFIX}_r5
IDS=1179795,1179796,1179797,1179798,1179799,1179825,1179826,1179827,1179861,1179862,1179863,1179864,1179865

echo "DATE=$(date -Is)"
echo HASHES
for file in ${R03}/jobs.tsv ${R03}/jobs.complete.json ${R5}/jobs.tsv ${R5}/site_bundle_submission.json ${PREFIX}_deployment.tsv; do
  printf '%s\t%s\n' "$(sha256sum ${file} | awk '{print $1}')" ${file}
done
echo STATES
sacct -j ${IDS} --format=JobID,JobName%42,State,Elapsed,ExitCode -P
echo QUEUE
squeue -j ${IDS//,/,} -o '%i|%j|%T|%M|%R'
echo ERROR_SCAN
hits=$(grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite loss|ValueError|\[.*FAIL.*\]' \
  ${R03} ${R4} ${R5} 2>/dev/null | head -80 || true)
if [[ -n ${hits} ]]; then printf '%s\n' "${hits}"; else echo CLEAN; fi
echo R0_FILES
find ${R03}/r0_holdout_map -maxdepth 2 -type f -printf '%p\t%s bytes\n' 2>/dev/null | sort | tail -40 || true
echo R0_LOG_TAILS
find ${R03} -type f \( -name '*.out' -o -name '*.err' -o -name 'train.out' \) -print0 2>/dev/null \
  | while IFS= read -r -d '' file; do echo "--- ${file}"; tail -25 "${file}"; done
