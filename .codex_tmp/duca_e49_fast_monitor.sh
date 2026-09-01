#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
PREFIX=${BASE}/duca_boundary_e49ef69_formal_20260722_155037
R03=${PREFIX}_r0_r3
R4=${PREFIX}_r4
R5=${PREFIX}_r5
IDS=1179795,1179796,1179797,1179798,1179799,1179825,1179826,1179827,1179861,1179862,1179863,1179864,1179865

echo "DATE=$(date -Is)"
sacct -j ${IDS} --format=JobID,JobName%40,State,Elapsed,ExitCode -P | grep -v '\.'
echo ARTIFACTS
find ${R03} ${R4} ${R5} -maxdepth 4 -type f \
  \( -name '*summary*.json' -o -name '*decision*.json' -o -name '*results*.json' \
     -o -name '*evaluation*.json' -o -name '*alignment*.json' -o -name '*gate*.json' \) \
  -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -80
echo MAP_LINES
grep -R -n -E 'Average-mAP|Avg-mAP|mAP at tIoU|selected_weakest|bootstrap|HEADROOM|GO_|KILL_' \
  ${R03}/r0_holdout_map ${R03}/logs 2>/dev/null | tail -100 || true
echo ERROR_SCAN
hits=$(grep -R -n -E 'Traceback|CUDA out of memory|OutOfMemory|non-finite loss|ValueError|\[.*FAIL.*\]' \
  ${R03} ${R4} ${R5} 2>/dev/null | head -60 || true)
if [[ -n ${hits} ]]; then printf '%s\n' "${hits}"; else echo CLEAN; fi
