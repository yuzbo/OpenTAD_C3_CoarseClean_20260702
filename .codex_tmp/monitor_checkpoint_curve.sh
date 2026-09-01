#!/usr/bin/env bash
set -u
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_checkpoint_curve_20260723_100219
date '+DATE %F %T %z'
squeue -j 1180868,1180869 -o '%.18i %.24j %.10T %.10M %.32R' || true
sacct -j 1180868,1180869 --format=JobIDRaw,JobName%28,State,Elapsed,ExitCode -P | grep -E '^[0-9]+\|' || true
find "${ROOT}" -type f \( -name 'evaluation.json' -o -name 'summary.json' -o -name '*.err' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -40
grep -R -n -E 'Traceback|CUDA out of memory|ValueError|RuntimeError|FAIL' "${ROOT}/logs" "${ROOT}"/*/epoch_*/eval.out 2>/dev/null | tail -30 || true
