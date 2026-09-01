#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
ARM="$ROOT/arms/two_stage_exact_uniform/official60"
EVAL="$ARM/eval.out"

date '+DATE %F %T %z'
squeue -j 1180637 -o '%.18i %.24j %.10T %.12M %.24R' || true
sacct -j 1180637 --format=JobID,JobName%30,State,Elapsed,ExitCode -P || true
echo '=== RESULT ==='
grep -n -E 'Average-mAP|mAP at tIoU|Testing Over|evaluation_status|complete_validation' "$EVAL" 2>/dev/null | tail -30 || true
echo '=== EVAL TAIL ==='
tail -12 "$EVAL" 2>/dev/null || true
echo '=== ARTIFACTS ==='
find "$ARM" -maxdepth 3 -type f \( -name '*terminal*.json' -o -name '*evaluation*.json' -o -name '*completion*.json' -o -name '*summary*.json' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort || true
exit 0
