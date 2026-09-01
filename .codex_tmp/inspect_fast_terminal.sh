#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107

date '+%F %T %z'
echo '=== JOB ==='
squeue -j 1180653 -o '%.18i %.28j %.10T %.10M %.34R' || true
sacct -j 1180653 --format=JobIDRaw,JobName%32,State,Elapsed,ExitCode -P || true
echo '=== RECENT FILES ==='
find "${ROOT}" -type f -mmin -45 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -60
echo '=== METRICS ==='
find "${ROOT}" -type f \( -name '*.out' -o -name '*.log' -o -name '*.json' -o -name '*.txt' \) -print0 2>/dev/null |
  xargs -0 -r grep -Hn -E 'Average-mAP|Avg-mAP|mAP at tIoU|mAP@|Testing Over|Training Over|terminal|state_dict_ema|Traceback|CUDA out of memory|non-finite|ValueError|\[FAIL\]' 2>/dev/null |
  tail -120 || true
echo '=== CHECKPOINTS ==='
find "${ROOT}" -type f \( -name 'epoch_59.pth' -o -name '*terminal*.json' -o -name '*completion*.json' -o -name '*summary*.json' \) -printf '%s %p\n' 2>/dev/null | sort
echo '=== EVAL TAIL ==='
tail -35 "${ROOT}/arms/trainfree_slowfast_fast_fusion_r2q3/official60/eval.out" 2>/dev/null || true
echo '=== COMPLETION JSON ==='
cat "${ROOT}/arms/trainfree_slowfast_fast_fusion_r2q3/completion.json" 2>/dev/null || true
echo '=== TERMINAL EVALUATION JSON ==='
cat "${ROOT}/arms/trainfree_slowfast_fast_fusion_r2q3/official60/terminal_evaluation.json" 2>/dev/null || true
