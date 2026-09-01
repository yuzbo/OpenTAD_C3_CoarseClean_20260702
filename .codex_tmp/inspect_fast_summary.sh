#!/usr/bin/env bash
set -u

ARM=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107/arms/trainfree_slowfast_fast_fusion_r2q3

date '+%F %T %z'
echo '=== COMPLETION ==='
cat "${ARM}/completion.json" 2>/dev/null || true
echo '=== TERMINAL EVALUATION ==='
cat "${ARM}/official60/terminal_evaluation.json" 2>/dev/null || true
echo '=== QUALITY/COST ARTIFACTS ==='
find "${ARM}" -type f \( -iname '*quality*.json' -o -iname '*selection*.json' -o -iname '*cost*.json' -o -iname '*audit*.json' \) -printf '%s %p\n' 2>/dev/null | sort
