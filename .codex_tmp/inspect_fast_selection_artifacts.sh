#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107/arms/trainfree_slowfast_fast_fusion_r2q3

date '+DATE %F %T %z'
echo '=== FILES ==='
find "$ROOT" -maxdepth 5 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -100
echo '=== JSON CONTENT ==='
find "$ROOT" -maxdepth 5 -type f \( -name '*selection*.json' -o -name '*cost*.json' -o -name '*completion*.json' -o -name '*terminal*.json' -o -name '*summary*.json' \) -print0 2>/dev/null | while IFS= read -r -d '' p; do
  echo "--- $p"
  cat "$p"
done
echo '=== DIAGNOSTIC LOG LINES ==='
grep -R -n -E 'selected|boundary|transition|overlap|max.hole|cluster|distance|recall|cost|latency|FLOP' "$ROOT"/official60/*.out "$ROOT"/official60/*.log "$ROOT"/*.json 2>/dev/null | tail -250 || true

exit 0
