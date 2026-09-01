source /etc/profile
set -u
echo '=== DATE ==='
date '+%Y-%m-%d %H:%M:%S %z'
echo '=== REPLAY PROCESS ==='
ps -eo pid,etime,%cpu,%mem,cmd | grep -E 'build_duca_r0_boundary_burst_oracles|codex_duca_boundary_d9f_replay' | grep -v grep || true
echo '=== REPLAY FILES ==='
ROOT=/data/run01/sczc063/yuzibo/tmp/duca_r0_determinism_d9fb398_v2_20260722
find "$ROOT" -maxdepth 2 -type f -printf '%P|%s|%TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort || true
