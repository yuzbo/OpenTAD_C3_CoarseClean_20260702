#!/usr/bin/env bash
set -u
date '+%F %T %z'
echo '===V8==='
squeue -j 1178989 -o '%i|%T|%M|%R' || true
sacct -j 1178989 --format=JobID,JobName%35,State,Elapsed,ExitCode -P || true
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_63e25eb_serial_20260721_2120
echo '===FILES==='
find "$ROOT" -maxdepth 5 -type f \( -name completion.json -o -name frontend_decision.json -o -name train.out \) -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -30
echo '===LOGTAIL==='
find "$ROOT" -type f -name train.out -print0 2>/dev/null | xargs -0 -r tail -n 4
echo '===CHECKPOINTS==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200 -type f -name epoch_131.pth -printf '%s|%p\n' 2>/dev/null
echo '===DISK==='
df -h /data/run01/sczc063/yuzibo | tail -1
