#!/usr/bin/env bash
set -euo pipefail
R=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3
echo "DATE=$(date -Is)"
squeue -j 1179795 -o '%i|%T|%M|%R'
sstat -j 1179795.batch --format=JobID,AveCPU,MaxRSS,MaxVMSize,AvePages,AveDiskRead,AveDiskWrite -P || true
echo FILE_ACTIVITY
find ${R}/r0_holdout_map/map ${R}/logs -type f -printf '%T@\t%s\t%p\n' 2>/dev/null | sort -nr | head -30
echo PROCESSES
scontrol show job 1179795 | grep -E 'JobState=|RunTime=|NodeList=|Command=|WorkDir=' || true
