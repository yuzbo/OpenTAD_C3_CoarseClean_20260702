#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
date '+%F %T %z'
ps -eo pid,etime,time,%cpu,rss,cmd | grep -E 'finalize_duca_r0_boundary_burst|duca-r0-bootstrap' | grep -v grep || true
sacct -j 1179517,1179533,1179602 --format=JobID,JobName%28,State,Elapsed,TotalCPU,ExitCode,NodeList%16 -P || true
squeue -j 1179517,1179533,1179602 || true
sstat -j 1179517.batch --format=JobID,AveCPU,AveRSS,MaxRSS,AveDiskRead,AveDiskWrite -P || true
