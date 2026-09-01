#!/usr/bin/env bash
set -euo pipefail
P=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map/map/R2Q3_privileged_boundary_burst/test.out
echo "DATE=$(date -Is)"
grep -E 'Average-mAP|mAP at tIoU|Testing Over' ${P} || true
tail -8 ${P}
