#!/usr/bin/env bash
set -euo pipefail
R=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3/r0_holdout_map/map/A_exact_uniform
echo "DATE=$(date -Is)"
echo TEST_TAIL
tail -160 ${R}/test.out
echo JSONS
find ${R} -maxdepth 3 -type f -name '*.json' -printf '%s\t%p\n' | sort -n
