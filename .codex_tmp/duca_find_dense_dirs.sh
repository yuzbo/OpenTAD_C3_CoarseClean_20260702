#!/usr/bin/env bash
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
find ${BASE}/projects/c3_lowres_action_probe -maxdepth 4 -type d -iname '*dense*' -print 2>/dev/null | head -80 || true
find ${BASE}/projects -maxdepth 3 -type d -iname '*dense*' -print 2>/dev/null | head -80 || true
