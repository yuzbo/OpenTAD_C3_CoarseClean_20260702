#!/usr/bin/env bash
set -euo pipefail

for repo in \
  /data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_ca40c9c_20260723 \
  /data/run01/sczc063/yuzibo/projects/opentad_duca_t1_26ce86d_20260723; do
  echo "=== ${repo} ==="
  sed -n '170,215p' "${repo}/scripts/run_duca_independent_official60_gpu1.sh" 2>/dev/null || true
  sed -n '250,285p' "${repo}/scripts/run_duca_independent_official60_gpu1.sh" 2>/dev/null || true
  python "${repo}/tools/test.py" --help 2>&1 | sed -n '1,180p' || true
done
