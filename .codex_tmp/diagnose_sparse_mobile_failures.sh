#!/usr/bin/env bash
set -u

SPARSE_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329/suite

echo '=== SPARSE D1 FULL GATE TRACE ==='
cat "${SPARSE_ROOT}/arms/d1/gate/full_model_gate.out" 2>/dev/null || true

echo '=== SPARSE FOUR CHECKPOINT HASHES ==='
for factor in d1 d2 d3 d4; do
  checkpoint="${SPARSE_ROOT}/arms/${factor}/p0/work/gpu1_id0/checkpoint/epoch_19.pth"
  if [ -f "${checkpoint}" ]; then
    stat -c '%n|%s' "${checkpoint}"
    sha256sum "${checkpoint}"
  else
    echo "MISSING ${checkpoint}"
  fi
done

echo '=== SPARSE CONFIG AND RECEIPTS ==='
find "${SPARSE_ROOT}/arms" -maxdepth 4 -type f \
  \( -name 'p0_summary.json' -o -name 'p0_winner.json' -o -name 'variant_validation.json' -o -name '*receipt*.json' \) \
  -print 2>/dev/null | sort

echo '=== MOBILENET EXACT TRACE ==='
MOBILE_LOG=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115/arms/trainfree_mobilenet_feature_change/official60/train.out
sed -n '660,725p' "${MOBILE_LOG}" 2>/dev/null || true
