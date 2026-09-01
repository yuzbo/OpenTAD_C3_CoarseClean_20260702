#!/usr/bin/env bash
set -u

JOBS=1180490,1180491,1180492,1180493,1180494,1180495,1180496,1180502,1180503,1180504,1180505,1180556,1180557,1180637,1180638,1180639,1180644,1180652,1180653,1180654,1180674,1180685,1180686,1180687,1180696,1180697
ROOTS=(
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_4c5604b_retry_20260723_1107
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_487a178_recovery_20260723_034220
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_ca40c9c_retry_20260723_035756
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_cee4ccd_recovery_20260723_042025
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_mobilenet_e30db0f_retry_20260723_042025
)

date '+%F %T %z'
echo '=== QUEUE ==='
squeue -j "${JOBS}" -o '%.18i %.26j %.10T %.10M %.30R' || true
echo '=== ACCOUNTING ==='
sacct -j "${JOBS}" --format=JobIDRaw,JobName%30,State,Elapsed,ExitCode -P | grep -E '^[0-9]+\|' || true

echo '=== RECENT ERRORS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f \( -name '*.err' -o -name 'train.out' -o -name '*.out' \) -mmin -90 -print0 2>/dev/null |
    xargs -0 -r grep -Hn -E 'Traceback|CUDA out of memory|out of memory|non-finite|ValueError|\[FAIL\]|AMP replay exhausted' 2>/dev/null |
    tail -40 || true
done

echo '=== LATEST TRAIN PROGRESS ==='
for root in "${ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -type f -name train.out -mmin -90 -print0 2>/dev/null | while IFS= read -r -d '' log; do
    line="$(grep -E '\[Train\]: Epoch|\[[0-9]{3}\]\[[0-9]{5}/[0-9]{5}\]|Average-mAP|mAP at tIoU|Training Over' "${log}" 2>/dev/null | tail -1)"
    if [ -n "${line}" ]; then
      printf '%s\t%s\n' "${log}" "${line}"
    fi
  done
done

echo '=== SPARSE GATE FAILURE CONTEXT ==='
for factor in d1 d2 d3 d4; do
  gate_log=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329/suite/arms/${factor}/gate/full_model_gate.out
  echo "--- ${factor}"
  tail -80 "${gate_log}" 2>/dev/null || true
done

echo '=== SPARSE P0 CHECKPOINTS ==='
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329/suite/arms \
  -path '*/p0/work/gpu1_id0/checkpoint/epoch_19.pth' -type f -print -exec sha256sum {} \; 2>/dev/null | sort

echo '=== MOBILENET LAZY FAILURE CONTEXT ==='
for arm in trainfree_mobilenet_feature_change trainfree_mobilenet_semantic trainfree_mobilenet_fusion_r2q3; do
  train_log=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_trainfree_mobilenet_4c5604b_retry2_20260723_1115/arms/${arm}/official60/train.out
  echo "--- ${arm}"
  tail -100 "${train_log}" 2>/dev/null || true
done
