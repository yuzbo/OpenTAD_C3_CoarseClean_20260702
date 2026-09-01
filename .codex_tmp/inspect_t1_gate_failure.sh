#!/usr/bin/env bash
set -u

ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_t1_trainfree_f81aef4_formal_20260723_1048
ARMS=(boundary_burst_r2q3_g0 t1_true_time_residual_g0 t1_reversed_time_residual_g0)

date '+%F %T %z'
sacct -j 1180637,1180638 --format=JobIDRaw,JobName%30,State,Elapsed,ExitCode -P || true

for arm in "${ARMS[@]}"; do
  echo "=== ${arm} ==="
  p0="${ROOT}/arms/${arm}/p0/work/gpu1_id0/checkpoint/epoch_19.pth"
  if [[ -f "${p0}" ]]; then
    stat -c 'checkpoint_size=%s' "${p0}"
    sha256sum "${p0}"
  else
    echo checkpoint_missing
  fi
  echo '-- gate tail --'
  tail -80 "${ROOT}/arms/${arm}/gate/full_model_gate.out" 2>/dev/null || true
  echo '-- p0 terminal --'
  grep -E '\[Train\]: Epoch|Training Over|non-finite|Traceback|FAIL' \
    "${ROOT}/arms/${arm}/p0/train.out" 2>/dev/null | tail -8 || true
done

echo '=== EXACT UNIFORM ==='
grep -E '\[Train\]: Epoch|Average-mAP|mAP at tIoU|Training Over|Traceback|non-finite|FAIL' \
  "${ROOT}/arms/two_stage_exact_uniform/official60/train.out" 2>/dev/null | tail -12 || true
