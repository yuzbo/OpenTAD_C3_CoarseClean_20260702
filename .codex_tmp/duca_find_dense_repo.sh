#!/usr/bin/env bash
set -euo pipefail
for repo in \
  /data/run01/sczc063/yuzibo/projects/opentad_dense_teacher_366b9951ef39_20260706_233128 \
  /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/code; do
  echo REPO=${repo}
  if [[ -d ${repo}/.git ]]; then
    git -C ${repo} rev-parse HEAD
    git -C ${repo} status --porcelain --untracked-files=normal | head -20
    test -f ${repo}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py && sha256sum ${repo}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py || true
  fi
done
