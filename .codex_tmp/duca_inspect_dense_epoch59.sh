#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
echo FILES
find ${ROOT} -maxdepth 4 -type f -printf '%p\t%s\n' | sort | head -160
echo LOG_TAIL
tail -80 ${ROOT}/train.out
echo CHECKPOINT_SHA
sha256sum ${ROOT}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
