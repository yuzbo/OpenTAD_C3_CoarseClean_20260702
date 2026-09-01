#!/usr/bin/env bash
set -euo pipefail
CFG=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800/work_dir/gpu1_id0/c3_dense_adatad_teacher_full_train.py
sed -n '1,220p' ${CFG}
