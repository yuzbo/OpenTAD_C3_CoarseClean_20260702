#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher
find ${ROOT} -maxdepth 7 -type f \( -name 'epoch_*.pth' -o -name '*summary*.json' -o -name 'completion.json' -o -name 'train.out' \) -printf '%p\t%s\n' 2>/dev/null | sort | tail -120 || true
