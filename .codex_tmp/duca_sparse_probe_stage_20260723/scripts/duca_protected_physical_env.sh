#!/usr/bin/env bash

[[ -n "${BASE:-}" ]] || {
  echo "[DUCA_PROTECTED_PHYSICAL_ENV][FAIL] BASE must be set" >&2
  return 1 2>/dev/null || exit 1
}

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHON="${BASE}/conda_envs/opentad/bin/python"
export DUCA_PROTECTED_ADATAD_PRETRAIN="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
export C3_OFFICIAL_ACTION_SEG_REPOS="${BASE}/projects/external_official_action_segmentation_repos_20260702"
export THUMOS14_ANNOTATION_PATH="${BASE}/thumos14/annotations/thumos_14_anno.json"
export THUMOS14_CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
