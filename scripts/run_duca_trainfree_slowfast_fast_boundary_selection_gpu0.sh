#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROVIDER="${PROVIDER:-slowfast_r50_fast}"
SUBSET="${SUBSET:-validation}"
MAX_VIDEOS="${MAX_VIDEOS:-20}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
BUDGET="${BUDGET:-384}"
CLIP_FRAMES="${CLIP_FRAMES:-32}"
FRAME_INTERVAL="${FRAME_INTERVAL:-2}"
CROP_SIZE="${CROP_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-2}"
BASELINES="${BASELINES:-boundary-first manual uniform oracle-actionness}"
RUN_TAG="${RUN_TAG:-duca_trainfree_slowfast_fast_boundary_$(date +%Y%m%d_%H%M%S_%z)}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
THUMOS14_ROOT="${THUMOS14_ROOT:-${YUZIBO_ROOT}/thumos14}"
ANNOTATION_JSON="${ANNOTATION_JSON:-${THUMOS14_ROOT}/annotations/thumos_14_anno.json}"
VIDEO_ROOTS="${VIDEO_ROOTS:-${THUMOS14_ROOT}/raw_data/video:${THUMOS14_ROOT}/test:${THUMOS14_ROOT}/train}"
OUT_ROOT="${OUT_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/trainfree_slowfast_fast_boundary/${RUN_TAG}}"

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export TORCH_HOME="${TORCH_HOME:-${BASE}/tmp/torch_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${TORCH_HOME}" "${OUT_ROOT}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "0" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  fail "expected GPU0 for SlowFast-Fast boundary diagnostic; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

[[ -f "${ANNOTATION_JSON}" ]] || fail "missing annotation: ${ANNOTATION_JSON}"
[[ -f "tools/bata/export_frozen_kinetics_actionness.py" ]] || fail "missing exporter"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON=python

ACTIONNESS_JSONL="${OUT_ROOT}/${PROVIDER}_${SUBSET}_actionness.jsonl"
ACTIONNESS_SUMMARY="${OUT_ROOT}/${PROVIDER}_${SUBSET}_actionness.summary.json"
ACTIONNESS_VALIDATION="${OUT_ROOT}/${PROVIDER}_${SUBSET}_actionness.validation.json"
COARSE_EVAL_JSONL="${OUT_ROOT}/${PROVIDER}_${SUBSET}_coarse_eval.actionness.jsonl"
COARSE_EVAL_SUMMARY="${OUT_ROOT}/${PROVIDER}_${SUBSET}_coarse_eval.summary.json"
COARSE_EVAL_VALIDATION="${OUT_ROOT}/${PROVIDER}_${SUBSET}_coarse_eval.validation.json"
SELECTION_AUDIT="${OUT_ROOT}/${PROVIDER}_${SUBSET}_selection.audit.jsonl"
SELECTION_SUMMARY="${OUT_ROOT}/${PROVIDER}_${SUBSET}_selection.summary.json"
SELECTION_VALIDATION="${OUT_ROOT}/${PROVIDER}_${SUBSET}_selection.validation.json"

echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] repo=${REPO_ROOT}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] provider=${PROVIDER} subset=${SUBSET} max_videos=${MAX_VIDEOS}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] dense_window_size=${DENSE_WINDOW_SIZE} budget=${BUDGET}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] clip_frames=${CLIP_FRAMES} frame_interval=${FRAME_INTERVAL} crop_size=${CROP_SIZE} batch_size=${BATCH_SIZE}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] selection_baselines=${BASELINES}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] roots=${VIDEO_ROOTS}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] out=${OUT_ROOT}"
echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] slurm_job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES} step_gpus=${SLURM_STEP_GPUS:-none}"

"${PYTHON}" -m py_compile \
  tools/bata/export_frozen_kinetics_actionness.py \
  tools/bata/eval_zero_shot_actionness.py \
  tools/bata/validate_zero_shot_actionness_eval.py \
  tools/bata/run_zero_shot_actionness_selection_eval.py \
  tools/bata/validate_zero_shot_selection_eval.py

"${PYTHON}" tools/bata/export_frozen_kinetics_actionness.py \
  --annotation-json "${ANNOTATION_JSON}" \
  --video-roots "${VIDEO_ROOTS}" \
  --output-jsonl "${ACTIONNESS_JSONL}" \
  --summary-json "${ACTIONNESS_SUMMARY}" \
  --provider "${PROVIDER}" \
  --subset "${SUBSET}" \
  --dense-window-size "${DENSE_WINDOW_SIZE}" \
  --clip-frames "${CLIP_FRAMES}" \
  --frame-interval "${FRAME_INTERVAL}" \
  --crop-size "${CROP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --max-videos "${MAX_VIDEOS}" \
  --slowfast-alpha 4 \
  --actionness-aux-weight 0.05 \
  2>&1 | tee "${OUT_ROOT}/export_${PROVIDER}.out"

"${PYTHON}" tools/bata/validate_zero_shot_actionness_eval.py \
  --actionness-jsonl "${ACTIONNESS_JSONL}" \
  --summary-json "${ACTIONNESS_SUMMARY}" \
  --validation-json "${ACTIONNESS_VALIDATION}" \
  2>&1 | tee "${OUT_ROOT}/validate_actionness.out"

"${PYTHON}" tools/bata/eval_zero_shot_actionness.py \
  --annotation-json "${ANNOTATION_JSON}" \
  --sample-jsonl "${ACTIONNESS_JSONL}" \
  --output-jsonl "${COARSE_EVAL_JSONL}" \
  --summary-json "${COARSE_EVAL_SUMMARY}" \
  --source-mode manual_jsonl \
  --manual-jsonl "${ACTIONNESS_JSONL}" \
  --recall-k "${BUDGET}" \
  2>&1 | tee "${OUT_ROOT}/coarse_eval.out"

"${PYTHON}" tools/bata/validate_zero_shot_actionness_eval.py \
  --actionness-jsonl "${COARSE_EVAL_JSONL}" \
  --summary-json "${COARSE_EVAL_SUMMARY}" \
  --validation-json "${COARSE_EVAL_VALIDATION}" \
  2>&1 | tee "${OUT_ROOT}/validate_coarse_eval.out"

"${PYTHON}" tools/bata/run_zero_shot_actionness_selection_eval.py \
  --annotation-json "${ANNOTATION_JSON}" \
  --actionness-jsonl "${COARSE_EVAL_JSONL}" \
  --audit-jsonl "${SELECTION_AUDIT}" \
  --summary-json "${SELECTION_SUMMARY}" \
  --budget "${BUDGET}" \
  --baselines ${BASELINES} \
  --boundary-radius "${BOUNDARY_RADIUS}" \
  2>&1 | tee "${OUT_ROOT}/selection_eval.out"

"${PYTHON}" tools/bata/validate_zero_shot_selection_eval.py \
  --audit-jsonl "${SELECTION_AUDIT}" \
  --summary-json "${SELECTION_SUMMARY}" \
  --validation-json "${SELECTION_VALIDATION}" \
  2>&1 | tee "${OUT_ROOT}/validate_selection.out"

echo "[DUCA_TRAINF_FREE_SLOWFAST_FAST] COMPLETE actionness=${ACTIONNESS_SUMMARY} coarse=${COARSE_EVAL_SUMMARY} selection=${SELECTION_SUMMARY}"
