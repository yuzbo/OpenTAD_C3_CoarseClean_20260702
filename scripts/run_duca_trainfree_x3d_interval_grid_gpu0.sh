#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_X3D_INTERVAL_GRID][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SUBSET="${SUBSET:-validation}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
BUDGET="${BUDGET:-384}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-2}"
BATCH_SIZE_XS="${BATCH_SIZE_XS:-16}"
BATCH_SIZE_S="${BATCH_SIZE_S:-12}"
CROP_SIZE_XS="${CROP_SIZE_XS:-160}"
CROP_SIZE_S="${CROP_SIZE_S:-182}"
PROVIDERS="${PROVIDERS:-x3d_xs x3d_s}"
FRAME_INTERVALS="${FRAME_INTERVALS:-1 2 4}"
RUN_TAG="${RUN_TAG:-duca_trainfree_x3d_interval_grid_$(date +%Y%m%d_%H%M%S_%z)}"
MATERIALIZE_FORMAL_JSONL="${MATERIALIZE_FORMAL_JSONL:-1}"
DUCA_X3D_FORMAL_PROVIDER="${DUCA_X3D_FORMAL_PROVIDER:-x3d_xs}"
DUCA_X3D_FORMAL_FRAME_INTERVAL="${DUCA_X3D_FORMAL_FRAME_INTERVAL:-2}"
DUCA_X3D_FORMAL_CLIP_FRAMES="${DUCA_X3D_FORMAL_CLIP_FRAMES:-}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
GRID_ROOT="${GRID_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/trainfree_frozen_actionness/${RUN_TAG}}"
FORMAL_X3D_ACTIONNESS_JSONL="${DUCA_X3D_ACTIONNESS_JSONL:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/trainfree_frozen_actionness/best_x3d_actionness.jsonl}"
FORMAL_X3D_MATERIALIZATION_SUMMARY="${FORMAL_X3D_MATERIALIZATION_SUMMARY:-${FORMAL_X3D_ACTIONNESS_JSONL%.jsonl}.materialization.json}"

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export TORCH_HOME="${TORCH_HOME:-${BASE}/tmp/torch_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${TORCH_HOME}" "${GRID_ROOT}"

if [[ "${CUDA_VISIBLE_DEVICES}" != "0" && -z "${SLURM_STEP_GPUS:-}" ]]; then
  fail "expected GPU0 for train-free X3D interval grid; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "[DUCA_X3D_INTERVAL_GRID] repo=${REPO_ROOT}"
echo "[DUCA_X3D_INTERVAL_GRID] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_X3D_INTERVAL_GRID] providers=${PROVIDERS}"
echo "[DUCA_X3D_INTERVAL_GRID] frame_intervals=${FRAME_INTERVALS}"
echo "[DUCA_X3D_INTERVAL_GRID] grid_root=${GRID_ROOT}"
echo "[DUCA_X3D_INTERVAL_GRID] formal_materialize=${MATERIALIZE_FORMAL_JSONL} provider=${DUCA_X3D_FORMAL_PROVIDER} frame_interval=${DUCA_X3D_FORMAL_FRAME_INTERVAL} output=${FORMAL_X3D_ACTIONNESS_JSONL}"

MANIFEST="${GRID_ROOT}/manifest.tsv"
printf "provider\tclip_frames\tframe_interval\tcrop_size\tbatch_size\tout_root\tstatus\n" > "${MANIFEST}"

for provider in ${PROVIDERS}; do
  case "${provider}" in
    x3d_xs|efficient_x3d_xs)
      clip_frames=4
      crop_size="${CROP_SIZE_XS}"
      batch_size="${BATCH_SIZE_XS}"
      ;;
    x3d_s|efficient_x3d_s)
      clip_frames=13
      crop_size="${CROP_SIZE_S}"
      batch_size="${BATCH_SIZE_S}"
      ;;
    *)
      fail "unsupported provider for interval grid: ${provider}"
      ;;
  esac

  for frame_interval in ${FRAME_INTERVALS}; do
    cell_tag="${RUN_TAG}_${provider}_t${clip_frames}x${frame_interval}"
    cell_root="${GRID_ROOT}/${provider}_t${clip_frames}x${frame_interval}"
    echo "[DUCA_X3D_INTERVAL_GRID] START provider=${provider} clip_frames=${clip_frames} frame_interval=${frame_interval}"
    PROVIDER="${provider}" \
    SUBSET="${SUBSET}" \
    MAX_VIDEOS="${MAX_VIDEOS}" \
    DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE}" \
    BUDGET="${BUDGET}" \
    CLIP_FRAMES="${clip_frames}" \
    FRAME_INTERVAL="${frame_interval}" \
    CROP_SIZE="${crop_size}" \
    BATCH_SIZE="${batch_size}" \
    BOUNDARY_RADIUS="${BOUNDARY_RADIUS}" \
    RUN_TAG="${cell_tag}" \
    OUT_ROOT="${cell_root}" \
    bash scripts/run_duca_trainfree_x3d_actionness_selection_gpu0.sh
    printf "%s\t%s\t%s\t%s\t%s\t%s\tcomplete\n" \
      "${provider}" "${clip_frames}" "${frame_interval}" "${crop_size}" "${batch_size}" "${cell_root}" >> "${MANIFEST}"
  done
done

SUMMARY_JSON="${GRID_ROOT}/x3d_interval_grid.summary.json"
SUMMARY_TSV="${GRID_ROOT}/x3d_interval_grid.summary.tsv"
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON=python

"${PYTHON}" -m py_compile \
  tools/bata/summarize_trainfree_x3d_interval_grid.py \
  tools/bata/materialize_trainfree_x3d_actionness.py

"${PYTHON}" tools/bata/summarize_trainfree_x3d_interval_grid.py \
  --manifest-tsv "${MANIFEST}" \
  --summary-json "${SUMMARY_JSON}" \
  --summary-tsv "${SUMMARY_TSV}" \
  --subset "${SUBSET}" \
  2>&1 | tee "${GRID_ROOT}/summarize_grid.out"

if [[ "${MATERIALIZE_FORMAL_JSONL}" == "1" ]]; then
  materialize_args=(
    --grid-summary-json "${SUMMARY_JSON}"
    --output-jsonl "${FORMAL_X3D_ACTIONNESS_JSONL}"
    --summary-json "${FORMAL_X3D_MATERIALIZATION_SUMMARY}"
    --selection-policy pre_registered
    --provider "${DUCA_X3D_FORMAL_PROVIDER}"
    --frame-interval "${DUCA_X3D_FORMAL_FRAME_INTERVAL}"
  )
  if [[ -n "${DUCA_X3D_FORMAL_CLIP_FRAMES}" ]]; then
    materialize_args+=(--clip-frames "${DUCA_X3D_FORMAL_CLIP_FRAMES}")
  fi
  "${PYTHON}" tools/bata/materialize_trainfree_x3d_actionness.py "${materialize_args[@]}" \
    2>&1 | tee "${GRID_ROOT}/materialize_formal_x3d_actionness.out"
fi

echo "[DUCA_X3D_INTERVAL_GRID] COMPLETE manifest=${MANIFEST} summary=${SUMMARY_JSON}"
