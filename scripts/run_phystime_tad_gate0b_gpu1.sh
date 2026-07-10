#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PHYSTIME_GATE0B][FAIL] $*" >&2
  exit 1
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${PHYSTIME_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  REPO_ROOT="${PHYSTIME_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
OUTPUT="${OUTPUT:-${BASE}/projects/phystime_tad/gate0b/precheck_${SLURM_JOB_ID:-local}.json}"
[[ -x "${PYTHON}" ]] || fail "Python environment is missing: ${PYTHON}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "outside Slurm this launcher is restricted to physical GPU1"
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${PYTHON}" -m py_compile \
  opentad/models/utils/phystime_geometry.py \
  opentad/models/projections/phystime_projection.py \
  opentad/models/dense_heads/phystime_head.py \
  opentad/models/detectors/phystime_tad.py \
  opentad/datasets/transforms/phystime.py \
  tools/bata/run_phystime_tad_precheck.py \
  tools/bata/run_phystime_real_data_gate.py

"${PYTHON}" -m pytest \
  tests/test_phystime_geometry.py \
  tests/test_phystime_measure_attention.py \
  tests/test_phystime_head.py \
  tests/test_phystime_data_pipeline.py \
  tests/test_phystime_detector.py \
  tests/test_phystime_config_precheck.py \
  tests/test_phystime_experiment_configs.py \
  tests/test_phystime_experiment_deployment.py \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  tests/test_c3_physical_grid_actionformer_candidate.py \
  tests/test_c3_physical_grid_round_trip.py -q

"${PYTHON}" tools/bata/run_phystime_tad_precheck.py \
  --config configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py \
  --device cuda:0 \
  --output "${OUTPUT}"

echo "[PHYSTIME_GATE0B] PASS output=${OUTPUT}"
