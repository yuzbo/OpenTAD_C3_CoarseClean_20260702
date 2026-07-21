#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_R0_HOLDOUT_MAP][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=exposure132
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_R0_CHECKPOINT:-}"
CHECKPOINT_SHA256="${DUCA_R0_CHECKPOINT_SHA256:-}"
EXPECTED_EPOCH="${DUCA_R0_CHECKPOINT_EPOCH:-131}"
OUTPUT_ROOT="${DUCA_R0_OUTPUT_ROOT:-}"
SPLIT_MANIFEST="${DUCA_FRONTEND_SPLIT_MANIFEST:-}"
SPLIT_SHA256="${DUCA_FRONTEND_SPLIT_MANIFEST_SHA256:-}"
SPLIT_ANNOTATION="${DUCA_SPLIT_ANNOTATION_PATH:-}"
SPLIT_ANNOTATION_SHA256="${DUCA_SPLIT_ANNOTATION_SHA256:-}"
TRAIN_BLOCK_LIST="${DUCA_FRONTEND_TRAIN_BLOCK_LIST:-}"
TRAIN_BLOCK_LIST_SHA256="${DUCA_FRONTEND_TRAIN_BLOCK_LIST_SHA256:-}"
HOLDOUT_BLOCK_LIST="${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST:-}"
HOLDOUT_BLOCK_LIST_SHA256="${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST_SHA256:-}"
EXPECTED_PRETRAIN="${DUCA_ADATAD_PRETRAIN_PATH:-}"
EXPECTED_PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm GPU is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "expected R0 checkpoint SHA256 is required"
[[ "${EXPECTED_PRETRAIN_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "expected AdaTAD pretrain SHA256 is required"
[[ -f "${CHECKPOINT}" && -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "checkpoint/pretrain is missing"
[[ "$(readlink -f "${EXPECTED_PRETRAIN}")" == "$(readlink -f "${ADATAD_PRETRAIN_PATH}")" ]] || fail "AdaTAD pretrain path drift"
[[ -n "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}" ]] || fail "fresh output root is required"
mkdir -p "${OUTPUT_ROOT}"

# The shared preflight delegates split reopening to validate_split_manifest.
"${PYTHON}" - "${OUTPUT_ROOT}/runtime_bindings.json" \
  "${SPLIT_MANIFEST}" "${SPLIT_SHA256}" \
  "${SPLIT_ANNOTATION}" "${SPLIT_ANNOTATION_SHA256}" \
  "${TRAIN_BLOCK_LIST}" "${TRAIN_BLOCK_LIST_SHA256}" \
  "${HOLDOUT_BLOCK_LIST}" "${HOLDOUT_BLOCK_LIST_SHA256}" \
  "${CHECKPOINT}" "${CHECKPOINT_SHA256}" \
  "${ADATAD_PRETRAIN_PATH}" "${EXPECTED_PRETRAIN_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

from tools.bata.select_duca_boundary_burst_candidates import (
    validate_r0_runtime_bindings,
)

out = Path(sys.argv[1])
values = sys.argv[2:]
payload = validate_r0_runtime_bindings(
    split_manifest=values[0],
    split_manifest_sha256=values[1],
    annotation_path=values[2],
    annotation_sha256=values[3],
    train_block_list=values[4],
    train_block_list_sha256=values[5],
    holdout_block_list=values[6],
    holdout_block_list_sha256=values[7],
    checkpoint_path=values[8],
    checkpoint_sha256=values[9],
    pretrain_path=values[10],
    pretrain_sha256=values[11],
)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

EVAL_BLOCKED="${OUTPUT_ROOT}/evaluation_blocked_videos.json"
"${PYTHON}" - "${HOLDOUT_BLOCK_LIST}" "${EVAL_BLOCKED}" <<'PY'
import json, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
videos = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if not videos or len(videos) != len(set(videos)):
    raise SystemExit("R0 evaluator blocked-video list must be nonempty and unique")
target.write_text(json.dumps(videos, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
export DUCA_R0_EVAL_BLOCKED_VIDEOS="${EVAL_BLOCKED}"

INPUT="${OUTPUT_ROOT}/holdout_inputs.jsonl"
INPUT_SUMMARY="${OUTPUT_ROOT}/holdout_inputs.summary.json"
FAMILIES="${OUTPUT_ROOT}/holdout_families.jsonl"
FAMILY_SUMMARY="${OUTPUT_ROOT}/holdout_families.summary.json"
"${PYTHON}" -m tools.bata.export_duca_allocation_ceiling_inputs \
  --config configs/adatad/thumos/duca_boundary_burst_r0_holdout_export.py \
  --checkpoint "${CHECKPOINT}" --output-jsonl "${INPUT}" \
  --summary-json "${INPUT_SUMMARY}" --split train --requested-budget 384 \
  --device cuda:0 --use-ema true --batch-size 1 --num-workers 2 \
  --coordinate-tolerance-frames 0

"${PYTHON}" -m tools.bata.build_duca_r0_boundary_burst_oracles \
  --input-jsonl "${INPUT}" \
  --config configs/adatad/thumos/duca_boundary_burst_r0_holdout_export.py \
  --output-jsonl "${FAMILIES}" --summary-json "${FAMILY_SUMMARY}" \
  --max-unselected-hole 2

export DUCA_ALLOCATION_ARTIFACT_PATH="${FAMILIES}"
export DUCA_ALLOCATION_ARTIFACT_SHA256="$(sha256sum "${FAMILIES}" | awk '{print $1}')"
families=(
  A_exact_uniform
  R2Q3_privileged_boundary_burst
  R4Q5_privileged_boundary_burst
  Z_unrestricted_gt_oracle
)
for family in "${families[@]}"; do
  export DUCA_ALLOCATION_FAMILY_KEY="${family}"
  if [[ "${family}" == A_exact_uniform ]]; then
    export DUCA_ALLOCATION_ALLOW_PRIVILEGED=0
  else
    export DUCA_ALLOCATION_ALLOW_PRIVILEGED=1
  fi
  root="${OUTPUT_ROOT}/map/${family}"
  mkdir -p "${root}"
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-r0-${SLURM_JOB_ID}-${family}" \
    tools/test.py configs/adatad/thumos/duca_boundary_burst_r0_selected_axis_replay.py \
    --checkpoint "${CHECKPOINT}" --checkpoint-state-key state_dict_ema \
    --expected-checkpoint-epoch "${EXPECTED_EPOCH}" \
    --metrics-json "${root}/metrics.json" --id 0 --seed 3407 \
    --cfg-options "work_dir=${root}/work" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
      "post_processing.save_dict=True" "inference.load_from_raw_predictions=False" \
    2>&1 | tee "${root}/test.out"
done

R0_CONFIG="configs/adatad/thumos/duca_boundary_burst_r0_selected_axis_replay.py"
R0_CONFIG_SHA256="$(sha256sum "${R0_CONFIG}" | awk '{print $1}')"
FAMILY_SUMMARY_SHA256="$(sha256sum "${FAMILY_SUMMARY}" | awk '{print $1}')"
EVAL_BLOCKED_SHA256="$(sha256sum "${EVAL_BLOCKED}" | awk '{print $1}')"
family_evaluations=()
for family in "${families[@]}"; do
  family_evaluations+=(--family-evaluation "${family}=${OUTPUT_ROOT}/map/${family}/metrics.json")
done
"${PYTHON}" -m tools.bata.finalize_duca_r0_boundary_burst \
  --expected-commit "${EXPECTED_COMMIT}" \
  "${family_evaluations[@]}" \
  --split-manifest "${SPLIT_MANIFEST}" --split-manifest-sha256 "${SPLIT_SHA256}" \
  --checkpoint "${CHECKPOINT}" --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --checkpoint-epoch "${EXPECTED_EPOCH}" \
  --config "${R0_CONFIG}" --config-sha256 "${R0_CONFIG_SHA256}" \
  --allocation-artifact "${FAMILIES}" \
  --allocation-artifact-sha256 "${DUCA_ALLOCATION_ARTIFACT_SHA256}" \
  --family-summary "${FAMILY_SUMMARY}" --family-summary-sha256 "${FAMILY_SUMMARY_SHA256}" \
  --pretrain "${ADATAD_PRETRAIN_PATH}" --pretrain-sha256 "${EXPECTED_PRETRAIN_SHA256}" \
  --blocked-videos "${EVAL_BLOCKED}" --blocked-videos-sha256 "${EVAL_BLOCKED_SHA256}" \
  --bootstrap-output "${OUTPUT_ROOT}/r0_bootstrap.json" \
  --summary-output "${OUTPUT_ROOT}/r0_summary.json" \
  --bootstrap-samples 1000 --bootstrap-seed 3407 --bootstrap-confidence 0.95 \
  --required-headroom-percentage-points 0.20

sha256sum "${OUTPUT_ROOT}/r0_summary.json" | awk '{print $1}' > "${OUTPUT_ROOT}/r0_summary.sha256"

echo "[DUCA_R0_HOLDOUT_MAP] completed ${OUTPUT_ROOT}/r0_summary.json"
