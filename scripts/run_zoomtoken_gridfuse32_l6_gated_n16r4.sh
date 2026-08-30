#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_GRIDFUSE32_L6][FAIL] invoke with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_GRIDFUSE32_L6][FAIL] %s\n' "$*" >&2
  exit 2
}

BASE="/data/run01/sczc063/yuzibo"
EXPECTED_COMMIT="${ZOOMTOKEN_GRIDFUSE_EXPECTED_COMMIT:?set the reviewed clean commit}"
ROOT="${ZOOMTOKEN_GRIDFUSE_SOURCE_ROOT:?set the reviewed clean checkout}"
RESULT_ROOT="${ZOOMTOKEN_GRIDFUSE_RESULT_ROOT:?set the immutable task result root}"
PHASE="${ZOOMTOKEN_GRIDFUSE_PHASE:-G0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
BRANCH="codex/zoomtoken-gridfuse32-l6-v001"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_gridfuse32_l6_prebackbone_seed42_v001.py"
CONTROL_CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"
R1_CHECKPOINT="${ZOOMTOKEN_GRIDFUSE_R1_CHECKPOINT:-${BASE}/projects/zoomtoken_official_prebackbone_r1_9e25c6d3_seed42_20260822T080108Z/cells/r1_strict_rect8x8_prebackbone_sparse_adapter/seed42/gpu2_id0/checkpoint/epoch_59.pth}"
ANNOTATION="${BASE}/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="${BASE}/thumos14/annotations/category_idx.txt"
VIDEO_ROOT="${BASE}/thumos14/raw_data/video"
PRETRAINED="${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full lowercase SHA'
[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'GridFuse32-L6 actions require a Slurm allocation'
case "${RESULT_ROOT}" in
  "${BASE}"/*) ;;
  *) fail 'result root leaves the remote write boundary' ;;
esac
[[ "${RESULT_ROOT}/" != "${ROOT}/"* ]] || fail 'result root must be outside the source checkout'
for path in "${CONFIG}" "${CONTROL_CONFIG}" "${R1_CHECKPOINT}" "${ANNOTATION}" "${CLASS_MAP}" "${PRETRAINED}" "${CONDA_ACTIVATE}"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done
[[ -d "${VIDEO_ROOT}" ]] || fail 'canonical THUMOS14 video root is missing'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source checkout is not clean'
git -C "${ROOT}" fetch --quiet origin "${BRANCH}"
[[ "$(git -C "${ROOT}" rev-parse FETCH_HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'fresh GitHub fetch does not resolve to the reviewed candidate'

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/profile
  set -u
fi
command -v module >/dev/null 2>&1 || fail 'environment-modules is unavailable'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${CONDA_ACTIVATE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail 'precheck requires a Slurm-visible GPU'
  python -m py_compile \
    opentad/models/backbones/vit_adapter.py \
    tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py \
    tools/bata/profile_zoomtoken_gridfuse32_l6_fullstack.py
  python -m pytest tests/test_zoomtoken_gridfuse32_l6.py -q
  python - "${CONFIG}" "${R1_CHECKPOINT}" <<'PY'
import sys
from mmengine import Config
import torch

config = Config.fromfile(sys.argv[1])
route = config.model.backbone.backbone.gridfuse32_l6
assert tuple(route.dense_block_indices) == tuple(range(6))
assert tuple(route.fused_block_indices) == tuple(range(6, 12))
checkpoint = torch.load(sys.argv[2], map_location="cpu")
assert int(checkpoint["epoch"]) == 59
assert "state_dict_ema" in checkpoint
print("[ZOOMTOKEN_GRIDFUSE32_L6][PRECHECK_READY]")
PY
  exit 0
fi

case "${PHASE}" in
  G0)
    [[ "${#visible_gpus[@]}" -eq 1 ]] || fail 'G0 requires exactly one visible GPU'
    [[ "${SLURM_CPUS_PER_TASK:-}" == "4" ]] || fail 'G0 requires --cpus-per-task=4'
    G0_ROOT="${RESULT_ROOT}/g0"
    [[ ! -e "${G0_ROOT}" ]] || fail 'exclusive G0 result root already exists'
    set +e
    python tools/bata/profile_zoomtoken_gridfuse32_l6_segment.py \
      --config "${CONFIG}" \
      --checkpoint "${R1_CHECKPOINT}" \
      --run-root "${G0_ROOT}" \
      --expected-commit "${EXPECTED_COMMIT}" \
      --warmup 100 \
      --iterations 500
    status=$?
    set -e
    exit "${status}"
    ;;
  G1)
    [[ "${#visible_gpus[@]}" -eq 2 ]] || fail 'G1 requires exactly two visible GPUs'
    [[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || fail 'G1 requires --cpus-per-task=8'
    G0_PROFILE="${RESULT_ROOT}/g0/profile.json"
    [[ -f "${G0_PROFILE}" ]] || fail 'G1 requires the immutable G0 profile'
    python - "${G0_PROFILE}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["status"] == "GRIDFUSE32_L6_G0_PASS_PENDING_G1"
assert payload["gate_passed"] is True
PY
    G1_ROOT="${RESULT_ROOT}/g1"
    G1_CELL="${G1_ROOT}/cell"
    [[ ! -e "${G1_ROOT}" ]] || fail 'exclusive G1 result root already exists'
    mkdir -p "${G1_ROOT}"
    set +e
    torchrun --nnodes=1 --nproc_per_node=2 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="zoomtoken-gridfuse32-l6-g1-${SLURM_JOB_ID}" \
      tools/train.py "${CONFIG}" --seed 42 --id 0 \
      --cfg-options \
      "work_dir=${G1_CELL}" \
      "zoomtoken_p1_config.source_commit=${EXPECTED_COMMIT}" \
      "dataset.train.ann_file=${ANNOTATION}" \
      "dataset.train.class_map=${CLASS_MAP}" \
      "dataset.train.data_path=${VIDEO_ROOT}" \
      "dataset.train.subset_name=training" \
      "dataset.val.ann_file=${ANNOTATION}" \
      "dataset.val.class_map=${CLASS_MAP}" \
      "dataset.val.data_path=${VIDEO_ROOT}" \
      "dataset.val.subset_name=validation" \
      "dataset.test.ann_file=${ANNOTATION}" \
      "dataset.test.class_map=${CLASS_MAP}" \
      "dataset.test.data_path=${VIDEO_ROOT}" \
      "dataset.test.subset_name=validation" \
      "evaluation.ground_truth_filename=${ANNOTATION}" \
      "model.backbone.custom.pretrain=${PRETRAINED}"
    status=$?
    set -e
    checkpoint="${G1_CELL}/gpu2_id0/checkpoint/epoch_59.pth"
    python - "${G1_ROOT}/terminal_receipt.json" "${status}" "${checkpoint}" "${EXPECTED_COMMIT}" <<'PY'
import json, os, sys
path, code, checkpoint, commit = sys.argv[1:]
payload = {
    "schema_version": "zoomtoken_gridfuse32_l6_g1_terminal_v001",
    "status": (
        "GRIDFUSE32_L6_G1_TRAINING_COMPLETED_PENDING_RESULT_INGEST"
        if int(code) == 0 and os.path.isfile(checkpoint)
        else "GRIDFUSE32_L6_G1_ENGINEERING_OR_PROTOCOL_BLOCKER"
    ),
    "exit_code": int(code),
    "source_commit": commit,
    "checkpoint": checkpoint if os.path.isfile(checkpoint) else None,
    "checkpoint_epoch": 59 if os.path.isfile(checkpoint) else None,
    "primary_state": "state_dict_ema" if os.path.isfile(checkpoint) else None,
}
with open(path, "x", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    exit "${status}"
    ;;
  G2)
    [[ "${#visible_gpus[@]}" -eq 1 ]] || fail 'G2 requires exactly one visible GPU'
    [[ "${SLURM_CPUS_PER_TASK:-}" == "5" ]] || fail 'G2 requires --cpus-per-task=5'
    G1_GATE="${ZOOMTOKEN_GRIDFUSE_G1_GATE_RECEIPT:?G2 requires the frozen G1 gate receipt}"
    CANDIDATE_CHECKPOINT="${ZOOMTOKEN_GRIDFUSE_CANDIDATE_CHECKPOINT:?G2 requires epoch-59 candidate EMA}"
    [[ -f "${G1_GATE}" && -f "${CANDIDATE_CHECKPOINT}" ]] || fail 'G2 gate or checkpoint is missing'
    python - "${G1_GATE}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["gate_passed"] is True
assert payload["status"] == "GRIDFUSE32_L6_G1_ACCURACY_PASS_PENDING_G2"
PY
    G2_ROOT="${RESULT_ROOT}/g2"
    [[ ! -e "${G2_ROOT}" ]] || fail 'exclusive G2 result root already exists'
    set +e
    torchrun --nnodes=1 --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
      --rdzv_id="zoomtoken-gridfuse32-l6-g2-${SLURM_JOB_ID}" \
      tools/bata/profile_zoomtoken_gridfuse32_l6_fullstack.py \
      --control-config "${CONTROL_CONFIG}" \
      --candidate-config "${CONFIG}" \
      --control-checkpoint "${R1_CHECKPOINT}" \
      --candidate-checkpoint "${CANDIDATE_CHECKPOINT}" \
      --annotation "${ANNOTATION}" \
      --class-map "${CLASS_MAP}" \
      --video-root "${VIDEO_ROOT}" \
      --run-root "${G2_ROOT}" \
      --expected-commit "${EXPECTED_COMMIT}"
    status=$?
    set -e
    exit "${status}"
    ;;
  *) fail 'phase must be G0, G1, or G2' ;;
esac
