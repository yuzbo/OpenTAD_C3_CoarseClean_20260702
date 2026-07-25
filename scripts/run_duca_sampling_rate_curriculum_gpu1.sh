#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_RATE_CURRICULUM][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_RATE_CURRICULUM_RUN_ROOT:-}"
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh run root is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"

STAGE1_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py"
STAGE2_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
STAGE1_WORK="${RUN_ROOT}/stage1/work"
STAGE2_WORK="${RUN_ROOT}/stage2/work"
mkdir -p "${RUN_ROOT}/stage1/quality" "${RUN_ROOT}/stage2"

if [[ -n "${DUCA_STAGE1_REUSE_CHECKPOINT:-}" ]]; then
  STAGE1_CHECKPOINT="${DUCA_STAGE1_REUSE_CHECKPOINT}"
  [[ -f "${STAGE1_CHECKPOINT}" ]] || fail "reused stage1 checkpoint is missing"
  [[ "${DUCA_STAGE1_REUSE_CHECKPOINT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "reused stage1 checkpoint hash is required"
  [[ "$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')" == "${DUCA_STAGE1_REUSE_CHECKPOINT_SHA256}" ]] || fail "reused stage1 checkpoint hash mismatch"
  [[ "${DUCA_STAGE1_REUSE_CHECKPOINT_EPOCH:-}" == "29" ]] || fail "reused stage1 checkpoint must be terminal epoch 29"
else
  "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-rate-curriculum-${SLURM_JOB_ID}-stage1" \
    tools/train.py "${STAGE1_CONFIG}" --id 0 --seed 3407 --cfg-options \
    "work_dir=${STAGE1_WORK}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${RUN_ROOT}/stage1/train.out"

  for epoch_one in 5 10 15 20 25 30; do
    epoch_zero=$((epoch_one - 1))
    checkpoint="${STAGE1_WORK}/gpu1_id0/checkpoint/epoch_${epoch_zero}.pth"
    quality_dir="${RUN_ROOT}/stage1/quality/epoch_${epoch_one}"
    [[ -f "${checkpoint}" ]] || fail "stage1 checkpoint is missing: ${checkpoint}"
    mkdir -p "${quality_dir}"
    "${PYTHON}" -m tools.bata.export_duca_selection_quality \
      --config "${STAGE1_CONFIG}" --checkpoint "${checkpoint}" \
      --output-jsonl "${quality_dir}/records.jsonl" \
      --summary-json "${quality_dir}/export.json" --split val --device cuda:0 \
      --use-ema true --seed 3407 \
      2>&1 | tee "${quality_dir}/export.out"
    "${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
      --records-jsonl "${quality_dir}/records.jsonl" --output-dir "${quality_dir}" \
      --bootstrap-samples 200 --random-seed 3407 \
      2>&1 | tee "${quality_dir}/analyze.out"
  done

  STAGE1_CHECKPOINT="${STAGE1_WORK}/gpu1_id0/checkpoint/epoch_29.pth"
  [[ -f "${STAGE1_CHECKPOINT}" ]] || fail "terminal stage1 EMA checkpoint is missing"
fi
export DUCA_STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT}"
export DUCA_STAGE1_CHECKPOINT_SHA256="$(sha256sum "${STAGE1_CHECKPOINT}" | awk '{print $1}')"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate-curriculum-${SLURM_JOB_ID}-stage2" \
  tools/train.py "${STAGE2_CONFIG}" --id 0 --seed 3407 --cfg-options \
  "work_dir=${STAGE2_WORK}" \
  "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/train.out"

STAGE2_CHECKPOINT="${STAGE2_WORK}/gpu1_id0/checkpoint/epoch_59.pth"
[[ -f "${STAGE2_CHECKPOINT}" ]] || fail "terminal stage2 EMA checkpoint is missing"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-rate-curriculum-${SLURM_JOB_ID}-stage2-eval" \
  tools/test.py "${STAGE2_CONFIG}" --checkpoint "${STAGE2_CHECKPOINT}" \
  --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch 59 \
  --metrics-json "${RUN_ROOT}/stage2/terminal_evaluation.json" --id 0 --seed 3407 \
  --cfg-options "work_dir=${RUN_ROOT}/stage2/eval" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_ROOT}/stage2/eval.out"

STAGE2_QUALITY="${RUN_ROOT}/stage2/selection_quality"
mkdir -p "${STAGE2_QUALITY}"
"${PYTHON}" -m tools.bata.export_duca_selection_quality \
  --config "${STAGE2_CONFIG}" --checkpoint "${STAGE2_CHECKPOINT}" \
  --output-jsonl "${STAGE2_QUALITY}/records.jsonl" \
  --summary-json "${STAGE2_QUALITY}/export.json" --split val --device cuda:0 \
  --use-ema true --seed 3407 \
  2>&1 | tee "${STAGE2_QUALITY}/export.out"
"${PYTHON}" -m tools.bata.analyze_duca_selection_quality \
  --records-jsonl "${STAGE2_QUALITY}/records.jsonl" --output-dir "${STAGE2_QUALITY}" \
  --bootstrap-samples 200 --random-seed 3407 \
  2>&1 | tee "${STAGE2_QUALITY}/analyze.out"

echo "[DUCA_RATE_CURRICULUM] completed stage1 and stage2 under ${RUN_ROOT}"
