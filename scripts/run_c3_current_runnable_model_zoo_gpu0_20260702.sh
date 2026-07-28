#!/bin/bash
set -uo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-${BASE}/OpenTAD_C3_CoarseClean_20260702}"
RUN_TAG="${RUN_TAG:-c3_current_runnable_model_zoo_gpu0_$(date '+%Y%m%d_%H%M%S_%z')}"
OUT_ROOT="${OUT_ROOT:-${BASE}/projects/c3_lowres_action_probe/outputs}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_TAG}}"
LOG_DIR="${RUN_DIR}/logs"
RUN_INDEX="${RUN_DIR}/run_index.tsv"
SKIP_INDEX="${RUN_DIR}/skipped_models.tsv"

CONFIG_ACTIONNESS="${CONFIG_ACTIONNESS:-configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py}"
CONFIG_BOUNDARY="${CONFIG_BOUNDARY:-configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py}"
ANN_FILE="${ANN_FILE:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
CLASS_MAP="${CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${BASE}/raw/Validation Data/validation}"
TEST_DATA_PATH="${TEST_DATA_PATH:-${BASE}/raw/Test Data/TH14_test_set_mp4}"

EPOCHS="${EPOCHS:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
VAL_EVERY_EPOCHS="${VAL_EVERY_EPOCHS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-20}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1.0e-4}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-train_loss}"
EARLY_STOP_MODE="${EARLY_STOP_MODE:-auto}"
COVERAGE_BUDGET_FRACTION="${COVERAGE_BUDGET_FRACTION:-0.5}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_C3_READERS="${RUN_C3_READERS:-0}"

DO_BEST_CHECKPOINT_INDIRECT_EVAL="${DO_BEST_CHECKPOINT_INDIRECT_EVAL:-1}"
INDIRECT_DENSE_WINDOW_SIZE="${INDIRECT_DENSE_WINDOW_SIZE:-768}"
INDIRECT_TARGET_LEN="${INDIRECT_TARGET_LEN:-384}"
INDIRECT_SELECTION_STRATEGY="${INDIRECT_SELECTION_STRATEGY:-delta_p_action}"
INDIRECT_MAX_VAL_BATCHES="${INDIRECT_MAX_VAL_BATCHES:-0}"
INDIRECT_WINDOW_OVERLAP_RATIO="${INDIRECT_WINDOW_OVERLAP_RATIO:-0.25}"

TCN_VARIANTS="${TCN_VARIANTS:-lite dilated multiscale motion residual gated separable_dilated causal_dilated ms_tcnpp c2f_tcn}"
OFFICIAL_BACKENDS="${OFFICIAL_BACKENDS:-official_video_mamba_asformer official_asformer official_fact official_ms_tcn2}"
PRIORITY_OFFICIAL_BACKENDS="${PRIORITY_OFFICIAL_BACKENDS:-official_video_mamba_asformer official_asformer official_fact official_ms_tcn2}"
PRIORITY_TCN_VARIANTS="${PRIORITY_TCN_VARIANTS:-c2f_tcn}"
MATRIX_IMAGE_96_IDS="${MATRIX_IMAGE_96_IDS:-timm_mobilenetv3_large_100_tsm_tcn timm_tf_efficientnetv2_b0_tcn timm_convnext_tiny_tcn timm_resnet18_tcn}"
MATRIX_IMAGE_224_IDS="${MATRIX_IMAGE_224_IDS:-timm_vit_tiny_patch16_224_temporal}"
MATRIX_VIDEO_112_IDS="${MATRIX_VIDEO_112_IDS:-torchvision_r3d_18 torchvision_r2plus1d_18 torchvision_mc3_18 torchvision_s3d torchvision_mvit_v2_s torchvision_swin3d_t pytorchvideo_x3d_xs pytorchvideo_x3d_s pytorchvideo_c2d_r50 pytorchvideo_i3d_r50}"
RUN_MOBILENET_BASELINES="${RUN_MOBILENET_BASELINES:-1}"
RUN_MATRIX_ZOO="${RUN_MATRIX_ZOO:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must be exactly 0 for this explicit GPU0 run, got '${CUDA_VISIBLE_DEVICES}'." >&2
  exit 44
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_NON_SLURM_TRAINING:-0}" != "1" ]]; then
  echo "Refusing to start training outside Slurm; set ALLOW_NON_SLURM_TRAINING=1 only inside a protected allocation." >&2
  exit 43
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}" || exit 2

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source "${BASE}/conda_envs/opentad/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-${BASE}/model_zoo_cache/c3_coarse_classifier/torch}"
export HF_HOME="${HF_HOME:-${BASE}/hf_cache}"
export http_proxy="${http_proxy:-http://u-MtfrT7:vH5orjDV@10.244.6.36:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

printf "model_id\tkind\tstatus\texit_code\tseconds\tout_dir\tlog_path\n" > "${RUN_INDEX}"
printf "model_id\treason\n" > "${SKIP_INDEX}"

log_msg() {
  echo "[$(date -Iseconds)] $*"
}

record_skip() {
  local model_id="$1"
  local reason="$2"
  printf "%s\t%s\n" "${model_id}" "${reason}" >> "${SKIP_INDEX}"
  log_msg "SKIP ${model_id}: ${reason}"
}

run_required() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/precheck_${name}.log"
  log_msg "PRECHECK ${name} -> ${log_path}"
  "$@" >"${log_path}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    log_msg "PRECHECK_FAILED ${name} rc=${rc}; see ${log_path}"
    exit "${rc}"
  fi
  log_msg "PRECHECK_OK ${name}"
}

run_model() {
  local model_id="$1"
  local kind="$2"
  local out_dir="$3"
  shift 3
  local log_path="${LOG_DIR}/${model_id}.log"
  local start_ts
  start_ts="$(date +%s)"
  mkdir -p "${out_dir}"
  log_msg "RUN ${model_id} (${kind}) -> ${out_dir}"
  "$@" >"${log_path}" 2>&1
  local rc=$?
  local end_ts
  end_ts="$(date +%s)"
  local seconds=$((end_ts - start_ts))
  local status="ok"
  if [[ ${rc} -ne 0 ]]; then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${model_id}" "${kind}" "${status}" "${rc}" "${seconds}" "${out_dir}" "${log_path}" >> "${RUN_INDEX}"
  log_msg "${status^^} ${model_id} rc=${rc} seconds=${seconds}"
  return 0
}

COMMON_DATA_ARGS=(
  --ann-file "${ANN_FILE}"
  --class-map "${CLASS_MAP}"
  --train-data-path "${TRAIN_DATA_PATH}"
  --val-data-path "${TEST_DATA_PATH}"
  --test-data-path "${TEST_DATA_PATH}"
)

COMMON_TRAIN_ARGS=(
  --device cuda
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --seed 0
  --max-train-batches "${MAX_TRAIN_BATCHES}"
  --max-val-batches "${MAX_VAL_BATCHES}"
  --val-every-epochs "${VAL_EVERY_EPOCHS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}"
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}"
  --early-stop-metric "${EARLY_STOP_METRIC}"
  --early-stop-mode "${EARLY_STOP_MODE}"
  --coverage-budget-fraction "${COVERAGE_BUDGET_FRACTION}"
  --boundary-radius "${BOUNDARY_RADIUS}"
  --log-every-batches 10
  --fast-lowres-pipeline
  --probe-window-size 384
  --save-checkpoint
  "${COMMON_DATA_ARGS[@]}"
)

run_indirect_eval() {
  local model_id="$1"
  local probe_model="$2"
  local train_out_dir="$3"
  local spatial_size="$4"
  local probe_window_size="$5"
  local config_path="$6"
  shift 6
  local checkpoint_path="${train_out_dir}/probe_reader.pth"
  if [[ "${DO_BEST_CHECKPOINT_INDIRECT_EVAL}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${checkpoint_path}" ]]; then
    record_skip "${model_id}_indirect_eval" "missing checkpoint ${checkpoint_path}"
    return 0
  fi
  local eval_dir="${train_out_dir}/best_checkpoint_indirect_val_${INDIRECT_DENSE_WINDOW_SIZE}_to_${INDIRECT_TARGET_LEN}"
  local samples_jsonl="${eval_dir}/samples.jsonl"
  local ledger_jsonl="${eval_dir}/value_transport_ledger_${INDIRECT_SELECTION_STRATEGY}_${INDIRECT_TARGET_LEN}.jsonl"
  local ledger_summary="${eval_dir}/value_transport_ledger_${INDIRECT_SELECTION_STRATEGY}_${INDIRECT_TARGET_LEN}.summary.json"
  mkdir -p "${eval_dir}"
  run_model "${model_id}_indirect_eval" "best-checkpoint-indirect-val" "${eval_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${config_path}" \
      --out-dir "${eval_dir}/probe_eval" \
      --device cuda \
      --epochs 1 \
      --batch-size 1 \
      --num-workers "${NUM_WORKERS}" \
      --seed 0 \
      --probe-model "${probe_model}" \
      --scout-spatial-size "${spatial_size}" \
      --max-train-batches 0 \
      --max-val-batches "${INDIRECT_MAX_VAL_BATCHES}" \
      --coverage-only \
      --coverage-budget-fraction "${COVERAGE_BUDGET_FRACTION}" \
      --boundary-radius "${BOUNDARY_RADIUS}" \
      --log-every-batches 25 \
      --fast-lowres-pipeline \
      --probe-window-size "${INDIRECT_DENSE_WINDOW_SIZE}" \
      --eval-window-overlap-ratio "${INDIRECT_WINDOW_OVERLAP_RATIO}" \
      --eval-include-all-windows \
      --probe-checkpoint "${checkpoint_path}" \
      --sample-jsonl "${samples_jsonl}" \
      "${COMMON_DATA_ARGS[@]}" \
      "$@"
  if [[ -f "${samples_jsonl}" ]]; then
    run_model "${model_id}_ledger_convert" "best-checkpoint-ledger-convert" "${eval_dir}" \
      python tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py \
        --input-jsonl "${samples_jsonl}" \
        --output-jsonl "${ledger_jsonl}" \
        --summary-json "${ledger_summary}" \
        --strategy "${INDIRECT_SELECTION_STRATEGY}" \
        --target-len "${INDIRECT_TARGET_LEN}" \
        --require-selected-count "${INDIRECT_TARGET_LEN}" \
        --allow-short-valid-ratio-count \
        --deduplicate-sample-id \
        --deploy-selection-ledger \
        --route-variant "c3_model_zoo_${model_id}_${INDIRECT_SELECTION_STRATEGY}_${INDIRECT_DENSE_WINDOW_SIZE}_to_${INDIRECT_TARGET_LEN}_val"
  fi
}

run_c3_reader() {
  local model_id="$1"
  local config_path="$2"
  local out_dir="${RUN_DIR}/${model_id}"
  run_model "${model_id}" "c3-reader" "${out_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${config_path}" \
      --out-dir "${out_dir}" \
      --batch-size 4 \
      --lr 1.0e-4 \
      --probe-model c3-reader \
      --scout-spatial-size 32 \
      "${COMMON_TRAIN_ARGS[@]}"
  run_indirect_eval "${model_id}" "c3-reader" "${out_dir}" 32 384 "${config_path}"
}

run_mobilenet() {
  local size="$1"
  local model_id="mobilenetv3_lowres_frame_actionness_${size}"
  local out_dir="${RUN_DIR}/${model_id}"
  run_model "${model_id}" "mobilenetv3" "${out_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${CONFIG_ACTIONNESS}" \
      --out-dir "${out_dir}" \
      --batch-size 4 \
      --lr 1.0e-4 \
      --probe-model mobilenetv3 \
      --mobilenet-sizes "${size}" \
      "${COMMON_TRAIN_ARGS[@]}"
  run_indirect_eval "${model_id}" "mobilenetv3" "${out_dir}" "${size}" 384 "${CONFIG_ACTIONNESS}" --mobilenet-sizes "${size}"
}

run_tcn() {
  local variant="$1"
  local model_id="temporal_tcn_${variant}"
  local out_dir="${RUN_DIR}/${model_id}"
  run_model "${model_id}" "temporal-tcn" "${out_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${CONFIG_ACTIONNESS}" \
      --out-dir "${out_dir}" \
      --batch-size 3 \
      --lr 1.0e-4 \
      --probe-model temporal-tcn \
      --scout-spatial-size 64 \
      --tcn-variants "${variant}" \
      "${COMMON_TRAIN_ARGS[@]}"
  run_indirect_eval "${model_id}" "temporal-tcn" "${out_dir}" 64 384 "${CONFIG_ACTIONNESS}" --tcn-variants "${variant}"
}

run_official() {
  local backend="$1"
  local model_id="official_action_seg_${backend}"
  local out_dir="${RUN_DIR}/${model_id}"
  if ! official_backend_available "${backend}"; then
    record_skip "${model_id}" "official backend unavailable; requires official implementation/dependencies and will not fall back to local lite prototype"
    return 0
  fi
  run_model "${model_id}" "official-action-seg" "${out_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${CONFIG_ACTIONNESS}" \
      --out-dir "${out_dir}" \
      --batch-size 2 \
      --lr 1.0e-4 \
      --probe-model official-action-seg \
      --scout-spatial-size 64 \
      --official-action-seg-backends "${backend}" \
      "${COMMON_TRAIN_ARGS[@]}"
  run_indirect_eval "${model_id}" "official-action-seg" "${out_dir}" 64 384 "${CONFIG_ACTIONNESS}" --official-action-seg-backends "${backend}"
}

official_backend_available() {
  local backend="$1"
  python - "${backend}" <<'PY'
import sys
from tools.bata import train_lowres_action_probe as probe

backend = sys.argv[1]
raise SystemExit(0 if probe.official_action_seg_backend_available(backend) else 1)
PY
}

run_matrix() {
  local model_id="$1"
  local spatial_size="$2"
  local batch_size="$3"
  local lr="$4"
  local train_window="$5"
  local out_dir="${RUN_DIR}/${model_id}"
  run_model "${model_id}" "matrix-zoo" "${out_dir}" \
    python -u tools/bata/train_lowres_action_probe.py \
      --config "${CONFIG_ACTIONNESS}" \
      --out-dir "${out_dir}" \
      --device cuda \
      --epochs "${EPOCHS}" \
      --batch-size "${batch_size}" \
      --num-workers "${NUM_WORKERS}" \
      --lr "${lr}" \
      --seed 0 \
      --probe-model matrix-zoo \
      --scout-spatial-size "${spatial_size}" \
      --matrix-model-ids "${model_id}" \
      --matrix-video-clip-len 16 \
      --matrix-video-anchor-stride 8 \
      --matrix-freeze-backbone \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-val-batches "${MAX_VAL_BATCHES}" \
      --val-every-epochs "${VAL_EVERY_EPOCHS}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE}" \
      --early-stop-min-epochs "${EARLY_STOP_MIN_EPOCHS}" \
      --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
      --early-stop-metric "${EARLY_STOP_METRIC}" \
      --early-stop-mode "${EARLY_STOP_MODE}" \
      --coverage-budget-fraction "${COVERAGE_BUDGET_FRACTION}" \
      --boundary-radius "${BOUNDARY_RADIUS}" \
      --log-every-batches 10 \
      --fast-lowres-pipeline \
      --probe-window-size "${train_window}" \
      --save-checkpoint \
      "${COMMON_DATA_ARGS[@]}"
  run_indirect_eval "${model_id}" "matrix-zoo" "${out_dir}" "${spatial_size}" "${train_window}" "${CONFIG_ACTIONNESS}" --matrix-model-ids "${model_id}"
}

word_in_list() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

log_msg "START RUN_TAG=${RUN_TAG}"
log_msg "HOST=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
log_msg "PROJECT_DIR=${PROJECT_DIR}"
log_msg "RUN_DIR=${RUN_DIR}"
log_msg "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_msg "RUN_C3_READERS=${RUN_C3_READERS}"
nvidia-smi || true

run_required cuda_check python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is not available inside the Slurm job")
PY

run_required py_compile python -m py_compile \
  tools/train.py \
  tools/test.py \
  tools/bata/train_lowres_action_probe.py \
  tools/bata/c3_coarse_classifier_model_matrix.py \
  tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py \
  tools/bata/validate_c3_asformer_delta_ledger_full_train.py

run_required focused_pytest python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_lowres_action_probe.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  tests/test_pc_ot_mras_frontend_ledger.py \
  -q

run_required model_zoo_dump python tools/bata/c3_coarse_classifier_model_matrix.py --print-zoo --dry-run

run_required tiny_forward python - <<'PY'
import torch
from tools.bata import train_lowres_action_probe as probe

device = "cuda"
frames64 = torch.rand(1, 8, 3, 64, 64, device=device)
valid = torch.ones(1, 8, dtype=torch.bool, device=device)
for variant in probe.SUPPORTED_TCN_VARIANTS:
    model = probe.C3TemporalTCNActionProbe(variant=variant, spatial_size=64, hidden_dim=32).to(device).eval()
    with torch.no_grad():
        logits = model(frames64, valid)
    assert tuple(logits.shape) == (1, 8), (variant, tuple(logits.shape))
    assert torch.isfinite(logits).all(), variant

mobile = probe.C3MobileNetV3ActionProbe(pretrained=False, freeze_backbone=True).to(device).eval()
with torch.no_grad():
    logits = mobile(frames64, valid)
assert tuple(logits.shape) == (1, 8)
assert torch.isfinite(logits).all()

for backend in ("official_ms_tcn2", "official_asformer", "official_fact"):
    if not probe.official_action_seg_backend_available(backend):
        raise SystemExit(f"official backend unavailable: {backend}")
    model = probe.C3OfficialActionSegmentationProbe(backend=backend, spatial_size=64, hidden_dim=16, num_layers=1).to(device).eval()
    with torch.no_grad():
        logits = model(frames64, valid)
    assert tuple(logits.shape) == (1, 8), (backend, tuple(logits.shape))
    assert torch.isfinite(logits).all(), backend

print("TINY_FORWARD_OK", flush=True)
PY

python tools/bata/c3_coarse_classifier_model_matrix.py --print-zoo --dry-run > "${RUN_DIR}/model_zoo.json"

if [[ "${RUN_C3_READERS}" == "1" ]]; then
  run_c3_reader "c3_reader_coarse_actionness" "${CONFIG_ACTIONNESS}"
  run_c3_reader "c3_reader_boundary_difficulty" "${CONFIG_BOUNDARY}"
else
  record_skip "c3_reader_coarse_actionness" "disabled by RUN_C3_READERS=${RUN_C3_READERS}; prioritizing non-c3-reader probes"
  record_skip "c3_reader_boundary_difficulty" "disabled by RUN_C3_READERS=${RUN_C3_READERS}; prioritizing non-c3-reader probes"
fi

if [[ "${RUN_MOBILENET_BASELINES}" == "1" ]]; then
  run_mobilenet 32
  run_mobilenet 64
else
  record_skip "mobilenetv3_lowres_frame_actionness_32" "disabled by RUN_MOBILENET_BASELINES=${RUN_MOBILENET_BASELINES}; prioritizing high-hope official/TCN probes"
  record_skip "mobilenetv3_lowres_frame_actionness_64" "disabled by RUN_MOBILENET_BASELINES=${RUN_MOBILENET_BASELINES}; prioritizing high-hope official/TCN probes"
fi

priority_official_array=(${PRIORITY_OFFICIAL_BACKENDS})
priority_tcn_array=(${PRIORITY_TCN_VARIANTS})

for backend in ${PRIORITY_OFFICIAL_BACKENDS}; do
  if word_in_list "${backend}" ${OFFICIAL_BACKENDS}; then
    run_official "${backend}"
  fi
done

for variant in ${PRIORITY_TCN_VARIANTS}; do
  if word_in_list "${variant}" ${TCN_VARIANTS}; then
    run_tcn "${variant}"
  fi
done

for variant in ${TCN_VARIANTS}; do
  if word_in_list "${variant}" "${priority_tcn_array[@]}"; then
    continue
  fi
  run_tcn "${variant}"
done

for backend in ${OFFICIAL_BACKENDS}; do
  if word_in_list "${backend}" "${priority_official_array[@]}"; then
    continue
  fi
  run_official "${backend}"
done

if [[ "${RUN_MATRIX_ZOO}" == "1" ]]; then
  for model_id in ${MATRIX_IMAGE_96_IDS}; do
    run_matrix "${model_id}" 96 2 1.0e-4 384
  done

  for model_id in ${MATRIX_IMAGE_224_IDS}; do
    run_matrix "${model_id}" 224 1 5.0e-5 384
  done

  for model_id in ${MATRIX_VIDEO_112_IDS}; do
    run_matrix "${model_id}" 112 1 5.0e-5 256
  done
else
  record_skip "matrix_zoo" "disabled by RUN_MATRIX_ZOO=${RUN_MATRIX_ZOO}; prioritizing high-hope official/TCN probes"
fi

record_skip "pytorchvideo_slowfast_r50" "training adapter intentionally disabled until two-pathway SlowFast wrapper is implemented"
record_skip "hf_videomae_small_kinetics" "hf_snapshot VideoMAE training adapter is not implemented"
record_skip "hf_videomae_base_kinetics" "hf_snapshot VideoMAE training adapter is not implemented"
record_skip "torchvision_swin3d_s" "second-wave heavy optional model; excluded from default current-runnable run"

log_msg "DONE RUN_TAG=${RUN_TAG}"
log_msg "RUN_INDEX=${RUN_INDEX}"
log_msg "SKIP_INDEX=${SKIP_INDEX}"
