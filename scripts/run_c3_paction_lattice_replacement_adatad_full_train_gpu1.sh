#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_PACTION_LATTICE_ADATAD][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN="${ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN:-0}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
RUN_TAG="${RUN_TAG:-c3_paction_lattice_replacement_adatad_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-}"
MASTER_PORT_LOW="${MASTER_PORT_LOW:-30000}"
MASTER_PORT_HIGH="${MASTER_PORT_HIGH:-60999}"
MASTER_PORT_MAX_ATTEMPTS="${MASTER_PORT_MAX_ATTEMPTS:-256}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${C3_PACTION_ADATAD_PRETRAIN_PATH:-${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"

if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
  fi
  if [[ "${CUDA_VISIBLE_DEVICES}" != "0" && "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
    fail "C3 p_action lattice AdaTAD full train must see one Slurm-bound GPU as logical 0/1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
    fail "C3 p_action lattice AdaTAD full train must use physical GPU1 outside Slurm remapping; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
fi

export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-${BASE}/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-${BASE}/raw/Test Data/TH14_test_set_mp4}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

CONFIG="${CONFIG:-configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train_exec.py}"
CONFIG_VALIDATOR="${CONFIG_VALIDATOR:-tools/bata/validate_c3_paction_learned_adatad_full_train.py}"
LATTICE_LEDGER_VALIDATOR="${LATTICE_LEDGER_VALIDATOR:-tools/bata/validate_paction_lattice_replacement_ledger.py}"
LATTICE_LEDGER_PIPELINE="${LATTICE_LEDGER_PIPELINE:-tools/bata/run_paction_lattice_replacement_ledger_pipeline.py}"

LEARNED_ROOT="${LEARNED_ROOT:-${BASE}/projects/c3_lowres_action_probe/paction_lattice_replacement_adatad/${RUN_TAG}}"
LEDGER_ROOT="${LEDGER_ROOT:-${LEARNED_ROOT}/ledgers}"
VALIDATION_DIR="${VALIDATION_DIR:-${LEARNED_ROOT}/validation}"
RUN_DIR_ROOT="${RUN_DIR_ROOT:-${LEARNED_ROOT}/logs}"
WORK_DIR_ROOT="${WORK_DIR_ROOT:-exps/thumos/adatad/c3_paction_lattice_replacement_adatad_full_train/${RUN_TAG}}"

PACTION_POLICY_CHECKPOINT="${PACTION_POLICY_CHECKPOINT:-}"
[[ -n "${PACTION_POLICY_CHECKPOINT}" ]] || fail "PACTION_POLICY_CHECKPOINT must point to a trained learned p_action policy checkpoint"

C3_PACTION_SOURCE_ROOT="${C3_PACTION_SOURCE_ROOT:-}"
if [[ -n "${C3_PACTION_SOURCE_ROOT}" ]]; then
  C3_PACTION_TRAIN_SOURCE_JSONL="${C3_PACTION_TRAIN_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/train/samples.jsonl}"
  C3_PACTION_VAL_SOURCE_JSONL="${C3_PACTION_VAL_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/val/samples.jsonl}"
  C3_PACTION_TEST_SOURCE_JSONL="${C3_PACTION_TEST_SOURCE_JSONL:-${C3_PACTION_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_PACTION_TRAIN_SOURCE_JSONL="${C3_PACTION_TRAIN_SOURCE_JSONL:-}"
  C3_PACTION_VAL_SOURCE_JSONL="${C3_PACTION_VAL_SOURCE_JSONL:-}"
  C3_PACTION_TEST_SOURCE_JSONL="${C3_PACTION_TEST_SOURCE_JSONL:-}"
fi

[[ -n "${C3_PACTION_TRAIN_SOURCE_JSONL}" ]] || fail "C3_PACTION_TRAIN_SOURCE_JSONL or C3_PACTION_SOURCE_ROOT is required"
[[ -n "${C3_PACTION_VAL_SOURCE_JSONL}" ]] || fail "C3_PACTION_VAL_SOURCE_JSONL or C3_PACTION_SOURCE_ROOT is required"
[[ -n "${C3_PACTION_TEST_SOURCE_JSONL}" ]] || fail "C3_PACTION_TEST_SOURCE_JSONL or C3_PACTION_SOURCE_ROOT is required"

PACTION_LATTICE_ADATAD_VARIANTS="${PACTION_LATTICE_ADATAD_VARIANTS:-paction_lattice_radius_score_only_move25}"
PACTION_LATTICE_FIXED_BUDGET="${PACTION_LATTICE_FIXED_BUDGET:-384}"
PACTION_LATTICE_DEVICE="${PACTION_LATTICE_DEVICE:-cuda}"
PACTION_LATTICE_LOCAL_RADIUS="${PACTION_LATTICE_LOCAL_RADIUS:-2}"
PACTION_LATTICE_DISTANCE_PENALTY="${PACTION_LATTICE_DISTANCE_PENALTY:-0.0}"
PACTION_LATTICE_GEOMETRY_DISTORTION_PENALTY="${PACTION_LATTICE_GEOMETRY_DISTORTION_PENALTY:-0.0}"
PACTION_LATTICE_MAX_GAP_GROWTH="${PACTION_LATTICE_MAX_GAP_GROWTH:-}"
PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE="${PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE:-0}"
PACTION_LATTICE_DISABLE_CHECKPOINT="${PACTION_LATTICE_DISABLE_CHECKPOINT:-0}"
PACTION_LATTICE_CHECKPOINT_INTERVAL="${PACTION_LATTICE_CHECKPOINT_INTERVAL:-2}"
PACTION_LATTICE_VAL_EVAL_INTERVAL="${PACTION_LATTICE_VAL_EVAL_INTERVAL:-5}"
PACTION_LATTICE_VAL_EVAL_INTERVAL_ANCHOR_EPOCH="${PACTION_LATTICE_VAL_EVAL_INTERVAL_ANCHOR_EPOCH:-5}"
PACTION_LATTICE_VAL_START_EPOCH="${PACTION_LATTICE_VAL_START_EPOCH:-4}"
PACTION_LATTICE_MIN_FREE_MB="${PACTION_LATTICE_MIN_FREE_MB:-2048}"
export C3_PACTION_ADATAD_DISABLE_CHECKPOINT="${C3_PACTION_ADATAD_DISABLE_CHECKPOINT:-${PACTION_LATTICE_DISABLE_CHECKPOINT}}"
export C3_PACTION_ADATAD_CHECKPOINT_INTERVAL="${C3_PACTION_ADATAD_CHECKPOINT_INTERVAL:-${PACTION_LATTICE_CHECKPOINT_INTERVAL}}"
export C3_PACTION_ADATAD_VAL_EVAL_INTERVAL="${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL:-${PACTION_LATTICE_VAL_EVAL_INTERVAL}}"
export C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH="${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH:-${PACTION_LATTICE_VAL_EVAL_INTERVAL_ANCHOR_EPOCH}}"
export C3_PACTION_ADATAD_VAL_START_EPOCH="${C3_PACTION_ADATAD_VAL_START_EPOCH:-${PACTION_LATTICE_VAL_START_EPOCH}}"

MAX_MAX_GAP="${MAX_MAX_GAP:-}"
MAX_P95_GAP="${MAX_P95_GAP:-}"
MAX_UNSELECTED_HOLE="${MAX_UNSELECTED_HOLE:-}"
MAX_P95_UNSELECTED_HOLE="${MAX_P95_UNSELECTED_HOLE:-}"
MAX_UNIFORM_SIMILARITY="${MAX_UNIFORM_SIMILARITY:-}"

resolve_path() {
  local raw="$1"
  if [[ "${raw}" == /* ]]; then
    echo "${raw}"
  elif [[ -f "${REPO_ROOT}/${raw}" ]]; then
    readlink -f "${REPO_ROOT}/${raw}"
  elif [[ -f "${BASE}/${raw}" ]]; then
    readlink -f "${BASE}/${raw}"
  else
    echo "${REPO_ROOT}/${raw}"
  fi
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

ADATAD_PRETRAIN_PATH="$(resolve_path "${ADATAD_PRETRAIN_PATH}")"
PACTION_POLICY_CHECKPOINT="$(resolve_path "${PACTION_POLICY_CHECKPOINT}")"
export C3_PACTION_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${CONFIG_VALIDATOR}"
require_file "${LATTICE_LEDGER_VALIDATOR}"
require_file "${LATTICE_LEDGER_PIPELINE}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${ADATAD_PRETRAIN_PATH}"
require_file "${PACTION_POLICY_CHECKPOINT}"
require_file "${C3_PACTION_TRAIN_SOURCE_JSONL}"
require_file "${C3_PACTION_VAL_SOURCE_JSONL}"
require_file "${C3_PACTION_TEST_SOURCE_JSONL}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

mkdir -p "${LEDGER_ROOT}" "${VALIDATION_DIR}" "${RUN_DIR_ROOT}" "${WORK_DIR_ROOT}"

pick_master_port() {
  local variant="$1"
  if [[ -n "${MASTER_PORT_BASE}" ]]; then
    local explicit_port="${MASTER_PORT_BASE}"
    MASTER_PORT_BASE="$((MASTER_PORT_BASE + 1))"
    echo "${explicit_port}"
    return 0
  fi
  "${PYTHON}" - "${RUN_TAG}" "${variant}" "${MASTER_PORT_LOW}" "${MASTER_PORT_HIGH}" "${MASTER_PORT_MAX_ATTEMPTS}" <<'PY'
import hashlib
import os
import socket
import sys

run_tag, variant = sys.argv[1], sys.argv[2]
low, high, max_attempts = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
if not (1024 <= low <= high <= 65535):
    raise SystemExit(f"invalid MASTER_PORT range: {low}-{high}")
span = high - low + 1
seed = "|".join(
    [
        run_tag,
        variant,
        os.environ.get("SLURM_JOB_ID", ""),
        os.environ.get("SLURM_STEP_ID", ""),
        str(os.getpid()),
    ]
)
start = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % span
for offset in range(min(max_attempts, span)):
    port = low + ((start + offset) % span)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind(("0.0.0.0", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit(f"no free MASTER_PORT found in {low}-{high} after {max_attempts} attempts")
PY
}

echo "[C3_PACTION_LATTICE_ADATAD] repo=${REPO_ROOT}"
echo "[C3_PACTION_LATTICE_ADATAD] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[C3_PACTION_LATTICE_ADATAD] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[C3_PACTION_LATTICE_ADATAD] slurm_step_gpus=${SLURM_STEP_GPUS:-none} slurm_job_gpus=${SLURM_JOB_GPUS:-none}"
echo "[C3_PACTION_LATTICE_ADATAD] precheck_only=${PRECHECK_ONLY} unlock=${ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN}"
echo "[C3_PACTION_LATTICE_ADATAD] learned_root=${LEARNED_ROOT}"
echo "[C3_PACTION_LATTICE_ADATAD] policy_checkpoint=${PACTION_POLICY_CHECKPOINT}"
echo "[C3_PACTION_LATTICE_ADATAD] adatad_pretrain_path=${C3_PACTION_ADATAD_PRETRAIN_PATH}"
echo "[C3_PACTION_LATTICE_ADATAD] variants=${PACTION_LATTICE_ADATAD_VARIANTS}"
echo "[C3_PACTION_LATTICE_ADATAD] disable_checkpoint=${C3_PACTION_ADATAD_DISABLE_CHECKPOINT} checkpoint_interval=${C3_PACTION_ADATAD_CHECKPOINT_INTERVAL}"
echo "[C3_PACTION_LATTICE_ADATAD] val_eval_interval=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL} val_eval_anchor=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH} val_start_epoch=${C3_PACTION_ADATAD_VAL_START_EPOCH}"

if [[ "${PRECHECK_ONLY}" != "1" && -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" && "${ALLOW_NON_SLURM_C3_PACTION_FULLTRAIN:-0}" != "1" ]]; then
  fail "formal full train must run inside a Slurm allocation/step; set PRECHECK_ONLY=1 for login-node checks"
fi

if [[ "${PACTION_LATTICE_DEVICE}" == cuda* ]]; then
  "${PYTHON}" - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable for PACTION_LATTICE_DEVICE=cuda; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
        f"SLURM_STEP_GPUS={os.environ.get('SLURM_STEP_GPUS', '')!r}, "
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', '')!r}. "
        "Inside a Slurm GPU step, use logical CUDA_VISIBLE_DEVICES=0."
    )
PY
fi

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  "${LATTICE_LEDGER_PIPELINE}" \
  tools/bata/apply_paction_lattice_replacement_policy.py \
  tools/bata/paction_lattice_replacement_policy.py \
  "${LATTICE_LEDGER_VALIDATOR}" \
  "${CONFIG_VALIDATOR}"

"${PYTHON}" -m pytest \
  tests/test_paction_lattice_replacement_policy.py \
  tests/test_apply_paction_lattice_replacement_policy.py \
  tests/test_paction_lattice_replacement_ledger_pipeline.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  -q

PACTION_POLICY_CHECKPOINT_SHA256="$("${PYTHON}" - "${PACTION_POLICY_CHECKPOINT}" <<'PY'
import hashlib
import sys
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
echo "[C3_PACTION_LATTICE_ADATAD] policy_checkpoint_sha256=${PACTION_POLICY_CHECKPOINT_SHA256}"

run_lattice_pipeline_for_split() {
  local split="$1"
  local input_jsonl="$2"
  local out_dir="${LEDGER_ROOT}/${split}"
  mkdir -p "${out_dir}"
  local args=(
    --input-jsonl "${input_jsonl}"
    --checkpoint-path "${PACTION_POLICY_CHECKPOINT}"
    --out-dir "${out_dir}"
    --summary-json "${out_dir}/pipeline.summary.json"
    --fixed-budget "${PACTION_LATTICE_FIXED_BUDGET}"
    --device "${PACTION_LATTICE_DEVICE}"
    --deploy-selection-ledger
    --local-radius "${PACTION_LATTICE_LOCAL_RADIUS}"
    --distance-penalty "${PACTION_LATTICE_DISTANCE_PENALTY}"
    --geometry-distortion-penalty "${PACTION_LATTICE_GEOMETRY_DISTORTION_PENALTY}"
    --variants ${PACTION_LATTICE_ADATAD_VARIANTS}
  )
  if [[ "${PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE}" == "1" ]]; then
    args+=(--allow-inferred-paction-positive-provenance)
  fi
  [[ -n "${PACTION_LATTICE_MAX_GAP_GROWTH}" ]] && args+=(--max-gap-growth "${PACTION_LATTICE_MAX_GAP_GROWTH}")
  [[ -n "${MAX_MAX_GAP}" ]] && args+=(--max-max-gap "${MAX_MAX_GAP}")
  [[ -n "${MAX_P95_GAP}" ]] && args+=(--max-p95-gap "${MAX_P95_GAP}")
  [[ -n "${MAX_UNSELECTED_HOLE}" ]] && args+=(--max-unselected-hole "${MAX_UNSELECTED_HOLE}")
  [[ -n "${MAX_P95_UNSELECTED_HOLE}" ]] && args+=(--max-p95-unselected-hole "${MAX_P95_UNSELECTED_HOLE}")
  [[ -n "${MAX_UNIFORM_SIMILARITY}" ]] && args+=(--max-uniform-similarity "${MAX_UNIFORM_SIMILARITY}")
  "${PYTHON}" "${LATTICE_LEDGER_PIPELINE}" "${args[@]}"
}

ledger_path_for_variant() {
  local split="$1"
  local variant="$2"
  echo "${LEDGER_ROOT}/${split}/value_transport_ledger_${variant}.jsonl"
}

sample_path_for_split() {
  local split="$1"
  echo "${LEDGER_ROOT}/${split}/samples.paction_lattice_replacement.jsonl"
}

metric_sample_path_for_split() {
  local split="$1"
  echo "${LEDGER_ROOT}/${split}/source.canonical_unique.jsonl"
}

validate_variant_split() {
  local variant="$1"
  local split="$2"
  local sample_jsonl
  local ledger_jsonl
  local metric_jsonl
  sample_jsonl="$(sample_path_for_split "${split}")"
  ledger_jsonl="$(ledger_path_for_variant "${split}" "${variant}")"
  metric_jsonl="$(metric_sample_path_for_split "${split}")"
  require_file "${sample_jsonl}"
  require_file "${ledger_jsonl}"
  require_file "${metric_jsonl}"
  local args=(
    --sample-jsonl "${sample_jsonl}"
    --metric-sample-jsonl "${metric_jsonl}"
    --ledger-jsonl "${ledger_jsonl}"
    --strategy "${variant}"
    --expected-target-len "${PACTION_LATTICE_FIXED_BUDGET}"
    --require-selected-count "${PACTION_LATTICE_FIXED_BUDGET}"
    --allow-short-valid-ratio-count
    --require-deployable
    --require-checkpoint-path "${PACTION_POLICY_CHECKPOINT}"
    --require-checkpoint-sha256 "${PACTION_POLICY_CHECKPOINT_SHA256}"
    --summary-json "${VALIDATION_DIR}/${split}_${variant}.validation.json"
  )
  [[ -n "${MAX_MAX_GAP}" ]] && args+=(--max-max-gap "${MAX_MAX_GAP}")
  [[ -n "${MAX_P95_GAP}" ]] && args+=(--max-p95-gap "${MAX_P95_GAP}")
  [[ -n "${MAX_UNSELECTED_HOLE}" ]] && args+=(--max-unselected-hole "${MAX_UNSELECTED_HOLE}")
  [[ -n "${MAX_P95_UNSELECTED_HOLE}" ]] && args+=(--max-p95-unselected-hole "${MAX_P95_UNSELECTED_HOLE}")
  [[ -n "${MAX_UNIFORM_SIMILARITY}" ]] && args+=(--max-uniform-similarity "${MAX_UNIFORM_SIMILARITY}")
  "${PYTHON}" "${LATTICE_LEDGER_VALIDATOR}" "${args[@]}"
}

run_adatad_variant() {
  local variant="$1"
  export C3_PACTION_LEDGER_VARIANT="${variant}"
  export C3_PACTION_LEARNED_LEDGER_ROOT="${LEDGER_ROOT}"
  export C3_PACTION_TRAIN_LEDGER_PATH
  export C3_PACTION_VAL_LEDGER_PATH
  export C3_PACTION_TEST_LEDGER_PATH
  C3_PACTION_TRAIN_LEDGER_PATH="$(ledger_path_for_variant train "${variant}")"
  C3_PACTION_VAL_LEDGER_PATH="$(ledger_path_for_variant val "${variant}")"
  C3_PACTION_TEST_LEDGER_PATH="$(ledger_path_for_variant test "${variant}")"
  export C3_PACTION_LEDGER_SOURCE="learned_paction_gap_loss_policy_checkpoint"
  export C3_PACTION_LEDGER_CONFIG_HASH="${PACTION_POLICY_CHECKPOINT_SHA256}"

  for split in train val test; do
    validate_variant_split "${variant}" "${split}"
  done

  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${CONFIG}" --require-ledger-files
  "${PYTHON}" "${CONFIG_VALIDATOR}" --config "${EXEC_CONFIG}" --require-ledger-files --allow-launch-unlocked

  if [[ "${PRECHECK_ONLY}" == "1" ]]; then
    echo "[C3_PACTION_LATTICE_ADATAD] PRECHECK_ONLY variant=${variant} target_len=${PACTION_LATTICE_FIXED_BUDGET} complete"
    return 0
  fi

  if [[ "${ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN}" != "1" ]]; then
    fail "ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN=1 is required for formal full train"
  fi
  local free_mb
  free_mb="$(df -Pm "${WORK_DIR_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ -z "${free_mb}" || "${free_mb}" -lt "${PACTION_LATTICE_MIN_FREE_MB}" ]]; then
    fail "insufficient free space for full train under ${WORK_DIR_ROOT}: free_mb=${free_mb:-unknown}, required_mb=${PACTION_LATTICE_MIN_FREE_MB}"
  fi

  local run_dir="${RUN_DIR_ROOT}/${variant}"
  local work_dir="${WORK_DIR_ROOT}/${variant}"
  local master_port
  mkdir -p "${run_dir}" "${work_dir}"
  master_port="$(pick_master_port "${variant}")"
  echo "[C3_PACTION_LATTICE_ADATAD] train variant=${variant} work_dir=${work_dir} master_port=${master_port}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port="${master_port}" \
    tools/train.py \
    "${EXEC_CONFIG}" \
    --id "${RUN_ID}" \
    --seed "${SEED}" \
    --cfg-options \
    "work_dir=${work_dir}" \
    "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    "workflow.disable_checkpoint=${C3_PACTION_ADATAD_DISABLE_CHECKPOINT}" \
    "workflow.checkpoint_interval=${C3_PACTION_ADATAD_CHECKPOINT_INTERVAL}" \
    "workflow.val_eval_interval=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL}" \
    "workflow.val_eval_interval_anchor_epoch=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH}" \
    "workflow.val_start_epoch=${C3_PACTION_ADATAD_VAL_START_EPOCH}" \
    2>&1 | tee "${run_dir}/train.out"
}

run_lattice_pipeline_for_split train "${C3_PACTION_TRAIN_SOURCE_JSONL}"
run_lattice_pipeline_for_split val "${C3_PACTION_VAL_SOURCE_JSONL}"
run_lattice_pipeline_for_split test "${C3_PACTION_TEST_SOURCE_JSONL}"

for variant in ${PACTION_LATTICE_ADATAD_VARIANTS}; do
  run_adatad_variant "${variant}"
done

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_PACTION_LATTICE_ADATAD] PRECHECK_ONLY all variants complete"
fi
