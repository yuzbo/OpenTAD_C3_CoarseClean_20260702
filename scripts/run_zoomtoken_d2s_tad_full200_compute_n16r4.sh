#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[D2S_TAD_FULL200_COMPUTE][FAIL] %s\n' "$*" >&2
  exit 2
}

require_control_free_value() {
  local name="$1"
  local value="$2"
  local character
  local ordinal
  local index
  [[ -n "${value}" ]] || fail "${name} must not be empty"
  for ((index = 0; index < ${#value}; index++)); do
    character="${value:index:1}"
    printf -v ordinal '%d' "'${character}"
    if ((ordinal < 32 || ordinal == 127)); then
      fail "${name} contains an ASCII control character"
    fi
  done
  [[ "${value}" != [[:space:]]* && "${value}" != *[[:space:]] ]] || \
    fail "${name} contains leading or trailing whitespace"
  [[ "${value}" != *,* ]] || fail "${name} contains a comma"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/d2s_tad_full200_compute}"
MANIFEST_DIR="${RUN_ROOT}/manifest"
CONTROL_DIR="${RUN_ROOT}/control"
WORK_DIR_ROOT="${RUN_ROOT}/work_dirs"
REC_DIR_ROOT="${RUN_ROOT}/recovery"
PRED_DIR="${RUN_ROOT}/predictions"
EVAL_DIR="${RUN_ROOT}/evaluation"
PROFILE_DIR="${RUN_ROOT}/profile"
ANNOTATION="${THUMOS_ANNOTATION:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
CLASS_MAP="${THUMOS_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
MEDIA_ROOT="${THUMOS_MEDIA_ROOT:-${BASE}/thumos14/raw_data/video}"
PRETRAINED="${VIDEOMAE_PRETRAINED:-${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
EXPECTED_COMMIT="${ZOOMTOKEN_EXPECTED_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
ZOOMTOKEN_SEEDS="${ZOOMTOKEN_SEEDS:-4407,4408,4409}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export ZOOMTOKEN_MATRIX_KIND=d2s

cd "${ROOT}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || \
  fail "candidate commit mismatch: ${ACTUAL_COMMIT} != ${EXPECTED_COMMIT}"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || \
  fail "candidate checkout has tracked changes"

# Optional Conda activation if on remote cluster
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi
CONDA_ACTIVATE="${BASE}/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ACTIVATE}"
fi

# Precheck mode
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf '[D2S_TAD_FULL200_COMPUTE][PRECHECK] Running static compilation...\n'
  python -m py_compile \
    opentad/models/backbones/d2s_videomae_wrapper.py \
    opentad/models/detectors/actionformer.py \
    tools/bata/d2s_tad_full200_compute.py \
    tools/bata/verify_d2s_patad_pre_run_witness.py \
    tools/bata/zoomtoken_full200_matrix_spec.py

  printf '[D2S_TAD_FULL200_COMPUTE][PRECHECK] Validating 3x3 D2S matrix and parameter disclosure...\n'
  python tools/bata/d2s_tad_full200_compute.py --root "${ROOT}"

  printf '[D2S_TAD_FULL200_COMPUTE][PRECHECK] Running test suite...\n'
  python -m pytest tests/test_d2s_tad_architecture.py \
         tests/test_d2s_tad_full200_compute.py -v

  printf '[D2S_TAD_FULL200_COMPUTE][PRECHECK] Running checkpoint-load and physical-skip witness...\n'
  python tools/bata/verify_d2s_patad_pre_run_witness.py \
    configs/adatad/thumos/continuous_roi_d2s_v3_u128_burst128_seed4407.py \
    --pretrained "${PRETRAINED}" \
    --matrix-kind d2s

  printf '[D2S_TAD_FULL200_COMPUTE][PRECHECK] PASS\n'
  exit 0
fi

# Formal execution validation
require_control_free_value "ROOT" "${ROOT}"
require_control_free_value "BASE" "${BASE}"
require_control_free_value "RUN_ROOT" "${RUN_ROOT}"
require_control_free_value "ANNOTATION" "${ANNOTATION}"
require_control_free_value "CLASS_MAP" "${CLASS_MAP}"
require_control_free_value "MEDIA_ROOT" "${MEDIA_ROOT}"
require_control_free_value "PRETRAINED" "${PRETRAINED}"
require_control_free_value "EXPECTED_COMMIT" "${EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal training requires Slurm"

mkdir -p "${MANIFEST_DIR}" "${CONTROL_DIR}" "${WORK_DIR_ROOT}" "${REC_DIR_ROOT}" "${PRED_DIR}" "${EVAL_DIR}" "${PROFILE_DIR}"

printf '[D2S_TAD_FULL200_COMPUTE] Building sealed data manifest...\n'
python tools/bata/d2s_tad_full200_compute.py \
  --annotation "${ANNOTATION}" \
  --class-map "${CLASS_MAP}" \
  --media-root "${MEDIA_ROOT}" \
  --manifest-dir "${MANIFEST_DIR}"

MANIFEST="${MANIFEST_DIR}/full_data_manifest.json"
[[ -f "${MANIFEST}" ]] || fail "manifest was not generated"

printf '[D2S_TAD_FULL200_COMPUTE] Validating matrix...\n'
MATRIX_RECEIPT="${CONTROL_DIR}/matrix_validation.json"
python tools/bata/d2s_tad_full200_compute.py \
  --root "${ROOT}" \
  --output "${MATRIX_RECEIPT}"

PROTOCOL_DOC="${ROOT}/docs/methods/d2s_tad_full200_compute_protocol.json"

python - "${ROOT}" "${MANIFEST}" "${CONTROL_DIR}" "${PRETRAINED}" "${EXPECTED_COMMIT}" "${PROTOCOL_DOC}" "${ZOOMTOKEN_SEEDS}" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
control_dir = Path(sys.argv[3])
pretrained = Path(sys.argv[4])
candidate_commit = sys.argv[5]
protocol_doc = Path(sys.argv[6])
seeds = [int(x.strip()) for x in sys.argv[7].split(",") if x.strip()]

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            d.update(b)
    return d.hexdigest()

arm_config_names = {
    "D160": "continuous_roi_s2_v3_d160",
    "G96": "continuous_roi_s2_v3_g96",
    "D2S-U128-B128": "continuous_roi_d2s_v3_u128_burst128",
}
arms = ["D160", "G96", "D2S-U128-B128"]

if not pretrained.is_file():
    raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained}")
if not protocol_doc.is_file():
    raise FileNotFoundError(f"protocol document not found: {protocol_doc}")

code_sha256 = hashlib.sha256(candidate_commit.encode("ascii")).hexdigest()
assert len(code_sha256) == 64, "protocol document SHA256 must be 64 chars"
pretrained_sha256 = sha256_file(pretrained)
assert len(pretrained_sha256) == 64

for arm in arms:
    for seed in seeds:
        config_name = f"{arm_config_names[arm]}_seed{seed}"
        config_path = root / "configs" / "adatad" / "thumos" / f"{config_name}.py"
        if not config_path.is_file():
            raise FileNotFoundError(f"cell config not found: {config_path}")
        config_sha256 = sha256_file(config_path)
        assert len(config_sha256) == 64
        hashes = {
            "code_sha256": code_sha256,
            "protocol_sha256": sha256_file(protocol_doc),
            "config_sha256": config_sha256,
            "annotation_sha256": manifest["annotation"]["sha256"],
            "class_map_sha256": manifest["class_map"]["sha256"],
            "media_manifest_sha256": manifest["media"]["records_sha256"],
            "pretrained_sha256": pretrained_sha256,
        }
        out = control_dir / f"identity_hashes_{arm}_seed{seed}.json"
        out.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote identity hashes: {out}")
PYEOF

POP_MANIFEST_SHA=$(python -c 'import json; print(json.load(open("'"${MANIFEST}"'"))["manifest_sha256"])')

# Step 1: Execute training cells across 3 arms x seeds
ARMS=(D160 G96 "D2S-U128-B128")
IFS=',' read -r -a SEED_ARRAY <<< "${ZOOMTOKEN_SEEDS}"

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEED_ARRAY[@]}"; do
    if [[ "${ARM}" == "D2S-U128-B128" ]]; then
      CONFIG="configs/adatad/thumos/continuous_roi_d2s_v3_u128_burst128_seed${SEED}.py"
    elif [[ "${ARM}" == "G96" ]]; then
      CONFIG="configs/adatad/thumos/continuous_roi_s2_v3_g96_seed${SEED}.py"
    else
      CONFIG="configs/adatad/thumos/continuous_roi_s2_v3_d160_seed${SEED}.py"
    fi
    CELL_WORK_DIR="${WORK_DIR_ROOT}/${ARM}_seed${SEED}"
    CELL_REC_DIR="${REC_DIR_ROOT}/${ARM}_seed${SEED}"
    mkdir -p "${CELL_REC_DIR}"
    CELL_ID_HASHES="${CONTROL_DIR}/identity_hashes_${ARM}_seed${SEED}.json"

    RANDOM_PORT=$(( 20000 + ( ${SLURM_JOB_ID:-$$} % 25000 ) + ( RANDOM % 5000 ) ))
    printf '[D2S_TAD_FULL200_COMPUTE] Running 2-GPU training for %s seed %d (port %d)...\n' "${ARM}" "${SEED}" "${RANDOM_PORT}"
    torchrun --nproc_per_node=2 --master_port="${RANDOM_PORT}" \
      tools/bata/continuous_roi_s2_v3_full200_compute_train.py \
      "${CONFIG}" \
      --matrix-kind d2s \
      --seed "${SEED}" \
      --expected-commit "${EXPECTED_COMMIT}" \
      --manifest "${MANIFEST}" \
      --identity-hashes "${CELL_ID_HASHES}" \
      --work-dir "${CELL_WORK_DIR}" \
      --recovery-dir "${CELL_REC_DIR}"
  done
done

# Step 2: Build post-training matrix artifact
POST_TRAIN_MATRIX="${CONTROL_DIR}/post_training_matrix.json"
python - "${ROOT}" "${WORK_DIR_ROOT}" "${EXPECTED_COMMIT}" "${POST_TRAIN_MATRIX}" "${ZOOMTOKEN_SEEDS}" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
work_dir_root = Path(sys.argv[2])
expected_commit = sys.argv[3]
out_path = Path(sys.argv[4])
seeds = [int(x.strip()) for x in sys.argv[5].split(",") if x.strip()]

arms = ["D160", "G96", "D2S-U128-B128"]
arm_config_names = {
    "D160": "continuous_roi_s2_v3_d160",
    "G96": "continuous_roi_s2_v3_g96",
    "D2S-U128-B128": "continuous_roi_d2s_v3_u128_burst128",
}

cells = []
for arm in arms:
    for seed in seeds:
        cell_work_dir = work_dir_root / f"{arm}_seed{seed}"
        checkpoint_path = cell_work_dir / "checkpoint" / "epoch_59.pth"
        terminal_receipt = cell_work_dir / "training_terminal_receipt.json"
        config_path = (
            root / "configs" / "adatad" / "thumos"
            / f"{arm_config_names[arm]}_seed{seed}.py"
        )
        for p in (checkpoint_path, terminal_receipt, config_path):
            if not p.is_file():
                raise FileNotFoundError(f"post-training artifact missing: {p}")
        cells.append({
            "arm": arm,
            "seed": seed,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "config_path": str(config_path.resolve()),
            "training_terminal_receipt_path": str(terminal_receipt.resolve()),
        })

matrix = {"candidate_commit": expected_commit, "cells": cells}
out_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote post-training matrix: {out_path}")
PYEOF

CHECKPOINT_SEAL="${CONTROL_DIR}/checkpoint_seal.json"
printf '[D2S_TAD_FULL200_COMPUTE] Sealing checkpoints...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_infer.py seal-checkpoints \
  --matrix "${POST_TRAIN_MATRIX}" \
  --population-manifest-sha256 "${POP_MANIFEST_SHA}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --output "${CHECKPOINT_SEAL}"

# Step 3: Label-free inference over 211 validation videos (792 ordered windows)
INFER_DIR_ROOT="${RUN_ROOT}/infer_work_dirs"
mkdir -p "${INFER_DIR_ROOT}"

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEED_ARRAY[@]}"; do
    CELL_INFER_WORK_DIR="${INFER_DIR_ROOT}/${ARM}_seed${SEED}"
    CELL_PRED="${PRED_DIR}/prediction_${ARM}_seed${SEED}.json"
    CELL_ID_HASHES="${CONTROL_DIR}/identity_hashes_${ARM}_seed${SEED}.json"

    printf '[D2S_TAD_FULL200_COMPUTE] Inferring %s seed %d...\n' "${ARM}" "${SEED}"
    python tools/bata/continuous_roi_s2_v3_full200_compute_infer.py infer-cell \
      --arm "${ARM}" \
      --seed "${SEED}" \
      --expected-commit "${EXPECTED_COMMIT}" \
      --manifest "${MANIFEST}" \
      --checkpoint-seal "${CHECKPOINT_SEAL}" \
      --identity-hashes "${CELL_ID_HASHES}" \
      --work-dir "${CELL_INFER_WORK_DIR}" \
      --output "${CELL_PRED}"
  done
done

# Step 4: Seal prediction bundles
PREDICTION_SEAL="${CONTROL_DIR}/prediction_seal.json"
printf '[D2S_TAD_FULL200_COMPUTE] Sealing prediction bundles...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_eval.py seal-predictions \
  --prediction-dir "${PRED_DIR}" \
  --checkpoint-seal "${CHECKPOINT_SEAL}" \
  --manifest "${MANIFEST}" \
  --output "${PREDICTION_SEAL}"

# Step 5: One-shot GT evaluation barrier
MARKER_PATH="${CONTROL_DIR}/single_gt_open.marker.json"
printf '[D2S_TAD_FULL200_COMPUTE] Running one-shot evaluation...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_eval.py evaluate-matrix \
  --prediction-seal "${PREDICTION_SEAL}" \
  --checkpoint-seal "${CHECKPOINT_SEAL}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --marker-path "${MARKER_PATH}" \
  --output-dir "${EVAL_DIR}"

printf '[D2S_TAD_FULL200_COMPUTE] Complete D2S-TAD execution finished.\n'
