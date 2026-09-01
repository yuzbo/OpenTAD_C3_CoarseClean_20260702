#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[S2_V3_FULL200_COMPUTE][FAIL] %s\n' "$*" >&2
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
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/continuous_roi_s2_v3_full200_compute}"
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
PRETRAINED="${VIDEOMAE_PRETRAINED:-${BASE}/pretrained/videomae_s_768x1_160_adapter.pth}"
EXPECTED_COMMIT="${ZOOMTOKEN_EXPECTED_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

cd "${ROOT}"

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

# Precheck mode: compile & validate configs, manifests, parameter surfaces, and unit tests
if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  printf '[S2_V3_FULL200_COMPUTE][PRECHECK] Running static compilation...\n'
  python -m py_compile \
    tools/bata/continuous_roi_s2_v3_full200_compute.py \
    tools/bata/continuous_roi_s2_v3_full200_compute_train.py \
    tools/bata/continuous_roi_s2_v3_full200_compute_infer.py \
    tools/bata/continuous_roi_s2_v3_full200_compute_eval.py \
    tools/bata/continuous_roi_s2_v3_full200_compute_profile.py

  printf '[S2_V3_FULL200_COMPUTE][PRECHECK] Validating 3x3 matrix and parameter surface fairness...\n'
  python tools/bata/continuous_roi_s2_v3_full200_compute.py --root "${ROOT}"

  printf '[S2_V3_FULL200_COMPUTE][PRECHECK] Running test suite...\n'
  pytest tests/test_continuous_roi_s2_v3_full200_compute.py \
         tests/test_continuous_roi_s2_v3_full200_compute_recovery.py \
         tests/test_continuous_roi_s2_v3_full200_compute_statistics.py -v

  printf '[S2_V3_FULL200_COMPUTE][PRECHECK] PASS\n'
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

printf '[S2_V3_FULL200_COMPUTE] Building sealed data manifest...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute.py \
  --annotation "${ANNOTATION}" \
  --class-map "${CLASS_MAP}" \
  --media-root "${MEDIA_ROOT}" \
  --manifest-dir "${MANIFEST_DIR}"

MANIFEST="${MANIFEST_DIR}/full_data_manifest.json"
[[ -f "${MANIFEST}" ]] || fail "manifest was not generated"

printf '[S2_V3_FULL200_COMPUTE] Validating matrix...\n'
MATRIX_RECEIPT="${CONTROL_DIR}/matrix_validation.json"
python tools/bata/continuous_roi_s2_v3_full200_compute.py \
  --root "${ROOT}" \
  --output "${MATRIX_RECEIPT}"

IDENTITY_HASHES="${CONTROL_DIR}/identity_hashes.json"
python -c '
import hashlib, json
from pathlib import Path
root = Path("'${ROOT}'").resolve()
manifest = json.loads(Path("'${MANIFEST}'").read_text(encoding="utf-8"))
def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            d.update(b)
    return d.hexdigest()
hashes = {
    "code_sha256": "'${EXPECTED_COMMIT}'",
    "protocol_sha256": sha(root / "docs/methods/continuous_roi_s2_v3_full200_compute_protocol.json"),
    "config_sha256": sha(root / "configs/adatad/thumos/continuous_roi_s2_v3_d160_seed4407.py"),
    "annotation_sha256": manifest["annotation"]["sha256"],
    "class_map_sha256": manifest["class_map"]["sha256"],
    "media_manifest_sha256": manifest["media"]["records_sha256"],
    "pretrained_sha256": sha("'${PRETRAINED}'"),
}
Path("'${IDENTITY_HASHES}'").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
print("Wrote identity hashes to", "'${IDENTITY_HASHES}'")
'

POP_MANIFEST_SHA=$(python -c 'import json; print(json.load(open("'${MANIFEST}'"))["manifest_sha256"])')

# Step 1: Execute 9 training cells across 3 arms x 3 seeds
ARMS=("D160" "G96" "U128-A0")
SEEDS=(4407 4408 4409)

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    if [[ "${ARM}" == "U128-A0" ]]; then
      CONFIG="configs/adatad/thumos/continuous_roi_s2_v3_u128_a0_seed${SEED}.py"
    elif [[ "${ARM}" == "G96" ]]; then
      CONFIG="configs/adatad/thumos/continuous_roi_s2_v3_g96_seed${SEED}.py"
    else
      CONFIG="configs/adatad/thumos/continuous_roi_s2_v3_d160_seed${SEED}.py"
    fi
    CELL_WORK_DIR="${WORK_DIR_ROOT}/${ARM}_seed${SEED}"
    CELL_REC_DIR="${REC_DIR_ROOT}/${ARM}_seed${SEED}"
    mkdir -p "${CELL_WORK_DIR}" "${CELL_REC_DIR}"

    printf '[S2_V3_FULL200_COMPUTE] Running 2-GPU training for %s seed %d...\n' "${ARM}" "${SEED}"
    torchrun --nproc_per_node=2 \
      tools/bata/continuous_roi_s2_v3_full200_compute_train.py \
      "${CONFIG}" \
      --seed "${SEED}" \
      --expected-commit "${EXPECTED_COMMIT}" \
      --manifest "${MANIFEST}" \
      --identity-hashes "${IDENTITY_HASHES}" \
      --work-dir "${CELL_WORK_DIR}" \
      --recovery-dir "${CELL_REC_DIR}"
  done
done

# Step 2: Seal all 9 checkpoints
CHECKPOINT_SEAL="${CONTROL_DIR}/checkpoint_seal.json"
printf '[S2_V3_FULL200_COMPUTE] Sealing all 9 checkpoints...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_infer.py seal-checkpoints \
  --matrix "${MATRIX_RECEIPT}" \
  --population-manifest-sha256 "${POP_MANIFEST_SHA}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --output "${CHECKPOINT_SEAL}"

# Step 3: Label-free inference over 211 validation videos (792 ordered windows)
for ARM in "${ARMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CELL_WORK_DIR="${WORK_DIR_ROOT}/${ARM}_seed${SEED}"
    CELL_PRED="${PRED_DIR}/prediction_${ARM}_seed${SEED}.json"

    printf '[S2_V3_FULL200_COMPUTE] Inferring %s seed %d...\n' "${ARM}" "${SEED}"
    python tools/bata/continuous_roi_s2_v3_full200_compute_infer.py infer-cell \
      --arm "${ARM}" \
      --seed "${SEED}" \
      --expected-commit "${EXPECTED_COMMIT}" \
      --manifest "${MANIFEST}" \
      --checkpoint-seal "${CHECKPOINT_SEAL}" \
      --identity-hashes "${IDENTITY_HASHES}" \
      --work-dir "${CELL_WORK_DIR}" \
      --output "${CELL_PRED}"
  done
done

# Step 4: Seal all 9 predictions
PREDICTION_SEAL="${CONTROL_DIR}/prediction_seal.json"
printf '[S2_V3_FULL200_COMPUTE] Sealing all 9 prediction bundles...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_eval.py seal-predictions \
  --prediction-dir "${PRED_DIR}" \
  --checkpoint-seal "${CHECKPOINT_SEAL}" \
  --manifest "${MANIFEST}" \
  --output "${PREDICTION_SEAL}"

# Step 5: One-shot GT evaluation barrier and 20,000 bootstrap replicates
MARKER_PATH="${CONTROL_DIR}/single_gt_open.marker.json"
printf '[S2_V3_FULL200_COMPUTE] Running one-shot evaluation and 20k bootstrap...\n'
python tools/bata/continuous_roi_s2_v3_full200_compute_eval.py evaluate-matrix \
  --prediction-seal "${PREDICTION_SEAL}" \
  --checkpoint-seal "${CHECKPOINT_SEAL}" \
  --manifest "${MANIFEST}" \
  --annotation "${ANNOTATION}" \
  --marker-path "${MARKER_PATH}" \
  --output-dir "${EVAL_DIR}"

# Step 6: Generate and compare C_exec arithmetic ledgers across all arms.
# This step produces the primary resource axis (rho_C <= 0.90 gate).
# Diagnostic-only metrics (latency/memory/energy) are also disclosed here.
# NOTE: this step requires that the profile tool receives pre-built event JSON
# files produced by the actual inference runs (via --input). The launcher
# invokes the comparison utility once all three arm ledgers are available.
C_EXEC_D160="${PROFILE_DIR}/c_exec_d160.json"
C_EXEC_G96="${PROFILE_DIR}/c_exec_g96.json"
C_EXEC_U128="${PROFILE_DIR}/c_exec_u128_a0.json"

# Collect per-arm C_exec ledger receipts produced during inference.
# Each arm's infer-cell is expected to have written its event JSON to PROFILE_DIR.
for ARM in "${ARMS[@]}"; do
  ARM_SLUG="${ARM//-/_}"
  ARM_SLUG="${ARM_SLUG// /_}"
  LEDGER_INPUT="${PROFILE_DIR}/c_exec_events_${ARM_SLUG}.json"
  [[ -f "${LEDGER_INPUT}" ]] || fail "C_exec event ledger missing for arm ${ARM}: ${LEDGER_INPUT}"
  case "${ARM}" in
    D160)    LEDGER_OUT="${C_EXEC_D160}" ;;
    G96)     LEDGER_OUT="${C_EXEC_G96}" ;;
    U128-A0) LEDGER_OUT="${C_EXEC_U128}" ;;
  esac
  printf '[S2_V3_FULL200_COMPUTE] Sealing C_exec ledger for %s...\n' "${ARM}"
  python tools/bata/continuous_roi_s2_v3_full200_compute_profile.py \
    --input "${LEDGER_INPUT}" \
    --output "${LEDGER_OUT}"
done

printf '[S2_V3_FULL200_COMPUTE] Comparing C_exec across arms...\n'
C_EXEC_COMPARISON="${PROFILE_DIR}/c_exec_comparison.json"
python - "${C_EXEC_D160}" "${C_EXEC_G96}" "${C_EXEC_U128}" "${C_EXEC_COMPARISON}" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from tools.bata.continuous_roi_s2_v3_full200_compute_profile import compare_c_exec_receipts
from tools.bata.continuous_roi_s2_v3_full200_compute import atomic_publish_json

receipts = {
    "D160": json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")),
    "G96": json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")),
    "U128-A0": json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")),
}
result = compare_c_exec_receipts(receipts)
atomic_publish_json(Path(sys.argv[4]), result)
gate = result["primary_exact_10u_le_9d"]
print(json.dumps({"status": "PASS" if gate else "GATE_FAIL_rho_C_gt_0.90",
                  "primary_gate_10u_le_9d": gate,
                  "ratio_u128_over_d160": result["ratio_disclosure"]["u128_a0_over_d160"]},
                 sort_keys=True))
if not gate:
    raise SystemExit(2)
PYEOF

printf '[S2_V3_FULL200_COMPUTE] Complete execution finished successfully.\n'
