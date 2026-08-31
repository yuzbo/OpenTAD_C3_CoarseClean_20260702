#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-mb-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_H65_MB_EVAL][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:?DUCA_EXPECTED_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "evaluation requires a clean checkout"

module load cuda/11.8
module load miniforge3/24.11
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CALIBRATION_DIR="${DUCA_CALIBRATION_DIR:?DUCA_CALIBRATION_DIR is required}"
SEAL_ROOT="${DUCA_SEAL_ROOT:?DUCA_SEAL_ROOT is required}"
OUTPUT_ROOT="${DUCA_EVALUATION_ROOT:?DUCA_EVALUATION_ROOT is required}"
[[ -r "$ANNOTATION_PATH" ]] || fail "official annotation unreadable"
[[ -f "$CALIBRATION_DIR/pre_run_calibration.json" ]] || fail "calibration report missing"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "one-time evaluation root already exists: $OUTPUT_ROOT"

SEAL_ARGS=()
for seed in 3407 3408 3409; do
  for view in control_k384 candidate_k384 candidate_mixed; do
    seal="$SEAL_ROOT/$seed/$view/prediction_seal.json"
    [[ -f "$seal" ]] || fail "sealed prediction matrix incomplete: $seal"
    SEAL_ARGS+=(--seal "$seal")
  done
done

mkdir -p "$OUTPUT_ROOT"
exec "$PYTHON" tools/bata/evaluate_duca_h65_system_multibudget_exposure.py \
  --annotation "$ANNOTATION_PATH" \
  --calibration "$CALIBRATION_DIR/pre_run_calibration.json" \
  --held-out-ids "$CALIBRATION_DIR/held_out_video_ids.txt" \
  --bootstrap-indices "$CALIBRATION_DIR/paired_bootstrap_indices.npy" \
  "${SEAL_ARGS[@]}" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output "$OUTPUT_ROOT/one_time_evaluation.json"
