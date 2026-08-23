#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-sc-cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_SINGLECLOCK_COST][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
cd "$ROOT"
EXPECTED_COMMIT="${DUCA_EVAL_COMMIT:?DUCA_EVAL_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "cost checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "cost checkout must be clean"

ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1

CLOCK_CHECKPOINT="${DUCA_CLOCK_CHECKPOINT:?DUCA_CLOCK_CHECKPOINT is required}"
CLOCK_SHA256="${DUCA_CLOCK_CHECKPOINT_SHA256:?DUCA_CLOCK_CHECKPOINT_SHA256 is required}"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:?DUCA_STAGE1_CHECKPOINT is required}"
STAGE1_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:?DUCA_STAGE1_CHECKPOINT_SHA256 is required}"
[[ -r "$CLOCK_CHECKPOINT" && -r "$STAGE1_CHECKPOINT" ]] || fail "checkpoint is unreadable"
[[ "$(sha256sum "$CLOCK_CHECKPOINT" | awk '{print $1}')" == "${CLOCK_SHA256,,}" ]] || fail "Clock checkpoint hash mismatch"
[[ "$(sha256sum "$STAGE1_CHECKPOINT" | awk '{print $1}')" == "${STAGE1_SHA256,,}" ]] || fail "Stage-1 checkpoint hash mismatch"
export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA256,,}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

OUTPUT_ROOT="${DUCA_COST_OUTPUT_ROOT:?DUCA_COST_OUTPUT_ROOT is required}"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "cost output root already exists"
mkdir -p "$OUTPUT_ROOT"
TRAINED_COMMIT="${DUCA_TRAINED_COMMIT:?DUCA_TRAINED_COMMIT is required}"
SESSION_ID="duca-singleclock-cost-${SLURM_JOB_ID}"
ON_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
ZERO_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4_gate_zero.py"

"$PYTHON" -m py_compile tools/bata/profile_duca_full_stack_cost.py \
  tools/bata/finalize_duca_h65_singleclock_cost.py
"$PYTHON" -m pytest tests/test_duca_h65_singleclock_cost.py \
  tests/test_profile_duca_full_stack_cost_cli.py -q

run_profile() {
  local role="$1" repeat="$2" order="$3" config="$4"
  PRECHECK_ONLY=0 ALLOW_RANDOM_INIT=0 CONFIG="$config" \
  PROFILE_CHECKPOINT="$CLOCK_CHECKPOINT" PROFILE_METHOD="h65-singleclock-$role" \
  PROFILE_WARMUP_SAMPLES=50 PROFILE_COMPLETE_OFFICIAL_WORKLOAD=1 \
  PROFILE_BATCH_SIZE=1 PROFILE_SAMPLE_POWER=1 \
  PROFILE_TRAINED_COMMIT="$TRAINED_COMMIT" PROFILE_EVIDENCE_COMMIT="$EXPECTED_COMMIT" \
  PROFILE_SESSION_ID="$SESSION_ID" PROFILE_PAIR_ID="singleclock-cost-r${repeat}" \
  PROFILE_REPEAT_INDEX="$repeat" PROFILE_ORDER_POSITION="$order" \
  OUTPUT_PREFIX="$OUTPUT_ROOT/${role}_r${repeat}" \
  bash scripts/run_duca_full_stack_cost_profile_gpu1.sh
}

# Sequential balanced ordering on one Slurm node; no concurrent workload.
run_profile on 1 1 "$ON_CONFIG"
run_profile gate_zero 1 2 "$ZERO_CONFIG"
run_profile gate_zero 2 1 "$ZERO_CONFIG"
run_profile on 2 2 "$ON_CONFIG"
run_profile on 3 1 "$ON_CONFIG"
run_profile gate_zero 3 2 "$ZERO_CONFIG"

"$PYTHON" tools/bata/finalize_duca_h65_singleclock_cost.py \
  --on-summary "$OUTPUT_ROOT/on_r1.summary.json" \
  --on-summary "$OUTPUT_ROOT/on_r2.summary.json" \
  --on-summary "$OUTPUT_ROOT/on_r3.summary.json" \
  --gate-zero-summary "$OUTPUT_ROOT/gate_zero_r1.summary.json" \
  --gate-zero-summary "$OUTPUT_ROOT/gate_zero_r2.summary.json" \
  --gate-zero-summary "$OUTPUT_ROOT/gate_zero_r3.summary.json" \
  --output "$OUTPUT_ROOT/singleclock_cost_gate.json"
echo "[DUCA_H65_SINGLECLOCK_COST] complete: $OUTPUT_ROOT/singleclock_cost_gate.json"
