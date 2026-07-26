#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "${name} is required" >&2
        exit 2
    fi
}

for name in \
    DUCA_DIAGNOSTIC_REPO_ROOT \
    DUCA_DIAGNOSTIC_EXPECTED_COMMIT \
    DUCA_DIAGNOSTIC_STAGE2_ROOT \
    DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT \
    DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT_SHA256 \
    DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT \
    DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT_SHA256 \
    DUCA_DIAGNOSTIC_PRETRAIN; do
    require_env "$name"
done

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
    cd "$DUCA_DIAGNOSTIC_REPO_ROOT"
    [[ "$(git rev-parse HEAD)" == "$DUCA_DIAGNOSTIC_EXPECTED_COMMIT" ]]
    [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
    [[ "$(sha256sum "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT" | awk '{print $1}')" == "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT_SHA256" ]]
    [[ "$(sha256sum "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT" | awk '{print $1}')" == "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT_SHA256" ]]
    python -m py_compile \
        tools/bata/diagnose_duca_stage2_nonfinite_loss.py \
        tools/bata/diagnose_duca_stage2_postupdate_nonfinite.py
    python -m pytest \
        tests/test_duca_stage2_nonfinite_diagnostic.py \
        tests/test_duca_stage2_postupdate_nonfinite_diagnostic.py -q
    echo "PRECHECK_OK"
    exit 0
fi

export DUCA_DIAGNOSTIC_TRIALS="${DUCA_DIAGNOSTIC_TRIALS:-8}"
export DUCA_DIAGNOSTIC_PREFIX_UPDATE_COUNT="${DUCA_DIAGNOSTIC_PREFIX_UPDATE_COUNT:-1}"
export DUCA_DIAGNOSTIC_TARGET_BATCH_INDEX="${DUCA_DIAGNOSTIC_TARGET_BATCH_INDEX:-$DUCA_DIAGNOSTIC_PREFIX_UPDATE_COUNT}"
short_commit="${DUCA_DIAGNOSTIC_EXPECTED_COMMIT:0:12}"
output_dir="${DUCA_DIAGNOSTIC_OUTPUT_ROOT:-$DUCA_DIAGNOSTIC_STAGE2_ROOT/stage2/nonfinite_diagnostics/epoch_10_prefix${DUCA_DIAGNOSTIC_PREFIX_UPDATE_COUNT}_target${DUCA_DIAGNOSTIC_TARGET_BATCH_INDEX}_${short_commit}}"
mkdir -p "$output_dir"

export DUCA_DIAGNOSTIC_OUTPUT_DIR="$output_dir"
job_id="$(sbatch --parsable \
    --job-name="duca-rate-e10-postupdate-diag" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --time=00:30:00 \
    --output="$output_dir/slurm-%j.out" \
    --error="$output_dir/slurm-%j.err" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
set +u
source /etc/profile
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
cd "$DUCA_DIAGNOSTIC_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$DUCA_DIAGNOSTIC_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "$(sha256sum "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT" | awk '{print $1}')" == "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT_SHA256" ]]
[[ "$(sha256sum "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT" | awk '{print $1}')" == "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT_SHA256" ]]
python -m tools.bata.diagnose_duca_stage2_postupdate_nonfinite \
    --config configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py \
    --checkpoint "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT" \
    --checkpoint-sha256 "$DUCA_DIAGNOSTIC_STAGE2_CHECKPOINT_SHA256" \
    --expected-checkpoint-epoch 9 \
    --expected-commit "$DUCA_DIAGNOSTIC_EXPECTED_COMMIT" \
    --pretrain "$DUCA_DIAGNOSTIC_PRETRAIN" \
    --stage1-checkpoint "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT" \
    --stage1-checkpoint-sha256 "$DUCA_DIAGNOSTIC_STAGE1_CHECKPOINT_SHA256" \
    --stage1-checkpoint-epoch 29 \
    --epoch 10 \
    --trials "$DUCA_DIAGNOSTIC_TRIALS" \
    --prefix-update-count "$DUCA_DIAGNOSTIC_PREFIX_UPDATE_COUNT" \
    --target-batch-index "$DUCA_DIAGNOSTIC_TARGET_BATCH_INDEX" \
    --output-json "$DUCA_DIAGNOSTIC_OUTPUT_DIR/report.json" \
    --device cuda:0 \
    > "$DUCA_DIAGNOSTIC_OUTPUT_DIR/diagnostic.out"
printf 'COMPLETED job=%s\n' "${SLURM_JOB_ID}" > "$DUCA_DIAGNOSTIC_OUTPUT_DIR/completion.txt"
SBATCH
)"
printf '%s\n' "$job_id" | tee "$output_dir/job_id.txt"
