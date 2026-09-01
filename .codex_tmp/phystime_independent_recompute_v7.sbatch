#!/usr/bin/env bash
#SBATCH --job-name=pt-ind-v7
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v7/slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v7/slurm-%j.err

set -euo pipefail
umask 027

RUN_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v7
RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v10
V16_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_approach_a_20260729_v16
ANNOTATION=/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json
POLICY="${RUNTIME}/configs/adatad/thumos/phystime_decode_cross_independent_policy_v1.json"
LINUX_PREFLIGHT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/diagnostic_closure_linux_preflight_20260730_v10.log
OUTPUT_DIR="${RUN_ROOT}/independent_recompute"
OUTPUT="${OUTPUT_DIR}/INDEPENDENT_RECOMPUTE.json"

test -d "${RUN_ROOT}"
test ! -e "${OUTPUT_DIR}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = c90849c7699629b69347f4076a117cc331b51636
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 97d3af185162ae8fd1283d58cb9a9978bed6b053
test -z "$(git -C "${RUNTIME}" status --porcelain)"
printf '%s  %s\n' 1221d35d429c489c954e886c77297f0070a563adcc4a84069fdc929ac74306e9 "${LINUX_PREFLIGHT}" | sha256sum -c -

module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir "${OUTPUT_DIR}"
{
    printf 'runtime_commit=%s\n' "$(git -C "${RUNTIME}" rev-parse HEAD)"
    printf 'runtime_tree=%s\n' "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'gpu_used_by_validator=false\n'
} > "${RUN_ROOT}/runtime_identity.txt"

sha256sum \
    "${ANNOTATION}" \
    "${POLICY}" \
    "${V16_ROOT}/selected_online/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/selected_ema/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/physical_online/DECODE_CROSS_COMPLETE.json" \
    "${V16_ROOT}/physical_ema/DECODE_CROSS_COMPLETE.json" \
    > "${RUN_ROOT}/input_sha256s.txt"

cd "${RUNTIME}"
python tools/bata/validate_phystime_independent_recompute.py \
    --completion "${V16_ROOT}/selected_online/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/selected_ema/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/physical_online/DECODE_CROSS_COMPLETE.json" \
    --completion "${V16_ROOT}/physical_ema/DECODE_CROSS_COMPLETE.json" \
    --annotation "${ANNOTATION}" \
    --policy "${POLICY}" \
    --output "${OUTPUT}" \
    --proposal-atol 0.0001 \
    --metric-atol 0.0001 \
    2>&1 | tee "${OUTPUT_DIR}/validator.log"

test -s "${OUTPUT}"
python - "${OUTPUT}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["schema_version"] == "phystime_independent_recompute_v2"
assert payload["validation_pass"] is True
assert payload["status"] == "tested"
assert payload["issues"] == []
implementation = payload["independent_implementation"]
assert implementation["geometry_dtype"] == "source_float32"
assert implementation["nms_dtype"] == "float64"
assert implementation["evaluation_dtype"] == "float64"
assert len(payload["completion_reports"]) == 4
for report in payload["completion_reports"]:
    assert report["validation_pass"] is True
    assert report["issues"] == []
    for axis in report["axis_reports"].values():
        assert axis["proposal_max_abs_error_seconds"] <= 0.0001
        assert axis["rebuilt_pre_cross_tolerance_match"] is True
        assert axis["independent_result_tolerance_match"] is True
        assert axis["max_abs_metric_delta"] <= 0.0001
PY
test -z "$(git status --porcelain)"
sha256sum \
    "${OUTPUT}" \
    "${OUTPUT_DIR}/validator.log" \
    "${RUN_ROOT}/input_sha256s.txt" \
    "${RUN_ROOT}/runtime_identity.txt" \
    > "${RUN_ROOT}/SHA256SUMS"
