#!/usr/bin/env bash
set -euo pipefail
umask 027

RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v8
LOG=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/diagnostic_closure_linux_preflight_20260729_v8.log

test -d "${RUNTIME}"
test ! -e "${LOG}"
test "$(git -C "${RUNTIME}" rev-parse HEAD)" = 2b074845497f6ada3314cb895f0d4ab2f4ce3eca
test "$(git -C "${RUNTIME}" rev-parse "HEAD^{tree}")" = 7779862c5422dc8e527b304bf881a760b0c90625
test -z "$(git -C "${RUNTIME}" status --porcelain)"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1

cd "${RUNTIME}"
{
    python -m py_compile \
        opentad/models/projections/phystime_projection.py \
        tools/bata/validate_phystime_independent_recompute.py \
        tools/bata/audit_sdpq_support_observability.py \
        tools/bata/build_actionformer_official_record.py \
        tools/bata/validate_actionformer_thumos_comparability.py
    python -m pytest \
        tests/test_actionformer_official_record_builder.py \
        tests/test_actionformer_thumos_comparability.py \
        tests/test_phystime_decode_cross_evidence_suite.py \
        tests/test_phystime_decode_cross_replay.py \
        tests/test_phystime_fullprecision_nms_replay.py \
        tests/test_phystime_independent_recompute.py \
        tests/test_sdpq_support_observability_audit.py \
        tests/test_support_decoupled_physical_query_head.py \
        -q
} 2>&1 | tee "${LOG}"

test -z "$(git status --porcelain)"
sha256sum "${LOG}"
