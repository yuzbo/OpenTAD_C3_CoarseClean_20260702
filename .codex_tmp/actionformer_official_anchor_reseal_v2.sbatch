#!/usr/bin/env bash
#SBATCH --job-name=af-anchor-v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v2/slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v2/slurm-%j.err

set -euo pipefail
umask 027

RUN_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v2
OFFICIAL_REPO=/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1
AUDIT_RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260730_v16
DATA_ARCHIVE=/data/run01/sczc063/yuzibo/datasets/actionformer_official_61ea7eb_20260729_v3/thumos.tar.gz
ANNOTATION="${OFFICIAL_REPO}/data/thumos/annotations/thumos14.json"
FEATURE_DIR="${OFFICIAL_REPO}/data/thumos/i3d_features"
CHECKPOINT="${OFFICIAL_REPO}/pretrained/thumos_i3d_reproduce/epoch_034.pth.tar"
TRAIN_LOG="${OFFICIAL_REPO}/pretrained/thumos_i3d_reproduce/thumos_reproduce_log.txt"
NMS_PREFLIGHT=/data/run01/sczc063/yuzibo/projects/phystime_tad/artifacts/actionformer_official_nms_preflight_v4.py
LINUX_PREFLIGHT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/diagnostic_closure_linux_preflight_20260730_v16.log
ENV_ACTIVATE=/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
EVAL_MODEL_DIR="${RUN_ROOT}/eval_model"
EVAL_CHECKPOINT="${EVAL_MODEL_DIR}/epoch_034.pth.tar"
RAW_SOURCE="${EVAL_MODEL_DIR}/eval_results.pkl"
RAW_COPY="${RUN_ROOT}/eval_results_epoch034.pkl"
EVAL_LOG="${RUN_ROOT}/official_eval_epoch034.log"
RECORD_DIR="${RUN_ROOT}/official_record"

test -d "${RUN_ROOT}"
test ! -e "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt"
test ! -e "${RUN_ROOT}/ARTIFACT_SHA256SUMS"
test ! -e "${RUN_ROOT}/SHA256SUMS"
test ! -e "${EVAL_MODEL_DIR}"
test ! -e "${RAW_COPY}"
test ! -e "${EVAL_LOG}"
test ! -e "${RECORD_DIR}"
test "${CUDA_VISIBLE_DEVICES:-}" != ""
test -r "${ENV_ACTIVATE}"

test "$(git -C "${OFFICIAL_REPO}" rev-parse HEAD)" = 61ea7eb9308a568b0cf45e3804830836e30061de
test "$(git -C "${OFFICIAL_REPO}" rev-parse "HEAD^{tree}")" = 7b06c5261ba244788c942a0d73e304581bc35154
test -z "$(git -C "${OFFICIAL_REPO}" status --porcelain)"
test "$(git -C "${AUDIT_RUNTIME}" rev-parse HEAD)" = e2a0d74f561b158c531d4909e72ecee69b153c16
test "$(git -C "${AUDIT_RUNTIME}" rev-parse "HEAD^{tree}")" = 0b6cb7996ee90f3209a78b78bbf7a55525e3badd
test -z "$(git -C "${AUDIT_RUNTIME}" status --porcelain)"
printf '%s  %s\n' 375f76ffbf7447af1035e694971ec9b2 "${DATA_ARCHIVE}" | md5sum -c -
printf '%s  %s\n' 0f33b9bb19f1d901e38b6fc2bcb4ded3917d8aec81401b1b2cf23e41ea2c4917 "${NMS_PREFLIGHT}" | sha256sum -c -
printf '%s  %s\n' 115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c "${LINUX_PREFLIGHT}" | sha256sum -c -
printf '%s  %s\n' 003b90506d549c01dd249344d782d45ef5379d977fd54eb0712f4d40640ff8d8 "${CHECKPOINT}" | sha256sum -c -
printf '%s  %s\n' 5d4bcb2d7634774292e43347e2f6b18f67e8f4d17f62e7655de37ca57ce79c02 "${TRAIN_LOG}" | sha256sum -c -

# Do not depend on a login-shell module function inside the allocation.
source "${ENV_ACTIVATE}"
test "$(command -v python)" = /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
test "$(python -c 'import torch; print(torch.__version__)')" = 2.0.1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${OFFICIAL_REPO}/libs/utils"
python setup.py build_ext --inplace 2>&1 | tee "${RUN_ROOT}/nms_build.log"
export PYTHONPATH="${OFFICIAL_REPO}/libs/utils:${OFFICIAL_REPO}"

cd "${OFFICIAL_REPO}"
python "${NMS_PREFLIGHT}" 2>&1 | tee "${RUN_ROOT}/nms_abi_probe.log"
python - <<'PY' 2>&1 | tee "${RUN_ROOT}/environment_probe.log"
import json
from pathlib import Path

import torch
import nms_1d_cpu

extension = Path(nms_1d_cpu.__file__).resolve()
expected = Path.cwd() / "libs" / "utils"
assert extension.parent == expected
assert torch.__version__ == "2.0.1"
assert torch.cuda.is_available()
print(
    json.dumps(
        {
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_visible_devices": __import__("os").environ["CUDA_VISIBLE_DEVICES"],
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nms_extension": str(extension),
            "nms_softnms_doc": nms_1d_cpu.softnms.__doc__,
        },
        sort_keys=True,
    )
)
PY

mkdir "${EVAL_MODEL_DIR}"
ln -s "${CHECKPOINT}" "${EVAL_CHECKPOINT}"
python ./eval.py ./configs/thumos_i3d.yaml "${EVAL_CHECKPOINT}" \
    --saveonly -p 10 2>&1 | tee "${RUN_ROOT}/official_saveonly_epoch034.log"
test -s "${RAW_SOURCE}"
cp --reflink=auto --preserve=mode,timestamps "${RAW_SOURCE}" "${RAW_COPY}"

python ./eval.py ./configs/thumos_i3d.yaml "${EVAL_CHECKPOINT}" \
    -p 10 2>&1 | tee "${EVAL_LOG}"
test -s "${EVAL_LOG}"
test -z "$(git status --porcelain)"

cd "${AUDIT_RUNTIME}"
export PYTHONPATH="${AUDIT_RUNTIME}"
python tools/bata/build_actionformer_official_record.py \
    --official-repo "${OFFICIAL_REPO}" \
    --data-archive "${DATA_ARCHIVE}" \
    --annotation "${ANNOTATION}" \
    --feature-dir "${FEATURE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --raw-predictions "${RAW_COPY}" \
    --train-log "${TRAIN_LOG}" \
    --eval-log "${EVAL_LOG}" \
    --output-dir "${RECORD_DIR}" \
    --record-id "actionformer-official-61ea7eb-epoch034-n16r4-reseal-20260730-v2" \
    --checkpoint-rule ema_epoch_034 \
    2>&1 | tee "${RUN_ROOT}/official_record_builder.log"

python tools/bata/validate_actionformer_thumos_comparability.py \
    --record "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json" \
    --output "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json" \
    --require-main-table \
    2>&1 | tee "${RUN_ROOT}/strict_comparability.log"

test -s "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json"
test -s "${RECORD_DIR}/OFFICIAL_COMPARABILITY_VERDICT.json"
test -s "${RECORD_DIR}/OFFICIAL_RECORD_COMPLETE.json"
test -s "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json"
python - \
    "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json" \
    "${RECORD_DIR}/OFFICIAL_COMPARABILITY_VERDICT.json" \
    "${RECORD_DIR}/OFFICIAL_RECORD_COMPLETE.json" \
    "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json" <<'PY'
import json
import sys

paths = sys.argv[1:]
record, builder_verdict, completion, strict_verdict = [
    json.load(open(path, "r", encoding="utf-8")) for path in paths
]
expected_receipts = {
    "config",
    "readme",
    "data_archive",
    "data_manifest",
    "annotation",
    "class_map",
    "observation_manifest",
    "checkpoint",
    "raw_predictions",
    "train_log",
    "eval_log",
    "evaluator_manifest",
    "environment_manifest",
    "metric_attestation",
    "run_manifest",
}
assert record["evidence_stratum"] == "official_reproduction"
assert record["protocol_family"] == "official_actionformer_i3d_stride4_v1"
assert record["source"]["commit"] == "61ea7eb9308a568b0cf45e3804830836e30061de"
assert record["source"]["tree"] == "7b06c5261ba244788c942a0d73e304581bc35154"
assert record["training"]["seed"] == 1234567891
assert record["model"]["effective_config_sha256"] == (
    "835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4"
)
assert set(record["receipts"]) == expected_receipts
assert len(record["receipts"]) == 15
assert completion["validation_pass"] is True
assert completion["main_table_eligible"] is True
assert completion["metrics"] == record["result"]["metrics"]
for verdict in (builder_verdict, strict_verdict):
    assert verdict["main_table_eligible"] is True
    assert verdict["ineligibility_reasons"] == []
PY
test -z "$(git -C "${OFFICIAL_REPO}" status --porcelain)"
test -z "$(git -C "${AUDIT_RUNTIME}" status --porcelain)"

sha256sum \
    "${RUN_ROOT}/environment_probe.log" \
    "${RUN_ROOT}/nms_abi_probe.log" \
    "${RUN_ROOT}/nms_build.log" \
    "${RUN_ROOT}/official_saveonly_epoch034.log" \
    "${RAW_COPY}" \
    "${EVAL_LOG}" \
    "${RUN_ROOT}/official_record_builder.log" \
    "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json" \
    "${RECORD_DIR}/OFFICIAL_COMPARABILITY_VERDICT.json" \
    "${RECORD_DIR}/OFFICIAL_RECORD_COMPLETE.json" \
    "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json" \
    "${RUN_ROOT}/strict_comparability.log" \
    > "${RUN_ROOT}/ARTIFACT_SHA256SUMS.tmp"
mv "${RUN_ROOT}/ARTIFACT_SHA256SUMS.tmp" "${RUN_ROOT}/ARTIFACT_SHA256SUMS"

{
    printf 'schema_version=official_actionformer_reseal_completion_v1\n'
    printf 'validation_pass=true\n'
    printf 'main_table_eligible=true\n'
    printf 'receipt_count=15\n'
    printf 'training_seed=1234567891\n'
    printf 'effective_config_sha256=%s\n' \
        835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'official_commit=%s\n' "$(git -C "${OFFICIAL_REPO}" rev-parse HEAD)"
    printf 'official_tree=%s\n' "$(git -C "${OFFICIAL_REPO}" rev-parse "HEAD^{tree}")"
    printf 'audit_commit=%s\n' "$(git -C "${AUDIT_RUNTIME}" rev-parse HEAD)"
    printf 'audit_tree=%s\n' "$(git -C "${AUDIT_RUNTIME}" rev-parse "HEAD^{tree}")"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'artifact_sha256s=%s\n' "$(sha256sum "${RUN_ROOT}/ARTIFACT_SHA256SUMS" | awk '{print $1}')"
} > "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt.tmp"
mv \
    "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt.tmp" \
    "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt"

sha256sum \
    "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt" \
    "${RUN_ROOT}/ARTIFACT_SHA256SUMS" \
    > "${RUN_ROOT}/SHA256SUMS.tmp"
mv "${RUN_ROOT}/SHA256SUMS.tmp" "${RUN_ROOT}/SHA256SUMS"
