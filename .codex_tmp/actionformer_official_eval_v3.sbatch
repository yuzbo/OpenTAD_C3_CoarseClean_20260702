#!/usr/bin/env bash
#SBATCH --job-name=af-official-61ea
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v3/slurm-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v3/slurm-%j.err

set -euo pipefail
umask 027

RUN_ROOT=/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_repro_20260729_v3
OFFICIAL_REPO=/data/run01/sczc063/yuzibo/projects/actionformer_official_61ea7eb_20260729_v1
AUDIT_RUNTIME=/data/run01/sczc063/yuzibo/projects/opentad_sparsehead_diagnostic_closure_20260729_v6
DATA_ARCHIVE=/data/run01/sczc063/yuzibo/datasets/actionformer_official_61ea7eb_20260729_v3/thumos.tar.gz
ANNOTATION="${OFFICIAL_REPO}/data/thumos/annotations/thumos14.json"
FEATURE_DIR="${OFFICIAL_REPO}/data/thumos/i3d_features"
CHECKPOINT="${OFFICIAL_REPO}/pretrained/thumos_i3d_reproduce/epoch_034.pth.tar"
TRAIN_LOG="${OFFICIAL_REPO}/pretrained/thumos_i3d_reproduce/thumos_reproduce_log.txt"
RAW_SOURCE="${OFFICIAL_REPO}/pretrained/thumos_i3d_reproduce/eval_results.pkl"
RAW_COPY="${RUN_ROOT}/eval_results_epoch034.pkl"
EVAL_LOG="${RUN_ROOT}/official_eval_epoch034.log"
RECORD_DIR="${RUN_ROOT}/official_record"

test -d "${RUN_ROOT}"
test ! -e "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt"
test ! -e "${RAW_COPY}"
test ! -e "${EVAL_LOG}"
test ! -e "${RECORD_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test "${CUDA_VISIBLE_DEVICES:-}" != ""
test "$(git -C "${OFFICIAL_REPO}" rev-parse HEAD)" = 61ea7eb9308a568b0cf45e3804830836e30061de
test "$(git -C "${OFFICIAL_REPO}" rev-parse "HEAD^{tree}")" = 7b06c5261ba244788c942a0d73e304581bc35154
test -z "$(git -C "${OFFICIAL_REPO}" status --porcelain)"
test "$(git -C "${AUDIT_RUNTIME}" rev-parse HEAD)" = 17a7b55071e3ac9da1f5a6611a84f52b98db3e21
test "$(git -C "${AUDIT_RUNTIME}" rev-parse "HEAD^{tree}")" = 7cad5c56392d1f7cf296158688083a4f5e327131
test -z "$(git -C "${AUDIT_RUNTIME}" status --porcelain)"
echo "375f76ffbf7447af1035e694971ec9b2  ${DATA_ARCHIVE}" | md5sum -c -

cd "${OFFICIAL_REPO}/libs/utils"
python setup.py build_ext --inplace 2>&1 | tee "${RUN_ROOT}/nms_build.log"

cd "${OFFICIAL_REPO}"
python - <<'PY' 2>&1 | tee "${RUN_ROOT}/environment_probe.log"
import json
import torch
import libs.utils.nms

assert torch.cuda.is_available()
print(
    json.dumps(
        {
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nms_import": "libs.utils.nms",
        },
        sort_keys=True,
    )
)
PY

python ./eval.py ./configs/thumos_i3d.yaml ./pretrained/thumos_i3d_reproduce/ \
    -epoch 34 --saveonly -p 10 2>&1 | tee "${RUN_ROOT}/official_saveonly_epoch034.log"
test -s "${RAW_SOURCE}"
cp --reflink=auto --preserve=mode,timestamps "${RAW_SOURCE}" "${RAW_COPY}"

python ./eval.py ./configs/thumos_i3d.yaml ./pretrained/thumos_i3d_reproduce/ \
    -epoch 34 -p 10 2>&1 | tee "${EVAL_LOG}"
test -s "${EVAL_LOG}"
test -z "$(git status --porcelain)"

cd "${AUDIT_RUNTIME}"
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
    --record-id "actionformer-official-61ea7eb-epoch034-n16r4-20260729" \
    --checkpoint-rule ema_epoch_034 \
    2>&1 | tee "${RUN_ROOT}/official_record_builder.log"

python tools/bata/validate_actionformer_thumos_comparability.py \
    --record "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json" \
    --output "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json" \
    --require-main-table \
    2>&1 | tee "${RUN_ROOT}/strict_comparability.log"

test -s "${RECORD_DIR}/OFFICIAL_RECORD_COMPLETE.json"
test -s "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json"
test -z "$(git -C "${OFFICIAL_REPO}" status --porcelain)"
test -z "$(git -C "${AUDIT_RUNTIME}" status --porcelain)"

{
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'official_commit=%s\n' "$(git -C "${OFFICIAL_REPO}" rev-parse HEAD)"
    printf 'official_tree=%s\n' "$(git -C "${OFFICIAL_REPO}" rev-parse "HEAD^{tree}")"
    printf 'audit_commit=%s\n' "$(git -C "${AUDIT_RUNTIME}" rev-parse HEAD)"
    printf 'audit_tree=%s\n' "$(git -C "${AUDIT_RUNTIME}" rev-parse "HEAD^{tree}")"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
} > "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt"

sha256sum \
    "${RUN_ROOT}/OFFICIAL_EVAL_STAGE_COMPLETE.txt" \
    "${RUN_ROOT}/environment_probe.log" \
    "${RUN_ROOT}/nms_build.log" \
    "${RUN_ROOT}/official_saveonly_epoch034.log" \
    "${RAW_COPY}" \
    "${EVAL_LOG}" \
    "${RECORD_DIR}/OFFICIAL_PROTOCOL_RECORD.json" \
    "${RECORD_DIR}/OFFICIAL_RECORD_COMPLETE.json" \
    "${RUN_ROOT}/STRICT_COMPARABILITY_VERDICT.json" \
    > "${RUN_ROOT}/SHA256SUMS"
