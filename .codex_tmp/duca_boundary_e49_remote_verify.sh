#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=e49ef69605e1f98a7217957483f93a8a64bfc348
DENSE_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_e49ef69_20260722
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
VERIFY=${BASE}/projects/c3_lowres_action_probe/duca_boundary_e49ef69_verify_20260722
DENSE_RUN=${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
DENSE_CONFIG=${DENSE_SNAPSHOT}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
DENSE_CHECKPOINT=${DENSE_RUN}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
DENSE_TRAINING=${DENSE_RUN}/work_dir/gpu1_id0/log.json
DENSE_EVALUATION=${DENSE_RUN}/train.out
DENSE_BINDING=${VERIFY}/dense_checkpoint_binding.json

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

if [[ ! -d ${SNAPSHOT}/.git ]]; then
  git clone --branch codex/duca-boundary-burst-20260722 --single-branch \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git \
    ${SNAPSHOT}
fi
cd ${SNAPSHOT}
git fetch origin codex/duca-boundary-burst-20260722
git checkout --detach ${COMMIT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

if [[ ! -d ${DENSE_SNAPSHOT}/.git ]]; then
  git clone --no-checkout \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git \
    ${DENSE_SNAPSHOT}
fi
git -C ${DENSE_SNAPSHOT} fetch origin ${DENSE_COMMIT}
git -C ${DENSE_SNAPSHOT} checkout --detach ${DENSE_COMMIT}
[[ "$(git -C ${DENSE_SNAPSHOT} rev-parse HEAD)" == "${DENSE_COMMIT}" ]]
[[ -z "$(git -C ${DENSE_SNAPSHOT} status --porcelain --untracked-files=normal)" ]]
[[ -f ${DENSE_CONFIG} && -f ${DENSE_CHECKPOINT} && -f ${DENSE_TRAINING} && -f ${DENSE_EVALUATION} ]]

python -m py_compile \
  tools/train.py tools/test.py \
  tools/bata/duca_selected_axis_training.py \
  tools/bata/duca_r5_paper_matrix.py \
  tools/bata/aggregate_duca_r5_paper_matrix.py \
  tools/bata/profile_duca_full_stack_cost.py \
  tools/bata/duca_trained_checkpoint_binding.py \
  tools/bata/run_duca_temporalmaxer_one_step.py

python -m pytest \
  tests/test_duca_boundary_burst_selection.py \
  tests/test_duca_boundary_burst_runtime_binding.py \
  tests/test_duca_boundary_burst_hard_swap_alignment.py \
  tests/test_duca_boundary_burst_full_model_gate.py \
  tests/test_duca_boundary_burst_configs.py \
  tests/test_duca_boundary_burst_artifact_contract.py \
  tests/test_aggregate_duca_r5_paper_matrix.py \
  tests/test_profile_duca_full_stack_cost_cli.py \
  tests/test_duca_selected_axis_optimization_configs.py \
  tests/test_duca_r5_paper_matrix.py \
  tests/test_duca_r0_holdout_replay.py \
  tests/test_duca_r0_evidence_contract.py \
  tests/test_duca_r0_boundary_burst_oracle.py \
  tests/test_duca_full_stack_cost.py \
  tests/test_duca_boundary_burst_submission_journal.py \
  tests/test_duca_trained_checkpoint_binding.py -q

python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

mkdir -p ${VERIFY}
python -c "from mmengine.config import Config; from tools.bata.profile_duca_full_stack_cost import _payload_fingerprint; from tools.bata.duca_trained_checkpoint_binding import build_trained_checkpoint_binding,write_trained_checkpoint_binding; cfg=Config.fromfile('${DENSE_CONFIG}'); payload=build_trained_checkpoint_binding(role='dense_adatad_baseline',git_commit='${DENSE_COMMIT}',config_path='${DENSE_CONFIG}',resolved_config_sha256=_payload_fingerprint(cfg),checkpoint_path='${DENSE_CHECKPOINT}',checkpoint_epoch=59,checkpoint_state_key='state_dict_ema',training_evidence_path='${DENSE_TRAINING}',evaluation_evidence_path='${DENSE_EVALUATION}'); write_trained_checkpoint_binding('${DENSE_BINDING}',payload)"
sha256sum ${DENSE_BINDING} | awk '{print $1}' > ${DENSE_BINDING}.sha256

python -m tools.bata.duca_r5_paper_matrix \
  --repo-root ${SNAPSHOT} \
  --output-dir ${VERIFY}/r5_generated \
  --uniform-config ${SNAPSHOT}/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py \
  --learned-config ${SNAPSHOT}/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py \
  --dense-config ${DENSE_CONFIG} \
  --dense-checkpoint ${DENSE_CHECKPOINT} \
  --dense-checkpoint-evidence ${DENSE_BINDING} \
  --dense-trained-commit ${DENSE_COMMIT} \
  --cluster n16r4 >/dev/null

bash -n scripts/launch_duca_r5_paper_matrix.sh
bash -n scripts/run_duca_full_stack_cost_profile_gpu1.sh
while IFS= read -r script; do bash -n "${script}"; done \
  < <(find ${VERIFY}/r5_generated/jobs -type f -name '*.sbatch' | sort)
[[ "$(find ${VERIFY}/r5_generated/configs -type f -name '*.py' | wc -l)" == 24 ]]
[[ "$(find ${VERIFY}/r5_generated/jobs -type f -name '*.sbatch' | wc -l)" == 35 ]]
[[ "$(sha256sum ${VERIFY}/r5_generated/matrix_summary.json | awk '{print $1}')" == \
   "$(cat ${VERIFY}/r5_generated/matrix_summary.json.sha256)" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

echo "VERIFY_OK commit=${COMMIT} snapshot=${SNAPSHOT} dense_snapshot=${DENSE_SNAPSHOT} generated=${VERIFY}/r5_generated"
