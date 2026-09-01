#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=62eb52bbc9c68d07c68aef7e53517b4716872870
BRANCH=codex/duca-boundary-burst-20260722
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_62eb52b_20260722
VERIFY=${BASE}/projects/c3_lowres_action_probe/duca_boundary_62eb52b_verify_20260722_2015
R0_CHECKPOINT=${BASE}/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth
DENSE_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
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
  git clone --branch ${BRANCH} --single-branch \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git \
    ${SNAPSHOT}
fi
cd ${SNAPSHOT}
git fetch origin ${BRANCH}
git checkout --detach ${COMMIT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

python -m py_compile \
  opentad/models/duca/acquisition.py \
  opentad/models/duca/structured_selection.py \
  opentad/models/duca/transition_only.py \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  tools/bata/duca_r5_paper_matrix.py \
  tools/bata/aggregate_duca_r5_paper_matrix.py

python -m pytest \
  tests/test_duca_transition_only.py \
  tests/test_duca_structured_selection.py \
  tests/test_duca_online_coarse_probe_actionness.py \
  tests/test_duca_boundary_burst_configs.py \
  tests/test_duca_frontend_p0_contract.py \
  tests/test_duca_r5_paper_matrix.py \
  tests/test_aggregate_duca_r5_paper_matrix.py \
  tests/test_duca_r0_r5_parallel_bundles.py -q

python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

bash -n \
  scripts/run_duca_independent_official60_gpu1.sh \
  scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
  scripts/submit_duca_r0_r5_parallel_bundles.sh

mkdir -p ${VERIFY}
[[ -f ${DENSE_CONFIG} && -f ${DENSE_CHECKPOINT} && -f ${DENSE_TRAINING} && -f ${DENSE_EVALUATION} ]]
python -c "from mmengine.config import Config; from tools.bata.profile_duca_full_stack_cost import _payload_fingerprint; from tools.bata.duca_trained_checkpoint_binding import build_trained_checkpoint_binding,write_trained_checkpoint_binding; cfg=Config.fromfile('${DENSE_CONFIG}'); payload=build_trained_checkpoint_binding(role='dense_adatad_baseline',git_commit='${DENSE_COMMIT}',config_path='${DENSE_CONFIG}',resolved_config_sha256=_payload_fingerprint(cfg),checkpoint_path='${DENSE_CHECKPOINT}',checkpoint_epoch=59,checkpoint_state_key='state_dict_ema',training_evidence_path='${DENSE_TRAINING}',evaluation_evidence_path='${DENSE_EVALUATION}'); write_trained_checkpoint_binding('${DENSE_BINDING}',payload)"
sha256sum ${DENSE_BINDING} | awk '{print $1}' > ${DENSE_BINDING}.sha256

export RUN_ROOT=${VERIFY}/parallel_precheck
export DUCA_EXPECTED_COMMIT=${COMMIT}
export DUCA_R0_CHECKPOINT=${R0_CHECKPOINT}
export DUCA_R0_CHECKPOINT_EPOCH=131
export DUCA_R5_DENSE_CONFIG=${DENSE_CONFIG}
export DUCA_R5_DENSE_CHECKPOINT=${DENSE_CHECKPOINT}
export DUCA_R5_DENSE_CHECKPOINT_EVIDENCE=${DENSE_BINDING}
export DUCA_R5_DENSE_TRAINED_COMMIT=${DENSE_COMMIT}
export PRECHECK_ONLY=1
bash scripts/submit_duca_r0_r5_parallel_bundles.sh

[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
sha256sum ${RUN_ROOT}/deployment_manifest.json > ${VERIFY}/deployment_manifest.sha256
echo "VERIFY_OK commit=${COMMIT} snapshot=${SNAPSHOT} verify=${VERIFY}"
