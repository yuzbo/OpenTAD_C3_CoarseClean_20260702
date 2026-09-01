#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=5426dff5f3fae03c74b3baff9d15d38527a47f11
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_5426dff_20260722
VERIFY=${BASE}/projects/c3_lowres_action_probe/duca_boundary_5426dff_verify_20260722

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

python -m py_compile \
  tools/train.py tools/test.py \
  tools/bata/duca_selected_axis_training.py \
  tools/bata/duca_r5_paper_matrix.py \
  tools/bata/aggregate_duca_r5_paper_matrix.py \
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
  tests/test_duca_boundary_burst_submission_journal.py -q

python -m pytest \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

mkdir -p ${VERIFY}
python -m tools.bata.duca_r5_paper_matrix \
  --repo-root ${SNAPSHOT} \
  --output-dir ${VERIFY}/r5_generated \
  --uniform-config ${SNAPSHOT}/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py \
  --learned-config ${SNAPSHOT}/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py \
  --cluster n16r4 >/dev/null

bash -n scripts/launch_duca_r5_paper_matrix.sh
while IFS= read -r script; do bash -n "${script}"; done \
  < <(find ${VERIFY}/r5_generated/jobs -type f -name '*.sbatch' | sort)
[[ "$(find ${VERIFY}/r5_generated/configs -type f -name '*.py' | wc -l)" == 24 ]]
[[ "$(find ${VERIFY}/r5_generated/jobs -type f -name '*.sbatch' | wc -l)" == 30 ]]
[[ "$(sha256sum ${VERIFY}/r5_generated/matrix_summary.json | awk '{print $1}')" == \
   "$(cat ${VERIFY}/r5_generated/matrix_summary.json.sha256)" ]]

echo "VERIFY_OK commit=${COMMIT} snapshot=${SNAPSHOT} generated=${VERIFY}/r5_generated"
