#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_SUBMIT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${THUMOS14_ANNOTATION_PATH}" ]] || fail "THUMOS annotation is missing"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/p0" \
  "${RUN_ROOT}/official60"
SPLIT_ROOT="${RUN_ROOT}/frontend_split"
"${PYTHON}" tools/bata/create_duca_frontend_split.py \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --output-dir "${SPLIT_ROOT}" \
  --seed 3407 \
  --holdout-fraction 0.20 \
  > "${RUN_ROOT}/frontend_split.out"
SPLIT_MANIFEST="${SPLIT_ROOT}/frontend_split_manifest.json"
SPLIT_SHA256="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"
TRAIN_BLOCK="${SPLIT_ROOT}/frontend_train_block_list.txt"
HOLDOUT_BLOCK="${SPLIT_ROOT}/frontend_holdout_block_list.txt"
DECISION="${RUN_ROOT}/frontend_decision.json"
CANDIDATE_MANIFEST="${RUN_ROOT}/frontend_candidate_manifest.json"
GATE_ROOT="${RUN_ROOT}/two_stage_gate"
GATE_SUITE="${GATE_ROOT}/gate_suite.json"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"

frontend_variants=(
  lr_control_c25_a50_s100
  lr_coarse50_action100_scorer25
  lr_coarse100_action200_scorer50
)
for variant in "${frontend_variants[@]}"; do
  job_file="${RUN_ROOT}/jobs/frontend_${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2f_${variant}_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --output=${RUN_ROOT}/logs/frontend_${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/frontend_${variant}-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_FRONTEND_VARIANT='${variant}'
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
export DUCA_FRONTEND_TRAIN_BLOCK_LIST='${TRAIN_BLOCK}'
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST='${HOLDOUT_BLOCK}'
export RUN_DIR='${RUN_ROOT}/p0/${variant}/run'
export WORK_DIR='${RUN_ROOT}/p0/${variant}/work'
bash scripts/run_duca_frontend_pretrain_variant_gpu1.sh
EOF
done

AGGREGATE_JOB="${RUN_ROOT}/jobs/frontend_aggregate.sbatch"
cat > "${AGGREGATE_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2f_aggregate_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=${RUN_ROOT}/logs/frontend_aggregate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/frontend_aggregate-%j.err
source /etc/profile
set -euo pipefail
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export PYTHONNOUSERSITE=1
'${PYTHON}' tools/bata/aggregate_duca_frontend_candidates.py \
  --expected-commit '${EXPECTED_COMMIT}' \
  --split-manifest '${SPLIT_MANIFEST}' \
  --split-manifest-sha256 '${SPLIT_SHA256}' \
  --receipt '${RUN_ROOT}/p0/lr_control_c25_a50_s100/run/completion.json' \
  --receipt '${RUN_ROOT}/p0/lr_coarse50_action100_scorer25/run/completion.json' \
  --receipt '${RUN_ROOT}/p0/lr_coarse100_action200_scorer50/run/completion.json' \
  --candidate-manifest '${CANDIDATE_MANIFEST}' \
  --decision-json '${DECISION}'
EOF

GATE_JOB="${RUN_ROOT}/jobs/two_stage_gate.sbatch"
cat > "${GATE_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2_gate_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=${RUN_ROOT}/logs/two_stage_gate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/two_stage_gate-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_FRONTEND_DECISION_JSON='${DECISION}'
export DUCA_FRONTEND_DECISION_SHA256="\$(sha256sum '${DECISION}' | awk '{print \$1}')"
export DUCA_TWO_STAGE_GATE_ROOT='${GATE_ROOT}'
bash scripts/run_duca_two_stage_curriculum_gate_gpu1.sh
EOF

official_variants=(
  two_stage_exact_uniform
  two_stage_scratch
  two_stage_pretrained_joint
  two_stage_pretrained_frozen
)
for variant in "${official_variants[@]}"; do
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2_${variant#two_stage_}_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_SELECTED_OPT_VARIANT='${variant}'
export DUCA_FRONTEND_DECISION_JSON='${DECISION}'
export DUCA_FRONTEND_DECISION_SHA256="\$(sha256sum '${DECISION}' | awk '{print \$1}')"
export DUCA_SELECTED_OPT_GATE_SUITE='${GATE_SUITE}'
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="\$(sha256sum '${GATE_SUITE}' | awk '{print \$1}')"
export RUN_DIR='${RUN_ROOT}/official60/${variant}/run'
export WORK_DIR='${RUN_ROOT}/official60/${variant}/work'
bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
EOF
done

FINAL_JOB="${RUN_ROOT}/jobs/final_aggregate.sbatch"
cat > "${FINAL_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2_final_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=${RUN_ROOT}/logs/final_aggregate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/final_aggregate-%j.err
source /etc/profile
set -euo pipefail
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export PYTHONNOUSERSITE=1
'${PYTHON}' tools/bata/aggregate_duca_two_stage_results.py \
  --expected-commit '${EXPECTED_COMMIT}' \
  --frontend-decision '${DECISION}' \
  --frontend-decision-sha256 "\$(sha256sum '${DECISION}' | awk '{print \$1}')" \
  --gate-suite '${GATE_SUITE}' \
  --gate-suite-sha256 "\$(sha256sum '${GATE_SUITE}' | awk '{print \$1}')" \
  --completion '${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.json' \
  --completion '${RUN_ROOT}/official60/two_stage_scratch/run/completion.json' \
  --completion '${RUN_ROOT}/official60/two_stage_pretrained_joint/run/completion.json' \
  --completion '${RUN_ROOT}/official60/two_stage_pretrained_frozen/run/completion.json' \
  --output-json '${RUN_ROOT}/final_suite_results.json'
EOF

chmod 0755 "${RUN_ROOT}"/jobs/*.sbatch
for job_file in "${RUN_ROOT}"/jobs/*.sbatch; do
  bash -n "${job_file}"
done
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_TWO_STAGE_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v scontrol >/dev/null 2>&1 || fail "scontrol is unavailable"
submitted=()
transaction_committed=0
cleanup_new_jobs() {
  [[ "${transaction_committed}" == "0" ]] || return 0
  local job
  for job in "${submitted[@]}"; do
    scancel --clusters="${TARGET_CLUSTER}" "${job}" >/dev/null 2>&1 || true
  done
}
trap cleanup_new_jobs EXIT

LAST_JOB_ID=""
submit_held() {
  local job_file="$1"
  shift
  local raw job_id
  raw="$(sbatch --hold --parsable --clusters="${TARGET_CLUSTER}" "$@" "${job_file}")"
  raw="${raw%%$'\n'*}"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch response: ${raw}"
  submitted+=("${job_id}")
  LAST_JOB_ID="${job_id}"
}

frontend_ids=()
for variant in "${frontend_variants[@]}"; do
  submit_held "${RUN_ROOT}/jobs/frontend_${variant}.sbatch"
  frontend_ids+=("${LAST_JOB_ID}")
done
frontend_dependency="afterok:$(IFS=:; echo "${frontend_ids[*]}")"
submit_held "${AGGREGATE_JOB}" --dependency="${frontend_dependency}"
aggregate_id="${LAST_JOB_ID}"
submit_held "${GATE_JOB}" --dependency="afterok:${aggregate_id}"
gate_id="${LAST_JOB_ID}"
official_ids=()
for variant in "${official_variants[@]}"; do
  submit_held "${RUN_ROOT}/jobs/${variant}.sbatch" --dependency="afterok:${gate_id}"
  official_ids+=("${LAST_JOB_ID}")
done
official_dependency="afterok:$(IFS=:; echo "${official_ids[*]}")"
submit_held "${FINAL_JOB}" --dependency="${official_dependency}"
final_id="${LAST_JOB_ID}"

{
  printf 'key\tjob_id\tdependency\n'
  for index in "${!frontend_variants[@]}"; do
    printf 'frontend_%s\t%s\tnone\n' "${frontend_variants[$index]}" "${frontend_ids[$index]}"
  done
  printf 'frontend_aggregate\t%s\t%s\n' "${aggregate_id}" "${frontend_dependency}"
  printf 'two_stage_gate\t%s\tafterok:%s\n' "${gate_id}" "${aggregate_id}"
  for index in "${!official_variants[@]}"; do
    printf '%s\t%s\tafterok:%s\n' "${official_variants[$index]}" "${official_ids[$index]}" "${gate_id}"
  done
  printf 'final_aggregate\t%s\t%s\n' "${final_id}" "${official_dependency}"
} > "${RUN_ROOT}/jobs.tsv"

for job in "${submitted[@]}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${job}"
done
transaction_committed=1
trap - EXIT
echo "[DUCA_TWO_STAGE_SUBMIT] submitted ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
