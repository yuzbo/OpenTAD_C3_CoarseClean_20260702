#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_BURST_SUBMIT][FAIL] $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
R0_CHECKPOINT="${DUCA_R0_CHECKPOINT:-}"
R0_CHECKPOINT_EPOCH="${DUCA_R0_CHECKPOINT_EPOCH:-131}"
[[ -n "${RUN_ROOT}" ]] || fail "RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${R0_CHECKPOINT}" ]] || fail "DUCA_R0_CHECKPOINT is required"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/submission"
[[ ! -f "${RUN_ROOT}/jobs.tsv" ]] || { cat "${RUN_ROOT}/jobs.tsv"; exit 0; }

SPLIT_ROOT="${RUN_ROOT}/frontend_split"
"${PYTHON}" tools/bata/create_duca_frontend_split.py \
  --annotation "${THUMOS14_ANNOTATION_PATH}" --output-dir "${SPLIT_ROOT}" \
  --seed 3407 --holdout-fraction 0.20 > "${RUN_ROOT}/frontend_split.out"
SPLIT_MANIFEST="${SPLIT_ROOT}/frontend_split_manifest.json"
SPLIT_SHA256="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"

write_header() {
  local path="$1" name="$2" time="$3" gpu="$4"
  cat >"${path}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=${time}
#SBATCH --output=${RUN_ROOT}/logs/${name}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${name}-%j.err
EOF
  [[ "${gpu}" == 1 ]] && echo '#SBATCH --gpus=1' >>"${path}"
  cat >>"${path}" <<EOF
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}' DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}' RUN_ROOT='${RUN_ROOT}'
EOF
}

P0_SBATCH="${RUN_ROOT}/submission/p0.sbatch"
write_header "${P0_SBATCH}" "burst_p0_${EXPECTED_COMMIT:0:7}" "2-00:00:00" 1
cat >>"${P0_SBATCH}" <<EOF
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
bash scripts/run_duca_boundary_burst_p0_gpu1.sh
EOF

R0_SBATCH="${RUN_ROOT}/submission/r0.sbatch"
write_header "${R0_SBATCH}" "burst_r0_${EXPECTED_COMMIT:0:7}" "2-00:00:00" 1
cat >>"${R0_SBATCH}" <<EOF
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST='${SPLIT_ROOT}/frontend_holdout_block_list.txt'
export DUCA_R0_CHECKPOINT='${R0_CHECKPOINT}'
export DUCA_R0_CHECKPOINT_EPOCH='${R0_CHECKPOINT_EPOCH}'
export DUCA_R0_OUTPUT_ROOT='${RUN_ROOT}/r0_holdout_map'
bash scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh
EOF

GATE_SBATCH="${RUN_ROOT}/submission/gate.sbatch"
write_header "${GATE_SBATCH}" "burst_gate_${EXPECTED_COMMIT:0:7}" "04:00:00" 1
cat >>"${GATE_SBATCH}" <<'EOF'
export DUCA_FRONTEND_DECISION_JSON="${RUN_ROOT}/frontend_decision.json"
export DUCA_FRONTEND_DECISION_SHA256="$(sha256sum "${DUCA_FRONTEND_DECISION_JSON}" | awk '{print $1}')"
export DUCA_BOUNDARY_BURST_GATE_ROOT="${RUN_ROOT}/full_model_gate"
bash scripts/run_duca_boundary_burst_gate_gpu1.sh
EOF

variants=(
  two_stage_exact_uniform
  gaussian_matched_g0
  boundary_burst_r2q3_g0
  boundary_burst_r4q5_g0
)
for variant in "${variants[@]}"; do
  file="${RUN_ROOT}/submission/${variant}.sbatch"
  write_header "${file}" "${variant}_${EXPECTED_COMMIT:0:7}" "3-00:00:00" 1
  cat >>"${file}" <<EOF
export DUCA_SELECTED_OPT_VARIANT='${variant}'
export DUCA_FRONTEND_DECISION_JSON="\${RUN_ROOT}/frontend_decision.json"
export DUCA_FRONTEND_DECISION_SHA256="\$(sha256sum "\${DUCA_FRONTEND_DECISION_JSON}" | awk '{print \$1}')"
export DUCA_SELECTED_OPT_GATE_SUITE="\${RUN_ROOT}/full_model_gate/gate_suite.json"
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="\$(sha256sum "\${DUCA_SELECTED_OPT_GATE_SUITE}" | awk '{print \$1}')"
export RUN_DIR="\${RUN_ROOT}/official60/${variant}/run"
export WORK_DIR="\${RUN_ROOT}/official60/${variant}/work"
bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
EOF
done

AGG_SBATCH="${RUN_ROOT}/submission/aggregate.sbatch"
write_header "${AGG_SBATCH}" "burst_agg_${EXPECTED_COMMIT:0:7}" "01:00:00" 0
cat >>"${AGG_SBATCH}" <<'EOF'
decision="${RUN_ROOT}/frontend_decision.json"
gate="${RUN_ROOT}/full_model_gate/gate_suite.json"
python -m tools.bata.aggregate_duca_boundary_burst_results \
  --expected-commit "${DUCA_EXPECTED_COMMIT}" \
  --decision "${decision}" --decision-sha256 "$(sha256sum "${decision}" | awk '{print $1}')" \
  --gate "${gate}" --gate-sha256 "$(sha256sum "${gate}" | awk '{print $1}')" \
  --completion "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.json" \
  --completion "${RUN_ROOT}/official60/gaussian_matched_g0/run/completion.json" \
  --completion "${RUN_ROOT}/official60/boundary_burst_r2q3_g0/run/completion.json" \
  --completion "${RUN_ROOT}/official60/boundary_burst_r4q5_g0/run/completion.json" \
  --output-json "${RUN_ROOT}/final_suite_results.json"
EOF

for file in "${P0_SBATCH}" "${R0_SBATCH}" "${GATE_SBATCH}" "${AGG_SBATCH}" "${RUN_ROOT}"/submission/*.sbatch; do
  bash -n "${file}"
done
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_BURST_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

r0="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${R0_SBATCH}")"; r0="${r0%%;*}"
p0="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${P0_SBATCH}")"; p0="${p0%%;*}"
gate="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --dependency="afterok:${p0}" "${GATE_SBATCH}")"; gate="${gate%%;*}"
train_ids=()
for variant in "${variants[@]}"; do
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --dependency="afterok:${gate}" "${RUN_ROOT}/submission/${variant}.sbatch")"
  train_ids+=("${raw%%;*}")
done
dependency="$(IFS=:; echo "${train_ids[*]}")"
aggregate="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --dependency="afterok:${dependency}" "${AGG_SBATCH}")"; aggregate="${aggregate%%;*}"
{
  printf 'key\tjob_id\tdependency\n'
  printf 'r0_holdout_map\t%s\tnone\n' "${r0}"
  printf 'p0\t%s\tnone\n' "${p0}"
  printf 'gate\t%s\tafterok:%s\n' "${gate}" "${p0}"
  for idx in "${!variants[@]}"; do printf '%s\t%s\tafterok:%s\n' "${variants[$idx]}" "${train_ids[$idx]}" "${gate}"; done
  printf 'aggregate\t%s\tafterok:%s\n' "${aggregate}" "${dependency}"
} > "${RUN_ROOT}/jobs.tsv"
cat "${RUN_ROOT}/jobs.tsv"
