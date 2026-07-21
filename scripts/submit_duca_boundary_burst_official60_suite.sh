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
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/submission"
[[ ! -f "${RUN_ROOT}/jobs.tsv" ]] || { cat "${RUN_ROOT}/jobs.tsv"; exit 0; }

SPLIT_ROOT="${RUN_ROOT}/frontend_split"
"${PYTHON}" -m tools.bata.create_duca_frontend_split \
  --annotation "${THUMOS14_ANNOTATION_PATH}" --output-dir "${SPLIT_ROOT}" \
  --seed 3407 --holdout-fraction 0.20 > "${RUN_ROOT}/frontend_split.out"
SPLIT_MANIFEST="${SPLIT_ROOT}/frontend_split_manifest.json"
SPLIT_SHA256="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.create_duca_frontend_split \
  --validate-manifest "${SPLIT_MANIFEST}" \
  --expected-manifest-sha256 "${SPLIT_SHA256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --train-block-list "${SPLIT_ROOT}/frontend_train_block_list.txt" \
  --holdout-block-list "${SPLIT_ROOT}/frontend_holdout_block_list.txt" \
  > "${RUN_ROOT}/frontend_split.validation.json"
SPLIT_ANNOTATION="${THUMOS14_ANNOTATION_PATH}"
SPLIT_TRAIN_BLOCK_LIST="${SPLIT_ROOT}/frontend_train_block_list.txt"
SPLIT_HOLDOUT_BLOCK_LIST="${SPLIT_ROOT}/frontend_holdout_block_list.txt"
SPLIT_ANNOTATION_SHA256="$(sha256sum "${SPLIT_ANNOTATION}" | awk '{print $1}')"
SPLIT_TRAIN_BLOCK_LIST_SHA256="$(sha256sum "${SPLIT_TRAIN_BLOCK_LIST}" | awk '{print $1}')"
SPLIT_HOLDOUT_BLOCK_LIST_SHA256="$(sha256sum "${SPLIT_HOLDOUT_BLOCK_LIST}" | awk '{print $1}')"
R0_CHECKPOINT_SHA256="$(sha256sum "${R0_CHECKPOINT}" | awk '{print $1}')"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"

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
export DUCA_ADATAD_PRETRAIN_PATH='${ADATAD_PRETRAIN_PATH}'
export DUCA_ADATAD_PRETRAIN_SHA256='${ADATAD_PRETRAIN_SHA256}'
EOF
}

P0_SBATCH="${RUN_ROOT}/submission/p0.sbatch"
write_header "${P0_SBATCH}" "burst_p0_${EXPECTED_COMMIT:0:7}" "2-00:00:00" 1
cat >>"${P0_SBATCH}" <<EOF
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
export DUCA_R0_SUMMARY_JSON='${RUN_ROOT}/r0_holdout_map/r0_summary.json'
export DUCA_R0_SUMMARY_SHA256_FILE='${RUN_ROOT}/r0_holdout_map/r0_summary.sha256'
bash scripts/run_duca_boundary_burst_p0_gpu1.sh
EOF

R0_SBATCH="${RUN_ROOT}/submission/r0.sbatch"
write_header "${R0_SBATCH}" "burst_r0_${EXPECTED_COMMIT:0:7}" "2-00:00:00" 1
cat >>"${R0_SBATCH}" <<EOF
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
export DUCA_SPLIT_ANNOTATION_PATH='${SPLIT_ANNOTATION}'
export DUCA_SPLIT_ANNOTATION_SHA256='${SPLIT_ANNOTATION_SHA256}'
export DUCA_FRONTEND_TRAIN_BLOCK_LIST='${SPLIT_TRAIN_BLOCK_LIST}'
export DUCA_FRONTEND_TRAIN_BLOCK_LIST_SHA256='${SPLIT_TRAIN_BLOCK_LIST_SHA256}'
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST='${SPLIT_HOLDOUT_BLOCK_LIST}'
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST_SHA256='${SPLIT_HOLDOUT_BLOCK_LIST_SHA256}'
export DUCA_R0_CHECKPOINT='${R0_CHECKPOINT}'
export DUCA_R0_CHECKPOINT_SHA256='${R0_CHECKPOINT_SHA256}'
export DUCA_R0_CHECKPOINT_EPOCH='${R0_CHECKPOINT_EPOCH}'
export DUCA_ADATAD_PRETRAIN_PATH='${ADATAD_PRETRAIN_PATH}'
export DUCA_ADATAD_PRETRAIN_SHA256='${ADATAD_PRETRAIN_SHA256}'
export DUCA_R0_OUTPUT_ROOT='${RUN_ROOT}/r0_holdout_map'
bash scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh
EOF

GATE_SBATCH="${RUN_ROOT}/submission/gate.sbatch"
write_header "${GATE_SBATCH}" "burst_gate_${EXPECTED_COMMIT:0:7}" "04:00:00" 1
cat >>"${GATE_SBATCH}" <<'EOF'
export DUCA_FRONTEND_DECISION_JSON="${RUN_ROOT}/frontend_decision.json"
IFS= read -r DUCA_FRONTEND_DECISION_SHA256 < "${RUN_ROOT}/frontend_decision.sha256"
[[ "${DUCA_FRONTEND_DECISION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 1
export DUCA_FRONTEND_DECISION_SHA256
export DUCA_BOUNDARY_BURST_GATE_ROOT="${RUN_ROOT}/full_model_gate"
bash scripts/run_duca_boundary_burst_gate_gpu1.sh
sha256sum "${RUN_ROOT}/full_model_gate/gate_suite.json" | awk '{print $1}' > \
  "${RUN_ROOT}/full_model_gate/gate_suite.sha256"
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
IFS= read -r DUCA_FRONTEND_DECISION_SHA256 < "\${RUN_ROOT}/frontend_decision.sha256"
[[ "\${DUCA_FRONTEND_DECISION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 1
export DUCA_FRONTEND_DECISION_SHA256
export DUCA_SELECTED_OPT_GATE_SUITE="\${RUN_ROOT}/full_model_gate/gate_suite.json"
IFS= read -r DUCA_SELECTED_OPT_GATE_SUITE_SHA256 < "\${RUN_ROOT}/full_model_gate/gate_suite.sha256"
[[ "\${DUCA_SELECTED_OPT_GATE_SUITE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 1
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256
export RUN_DIR="\${RUN_ROOT}/official60/${variant}/run"
export WORK_DIR="\${RUN_ROOT}/official60/${variant}/work"
bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
sha256sum "\${RUN_DIR}/completion.json" | awk '{print \$1}' > \
  "\${RUN_DIR}/completion.sha256"
EOF
done

AGG_SBATCH="${RUN_ROOT}/submission/aggregate.sbatch"
write_header "${AGG_SBATCH}" "burst_agg_${EXPECTED_COMMIT:0:7}" "01:00:00" 0
cat >>"${AGG_SBATCH}" <<'EOF'
decision="${RUN_ROOT}/frontend_decision.json"
gate="${RUN_ROOT}/full_model_gate/gate_suite.json"
read_seal() {
  local seal="$1" value
  [[ -f "${seal}" ]] || return 1
  IFS= read -r value < "${seal}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s' "${value}"
}
decision_sha256="$(read_seal "${RUN_ROOT}/frontend_decision.sha256")"
gate_sha256="$(read_seal "${RUN_ROOT}/full_model_gate/gate_suite.sha256")"
python -m tools.bata.aggregate_duca_boundary_burst_results \
  --expected-commit "${DUCA_EXPECTED_COMMIT}" \
  --decision "${decision}" --decision-sha256 "${decision_sha256}" \
  --gate "${gate}" --gate-sha256 "${gate_sha256}" \
  --completion "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.sha256")" \
  --completion "${RUN_ROOT}/official60/gaussian_matched_g0/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/gaussian_matched_g0/run/completion.sha256")" \
  --completion "${RUN_ROOT}/official60/boundary_burst_r2q3_g0/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/boundary_burst_r2q3_g0/run/completion.sha256")" \
  --completion "${RUN_ROOT}/official60/boundary_burst_r4q5_g0/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/boundary_burst_r4q5_g0/run/completion.sha256")" \
  --output-json "${RUN_ROOT}/final_suite_results.json"
EOF

for file in "${P0_SBATCH}" "${R0_SBATCH}" "${GATE_SBATCH}" "${AGG_SBATCH}" "${RUN_ROOT}"/submission/*.sbatch; do
  bash -n "${file}"
done
"${PYTHON}" - "${RUN_ROOT}/submission_manifest.json" "${EXPECTED_COMMIT}" \
  "${SPLIT_MANIFEST}" "${SPLIT_SHA256}" \
  "${SPLIT_ANNOTATION}" "${SPLIT_ANNOTATION_SHA256}" \
  "${SPLIT_TRAIN_BLOCK_LIST}" "${SPLIT_TRAIN_BLOCK_LIST_SHA256}" \
  "${SPLIT_HOLDOUT_BLOCK_LIST}" "${SPLIT_HOLDOUT_BLOCK_LIST_SHA256}" \
  "${R0_CHECKPOINT}" "${R0_CHECKPOINT_SHA256}" \
  "${ADATAD_PRETRAIN_PATH}" "${ADATAD_PRETRAIN_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    out,
    commit,
    split,
    split_sha,
    annotation,
    annotation_sha,
    train_block,
    train_block_sha,
    holdout_block,
    holdout_block_sha,
    checkpoint,
    checkpoint_sha,
    pretrain,
    pretrain_sha,
) = sys.argv[1:]
checkpoint_path = Path(checkpoint).expanduser().resolve()
pretrain_path = Path(pretrain).expanduser().resolve()
for path, expected, label in (
    (checkpoint_path, checkpoint_sha, "R0 checkpoint"),
    (pretrain_path, pretrain_sha, "AdaTAD pretrain"),
):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{label} drifted while sealing submission")
payload = {
    "schema": "duca_boundary_burst_submission_v1",
    "ok": True,
    "git_commit": commit,
    "split_manifest_path": str(Path(split).resolve()),
    "split_manifest_sha256": split_sha,
    "split_reference_bindings": {
        "annotation_path": str(Path(annotation).resolve()),
        "annotation_sha256": annotation_sha,
        "train_block_list": str(Path(train_block).resolve()),
        "train_block_list_sha256": train_block_sha,
        "holdout_block_list": str(Path(holdout_block).resolve()),
        "holdout_block_list_sha256": holdout_block_sha,
    },
    "r0_checkpoint_path": str(checkpoint_path),
    "r0_checkpoint_sha256": checkpoint_sha,
    "adatad_pretrain_path": str(pretrain_path),
    "adatad_pretrain_sha256": pretrain_sha,
    "dependency_contract": {
        "r0_holdout_map": "none",
        "p0": "afterok:r0_holdout_map",
        "gate": "afterok:p0",
        "official60_arms": "afterok:gate",
        "aggregate": "afterok:all_official60_arms",
    },
    "r0_positive_headroom_required": True,
    "uniform_frontend_checkpoint": None,
    "learned_frontend_checkpoint_source": "variant_matched_p0_winner",
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_BURST_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

r0="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${R0_SBATCH}")"; r0="${r0%%;*}"
p0="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --dependency="afterok:${r0}" "${P0_SBATCH}")"; p0="${p0%%;*}"
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
  printf 'p0\t%s\tafterok:%s\n' "${p0}" "${r0}"
  printf 'gate\t%s\tafterok:%s\n' "${gate}" "${p0}"
  for idx in "${!variants[@]}"; do printf '%s\t%s\tafterok:%s\n' "${variants[$idx]}" "${train_ids[$idx]}" "${gate}"; done
  printf 'aggregate\t%s\tafterok:%s\n' "${aggregate}" "${dependency}"
} > "${RUN_ROOT}/jobs.tsv"
cat "${RUN_ROOT}/jobs.tsv"
