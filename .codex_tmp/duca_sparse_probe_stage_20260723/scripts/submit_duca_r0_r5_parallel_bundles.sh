#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_R0_R5_PARALLEL_SUBMIT][FAIL] $*" >&2; exit 1; }

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
UNIFORM_CONFIG="${DUCA_R5_UNIFORM_CONFIG:-${REPO_ROOT}/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py}"
LEARNED_CONFIG="${DUCA_R5_LEARNED_CONFIG:-${REPO_ROOT}/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py}"
DENSE_CONFIG="${DUCA_R5_DENSE_CONFIG:-}"
DENSE_CHECKPOINT="${DUCA_R5_DENSE_CHECKPOINT:-}"
DENSE_EVIDENCE="${DUCA_R5_DENSE_CHECKPOINT_EVIDENCE:-}"
DENSE_TRAINED_COMMIT="${DUCA_R5_DENSE_TRAINED_COMMIT:-b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f}"
R5_BUDGETS="${DUCA_R5_BUDGETS:-384 320 256 192 128}"
R5_SEEDS="${DUCA_R5_SEEDS:-3407 5801 8123}"
read -r -a r5_budget_values <<<"${R5_BUDGETS}"
read -r -a r5_seed_values <<<"${R5_SEEDS}"
PREREGISTERED_FAMILY=R2Q3_privileged_boundary_burst

[[ -n "${RUN_ROOT}" && ! -e "${RUN_ROOT}" ]] || fail "fresh RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "clean final-HEAD tree is required"
[[ -f "${R0_CHECKPOINT}" ]] || fail "DUCA_R0_CHECKPOINT is required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"
[[ -f "${DENSE_CONFIG}" && -f "${DENSE_CHECKPOINT}" && -f "${DENSE_EVIDENCE}" ]] \
  || fail "dense config, checkpoint, and checkpoint evidence are required"
[[ -f "${UNIFORM_CONFIG}" && -f "${LEARNED_CONFIG}" ]] \
  || fail "R5 source configs are missing"
[[ -z "${DUCA_PREREGISTERED_PROJECTED_FAMILY:-}" \
  || "${DUCA_PREREGISTERED_PROJECTED_FAMILY}" == "${PREREGISTERED_FAMILY}" ]] \
  || fail "this deployment preregisters only R2Q3"
[[ "$(basename "${LEARNED_CONFIG}")" == \
   duca_boundary_burst_g1_protected_fixed384_official60.py ]] \
  || fail "formal self-contained R5 currently binds the existing R2Q3 G1 config"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/bundles"

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

R0_CHECKPOINT_SHA256="$(sha256sum "${R0_CHECKPOINT}" | awk '{print $1}')"
ADATAD_PRETRAIN_SHA256="$(sha256sum "${ADATAD_PRETRAIN_PATH}" | awk '{print $1}')"
ANNOTATION_SHA256="$(sha256sum "${THUMOS14_ANNOTATION_PATH}" | awk '{print $1}')"
TRAIN_BLOCK="${SPLIT_ROOT}/frontend_train_block_list.txt"
HOLDOUT_BLOCK="${SPLIT_ROOT}/frontend_holdout_block_list.txt"
TRAIN_BLOCK_SHA256="$(sha256sum "${TRAIN_BLOCK}" | awk '{print $1}')"
HOLDOUT_BLOCK_SHA256="$(sha256sum "${HOLDOUT_BLOCK}" | awk '{print $1}')"
export DUCA_REPO_ROOT="${REPO_ROOT}"
export DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
export ADATAD_PRETRAIN_PATH
export ADATAD_PRETRAIN_SHA256

R5_ROOT="${RUN_ROOT}/r5"
"${PYTHON}" -m tools.bata.duca_r5_paper_matrix \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${R5_ROOT}" \
  --uniform-config "${UNIFORM_CONFIG}" \
  --learned-config "${LEARNED_CONFIG}" \
  --dense-config "${DENSE_CONFIG}" \
  --dense-checkpoint "${DENSE_CHECKPOINT}" \
  --dense-checkpoint-evidence "${DENSE_EVIDENCE}" \
  --dense-trained-commit "${DENSE_TRAINED_COMMIT}" \
  --cluster "${TARGET_CLUSTER}" \
  --budgets "${r5_budget_values[@]}" \
  --seeds "${r5_seed_values[@]}" \
  > "${RUN_ROOT}/r5_matrix_generation.out"

bundle_roles=(
  r0_r1
  r2_r3_core
  r2_r3_adapted
  shared_bootstrap
  r4_r2q3
  r5_all
)
declare -A bundle_gpus=(
  [r0_r1]=1
  [r2_r3_core]=2
  [r2_r3_adapted]=2
  [shared_bootstrap]=2
  [r4_r2q3]=2
  [r5_all]=4
)

write_bundle_sbatch() {
  local role="$1" gpus="${bundle_gpus[$1]}"
  local cpus_per_task=4
  [[ "${gpus}" == 1 ]] && cpus_per_task=8
  local file="${RUN_ROOT}/jobs/${role}.sbatch"
  cat > "${file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dr5_${role}_${EXPECTED_COMMIT:0:7}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=${gpus}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --gpus=${gpus}
#SBATCH --time=7-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${role}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${role}-%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_REPO_ROOT='${REPO_ROOT}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_PARALLEL_BUNDLE_ROLE='${role}'
export DUCA_PARALLEL_BUNDLE_ROOT='${RUN_ROOT}/bundles/${role}'
export DUCA_ADATAD_PRETRAIN_PATH='${ADATAD_PRETRAIN_PATH}'
export DUCA_ADATAD_PRETRAIN_SHA256='${ADATAD_PRETRAIN_SHA256}'
export DUCA_R5_MATRIX_ROOT='${R5_ROOT}'
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
export DUCA_SPLIT_ANNOTATION_PATH='${THUMOS14_ANNOTATION_PATH}'
export DUCA_SPLIT_ANNOTATION_SHA256='${ANNOTATION_SHA256}'
export DUCA_FRONTEND_TRAIN_BLOCK_LIST='${TRAIN_BLOCK}'
export DUCA_FRONTEND_TRAIN_BLOCK_LIST_SHA256='${TRAIN_BLOCK_SHA256}'
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST='${HOLDOUT_BLOCK}'
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST_SHA256='${HOLDOUT_BLOCK_SHA256}'
export DUCA_R0_CHECKPOINT='${R0_CHECKPOINT}'
export DUCA_R0_CHECKPOINT_SHA256='${R0_CHECKPOINT_SHA256}'
export DUCA_R0_CHECKPOINT_EPOCH='${R0_CHECKPOINT_EPOCH}'
export DUCA_R0_BOOTSTRAP_WORKERS=8
export DUCA_PREREGISTERED_PROJECTED_FAMILY='${PREREGISTERED_FAMILY}'
export DUCA_SHARED_BOOTSTRAP_RECEIPT='${RUN_ROOT}/bundles/shared_bootstrap/bootstrap_receipt.json'
bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh
EOF
  bash -n "${file}"
}

for role in "${bundle_roles[@]}"; do write_bundle_sbatch "${role}"; done

"${PYTHON}" - "${RUN_ROOT}/deployment_manifest.json" \
  "${EXPECTED_COMMIT}" "${RUN_ROOT}" "${R5_ROOT}" \
  "${UNIFORM_CONFIG}" "${LEARNED_CONFIG}" "${bundle_roles[@]}" <<'PY'
import hashlib
import sys
from pathlib import Path
from tools.bata.duca_selected_axis_training import atomic_write_json

output, commit, run_root, r5_root, uniform, learned, *roles = sys.argv[1:]
r5 = Path(r5_root).resolve()
rows = [line.split("\t") for line in (r5 / "cells.tsv").read_text(encoding="utf-8").splitlines()[1:]]
summary = __import__("json").loads((r5 / "matrix_summary.json").read_text(encoding="utf-8"))
expected_count = len(summary["backends"]) * len(summary["arms"]) * len(summary["budgets"]) * len(summary["seeds"])
if len(rows) != expected_count:
    raise SystemExit("parallel deployment R5 cell count differs from requested axes")
coverage = {}
for row in rows:
    coverage.setdefault(f"{row[1]}_{row[2]}", []).append(row[0])
expected_group = len(summary["budgets"]) * len(summary["seeds"])
if len(coverage) != 4 or set(map(len, coverage.values())) != {expected_group}:
    raise SystemExit("R5 backend-by-arm coverage differs from requested axes")
jobs = []
for role in roles:
    path = Path(run_root).resolve() / "jobs" / f"{role}.sbatch"
    dependency = "afterok:shared_bootstrap" if role in {"r4_r2q3", "r5_all"} else None
    jobs.append({
        "role": role,
        "dependency": dependency,
        "sbatch": str(path),
        "sbatch_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
atomic_write_json(Path(output).resolve(), {
    "schema": "duca_r0_r5_shared_bootstrap_parallel_deployment_v2",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "run_root": str(Path(run_root).resolve()),
    "external_read_only_inputs": [
        "R0 checkpoint", "AdaTAD pretrain", "THUMOS split",
        "sealed dense AdaTAD baseline checkpoint and evidence",
    ],
    "forbidden_external_trainable_artifacts": [
        "frontend_decision", "gate_suite", "terminal_u_g0_suite", "alignment"
    ],
    "dependency_null_bundle_count": 4,
    "shared_trainable_prerequisite_count": 1,
    "shared_bootstrap_consumers": ["r4_r2q3", "r5_all"],
    "duplicate_trainable_bootstrap_count": 0,
    "preregistered_projected_family": "R2Q3_privileged_boundary_burst",
    "family_routing_source": "preregistered_fixed_candidate",
    "r0_role": "detector_seen_training_internal_non_routing_diagnostic",
    "r4_r5_continuation_rule": (
        "official60_hard_adapted_r2q3_g0_average_mAP_strictly_greater_than_"
        "exact_uniform"
    ),
    "r5_cell_count": len(rows),
    "r5_budgets": summary["budgets"],
    "r5_seeds": summary["seeds"],
    "r5_coverage": coverage,
    "source_configs": {
        "uniform": str(Path(uniform).resolve()),
        "learned": str(Path(learned).resolve()),
    },
    "jobs": jobs,
})
PY

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_R0_R5_PARALLEL_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

printf 'role\tjob_id\tdependency\tgpus\tsbatch\n' > "${RUN_ROOT}/jobs.tsv"
declare -A job_ids=()
dependency_null_roles=(r0_r1 r2_r3_core r2_r3_adapted shared_bootstrap)
for role in "${dependency_null_roles[@]}"; do
  sbatch_file="${RUN_ROOT}/jobs/${role}.sbatch"
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" "${sbatch_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job id for ${role}"
  job_ids["${role}"]="${job_id}"
  printf '%s\t%s\tnone\t%s\t%s\n' "${role}" "${job_id}" \
    "${bundle_gpus[$role]}" "${sbatch_file}" >> "${RUN_ROOT}/jobs.tsv"
done

shared_dependency="afterok:${job_ids[shared_bootstrap]}"
for role in r4_r2q3 r5_all; do
  sbatch_file="${RUN_ROOT}/jobs/${role}.sbatch"
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="${shared_dependency}" "${sbatch_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job id for ${role}"
  job_ids["${role}"]="${job_id}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${role}" "${job_id}" \
    "${shared_dependency}" "${bundle_gpus[$role]}" "${sbatch_file}" \
    >> "${RUN_ROOT}/jobs.tsv"
done

aggregate_dependency="afterok:${job_ids[r5_all]}"
aggregate_sbatch="${R5_ROOT}/jobs/aggregate.sbatch"
aggregate_raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --gpus=1 \
  --dependency="${aggregate_dependency}" "${aggregate_sbatch}")"
aggregate_job="${aggregate_raw%%;*}"
[[ "${aggregate_job}" =~ ^[1-9][0-9]*$ ]] || fail "invalid aggregate job id"
printf 'aggregate\t%s\t%s\t1\t%s\n' "${aggregate_job}" \
  "${aggregate_dependency}" "${aggregate_sbatch}" >> "${RUN_ROOT}/jobs.tsv"
sha256sum "${RUN_ROOT}/jobs.tsv" | awk '{print $1}' > "${RUN_ROOT}/jobs.tsv.sha256"

echo "[DUCA_R0_R5_PARALLEL_SUBMIT] submitted one shared bootstrap, four independent jobs, two consumers and R5 aggregate"
cat "${RUN_ROOT}/jobs.tsv"
