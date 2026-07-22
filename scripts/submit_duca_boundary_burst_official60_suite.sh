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
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/submission"
JOBS_TSV="${RUN_ROOT}/jobs.tsv"
JOBS_COMPLETE="${RUN_ROOT}/jobs.complete.json"
journal() {
  "${PYTHON}" -m tools.bata.duca_boundary_burst_submission_journal \
    --journal "${JOBS_TSV}" --seal "${JOBS_COMPLETE}" \
    --expected-commit "${EXPECTED_COMMIT}" --target-cluster "${TARGET_CLUSTER}" "$@"
}
if [[ -e "${JOBS_TSV}" || -e "${JOBS_COMPLETE}" ]]; then
  journal inspect >/dev/null \
    || fail "partial or invalid submission journal exists; reconcile Slurm manually"
  cat "${JOBS_TSV}"
  exit 0
fi
[[ -f "${R0_CHECKPOINT}" ]] || fail "DUCA_R0_CHECKPOINT is required"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain is missing"

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
export DUCA_TARGET_CLUSTER='${TARGET_CLUSTER}'
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
  "${RUN_ROOT}/full_model_gate/gate_suite.sha256.tmp"
mv -f "${RUN_ROOT}/full_model_gate/gate_suite.sha256.tmp" \
  "${RUN_ROOT}/full_model_gate/gate_suite.sha256"
EOF

main_worker_roles=(
  two_stage_exact_uniform
  r0_selected_boundary_burst_g0
)
for worker_role in "${main_worker_roles[@]}"; do
  file="${RUN_ROOT}/submission/${worker_role}.sbatch"
  write_header "${file}" "${worker_role}_${EXPECTED_COMMIT:0:7}" "3-00:00:00" 1
  cat >>"${file}" <<EOF
set +e
(
set -euo pipefail
export DUCA_FRONTEND_DECISION_JSON="\${RUN_ROOT}/frontend_decision.json"
IFS= read -r DUCA_FRONTEND_DECISION_SHA256 < "\${RUN_ROOT}/frontend_decision.sha256"
[[ "\${DUCA_FRONTEND_DECISION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 1
export DUCA_FRONTEND_DECISION_SHA256
export DUCA_SELECTED_OPT_GATE_SUITE="\${RUN_ROOT}/full_model_gate/gate_suite.json"
IFS= read -r DUCA_SELECTED_OPT_GATE_SUITE_SHA256 < "\${RUN_ROOT}/full_model_gate/gate_suite.sha256"
[[ "\${DUCA_SELECTED_OPT_GATE_SUITE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || exit 1
export DUCA_SELECTED_OPT_GATE_SUITE_SHA256
resolved_variant="\$("\${PYTHON}" - \
  "\${DUCA_FRONTEND_DECISION_JSON}" "\${DUCA_FRONTEND_DECISION_SHA256}" \
  "\${DUCA_SELECTED_OPT_GATE_SUITE}" "\${DUCA_SELECTED_OPT_GATE_SUITE_SHA256}" \
  "\${DUCA_EXPECTED_COMMIT}" '${worker_role}' <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    UNIFORM_OFFICIAL_VARIANT,
    validate_frontend_decision,
    validate_full_model_gate,
)

decision = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
validate_full_model_gate(
    gate_path=sys.argv[3],
    gate_sha256=sys.argv[4],
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
worker_role = sys.argv[6]
if worker_role == UNIFORM_OFFICIAL_VARIANT:
    variant = UNIFORM_OFFICIAL_VARIANT
elif worker_role == "r0_selected_boundary_burst_g0":
    variant = decision["family_routing"]["selected_official60_variant"]
else:
    raise SystemExit(f"unsupported main worker role: {worker_role}")
required = decision["family_routing"]["required_official60_variants"]
if variant not in required:
    raise SystemExit(f"refusing to run non-main official60 variant: {variant}")
if variant not in (
    UNIFORM_OFFICIAL_VARIANT,
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
):
    raise SystemExit(f"invalid routed official60 variant: {variant}")
print(variant)
PY
)"
export DUCA_SELECTED_OPT_VARIANT="\${resolved_variant}"
export RUN_DIR="\${RUN_ROOT}/official60/\${resolved_variant}/run"
export WORK_DIR="\${RUN_ROOT}/official60/\${resolved_variant}/work"
bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
sha256sum "\${RUN_DIR}/completion.json" | awk '{print \$1}' > \
  "\${RUN_DIR}/completion.sha256.tmp"
mv -f "\${RUN_DIR}/completion.sha256.tmp" "\${RUN_DIR}/completion.sha256"
)
role_status=\$?
if [[ \${role_status} -ne 0 ]]; then
  "\${PYTHON}" - "\${RUN_ROOT}/official60_worker_failures/${worker_role}.json" \
    "\${DUCA_EXPECTED_COMMIT}" '${worker_role}' "\${role_status}" <<'PY'
import sys
from pathlib import Path

from tools.bata.select_duca_boundary_burst_candidates import _atomic_write_json

_atomic_write_json(
    Path(sys.argv[1]).resolve(),
    {
        "schema": "duca_boundary_burst_official_role_failure_v1",
        "ok": False,
        "fail_closed": True,
        "git_commit": sys.argv[2],
        "worker_role": sys.argv[3],
        "worker_exit_code": int(sys.argv[4]),
        "status": "worker_failed_aggregate_will_adjudicate_required_main",
    },
    require_absent=False,
)
PY
fi
exit 0
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
selected_variant="$("${PYTHON}" - "${decision}" "${decision_sha256}" \
  "${gate}" "${gate_sha256}" "${DUCA_EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import (
    validate_frontend_decision,
    validate_full_model_gate,
)

decision = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
validate_full_model_gate(
    gate_path=sys.argv[3],
    gate_sha256=sys.argv[4],
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[5],
)
print(decision["family_routing"]["selected_official60_variant"])
PY
)"
[[ "${selected_variant}" == boundary_burst_r2q3_g0 \
  || "${selected_variant}" == boundary_burst_r4q5_g0 ]] || exit 1
"${PYTHON}" -m tools.bata.aggregate_duca_boundary_burst_results \
  --expected-commit "${DUCA_EXPECTED_COMMIT}" \
  --decision "${decision}" --decision-sha256 "${decision_sha256}" \
  --gate "${gate}" --gate-sha256 "${gate_sha256}" \
  --completion "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/two_stage_exact_uniform/run/completion.sha256")" \
  --completion "${RUN_ROOT}/official60/${selected_variant}/run/completion.json" \
  --completion-sha256 "$(read_seal "${RUN_ROOT}/official60/${selected_variant}/run/completion.sha256")" \
  --output-json "${RUN_ROOT}/final_suite_results.json"
EOF

for file in "${P0_SBATCH}" "${R0_SBATCH}" "${GATE_SBATCH}" "${AGG_SBATCH}" "${RUN_ROOT}"/submission/*.sbatch; do
  bash -n "${file}"
done
"${PYTHON}" - "${RUN_ROOT}/submission_manifest.json" \
  "${RUN_ROOT}/submission_manifest.sha256" "${EXPECTED_COMMIT}" \
  "${SPLIT_MANIFEST}" "${SPLIT_SHA256}" \
  "${SPLIT_ANNOTATION}" "${SPLIT_ANNOTATION_SHA256}" \
  "${SPLIT_TRAIN_BLOCK_LIST}" "${SPLIT_TRAIN_BLOCK_LIST_SHA256}" \
  "${SPLIT_HOLDOUT_BLOCK_LIST}" "${SPLIT_HOLDOUT_BLOCK_LIST_SHA256}" \
  "${R0_CHECKPOINT}" "${R0_CHECKPOINT_SHA256}" \
  "${ADATAD_PRETRAIN_PATH}" "${ADATAD_PRETRAIN_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from tools.bata.duca_p0_evaluation import canonical_sha256
from tools.bata.select_duca_boundary_burst_candidates import (
    R0_PROJECTED_FAMILY_ROUTES,
    _atomic_write_json,
)

(
    out,
    seal_out,
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
run_root = Path(out).resolve().parent
for path, expected, label in (
    (checkpoint_path, checkpoint_sha, "R0 checkpoint"),
    (pretrain_path, pretrain_sha, "AdaTAD pretrain"),
):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"{label} drifted while sealing submission")
sbatch_roles = {
    "r0_holdout_map": "r0.sbatch",
    "p0": "p0.sbatch",
    "gate": "gate.sbatch",
    "two_stage_exact_uniform": "two_stage_exact_uniform.sbatch",
    "r0_selected_boundary_burst_g0": "r0_selected_boundary_burst_g0.sbatch",
    "aggregate": "aggregate.sbatch",
}
generated_sbatch_artifacts = {}
for role, filename in sbatch_roles.items():
    path = (run_root / "submission" / filename).resolve()
    if not path.is_file():
        raise SystemExit(f"generated sbatch is missing for {role}: {path}")
    generated_sbatch_artifacts[role] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
payload = {
    "schema": "duca_boundary_burst_submission_v2",
    "ok": True,
    "fail_closed": True,
    "git_commit": commit,
    "run_root": str(run_root),
    "generated_sbatch_artifacts": generated_sbatch_artifacts,
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
    "r0_summary_contract": {
        "path": str(Path(out).parent / "r0_holdout_map" / "r0_summary.json"),
        "sha256_seal_path": str(
            Path(out).parent / "r0_holdout_map" / "r0_summary.sha256"
        ),
        "git_commit": commit,
        "consumer": "family_routing_manifest_v1",
    },
    "projected_family_mappings": {
        family: {
            "p0_variant": route["p0_variant"],
            "official60_variant": route["official60_variant"],
        }
        for family, route in R0_PROJECTED_FAMILY_ROUTES.items()
    },
    "dependency_contract": {
        "r0_holdout_map": "none",
        "p0": "afterok:r0_holdout_map",
        "gate": "afterok:p0",
        "two_stage_exact_uniform": "afterok:gate",
        "r0_selected_boundary_burst_g0": "afterok:gate",
        "aggregate": "afterok:matched_u_plus_r0_selected_g0",
    },
    "runtime_selected_g0_worker": {
        "role": "r0_selected_boundary_burst_g0",
        "routing_source": str(run_root / "frontend_decision.json"),
        "routing_source_sha256_seal": str(
            run_root / "frontend_decision.sha256"
        ),
        "full_model_gate": str(run_root / "full_model_gate" / "gate_suite.json"),
        "allowed_variants": [
            "boundary_burst_r2q3_g0",
            "boundary_burst_r4q5_g0",
        ],
        "runner": "scripts/run_duca_two_stage_curriculum_variant_gpu1.sh",
    },
    "required_p0_policy": "r0_selected_projected_family_only",
    "required_official60_policy": [
        "two_stage_exact_uniform",
        "r0_selected_boundary_burst_g0",
    ],
    "diagnostic_official60_roles": [
        "gaussian_matched_g0",
        "r0_unselected_boundary_burst_g0",
    ],
    "diagnostic_submission_policy": "not_submitted_by_main_dag",
    "diagnostic_failures_block_main": False,
    "official60_job_envelope": (
        "worker failures become atomic role_failure evidence; aggregate requires "
        "sealed U plus the R0-selected G0 completion"
    ),
    "required_main_failure_blocks_final_aggregate": True,
    "aggregate_inputs": "matched_u_plus_r0_selected_g0_only",
    "r0_positive_headroom_required": True,
    "uniform_frontend_checkpoint": None,
    "learned_frontend_checkpoint_source": "variant_matched_p0_winner",
}
payload["manifest_sha256"] = canonical_sha256(payload)
output = Path(out).resolve()
_atomic_write_json(output, payload, require_absent=True)
seal = Path(seal_out).resolve()
temporary = seal.with_name(f".{seal.name}.{uuid4().hex}.tmp")
try:
    with temporary.open("xb") as handle:
        handle.write((hashlib.sha256(output.read_bytes()).hexdigest() + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, seal)
finally:
    temporary.unlink(missing_ok=True)
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_BURST_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

journal initialize

SUBMITTED_JOB_ID=""
submit_and_record() {
  local role="$1" dependency="$2" sbatch_file="$3"
  local raw job_id
  local sbatch_args=(--parsable --clusters="${TARGET_CLUSTER}")
  if [[ "${dependency}" != "none" ]]; then
    sbatch_args+=(--dependency="${dependency}")
  fi
  journal reserve --role "${role}" --dependency "${dependency}"
  if ! raw="$(sbatch "${sbatch_args[@]}" "${sbatch_file}")"; then
    fail "sbatch failed for ${role}; partial journal retained for manual reconciliation"
  fi
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] \
    || fail "invalid sbatch response for ${role}; partial journal retained"
  journal record --role "${role}" --job-id "${job_id}" \
    --dependency "${dependency}"
  SUBMITTED_JOB_ID="${job_id}"
}

submit_and_record "r0_holdout_map" "none" "${R0_SBATCH}"
r0="${SUBMITTED_JOB_ID}"
p0_dependency="afterok:${r0}"
submit_and_record "p0" "${p0_dependency}" "${P0_SBATCH}"
p0="${SUBMITTED_JOB_ID}"
gate_dependency="afterok:${p0}"
submit_and_record "gate" "${gate_dependency}" "${GATE_SBATCH}"
gate="${SUBMITTED_JOB_ID}"
main_dependency="afterok:${gate}"
submit_and_record "two_stage_exact_uniform" "${main_dependency}" \
  "${RUN_ROOT}/submission/two_stage_exact_uniform.sbatch"
uniform="${SUBMITTED_JOB_ID}"
submit_and_record "r0_selected_boundary_burst_g0" "${main_dependency}" \
  "${RUN_ROOT}/submission/r0_selected_boundary_burst_g0.sbatch"
selected_g0="${SUBMITTED_JOB_ID}"
aggregate_dependency="afterok:${uniform}:${selected_g0}"
submit_and_record "aggregate" "${aggregate_dependency}" "${AGG_SBATCH}"
journal seal
journal inspect >/dev/null
cat "${JOBS_TSV}"
