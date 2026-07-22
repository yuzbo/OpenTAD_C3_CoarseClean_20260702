#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_R0_R5_PARALLEL_BUNDLE][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

ROLE="${DUCA_PARALLEL_BUNDLE_ROLE:-}"
BUNDLE_ROOT="${DUCA_PARALLEL_BUNDLE_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
FROZEN_PRETRAIN="${DUCA_ADATAD_PRETRAIN_PATH:-}"
FROZEN_PRETRAIN_SHA256="${DUCA_ADATAD_PRETRAIN_SHA256:-}"
PREREGISTERED_FAMILY=R2Q3_privileged_boundary_burst

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "a Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "clean tree required"
[[ -n "${BUNDLE_ROOT}" && ! -e "${BUNDLE_ROOT}" ]] \
  || fail "fresh DUCA_PARALLEL_BUNDLE_ROOT is required"
[[ -f "${FROZEN_PRETRAIN}" ]] || fail "AdaTAD pretrain is missing"
[[ "${FROZEN_PRETRAIN_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "AdaTAD pretrain SHA256 is required"
[[ -z "${DUCA_PREREGISTERED_PROJECTED_FAMILY:-}" \
  || "${DUCA_PREREGISTERED_PROJECTED_FAMILY}" == "${PREREGISTERED_FAMILY}" ]] \
  || fail "external family routing differs from preregistered R2Q3"
export DUCA_PREREGISTERED_PROJECTED_FAMILY="${PREREGISTERED_FAMILY}"
[[ "$(sha256sum "${FROZEN_PRETRAIN}" | awk '{print $1}')" == \
   "${FROZEN_PRETRAIN_SHA256}" ]] || fail "AdaTAD pretrain drift"
mkdir -p "${BUNDLE_ROOT}"

require_one_gpu() {
  [[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == 1 ]] \
    || fail "this bundle step requires exactly one Slurm-visible GPU"
}

seal_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "cannot seal missing artifact: ${path}"
  sha256sum "${path}" | awk '{print $1}' > "${path}.sha256.tmp"
  mv -f "${path}.sha256.tmp" "${path}.sha256"
}

read_seal() {
  local path="$1" value
  [[ -f "${path}" ]] || fail "missing SHA256 seal: ${path}"
  IFS= read -r value < "${path}"
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA256 seal: ${path}"
  printf '%s' "${value}"
}

write_completion() {
  local output="$1" role="$2" detail="$3"
  "${PYTHON}" - "${output}" "${EXPECTED_COMMIT}" "${role}" "${detail}" <<'PY'
import sys
from pathlib import Path
from tools.bata.duca_selected_axis_training import atomic_write_json

output, commit, role, detail = sys.argv[1:]
atomic_write_json(
    Path(output).resolve(),
    {
        "schema": "duca_r0_r5_parallel_bundle_completion_v2",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "bundle_role": role,
        "detail": detail,
    },
)
PY
}

run_r0() {
  local output_root="$1"
  export DUCA_R0_OUTPUT_ROOT="${output_root}"
  export DUCA_R0_BOOTSTRAP_WORKERS="${DUCA_R0_BOOTSTRAP_WORKERS:-8}"
  bash scripts/run_duca_boundary_burst_r0_holdout_map_gpu1.sh
}

run_curriculum_arm() {
  local variant="$1" root="$2"
  export DUCA_SELECTED_OPT_VARIANT="${variant}"
  export RUN_DIR="${root}/run"
  export WORK_DIR="${root}/work"
  bash scripts/run_duca_two_stage_curriculum_variant_gpu1.sh
  seal_file "${RUN_DIR}/completion.json"
}

resolve_preregistered_route() {
  local decision="$1" decision_sha="$2"
  "${PYTHON}" - "${decision}" "${decision_sha}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from tools.bata.select_duca_boundary_burst_candidates import validate_frontend_decision

decision = validate_frontend_decision(
    decision_path=sys.argv[1],
    decision_sha256=sys.argv[2],
    expected_commit=sys.argv[3],
)
route = decision["family_routing"]
print(route["selected_p0_variant"])
print(route["selected_official60_variant"])
print(route["preregistered_projected_family"])
PY
}

# Produce every trainable prerequisite from the current exact commit. Nothing
# in this function accepts a frontend decision, gate, terminal suite, alignment,
# or selector checkpoint from outside this bundle.
run_current_commit_bootstrap() {
  require_one_gpu
  local root="${BUNDLE_ROOT}"
  local r0_root="${root}/r0_holdout_map"
  run_r0 "${r0_root}"

  export RUN_ROOT="${root}"
  export DUCA_R0_PRODUCER_COMMIT="${EXPECTED_COMMIT}"
  export DUCA_R0_SUMMARY_JSON="${r0_root}/r0_summary.json"
  export DUCA_R0_SUMMARY_SHA256_FILE="${r0_root}/r0_summary.sha256"
  mkdir -p "${root}/frontend_split"
  cp -- "${DUCA_FRONTEND_TRAIN_BLOCK_LIST}" \
    "${root}/frontend_split/frontend_train_block_list.txt"
  cp -- "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}" \
    "${root}/frontend_split/frontend_holdout_block_list.txt"
  [[ "$(sha256sum "${root}/frontend_split/frontend_train_block_list.txt" | awk '{print $1}')" == \
     "${DUCA_FRONTEND_TRAIN_BLOCK_LIST_SHA256}" ]] \
    || fail "copied frontend train split drifted"
  [[ "$(sha256sum "${root}/frontend_split/frontend_holdout_block_list.txt" | awk '{print $1}')" == \
     "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST_SHA256}" ]] \
    || fail "copied frontend holdout split drifted"
  bash scripts/run_duca_boundary_burst_p0_gpu1.sh

  local decision="${root}/frontend_decision.json"
  local decision_sha
  decision_sha="$(read_seal "${root}/frontend_decision.sha256")"
  export DUCA_FRONTEND_DECISION_JSON="${decision}"
  export DUCA_FRONTEND_DECISION_SHA256="${decision_sha}"
  export DUCA_BOUNDARY_BURST_GATE_ROOT="${root}/full_model_gate"
  bash scripts/run_duca_boundary_burst_gate_gpu1.sh
  local gate="${root}/full_model_gate/gate_suite.json"
  seal_file "${gate}"
  local gate_sha
  gate_sha="$(read_seal "${gate}.sha256")"
  export DUCA_SELECTED_OPT_GATE_SUITE="${gate}"
  export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="${gate_sha}"

  readarray -t route < <(resolve_preregistered_route "${decision}" "${decision_sha}")
  [[ "${#route[@]}" == 3 && "${route[2]}" == "${PREREGISTERED_FAMILY}" ]] \
    || fail "current-commit preregistered family routing failed"
  local selected_g0="${route[1]}"
  run_curriculum_arm two_stage_exact_uniform \
    "${root}/official60/two_stage_exact_uniform"
  run_curriculum_arm "${selected_g0}" \
    "${root}/official60/${selected_g0}"

  local uniform_completion="${root}/official60/two_stage_exact_uniform/run/completion.json"
  local g0_completion="${root}/official60/${selected_g0}/run/completion.json"
  local terminal="${root}/terminal_u_g0_suite.json"
  "${PYTHON}" -m tools.bata.aggregate_duca_boundary_burst_results \
    --expected-commit "${EXPECTED_COMMIT}" \
    --decision "${decision}" --decision-sha256 "${decision_sha}" \
    --gate "${gate}" --gate-sha256 "${gate_sha}" \
    --completion "${uniform_completion}" \
    --completion-sha256 "$(read_seal "${uniform_completion}.sha256")" \
    --completion "${g0_completion}" \
    --completion-sha256 "$(read_seal "${g0_completion}.sha256")" \
    --output-json "${terminal}"
  seal_file "${terminal}"

  export DUCA_BOUNDARY_BURST_TERMINAL_SUITE="${terminal}"
  export DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256="$(read_seal "${terminal}.sha256")"
  export DUCA_BOUNDARY_BURST_ALIGNMENT_ROOT="${root}/r4_alignment"
  bash scripts/run_duca_boundary_burst_hard_swap_alignment_gpu1.sh

  "${PYTHON}" - "${root}/bootstrap_receipt.json" "${EXPECTED_COMMIT}" \
    "${decision}" "${decision_sha}" "${gate}" "${gate_sha}" \
    "${terminal}" "$(read_seal "${terminal}.sha256")" \
    "${root}/r4_alignment/alignment.json" \
    "$(read_seal "${root}/r4_alignment/alignment.json.sha256")" \
    "${route[0]}" "${selected_g0}" "${route[2]}" <<'PY'
import json
import sys
from pathlib import Path
from tools.bata.duca_selected_axis_training import atomic_write_json

(
    output, commit, decision, decision_sha, gate, gate_sha,
    terminal, terminal_sha, alignment, alignment_sha,
    selected_p0, selected_g0, selected_family,
) = sys.argv[1:]
decision_payload = json.loads(Path(decision).read_text(encoding="utf-8"))
diagnostic = decision_payload.get("r0_diagnostic_provenance")
if (
    selected_family != "R2Q3_privileged_boundary_burst"
    or decision_payload.get("preregistered_projected_family") != selected_family
    or not isinstance(diagnostic, dict)
    or diagnostic.get("routing_authority") is not False
):
    raise SystemExit("bootstrap preregistration/non-routing R0 contract drift")
atomic_write_json(
    Path(output).resolve(),
    {
        "schema": "duca_current_commit_bootstrap_v2",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "selected_p0_variant": selected_p0,
        "selected_g0_variant": selected_g0,
        "preregistered_projected_family": selected_family,
        "routing_source": "preregistered_fixed_candidate",
        "r0_diagnostic_provenance": diagnostic,
        "continuation_rule": decision_payload["continuation_rule"],
        "frontend_decision": {"path": decision, "sha256": decision_sha},
        "gate_suite": {"path": gate, "sha256": gate_sha},
        "terminal_u_g0_suite": {"path": terminal, "sha256": terminal_sha},
        "alignment": {"path": alignment, "sha256": alignment_sha},
    },
)
PY
}

load_bootstrap_exports() {
  local receipt="$1"
  readarray -t bootstrap < <("${PYTHON}" - "${receipt}" "${EXPECTED_COMMIT}" <<'PY'
import json, sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("schema") != "duca_current_commit_bootstrap_v2" or not p.get("ok"):
    raise SystemExit("invalid current-commit bootstrap receipt")
if p.get("git_commit") != sys.argv[2]:
    raise SystemExit("bootstrap commit drift")
if (
    p.get("preregistered_projected_family")
    != "R2Q3_privileged_boundary_burst"
    or p.get("routing_source") != "preregistered_fixed_candidate"
    or p.get("r0_diagnostic_provenance", {}).get("routing_authority") is not False
    or p.get("continuation_rule", {}).get("decision_source")
    != "official60_hard_adapted_g0_vs_exact_uniform"
):
    raise SystemExit("bootstrap routing/continuation contract drift")
for key in ("frontend_decision", "gate_suite", "terminal_u_g0_suite", "alignment"):
    row = p[key]
    path = Path(row["path"])
    import hashlib
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit(f"bootstrap artifact drift: {key}")
print(p["frontend_decision"]["path"])
print(p["frontend_decision"]["sha256"])
print(p["gate_suite"]["path"])
print(p["gate_suite"]["sha256"])
print(p["alignment"]["path"])
print(p["alignment"]["sha256"])
print(p["preregistered_projected_family"])
PY
  )
  [[ "${#bootstrap[@]}" == 7 ]] || fail "bootstrap export resolution failed"
  export DUCA_FRONTEND_DECISION_JSON="${bootstrap[0]}"
  export DUCA_FRONTEND_DECISION_SHA256="${bootstrap[1]}"
  export DUCA_SELECTED_OPT_GATE_SUITE="${bootstrap[2]}"
  export DUCA_SELECTED_OPT_GATE_SUITE_SHA256="${bootstrap[3]}"
  export DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${bootstrap[4]}"
  export DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${bootstrap[5]}"
}

run_r0_r1() {
  require_one_gpu
  "${PYTHON}" -m pytest \
    tests/test_duca_boundary_burst_configs.py \
    tests/test_duca_frontend_p0_contract.py \
    tests/test_duca_transition_only.py \
    tests/test_duca_r5_paper_matrix.py -q \
    2>&1 | tee "${BUNDLE_ROOT}/r1_focused_contracts.out"
  run_r0 "${BUNDLE_ROOT}/r0_holdout_map"
  write_completion "${BUNDLE_ROOT}/completion.json" "${ROLE}" \
    "R0 frozen-detector mAP diagnostic and R1 focused contracts completed"
}

run_independent_child() {
  require_one_gpu
  local variant="${DUCA_PARALLEL_CHILD_VARIANT:-}"
  export DUCA_INDEPENDENT_VARIANT="${variant}"
  export DUCA_INDEPENDENT_ARM_ROOT="${BUNDLE_ROOT}/arm"
  bash scripts/run_duca_independent_official60_gpu1.sh
}

run_r2_r3() {
  local variants=()
  if [[ "${ROLE}" == r2_r3_core ]]; then
    variants=(
      two_stage_exact_uniform
      boundary_burst_r2q3_soft_detached_g0
      boundary_burst_r2q3_hard_detached_g0
    )
  elif [[ "${ROLE}" == r2_r3_adapted ]]; then
    variants=(
      boundary_burst_r2q3_soft_adapted_g0
      boundary_burst_r2q3_g0
      boundary_burst_r4q5_g0
    )
  else
    fail "invalid R2/R3 bundle role: ${ROLE}"
  fi
  local pids=()
  for variant in "${variants[@]}"; do
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
      env DUCA_PARALLEL_BUNDLE_ROLE=r2_r3_child \
      DUCA_PARALLEL_CHILD_VARIANT="${variant}" \
      DUCA_PARALLEL_BUNDLE_ROOT="${BUNDLE_ROOT}/${variant}" \
      bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
      > "${BUNDLE_ROOT}/${variant}.out" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == 0 ]] || fail "at least one R2/R3 arm failed"
  write_completion "${BUNDLE_ROOT}/completion.json" "${ROLE}" \
    "three preregistered R2/R3 arms completed; each learned arm ran P0, gate and official-60"
}

run_feedback_child() {
  require_one_gpu
  local variant="${DUCA_PARALLEL_CHILD_VARIANT:-}"
  run_curriculum_arm "${variant}" "${BUNDLE_ROOT}/${variant}"
}

run_r4() {
  local bootstrap_root="${BUNDLE_ROOT}/current_commit_bootstrap"
  srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
    env DUCA_PARALLEL_BUNDLE_ROLE=current_commit_bootstrap \
    DUCA_PARALLEL_BUNDLE_ROOT="${bootstrap_root}" \
    bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
    > "${BUNDLE_ROOT}/bootstrap.out" 2>&1
  load_bootstrap_exports "${bootstrap_root}/bootstrap_receipt.json"
  [[ "${bootstrap[6]}" == R2Q3_privileged_boundary_burst ]] \
    || fail "formal R4 bundle differs from preregistered R2Q3"

  local variants=(boundary_burst_r2q3_g1 boundary_burst_r2q3_g2)
  local pids=()
  for variant in "${variants[@]}"; do
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
      env DUCA_PARALLEL_BUNDLE_ROLE=r4_feedback_child \
      DUCA_PARALLEL_CHILD_VARIANT="${variant}" \
      DUCA_PARALLEL_BUNDLE_ROOT="${BUNDLE_ROOT}/feedback_${variant}" \
      DUCA_FRONTEND_DECISION_JSON="${DUCA_FRONTEND_DECISION_JSON}" \
      DUCA_FRONTEND_DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256}" \
      DUCA_SELECTED_OPT_GATE_SUITE="${DUCA_SELECTED_OPT_GATE_SUITE}" \
      DUCA_SELECTED_OPT_GATE_SUITE_SHA256="${DUCA_SELECTED_OPT_GATE_SUITE_SHA256}" \
      DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON}" \
      DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256}" \
      bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
      > "${BUNDLE_ROOT}/${variant}.out" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == 0 ]] || fail "R4 G1/G2 failed"
  write_completion "${BUNDLE_ROOT}/completion.json" "${ROLE}" \
    "current-commit P0, U/G0, terminal, alignment, G1 and G2 completed"
}

run_r5_gate_child() {
  require_one_gpu
  export R5_FRONTEND_DECISION="${DUCA_FRONTEND_DECISION_JSON}"
  export R5_FRONTEND_DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256}"
  export R5_ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON}"
  export R5_ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256}"
  export ADATAD_PRETRAIN_PATH="${FROZEN_PRETRAIN}"
  export ADATAD_PRETRAIN_SHA256="${FROZEN_PRETRAIN_SHA256}"
  bash "${DUCA_R5_MATRIX_ROOT}/jobs/temporalmaxer_one_step.sbatch"
}

run_r5_group_child() {
  require_one_gpu
  local backend="${DUCA_PARALLEL_R5_BACKEND:-}"
  local arm="${DUCA_PARALLEL_R5_ARM:-}"
  local matrix_root="${DUCA_R5_MATRIX_ROOT:-}"
  export R5_FRONTEND_DECISION="${DUCA_FRONTEND_DECISION_JSON}"
  export R5_FRONTEND_DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256}"
  export R5_ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON}"
  export R5_ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256}"
  export ADATAD_PRETRAIN_PATH="${FROZEN_PRETRAIN}"
  export ADATAD_PRETRAIN_SHA256="${FROZEN_PRETRAIN_SHA256}"

  local count=0
  while IFS=$'\t' read -r cell_id row_backend row_arm budget seed \
    config config_sha sbatch_file sbatch_sha; do
    [[ "${cell_id}" == id ]] && continue
    [[ "${row_backend}" == "${backend}" && "${row_arm}" == "${arm}" ]] \
      || continue
    [[ "$(sha256sum "${config}" | awk '{print $1}')" == "${config_sha}" ]] \
      || fail "R5 config drift: ${cell_id}"
    [[ "$(sha256sum "${sbatch_file}" | awk '{print $1}')" == "${sbatch_sha}" ]] \
      || fail "R5 sbatch drift: ${cell_id}"
    bash "${sbatch_file}"
    count=$((count + 1))
    if [[ "${seed}" == 3407 && "${backend}" == actionformer ]]; then
      local cost_sbatch
      cost_sbatch="$(awk -F '\t' -v source="${cell_id}" \
        'NR > 1 && $3 == source {print $4}' "${matrix_root}/costs.tsv")"
      [[ -n "${cost_sbatch}" ]] || fail "missing R5 cost job for ${cell_id}"
      bash "${cost_sbatch}"
    fi
  done < "${matrix_root}/cells.tsv"
  local expected_count
  expected_count="$(awk -F '\t' -v backend="${backend}" -v arm="${arm}" \
    'NR > 1 && $2 == backend && $3 == arm {count += 1} END {print count + 0}' \
    "${matrix_root}/cells.tsv")"
  [[ "${expected_count}" -gt 0 && "${count}" == "${expected_count}" ]] \
    || fail "R5 group cell count drift: expected=${expected_count} observed=${count}"
  write_completion "${BUNDLE_ROOT}/completion.json" "${ROLE}" \
    "${count} R5 ${backend}/${arm} cells and applicable same-backend costs completed"
}

run_r5_all() {
  local matrix_root="${DUCA_R5_MATRIX_ROOT:-}"
  [[ -d "${matrix_root}" ]] || fail "generated R5 matrix root is missing"
  local bootstrap_root="${BUNDLE_ROOT}/current_commit_bootstrap"
  srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
    env DUCA_PARALLEL_BUNDLE_ROLE=current_commit_bootstrap \
    DUCA_PARALLEL_BUNDLE_ROOT="${bootstrap_root}" \
    bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
    > "${BUNDLE_ROOT}/bootstrap.out" 2>&1
  load_bootstrap_exports "${bootstrap_root}/bootstrap_receipt.json"
  [[ "${bootstrap[6]}" == R2Q3_privileged_boundary_burst ]] \
    || fail "formal R5 learned source differs from preregistered R2Q3"

  srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
    env DUCA_PARALLEL_BUNDLE_ROLE=r5_gate_child \
    DUCA_PARALLEL_BUNDLE_ROOT="${BUNDLE_ROOT}/mechanism_gate" \
    DUCA_R5_MATRIX_ROOT="${matrix_root}" \
    DUCA_FRONTEND_DECISION_JSON="${DUCA_FRONTEND_DECISION_JSON}" \
    DUCA_FRONTEND_DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256}" \
    DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON}" \
    DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256}" \
    bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
    > "${BUNDLE_ROOT}/mechanism_gate.out" 2>&1

  local groups=(
    actionformer:uniform
    actionformer:learned
    temporalmaxer:uniform
    temporalmaxer:learned
  )
  local pids=()
  for group in "${groups[@]}"; do
    IFS=: read -r backend arm <<<"${group}"
    local group_id="${backend}_${arm}"
    srun --exclusive --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 \
      env DUCA_PARALLEL_BUNDLE_ROLE=r5_group_child \
      DUCA_PARALLEL_BUNDLE_ROOT="${BUNDLE_ROOT}/${group_id}" \
      DUCA_PARALLEL_R5_BACKEND="${backend}" \
      DUCA_PARALLEL_R5_ARM="${arm}" \
      DUCA_R5_MATRIX_ROOT="${matrix_root}" \
      DUCA_FRONTEND_DECISION_JSON="${DUCA_FRONTEND_DECISION_JSON}" \
      DUCA_FRONTEND_DECISION_SHA256="${DUCA_FRONTEND_DECISION_SHA256}" \
      DUCA_BOUNDARY_BURST_ALIGNMENT_JSON="${DUCA_BOUNDARY_BURST_ALIGNMENT_JSON}" \
      DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256="${DUCA_BOUNDARY_BURST_ALIGNMENT_SHA256}" \
      bash scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh \
      > "${BUNDLE_ROOT}/${group_id}.out" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [[ "${failed}" == 0 ]] || fail "at least one R5 backend-by-arm group failed"
  write_completion "${BUNDLE_ROOT}/completion.json" "${ROLE}" \
    "all requested R5 cells and paired ActionFormer candidate/dense cost profiles completed"
}

case "${ROLE}" in
  r0_r1) run_r0_r1 ;;
  r2_r3_core|r2_r3_adapted) run_r2_r3 ;;
  r2_r3_child) run_independent_child ;;
  current_commit_bootstrap) run_current_commit_bootstrap ;;
  r4_r2q3) run_r4 ;;
  r4_feedback_child) run_feedback_child ;;
  r5_all) run_r5_all ;;
  r5_gate_child) run_r5_gate_child ;;
  r5_group_child) run_r5_group_child ;;
  *) fail "unknown bundle role: ${ROLE}" ;;
esac

echo "[DUCA_R0_R5_PARALLEL_BUNDLE] completed ${ROLE}"
