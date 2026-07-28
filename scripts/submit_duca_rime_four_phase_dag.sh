#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_RIME_FOUR_PHASE_SUBMIT][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

dense_recovery_mode="${DUCA_RIME_DENSE_RECOVERY_MODE:-fresh_train}"
[[ "${dense_recovery_mode}" == fresh_train || "${dense_recovery_mode}" == salvage ]] \
  || fail "DUCA_RIME_DENSE_RECOVERY_MODE must be fresh_train or salvage"
export DUCA_RIME_DENSE_RECOVERY_MODE="${dense_recovery_mode}"
[[ "${DUCA_RIME_ENABLE_PHASE4:-0}" == 0 ]] \
  || fail "this recovery DAG keeps official-final Phase 4 sealed"
export DUCA_RIME_ENABLE_PHASE4=0

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_DEPLOYMENT_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_PHASE1_PROTOCOL_MANIFEST \
  DUCA_RIME_PHASE1_PROTOCOL_MANIFEST_SHA256 \
  DUCA_RIME_RELEASED_DENSE_CHECKPOINT \
  DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256 \
  DUCA_RIME_LOCAL_DENSE_CHECKPOINT \
  DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_CANDIDATE_COSTS \
  DUCA_RIME_TARGET_MEAN_COST \
  DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST \
  DUCA_RIME_DECODER_FAMILY \
  DUCA_RIME_RISK_WEIGHT \
  DUCA_RIME_RISK_THRESHOLD \
  DUCA_RIME_O4_MAX_BRIER \
  DUCA_RIME_O4_MAX_ECE \
  DUCA_RIME_O4_MIN_COVERAGE \
  DUCA_RIME_O4_MAX_LOW_RISK_FAILURE \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
  export "${name}"
done

[[ "${DUCA_RIME_CANDIDATE_BUDGETS}" == "192,256,384,512" ]] \
  || fail "candidate-budget protocol drift"
[[ "${DUCA_RIME_CANDIDATE_COSTS}" == "192,256,384,512" ]] \
  || fail "candidate-cost proxy protocol drift"
[[ "${DUCA_RIME_TARGET_MEAN_COST}" == "384" ]] \
  || fail "primary mean-cost protocol drift"
[[ "${DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST}" == "192" ]] \
  || fail "floor-panel mean-cost protocol drift"
[[ "${DUCA_RIME_DECODER_FAMILY}" == "weak_overlap" ]] \
  || fail "decoder-family protocol drift"
[[ "${DUCA_RIME_RISK_WEIGHT}" == "1.0" ]] \
  || fail "risk-weight protocol drift"
[[ "${DUCA_RIME_RISK_THRESHOLD}" == "0.35" ]] \
  || fail "risk-threshold protocol drift"
[[ "${DUCA_RIME_O4_MAX_BRIER}" == "0.25" ]] \
  || fail "O4 Brier threshold drift"
[[ "${DUCA_RIME_O4_MAX_ECE}" == "0.10" ]] \
  || fail "O4 ECE threshold drift"
[[ "${DUCA_RIME_O4_MIN_COVERAGE}" == "0.25" ]] \
  || fail "O4 coverage threshold drift"
[[ "${DUCA_RIME_O4_MAX_LOW_RISK_FAILURE}" == "0.35" ]] \
  || fail "O4 low-risk failure threshold drift"
[[ "${DUCA_RIME_SHORT_MAX_SECONDS}" == "2.0" ]] \
  || fail "short-duration stratum drift"
[[ "${DUCA_RIME_MEDIUM_MAX_SECONDS}" == "8.0" ]] \
  || fail "medium-duration stratum drift"

[[ -z "${SLURM_JOB_ID:-}" ]] \
  || fail "the four-phase DAG must be submitted from a login node"
command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v scontrol >/dev/null || fail "scontrol is unavailable"
command -v scancel >/dev/null || fail "scancel is unavailable"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ ! -e "${DUCA_RIME_DEPLOYMENT_ROOT}" ]] \
  || fail "a fresh deployment root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

for binding in \
  "${DUCA_RIME_SPLIT_MANIFEST}|${DUCA_RIME_SPLIT_MANIFEST_SHA256}|split manifest" \
  "${DUCA_RIME_PRETRAIN_PATH}|${DUCA_RIME_PRETRAIN_SHA256}|VideoMAE pretrain" \
  "${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST}|${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST_SHA256}|physical protocol" \
  "${DUCA_RIME_RELEASED_DENSE_CHECKPOINT}|${DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256}|released dense checkpoint" \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT}|${DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256}|local dense checkpoint" \
  "${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT}|${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256}|uniform checkpoint" \
  "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT}|${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256}|no-probe checkpoint" \
  "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT}|${DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256}|probe checkpoint"; do
  IFS='|' read -r path expected label <<<"${binding}"
  check_sha256 "${path}" "${expected}" "${label}"
done

export DUCA_RIME_PHASE1_SPLIT_ROLE=certification_development
export DUCA_RIME_PHASE1_DENSE_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_dense_phase1_control.py"
export DUCA_RIME_PHASE1_UNIFORM_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_uniform_phase1_control.py"
export DUCA_RIME_PHASE1_NO_PROBE_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_no_probe_uniform_phase1_cost.py"
export DUCA_RIME_PHASE1_PROBE_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_probe_uniform_phase1_cost.py"
export DUCA_RIME_PHASE2_MIXED_K_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_uniform_mixed_k_total60.py"
export DUCA_RIME_DENSE_ACTIONFORMER_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_dense_actionformer_total60.py"
export DUCA_RIME_DENSE_TRIDET_CONFIG="${DUCA_RIME_REPO_ROOT}/configs/adatad/thumos/duca_rime_dense_tridet_total60.py"

export DUCA_RIME_CODE_GATE_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/code_gate"
export DUCA_RIME_PHASE1_PIPELINE_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase1"
export DUCA_RIME_DENSE_ACTIONFORMER_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/dense_actionformer"
export DUCA_RIME_DENSE_TRIDET_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/dense_tridet"
export DUCA_RIME_PHASE2_PIPELINE_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase2"
export DUCA_RIME_PHASE3_CONTROLLER_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase3_controller"
export DUCA_RIME_PHASE3_ASSET_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase3_assets"
export DUCA_RIME_PHASE3_BUNDLE_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase3"
export DUCA_RIME_PHASE4_CELLS_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase4/cells"
export DUCA_RIME_PHASE4_SUBMISSION_ROOT="${DUCA_RIME_DEPLOYMENT_ROOT}/phase4/submission"
export DUCA_RIME_CODE_GATE_RECEIPT="${DUCA_RIME_CODE_GATE_ROOT}/gate.receipt"
export DUCA_RIME_PHASE1_PIPELINE_RECEIPT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/pipeline_receipt.json"
export DUCA_RIME_PHASE2_PIPELINE_RECEIPT="${DUCA_RIME_PHASE2_PIPELINE_ROOT}/evidence/pipeline_receipt.json"

export DUCA_RIME_DENSE_CONFIG_ACTIONFORMER="${DUCA_RIME_DENSE_ACTIONFORMER_CONFIG}"
export DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER="${DUCA_RIME_DENSE_ACTIONFORMER_ROOT}/train/gpu1_id0/checkpoint/terminal_ema.pth"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER="${DUCA_RIME_DENSE_ACTIONFORMER_ROOT}/checkpoint_evidence.json"
export DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_RIME_DENSE_CONFIG_TRIDET="${DUCA_RIME_DENSE_TRIDET_CONFIG}"
export DUCA_RIME_DENSE_CHECKPOINT_TRIDET="${DUCA_RIME_DENSE_TRIDET_ROOT}/train/gpu1_id0/checkpoint/terminal_ema.pth"
export DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET="${DUCA_RIME_DENSE_TRIDET_ROOT}/checkpoint_evidence.json"
export DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET="${DUCA_RIME_EXPECTED_COMMIT}"

salvage_manifest_path=""
salvage_manifest_sha256=""
salvage_source_commit=""
if [[ "${dense_recovery_mode}" == salvage ]]; then
  required DUCA_RIME_DENSE_SALVAGE_MANIFEST
  required DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256
  export DUCA_RIME_DENSE_SALVAGE_MANIFEST
  export DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256
  check_sha256 \
    "${DUCA_RIME_DENSE_SALVAGE_MANIFEST}" \
    "${DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256}" \
    "dense salvage manifest"
  readarray -t salvage_values < <(
    python - \
      "${DUCA_RIME_DENSE_SALVAGE_MANIFEST}" \
      "${DUCA_RIME_EXPECTED_COMMIT}" \
      "${DUCA_RIME_DENSE_ACTIONFORMER_ROOT}" \
      "${DUCA_RIME_DENSE_TRIDET_ROOT}" <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    payload.get("schema_version") != "duca_rime_dense_salvage_manifest_v1"
    or payload.get("status") != "frozen"
    or payload.get("uses_official_final") is not False
    or payload.get("recovery_git_commit") != sys.argv[2]
    or payload.get("failed_transaction", {}).get("terminal_state")
    != "failed_closed"
):
    raise SystemExit("dense salvage manifest contract drift")
source_commit = str(payload.get("failed_transaction", {}).get("git_commit", ""))
if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
    raise SystemExit("dense salvage source commit is invalid")
expected_outputs = {
    "ActionFormer": pathlib.Path(sys.argv[3]).resolve(),
    "TriDet": pathlib.Path(sys.argv[4]).resolve(),
}
for backend, expected_output in expected_outputs.items():
    source = payload.get("sources", {}).get(backend, {})
    if (
        source.get("backend") != backend
        or source.get("original_job_state") != "FAILED"
        or pathlib.Path(str(source.get("output_root", ""))).resolve()
        != expected_output
    ):
        raise SystemExit(f"dense salvage {backend} source/output contract drift")
print(manifest_path)
print(source_commit)
PY
  )
  [[ "${#salvage_values[@]}" == 2 ]] \
    || fail "failed to resolve dense salvage manifest"
  salvage_manifest_path="${salvage_values[0]}"
  salvage_manifest_sha256="${DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256}"
  salvage_source_commit="${salvage_values[1]}"
  export DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER="${salvage_source_commit}"
  export DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET="${salvage_source_commit}"
fi

mkdir -p "${DUCA_RIME_DEPLOYMENT_ROOT}/logs"
bootstrap="source /etc/profile && module load cuda/11.8 && module load miniforge3/24.11 && source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate"
job_ids=()
submitted_job=""
cleanup_held_jobs() {
  if [[ "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_held_jobs ERR INT TERM

submit_job() {
  local name="$1" time_limit="$2" dependency="$3" command="$4"
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi
  local job
  job="$(
    sbatch \
      --parsable \
      --hold \
      --partition=gpu \
      --gres=gpu:1 \
      --cpus-per-task=8 \
      --time="${time_limit}" \
      --job-name="${name}" \
      --chdir="${DUCA_RIME_REPO_ROOT}" \
      --output="${DUCA_RIME_DEPLOYMENT_ROOT}/logs/%x-%j.out" \
      "${dependency_args[@]}" \
      --export=ALL \
      --wrap="exec /bin/bash -lc '${bootstrap} && exec ${command}'"
  )"
  job="${job%%;*}"
  job_ids+=("${job}")
  submitted_job="${job}"
}

submit_job rime-code 04:00:00 "" scripts/run_duca_rime_code_gate.sh
code_job="${submitted_job}"
submit_job \
  rime-phase1 \
  3-00:00:00 \
  "${code_job}" \
  scripts/run_duca_rime_phase1_evidence_pipeline.sh
phase1_job="${submitted_job}"
submit_job \
  "rime-dense-af-${dense_recovery_mode}" \
  "$([[ "${dense_recovery_mode}" == salvage ]] && echo 12:00:00 || echo 5-00:00:00)" \
  "${code_job}" \
  "$(
    if [[ "${dense_recovery_mode}" == salvage ]]; then
      printf '%s' \
        "env DUCA_RIME_DENSE_SALVAGE_BACKEND=ActionFormer" \
        " DUCA_RIME_DENSE_SALVAGE_ROOT=${DUCA_RIME_DENSE_ACTIONFORMER_ROOT}" \
        " DUCA_RIME_DENSE_SALVAGE_CONFIG=${DUCA_RIME_DENSE_ACTIONFORMER_CONFIG}" \
        " bash scripts/run_duca_rime_dense_salvage.sh"
    else
      printf '%s' scripts/run_duca_rime_dense_actionformer_train.sh
    fi
  )"
dense_actionformer_job="${submitted_job}"
submit_job \
  "rime-dense-td-${dense_recovery_mode}" \
  "$([[ "${dense_recovery_mode}" == salvage ]] && echo 12:00:00 || echo 5-00:00:00)" \
  "${code_job}" \
  "$(
    if [[ "${dense_recovery_mode}" == salvage ]]; then
      printf '%s' \
        "env DUCA_RIME_DENSE_SALVAGE_BACKEND=TriDet" \
        " DUCA_RIME_DENSE_SALVAGE_ROOT=${DUCA_RIME_DENSE_TRIDET_ROOT}" \
        " DUCA_RIME_DENSE_SALVAGE_CONFIG=${DUCA_RIME_DENSE_TRIDET_CONFIG}" \
        " bash scripts/run_duca_rime_dense_salvage.sh"
    else
      printf '%s' scripts/run_duca_rime_dense_tridet_train.sh
    fi
  )"
dense_tridet_job="${submitted_job}"
submit_job \
  rime-phase2 \
  7-00:00:00 \
  "${phase1_job}" \
  scripts/run_duca_rime_phase2_train_and_evidence_pipeline.sh
phase2_job="${submitted_job}"
phase3_dependency="${phase2_job}:${dense_actionformer_job}:${dense_tridet_job}"
submit_job \
  rime-phase3-controller \
  12:00:00 \
  "${phase3_dependency}" \
  scripts/run_duca_rime_phase3_submit_controller.sh
phase3_controller_job="${submitted_job}"

python - \
  "${DUCA_RIME_DEPLOYMENT_ROOT}/submission_manifest.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${code_job}" \
  "${phase1_job}" \
  "${dense_actionformer_job}" \
  "${dense_tridet_job}" \
  "${phase2_job}" \
  "${phase3_controller_job}" \
  "${dense_recovery_mode}" \
  "${salvage_manifest_path}" \
  "${salvage_manifest_sha256}" \
  "${salvage_source_commit}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1]).resolve()
commit = sys.argv[2]
names = (
    "code_gate",
    "phase1",
    "dense_actionformer",
    "dense_tridet",
    "phase2",
    "phase3_controller",
)
jobs = dict(zip(names, sys.argv[3:]))
recovery_mode, salvage_manifest, salvage_manifest_sha, salvage_source_commit = (
    sys.argv[9:13]
)
payload = {
    "schema_version": "duca_rime_four_phase_submission_v2",
    "status": "held_complete",
    "git_commit": commit,
    "jobs": jobs,
    "dependencies": {
        "phase1": [jobs["code_gate"]],
        "dense_actionformer": [jobs["code_gate"]],
        "dense_tridet": [jobs["code_gate"]],
        "phase2": [jobs["phase1"]],
        "phase3_controller": [
            jobs["phase2"],
            jobs["dense_actionformer"],
            jobs["dense_tridet"],
        ],
    },
    "dense_recovery": {
        "mode": recovery_mode,
        "salvage_manifest_path": salvage_manifest or None,
        "salvage_manifest_sha256": salvage_manifest_sha or None,
        "source_training_git_commit": salvage_source_commit or None,
        "source_jobs_remain_failed": recovery_mode == "salvage",
        "failed_transaction_mutated": False,
        "claim_scope": (
            "engineering_dense_reference_recovery_not_method_evidence"
            if recovery_mode == "salvage"
            else "fresh_dense_reference_training"
        ),
    },
    "frozen_protocol_inputs": {
        "candidate_budgets": os.environ["DUCA_RIME_CANDIDATE_BUDGETS"],
        "candidate_costs": os.environ["DUCA_RIME_CANDIDATE_COSTS"],
        "target_mean_cost": os.environ["DUCA_RIME_TARGET_MEAN_COST"],
        "phase4_second_target_mean_cost": os.environ[
            "DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST"
        ],
        "decoder_family": os.environ["DUCA_RIME_DECODER_FAMILY"],
        "risk_weight": os.environ["DUCA_RIME_RISK_WEIGHT"],
        "risk_threshold": os.environ["DUCA_RIME_RISK_THRESHOLD"],
        "o4_max_brier": os.environ["DUCA_RIME_O4_MAX_BRIER"],
        "o4_max_ece": os.environ["DUCA_RIME_O4_MAX_ECE"],
        "o4_min_coverage": os.environ["DUCA_RIME_O4_MIN_COVERAGE"],
        "o4_max_low_risk_failure": os.environ[
            "DUCA_RIME_O4_MAX_LOW_RISK_FAILURE"
        ],
        "short_max_seconds": os.environ["DUCA_RIME_SHORT_MAX_SECONDS"],
        "medium_max_seconds": os.environ["DUCA_RIME_MEDIUM_MAX_SECONDS"],
    },
    "expected_terminal_artifacts": {
        "phase1": str(target.parent / "phase1" / "pipeline_receipt.json"),
        "dense_actionformer": str(
            target.parent / "dense_actionformer" / "checkpoint_evidence.json"
        ),
        "dense_tridet": str(
            target.parent / "dense_tridet" / "checkpoint_evidence.json"
        ),
        "phase2": str(target.parent / "phase2" / "pipeline_receipt.json"),
        "phase3": str(target.parent / "phase3" / "seal" / "phase3_receipt.json"),
        "phase4": None,
    },
    "phase4_submission_enabled": False,
    "official_final_sealed": True,
    "release_is_transactional": True,
}
text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

def atomic_write(path, content):
    path = pathlib.Path(path)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

atomic_write(target, text)
atomic_write(str(target) + ".sha256", f"{digest}  {target.name}\n")
atomic_write(
    str(target) + ".receipt.json",
    json.dumps(
        {
            "schema_version": "duca_rime_four_phase_submission_receipt_v1",
            "status": "held_complete",
            "git_commit": commit,
            "manifest_path": str(target),
            "manifest_sha256": digest,
            "job_ids": list(jobs.values()),
            "released": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
)
PY

job_ids_tmp="${DUCA_RIME_DEPLOYMENT_ROOT}/.job_ids.txt.partial.$$"
printf '%s\n' "${job_ids[@]}" > "${job_ids_tmp}"
mv "${job_ids_tmp}" "${DUCA_RIME_DEPLOYMENT_ROOT}/job_ids.txt"
release_list="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "${release_list}"
trap - ERR INT TERM
python - "${DUCA_RIME_DEPLOYMENT_ROOT}/submission_manifest.json.receipt.json" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = json.loads(target.read_text(encoding="utf-8"))
payload["status"] = "released"
payload["released"] = True
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
echo \
  "[DUCA_RIME_FOUR_PHASE_SUBMIT] RELEASED ${job_ids[*]} (Phase-3/4 child DAGs are fail-closed)"
