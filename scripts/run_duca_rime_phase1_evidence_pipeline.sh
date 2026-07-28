#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1_EVIDENCE][FAIL] $*" >&2
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

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE1_PIPELINE_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE1_SPLIT_ROLE \
  DUCA_RIME_CODE_GATE_RECEIPT \
  DUCA_RIME_PHASE1_DENSE_CONFIG \
  DUCA_RIME_RELEASED_DENSE_CHECKPOINT \
  DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256 \
  DUCA_RIME_LOCAL_DENSE_CHECKPOINT \
  DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_UNIFORM_CONFIG \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT \
  DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_NO_PROBE_CONFIG \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_NO_PROBE_TRAINED_COMMIT \
  DUCA_RIME_PHASE1_PROBE_CONFIG \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT \
  DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256 \
  DUCA_RIME_PHASE1_PROBE_TRAINED_COMMIT \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_PHASE1_PROTOCOL_MANIFEST \
  DUCA_RIME_PHASE1_PROTOCOL_MANIFEST_SHA256 \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-1 evidence production must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ ! -e "${DUCA_RIME_PHASE1_PIPELINE_ROOT}" ]] \
  || fail "a fresh Phase-1 evidence root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ -f "${DUCA_RIME_CODE_GATE_RECEIPT}" ]] \
  || fail "code-gate receipt is missing"
export DUCA_RIME_CODE_GATE_RECEIPT_SHA256="$(
  sha256sum "${DUCA_RIME_CODE_GATE_RECEIPT}" | awk '{print $1}'
)"
for binding in \
  "${DUCA_RIME_SPLIT_MANIFEST}|${DUCA_RIME_SPLIT_MANIFEST_SHA256}|split manifest" \
  "${DUCA_RIME_CODE_GATE_RECEIPT}|${DUCA_RIME_CODE_GATE_RECEIPT_SHA256}|code gate" \
  "${DUCA_RIME_RELEASED_DENSE_CHECKPOINT}|${DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256}|released dense checkpoint" \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT}|${DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256}|local dense checkpoint" \
  "${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT}|${DUCA_RIME_PHASE1_UNIFORM_CHECKPOINT_SHA256}|uniform checkpoint" \
  "${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT}|${DUCA_RIME_PHASE1_NO_PROBE_CHECKPOINT_SHA256}|no-probe checkpoint" \
  "${DUCA_RIME_PHASE1_PROBE_CHECKPOINT}|${DUCA_RIME_PHASE1_PROBE_CHECKPOINT_SHA256}|probe checkpoint" \
  "${DUCA_RIME_PRETRAIN_PATH}|${DUCA_RIME_PRETRAIN_SHA256}|VideoMAE pretrain" \
  "${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST}|${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST_SHA256}|physical protocol"; do
  IFS='|' read -r path expected label <<<"${binding}"
  check_sha256 "${path}" "${expected}" "${label}"
done

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_PHASE1_SPLIT_ROLE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
role = manifest["train_roles"][sys.argv[2]]
print(role["block_list_path"])
print(role["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 2 ]] \
  || fail "failed to resolve the Phase-1 development role"
check_sha256 "${split_values[0]}" "${split_values[1]}" "development block list"
export DUCA_RIME_PHASE1_EVAL_BLOCK_LIST="${split_values[0]}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  export PRECHECK_ONLY=1
  export DUCA_RIME_PHASE1_DENSE_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/precheck-dense"
  export DUCA_RIME_PHASE1_DENSE_VARIANT=released_dense
  export DUCA_RIME_PHASE1_DENSE_CHECKPOINT="${DUCA_RIME_RELEASED_DENSE_CHECKPOINT}"
  export DUCA_RIME_PHASE1_DENSE_CHECKPOINT_SHA256="${DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256}"
  scripts/run_duca_rime_phase1_dense_eval.sh
  export DUCA_RIME_PHASE1_UNIFORM_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/precheck-uniform"
  export DUCA_RIME_FIXED_BUDGET=384
  scripts/run_duca_rime_phase1_uniform_eval.sh
  export DUCA_RIME_PHASE1_COST_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/precheck-cost"
  export DUCA_RIME_ADATAD_PRETRAIN="${DUCA_RIME_PRETRAIN_PATH}"
  export DUCA_RIME_ADATAD_PRETRAIN_SHA256="${DUCA_RIME_PRETRAIN_SHA256}"
  export DUCA_RIME_PHASE1_PROFILE_SESSION_ID=precheck
  export DUCA_RIME_PHASE1_PROFILE_PAIR_ID=precheck
  scripts/run_duca_rime_phase1_cost_controls.sh
  echo "[DUCA_RIME_PHASE1_EVIDENCE] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE1_PIPELINE_ROOT}"
run_dense() {
  local variant="$1" checkpoint="$2" checkpoint_sha="$3" output="$4"
  export DUCA_RIME_PHASE1_DENSE_VARIANT="${variant}"
  export DUCA_RIME_PHASE1_DENSE_CHECKPOINT="${checkpoint}"
  export DUCA_RIME_PHASE1_DENSE_CHECKPOINT_SHA256="${checkpoint_sha}"
  export DUCA_RIME_PHASE1_DENSE_ROOT="${output}"
  export DUCA_RIME_EVAL_SEED=3407
  scripts/run_duca_rime_phase1_dense_eval.sh
}

run_dense \
  released_dense \
  "${DUCA_RIME_RELEASED_DENSE_CHECKPOINT}" \
  "${DUCA_RIME_RELEASED_DENSE_CHECKPOINT_SHA256}" \
  "${DUCA_RIME_PHASE1_PIPELINE_ROOT}/released_dense"
run_dense \
  local_dense \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT}" \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256}" \
  "${DUCA_RIME_PHASE1_PIPELINE_ROOT}/local_dense_a"
run_dense \
  local_dense \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT}" \
  "${DUCA_RIME_LOCAL_DENSE_CHECKPOINT_SHA256}" \
  "${DUCA_RIME_PHASE1_PIPELINE_ROOT}/local_dense_b"

for budget in 384 192; do
  export DUCA_RIME_FIXED_BUDGET="${budget}"
  export DUCA_RIME_PHASE1_UNIFORM_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/uniform_k${budget}"
  export DUCA_RIME_EVAL_SEED=3407
  scripts/run_duca_rime_phase1_uniform_eval.sh
done

wrapper_gate="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/wrapper_gate.json"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON="${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST}"
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256="${DUCA_RIME_PHASE1_PROTOCOL_MANIFEST_SHA256}"
export DUCA_PROTECTED_GATE_ARM=protected_e2e
export DUCA_PROTECTED_GATE_OUTPUT_JSON="${wrapper_gate}"
scripts/run_duca_protected_physical_full_model_gate_gpu1.sh

export DUCA_RIME_PHASE1_COST_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/cost"
export DUCA_RIME_ADATAD_PRETRAIN="${DUCA_RIME_PRETRAIN_PATH}"
export DUCA_RIME_ADATAD_PRETRAIN_SHA256="${DUCA_RIME_PRETRAIN_SHA256}"
export DUCA_RIME_PHASE1_PROFILE_SESSION_ID="rime-phase1-${DUCA_RIME_EXPECTED_COMMIT:0:12}"
export DUCA_RIME_PHASE1_PROFILE_PAIR_ID="rime-phase1-${SLURM_JOB_ID}"
scripts/run_duca_rime_phase1_cost_controls.sh

export DUCA_RIME_PHASE1_SEAL_ROOT="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/seal"
export DUCA_RIME_PHASE0_REPLICATE_A_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/local_dense_a/localization_metrics.json"
export DUCA_RIME_PHASE0_REPLICATE_A_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE0_REPLICATE_A_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_PHASE0_REPLICATE_B_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/local_dense_b/localization_metrics.json"
export DUCA_RIME_PHASE0_REPLICATE_B_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE0_REPLICATE_B_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_RELEASED_DENSE_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/released_dense/localization_metrics.json"
export DUCA_RIME_RELEASED_DENSE_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_RELEASED_DENSE_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_LOCAL_DENSE_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/local_dense_a/localization_metrics.json"
export DUCA_RIME_LOCAL_DENSE_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_LOCAL_DENSE_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_UNIFORM_K384_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/uniform_k384/localization_metrics.json"
export DUCA_RIME_UNIFORM_K384_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_UNIFORM_K384_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/uniform_k384/inference_ledger_summary.json"
export DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY_SHA256="$(
  sha256sum "${DUCA_RIME_UNIFORM_K384_LEDGER_SUMMARY}" | awk '{print $1}'
)"
export DUCA_RIME_UNIFORM_K192_METRICS="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/uniform_k192/localization_metrics.json"
export DUCA_RIME_UNIFORM_K192_METRICS_SHA256="$(
  sha256sum "${DUCA_RIME_UNIFORM_K192_METRICS}" | awk '{print $1}'
)"
export DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/uniform_k192/inference_ledger_summary.json"
export DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY_SHA256="$(
  sha256sum "${DUCA_RIME_UNIFORM_K192_LEDGER_SUMMARY}" | awk '{print $1}'
)"
export DUCA_RIME_WRAPPER_GATE="${wrapper_gate}"
export DUCA_RIME_WRAPPER_GATE_SHA256="$(sha256sum "${wrapper_gate}" | awk '{print $1}')"
export DUCA_RIME_NO_PROBE_PROFILE="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/cost/no_probe_uniform.summary.json"
export DUCA_RIME_NO_PROBE_PROFILE_SHA256="$(
  sha256sum "${DUCA_RIME_NO_PROBE_PROFILE}" | awk '{print $1}'
)"
export DUCA_RIME_PROBE_PROFILE="${DUCA_RIME_PHASE1_PIPELINE_ROOT}/cost/probe_uniform.summary.json"
export DUCA_RIME_PROBE_PROFILE_SHA256="$(
  sha256sum "${DUCA_RIME_PROBE_PROFILE}" | awk '{print $1}'
)"
scripts/run_duca_rime_phase1_seal.sh

python - \
  "${DUCA_RIME_PHASE1_PIPELINE_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
phase1 = root / "seal" / "phase1_receipt.json"
if not phase1.is_file():
    raise SystemExit("Phase-1 seal is missing")
payload = {
    "schema_version": "duca_rime_phase1_evidence_pipeline_receipt_v1",
    "status": "passed",
    "git_commit": sys.argv[2],
    "slurm_job_id": sys.argv[3],
    "phase1_receipt_path": str(phase1),
    "phase1_receipt_sha256": hashlib.sha256(phase1.read_bytes()).hexdigest(),
    "official_final_subset_consumed": False,
}
target = root / "pipeline_receipt.json"
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
echo "[DUCA_RIME_PHASE1_EVIDENCE] PASS ${DUCA_RIME_PHASE1_PIPELINE_ROOT}"
