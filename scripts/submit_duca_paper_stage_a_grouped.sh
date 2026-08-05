#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_STAGE_A_GROUPED_SUBMIT][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_PAPER_REPO_ROOT \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_RUN_ROOT \
  DUCA_PAPER_PRETRAIN_PATH \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_PATH \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_PATH \
  DUCA_PAPER_CLASS_MAP_SHA256 \
  DUCA_PAPER_CODE_GATE_RECEIPT \
  DUCA_PAPER_CODE_GATE_RECEIPT_SHA256 \
  DUCA_PAPER_SHORT_WINDOW_GATE_JSON \
  DUCA_PAPER_SHORT_WINDOW_GATE_SHA256 \
  DUCA_PAPER_NUMERIC_GATE_JSON \
  DUCA_PAPER_NUMERIC_GATE_SHA256 \
  DUCA_PAPER_EXACT211_UID_GATE_JSON \
  DUCA_PAPER_EXACT211_UID_GATE_SHA256; do
  required "${name}"
done

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${DUCA_PAPER_REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
DUCA_PAPER_RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "DUCA_PAPER_RUN_ROOT" \
    "${DUCA_PAPER_REPO_ROOT}" \
    "${BASE}" \
    "${DUCA_PAPER_RUN_ROOT}"
)" || fail "Stage-A root violates the formal path contract"

command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v scontrol >/dev/null || fail "scontrol is unavailable"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
old_failed_commit="2df0103ec1c26ff7cff7ed15f399e78e640df211"
old_failed_root="/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_2df0103e_20260802_120351"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" != "${old_failed_commit}" ]] \
  || fail "the immutable failed Stage-A source cannot be redeployed"
[[ "${DUCA_PAPER_RUN_ROOT}" != "${old_failed_root}" ]] \
  || fail "the immutable failed Stage-A transaction root cannot be reused"
cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_PAPER_RUN_ROOT}" ]] || fail "a fresh Stage-A root is required"

active_jobs="$(squeue -h -u "$(id -un)" | wc -l)"
max_jobs="${DUCA_PAPER_MAX_SUBMIT_JOBS:-16}"
[[ "${max_jobs}" =~ ^[0-9]+$ && "${active_jobs}" =~ ^[0-9]+$ ]] \
  || fail "invalid scheduler job-count contract"
(( active_jobs + 7 <= max_jobs )) \
  || fail "seven scheduler slots are not available for isolated grouped Stage A"

for binding in \
  "${DUCA_PAPER_PRETRAIN_PATH}|${DUCA_PAPER_PRETRAIN_SHA256}|VideoMAE initialization" \
  "${DUCA_PAPER_ANNOTATION_PATH}|${DUCA_PAPER_ANNOTATION_SHA256}|THUMOS14 annotation" \
  "${DUCA_PAPER_CLASS_MAP_PATH}|${DUCA_PAPER_CLASS_MAP_SHA256}|THUMOS14 class map" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}|${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}|clean Linux/PyTorch code gate" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}|${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}|real short-window Slurm gate" \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}|${DUCA_PAPER_NUMERIC_GATE_SHA256}|production-like learned numeric gate" \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}|${DUCA_PAPER_EXACT211_UID_GATE_SHA256}|exact-211 physical UID gate"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | cut -d ' ' -f 1)" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done
python -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
python -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}"
python -m tools.bata.validate_duca_paper_numeric_gate \
  --receipt "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_NUMERIC_GATE_SHA256}"
python -m tools.bata.validate_duca_paper_exact211_uid_gate \
  --receipt "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}"

mkdir -p "${DUCA_PAPER_RUN_ROOT}/logs"
export DUCA_PAPER_CELLS_ROOT="${DUCA_PAPER_RUN_ROOT}/cells"
export DUCA_PAPER_MATRIX_ROOT="${DUCA_PAPER_RUN_ROOT}/matrix"
export DUCA_PAPER_MATRIX_MANIFEST="${DUCA_PAPER_RUN_ROOT}/protocol_manifest.json"
python -m tools.bata.build_duca_paper_matrix_manifest \
  --repo-root "${DUCA_PAPER_REPO_ROOT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --pretrain "${DUCA_PAPER_PRETRAIN_PATH}" \
  --annotation "${DUCA_PAPER_ANNOTATION_PATH}" \
  --class-map "${DUCA_PAPER_CLASS_MAP_PATH}" \
  --short-window-gate "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --numeric-gate "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  --exact211-uid-gate "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  --output "${DUCA_PAPER_MATRIX_MANIFEST}"
export DUCA_PAPER_MATRIX_MANIFEST_SHA256="$(
  sha256sum "${DUCA_PAPER_MATRIX_MANIFEST}" | cut -d ' ' -f 1
)"

export DUCA_PAPER_REPO_ROOT DUCA_PAPER_EXPECTED_COMMIT
export DUCA_PAPER_PRETRAIN_PATH DUCA_PAPER_PRETRAIN_SHA256
export DUCA_PAPER_ANNOTATION_PATH DUCA_PAPER_ANNOTATION_SHA256
export DUCA_PAPER_CLASS_MAP_PATH DUCA_PAPER_CLASS_MAP_SHA256
export DUCA_PAPER_CODE_GATE_RECEIPT DUCA_PAPER_CODE_GATE_RECEIPT_SHA256
export DUCA_PAPER_SHORT_WINDOW_GATE_JSON DUCA_PAPER_SHORT_WINDOW_GATE_SHA256
export DUCA_PAPER_NUMERIC_GATE_JSON DUCA_PAPER_NUMERIC_GATE_SHA256
export DUCA_PAPER_EXACT211_UID_GATE_JSON DUCA_PAPER_EXACT211_UID_GATE_SHA256
export DUCA_PAPER_CELLS_ROOT DUCA_PAPER_MATRIX_ROOT
export DUCA_PAPER_MATRIX_MANIFEST DUCA_PAPER_MATRIX_MANIFEST_SHA256

job_ids=()
cleanup_held_jobs() {
  if [[ "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_held_jobs ERR INT TERM

for seed in 5801 8123 12011; do
  for group in controls duca; do
    job_id="$(
      sbatch \
        --parsable \
        --hold \
        --partition=gpu \
        --gres=gpu:2 \
        --cpus-per-task="${DUCA_PAPER_CPUS:-16}" \
        --time="${DUCA_PAPER_GROUPED_TIME:-7-00:00:00}" \
        --job-name="ducaA-${group}-s${seed}" \
        --output="${DUCA_PAPER_RUN_ROOT}/logs/%x-%j.out" \
        --export="ALL,DUCA_PAPER_SEED=${seed},DUCA_PAPER_GROUP=${group}" \
        --wrap="/bin/bash ${DUCA_PAPER_REPO_ROOT}/scripts/run_duca_paper_stage_a_seed.sh"
    )"
    job_ids+=("${job_id%%;*}")
  done
done
[[ "${#job_ids[@]}" == 6 ]] || fail "Stage-A did not create six isolated seed groups"

dependency="$(IFS=:; echo "${job_ids[*]}")"
seal_job="$(
  sbatch \
    --parsable \
    --hold \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --time=02:00:00 \
    --job-name=ducaA-seal \
    --dependency="afterok:${dependency}" \
    --output="${DUCA_PAPER_RUN_ROOT}/logs/%x-%j.out" \
    --export="ALL" \
    --wrap="/bin/bash ${DUCA_PAPER_REPO_ROOT}/scripts/run_duca_paper_stage_a_seal.sh"
)"
seal_job="${seal_job%%;*}"
job_ids+=("${seal_job}")

python - \
  "${DUCA_PAPER_RUN_ROOT}/submission_manifest.json" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${DUCA_PAPER_MATRIX_MANIFEST}" \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  "${DUCA_PAPER_NUMERIC_GATE_SHA256}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}" \
  "${DUCA_PAPER_CELLS_ROOT}" \
  "${seal_job}" \
  "${job_ids[@]:0:6}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    output,
    commit,
    protocol,
    protocol_sha,
    code_gate,
    code_gate_sha,
    short_gate,
    short_gate_sha,
    numeric_gate,
    numeric_gate_sha,
    exact211_gate,
    exact211_gate_sha,
    cells_root,
    seal_job,
    *group_jobs,
) = sys.argv[1:]
specs = [
    (seed, group)
    for seed in (5801, 8123, 12011)
    for group in ("controls", "duca")
]
if len(group_jobs) != 6:
    raise SystemExit("grouped Stage-A submission cardinality drift")
payload = {
    "schema_version": "duca_paper_stage_a_grouped_submission_v2",
    "status": "held_complete",
    "git_commit": commit,
    "protocol_manifest_path": str(pathlib.Path(protocol).resolve()),
    "protocol_manifest_sha256": protocol_sha,
    "code_gate_path": str(pathlib.Path(code_gate).resolve()),
    "code_gate_sha256": code_gate_sha,
    "short_window_gate_path": str(pathlib.Path(short_gate).resolve()),
    "short_window_gate_sha256": short_gate_sha,
    "numeric_gate_path": str(pathlib.Path(numeric_gate).resolve()),
    "numeric_gate_sha256": numeric_gate_sha,
    "exact211_uid_gate_path": str(pathlib.Path(exact211_gate).resolve()),
    "exact211_uid_gate_sha256": exact211_gate_sha,
    "cells_root": str(pathlib.Path(cells_root).resolve()),
    "logical_cell_count": 12,
    "scheduler_job_count": 7,
    "sequential_scheduler_grouping_only": True,
    "mixed_k_failure_blocks_duca_arm": False,
    "seed_group_jobs": [
        {
            "seed": seed,
            "group": group,
            "slurm_job_id": job_id,
            "arms": (
                [
                    "dense",
                    "uniform_fixed_k384",
                    "uniform_mixed_train_k384_eval",
                ]
                if group == "controls"
                else ["duca_fixed_k384"]
            ),
        }
        for (seed, group), job_id in zip(specs, group_jobs)
    ],
    "seal_job_id": seal_job,
    "release_is_transactional": True,
    "phase_b_submitted": False,
    "single_seed_claim_allowed": False,
}
target = pathlib.Path(output)
text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
with target.open("x", encoding="utf-8") as handle:
    handle.write(text)
    handle.flush()
    os.fsync(handle.fileno())
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
pathlib.Path(str(target) + ".sha256").write_text(
    f"{digest}  {target.name}\n", encoding="utf-8"
)
receipt = {
    "schema_version": "duca_paper_stage_a_grouped_submission_receipt_v2",
    "status": "held_complete",
    "git_commit": commit,
    "code_gate_path": str(pathlib.Path(code_gate).resolve()),
    "code_gate_sha256": code_gate_sha,
    "short_window_gate_path": str(pathlib.Path(short_gate).resolve()),
    "short_window_gate_sha256": short_gate_sha,
    "numeric_gate_path": str(pathlib.Path(numeric_gate).resolve()),
    "numeric_gate_sha256": numeric_gate_sha,
    "exact211_uid_gate_path": str(pathlib.Path(exact211_gate).resolve()),
    "exact211_uid_gate_sha256": exact211_gate_sha,
    "submission_manifest_path": str(target.resolve()),
    "submission_manifest_sha256": digest,
    "seed_group_job_ids": group_jobs,
    "seal_job_id": seal_job,
    "released": False,
}
pathlib.Path(str(target) + ".receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf '%s\n' "${job_ids[@]:0:6}" > "${DUCA_PAPER_RUN_ROOT}/seed_group_job_ids.txt"
printf '%s\n' "${seal_job}" > "${DUCA_PAPER_RUN_ROOT}/seal_job_id.txt"
release_list="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "${release_list}"
python - "${DUCA_PAPER_RUN_ROOT}/submission_manifest.json.receipt.json" <<'PY'
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
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
sha256sum "${DUCA_PAPER_RUN_ROOT}/submission_manifest.json.receipt.json" \
  > "${DUCA_PAPER_RUN_ROOT}/submission_manifest.json.receipt.sha256"
trap - ERR INT TERM

echo "[DUCA_PAPER_STAGE_A_GROUPED_SUBMIT] RELEASED 6 isolated seed groups; seal ${seal_job}"
