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
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v scontrol >/dev/null || fail "scontrol is unavailable"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
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
(( active_jobs + 4 <= max_jobs )) \
  || fail "four scheduler slots are not available for grouped Stage A"

for binding in \
  "${DUCA_PAPER_PRETRAIN_PATH}|${DUCA_PAPER_PRETRAIN_SHA256}|VideoMAE initialization" \
  "${DUCA_PAPER_ANNOTATION_PATH}|${DUCA_PAPER_ANNOTATION_SHA256}|THUMOS14 annotation" \
  "${DUCA_PAPER_CLASS_MAP_PATH}|${DUCA_PAPER_CLASS_MAP_SHA256}|THUMOS14 class map"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | cut -d ' ' -f 1)" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done

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
  --output "${DUCA_PAPER_MATRIX_MANIFEST}"
export DUCA_PAPER_MATRIX_MANIFEST_SHA256="$(
  sha256sum "${DUCA_PAPER_MATRIX_MANIFEST}" | cut -d ' ' -f 1
)"

export DUCA_PAPER_REPO_ROOT DUCA_PAPER_EXPECTED_COMMIT
export DUCA_PAPER_PRETRAIN_PATH DUCA_PAPER_PRETRAIN_SHA256
export DUCA_PAPER_ANNOTATION_PATH DUCA_PAPER_ANNOTATION_SHA256
export DUCA_PAPER_CLASS_MAP_PATH DUCA_PAPER_CLASS_MAP_SHA256
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
  job_id="$(
    sbatch \
      --parsable \
      --hold \
      --partition=gpu \
      --gres=gpu:2 \
      --cpus-per-task="${DUCA_PAPER_CPUS:-16}" \
      --time="${DUCA_PAPER_GROUPED_TIME:-7-00:00:00}" \
      --job-name="ducaA-seed${seed}" \
      --output="${DUCA_PAPER_RUN_ROOT}/logs/%x-%j.out" \
      --export="ALL,DUCA_PAPER_SEED=${seed}" \
      --wrap="/bin/bash ${DUCA_PAPER_REPO_ROOT}/scripts/run_duca_paper_stage_a_seed.sh"
  )"
  job_ids+=("${job_id%%;*}")
done
[[ "${#job_ids[@]}" == 3 ]] || fail "Stage-A did not create three seed groups"

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
  "${DUCA_PAPER_CELLS_ROOT}" \
  "${seal_job}" \
  "${job_ids[@]:0:3}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output, commit, protocol, protocol_sha, cells_root, seal_job, *seed_jobs = sys.argv[1:]
arms = (
    "dense",
    "uniform_fixed_k384",
    "uniform_mixed_train_k384_eval",
    "duca_fixed_k384",
)
seeds = (5801, 8123, 12011)
if len(seed_jobs) != 3:
    raise SystemExit("grouped Stage-A submission cardinality drift")
payload = {
    "schema_version": "duca_paper_stage_a_grouped_submission_v1",
    "status": "held_complete",
    "git_commit": commit,
    "protocol_manifest_path": str(pathlib.Path(protocol).resolve()),
    "protocol_manifest_sha256": protocol_sha,
    "cells_root": str(pathlib.Path(cells_root).resolve()),
    "logical_cell_count": 12,
    "scheduler_job_count": 4,
    "sequential_scheduler_grouping_only": True,
    "seed_jobs": [
        {
            "seed": seed,
            "slurm_job_id": job_id,
            "arms": list(arms),
        }
        for seed, job_id in zip(seeds, seed_jobs)
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
    "schema_version": "duca_paper_stage_a_grouped_submission_receipt_v1",
    "status": "held_complete",
    "git_commit": commit,
    "submission_manifest_path": str(target.resolve()),
    "submission_manifest_sha256": digest,
    "seed_job_ids": seed_jobs,
    "seal_job_id": seal_job,
    "released": False,
}
pathlib.Path(str(target) + ".receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf '%s\n' "${job_ids[@]:0:3}" > "${DUCA_PAPER_RUN_ROOT}/seed_job_ids.txt"
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
trap - ERR INT TERM

echo "[DUCA_PAPER_STAGE_A_GROUPED_SUBMIT] RELEASED 3 seed groups; seal ${seal_job}"
