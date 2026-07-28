#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_SUBMIT][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE4_AUTHORIZATION \
  DUCA_RIME_PHASE4_AUTHORIZATION_SHA256 \
  DUCA_RIME_PHASE4_CELLS_ROOT \
  DUCA_RIME_PHASE4_SUBMISSION_ROOT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_PHASE2_PROTOCOL_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_PHASE3_ASSET_RECEIPT \
  DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_DENSE_CONFIG_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER \
  DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER \
  DUCA_RIME_DENSE_CONFIG_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_TRIDET \
  DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET; do
  required "${name}"
  export "${name}"
done

if [[ -n "${SLURM_JOB_ID:-}" && "${DUCA_RIME_SUBMIT_CONTROLLER:-0}" != 1 ]]; then
  fail "matrix submission requires a login node or the registered submit controller"
fi
command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v scontrol >/dev/null || fail "scontrol is unavailable"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE4_CELLS_ROOT}" ]] \
  || fail "a fresh Phase-4 cells root is required"
[[ ! -e "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}" ]] \
  || fail "a fresh submission root is required"
[[ "$(sha256sum "${DUCA_RIME_PHASE4_AUTHORIZATION}" | awk '{print $1}')" == "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" ]] \
  || fail "Phase-4 authorization SHA-256 drift"
for binding in \
  "${DUCA_RIME_PHASE2_RECEIPT}|${DUCA_RIME_PHASE2_RECEIPT_SHA256}|Phase-2 receipt" \
  "${DUCA_RIME_SPLIT_MANIFEST}|${DUCA_RIME_SPLIT_MANIFEST_SHA256}|split manifest" \
  "${DUCA_RIME_TARGETS_JSONL}|${DUCA_RIME_TARGETS_SHA256}|training targets" \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT}|${DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256}|Phase-3 asset receipt" \
  "${DUCA_RIME_PRETRAIN_PATH}|${DUCA_RIME_PRETRAIN_SHA256}|VideoMAE pretrain" \
  "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER}|${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER}|ActionFormer dense evidence" \
  "${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET}|${DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_TRIDET}|TriDet dense evidence"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done
for path in \
  "${DUCA_RIME_DENSE_CONFIG_ACTIONFORMER}" \
  "${DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER}" \
  "${DUCA_RIME_DENSE_CONFIG_TRIDET}" \
  "${DUCA_RIME_DENSE_CHECKPOINT_TRIDET}"; do
  [[ -f "${path}" ]] || fail "dense dependency is missing: ${path}"
done

python - \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT}" \
  "${DUCA_RIME_TARGETS_SHA256}" <<'PY'
import json
import hashlib
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
asset = json.load(open(sys.argv[5], encoding="utf-8"))
if (
    row.get("schema_version") != "duca_rime_stage_receipt_v1"
    or row.get("phase") != "phase4_authorization"
    or row.get("status") != "authorized"
    or row.get("gate_pass") is not True
    or row.get("git_commit") != sys.argv[2]
    or row.get("formal_seeds") != [5801, 8123, 12011]
    or row.get("required_detectors") != ["ActionFormer", "TriDet"]
    or row.get("required_budget_panels") != [384, 192]
    or row.get("official_final_subset_consumed") is not False
    or row.get("phase2_receipt", {}).get("sha256") != sys.argv[3]
    or row.get("split_manifest", {}).get("sha256") != sys.argv[4]
    or not row.get("official_final_video_ids")
    or hashlib.sha256(
        json.dumps(
            row["official_final_video_ids"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest() != row.get("official_final_video_ids_sha256")
    or asset.get("schema_version") != "duca_rime_phase3_asset_receipt_v1"
    or asset.get("status") != "passed"
    or asset.get("git_commit") != sys.argv[2]
    or asset.get("artifacts", {}).get("training_targets", {}).get("sha256")
    != sys.argv[6]
):
    raise SystemExit("Phase-4 authorization is not the frozen 12-cell contract")
PY

mkdir -p \
  "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/logs" \
  "$(dirname "${DUCA_RIME_PHASE4_CELLS_ROOT}")"

job_ids=()
cleanup_held_jobs() {
  if [[ "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_held_jobs ERR INT TERM

for backend in ActionFormer TriDet; do
  for target in 384 192; do
    for seed in 5801 8123 12011; do
      cell_root="${DUCA_RIME_PHASE4_CELLS_ROOT}/${backend}/K${target}/seed${seed}"
      job_id="$(
        sbatch \
          --parsable \
          --hold \
          --partition=gpu \
          --gres=gpu:1 \
          --cpus-per-task="${DUCA_RIME_PHASE4_CPUS:-8}" \
          --time="${DUCA_RIME_PHASE4_TIME:-7-00:00:00}" \
          --job-name="rime4-${backend:0:2}-k${target}-s${seed}" \
          --output="${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/logs/%x-%j.out" \
          --export="ALL,DUCA_RIME_PHASE4_BACKEND=${backend},DUCA_RIME_PHASE4_TARGET=${target},DUCA_RIME_PHASE4_SEED=${seed},DUCA_RIME_PHASE4_CELL_ROOT=${cell_root}" \
          scripts/run_duca_rime_phase4_cell_pipeline.sh
      )"
      job_ids+=("${job_id%%;*}")
    done
  done
done
[[ "${#job_ids[@]}" == 12 ]] || fail "matrix did not create exactly 12 held cells"

dependency="$(IFS=:; echo "${job_ids[*]}")"
export DUCA_RIME_PHASE4_MATRIX_ROOT="${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/matrix"
seal_job="$(
  sbatch \
    --parsable \
    --hold \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --time=02:00:00 \
    --job-name=rime4-seal \
    --dependency="afterok:${dependency}" \
    --output="${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/logs/%x-%j.out" \
    --export="ALL,DUCA_RIME_PHASE4_MATRIX_ROOT=${DUCA_RIME_PHASE4_MATRIX_ROOT}" \
    scripts/run_duca_rime_phase4_seal_matrix.sh
)"
seal_job="${seal_job%%;*}"
job_ids+=("${seal_job}")

python - \
  "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION}" \
  "${DUCA_RIME_PHASE4_AUTHORIZATION_SHA256}" \
  "${DUCA_RIME_PHASE4_CELLS_ROOT}" \
  "${seal_job}" \
  "${job_ids[@]:0:12}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    output,
    commit,
    authorization,
    authorization_sha,
    cells_root,
    seal_job,
    *cell_jobs,
) = sys.argv[1:]
backends = ("ActionFormer", "TriDet")
targets = (384, 192)
seeds = (5801, 8123, 12011)
coordinates = [
    {"backend": backend, "target_mean_cost": target, "seed": seed}
    for backend in backends
    for target in targets
    for seed in seeds
]
if len(cell_jobs) != 12 or len(coordinates) != 12:
    raise SystemExit("submission manifest cardinality drift")
payload = {
    "schema_version": "duca_rime_phase4_submission_v1",
    "status": "held_complete",
    "git_commit": commit,
    "authorization_path": str(pathlib.Path(authorization).resolve()),
    "authorization_sha256": authorization_sha,
    "cells_root": str(pathlib.Path(cells_root).resolve()),
    "cell_count": 12,
    "cells": [
        {**coordinate, "slurm_job_id": job_id}
        for coordinate, job_id in zip(coordinates, cell_jobs)
    ],
    "seal_job_id": seal_job,
    "release_is_transactional": True,
}
target = pathlib.Path(output)
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
atomic_write(
    str(target) + ".sha256",
    f"{digest}  {target.name}\n",
)
receipt = {
    "schema_version": "duca_rime_phase4_submission_receipt_v1",
    "status": "held_complete",
    "git_commit": commit,
    "submission_manifest_path": str(target.resolve()),
    "submission_manifest_sha256": digest,
    "cell_job_ids": cell_jobs,
    "seal_job_id": seal_job,
    "released": False,
}
atomic_write(
    str(target) + ".receipt.json",
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
)
PY
[[ "$(sha256sum "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json" | awk '{print $1}')" == \
  "$(awk '{print $1}' "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json.sha256")" ]] \
  || fail "submission manifest verification failed before release"
[[ -f "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json.receipt.json" ]] \
  || fail "submission receipt is missing before release"

cell_ids_tmp="${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/.cell_job_ids.txt.partial.$$"
seal_id_tmp="${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/.seal_job_id.txt.partial.$$"
printf '%s\n' "${job_ids[@]:0:12}" > "${cell_ids_tmp}"
printf '%s\n' "${seal_job}" > "${seal_id_tmp}"
mv "${cell_ids_tmp}" "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/cell_job_ids.txt"
mv "${seal_id_tmp}" "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/seal_job_id.txt"
release_list="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "${release_list}"
trap - ERR INT TERM
python - "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json.receipt.json" <<'PY'
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

echo "[DUCA_RIME_PHASE4_SUBMIT] RELEASED 12 cells; seal job ${seal_job}"
