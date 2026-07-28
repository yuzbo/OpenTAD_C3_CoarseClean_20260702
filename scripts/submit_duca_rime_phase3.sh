#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_SUBMIT][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE3_BUNDLE_ROOT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON \
  DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_BUDGET_PROTOCOL_JSON \
  DUCA_RIME_BUDGET_PROTOCOL_SHA256 \
  DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL \
  DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256 \
  DUCA_RIME_ADAPTOK_REPLAY_JSONL \
  DUCA_RIME_ADAPTOK_REPLAY_SHA256 \
  DUCA_RIME_PHASE3_ASSET_RECEIPT \
  DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256 \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_DENSE_CONFIG_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER \
  DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER; do
  required "${name}"
  export "${name}"
done

if [[ -n "${SLURM_JOB_ID:-}" && "${DUCA_RIME_SUBMIT_CONTROLLER:-0}" != 1 ]]; then
  fail "Phase-3 submission requires a login node or the registered submit controller"
fi
command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v scontrol >/dev/null || fail "scontrol is unavailable"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_RIME_PHASE3_BUNDLE_ROOT}" ]] \
  || fail "a fresh Phase-3 bundle root is required"

[[ "$(sha256sum "${DUCA_RIME_SPLIT_MANIFEST}" | awk '{print $1}')" == "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" ]] \
  || fail "split manifest SHA-256 drift"
for binding in \
  "${DUCA_RIME_PHASE2_RECEIPT}|${DUCA_RIME_PHASE2_RECEIPT_SHA256}|Phase-2 receipt" \
  "${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON}|${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256}|training exposure" \
  "${DUCA_RIME_PRETRAIN_PATH}|${DUCA_RIME_PRETRAIN_SHA256}|VideoMAE pretrain" \
  "${DUCA_RIME_TARGETS_JSONL}|${DUCA_RIME_TARGETS_SHA256}|training targets" \
  "${DUCA_RIME_BUDGET_PROTOCOL_JSON}|${DUCA_RIME_BUDGET_PROTOCOL_SHA256}|budget protocol" \
  "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL}|${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256}|D-shuffle training replay" \
  "${DUCA_RIME_ADAPTOK_REPLAY_JSONL}|${DUCA_RIME_ADAPTOK_REPLAY_SHA256}|AdapTok replay" \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT}|${DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256}|asset receipt"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done
python - \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_TARGETS_SHA256}" \
  "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256}" \
  "${DUCA_RIME_ADAPTOK_REPLAY_SHA256}" \
  "${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256}" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "training_targets": sys.argv[3],
    "dshuffle_training_replay": sys.argv[4],
    "adaptok_replay": sys.argv[5],
    "training_exposure": sys.argv[6],
}
if (
    row.get("schema_version") != "duca_rime_phase3_asset_receipt_v1"
    or row.get("status") != "passed"
    or row.get("git_commit") != sys.argv[2]
    or row.get("official_final_subset_consumed") is not False
    or any(
        row.get("artifacts", {}).get(name, {}).get("sha256") != digest
        for name, digest in expected.items()
    )
):
    raise SystemExit("Phase-3 asset receipt does not bind the submitted assets")
PY
readarray -t split_values < <(python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    row = manifest["train_roles"][role]
    print(row["block_list_path"])
    print(row["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] || fail "failed to resolve train/development roles"
[[ "$(sha256sum "${split_values[0]}" | awk '{print $1}')" == "${split_values[1]}" ]] \
  || fail "train block-list SHA-256 drift"
[[ "$(sha256sum "${split_values[2]}" | awk '{print $1}')" == "${split_values[3]}" ]] \
  || fail "development block-list SHA-256 drift"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"
export DUCA_RIME_TARGET_MEAN_COST=384
export DUCA_RIME_FIXED_BUDGET=384

mkdir -p "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/logs"
arms=(RIME-full U-fixed F-bound D-no-risk AdapTok-TAD D-shuffle)
job_ids=()
declare -A arm_jobs
cleanup_held_jobs() {
  if [[ "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_held_jobs ERR INT TERM

for arm in "${arms[@]}"; do
  name="$(printf '%s' "${arm}" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
  dependency_args=()
  if [[ "${arm}" == D-shuffle ]]; then
    dependency_args=(--dependency="afterok:${arm_jobs[RIME-full]}")
  fi
  job="$(
    sbatch \
      --parsable \
      --hold \
      --partition=gpu \
      --gres=gpu:1 \
      --cpus-per-task="${DUCA_RIME_PHASE3_CPUS:-8}" \
      --time="${DUCA_RIME_PHASE3_TIME:-4-00:00:00}" \
      --job-name="rime3-${name}" \
      --output="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/logs/%x-%j.out" \
      "${dependency_args[@]}" \
      --export="ALL,DUCA_RIME_PHASE3_ARM=${arm}" \
      scripts/run_duca_rime_phase3_arm_pipeline.sh
  )"
  job="${job%%;*}"
  job_ids+=("${job}")
  arm_jobs["${arm}"]="${job}"
done

cost_job="$(
  sbatch \
    --parsable \
    --hold \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task="${DUCA_RIME_PHASE3_CPUS:-8}" \
    --time=08:00:00 \
    --job-name=rime3-cost \
    --dependency="afterok:${arm_jobs[U-fixed]}:${arm_jobs[RIME-full]}" \
    --output="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/logs/%x-%j.out" \
    --export=ALL \
    scripts/run_duca_rime_phase3_cost_from_roots.sh
)"
cost_job="${cost_job%%;*}"
job_ids+=("${cost_job}")

all_dependencies="$(IFS=:; echo "${job_ids[*]}")"
seal_root="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/seal"
seal_exports="ALL"
for arm in U-fixed U-same-K F-bound D-shuffle D-no-risk AdapTok-TAD RIME-full; do
  key="$(printf '%s' "${arm}" | tr '[:lower:]-' '[:upper:]_')"
  seal_exports+=",DUCA_RIME_PHASE3_${key}_EVAL_ROOT=${DUCA_RIME_PHASE3_BUNDLE_ROOT}/arms/${key}/eval"
done
seal_exports+=",DUCA_RIME_PHASE3_SEAL_ROOT=${seal_root}"
seal_exports+=",DUCA_RIME_PHASE3_COST_EVIDENCE=${DUCA_RIME_PHASE3_BUNDLE_ROOT}/cost/paired_cost.json"
seal_job="$(
  sbatch \
    --parsable \
    --hold \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --time=02:00:00 \
    --job-name=rime3-seal \
    --dependency="afterok:${all_dependencies}" \
    --output="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/logs/%x-%j.out" \
    --export="${seal_exports}" \
    --wrap="export DUCA_RIME_PHASE3_COST_EVIDENCE_SHA256=\$(sha256sum \"${DUCA_RIME_PHASE3_BUNDLE_ROOT}/cost/paired_cost.json\" | awk '{print \$1}'); exec scripts/run_duca_rime_phase3_seal.sh"
)"
seal_job="${seal_job%%;*}"
job_ids+=("${seal_job}")

python - \
  "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${cost_job}" \
  "${seal_job}" \
  "${arms[@]}" \
  -- "${job_ids[@]:0:6}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output, commit, cost_job, seal_job, *rest = sys.argv[1:]
separator = rest.index("--")
arms = rest[:separator]
jobs = rest[separator + 1 :]
if len(arms) != 6 or len(jobs) != 6:
    raise SystemExit("Phase-3 submission cardinality drift")
payload = {
    "schema_version": "duca_rime_phase3_submission_v1",
    "status": "held_complete",
    "git_commit": commit,
    "training_job_count": 6,
    "u_same_k_training_job_count": 0,
    "arms": [
        {"arm": arm, "slurm_job_id": job}
        for arm, job in zip(arms, jobs)
    ],
    "cost_job_id": cost_job,
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
    "schema_version": "duca_rime_phase3_submission_receipt_v1",
    "status": "held_complete",
    "git_commit": commit,
    "submission_manifest_path": str(target.resolve()),
    "submission_manifest_sha256": digest,
    "training_job_ids": jobs,
    "cost_job_id": cost_job,
    "seal_job_id": seal_job,
    "released": False,
}
atomic_write(
    str(target) + ".receipt.json",
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
)
PY
[[ "$(sha256sum "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json" | awk '{print $1}')" == \
  "$(awk '{print $1}' "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json.sha256")" ]] \
  || fail "submission manifest verification failed before release"
[[ -f "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json.receipt.json" ]] \
  || fail "submission receipt is missing before release"

job_ids_tmp="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/.job_ids.txt.partial.$$"
printf '%s\n' "${job_ids[@]}" > "${job_ids_tmp}"
mv "${job_ids_tmp}" "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/job_ids.txt"
release_list="$(IFS=,; echo "${job_ids[*]}")"
scontrol release "${release_list}"
trap - ERR INT TERM
python - "${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json.receipt.json" <<'PY'
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
echo "[DUCA_RIME_PHASE3_SUBMIT] RELEASED 6 training arms; seal job ${seal_job}"
