#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_SUBMIT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
RUN_ROOT="${RUN_ROOT:-}"
SUBMIT_COST="${SUBMIT_COST:-1}"
[[ -d "${RUN_ROOT}" ]] || fail "RUN_ROOT must name a prepared suite"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
[[ -f "${MANIFEST}" ]] || fail "suite manifest is missing"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "submission requires a clean tree"
command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v flock >/dev/null 2>&1 || fail "flock is unavailable"

readarray -t binding < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("schema") != "duca_cellcf_suite_manifest_v1" or p.get("ok") is not True:
    raise SystemExit("invalid CellCF suite manifest")
print(p["git_commit"])
print(p["seed"])
print(p["real_loader_gate"]["path"])
print(p["ddp_pilot"]["path"])
PY
)
EXPECTED_COMMIT="${binding[0]}"
SEED="${binding[1]}"
GATE_JSON="${binding[2]}"
PILOT_JSON="${binding[3]}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from suite manifest"

RECEIPT_DIR="${RUN_ROOT}/submission_receipts"
mkdir -p "${RECEIPT_DIR}"
exec 9>"${RECEIPT_DIR}/submit.lock"
flock -n 9 || fail "another submission process holds the suite lock"

submit_once() {
  local name="$1"
  local job_file="$2"
  local dependency="${3:-}"
  local receipt="${RECEIPT_DIR}/${name}.json"
  if [[ -f "${receipt}" ]]; then
    "${PYTHON}" - "${receipt}" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
job_id = p.get("job_id")
if not isinstance(job_id, int) or job_id <= 0:
    raise SystemExit("invalid existing submission receipt")
print(job_id)
PY
    return
  fi
  local args=(--parsable --comment="cellcf-${EXPECTED_COMMIT:0:12}-seed${SEED}-${name}")
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="${dependency}")
  fi
  local raw job_id
  raw="$(sbatch "${args[@]}" "${job_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: ${raw}"
  "${PYTHON}" - "${receipt}" "${name}" "${job_file}" "${job_id}" "${raw}" "${EXPECTED_COMMIT}" "${SEED}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

out, name, job_file, job_id, raw, commit, seed = sys.argv[1:]
payload = {
    "schema": "duca_cellcf_slurm_submission_v1",
    "name": name,
    "job_file": str(Path(job_file).resolve()),
    "job_id": int(job_id),
    "raw_sbatch_response": raw,
    "git_commit": commit,
    "seed": int(seed),
}
target = Path(out)
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
  printf '%s\n' "${job_id}"
}

variants=(uniform transition_beta0 cellcf)
job_ids=()
for variant in "${variants[@]}"; do
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  [[ -f "${job_file}" ]] || fail "missing prepared job: ${job_file}"
  job_ids+=("$(submit_once "${variant}" "${job_file}")")
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
aggregate_job="${RUN_ROOT}/jobs/aggregate.sbatch"
cat > "${aggregate_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=cellcf-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=${RUN_ROOT}/logs/aggregate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/aggregate-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export BASE='${BASE}'
'${PYTHON}' -m tools.bata.validate_duca_cellcf_suite \
  --repo-root '${REPO_ROOT}' --seed '${SEED}' \
  --expected-commit '${EXPECTED_COMMIT}' --require-clean \
  --gate-json '${GATE_JSON}' --pilot-json '${PILOT_JSON}' \
  --post-run-evidence 'uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  --post-run-evidence 'transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  --post-run-evidence 'cellcf=${RUN_ROOT}/logs/cellcf/post_run_evidence.json' \
  --output-json '${RUN_ROOT}/final_suite_evidence.json'
EOF
chmod 0755 "${aggregate_job}"
bash -n "${aggregate_job}"
aggregate_id="$(submit_once aggregate "${aggregate_job}" "afterok:${dependency}")"

cost_id=""
if [[ "${SUBMIT_COST}" == "1" ]]; then
  cost_job="${RUN_ROOT}/jobs/cost.sbatch"
  cat > "${cost_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=cellcf-cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}/logs/cost-%j.out
#SBATCH --error=${RUN_ROOT}/logs/cost-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CELLCF_CHECKPOINT='${RUN_ROOT}/work_dirs/cellcf/gpu1_id0/checkpoint/epoch_131.pth'
export DUCA_CELLCF_COST_ROOT='${RUN_ROOT}/cost'
bash scripts/run_duca_cellcf_cost_pair.sh
EOF
  chmod 0755 "${cost_job}"
  bash -n "${cost_job}"
  cost_id="$(submit_once cost "${cost_job}" "afterok:${job_ids[2]}")"
fi

printf 'variant\tseed\tcommit\tsbatch_file\tjob_id\tstatus\n' > "${RUN_ROOT}/jobs.submitted.tsv"
for index in "${!variants[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\tSUBMITTED\n' \
    "${variants[$index]}" "${SEED}" "${EXPECTED_COMMIT}" \
    "${RUN_ROOT}/jobs/${variants[$index]}.sbatch" "${job_ids[$index]}" \
    >> "${RUN_ROOT}/jobs.submitted.tsv"
done
printf 'aggregate\t%s\t%s\t%s\t%s\tDEPENDENCY_SUBMITTED\n' \
  "${SEED}" "${EXPECTED_COMMIT}" "${aggregate_job}" "${aggregate_id}" \
  >> "${RUN_ROOT}/jobs.submitted.tsv"
if [[ -n "${cost_id}" ]]; then
  printf 'cost\t%s\t%s\t%s\t%s\tDEPENDENCY_SUBMITTED\n' \
    "${SEED}" "${EXPECTED_COMMIT}" "${RUN_ROOT}/jobs/cost.sbatch" "${cost_id}" \
    >> "${RUN_ROOT}/jobs.submitted.tsv"
fi
echo "[DUCA_CELLCF_SUBMIT] train=${job_ids[*]} aggregate=${aggregate_id} cost=${cost_id:-disabled} root=${RUN_ROOT}"
