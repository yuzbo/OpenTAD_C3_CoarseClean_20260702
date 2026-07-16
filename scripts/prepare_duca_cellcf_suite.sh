#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_PREPARE][FAIL] $*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

resolve_target_cluster() {
  local cluster="${DUCA_CELLCF_TARGET_CLUSTER:-${SLURM_CLUSTER_NAME:-}}"
  if [[ -z "${cluster}" ]]; then
    command -v scontrol >/dev/null 2>&1 || fail \
      "set DUCA_CELLCF_TARGET_CLUSTER or run preparation where scontrol is available"
    cluster="$(scontrol show config | awk -F= \
      '/^[[:space:]]*ClusterName[[:space:]]*=/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
  fi
  [[ "${cluster}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid Slurm cluster identity: ${cluster}"
  printf '%s\n' "${cluster}"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
SEED="${SEED:-0}"
CURRENT_HEAD="$(git rev-parse HEAD)"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_cellcf_${CURRENT_HEAD:0:7}_seed${SEED}}"
GATE_JSON="${DUCA_CELLCF_GATE_JSON:-}"
PILOT_JSON="${DUCA_CELLCF_DDP_PILOT_JSON:-}"
TARGET_CLUSTER="$(resolve_target_cluster)"

[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "suite preparation requires a clean tree"
[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -f "${GATE_JSON}" ]] || fail "real-loader gate JSON is missing"
[[ -f "${PILOT_JSON}" ]] || fail "DDP pilot JSON is missing"
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"
[[ "${DUCA_OFFICIAL_ADATAD_CHECKPOINT_INTERVAL}" == "5" ]] || fail \
  "formal CellCF training must preserve checkpoint-every-5"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/work_dirs"
CANONICAL_ENV_FILE="${RUN_ROOT}/canonical_env.tsv"
duca_cellcf_canonical_env_payload > "${CANONICAL_ENV_FILE}"
CANONICAL_ENV_SHA256="$(sha256_file "${CANONICAL_ENV_FILE}")"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
"${PYTHON}" -m tools.bata.validate_duca_cellcf_suite \
  --repo-root "${REPO_ROOT}" --seed "${SEED}" \
  --expected-commit "${EXPECTED_COMMIT}" --require-clean \
  --gate-json "${GATE_JSON}" --pilot-json "${PILOT_JSON}" \
  --output-json "${MANIFEST}"
MANIFEST_SHA256="$(sha256_file "${MANIFEST}")"
GATE_SHA256="$(sha256_file "${GATE_JSON}")"
PILOT_SHA256="$(sha256_file "${PILOT_JSON}")"

AGGREGATE_EVIDENCE="${RUN_ROOT}/aggregate_suite_evidence.json"
COST_EVIDENCE="${RUN_ROOT}/cost/cellcf_vs_bare_uniform.json"
FINAL_EVIDENCE="${RUN_ROOT}/final_suite_evidence.json"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"
variants=(uniform transition_beta0 cellcf)
arm_job_names=(
  "cellcf-uniform-s${SEED}-${SHORT_COMMIT}"
  "cellcf-transition_beta0-s${SEED}-${SHORT_COMMIT}"
  "cellcf-cellcf-s${SEED}-${SHORT_COMMIT}"
)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  job_name="${arm_job_names[$index]}"
  readarray -t binding < <("${PYTHON}" - "${MANIFEST}" "${variant}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
variant = next(item for item in payload["variants"] if item["name"] == sys.argv[2])
data = payload["reference_data_artifacts"]
for value in (
    variant["resolved_config_sha256"],
    payload["shared_protocol_sha256"],
    payload["ordered_exposure_sha256"],
    data["evaluation_annotation_path"],
    data["evaluation_annotation_sha256"],
    data["evaluation_class_map_path"],
    data["evaluation_class_map_sha256"],
    data["evaluation_config_sha256"],
):
    print(value)
PY
)
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
# DUCA_CELLCF_DEPENDENCY_ROLE=none
set -euo pipefail
cd '${REPO_ROOT}'
[[ "\$(sha256sum '${MANIFEST}' | awk '{print \$1}')" == '${MANIFEST_SHA256}' ]] || { echo '[DUCA_CELLCF_JOB][FAIL] suite manifest hash drift' >&2; exit 1; }
[[ "\$(git rev-parse HEAD)" == '${EXPECTED_COMMIT}' ]] || { echo '[DUCA_CELLCF_JOB][FAIL] commit drift' >&2; exit 1; }
[[ "\${SLURM_CLUSTER_NAME:-}" == '${TARGET_CLUSTER}' ]] || { echo '[DUCA_CELLCF_JOB][FAIL] Slurm cluster drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${job_name}' ]] || { echo '[DUCA_CELLCF_JOB][FAIL] Slurm job-name drift' >&2; exit 1; }
export BASE='${BASE}'
export DUCA_CELLCF_VARIANT='${variant}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CELLCF_GATE_JSON='${GATE_JSON}'
export DUCA_CELLCF_DDP_PILOT_JSON='${PILOT_JSON}'
export DUCA_CELLCF_GATE_SHA256='${GATE_SHA256}'
export DUCA_CELLCF_DDP_PILOT_SHA256='${PILOT_SHA256}'
export DUCA_CELLCF_RESOLVED_CONFIG_SHA256='${binding[0]}'
export DUCA_CELLCF_PROTOCOL_SHA256='${binding[1]}'
export DUCA_CELLCF_ORDER_SHA256='${binding[2]}'
export DUCA_CELLCF_ANNOTATION_PATH='${binding[3]}'
export DUCA_CELLCF_ANNOTATION_SHA256='${binding[4]}'
export DUCA_CELLCF_CLASS_MAP_PATH='${binding[5]}'
export DUCA_CELLCF_CLASS_MAP_SHA256='${binding[6]}'
export DUCA_CELLCF_EVALUATION_CONFIG_SHA256='${binding[7]}'
export DUCA_CELLCF_CANONICAL_ENV_FILE='${CANONICAL_ENV_FILE}'
export DUCA_CELLCF_CANONICAL_ENV_SHA256='${CANONICAL_ENV_SHA256}'
export DUCA_CELLCF_SUITE_MANIFEST='${MANIFEST}'
export DUCA_CELLCF_SUITE_MANIFEST_SHA256='${MANIFEST_SHA256}'
export DUCA_CELLCF_TARGET_CLUSTER='${TARGET_CLUSTER}'
export SEED='${SEED}'
export RUN_DIR='${RUN_ROOT}/logs/${variant}'
export WORK_DIR='${RUN_ROOT}/work_dirs/${variant}'
bash scripts/run_duca_cellcf_variant.sh
EOF
  chmod 0755 "${job_file}"
done

aggregate_job_name="cellcf-aggregate-s${SEED}-${SHORT_COMMIT}"
aggregate_job="${RUN_ROOT}/jobs/aggregate.sbatch"
cat > "${aggregate_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${aggregate_job_name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=${RUN_ROOT}/logs/aggregate-%j.out
#SBATCH --error=${RUN_ROOT}/logs/aggregate-%j.err
# DUCA_CELLCF_DEPENDENCY_ROLE=afterok_three_arms
set -euo pipefail
cd '${REPO_ROOT}'
[[ "\$(sha256sum '${MANIFEST}' | awk '{print \$1}')" == '${MANIFEST_SHA256}' ]] || { echo '[DUCA_CELLCF_AGGREGATE][FAIL] suite manifest hash drift' >&2; exit 1; }
[[ "\$(git rev-parse HEAD)" == '${EXPECTED_COMMIT}' ]] || { echo '[DUCA_CELLCF_AGGREGATE][FAIL] commit drift' >&2; exit 1; }
[[ "\${SLURM_CLUSTER_NAME:-}" == '${TARGET_CLUSTER}' ]] || { echo '[DUCA_CELLCF_AGGREGATE][FAIL] Slurm cluster drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${aggregate_job_name}' ]] || { echo '[DUCA_CELLCF_AGGREGATE][FAIL] Slurm job-name drift' >&2; exit 1; }
export BASE='${BASE}'
'${PYTHON}' -m tools.bata.validate_duca_cellcf_suite \
  --repo-root '${REPO_ROOT}' --seed '${SEED}' \
  --expected-commit '${EXPECTED_COMMIT}' --require-clean \
  --gate-json '${GATE_JSON}' --pilot-json '${PILOT_JSON}' \
  --post-run-evidence 'uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  --post-run-evidence 'transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  --post-run-evidence 'cellcf=${RUN_ROOT}/logs/cellcf/post_run_evidence.json' \
  --output-json '${AGGREGATE_EVIDENCE}'
EOF
chmod 0755 "${aggregate_job}"

cost_job_name="cellcf-cost-s${SEED}-${SHORT_COMMIT}"
cost_job="${RUN_ROOT}/jobs/cost.sbatch"
cat > "${cost_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${cost_job_name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}/logs/cost-%j.out
#SBATCH --error=${RUN_ROOT}/logs/cost-%j.err
# DUCA_CELLCF_DEPENDENCY_ROLE=afterok_aggregate
set -euo pipefail
cd '${REPO_ROOT}'
[[ "\$(sha256sum '${MANIFEST}' | awk '{print \$1}')" == '${MANIFEST_SHA256}' ]] || { echo '[DUCA_CELLCF_COST_JOB][FAIL] suite manifest hash drift' >&2; exit 1; }
[[ "\$(git rev-parse HEAD)" == '${EXPECTED_COMMIT}' ]] || { echo '[DUCA_CELLCF_COST_JOB][FAIL] commit drift' >&2; exit 1; }
[[ "\${SLURM_CLUSTER_NAME:-}" == '${TARGET_CLUSTER}' ]] || { echo '[DUCA_CELLCF_COST_JOB][FAIL] Slurm cluster drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${cost_job_name}' ]] || { echo '[DUCA_CELLCF_COST_JOB][FAIL] Slurm job-name drift' >&2; exit 1; }
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CELLCF_CHECKPOINT='${RUN_ROOT}/work_dirs/cellcf/gpu1_id0/checkpoint/epoch_131.pth'
export DUCA_CELLCF_COST_ROOT='${RUN_ROOT}/cost'
export DUCA_CELLCF_SUITE_MANIFEST='${MANIFEST}'
export DUCA_CELLCF_SUITE_MANIFEST_SHA256='${MANIFEST_SHA256}'
export DUCA_CELLCF_AGGREGATE_EVIDENCE='${AGGREGATE_EVIDENCE}'
export DUCA_CELLCF_COST_EVIDENCE='${COST_EVIDENCE}'
export DUCA_CELLCF_TARGET_CLUSTER='${TARGET_CLUSTER}'
export SEED='${SEED}'
CELLCF_POST_RUN_EVIDENCE='${RUN_ROOT}/logs/cellcf/post_run_evidence.json'
readarray -t cost_binding < <('${PYTHON}' - '${AGGREGATE_EVIDENCE}' \
  "\${CELLCF_POST_RUN_EVIDENCE}" '${EXPECTED_COMMIT}' '${SEED}' \
  '${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  '${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' <<'PY'
import hashlib
import json
import sys
from pathlib import Path

aggregate_path = Path(sys.argv[1]).resolve()
cellcf_path = Path(sys.argv[2]).resolve()
commit = sys.argv[3]
seed = int(sys.argv[4])
expected_paths = {
    "uniform": Path(sys.argv[5]).resolve(),
    "transition_beta0": Path(sys.argv[6]).resolve(),
    "cellcf": cellcf_path,
}
payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
if payload.get("schema") != "duca_cellcf_suite_manifest_v1" or payload.get("ok") is not True:
    raise SystemExit("aggregate evidence has an invalid schema or status")
if payload.get("status") != "runs_complete_cost_pending":
    raise SystemExit("aggregate evidence is not the expected cost-pending artifact")
if payload.get("git_commit") != commit or payload.get("seed") != seed:
    raise SystemExit("aggregate evidence commit or seed mismatch")
completed = payload.get("completed_runs")
if not isinstance(completed, dict) or set(completed) != set(expected_paths):
    raise SystemExit("aggregate evidence does not cover exactly three CellCF arms")
for variant, expected_path in expected_paths.items():
    record = completed[variant]
    actual_path = Path(str(record.get("path") or "")).resolve()
    if actual_path != expected_path or not actual_path.is_file():
        raise SystemExit(f"aggregate {variant} post-run path mismatch")
    actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if record.get("sha256") != actual_sha:
        raise SystemExit(f"aggregate {variant} post-run hash mismatch")
    if variant == "cellcf":
        print(actual_sha)
PY
)
[[ "\${#cost_binding[@]}" == "1" && "\${cost_binding[0]}" =~ ^[0-9a-f]{64}$ ]] || { echo '[DUCA_CELLCF_COST_JOB][FAIL] aggregate did not bind CellCF post-run evidence' >&2; exit 1; }
export DUCA_CELLCF_POST_RUN_EVIDENCE_JSON="\${CELLCF_POST_RUN_EVIDENCE}"
export DUCA_CELLCF_POST_RUN_EVIDENCE_SHA256="\${cost_binding[0]}"
bash scripts/run_duca_cellcf_cost_pair.sh
EOF
chmod 0755 "${cost_job}"

completion_job_name="cellcf-completion-s${SEED}-${SHORT_COMMIT}"
completion_job="${RUN_ROOT}/jobs/completion.sbatch"
cat > "${completion_job}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${completion_job_name}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=${RUN_ROOT}/logs/completion-%j.out
#SBATCH --error=${RUN_ROOT}/logs/completion-%j.err
# DUCA_CELLCF_DEPENDENCY_ROLE=afterok_aggregate_and_cost
set -euo pipefail
cd '${REPO_ROOT}'
[[ "\$(sha256sum '${MANIFEST}' | awk '{print \$1}')" == '${MANIFEST_SHA256}' ]] || { echo '[DUCA_CELLCF_COMPLETION][FAIL] suite manifest hash drift' >&2; exit 1; }
[[ "\$(git rev-parse HEAD)" == '${EXPECTED_COMMIT}' ]] || { echo '[DUCA_CELLCF_COMPLETION][FAIL] commit drift' >&2; exit 1; }
[[ "\${SLURM_CLUSTER_NAME:-}" == '${TARGET_CLUSTER}' ]] || { echo '[DUCA_CELLCF_COMPLETION][FAIL] Slurm cluster drift' >&2; exit 1; }
[[ "\${SLURM_JOB_NAME:-}" == '${completion_job_name}' ]] || { echo '[DUCA_CELLCF_COMPLETION][FAIL] Slurm job-name drift' >&2; exit 1; }
export BASE='${BASE}'
'${PYTHON}' - '${AGGREGATE_EVIDENCE}' '${EXPECTED_COMMIT}' '${SEED}' \
  '${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  '${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  '${RUN_ROOT}/logs/cellcf/post_run_evidence.json' <<'PY'
import hashlib
import json
import sys
from pathlib import Path

aggregate_path = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
seed = int(sys.argv[3])
expected_paths = {
    "uniform": Path(sys.argv[4]).resolve(),
    "transition_beta0": Path(sys.argv[5]).resolve(),
    "cellcf": Path(sys.argv[6]).resolve(),
}
payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
if payload.get("schema") != "duca_cellcf_suite_manifest_v1" or payload.get("ok") is not True:
    raise SystemExit("aggregate evidence has an invalid schema or status")
if payload.get("status") != "runs_complete_cost_pending":
    raise SystemExit("aggregate evidence is not the expected cost-pending artifact")
if payload.get("git_commit") != commit or payload.get("seed") != seed:
    raise SystemExit("aggregate evidence commit or seed mismatch")
completed = payload.get("completed_runs")
if not isinstance(completed, dict) or set(completed) != set(expected_paths):
    raise SystemExit("aggregate evidence does not cover exactly three CellCF arms")
for variant, expected_path in expected_paths.items():
    record = completed[variant]
    actual_path = Path(str(record.get("path") or "")).resolve()
    if actual_path != expected_path or not actual_path.is_file():
        raise SystemExit(f"aggregate {variant} post-run path mismatch")
    if record.get("sha256") != hashlib.sha256(actual_path.read_bytes()).hexdigest():
        raise SystemExit(f"aggregate {variant} post-run hash mismatch")
PY
'${PYTHON}' -m tools.bata.validate_duca_cellcf_suite \
  --repo-root '${REPO_ROOT}' --seed '${SEED}' \
  --expected-commit '${EXPECTED_COMMIT}' --require-clean \
  --gate-json '${GATE_JSON}' --pilot-json '${PILOT_JSON}' \
  --post-run-evidence 'uniform=${RUN_ROOT}/logs/uniform/post_run_evidence.json' \
  --post-run-evidence 'transition_beta0=${RUN_ROOT}/logs/transition_beta0/post_run_evidence.json' \
  --post-run-evidence 'cellcf=${RUN_ROOT}/logs/cellcf/post_run_evidence.json' \
  --cost-evidence '${COST_EVIDENCE}' \
  --require-cost-evidence \
  --output-json '${FINAL_EVIDENCE}'
EOF
chmod 0755 "${completion_job}"

for job_file in "${RUN_ROOT}"/jobs/*.sbatch; do
  bash -n "${job_file}" || fail "generated job file has invalid syntax: ${job_file}"
done

job_keys=(uniform transition_beta0 cellcf aggregate cost completion)
job_names=(
  "${arm_job_names[@]}"
  "${aggregate_job_name}"
  "${cost_job_name}"
  "${completion_job_name}"
)
dependency_roles=(
  none
  none
  none
  afterok_three_arms
  afterok_aggregate
  afterok_aggregate_and_cost
)
JOBS_TSV="${RUN_ROOT}/jobs.tsv"
jobs_tmp="${JOBS_TSV}.tmp.$$"
printf 'job_key\tseed\tcommit\tjob_name\tdependency_role\tcluster\tmanifest_sha256\tsbatch_file\tsbatch_sha256\tstatus\n' > "${jobs_tmp}"
for index in "${!job_keys[@]}"; do
  key="${job_keys[$index]}"
  job_file="${RUN_ROOT}/jobs/${key}.sbatch"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${key}" "${SEED}" "${EXPECTED_COMMIT}" "${job_names[$index]}" \
    "${dependency_roles[$index]}" "${TARGET_CLUSTER}" "${MANIFEST_SHA256}" \
    "${job_file}" "$(sha256_file "${job_file}")" "PREPARED_NOT_SUBMITTED" \
    >> "${jobs_tmp}"
done
mv "${jobs_tmp}" "${JOBS_TSV}"
JOBS_TSV_SHA256="$(sha256_file "${JOBS_TSV}")"

PREPARED_SUBMISSION="${RUN_ROOT}/prepared_submission.json"
"${PYTHON}" - "${PREPARED_SUBMISSION}" "${MANIFEST}" "${MANIFEST_SHA256}" \
  "${EXPECTED_COMMIT}" "${SEED}" "${TARGET_CLUSTER}" \
  "${CANONICAL_ENV_FILE}" "${CANONICAL_ENV_SHA256}" \
  "${JOBS_TSV}" "${JOBS_TSV_SHA256}" <<'PY'
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

(
    output,
    manifest,
    manifest_sha256,
    commit,
    seed,
    cluster,
    canonical_env,
    canonical_env_sha256,
    jobs_tsv,
    jobs_tsv_sha256,
) = sys.argv[1:]
with open(jobs_tsv, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
jobs = []
for row in rows:
    jobs.append(
        {
            "key": row["job_key"],
            "job_name": row["job_name"],
            "dependency_role": row["dependency_role"],
            "job_file": str(Path(row["sbatch_file"]).resolve()),
            "job_file_sha256": row["sbatch_sha256"],
        }
    )
payload = {
    "schema": "duca_cellcf_prepared_submission_v1",
    "git_commit": commit,
    "seed": int(seed),
    "target_cluster": cluster,
    "checkpoint_interval": 5,
    "suite_manifest": str(Path(manifest).resolve()),
    "suite_manifest_sha256": manifest_sha256,
    "canonical_env_file": str(Path(canonical_env).resolve()),
    "canonical_env_sha256": canonical_env_sha256,
    "jobs_tsv": str(Path(jobs_tsv).resolve()),
    "jobs_tsv_sha256": jobs_tsv_sha256,
    "job_order": [job["key"] for job in jobs],
    "jobs": jobs,
}
target = Path(output)
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
PREPARED_SUBMISSION_SHA256="$(sha256_file "${PREPARED_SUBMISSION}")"
printf '%s\n' "${PREPARED_SUBMISSION_SHA256}" > "${PREPARED_SUBMISSION}.sha256"

echo "[DUCA_CELLCF_PREPARE] prepared ${RUN_ROOT} for cluster ${TARGET_CLUSTER}; no job submitted"
echo "[DUCA_CELLCF_PREPARE] manifest=${MANIFEST_SHA256} submission=${PREPARED_SUBMISSION_SHA256}"
