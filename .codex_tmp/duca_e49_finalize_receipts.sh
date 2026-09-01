#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
PREFIX=${BASE}/duca_boundary_e49ef69_formal_20260722_155037
R03=${PREFIX}_r0_r3
R4=${PREFIX}_r4
R5=${PREFIX}_r5
COMMIT=e49ef69605e1f98a7217957483f93a8a64bfc348
MEMBERS=${R5}/bundle_members.tsv

[[ $(awk 'END {print NR-1}' ${R5}/cells.tsv) -eq 24 ]]
[[ $(awk 'END {print NR-1}' ${R5}/costs.tsv) -eq 9 ]]
[[ $(awk 'END {print NR-1}' ${MEMBERS}) -eq 34 ]]
[[ "$(sha256sum ${MEMBERS} | awk '{print $1}')" == "$(tr -d '[:space:]' < ${MEMBERS}.sha256)" ]]
[[ "$(sha256sum ${R5}/jobs.tsv | awk '{print $1}')" == "$(tr -d '[:space:]' < ${R5}/jobs.tsv.sha256)" ]]
for script in ${R5}/jobs/train_bundle_{0,1,2,3}.sbatch ${R5}/jobs/postprocess_bundle.sbatch; do
  [[ "$(sha256sum ${script} | awk '{print $1}')" == "$(tr -d '[:space:]' < ${script}.sha256)" ]]
done

python3 - ${R5} ${COMMIT} ${MEMBERS} <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
members = Path(sys.argv[3]).resolve()
with (root / "jobs.tsv").open(encoding="utf-8", newline="") as handle:
    jobs = list(csv.DictReader(handle, delimiter="\t"))
with (root / "jobs.individual_partial.tsv").open(encoding="utf-8", newline="") as handle:
    partial = list(csv.DictReader(handle, delimiter="\t"))
cancelled = [row["job_id"] for row in partial if row["role"] != "temporalmaxer_one_step"]
payload = {
    "schema": "duca_r5_site_bundle_submission_v1",
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "reason": "n16r4 AssocMaxSubmitJobLimit",
    "experiment_content_changed": False,
    "equivalence": (
        "Every original hash-bound cell/cost sbatch is executed unchanged; "
        "only six cells per GPU allocation are serialized."
    ),
    "mechanism_gate_job": jobs[0]["job_id"],
    "train_bundle_jobs": [row["job_id"] for row in jobs if row["role"].startswith("train_bundle_")],
    "postprocess_job": jobs[-1]["job_id"],
    "cancelled_duplicate_individual_job_ids": cancelled,
    "train_bundle_count": 4,
    "cells_executed": 24,
    "cost_profiles_executed": 9,
    "final_aggregate_executed": True,
    "bundle_members_path": str(members),
    "bundle_members_sha256": hashlib.sha256(members.read_bytes()).hexdigest(),
    "jobs_path": str(root / "jobs.tsv"),
    "jobs_sha256": hashlib.sha256((root / "jobs.tsv").read_bytes()).hexdigest(),
    "partial_individual_jobs_path": str(root / "jobs.individual_partial.tsv"),
    "partial_individual_jobs_sha256": hashlib.sha256((root / "jobs.individual_partial.tsv").read_bytes()).hexdigest(),
}
output = root / "site_bundle_submission.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "site_bundle_submission.json.sha256").write_text(
    hashlib.sha256(output.read_bytes()).hexdigest() + "\n", encoding="utf-8"
)
PY

RECEIPT=${PREFIX}_deployment.tsv
{
  printf 'field\tvalue\n'
  printf 'exact_commit\t%s\n' ${COMMIT}
  printf 'snapshot\t/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_e49ef69_20260722\n'
  printf 'r0_r3_root\t%s\n' ${R03}
  printf 'r0_r3_jobs\t%s\n' ${R03}/jobs.tsv
  printf 'r3_aggregate_job\t1179825\n'
  printf 'r4_root\t%s\n' ${R4}
  printf 'r4_job\t1179826\n'
  printf 'r5_root\t%s\n' ${R5}
  printf 'r5_gate_job\t1179827\n'
  printf 'r5_train_bundle_jobs\t1179861,1179862,1179863,1179864\n'
  printf 'r5_postprocess_job\t1179865\n'
  printf 'r5_submission_mode\tsite_bundle_equivalent\n'
  printf 'r5_cells\t24\n'
  printf 'r5_cost_profiles\t9\n'
  printf 'r5_jobs\t%s\n' ${R5}/jobs.tsv
  printf 'r5_bundle_manifest\t%s\n' ${R5}/site_bundle_submission.json
} > ${RECEIPT}
sha256sum ${RECEIPT} | awk '{print $1}' > ${RECEIPT}.sha256

echo RECEIPTS_OK
echo RECEIPT=${RECEIPT}
echo RECEIPT_SHA=$(cat ${RECEIPT}.sha256)
echo R5_MANIFEST_SHA=$(cat ${R5}/site_bundle_submission.json.sha256)
echo JOBS
cat ${R5}/jobs.tsv
echo BUNDLE_COUNTS
awk -F '\t' 'NR > 1 {counts[$1]++} END {for (key in counts) print key, counts[key]}' ${MEMBERS} | sort
echo CANCELLED_DUPLICATE_STATES
sacct -j 1179828,1179829,1179830,1179831,1179832,1179833 \
  --format=JobID,State,ExitCode -P
