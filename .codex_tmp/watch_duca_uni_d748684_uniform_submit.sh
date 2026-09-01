#!/usr/bin/env bash
set -euo pipefail

GATE_JOB=1177687
BASE=/data/run01/sczc063/yuzibo
RUN_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_official60_20260721_0330
GATE_ROOT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_uni_d748684_gate_20260721_0325
ARM_SCRIPT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/run_duca_uni_d748684_official60_arm.sh
COMPLETE_SCRIPT=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/complete_duca_uni_d748684_official60.sh
AUTHORIZATION_JSON="${GATE_ROOT}/authorization.json"
RECEIPT="${RUN_ROOT}/uniform_and_completion_submit_receipt.tsv"
MAX_POLLS=5760
SLEEP_SECONDS=60

[[ ! -e "${RECEIPT}" ]]
if squeue --clusters=n16r4 -h -n du748_uniform -o '%A' | grep -q .; then
  echo "an active du748_uniform Job already exists" >&2
  exit 2
fi

job_state() {
  sacct -X -j "$1" --format=State -n -P | head -n 1 | cut -d' ' -f1
}

for ((poll = 1; poll <= MAX_POLLS; poll += 1)); do
  state="$(job_state "${GATE_JOB}")"
  case "${state}" in
    COMPLETED)
      [[ -f "${AUTHORIZATION_JSON}" ]]
      grep -q '"ok": true' "${AUTHORIZATION_JSON}"
      break
      ;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
      echo "gate ${GATE_JOB} ended as ${state}; refusing official-60" >&2
      exit 3
      ;;
    *)
      sleep "${SLEEP_SECONDS}"
      ;;
  esac
done

[[ "$(job_state "${GATE_JOB}")" == "COMPLETED" ]]

uniform_response="$(
  sbatch --parsable --clusters=n16r4 \
    --job-name=du748_uniform \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task=8 \
    --time=3-00:00:00 \
    --output="${RUN_ROOT}/logs/exact_uniform-%j.out" \
    --error="${RUN_ROOT}/logs/exact_uniform-%j.err" \
    --export=ALL,DUCA_VARIANT=exact_uniform \
    "${ARM_SCRIPT}"
)"
uniform_job="${uniform_response%%;*}"
[[ "${uniform_job}" =~ ^[1-9][0-9]*$ ]]
printf 'exact_uniform\t%s\tafter_gate_completed\tSUBMITTED_BY_WATCHER\n' \
  "${uniform_job}" >> "${RUN_ROOT}/jobs.tsv"
printf 'key\tjob_id\nexact_uniform\t%s\n' "${uniform_job}" > "${RECEIPT}"

arm_dependency="afterok:1177690:1177691:1177692:${uniform_job}"
for ((poll = 1; poll <= MAX_POLLS; poll += 1)); do
  set +e
  complete_response="$(
    sbatch --parsable --clusters=n16r4 \
      --dependency="${arm_dependency}" \
      --job-name=du748_complete \
      --nodes=1 \
      --ntasks=1 \
      --gpus=1 \
      --cpus-per-task=2 \
      --time=01:00:00 \
      --output="${RUN_ROOT}/logs/complete-%j.out" \
      --error="${RUN_ROOT}/logs/complete-%j.err" \
      "${COMPLETE_SCRIPT}" 2>"${RUN_ROOT}/logs/complete-submit.err"
  )"
  submit_status=$?
  set -e
  if [[ "${submit_status}" == "0" ]]; then
    complete_job="${complete_response%%;*}"
    [[ "${complete_job}" =~ ^[1-9][0-9]*$ ]]
    printf 'complete\t%s\t%s\tSUBMITTED_BY_WATCHER\n' \
      "${complete_job}" "${arm_dependency}" >> "${RUN_ROOT}/jobs.tsv"
    printf 'complete\t%s\n' "${complete_job}" >> "${RECEIPT}"
    exit 0
  fi
  if grep -q 'AssocMaxSubmitJobLimit' "${RUN_ROOT}/logs/complete-submit.err"; then
    sleep "${SLEEP_SECONDS}"
    continue
  fi
  cat "${RUN_ROOT}/logs/complete-submit.err" >&2
  exit "${submit_status}"
done

echo "timed out waiting to submit completion job" >&2
exit 4
