#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_DDP_PILOT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
REAL_LOADER_GATE_JSON="${DUCA_CELLCF_GATE_JSON:-}"
EXPECTED_GATE_SHA256="${DUCA_CELLCF_GATE_SHA256:-${DUCA_CELLCF_REAL_LOADER_GATE_SHA256:-}}"
SEED="${SEED:-0}"

[[ -n "${EXPECTED_COMMIT}" ]] || fail "DUCA_EXPECTED_COMMIT is required"
[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "pilot requires a clean exact-commit checkout"
[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "AdaTAD pretrain missing: ${ADATAD_PRETRAIN_PATH}"
[[ -f "${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py" ]] \
  || fail "official ASFormer source is missing"
[[ -n "${REAL_LOADER_GATE_JSON}" && -f "${REAL_LOADER_GATE_JSON}" ]] \
  || fail "DUCA_CELLCF_GATE_JSON must name the real-loader CUDA gate"
[[ "${EXPECTED_GATE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "DUCA_CELLCF_GATE_SHA256 must be an exact lowercase SHA256"
[[ "${SEED}" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"

REAL_LOADER_GATE_JSON="$("${PYTHON}" -c \
  'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
  "${REAL_LOADER_GATE_JSON}")"
OBSERVED_GATE_SHA256="$(sha256sum "${REAL_LOADER_GATE_JSON}" | awk '{print $1}')"
[[ "${OBSERVED_GATE_SHA256}" == "${EXPECTED_GATE_SHA256}" ]] \
  || fail "real-loader gate SHA256 differs from the frozen deployment binding"

preflight_dir="$(mktemp -d "${TMPDIR:-/tmp}/duca_cellcf_ddp_preflight.XXXXXX")"
preflight_json="${preflight_dir}/preflight.json"
cleanup_preflight() {
  rm -rf "${preflight_dir}"
}
trap cleanup_preflight EXIT
"${PYTHON}" -m tools.bata.validate_duca_cellcf_ddp_pilot \
  --repo-root "${REPO_ROOT}" \
  --real-loader-gate-json "${REAL_LOADER_GATE_JSON}" \
  --expected-real-loader-gate-sha256 "${EXPECTED_GATE_SHA256}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --precheck-only \
  --output-json "${preflight_json}" \
  >/dev/null

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_CELLCF_DDP_PILOT] PRECHECK PASS commit=${EXPECTED_COMMIT} gate=${EXPECTED_GATE_SHA256}"
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] \
  || fail "pilot must run inside one numeric Slurm allocation"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
  || fail "pilot must not run as a Slurm array"
[[ "${SLURM_NNODES:-1}" == "1" ]] || fail "pilot requires exactly one Slurm node"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || fail "pilot requires exactly one Slurm task"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose an allocated GPU"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
VISIBLE_GPU_COUNT="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${VISIBLE_GPU_COUNT}" == "1" ]] || fail "pilot requires exactly one Slurm-visible GPU"

RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_cellcf_ddp_pilot_${DUCA_CELLCF_TRAINING_PROFILE}_${CURRENT_HEAD:0:12}_${EXPECTED_GATE_SHA256:0:12}}"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "RUN_ROOT violates the formal path contract"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/probes" "${RUN_ROOT}/work_dirs" "${RUN_ROOT}/logs"
CANONICAL_ENV_FILE="${RUN_ROOT}/canonical_env.tsv"
duca_cellcf_canonical_env_payload >"${CANONICAL_ENV_FILE}"

PILOT_NONCE="${SLURM_JOB_ID}-${CURRENT_HEAD}-$(date +%s%N)"
OFFICIAL_ASFORMER_SOURCE="${C3_OFFICIAL_ACTION_SEG_REPOS}/ASFormer/model.py"
MANIFEST_PATH="${RUN_ROOT}/manifest.json"
"${PYTHON}" - "${MANIFEST_PATH}" "${CURRENT_HEAD}" "${REAL_LOADER_GATE_JSON}" \
  "${EXPECTED_GATE_SHA256}" "${SEED}" "${SLURM_JOB_ID}" "${PILOT_NONCE}" \
  "${ADATAD_PRETRAIN_PATH}" "${OFFICIAL_ASFORMER_SOURCE}" "${CANONICAL_ENV_FILE}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


(
    output,
    commit,
    gate,
    gate_sha,
    seed,
    slurm_job_id,
    nonce,
    checkpoint,
    official_source,
    canonical_env,
) = sys.argv[1:]
payload = {
    "schema": "duca_cellcf_ddp_pilot_run_v1",
    "git_commit": commit,
    "real_loader_gate_json": str(Path(gate).resolve()),
    "real_loader_gate_sha256": gate_sha,
    "real_loader_gate_git_commit": commit,
    "seed": int(seed),
    "slurm_job_id": slurm_job_id,
    "slurm_array_job_id": None,
    "pilot_nonce": nonce,
    "world_size": 1,
    "nproc_per_node": 1,
    "torch_distributed_entrypoint": "python -m torch.distributed.run",
    "task": "offline_temporal_action_detection",
    "fixed_k": 384,
    "dense_window_size": 768,
    "training_profile": os.environ["DUCA_CELLCF_TRAINING_PROFILE"],
    "formal_end_epoch": int(os.environ["DUCA_CELLCF_END_EPOCH"]),
    "checkpoint_interval": 5,
    "pilot_checkpoint_disabled": True,
    "successful_updates_per_arm": 10,
    "forced_amp_overflow_attempts_per_arm": 1,
    "variants": ["uniform", "transition_beta0", "cellcf"],
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": sha256(checkpoint),
    "official_asformer_source": str(Path(official_source).resolve()),
    "official_asformer_source_sha256": sha256(official_source),
    "canonical_env_path": str(Path(canonical_env).resolve()),
    "canonical_env_sha256": sha256(canonical_env),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
RUN_MANIFEST_SHA256="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"

variants=(uniform transition_beta0 cellcf)
pilot_configs=(
  configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_p0_ddp_pilot.py
  configs/adatad/thumos/duca_cellcf_fixed384_p0_ddp_pilot.py
)
formal_configs=(
  configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py
  configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py
  configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py
)

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  pilot_config="${pilot_configs[$index]}"
  formal_config="${formal_configs[$index]}"
  probe_json="${RUN_ROOT}/probes/${variant}.training_probe.json"
  context_json="${RUN_ROOT}/probes/${variant}.context.json"
  work_dir="${RUN_ROOT}/work_dirs/${variant}"

  "${PYTHON}" - "${context_json}" "${CURRENT_HEAD}" "${variant}" "${SEED}" \
    "${SLURM_JOB_ID}" "${PILOT_NONCE}" "${REPO_ROOT}/${pilot_config}" \
    "${REPO_ROOT}/${formal_config}" "${probe_json}" "${work_dir}" \
    "${ADATAD_PRETRAIN_PATH}" "${REAL_LOADER_GATE_JSON}" "${EXPECTED_GATE_SHA256}" \
    "${MANIFEST_PATH}" "${RUN_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


(
    context_path,
    commit,
    variant,
    seed,
    slurm_job_id,
    nonce,
    pilot_config,
    formal_config,
    probe_json,
    work_dir,
    checkpoint,
    gate,
    gate_sha,
    manifest,
    manifest_sha,
) = sys.argv[1:]
payload = {
    "schema": "duca_cellcf_ddp_pilot_context_v1",
    "git_commit": commit,
    "variant": variant,
    "seed": int(seed),
    "slurm_job_id": slurm_job_id,
    "pilot_nonce": nonce,
    "world_size": 1,
    "source_config_path": str(Path(pilot_config).resolve()),
    "source_config_sha256": sha256(pilot_config),
    "formal_config_path": str(Path(formal_config).resolve()),
    "formal_config_sha256": sha256(formal_config),
    "training_probe_json": str(Path(probe_json).resolve()),
    "context_json": str(Path(context_path).resolve()),
    "work_dir": str(Path(work_dir).resolve()),
    "checkpoint_path": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": sha256(checkpoint),
    "real_loader_gate_json": str(Path(gate).resolve()),
    "real_loader_gate_sha256": gate_sha,
    "run_manifest_path": str(Path(manifest).resolve()),
    "run_manifest_sha256": manifest_sha,
    "task": "offline_temporal_action_detection",
    "fixed_k": 384,
    "training_profile": os.environ["DUCA_CELLCF_TRAINING_PROFILE"],
    "formal_end_epoch": int(os.environ["DUCA_CELLCF_END_EPOCH"]),
    "checkpoint_interval": 5,
    "pilot_checkpoint_disabled": True,
}
Path(context_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

  export DUCA_TRAINING_PROBE_JSON="${probe_json}"
  export DUCA_TRAINING_PROBE_CONTEXT_JSON="${context_json}"
  echo "[DUCA_CELLCF_DDP_PILOT] starting ${variant} in Slurm job ${SLURM_JOB_ID}"
  "${PYTHON}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-cellcf-${SLURM_JOB_ID}-${variant}" \
    tools/train.py \
    "${pilot_config}" \
    --id "${index}" \
    --seed "${SEED}" \
    --cfg-options \
      "work_dir=${work_dir}" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
    2>&1 | tee "${RUN_ROOT}/logs/${variant}.train.out"
  [[ -s "${probe_json}" ]] || fail "${variant} did not emit a non-empty training probe"
  checkpoint_file="$(find "${work_dir}" -type f -name '*.pth' -print -quit)"
  [[ -z "${checkpoint_file}" ]] || fail "${variant} pilot wrote a forbidden checkpoint: ${checkpoint_file}"
done
unset DUCA_TRAINING_PROBE_JSON DUCA_TRAINING_PROBE_CONTEXT_JSON

PILOT_ARTIFACT="${RUN_ROOT}/duca_cellcf_ddp_pilot_suite.json"
"${PYTHON}" -m tools.bata.validate_duca_cellcf_ddp_pilot \
  --repo-root "${REPO_ROOT}" \
  --probe-dir "${RUN_ROOT}/probes" \
  --real-loader-gate-json "${REAL_LOADER_GATE_JSON}" \
  --expected-real-loader-gate-sha256 "${EXPECTED_GATE_SHA256}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --output-json "${PILOT_ARTIFACT}" \
  >"${RUN_ROOT}/duca_cellcf_ddp_pilot_validation.out"

"${PYTHON}" - "${PILOT_ARTIFACT}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("schema") != "duca_cellcf_ddp_pilot_suite_v1" or payload.get("ok") is not True:
    raise SystemExit("CellCF DDP pilot artifact failed its final schema/status check")
PY

echo "[DUCA_CELLCF_DDP_PILOT] PASS ${PILOT_ARTIFACT}"
