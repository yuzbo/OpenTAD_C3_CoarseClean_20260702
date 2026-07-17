#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_CONVERGENCE][FAIL] $*" >&2
  exit 1
}

EVIDENCE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${EVIDENCE_REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
TRAINED_REPO_ROOT="${DUCA_CELLCF_TRAINED_REPO_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
RUN_ROOT="${DUCA_CELLCF_FORMAL_RUN_ROOT:-}"
VARIANT="${DUCA_CELLCF_VARIANT:-}"
SEED="${SEED:-0}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "trajectory evaluation must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ -d "${TRAINED_REPO_ROOT}" ]] || fail "exact trained repository is missing"
[[ -n "${EXPECTED_COMMIT}" ]] || fail "DUCA_EXPECTED_COMMIT is required"
[[ -d "${RUN_ROOT}" ]] || fail "formal run root is missing"
RUN_ROOT="$(
  duca_cellcf_require_external_path \
    "RUN_ROOT" "${EVIDENCE_REPO_ROOT}" "${BASE}" "${RUN_ROOT}"
)" || fail "RUN_ROOT violates the formal path contract"
case "${RUN_ROOT}/" in
  "$(realpath -e -- "${TRAINED_REPO_ROOT}")/"*)
    fail "formal run root must stay outside the trained worktree"
    ;;
esac
case "${VARIANT}" in
  uniform)
    CONFIG="configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py"
    ;;
  transition_beta0)
    CONFIG="configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py"
    ;;
  cellcf)
    CONFIG="configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py"
    ;;
  *)
    fail "DUCA_CELLCF_VARIANT must be uniform, transition_beta0, or cellcf"
    ;;
esac

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd "${TRAINED_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "trained checkout commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "trained checkout is dirty"
IGNORED_PYTHON_SOURCES="$(
  git ls-files --others --ignored --exclude-standard -- \
    '*.py' '*.pth' 'sitecustomize.py' 'usercustomize.py'
)"
[[ -z "${IGNORED_PYTHON_SOURCES}" ]] \
  || fail "ignored Python sources could shadow the trained checkout"
unset IGNORED_PYTHON_SOURCES
export DUCA_CELLCF_TRAINING_PROFILE=exposure132
source "${TRAINED_REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one Slurm-visible GPU is required"

POST_RUN="${RUN_ROOT}/logs/${VARIANT}/post_run_evidence.json"
[[ -f "${POST_RUN}" ]] || fail "sealed post-run evidence is missing"
readarray -t binding < <("${PYTHON}" - "${POST_RUN}" "${VARIANT}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
variant = sys.argv[2]
commit = sys.argv[3]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "duca_cellcf_post_run_evidence_v1" or payload.get("ok") is not True:
    raise SystemExit("post-run evidence is invalid")
if payload.get("variant") != variant or payload.get("git_commit") != commit:
    raise SystemExit("post-run variant/commit mismatch")
if payload.get("training_profile", "exposure132") != "exposure132":
    raise SystemExit("fixed trajectory only accepts exposure132 evidence")
if payload.get("successful_optimizer_updates") != 13200:
    raise SystemExit("post-run terminal update count mismatch")
if payload.get("checkpoint_epoch") != 131 or payload.get("checkpoint_state_key") != "state_dict_ema":
    raise SystemExit("post-run terminal binding mismatch")
print(payload["checkpoint_path"])
print(payload["terminal_evaluation_path"])
PY
)
[[ "${#binding[@]}" == "2" ]] || fail "failed to parse post-run binding"
TERMINAL_CHECKPOINT="${binding[0]}"
TERMINAL_EVALUATION="${binding[1]}"
CHECKPOINT_DIR="$(dirname "${TERMINAL_CHECKPOINT}")"
OUTPUT_ROOT="${RUN_ROOT}/convergence/${VARIANT}"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "refusing to overwrite trajectory evidence"
mkdir -p "${OUTPUT_ROOT}"

EVALUATION_RUNTIME_HASHES=()
for epoch in 59 89; do
  checkpoint="${CHECKPOINT_DIR}/epoch_${epoch}.pth"
  sidecar="${checkpoint}.metadata.json"
  eval_root="${OUTPUT_ROOT}/epoch_${epoch}"
  metrics_json="${eval_root}/evaluation.json"
  [[ -f "${checkpoint}" && -f "${sidecar}" ]] || fail "epoch ${epoch} checkpoint evidence is incomplete"
  [[ ! -e "${metrics_json}" ]] || fail "refusing to overwrite epoch ${epoch} evaluation"
  mkdir -p "${eval_root}"
  DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256="$("${PYTHON}" - \
    "${CONFIG}" "${eval_root}/work" "${ADATAD_PRETRAIN_PATH}" <<'PY'
import sys

from tools.bata.duca_cellcf_training import expected_runtime_config_sha256

config, work_dir, pretrain = sys.argv[1:]
print(
    expected_runtime_config_sha256(
        config,
        {
            "work_dir": work_dir,
            "model.backbone.custom.pretrain": pretrain,
            "post_processing.save_dict": True,
            "inference.load_from_raw_predictions": False,
        },
        experiment_id=0,
        gpu_num=1,
        entrypoint="tools/test.py",
    )
)
PY
)"
  [[ "${DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "epoch ${epoch} runtime config hash is invalid"
  export DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256
  EVALUATION_RUNTIME_HASHES+=("${DUCA_CELLCF_EVAL_RUNTIME_CONFIG_SHA256}")
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="cellcf-convergence-${SLURM_JOB_ID}-${VARIANT}-${epoch}" \
    tools/test.py "${CONFIG}" --checkpoint "${checkpoint}" \
    --checkpoint-state-key state_dict_ema --expected-checkpoint-epoch "${epoch}" \
    --metrics-json "${metrics_json}" --id 0 --seed "${SEED}" \
    --cfg-options "work_dir=${eval_root}/work" \
      "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
      "post_processing.save_dict=True" "inference.load_from_raw_predictions=False" \
    2>&1 | tee "${eval_root}/evaluation.out"
done

[[ -f "${TERMINAL_EVALUATION}" ]] || fail "sealed terminal evaluation is missing"
"${PYTHON}" - "${OUTPUT_ROOT}/variant_complete.json" "${POST_RUN}" \
  "${OUTPUT_ROOT}/epoch_59/evaluation.json" \
  "${OUTPUT_ROOT}/epoch_89/evaluation.json" "${TERMINAL_EVALUATION}" \
  "${VARIANT}" "${SEED}" "${EXPECTED_COMMIT}" \
  "${EVALUATION_RUNTIME_HASHES[0]}" "${EVALUATION_RUNTIME_HASHES[1]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

output = Path(sys.argv[1]).resolve()
paths = [Path(value).resolve() for value in sys.argv[2:6]]
variant = sys.argv[6]
seed = int(sys.argv[7])
commit = sys.argv[8]
runtime_hashes = sys.argv[9:11]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"missing trajectory artifact: {path}")
payload = {
    "schema": "duca_cellcf_convergence_variant_receipt_v1",
    "ok": True,
    "task": "offline_temporal_action_detection",
    "git_commit": commit,
    "training_profile": "exposure132",
    "variant": variant,
    "seed": seed,
    "artifacts": [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ],
}
evaluation_runtime_hashes = {}
for epoch, evaluation_path in zip((59, 89, 131), paths[1:]):
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    runtime_hash = evaluation.get("runtime_config_sha256")
    if not isinstance(runtime_hash, str) or len(runtime_hash) != 64:
        raise SystemExit(f"epoch {epoch} evaluation runtime hash is invalid")
    evaluation_runtime_hashes[str(epoch)] = runtime_hash
if evaluation_runtime_hashes["59"] != runtime_hashes[0]:
    raise SystemExit("epoch 59 runtime hash differs from the launched config")
if evaluation_runtime_hashes["89"] != runtime_hashes[1]:
    raise SystemExit("epoch 89 runtime hash differs from the launched config")
payload["evaluation_runtime_config_sha256"] = evaluation_runtime_hashes
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
).hexdigest()
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

echo "[DUCA_CELLCF_CONVERGENCE] ${VARIANT} fixed epoch 59/89 diagnostics complete"
