#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_DENSE_TRIDET][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_DENSE_TRIDET_CONFIG \
  DUCA_RIME_DENSE_TRIDET_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256; do
  required "${name}"
done

dense_backend="${DUCA_RIME_DENSE_BACKEND:-TriDet}"
[[ "${dense_backend}" == ActionFormer || "${dense_backend}" == TriDet ]] \
  || fail "dense backend must be ActionFormer or TriDet"

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "dense reference training must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "a complete exact Git worktree is required"
[[ ! -e "${DUCA_RIME_DENSE_TRIDET_ROOT}" ]] \
  || fail "a fresh dense reference root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"

readarray -t split_values < <(
  python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

from tools.bata.create_duca_rime_splits import validate_rime_splits

validation = validate_rime_splits(sys.argv[1])
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    row = manifest["train_roles"][role]
    print(row["block_list_path"])
    print(row["block_list_sha256"])
print(validation["assignment_sha256"])
PY
)
[[ "${#split_values[@]}" == 5 ]] \
  || fail "failed to resolve frozen dense TriDet roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

readarray -t config_values < <(
  python - \
    "${DUCA_RIME_DENSE_TRIDET_CONFIG}" \
    "${DUCA_RIME_PRETRAIN_PATH}" \
    "${dense_backend}" <<'PY'
import hashlib
import json
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
cfg.model.backbone.custom.pretrain = sys.argv[2]
backend = sys.argv[3]
expected_protocol = {
    "ActionFormer": "duca_rime_dense_actionformer_cost_baseline_v1",
    "TriDet": "duca_rime_dense_tridet_cost_baseline_v1",
}[backend]
if (
    cfg.workflow.formal_protocol != expected_protocol
    or cfg.model.type != backend
    or int(cfg.model.projection.in_channels) != 384
    or int(cfg.model.projection.max_seq_len) != 768
    or cfg.model.get("frame_selector") is not None
    or cfg.duca_rime_dense_contract.detector_backend != backend
    or cfg.duca_rime_dense_contract.selector is not None
    or cfg.duca_rime_dense_contract.official_final_subset_consumed is not False
):
    raise SystemExit(f"dense {backend} config contract drift")
canonical = json.dumps(
    cfg.to_dict(),
    sort_keys=True,
    separators=(",", ":"),
    default=str,
)
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
PY
)
[[ "${#config_values[@]}" == 1 ]] \
  || fail "failed to seal dense reference config"
resolved_config_sha256="${config_values[0]}"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_RIME_DENSE_TRIDET] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_DENSE_TRIDET_ROOT}"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-train" --nproc_per_node=1 tools/train.py \
  "${DUCA_RIME_DENSE_TRIDET_CONFIG}" \
  --seed 3407 \
  --id 0 \
  --cfg-options \
  "work_dir=${DUCA_RIME_DENSE_TRIDET_ROOT}/train" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

actual_root="${DUCA_RIME_DENSE_TRIDET_ROOT}/train/gpu1_id0"
full_checkpoint="${actual_root}/checkpoint/epoch_59.pth"
checkpoint="${actual_root}/checkpoint/terminal_ema.pth"
python tools/bata/compact_duca_rime_checkpoint.py \
  --source "${full_checkpoint}" \
  --output "${checkpoint}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --remove-source

python - \
  "${DUCA_RIME_DENSE_TRIDET_ROOT}/training_evidence.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${checkpoint}" \
  "${resolved_config_sha256}" \
  "${split_values[4]}" \
  "${SLURM_JOB_ID}" \
  "${dense_backend}" <<'PY'
import hashlib
import json
import os
import sys

output, commit, checkpoint, config_sha, assignment_sha, job_id, backend = sys.argv[1:]
payload = {
    "schema_version": f"duca_rime_dense_{backend.lower()}_training_evidence_v1",
    "status": "complete",
    "git_commit": commit,
    "slurm_job_id": job_id,
    "seed": 3407,
    "terminal_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "checkpoint_path": os.path.abspath(checkpoint),
    "checkpoint_sha256": hashlib.sha256(open(checkpoint, "rb").read()).hexdigest(),
    "resolved_config_sha256": config_sha,
    "split_assignment_sha256": assignment_sha,
    "detector_backend": backend,
    "uses_official_final": False,
}
temporary = output + f".partial.{os.getpid()}"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-eval" --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_DENSE_TRIDET_CONFIG}" \
  --checkpoint "${checkpoint}" \
  --seed 3407 \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${DUCA_RIME_DENSE_TRIDET_ROOT}/evaluation_evidence.json" \
  --cfg-options \
  "work_dir=${DUCA_RIME_DENSE_TRIDET_ROOT}/evaluation" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

python - \
  "${DUCA_RIME_DENSE_TRIDET_CONFIG}" \
  "${resolved_config_sha256}" \
  "${checkpoint}" \
  "${DUCA_RIME_DENSE_TRIDET_ROOT}/training_evidence.json" \
  "${DUCA_RIME_DENSE_TRIDET_ROOT}/evaluation_evidence.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${DUCA_RIME_DENSE_TRIDET_ROOT}/checkpoint_evidence.json" <<'PY'
import json
import sys

from tools.bata.duca_trained_checkpoint_binding import (
    build_trained_checkpoint_binding,
    write_trained_checkpoint_binding,
)

config, resolved, checkpoint, training, evaluation, commit, output = sys.argv[1:]
payload = build_trained_checkpoint_binding(
    role="dense_adatad_baseline",
    git_commit=commit,
    config_path=config,
    resolved_config_sha256=resolved,
    checkpoint_path=checkpoint,
    checkpoint_epoch=59,
    checkpoint_state_key="state_dict_ema",
    training_evidence_path=training,
    evaluation_evidence_path=evaluation,
)
write_trained_checkpoint_binding(output, payload)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python - \
  "${DUCA_RIME_DENSE_TRIDET_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${checkpoint}" \
  "${resolved_config_sha256}" \
  "${SLURM_JOB_ID}" \
  "${dense_backend}" <<'PY'
import hashlib
import json
import os
import sys

root, commit, checkpoint, config_sha, job_id, backend = sys.argv[1:]
evidence = os.path.join(root, "checkpoint_evidence.json")
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
payload = {
    "schema_version": f"duca_rime_dense_{backend.lower()}_receipt_v1",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": job_id,
    "detector_backend": backend,
    "checkpoint_path": os.path.abspath(checkpoint),
    "checkpoint_sha256": sha(checkpoint),
    "checkpoint_evidence_path": os.path.abspath(evidence),
    "checkpoint_evidence_sha256": sha(evidence),
    "resolved_config_sha256": config_sha,
    "uses_official_final": False,
    "claim_scope": (
        f"trained_dense_{backend.lower()}_cost_reference_not_candidate_method"
    ),
}
output = os.path.join(root, "training_receipt.json")
temporary = output + f".partial.{os.getpid()}"
with open(temporary, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

echo "[DUCA_RIME_DENSE_REFERENCE] PASS ${dense_backend} ${DUCA_RIME_DENSE_TRIDET_ROOT}"
