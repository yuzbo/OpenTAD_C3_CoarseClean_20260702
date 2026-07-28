#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_DENSE_SALVAGE][FAIL] $*" >&2
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
  DUCA_RIME_DENSE_SALVAGE_MANIFEST \
  DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256 \
  DUCA_RIME_DENSE_SALVAGE_BACKEND \
  DUCA_RIME_DENSE_SALVAGE_ROOT \
  DUCA_RIME_DENSE_SALVAGE_CONFIG \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "dense checkpoint salvage/evaluation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact recovery commit is required"
[[ "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" == ActionFormer \
  || "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" == TriDet ]] \
  || fail "salvage backend must be ActionFormer or TriDet"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "salvage requires a complete exact Git worktree"
[[ ! -e "${DUCA_RIME_DENSE_SALVAGE_ROOT}" ]] \
  || fail "salvage output root must be fresh"

cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
python - \
  "${DUCA_RIME_REPO_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import os
import subprocess
import sys

repo_root, required_commit = sys.argv[1:]
expected_commit = os.environ.get("DUCA_EXPECTED_COMMIT")
observed_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=repo_root,
    text=True,
    encoding="utf-8",
).strip()
if expected_commit != required_commit or observed_commit != required_commit:
    raise SystemExit("dense salvage formal evaluator commit bridge drift")
PY
check_sha256 \
  "${DUCA_RIME_DENSE_SALVAGE_MANIFEST}" \
  "${DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256}" \
  "dense salvage manifest"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"

readarray -t manifest_values < <(
  python - \
    "${DUCA_RIME_DENSE_SALVAGE_MANIFEST}" \
    "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" \
    "${DUCA_RIME_DENSE_SALVAGE_ROOT}" <<'PY'
import json
import pathlib
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
backend = sys.argv[2]
source = manifest["sources"][backend]
if pathlib.Path(source["output_root"]).resolve() != pathlib.Path(sys.argv[3]).resolve():
    raise SystemExit("salvage output root differs from frozen manifest")
print(manifest["failed_transaction"]["git_commit"])
print(source["source_job_id"])
print(source["seed"])
print(source["variant"])
PY
)
[[ "${#manifest_values[@]}" == 4 ]] \
  || fail "failed to resolve salvage source identity"
source_training_commit="${manifest_values[0]}"
source_job_id="${manifest_values[1]}"
source_seed="${manifest_values[2]}"
source_variant="${manifest_values[3]}"
[[ "${source_training_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "salvage source commit is invalid"

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
  || fail "failed to resolve frozen dense salvage roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"

readarray -t config_values < <(
  python - \
    "${DUCA_RIME_DENSE_SALVAGE_CONFIG}" \
    "${DUCA_RIME_PRETRAIN_PATH}" \
    "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" <<'PY'
import hashlib
import json
import os
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
    or cfg.model.get("frame_selector") is not None
    or cfg.duca_rime_dense_contract.detector_backend != backend
    or cfg.duca_rime_dense_contract.official_final_subset_consumed is not False
    or bool(cfg.model.backbone.backbone.with_cp)
    or cfg.evaluation.subset != "training"
    or cfg.evaluation.blocked_videos
    != os.environ["DUCA_RIME_DEVELOPMENT_BLOCK_LIST"]
):
    raise SystemExit(f"dense salvage {backend} config contract drift")
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
  || fail "failed to seal salvage evaluation config"
resolved_config_sha256="${config_values[0]}"

run_salvage() {
  python -m tools.bata.salvage_duca_rime_dense_checkpoint \
    --manifest "${DUCA_RIME_DENSE_SALVAGE_MANIFEST}" \
    --manifest-sha256 "${DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256}" \
    --backend "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" \
    --expected-recovery-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
    --output-root "${DUCA_RIME_DENSE_SALVAGE_ROOT}" \
    "$@"
}
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  run_salvage --precheck-only
  echo "[DUCA_RIME_DENSE_SALVAGE] PRECHECK PASS ${DUCA_RIME_DENSE_SALVAGE_BACKEND}"
  exit 0
fi

run_salvage
checkpoint="${DUCA_RIME_DENSE_SALVAGE_ROOT}/checkpoint/terminal_ema.pth"
salvage_receipt="${DUCA_RIME_DENSE_SALVAGE_ROOT}/salvage_receipt.json"
[[ -f "${checkpoint}" && -f "${salvage_receipt}" ]] \
  || fail "salvage tool did not emit checkpoint and receipt"

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-salvage-eval" --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_DENSE_SALVAGE_CONFIG}" \
  --checkpoint "${checkpoint}" \
  --seed "${source_seed}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${DUCA_RIME_DENSE_SALVAGE_ROOT}/evaluation_evidence.json" \
  --cfg-options \
  "work_dir=${DUCA_RIME_DENSE_SALVAGE_ROOT}/evaluation" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

python - \
  "${DUCA_RIME_DENSE_SALVAGE_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${source_training_commit}" \
  "${source_job_id}" \
  "${source_seed}" \
  "${source_variant}" \
  "${DUCA_RIME_DENSE_SALVAGE_BACKEND}" \
  "${resolved_config_sha256}" \
  "${split_values[4]}" \
  "${DUCA_RIME_DENSE_SALVAGE_CONFIG}" \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "${checkpoint}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

from tools.bata.duca_trained_checkpoint_binding import (
    build_trained_checkpoint_binding,
    write_trained_checkpoint_binding,
)

(
    root,
    recovery_commit,
    source_commit,
    source_job_id,
    seed,
    variant,
    backend,
    resolved_config_sha,
    split_assignment_sha,
    config_path,
    evaluation_pretrain_path,
    evaluation_pretrain_sha,
    checkpoint_path,
) = sys.argv[1:]
root = pathlib.Path(root).resolve()
salvage_path = root / "salvage_receipt.json"
evaluation_path = root / "evaluation_evidence.json"
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
if sha(evaluation_pretrain_path) != evaluation_pretrain_sha:
    raise SystemExit("recovery evaluation pretrain SHA-256 drift")
salvage = json.loads(salvage_path.read_text(encoding="utf-8"))
if (
    salvage.get("status") != "passed"
    or salvage.get("recovery_git_commit") != recovery_commit
    or salvage.get("source_training_git_commit") != source_commit
    or salvage.get("source_job_id") != source_job_id
    or salvage.get("backend") != backend
    or salvage.get("uses_official_final") is not False
    or salvage.get("source_root_mutated") is not False
    or salvage.get("original_job_reclassified_as_success") is not False
):
    raise SystemExit("dense salvage receipt violates its immutable source contract")

training_evidence = {
    "schema_version": "duca_rime_dense_salvaged_source_evidence_v1",
    "status": "salvaged_from_failed_post_training_job",
    "source_training_git_commit": source_commit,
    "recovery_git_commit": recovery_commit,
    "source_slurm_job_id": source_job_id,
    "source_job_state": "FAILED",
    "seed_external_manifest": int(seed),
    "variant_external_manifest": variant,
    "detector_backend": backend,
    "terminal_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "checkpoint_path": str(pathlib.Path(checkpoint_path).resolve()),
    "checkpoint_sha256": sha(checkpoint_path),
    "salvage_receipt_path": str(salvage_path),
    "salvage_receipt_sha256": sha(salvage_path),
    "split_assignment_sha256": split_assignment_sha,
    "recovery_evaluation_pretrain_path": str(
        pathlib.Path(evaluation_pretrain_path).resolve()
    ),
    "recovery_evaluation_pretrain_sha256": evaluation_pretrain_sha,
    "embedded_training_provenance": False,
    "uses_official_final": False,
    "claim_scope": "engineering_dense_reference_recovery_not_method_evidence",
}
training_path = root / "source_training_evidence.json"
with training_path.open("x", encoding="utf-8") as handle:
    json.dump(training_evidence, handle, indent=2, sort_keys=True)
    handle.write("\n")

binding = build_trained_checkpoint_binding(
    role="dense_adatad_baseline",
    git_commit=source_commit,
    config_path=config_path,
    resolved_config_sha256=resolved_config_sha,
    checkpoint_path=checkpoint_path,
    checkpoint_epoch=59,
    checkpoint_state_key="state_dict_ema",
    training_evidence_path=training_path,
    evaluation_evidence_path=evaluation_path,
)
binding_path = root / "checkpoint_evidence.json"
write_trained_checkpoint_binding(binding_path, binding)
receipt = {
    "schema_version": "duca_rime_dense_salvage_evaluation_receipt_v1",
    "status": "passed",
    "recovery_git_commit": recovery_commit,
    "source_training_git_commit": source_commit,
    "source_slurm_job_id": source_job_id,
    "source_job_state": "FAILED",
    "detector_backend": backend,
    "checkpoint_path": str(pathlib.Path(checkpoint_path).resolve()),
    "checkpoint_sha256": sha(checkpoint_path),
    "salvage_receipt_path": str(salvage_path),
    "salvage_receipt_sha256": sha(salvage_path),
    "evaluation_evidence_path": str(evaluation_path),
    "evaluation_evidence_sha256": sha(evaluation_path),
    "checkpoint_evidence_path": str(binding_path),
    "checkpoint_evidence_sha256": sha(binding_path),
    "evaluation_pretrain_path": str(
        pathlib.Path(evaluation_pretrain_path).resolve()
    ),
    "evaluation_pretrain_sha256": evaluation_pretrain_sha,
    "uses_official_final": False,
    "energy_evidence_available": False,
    "original_job_reclassified_as_success": False,
    "claim_scope": "engineering_dense_reference_recovery_not_method_evidence",
}
target = root / "recovery_receipt.json"
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY

echo \
  "[DUCA_RIME_DENSE_SALVAGE] PASS ${DUCA_RIME_DENSE_SALVAGE_BACKEND} ${DUCA_RIME_DENSE_SALVAGE_ROOT}"
