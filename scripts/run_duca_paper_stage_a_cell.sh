#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_STAGE_A_CELL][FAIL] $*" >&2
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
  DUCA_PAPER_REPO_ROOT \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_ARM \
  DUCA_PAPER_CONFIG \
  DUCA_PAPER_CELL_ROOT \
  DUCA_PAPER_SEED \
  DUCA_PAPER_MATRIX_MANIFEST \
  DUCA_PAPER_MATRIX_MANIFEST_SHA256 \
  DUCA_PAPER_PRETRAIN_PATH \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_PATH \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_PATH \
  DUCA_PAPER_CLASS_MAP_SHA256 \
  DUCA_PAPER_CODE_GATE_RECEIPT \
  DUCA_PAPER_CODE_GATE_RECEIPT_SHA256 \
  DUCA_PAPER_SHORT_WINDOW_GATE_JSON \
  DUCA_PAPER_SHORT_WINDOW_GATE_SHA256 \
  DUCA_PAPER_NUMERIC_GATE_JSON \
  DUCA_PAPER_NUMERIC_GATE_SHA256 \
  DUCA_PAPER_EXACT211_UID_GATE_JSON \
  DUCA_PAPER_EXACT211_UID_GATE_SHA256; do
  required "${name}"
done

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${DUCA_PAPER_REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
DUCA_PAPER_CELL_ROOT="$(
  duca_cellcf_require_external_path \
    "DUCA_PAPER_CELL_ROOT" \
    "${DUCA_PAPER_REPO_ROOT}" \
    "${BASE}" \
    "${DUCA_PAPER_CELL_ROOT}"
)" || fail "Stage-A cell root violates the formal path contract"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Stage-A cells must run inside Slurm"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ -d "${DUCA_PAPER_REPO_ROOT}/.git" ]] \
  || fail "a complete Git checkout is required"
cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_PAPER_CELL_ROOT}" ]] || fail "a fresh cell root is required"
[[ -f "${DUCA_PAPER_CONFIG}" ]] || fail "cell config is missing"
check_sha256 \
  "${DUCA_PAPER_MATRIX_MANIFEST}" \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" \
  "Stage-A matrix manifest"
check_sha256 \
  "${DUCA_PAPER_PRETRAIN_PATH}" \
  "${DUCA_PAPER_PRETRAIN_SHA256}" \
  "VideoMAE initialization"
check_sha256 \
  "${DUCA_PAPER_ANNOTATION_PATH}" \
  "${DUCA_PAPER_ANNOTATION_SHA256}" \
  "THUMOS14 annotation"
check_sha256 \
  "${DUCA_PAPER_CLASS_MAP_PATH}" \
  "${DUCA_PAPER_CLASS_MAP_SHA256}" \
  "THUMOS14 class map"
check_sha256 \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "real natural-short-window heavy-backbone gate"
check_sha256 \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "clean Linux/PyTorch code gate"
check_sha256 \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  "${DUCA_PAPER_NUMERIC_GATE_SHA256}" \
  "production-like learned numeric gate"
check_sha256 \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}" \
  "exact-211 physical UID gate"
python -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
python -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}"
python -m tools.bata.validate_duca_paper_numeric_gate \
  --receipt "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_NUMERIC_GATE_SHA256}"
python -m tools.bata.validate_duca_paper_exact211_uid_gate \
  --receipt "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}"

readarray -t config_values < <(python - \
  "${DUCA_PAPER_MATRIX_MANIFEST}" \
  "${DUCA_PAPER_CONFIG}" \
  "${DUCA_PAPER_ARM}" \
  "${DUCA_PAPER_SEED}" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${DUCA_PAPER_ANNOTATION_SHA256}" \
  "${DUCA_PAPER_CLASS_MAP_SHA256}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  "${DUCA_PAPER_NUMERIC_GATE_SHA256}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

from mmengine.config import Config
from tools.bata import duca_paper_training

(
    manifest_path,
    config_path,
    arm,
    seed,
    commit,
    annotation_sha,
    class_map_sha,
    code_gate_path,
    code_gate_sha,
    short_gate_path,
    short_gate_sha,
    numeric_gate_path,
    numeric_gate_sha,
    exact211_gate_path,
    exact211_gate_sha,
) = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
cfg = Config.fromfile(config_path)
contract = duca_paper_training.validate_static_config(cfg)
repo = pathlib.Path.cwd().resolve()
source = pathlib.Path(config_path).resolve()
relative = source.relative_to(repo).as_posix()
record = manifest.get("configs", {}).get(arm, {})
code_gate = manifest.get("prerequisite_gates", {}).get(
    "clean_linux_pytorch_code", {}
)
short_gate = manifest.get("prerequisite_gates", {}).get(
    "real_natural_short_window_heavy_backbone", {}
)
numeric_gate = manifest.get("prerequisite_gates", {}).get(
    "production_like_learned_exactk_numeric", {}
)
exact211_gate = manifest.get("prerequisite_gates", {}).get(
    "exact211_physical_uid_metadata", {}
)
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
resolved = duca_paper_training.canonical_sha256(cfg.to_dict())
if (
    manifest.get("schema_version") != duca_paper_training.MATRIX_SCHEMA
    or manifest.get("status") != "frozen"
    or manifest.get("git_commit") != commit
    or {"arm": arm, "seed": int(seed)} not in manifest.get("cells", [])
    or contract.get("variant") != arm
    or record.get("path") != relative
    or record.get("sha256") != sha(source)
    or record.get("resolved_sha256") != resolved
    or manifest.get("assets", {}).get("annotation_sha256") != annotation_sha
    or manifest.get("assets", {}).get("class_map_sha256") != class_map_sha
    or code_gate.get("git_commit") != commit
    or code_gate.get("status") != "passed"
    or pathlib.Path(str(code_gate.get("path", ""))).resolve()
    != pathlib.Path(code_gate_path).resolve()
    or code_gate.get("sha256") != code_gate_sha
    or short_gate.get("git_commit") != commit
    or short_gate.get("status") != "passed"
    or pathlib.Path(str(short_gate.get("path", ""))).resolve()
    != pathlib.Path(short_gate_path).resolve()
    or short_gate.get("sha256") != short_gate_sha
    or numeric_gate.get("git_commit") != commit
    or numeric_gate.get("status") != "passed"
    or pathlib.Path(str(numeric_gate.get("path", ""))).resolve()
    != pathlib.Path(numeric_gate_path).resolve()
    or numeric_gate.get("sha256") != numeric_gate_sha
    or exact211_gate.get("git_commit") != commit
    or exact211_gate.get("status") != "passed"
    or pathlib.Path(str(exact211_gate.get("path", ""))).resolve()
    != pathlib.Path(exact211_gate_path).resolve()
    or exact211_gate.get("sha256") != exact211_gate_sha
):
    raise SystemExit("Stage-A cell differs from the frozen matrix")
for path, expected, label in (
    (cfg.evaluation.ground_truth_filename, annotation_sha, "annotation"),
    (cfg.dataset.test.class_map, class_map_sha, "class map"),
):
    if not pathlib.Path(path).is_file() or sha(path) != expected:
        raise SystemExit(f"runtime {label} binding drift")
print(resolved)
print(int(cfg.duca_paper_cell.evaluation_heavy_k))
PY
)
[[ "${#config_values[@]}" == 2 ]] || fail "failed to resolve the cell config"

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_PAPER_STAGE_A_CELL] PRECHECK PASS ${DUCA_PAPER_ARM} seed${DUCA_PAPER_SEED}"
  exit 0
fi

mkdir -p "${DUCA_PAPER_CELL_ROOT}"
export DUCA_EXPECTED_COMMIT="${DUCA_PAPER_EXPECTED_COMMIT}"
export DUCA_PAPER_RESOLVED_CONFIG_SHA256="${config_values[0]}"

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-train" --nproc_per_node=2 tools/train.py \
  "${DUCA_PAPER_CONFIG}" \
  --seed "${DUCA_PAPER_SEED}" \
  --id 0 \
  --cfg-options \
  "work_dir=${DUCA_PAPER_CELL_ROOT}/train" \
  "model.backbone.custom.pretrain=${DUCA_PAPER_PRETRAIN_PATH}"

train_root="${DUCA_PAPER_CELL_ROOT}/train/gpu2_id0"
audit="${train_root}/duca_paper_training_audit.json"
full_checkpoint="${train_root}/checkpoint/epoch_59.pth"
checkpoint="${train_root}/checkpoint/terminal_ema.pth"
python -m tools.bata.compact_duca_rime_checkpoint \
  --source "${full_checkpoint}" \
  --output "${checkpoint}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --remove-source

training_receipt="${DUCA_PAPER_CELL_ROOT}/training_receipt.json"
python - \
  "${audit}" \
  "${checkpoint}" \
  "${DUCA_PAPER_ARM}" \
  "${DUCA_PAPER_SEED}" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${DUCA_PAPER_MATRIX_MANIFEST}" \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  "${DUCA_PAPER_NUMERIC_GATE_SHA256}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}" \
  "${training_receipt}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    audit_path,
    checkpoint_path,
    arm,
    seed,
    commit,
    matrix_path,
    matrix_sha,
    code_gate_path,
    code_gate_sha,
    short_gate_path,
    short_gate_sha,
    numeric_gate_path,
    numeric_gate_sha,
    exact211_gate_path,
    exact211_gate_sha,
    output,
) = sys.argv[1:]
compaction_path = checkpoint_path + ".receipt.json"
for path in (
    audit_path,
    checkpoint_path,
    compaction_path,
    matrix_path,
    code_gate_path,
    short_gate_path,
    numeric_gate_path,
    exact211_gate_path,
):
    if not os.path.isfile(path):
        raise SystemExit(f"terminal training evidence is missing: {path}")
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
audit = json.load(open(audit_path, encoding="utf-8"))
compaction = json.load(open(compaction_path, encoding="utf-8"))
updates = audit.get("update_audit", {})
loader = audit.get("train_loader_contract", {})
budget = audit.get("budget_ledger_summary", {})
expected_budget_rows = 0 if arm == "dense" else 12000
if (
    audit.get("status") != "complete"
    or audit.get("git_commit") != commit
    or audit.get("variant") != arm
    or int(audit.get("seed", -1)) != int(seed)
    or audit.get("training_consumes_validation") is not False
    or int(audit.get("train_video_count", -1)) != 200
    or int(audit.get("evaluation_video_count", -1)) != 211
    or int(audit.get("world_size", -1)) != 2
    or int(audit.get("global_batch_size", -1)) != 2
    or int(loader.get("dataset", {}).get("video_count", -1)) != 200
    or int(loader.get("per_video_exposure_count", -1)) != 60
    or int(updates.get("successful_optimizer_updates", -1)) != 6000
    or int(updates.get("scheduler_updates", -1)) != 6000
    or int(updates.get("ema_updates", -1)) != 6000
    or budget.get("schema_version") != "duca_paper_committed_budget_summary_v1"
    or budget.get("arm") != arm
    or int(budget.get("epochs", -1)) != 60
    or int(budget.get("row_count", -1)) != expected_budget_rows
    or len(str(budget.get("budget_summary_sha256", ""))) != 64
    or compaction.get("schema_version") != "duca_rime_compact_checkpoint_receipt_v1"
    or compaction.get("status") != "passed"
    or compaction.get("evaluation_equivalent") is not True
    or compaction.get("training_resume_supported") is not False
    or sha(matrix_path) != matrix_sha
    or sha(code_gate_path) != code_gate_sha
    or sha(short_gate_path) != short_gate_sha
    or sha(numeric_gate_path) != numeric_gate_sha
    or sha(exact211_gate_path) != exact211_gate_sha
):
    raise SystemExit("terminal training evidence violates the frozen Stage-A contract")
payload = {
    "schema_version": "duca_paper_full200_training_receipt_v2",
    "status": "passed",
    "git_commit": commit,
    "arm": arm,
    "seed": int(seed),
    "train_video_count": 200,
    "evaluation_video_count": 211,
    "world_size": 2,
    "global_batch_size": 2,
    "successful_optimizer_updates": 6000,
    "training_consumed_validation": False,
    "budget_ledger_summary": budget,
    "budget_summary_sha256": budget["budget_summary_sha256"],
    "training_audit_path": str(pathlib.Path(audit_path).resolve()),
    "training_audit_sha256": sha(audit_path),
    "checkpoint_path": str(pathlib.Path(checkpoint_path).resolve()),
    "checkpoint_sha256": sha(checkpoint_path),
    "checkpoint_epoch": 59,
    "checkpoint_state_key": "state_dict_ema",
    "checkpoint_compaction_receipt_path": str(pathlib.Path(compaction_path).resolve()),
    "checkpoint_compaction_receipt_sha256": sha(compaction_path),
    "matrix_manifest_path": str(pathlib.Path(matrix_path).resolve()),
    "matrix_manifest_sha256": matrix_sha,
    "code_gate_path": str(pathlib.Path(code_gate_path).resolve()),
    "code_gate_sha256": code_gate_sha,
    "short_window_gate_path": str(pathlib.Path(short_gate_path).resolve()),
    "short_window_gate_sha256": short_gate_sha,
    "numeric_gate_path": str(pathlib.Path(numeric_gate_path).resolve()),
    "numeric_gate_sha256": numeric_gate_sha,
    "exact211_uid_gate_path": str(pathlib.Path(exact211_gate_path).resolve()),
    "exact211_uid_gate_sha256": exact211_gate_sha,
    "single_seed_claim_allowed": False,
}
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
training_receipt_sha="$(sha256sum "${training_receipt}" | awk '{print $1}')"
export DUCA_PAPER_TRAINING_RECEIPT="${training_receipt}"
export DUCA_PAPER_TRAINING_RECEIPT_SHA256="${training_receipt_sha}"

terminal_evaluation="${DUCA_PAPER_CELL_ROOT}/terminal_evaluation.json"
if [[ "${DUCA_PAPER_ARM}" == dense ]]; then
  unset DUCA_RIME_INFERENCE_LEDGER_ROOT || true
  unset DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256 || true
else
  export DUCA_RIME_INFERENCE_LEDGER_ROOT="${DUCA_PAPER_CELL_ROOT}/eval_budget_ledger"
  export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256="${DUCA_PAPER_MATRIX_MANIFEST_SHA256}"
  [[ ! -e "${DUCA_RIME_INFERENCE_LEDGER_ROOT}" ]] \
    || fail "a fresh evaluation budget ledger root is required"
fi
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-eval" --nproc_per_node=1 tools/test.py \
  "${DUCA_PAPER_CONFIG}" \
  --checkpoint "${checkpoint}" \
  --seed "${DUCA_PAPER_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${terminal_evaluation}" \
  --cfg-options \
  "work_dir=${DUCA_PAPER_CELL_ROOT}/eval" \
  "model.backbone.custom.pretrain=${DUCA_PAPER_PRETRAIN_PATH}"

python - \
  "${terminal_evaluation}" \
  "${training_receipt}" \
  "${DUCA_PAPER_ARM}" \
  "${DUCA_PAPER_SEED}" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${config_values[1]}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${DUCA_PAPER_NUMERIC_GATE_JSON}" \
  "${DUCA_PAPER_NUMERIC_GATE_SHA256}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_JSON}" \
  "${DUCA_PAPER_EXACT211_UID_GATE_SHA256}" \
  "${DUCA_PAPER_CELL_ROOT}/cell.receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    evaluation_path,
    training_path,
    arm,
    seed,
    commit,
    expected_k,
    code_gate_path,
    code_gate_sha,
    short_gate_path,
    short_gate_sha,
    numeric_gate_path,
    numeric_gate_sha,
    exact211_gate_path,
    exact211_gate_sha,
    output,
) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
evaluation = json.load(open(evaluation_path, encoding="utf-8"))
exact = evaluation.get("exact211_execution", {})
budget = evaluation.get("budget_execution", {})
training_identity = evaluation.get("training_identity", {})
if (
    evaluation.get("schema_version") != "duca_paper_full211_terminal_evaluation_v2"
    or evaluation.get("git_commit") != commit
    or evaluation.get("variant") != arm
    or int(evaluation.get("seed", -1)) != int(seed)
    or int(evaluation.get("train_video_count", -1)) != 200
    or int(evaluation.get("evaluation_video_count", -1)) != 211
    or evaluation.get("training_consumed_validation") is not False
    or evaluation.get("runtime_gt_input_to_selector") is not False
    or int(evaluation.get("evaluation_heavy_k", -1)) != int(expected_k)
    or evaluation.get("runtime_heavy_k_contract_enforced") is not True
    or exact.get("official_open_tad_pipeline_completed") is not True
    or int(exact.get("evaluation_video_count", -1)) != 211
    or training_identity.get("training_receipt_sha256") != sha(training_path)
    or sha(code_gate_path) != code_gate_sha
    or sha(short_gate_path) != short_gate_sha
    or sha(numeric_gate_path) != numeric_gate_sha
    or sha(exact211_gate_path) != exact211_gate_sha
    or budget.get("schema_version") != "duca_paper_exact211_budget_execution_v1"
    or budget.get("arm") != arm
    or budget.get("requested_budget_is_dynamic") is not False
    or (
        arm != "dense"
        and len(str(budget.get("window_budget_vector_sha256", ""))) != 64
    )
    or not isinstance(evaluation.get("metrics"), dict)
):
    raise SystemExit("terminal evaluation violates the exact-211 Stage-A contract")
payload = {
    "schema_version": "duca_paper_stage_a_cell_receipt_v2",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "arm": arm,
    "seed": int(seed),
    "training_receipt_path": str(pathlib.Path(training_path).resolve()),
    "training_receipt_sha256": sha(training_path),
    "terminal_evaluation_path": str(pathlib.Path(evaluation_path).resolve()),
    "terminal_evaluation_sha256": sha(evaluation_path),
    "code_gate_path": str(pathlib.Path(code_gate_path).resolve()),
    "code_gate_sha256": code_gate_sha,
    "short_window_gate_path": str(pathlib.Path(short_gate_path).resolve()),
    "short_window_gate_sha256": short_gate_sha,
    "numeric_gate_path": str(pathlib.Path(numeric_gate_path).resolve()),
    "numeric_gate_sha256": numeric_gate_sha,
    "exact211_uid_gate_path": str(pathlib.Path(exact211_gate_path).resolve()),
    "exact211_uid_gate_sha256": exact211_gate_sha,
    "evaluation_budget_execution_sha256": budget["content_sha256"],
    "window_budget_vector_sha256": budget.get("window_budget_vector_sha256"),
    "exact_train_video_count": 200,
    "exact_evaluation_video_count": 211,
    "paper_claim_ready": False,
    "requires_complete_three_seed_matrix": True,
}
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "[DUCA_PAPER_STAGE_A_CELL] PASS ${DUCA_PAPER_ARM} seed${DUCA_PAPER_SEED}"
