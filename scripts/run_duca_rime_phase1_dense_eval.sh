#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE1_DENSE][FAIL] $*" >&2
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
  DUCA_RIME_PHASE1_DENSE_ROOT \
  DUCA_RIME_PHASE1_DENSE_CONFIG \
  DUCA_RIME_PHASE1_DENSE_VARIANT \
  DUCA_RIME_PHASE1_DENSE_CHECKPOINT \
  DUCA_RIME_PHASE1_DENSE_CHECKPOINT_SHA256 \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_PHASE1_SPLIT_ROLE \
  DUCA_RIME_EVAL_SEED \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-1 dense evaluation must run inside Slurm"
[[ "${DUCA_RIME_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected Git commit is required"
[[ "${DUCA_RIME_PHASE1_DENSE_VARIANT}" =~ ^(released_dense|local_dense)$ ]] \
  || fail "dense variant must be released_dense or local_dense"
[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]] \
  || fail "Phase-1 dense evaluation requires a complete Git worktree"
[[ ! -e "${DUCA_RIME_PHASE1_DENSE_ROOT}" ]] \
  || fail "a fresh Phase-1 dense root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_PHASE1_DENSE_CHECKPOINT}" \
  "${DUCA_RIME_PHASE1_DENSE_CHECKPOINT_SHA256}" \
  "dense checkpoint"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "RIME split manifest"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE initialization"
python tools/bata/create_duca_rime_splits.py \
  --validate-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --expected-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  > /dev/null

readarray -t split_values < <(
  python - \
    "${DUCA_RIME_SPLIT_MANIFEST}" \
    "${DUCA_RIME_PHASE1_SPLIT_ROLE}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
if role not in manifest["train_roles"]:
    raise SystemExit("Phase-1 dense split role is not registered")
print(manifest["train_roles"][role]["block_list_path"])
print(manifest["train_roles"][role]["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 2 ]] || fail "failed to resolve Phase-1 split role"
check_sha256 "${split_values[0]}" "${split_values[1]}" "Phase-1 block list"
export DUCA_RIME_PHASE1_EVAL_BLOCK_LIST="${split_values[0]}"
export DUCA_EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT}"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python - \
    "${DUCA_RIME_PHASE1_DENSE_CONFIG}" \
    "${DUCA_RIME_PRETRAIN_PATH}" <<'PY'
import pathlib
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
cfg.model.backbone.custom.pretrain = sys.argv[2]
assert cfg.workflow.formal_protocol == "duca_rime_phase1_dense_control_v1"
assert cfg.duca_rime_baseline_contract.phase == 1
assert cfg.duca_rime_baseline_contract.checkpoint_compatibility_mode == "strict_exact_v1"
assert cfg.evaluation.subset == "training"
assert pathlib.Path(cfg.model.backbone.custom.pretrain).resolve() == pathlib.Path(
    sys.argv[2]
).resolve()
PY
  echo "[DUCA_RIME_PHASE1_DENSE] PRECHECK PASS"
  exit 0
fi

mkdir -p "${DUCA_RIME_PHASE1_DENSE_ROOT}"
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}" --nproc_per_node=1 tools/test.py \
  "${DUCA_RIME_PHASE1_DENSE_CONFIG}" \
  --checkpoint "${DUCA_RIME_PHASE1_DENSE_CHECKPOINT}" \
  --seed "${DUCA_RIME_EVAL_SEED}" \
  --id 0 \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json \
  "${DUCA_RIME_PHASE1_DENSE_ROOT}/terminal_evaluation.json" \
  --cfg-options \
  "work_dir=${DUCA_RIME_PHASE1_DENSE_ROOT}/runtime" \
  "model.backbone.custom.pretrain=${DUCA_RIME_PRETRAIN_PATH}"

python -m tools.bata.evaluate_duca_rime_predictions \
  --terminal-evaluation \
  "${DUCA_RIME_PHASE1_DENSE_ROOT}/terminal_evaluation.json" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --phase 1 \
  --split-role "${DUCA_RIME_PHASE1_SPLIT_ROLE}" \
  --short-max-seconds "${DUCA_RIME_SHORT_MAX_SECONDS}" \
  --medium-max-seconds "${DUCA_RIME_MEDIUM_MAX_SECONDS}" \
  --output "${DUCA_RIME_PHASE1_DENSE_ROOT}/localization_metrics.json"

printf '%s\n' \
  "schema=duca_rime_phase1_dense_evaluation_receipt_v1" \
  "status=passed" \
  "commit=${DUCA_RIME_EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  "variant=${DUCA_RIME_PHASE1_DENSE_VARIANT}" \
  "split_role=${DUCA_RIME_PHASE1_SPLIT_ROLE}" \
  "seed=${DUCA_RIME_EVAL_SEED}" \
  "checkpoint_sha256=${DUCA_RIME_PHASE1_DENSE_CHECKPOINT_SHA256}" \
  "pretrain_sha256=${DUCA_RIME_PRETRAIN_SHA256}" \
  "terminal_evaluation_sha256=$(sha256sum "${DUCA_RIME_PHASE1_DENSE_ROOT}/terminal_evaluation.json" | awk '{print $1}')" \
  "localization_metrics_sha256=$(sha256sum "${DUCA_RIME_PHASE1_DENSE_ROOT}/localization_metrics.json" | awk '{print $1}')" \
  > "${DUCA_RIME_PHASE1_DENSE_ROOT}/evaluation.receipt"
echo "[DUCA_RIME_PHASE1_DENSE] PASS ${DUCA_RIME_PHASE1_DENSE_ROOT}"
