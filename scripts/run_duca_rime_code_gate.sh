#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_CODE_GATE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="${DUCA_RIME_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPECTED_COMMIT="${DUCA_RIME_EXPECTED_COMMIT:-}"
OUTPUT_ROOT="${DUCA_RIME_CODE_GATE_ROOT:-}"
PYTHON="${PYTHON:-python}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the code gate must run inside Slurm"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "an exact expected commit is required"
[[ -n "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}" ]] || fail "a fresh external output root is required"
[[ -d "${REPO_ROOT}" ]] || fail "repository snapshot is missing"
cd "${REPO_ROOT}"

if [[ -d .git ]]; then
  [[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "Git commit drift"
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Git tree is dirty"
else
  SOURCE_MANIFEST="${REPO_ROOT}/.codex_source_manifest"
  [[ -f "${SOURCE_MANIFEST}" ]] || fail "archive source manifest is missing"
  grep -qx "commit=${EXPECTED_COMMIT}" "${SOURCE_MANIFEST}" \
    || fail "archive commit drift"
  OVERLAY_MANIFEST="${DUCA_RIME_OVERLAY_SHA256_MANIFEST:-}"
  [[ -f "${OVERLAY_MANIFEST}" ]] || fail "overlay SHA-256 manifest is required"
  sha256sum -c "${OVERLAY_MANIFEST}"
fi

mkdir -p "${OUTPUT_ROOT}/fixtures" "${OUTPUT_ROOT}/logs"
printf 'fixture-block-entry\n' > "${OUTPUT_ROOT}/fixtures/train_block.txt"
printf 'fixture-block-entry\n' > "${OUTPUT_ROOT}/fixtures/development_block.txt"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/targets.jsonl"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/replay.jsonl"
printf '{}\n' > "${OUTPUT_ROOT}/fixtures/protocol.json"

export DUCA_RIME_TRAIN_BLOCK_LIST="${OUTPUT_ROOT}/fixtures/train_block.txt"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${OUTPUT_ROOT}/fixtures/development_block.txt"
export DUCA_RIME_TARGETS_JSONL="${OUTPUT_ROOT}/fixtures/targets.jsonl"
export DUCA_RIME_TARGETS_SHA256="$(sha256sum "${DUCA_RIME_TARGETS_JSONL}" | awk '{print $1}')"
export DUCA_RIME_REPLAY_JSONL="${OUTPUT_ROOT}/fixtures/replay.jsonl"
export DUCA_RIME_REPLAY_SHA256="$(sha256sum "${DUCA_RIME_REPLAY_JSONL}" | awk '{print $1}')"
export DUCA_RIME_BUDGET_PROTOCOL_JSON="${OUTPUT_ROOT}/fixtures/protocol.json"
export DUCA_RIME_BUDGET_PROTOCOL_SHA256="$(
  sha256sum "${DUCA_RIME_BUDGET_PROTOCOL_JSON}" | awk '{print $1}'
)"

"${PYTHON}" -m py_compile \
  tools/train.py \
  tools/bata/train_lowres_action_probe.py \
  tools/bata/create_duca_rime_splits.py \
  tools/bata/duca_rime_phase2.py \
  tools/bata/build_duca_rime_budget_replay.py \
  opentad/models/duca/rime.py \
  opentad/models/selectors/duca_rime_frame_selector.py \
  opentad/models/backbones/backbone_wrapper.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/detectors/tridet.py \
  opentad/models/dense_heads/tridet_head.py

"${PYTHON}" -m pytest \
  tests/test_duca_rime.py \
  tests/test_duca_rime_phase2.py \
  tests/test_duca_rime_tridet.py \
  tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  -q 2>&1 | tee "${OUTPUT_ROOT}/logs/pytest.out"

"${PYTHON}" - <<'PY' 2>&1 | tee "${OUTPUT_ROOT}/logs/configs.out"
from mmengine.config import Config

configs = (
    "configs/adatad/thumos/duca_rime_uniform_fixed384_total60.py",
    "configs/adatad/thumos/duca_rime_fixed_bound_total60.py",
    "configs/adatad/thumos/duca_rime_dynamic_no_risk_total60.py",
    "configs/adatad/thumos/duca_rime_dynamic_shuffle_total60.py",
    "configs/adatad/thumos/duca_adaptok_tad_direct_total60.py",
    "configs/adatad/thumos/duca_rime_full_total60.py",
    "configs/adatad/thumos/duca_rime_full_tridet_total60.py",
    "configs/adatad/thumos/duca_rime_uniform_same_k_eval.py",
)
for path in configs:
    cfg = Config.fromfile(path)
    assert cfg.solver.train.batch_size == 1
    if "duca_rime_contract" in cfg:
        assert cfg.duca_rime_contract.pad_to_kmax is False
        assert cfg.duca_rime_contract.execution_quantum == 16
    print(f"CONFIG_OK {path}")
PY

printf '%s\n' \
  "schema=duca_rime_code_gate_v1" \
  "status=passed" \
  "commit=${EXPECTED_COMMIT}" \
  "slurm_job_id=${SLURM_JOB_ID}" \
  > "${OUTPUT_ROOT}/gate.receipt"

echo "[DUCA_RIME_CODE_GATE] PASS ${OUTPUT_ROOT}/gate.receipt"
