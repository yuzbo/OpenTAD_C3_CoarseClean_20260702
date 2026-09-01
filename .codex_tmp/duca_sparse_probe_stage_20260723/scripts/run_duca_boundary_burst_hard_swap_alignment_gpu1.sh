#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_BURST_R4_ALIGNMENT][FAIL] $*" >&2; exit 1; }

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TERMINAL_SUITE="${DUCA_BOUNDARY_BURST_TERMINAL_SUITE:-}"
TERMINAL_SUITE_SHA256="${DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256:-}"
OUTPUT_ROOT="${DUCA_BOUNDARY_BURST_ALIGNMENT_ROOT:-}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || fail "one Slurm GPU allocation is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${TERMINAL_SUITE}" ]] || fail "sealed terminal U/G0 suite is missing"
[[ "${TERMINAL_SUITE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "terminal suite SHA256 is invalid"
[[ "$(sha256sum "${TERMINAL_SUITE}" | awk '{print $1}')" == "${TERMINAL_SUITE_SHA256}" ]] \
  || fail "terminal suite hash drift"
[[ -n "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}" ]] || fail "fresh alignment root is required"
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] \
  || fail "exactly one logical GPU is required"

mkdir -p "${OUTPUT_ROOT}/contracts" "${OUTPUT_ROOT}/full_model" \
  "${OUTPUT_ROOT}/shards"
CONTEXT="${OUTPUT_ROOT}/alignment_context.json"
ARTIFACT="${OUTPUT_ROOT}/alignment.json"

"${PYTHON}" -m tools.bata.duca_boundary_burst_hard_swap_alignment \
  prepare-context \
  --expected-commit "${EXPECTED_COMMIT}" \
  --terminal-suite "${TERMINAL_SUITE}" \
  --terminal-suite-sha256 "${TERMINAL_SUITE_SHA256}" \
  --output-json "${CONTEXT}"
CONTEXT_SHA256="$(sha256sum "${CONTEXT}" | awk '{print $1}')"

readarray -t route < <("${PYTHON}" - "${CONTEXT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
variants = payload["authorized_variants"]
for stage in ("G1", "G2"):
    matches = [(variant, row) for variant, row in variants.items() if row["stage"] == stage]
    if len(matches) != 1:
        raise SystemExit(f"context does not contain exactly one {stage} variant")
    variant, row = matches[0]
    print(variant)
    print(row["config_path"])
print(payload["adatad_pretrain"]["path"])
print(payload["adatad_pretrain"]["sha256"])
print(payload["p0_initialization"]["path"])
print(payload["p0_initialization"]["sha256"])
print(payload["p0_initialization"]["epoch"])
PY
)
G1_VARIANT="${route[0]}"
G1_CONFIG="${route[1]}"
G2_VARIANT="${route[2]}"
G2_CONFIG="${route[3]}"
ADATAD_PRETRAIN="${route[4]}"
ADATAD_PRETRAIN_SHA256="${route[5]}"
export DUCA_FRONTEND_CHECKPOINT="${route[6]}"
export DUCA_FRONTEND_CHECKPOINT_SHA256="${route[7]}"
export DUCA_FRONTEND_CHECKPOINT_EPOCH="${route[8]}"

for item in "${G1_VARIANT}:${G1_CONFIG}" "${G2_VARIANT}:${G2_CONFIG}"; do
  IFS=: read -r variant config <<<"${item}"
  contract="${OUTPUT_ROOT}/contracts/${variant}.json"
  gate="${OUTPUT_ROOT}/full_model/${variant}.json"
  "${PYTHON}" tools/bata/validate_duca_protected_e2e_official60.py \
    --config "${config}" --output-json "${contract}"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-r4-gate-${SLURM_JOB_ID}-${variant}" \
    tools/bata/run_duca_protected_e2e_exact_full_model_gate.py \
    --config "${config}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --adatad-pretrain "${ADATAD_PRETRAIN}" \
    --adatad-pretrain-sha256 "${ADATAD_PRETRAIN_SHA256}" \
    --output-json "${gate}"
done

for stratum in short medium long; do
  output="${OUTPUT_ROOT}/shards/${stratum}.json"
  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-r4-swap-${SLURM_JOB_ID}-${stratum}" \
    -m tools.bata.duca_boundary_burst_hard_swap_alignment \
    run-shard \
    --stratum "${stratum}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --context "${CONTEXT}" \
    --context-sha256 "${CONTEXT_SHA256}" \
    --output-json "${output}"
done

G1_GATE="${OUTPUT_ROOT}/full_model/${G1_VARIANT}.json"
G2_GATE="${OUTPUT_ROOT}/full_model/${G2_VARIANT}.json"
"${PYTHON}" -m tools.bata.duca_boundary_burst_hard_swap_alignment aggregate \
  --expected-commit "${EXPECTED_COMMIT}" \
  --context "${CONTEXT}" \
  --context-sha256 "${CONTEXT_SHA256}" \
  --short-shard "${OUTPUT_ROOT}/shards/short.json" \
  --short-shard-sha256 "$(sha256sum "${OUTPUT_ROOT}/shards/short.json" | awk '{print $1}')" \
  --medium-shard "${OUTPUT_ROOT}/shards/medium.json" \
  --medium-shard-sha256 "$(sha256sum "${OUTPUT_ROOT}/shards/medium.json" | awk '{print $1}')" \
  --long-shard "${OUTPUT_ROOT}/shards/long.json" \
  --long-shard-sha256 "$(sha256sum "${OUTPUT_ROOT}/shards/long.json" | awk '{print $1}')" \
  --g1-variant "${G1_VARIANT}" \
  --g1-gate "${G1_GATE}" \
  --g1-gate-sha256 "$(sha256sum "${G1_GATE}" | awk '{print $1}')" \
  --g2-variant "${G2_VARIANT}" \
  --g2-gate "${G2_GATE}" \
  --g2-gate-sha256 "$(sha256sum "${G2_GATE}" | awk '{print $1}')" \
  --output-json "${ARTIFACT}"

ARTIFACT_SHA256="$(sha256sum "${ARTIFACT}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.duca_boundary_burst_hard_swap_alignment validate \
  --expected-commit "${EXPECTED_COMMIT}" \
  --artifact "${ARTIFACT}" \
  --artifact-sha256 "${ARTIFACT_SHA256}"
printf '%s\n' "${ARTIFACT_SHA256}" > "${ARTIFACT}.sha256.tmp"
mv -f "${ARTIFACT}.sha256.tmp" "${ARTIFACT}.sha256"

echo "[DUCA_BURST_R4_ALIGNMENT] passed ${ARTIFACT}"
