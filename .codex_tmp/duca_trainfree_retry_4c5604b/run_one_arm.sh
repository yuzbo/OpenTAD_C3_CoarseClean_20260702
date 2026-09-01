#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant is required}"
snapshot="${DUCA_SNAPSHOT:?DUCA_SNAPSHOT is required}"
run_root="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required}"
expected_commit="${DUCA_EXPECTED_COMMIT:?DUCA_EXPECTED_COMMIT is required}"
pretrain_path="${DUCA_ADATAD_PRETRAIN_PATH:?DUCA_ADATAD_PRETRAIN_PATH is required}"
pretrain_sha256="${DUCA_ADATAD_PRETRAIN_SHA256:?DUCA_ADATAD_PRETRAIN_SHA256 is required}"
python_bin="/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python"
arm_root="${run_root}/arms/${variant}"
export TORCH_HOME="/data/run01/sczc063/yuzibo/.cache/torch"

if [[ "${variant}" == "trainfree_slowfast_fast_fusion_r2q3" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  cd "${snapshot}"
  "${python_bin}" - <<'PY'
import json
import torch

from tools.bata.train_lowres_action_probe import C3SlowFastFastFrozenProbe

probe = C3SlowFastFastFrozenProbe(pretrained=True).to("cuda").eval()
frames = torch.rand(1, 16, 3, 64, 64, device="cuda")
valid = torch.ones(1, 16, dtype=torch.bool, device="cuda")
with torch.no_grad():
    output = probe(frames, valid, return_hidden=True)
hidden = output["hidden"]
if hidden.ndim != 3 or hidden.shape[:2] != (1, 16):
    raise SystemExit(f"invalid Fast-pathway hidden shape: {tuple(hidden.shape)}")
if any(parameter.requires_grad for parameter in probe.parameters()):
    raise SystemExit("Fast-pathway frozen probe exposes trainable parameters")
print(json.dumps({
    "ok": True,
    "pathway": "fast_only",
    "slow_path_executed": False,
    "lateral_fusion_executed": False,
    "hidden_shape": list(hidden.shape),
}, sort_keys=True))
PY
fi

export DUCA_REPO_ROOT="${snapshot}"
export DUCA_EXPECTED_COMMIT="${expected_commit}"
export DUCA_INDEPENDENT_VARIANT="${variant}"
export DUCA_INDEPENDENT_ARM_ROOT="${arm_root}"
export DUCA_ADATAD_PRETRAIN_PATH="${pretrain_path}"
export DUCA_ADATAD_PRETRAIN_SHA256="${pretrain_sha256}"

exec bash "${snapshot}/scripts/run_duca_independent_official60_gpu1.sh"
