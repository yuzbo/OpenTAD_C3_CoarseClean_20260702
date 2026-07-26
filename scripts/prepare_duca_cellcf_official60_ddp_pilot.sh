#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DUCA_CELLCF_TRAINING_PROFILE=official60
exec bash "${REPO_ROOT}/scripts/prepare_duca_cellcf_ddp_pilot.sh" "$@"
