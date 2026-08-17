#!/usr/bin/env bash
set -euo pipefail
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  python tools/bata/validate_duca_semantic_indirect_n16r4.py
  exit 0
fi
future_pilot_argv=(python tools/bata/validate_duca_semantic_indirect_n16r4.py --config configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py --arm actionness_boundary_dynamic_k_headline --pilot)
future_full_argv=(python tools/bata/validate_duca_semantic_indirect_n16r4.py --config configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py --arm actionness_boundary_dynamic_k_headline --full)
case "${1:-}" in pilot|full) echo "future declared $1 argv only; execution intentionally not implemented"; exit 2;; *) echo "PRECHECK_ONLY=1 or declared pilot/full" >&2; exit 2;; esac
