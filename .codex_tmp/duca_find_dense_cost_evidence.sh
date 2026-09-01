#!/usr/bin/env bash
set -euo pipefail
BASE=/data/run01/sczc063/yuzibo
find ${BASE}/projects -type f -path '*/checkpoint/epoch_59.pth' -printf '%p\n' 2>/dev/null | grep -Ei 'dense|e2e|adatad' | head -80 || true
