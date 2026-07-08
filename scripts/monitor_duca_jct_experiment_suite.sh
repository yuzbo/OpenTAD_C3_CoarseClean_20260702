#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DEPLOYMENT_SUMMARY="${1:-${DEPLOYMENT_SUMMARY:-}}"
if [[ -z "${DEPLOYMENT_SUMMARY}" ]]; then
  echo "usage: $0 /path/to/deployment_summary.json" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
RUN_ROOT="$("${PYTHON}" - "${DEPLOYMENT_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(payload.get("run_root") or str(Path(sys.argv[1]).parent))
PY
)"

SQUEUE_TEXT="${SQUEUE_TEXT:-${RUN_ROOT}/squeue_snapshot.tsv}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/duca_jct_suite_monitor.summary.json}"
mkdir -p "$(dirname "${SQUEUE_TEXT}")" "$(dirname "${OUTPUT_JSON}")"

if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER:-$(whoami)}" -h -o "%i|%j|%T|%r" > "${SQUEUE_TEXT}" || true
else
  : > "${SQUEUE_TEXT}"
fi

"${PYTHON}" tools/bata/monitor_duca_jct_experiment_suite.py \
  --deployment-summary "${DEPLOYMENT_SUMMARY}" \
  --squeue-text "${SQUEUE_TEXT}" \
  --output-json "${OUTPUT_JSON}" \
  --print-table

echo "[DUCA_JCT_MONITOR] summary_json=${OUTPUT_JSON}"
