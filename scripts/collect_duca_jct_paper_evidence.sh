#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MONITOR_SUMMARY="${1:-${MONITOR_SUMMARY:-}}"
if [[ -z "${MONITOR_SUMMARY}" ]]; then
  echo "usage: $0 /path/to/duca_jct_suite_monitor.summary.json [matched_baselines.json]" >&2
  exit 2
fi

BASELINE_SUMMARY="${2:-${BASELINE_SUMMARY:-}}"
PYTHON="${PYTHON:-python}"
RUN_ROOT="$("${PYTHON}" - "${MONITOR_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(payload.get("run_root") or str(Path(sys.argv[1]).parent))
PY
)"

OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/duca_jct_paper_evidence.summary.json}"
OUTPUT_TSV="${OUTPUT_TSV:-${RUN_ROOT}/duca_jct_paper_evidence.table.tsv}"

args=(
  --monitor-summary "${MONITOR_SUMMARY}"
  --output-json "${OUTPUT_JSON}"
  --output-tsv "${OUTPUT_TSV}"
)
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi

"${PYTHON}" tools/bata/collect_duca_jct_paper_evidence.py "${args[@]}"
echo "[DUCA_JCT_EVIDENCE] summary_json=${OUTPUT_JSON}"
echo "[DUCA_JCT_EVIDENCE] table_tsv=${OUTPUT_TSV}"
