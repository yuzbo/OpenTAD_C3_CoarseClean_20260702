from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tools" / "bata" / "run_duca_online_plugin_smoke.py"


def test_duca_online_plugin_smoke_runs_and_reports_contract(tmp_path: Path) -> None:
    output_json = tmp_path / "duca_online_plugin_smoke.json"

    completed = subprocess.run(
        [sys.executable, str(SMOKE), "--output-json", str(output_json)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert stdout_payload == payload
    assert payload["selected_count"] == 384
    assert payload["budget"] == 384
    assert payload["detector_input_length"] == 384
    assert payload["budget_violation_rate"] == 0.0
    assert payload["uses_ledger_for_decision"] is False
    assert payload["teacher_utility_used_train_only"] is True
