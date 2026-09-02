import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "bata" / "dispatch_duca_multibranch.py"


def test_dispatch_plan_is_fail_closed_for_current_admission():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "plan", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 3
    payload = json.loads(proc.stdout)
    assert payload["status"] == "BLOCKED"
    assert any("DUCA_UNIFIED" in reason for reason in payload["blocked_reasons"])
    assert payload["job_ids"] == []
