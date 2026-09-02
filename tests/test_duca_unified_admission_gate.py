"""Fail-closed checks for the currently incomplete Unified cells."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "scripts" / "duca_unified_fullmatrix" / "matrix.json"
SUBMIT = ROOT / "scripts" / "duca_unified_fullmatrix" / "submit_all.sh"


def test_unimplemented_cells_are_marked_and_submission_is_fail_closed():
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    rows = payload["rows"]
    blocked = {row["arm_id"] for row in rows if row["admission_status"] != "READY_FOR_GATE"}
    assert blocked == {"D1", "F11", "G10", "G11", "H0"}
    assert payload["implementation_gate"]["cost_benchmark_status"] == "BLOCKED_UNIMPLEMENTED"
    submit_text = SUBMIT.read_text(encoding="utf-8")
    assert "DUCA Unified formal submission is blocked" in submit_text
    assert "DUCA Unified cost benchmark is not implemented" in submit_text
