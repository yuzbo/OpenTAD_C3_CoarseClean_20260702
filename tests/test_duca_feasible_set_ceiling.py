from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import diagnose_duca_feasible_set_ceiling as ceiling


def test_counterfactual_ceiling_enforces_constraints_and_uses_detector_loss(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    row = {"sample_id": "v|0", "video_id": "v", "valid_len": 8, "budget": 4, "max_hole": 2, "candidates": [
        {"candidate_id": "uniform", "selected_positions": [0, 2, 5, 7], "detector_loss": 2.0, "is_reference": True},
        {"candidate_id": "best", "selected_positions": [0, 2, 4, 7], "detector_loss": 1.0},
        {"candidate_id": "illegal", "selected_positions": [0, 1, 2, 3], "detector_loss": 0.0},
    ]}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = ceiling.run(source, tmp_path / "out")
    assert result["mean_counterfactual_gain"] == pytest.approx(1.0)
    assert result["contract"]["paper_deployable"] is False
    assert (tmp_path / "out" / "feasible_set_counterfactual_per_sample.csv").is_file()


def test_counterfactual_ceiling_rejects_proxy_without_detector_loss() -> None:
    with pytest.raises(ValueError, match="detector_loss"):
        ceiling.evaluate_record({"sample_id": "x", "valid_len": 4, "budget": 2, "max_hole": 2, "candidates": [{"selected_positions": [0, 3], "boundary_score": 1.0}]})
