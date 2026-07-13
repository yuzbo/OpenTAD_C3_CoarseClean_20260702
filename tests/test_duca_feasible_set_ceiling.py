from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import diagnose_duca_feasible_set_ceiling as ceiling


def _provenance() -> dict:
    return {
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "geometry_sha256": "e" * 64,
        "gt_sha256": "f" * 64,
        "evaluator_identity": "ActionFormer:cls_loss+reg_loss",
    }


def _candidate(candidate_id: str, positions: list[int], loss: float, **extra: object) -> dict:
    return {"candidate_id": candidate_id, "selected_positions": positions, "detector_loss": loss,
            "detector_loss_components": {"cls_loss": loss * .6, "reg_loss": loss * .4}, **extra}


def test_counterfactual_ceiling_enforces_constraints_and_uses_detector_loss(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    row = {"sample_id": "v|0", "video_id": "v", "valid_len": 8, "budget": 4, "max_hole": 2, "provenance": _provenance(), "candidates": [
        _candidate("uniform", [0, 2, 5, 7], 2.0, is_reference=True),
        _candidate("best", [0, 2, 4, 7], 1.0),
        _candidate("illegal", [0, 1, 2, 3], 0.0),
    ]}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = ceiling.run(source, tmp_path / "out")
    assert result["mean_counterfactual_gain"] == pytest.approx(1.0)
    assert result["contract"]["paper_deployable"] is False
    assert result["diagnostic_role"].endswith("not_upper_bound")
    assert (tmp_path / "out" / "supplied_candidate_counterfactual_per_sample.csv").is_file()


def test_counterfactual_ceiling_rejects_proxy_without_detector_loss() -> None:
    with pytest.raises(ValueError, match="detector_loss"):
        ceiling.evaluate_record({"sample_id": "x", "valid_len": 4, "budget": 2, "max_hole": 2, "provenance": _provenance(), "candidates": [{"candidate_id": "x", "selected_positions": [0, 3], "boundary_score": 1.0}]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_counterfactual_rejects_non_finite_loss(bad: float) -> None:
    row = {"sample_id": "x", "valid_len": 4, "budget": 2, "max_hole": 2, "provenance": _provenance(),
           "candidates": [_candidate("ref", [0, 3], bad, is_reference=True)]}
    with pytest.raises(ValueError, match="finite"):
        ceiling.evaluate_record(row)


def test_counterfactual_rejects_duplicate_selection_and_multiple_reference() -> None:
    base = {"sample_id": "x", "valid_len": 4, "budget": 2, "max_hole": 2, "provenance": _provenance()}
    with pytest.raises(ValueError, match="duplicate candidate selection"):
        ceiling.evaluate_record({**base, "candidates": [_candidate("a", [0, 3], 1, is_reference=True), _candidate("b", [0, 3], 2)]})
    with pytest.raises(ValueError, match="exactly one"):
        ceiling.evaluate_record({**base, "candidates": [_candidate("a", [0, 2], 1, is_reference=True), _candidate("b", [1, 3], 2, is_reference=True)]})
