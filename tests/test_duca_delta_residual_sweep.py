from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from opentad.models.duca.structured_selection import global_structured_topk
from tools.bata.audit_duca_delta_residual_sweep import (
    parse_gamma_grid,
    run_residual_sweep,
    standardized_residual_scores,
)


def _record() -> dict:
    return {
        "schema_version": "duca_selection_quality_record_v2",
        "sample_id": "video|0",
        "valid_len": 8,
        "gt_segments": [[2.0, 6.0]],
        "transition_policy_scores": [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5],
        "abs_delta_p_action": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "source": {
            "config": "cfg.py",
            "checkpoint": "epoch_4.pth",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_state_key": "state_dict_ema",
            "git_commit": "b" * 40,
            "selector_only_inference": True,
            "detector_backbone_executed": False,
            "uses_gt_for_selection": False,
        },
    }


def test_gamma_zero_preserves_the_learned_hard_path() -> None:
    learned = torch.tensor(_record()["transition_policy_scores"])
    delta = torch.tensor(_record()["abs_delta_p_action"])
    residual, _ = standardized_residual_scores(learned, delta, 0.0)
    original = global_structured_topk(learned[None, :], k=4, max_unselected_hole=2)
    normalized = global_structured_topk(residual[None, :], k=4, max_unselected_hole=2)
    assert torch.equal(original.selected_positions, normalized.selected_positions)


def test_residual_sweep_is_read_only_and_hash_bound(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    report = run_residual_sweep(
        records_jsonl=records,
        output_json=output,
        gamma_grid="0,0.5,1",
        budget=4,
        max_unselected_hole=2,
    )
    assert report["status"] == "complete"
    assert report["aggregate"]["sample_count"] == 1
    assert report["read_only_contract"]["gt_used_for_selection"] is False
    assert report["read_only_contract"]["model_selection_allowed"] is False
    assert report["provenance"]["records_sha256"]
    assert output.exists()


def test_gamma_grid_fails_closed() -> None:
    assert parse_gamma_grid("0,0.25,1") == [0.0, 0.25, 1.0]
    with pytest.raises(ValueError, match="start at zero"):
        parse_gamma_grid("0.1,1")
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_gamma_grid("0,1,0.5")
