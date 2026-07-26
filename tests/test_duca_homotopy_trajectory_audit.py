from __future__ import annotations

import json

import torch

from opentad.models.duca.structured_selection import exact_uniform_reference_scores
from tools.bata.audit_duca_homotopy_trajectory import (
    _load_fixed_batch,
    audit_tensor_trajectory,
    run_records_audit,
)


def _audit(scores: torch.Tensor, *, budget: int, max_hole: int, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    return audit_tensor_trajectory(
        scores,
        torch.ones_like(scores, dtype=torch.bool),
        budget=budget,
        max_unselected_hole=max_hole,
        alpha_grid=alphas,
        temperature=0.7,
        sample_ids=["fixed-validation-row"],
    )


def test_identical_reference_scores_keep_the_hard_path_completely_unchanged() -> None:
    valid = torch.ones(1, 8, dtype=torch.bool)
    reference = exact_uniform_reference_scores(torch.zeros(1, 8), valid, k=4)

    report = _audit(reference, budget=4, max_hole=2)
    sample = report["samples"][0]

    assert sample["trajectory_summary"]["unique_hard_path_count"] == 1
    assert sample["trajectory_summary"]["max_single_step_hard_swaps"] == 0
    assert sample["trajectory_summary"]["first_change_alpha"] is None
    assert sample["trajectory_summary"]["last_change_alpha"] is None
    assert sample["trajectory_summary"]["longest_hard_path_unchanged_interval"]["alpha_start"] == 0.0
    assert sample["trajectory_summary"]["longest_hard_path_unchanged_interval"]["alpha_end"] == 1.0
    assert all(step["exact_uniform_overlap_rate"] == 1.0 for step in sample["alpha_trajectory"])


def test_single_threshold_can_trigger_a_large_hard_jump_while_soft_occupancy_moves_smoothly() -> None:
    valid = torch.ones(1, 8, dtype=torch.bool)
    reference = exact_uniform_reference_scores(torch.zeros(1, 8), valid, k=4)

    report = _audit(-reference, budget=4, max_hole=2, alphas=(0.0, 0.49, 0.51, 1.0))
    sample = report["samples"][0]
    pairs = sample["adjacent_alpha"]

    assert sample["trajectory_summary"]["unique_hard_path_count"] == 2
    assert sample["trajectory_summary"]["hard_path_change_count"] == 1
    assert sample["trajectory_summary"]["first_change_alpha"] == 0.51
    assert sample["trajectory_summary"]["max_single_step_hard_swaps"] == 4
    assert pairs[1]["hard_swap_count"] == 4
    assert 0.0 < pairs[1]["soft_occupancy_l1"] < 4.0
    assert pairs[0]["hard_swap_count"] == 0
    assert pairs[0]["soft_changed_without_hard_swap"] is True


def test_short_window_selects_every_valid_position_and_reports_effective_k() -> None:
    scores = torch.tensor([[3.0, -2.0, 0.5]], dtype=torch.float32)

    report = _audit(scores, budget=4, max_hole=2, alphas=(0.0, 0.5, 1.0))
    sample = report["samples"][0]

    assert report["aggregate"]["valid_len_le_budget_ratio"] == 1.0
    assert sample["effective_k"] == 3
    assert all(step["hard_positions"] == [0, 1, 2] for step in sample["alpha_trajectory"])
    assert all(step["geometry"]["max_hole"] == 0 for step in sample["alpha_trajectory"])
    assert all(step["geometry"]["adjacent_selection_rate"] == 1.0 for step in sample["alpha_trajectory"])
    assert all(step["geometry"]["longest_contiguous_selected_run"] == 3 for step in sample["alpha_trajectory"])


def test_g2_contract_preserves_exact_k_and_never_leaves_more_than_two_unselected() -> None:
    scores = torch.tensor([[20.0, 19.0, 18.0, 17.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0]])

    report = _audit(scores, budget=4, max_hole=2, alphas=(0.0, 0.3, 0.7, 1.0))
    sample = report["samples"][0]

    assert sample["max_unselected_hole"] == 2
    for step in sample["alpha_trajectory"]:
        assert len(step["hard_positions"]) == 4
        assert step["geometry"]["max_hole"] <= 2
        assert sum(step["geometry"]["selected_gap_histogram"].values()) == 3


def test_real_loader_uint8_fixed_batch_is_accepted(tmp_path) -> None:
    path = tmp_path / "batch.pth"
    torch.save(
        {
            "batch": {
                "inputs": torch.zeros(1, 3, 4, 2, 2, dtype=torch.uint8),
                "masks": torch.ones(1, 4, dtype=torch.bool),
                "sample_ids": ["video|0"],
            }
        },
        path,
    )

    loaded = _load_fixed_batch(path)

    assert loaded["inputs"].dtype == torch.uint8
    assert loaded["sample_ids"] == ["video|0"]


def test_records_mode_reopens_one_consistent_hash_bound_source(tmp_path) -> None:
    records = tmp_path / "records.jsonl"
    source = {
        "config": "/audit/config.py",
        "checkpoint": "/audit/epoch_4.pth",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_state_key": "state_dict_ema",
        "git_commit": "b" * 40,
        "selector_only_inference": True,
        "detector_backbone_executed": False,
        "uses_gt_for_selection": False,
    }
    rows = []
    for index in range(2):
        rows.append(
            {
                "schema_version": "duca_selection_quality_record_v2",
                "sample_id": f"video|{index}",
                "valid_len": 8 - index,
                "transition_policy_scores": [float(value) for value in range(8 - index)],
                "gt_segments": [[1.0, 3.0]],
                "source": source,
            }
        )
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "audit.json"

    report = run_records_audit(
        records_jsonl=records,
        output_json=output,
        alpha_grid=(0.0, 0.5, 1.0),
        budget=4,
        max_unselected_hole=2,
        temperature=0.7,
        evaluate_gt=True,
    )

    assert report["aggregate"]["sample_count"] == 2
    assert report["provenance"]["record_source"]["checkpoint_sha256"] == "a" * 64
    assert report["read_only_contract"]["selector_constructed"] is False
    assert output.is_file()
