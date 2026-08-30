from __future__ import annotations

import json
import os

import pytest

if os.name == "nt":
    pytest.skip(
        "local Windows torch/c10.dll import is unstable; Linux remote runs this suite",
        allow_module_level=True,
    )

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.duca import (  # noqa: E402
    SignedTwoSidedMarginalUtilityHead,
    allocate_equal_budget_marginal_reallocation,
    allocate_video_budgets_exact,
    build_frozen_scout_marginal_features,
    detached_three_budget_prefix_utilities,
    interpolate_acquisition_time_to_detector_grid,
    marginal_budget_accounting,
    nested_h65_budget_prefixes,
    validate_real_heavy_observation_tensor,
)
from tools.bata.run_duca_marginal_frozen_h65_probe import (  # noqa: E402
    _allocate_rows_by_video,
    _build_parser,
    _derive_cap_release_neighborhood,
    _json_block_list_for_evaluator,
    _stage_paths,
    _write_cap_release_neighborhood_result,
    _write_cap_release_result,
)


def test_nested_h65_prefixes_preserve_the_sealed_baseline_and_are_nested() -> None:
    valid_lengths = torch.tensor([67, 300, 401, 600])
    baseline = torch.full((4, 384), -1, dtype=torch.long)
    valid = torch.zeros(4, 600, dtype=torch.bool)
    for row, valid_length in enumerate(valid_lengths.tolist()):
        baseline_count = min(valid_length, 384)
        baseline[row, :baseline_count] = torch.arange(baseline_count)
        valid[row, :valid_length] = True
    priority = -torch.arange(600, dtype=torch.float32)[None, :].expand(4, -1)

    result = nested_h65_budget_prefixes(
        baseline,
        priority,
        valid,
        budgets=(256, 384, 512),
        baseline_budget=384,
    )

    assert torch.equal(result.positions_by_budget[384], baseline)
    assert torch.equal(result.actual_count_by_budget[256], torch.tensor([67, 256, 256, 256]))
    assert torch.equal(result.actual_count_by_budget[384], torch.tensor([67, 300, 384, 384]))
    assert torch.equal(result.actual_count_by_budget[512], torch.tensor([67, 300, 401, 512]))
    assert torch.equal(result.effective_budget_by_requested[256], torch.tensor([384, 256, 256, 256]))
    assert torch.equal(result.effective_budget_by_requested[512], torch.tensor([384, 384, 512, 512]))
    assert torch.equal(result.execution_slots_by_budget[256], torch.tensor([384, 256, 256, 256]))
    assert torch.equal(result.execution_slots_by_budget[384], torch.tensor([384, 384, 384, 384]))
    assert torch.equal(result.execution_slots_by_budget[512], torch.tensor([384, 384, 416, 512]))
    assert torch.equal(result.positions_by_budget[256][0, :67], baseline[0, :67])
    assert torch.equal(result.positions_by_budget[512][1, :300], baseline[1, :300])
    assert torch.equal(result.execution_positions_by_budget[512][2][:401], torch.arange(401))
    assert torch.equal(result.execution_positions_by_budget[512][2][401:], torch.full((15,), -1))


def test_three_budget_prefix_utility_targets_are_signed_and_detached() -> None:
    valid_lengths = torch.tensor([67, 300, 401])
    baseline = torch.full((3, 384), -1, dtype=torch.long)
    valid = torch.zeros(3, 401, dtype=torch.bool)
    for row, valid_length in enumerate(valid_lengths.tolist()):
        baseline[row, : min(valid_length, 384)] = torch.arange(min(valid_length, 384))
        valid[row, :valid_length] = True
    priority = -torch.arange(401, dtype=torch.float32)[None, :].expand(3, -1)
    nested = nested_h65_budget_prefixes(baseline, priority, valid)
    calls = []

    def evaluate(_positions, budget, counts):
        calls.append((budget, counts.clone()))
        value = {256: 5.0, 384: 3.0, 512: 2.0}[budget]
        return torch.full((counts.numel(),), value, requires_grad=True)

    output = detached_three_budget_prefix_utilities(
        nested.positions_by_budget,
        evaluate,
        actual_count_by_budget=nested.actual_count_by_budget,
    )

    assert torch.equal(output["downgrade_penalty"], torch.tensor([0.0, 2.0, 2.0]))
    assert torch.equal(output["upgrade_gain"], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(output["downgrade_target_valid"], torch.tensor([False, True, True]))
    assert torch.equal(output["upgrade_target_valid"], torch.tensor([False, False, True]))
    assert [(budget, counts.numel()) for budget, counts in calls] == [(384, 3), (256, 2), (512, 1)]
    assert output["downgrade_penalty"].requires_grad is False
    assert output["upgrade_gain"].requires_grad is False
    assert output["direct_detector_gradient"] is False


def test_variable_acquisition_time_mapping_is_monotone_and_k384_exact() -> None:
    k384 = torch.arange(384, dtype=torch.long)[None]
    exact = interpolate_acquisition_time_to_detector_grid(
        k384,
        torch.tensor([384]),
        detector_length=384,
    )
    assert torch.equal(exact, k384.float())

    k256 = torch.arange(0, 512, 2, dtype=torch.long)[None]
    expanded = interpolate_acquisition_time_to_detector_grid(
        k256,
        torch.tensor([256]),
        detector_length=384,
    )
    assert expanded.shape == (1, 384)
    assert torch.all(expanded[:, 1:] > expanded[:, :-1])
    assert float(expanded.min().item()) >= 0.0
    assert float(expanded.max().item()) <= 511.0


def test_signed_marginal_head_does_not_clamp_negative_utilities() -> None:
    head = SignedTwoSidedMarginalUtilityHead(input_dim=3, hidden_dim=4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[-1].bias.copy_(torch.tensor([-2.0, 3.0]))

    output = head(torch.zeros(2, 3))

    assert torch.equal(output["downgrade_penalty"], torch.tensor([-2.0, -2.0]))
    assert torch.equal(output["upgrade_gain"], torch.tensor([3.0, 3.0]))


def test_frozen_scout_features_are_detached_and_ignore_train_only_targets() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_(True)
    state = {
        "valid_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        "coarse_hidden_features": hidden,
        "p_action": torch.tensor([[0.1, 0.4, 0.8, 0.2], [0.6, 0.5, 0.2, 0.0]]),
        "transition_score": torch.tensor([[0.2, 0.9, 0.3, 0.1], [0.1, 0.3, 0.7, 0.0]]),
        "uncertainty": torch.tensor([[0.7, 0.5, 0.2, 0.9], [0.4, 0.5, 0.8, 0.0]]),
        "action_target": torch.ones(2, 4),
    }
    baseline = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])

    first = build_frozen_scout_marginal_features(state, baseline)
    state["action_target"] = torch.zeros(2, 4)
    state["gt_segments"] = torch.full((2, 4), 99.0)
    second = build_frozen_scout_marginal_features(state, baseline)

    assert first.shape == (2, 3 + 3 * 3 + 4)
    assert first.requires_grad is False
    assert torch.equal(first, second)


def test_equal_budget_allocator_transfers_cost_only_for_positive_total_utility() -> None:
    result = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.tensor([0.1, 1.0, 1.0, 1.0]),
        upgrade_gain=torch.tensor([0.1, 2.0, 0.5, 0.4]),
        valid_observations=torch.full((4,), 512),
    )

    assert result.feasible is True
    assert torch.equal(result.budget, torch.tensor([256, 512, 384, 384]))
    assert int(result.actual_cost.sum().item()) == 4 * 384
    assert int(result.changed_mask.sum().item()) == 2
    assert float(result.predicted_total_utility.item()) == pytest.approx(1.9)

    no_transfer = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.ones(4),
        upgrade_gain=torch.full((4,), 0.5),
        valid_observations=torch.full((4,), 512),
    )
    assert no_transfer.feasible is True
    assert torch.equal(no_transfer.budget, torch.full((4,), 384))
    assert int(no_transfer.changed_mask.sum().item()) == 0


def test_allocator_accounts_for_real_short_window_cost_and_reports_infeasibility() -> None:
    feasible = allocate_video_budgets_exact(
        relative_utility=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 2.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.4]]),
        valid_observations=torch.tensor([300, 428, 512, 512]),
        budget_levels=(256, 384, 512),
        baseline_budget=384,
        max_changed_fraction=0.5,
    )
    assert feasible.feasible is True
    assert torch.equal(feasible.effective_budget, torch.tensor([256, 512, 384, 384]))
    assert torch.equal(feasible.actual_cost, torch.tensor([256, 428, 384, 384]))
    assert feasible.target_actual_cost == 1452

    baseline = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.zeros(4),
        upgrade_gain=torch.ones(4),
        valid_observations=torch.full((4,), 300),
    )
    assert baseline.feasible is True
    assert torch.equal(baseline.effective_budget, torch.full((4,), 384))
    assert int(baseline.actual_cost.sum().item()) == 1200
    assert "all-K384 fallback" in baseline.reason


def test_budget_grouped_heavy_execution_supports_partial_final_packet() -> None:
    observations = torch.zeros(1, 1, 3, 416, 2, 2)
    acquisition_mask = torch.arange(416)[None, :] < 401
    assert validate_real_heavy_observation_tensor(
        observations,
        actual_observations=401,
        execution_slots=416,
        acquisition_mask=acquisition_mask,
    ) is observations

    historical = torch.zeros(1, 1, 3, 384, 2, 2)
    historical_mask = torch.arange(384)[None, :] < 67
    assert validate_real_heavy_observation_tensor(
        historical,
        actual_observations=67,
        execution_slots=384,
        acquisition_mask=historical_mask,
        baseline_execution=True,
    ) is historical


def test_nominal_k256_rejects_a_tensor_padded_to_512_observations() -> None:
    with pytest.raises(ValueError, match="does not match execution_slots"):
        validate_real_heavy_observation_tensor(
            torch.zeros(1, 1, 3, 512, 2, 2),
            actual_observations=401,
            execution_slots=416,
            acquisition_mask=torch.arange(416)[None, :] < 401,
        )


def test_short_window_accounting_contract_examples() -> None:
    valid = torch.tensor([67, 300, 401, 600])
    lower = marginal_budget_accounting(valid, 256)
    baseline = marginal_budget_accounting(valid, 384)
    upper = marginal_budget_accounting(valid, 512)
    assert torch.equal(lower["actual_cost"], torch.tensor([67, 256, 256, 256]))
    assert torch.equal(baseline["actual_cost"], torch.tensor([67, 300, 384, 384]))
    assert torch.equal(upper["actual_cost"], torch.tensor([67, 300, 401, 512]))
    assert torch.equal(upper["execution_slots"], torch.tensor([384, 384, 416, 512]))


def test_text_block_list_is_serialized_for_the_map_evaluator(tmp_path) -> None:
    source = tmp_path / "frontend_holdout_block_list.txt"
    source.write_text("video_train_2\nvideo_train_1\n", encoding="utf-8")

    target = _json_block_list_for_evaluator(source)

    assert target == source.with_suffix(".evaluator.json")
    assert json.loads(target.read_text(encoding="utf-8")) == [
        "video_train_2",
        "video_train_1",
    ]
    assert _json_block_list_for_evaluator(source) == target


def test_cap_release_allows_two_positive_transfers_without_changing_cost() -> None:
    rows = [
        {
            "video_id": "video_a",
            "sample_id": f"video_a|{index}",
            "valid_observations": 512,
        }
        for index in range(4)
    ]
    downgrade = [0.1, 0.2, 9.0, 9.0]
    upgrade = [0.0, 0.0, 2.0, 1.5]

    default_budgets, default_summary = _allocate_rows_by_video(
        rows,
        downgrade=downgrade,
        upgrade=upgrade,
    )
    explicit_capped, _ = _allocate_rows_by_video(
        rows,
        downgrade=downgrade,
        upgrade=upgrade,
        max_changed_fraction=0.5,
    )
    released_budgets, released_summary = _allocate_rows_by_video(
        rows,
        downgrade=downgrade,
        upgrade=upgrade,
        max_changed_fraction=1.0,
    )

    assert default_budgets == explicit_capped == [256, 384, 512, 384]
    assert released_budgets == [256, 256, 512, 512]
    assert sum(default_summary["video_a"]["actual_cost"]) == 4 * 384
    assert sum(released_summary["video_a"]["actual_cost"]) == 4 * 384
    assert default_summary["video_a"]["actual_budget_error"] == 0
    assert released_summary["video_a"]["actual_budget_error"] == 0


def test_cap_release_preserves_the_frozen_allocator_tie_break() -> None:
    rows = [
        {
            "video_id": "video_a",
            "sample_id": f"video_a|{index}",
            "valid_observations": 512,
        }
        for index in range(4)
    ]

    budgets, _summary = _allocate_rows_by_video(
        rows,
        downgrade=[0.1] * 4,
        upgrade=[1.0] * 4,
        max_changed_fraction=0.5,
    )

    assert budgets == [384, 384, 256, 512]


def test_cap_release_result_has_an_independent_stage_and_never_overwrites_probe(
    tmp_path,
) -> None:
    original = '{"status":"ORACLE_HEADROOM_GRAY_ZONE_RETURN_TO_PRO"}\n'
    paths = _stage_paths(tmp_path)
    paths["result"].write_text(original, encoding="utf-8")

    _write_cap_release_result(tmp_path, {"status": "CAP_RELEASE_TEST"})

    assert paths["result"].read_text(encoding="utf-8") == original
    assert json.loads(paths["cap_release_result"].read_text(encoding="utf-8")) == {
        "status": "CAP_RELEASE_TEST"
    }
    args = _build_parser().parse_args(
        ["--stage", "oracle-cap-release", "--output-dir", str(tmp_path)]
    )
    assert args.stage == "oracle-cap-release"


def test_cap_release_neighborhood_derives_all_exact_cost_states_without_pairing() -> None:
    rows = []
    capped = []
    released = []
    for video_number in range(4):
        video_id = f"video_{video_number}"
        for window, released_budget in enumerate((256, 512)):
            rows.append(
                {
                    "video_id": video_id,
                    "sample_id": f"{video_id}|{window}",
                    "budget_accounting": {
                        "256": {"actual_cost": 93},
                        "384": {"actual_cost": 100},
                        "512": {"actual_cost": 107},
                    },
                }
            )
            capped.append(384)
            released.append(released_budget)
    special_video = "video_validation_0000419"
    for window, released_budget in enumerate((256, 256, 512, 512)):
        rows.append(
            {
                "video_id": special_video,
                "sample_id": f"{special_video}|{window}",
                "budget_accounting": {
                    "256": {"actual_cost": 93},
                    "384": {"actual_cost": 100},
                    "512": {"actual_cost": 107},
                },
            }
        )
        capped.append(384)
        released.append(released_budget)

    neighborhood = _derive_cap_release_neighborhood(
        rows,
        capped_budgets=capped,
        released_budgets=released,
    )

    assert neighborhood["difference_video_count"] == 5
    assert neighborhood["difference_window_count"] == 12
    assert neighborhood["video_state_counts"] == {
        "video_0": 2,
        "video_1": 2,
        "video_2": 2,
        "video_3": 2,
        special_video: 6,
    }
    assert neighborhood["joint_state_count"] == 96
    assert neighborhood["minimal_transfer_count"] == 8
    assert neighborhood["net_transfer_group_count"] == 6
    assert neighborhood["global_target_observation_cost"] == 1200
    assert all(
        state["actual_observation_cost"] == 1200
        for state in neighborhood["joint_states"]
    )

    special_transfers = [
        set(transfer["released_sample_ids"])
        for transfer in neighborhood["minimal_transfers"]
        if transfer["video_id"] == special_video
    ]
    expected_special_transfers = [
        {f"{special_video}|{down}", f"{special_video}|{up}"}
        for down in (0, 1)
        for up in (2, 3)
    ]
    assert len(special_transfers) == 4
    assert {frozenset(value) for value in special_transfers} == {
        frozenset(value) for value in expected_special_transfers
    }
    assert len(neighborhood["full_release_decompositions"]) == 2
    assert all(
        len(decomposition) == 6
        for decomposition in neighborhood["full_release_decompositions"]
    )


def test_cap_release_neighborhood_result_is_independent_and_parser_exposes_stage(
    tmp_path,
) -> None:
    paths = _stage_paths(tmp_path)
    paths["result"].write_text('{"status":"ORIGINAL"}\n', encoding="utf-8")
    paths["cap_release_result"].write_text(
        '{"status":"CAP_RELEASE"}\n', encoding="utf-8"
    )
    original_probe = paths["result"].read_bytes()
    original_cap_release = paths["cap_release_result"].read_bytes()

    _write_cap_release_neighborhood_result(
        tmp_path, {"status": "NEIGHBORHOOD_TEST"}
    )

    assert paths["result"].read_bytes() == original_probe
    assert paths["cap_release_result"].read_bytes() == original_cap_release
    assert json.loads(
        paths["cap_release_neighborhood_result"].read_text(encoding="utf-8")
    ) == {"status": "NEIGHBORHOOD_TEST"}
    args = _build_parser().parse_args(
        [
            "--stage",
            "oracle-cap-release-neighborhood",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.stage == "oracle-cap-release-neighborhood"
