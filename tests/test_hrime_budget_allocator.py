from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "opentad" / "models" / "duca" / "hrime.py"
SPEC = importlib.util.spec_from_file_location("hrime_core_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
hrime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hrime
SPEC.loader.exec_module(hrime)


def _sha(label: str) -> str:
    return hrime.canonical_sha256({"label": label})


def _window(key: str, options):
    return hrime.MCKPWindow(
        key,
        tuple(options),
        option_source_sha256=_sha(f"options-{key}"),
    )


def test_short_window_deduplicates_nominal_aliases_into_true_effective_k():
    feasible = hrime.canonicalize_effective_k_options(
        231,
        (192, 256, 384, 512),
    )
    assert feasible.effective_ks == (192, 224)
    assert feasible.nominal_to_effective == {
        192: 192,
        256: 224,
        384: 224,
        512: 224,
    }
    choice = feasible.choice_for_effective_k(224)
    assert choice.nominal_budgets == (256, 384, 512)
    assert choice.canonical_nominal_budget == 256


def test_window_shorter_than_execution_quantum_fails_closed():
    with pytest.raises(ValueError, match="shorter than one"):
        hrime.canonicalize_effective_k_options(
            15,
            (192, 256, 384, 512),
        )


def test_fractional_or_non_quantized_budget_inputs_fail_closed():
    with pytest.raises(ValueError, match="exact integer"):
        hrime.project_budget_to_reachable(
            430.5,
            ((192, 256), (192, 224)),
        )
    with pytest.raises(ValueError, match="quantum aligned"):
        hrime.solve_exact_mckp(
            (
                _window(
                    "w0",
                    (hrime.MCKPOption(15, utility=1, risk=0),),
                ),
            ),
            target_total_cost=15,
            beta=0,
            allocation_context_sha256=_sha("allocation"),
        )


def test_reachable_projection_reports_raw_projection_and_zero_solver_slack():
    projection = hrime.project_budget_to_reachable(
        430,
        ((192, 256), (192, 224)),
    )
    assert projection.reachable_budget == 416
    assert projection.projection_unused_budget == 14
    assert projection.solver_unused_budget == 0
    assert projection.minimum_reachable_budget == 384
    assert projection.maximum_reachable_budget == 480
    assert projection.reachable_total_count == 4


def test_budget_below_minimum_feasible_total_fails_closed():
    with pytest.raises(ValueError, match="below the minimum"):
        hrime.project_budget_to_reachable(
            383,
            ((192, 256), (192, 224)),
        )


def test_exact_mckp_maximizes_objective_at_exact_reachable_total():
    windows = (
        _window(
            "w0",
            (
                hrime.MCKPOption(16, utility="1", risk="0.1"),
                hrime.MCKPOption(32, utility="5", risk="0.2"),
            ),
        ),
        _window(
            "w1",
            (
                hrime.MCKPOption(16, utility="4", risk="0.2"),
                hrime.MCKPOption(32, utility="3", risk="0.1"),
            ),
        ),
    )
    result = hrime.solve_exact_mckp(
        windows,
        target_total_cost=48,
        beta="1",
        allocation_context_sha256=_sha("allocation"),
    )
    assert result.assignment == (32, 16)
    assert result.realized_total_cost == 48
    assert result.solver_unused_budget == 0
    assert result.to_receipt()["status"] == "solved_exact"


def test_exact_mckp_tie_breaks_by_lower_risk_then_lexicographic_assignment():
    lower_risk = (
        _window(
            "w0",
            (
                hrime.MCKPOption(16, utility="1", risk="0"),
                hrime.MCKPOption(32, utility="2", risk="1"),
            ),
        ),
        _window(
            "w1",
            (
                hrime.MCKPOption(16, utility="2", risk="1"),
                hrime.MCKPOption(32, utility="1", risk="0"),
            ),
        ),
    )
    result = hrime.solve_exact_mckp(
        lower_risk,
        target_total_cost=48,
        beta="1",
        allocation_context_sha256=_sha("allocation"),
    )
    assert result.assignment == (16, 32)
    assert result.total_risk_int == 0

    exact_tie = tuple(
        _window(
            key,
            (
                hrime.MCKPOption(16, utility="1", risk="0"),
                hrime.MCKPOption(32, utility="1", risk="0"),
            ),
        )
        for key in ("w0", "w1")
    )
    tied = hrime.solve_exact_mckp(
        exact_tie,
        target_total_cost=48,
        beta="0",
        allocation_context_sha256=_sha("allocation"),
    )
    assert tied.assignment == (16, 32)


def test_exact_mckp_is_bit_exact_and_rejects_unreachable_target():
    windows = (
        _window(
            "w0",
            (
                hrime.MCKPOption(16, utility="0.1234564", risk="0.3333333"),
                hrime.MCKPOption(32, utility="0.1234566", risk="0.1111111"),
            ),
        ),
        _window(
            "w1",
            (
                hrime.MCKPOption(16, utility="1.2", risk="0.2"),
                hrime.MCKPOption(32, utility="1.3", risk="0.3"),
            ),
        ),
    )
    first = hrime.solve_exact_mckp(
        windows,
        target_total_cost=48,
        beta="0.75",
        allocation_context_sha256=_sha("allocation"),
    )
    second = hrime.solve_exact_mckp(
        windows,
        target_total_cost=48,
        beta="0.7500",
        allocation_context_sha256=_sha("allocation"),
    )
    assert first == second
    assert first.solver_input_sha256 == second.solver_input_sha256
    assert first.assignment_sha256 == second.assignment_sha256
    assert float(first.maximum_component_quantization_error) <= 0.5e-6
    with pytest.raises(ValueError, match="not reachable"):
        hrime.solve_exact_mckp(
            windows,
            target_total_cost=47,
            beta="0.75",
            allocation_context_sha256=_sha("allocation"),
        )


def test_exact_mckp_agrees_with_brute_force_on_small_integer_panel():
    windows = (
        _window(
            "w0",
            (
                hrime.MCKPOption(16, utility=1, risk=0),
                hrime.MCKPOption(32, utility=7, risk=1),
                hrime.MCKPOption(48, utility=8, risk=1),
            ),
        ),
        _window(
            "w1",
            (
                hrime.MCKPOption(16, utility=5, risk=1),
                hrime.MCKPOption(32, utility=6, risk=0),
                hrime.MCKPOption(48, utility=9, risk=2),
            ),
        ),
        _window(
            "w2",
            (
                hrime.MCKPOption(16, utility=2, risk=0),
                hrime.MCKPOption(32, utility=4, risk=1),
                hrime.MCKPOption(48, utility=10, risk=3),
            ),
        ),
    )
    target = 96
    beta = 1
    feasible = []
    for indices in itertools.product(range(3), repeat=3):
        options = tuple(
            windows[window_index].options[option_index]
            for window_index, option_index in enumerate(indices)
        )
        assignment = tuple(option.effective_k for option in options)
        if sum(assignment) != target:
            continue
        objective = sum(int(option.utility) - beta * int(option.risk) for option in options)
        risk = sum(int(option.risk) for option in options)
        feasible.append((objective, risk, assignment))
    expected = sorted(feasible, key=lambda row: (-row[0], row[1], row[2]))[0]
    result = hrime.solve_exact_mckp(
        windows,
        target_total_cost=target,
        beta=beta,
        allocation_context_sha256=_sha("allocation"),
    )
    assert result.assignment == expected[2]
    assert result.total_objective_int == expected[0] * hrime.HRIME_SCORE_SCALE
    assert result.total_risk_int == expected[1] * hrime.HRIME_SCORE_SCALE


def _group_and_feasible_sets():
    grouped = hrime.group_video_windows(
        (
            hrime.VideoWindowRef(
                video_id="video-a",
                window_start_frame=768,
                valid_length=231,
                source_index=1,
                cheap_feature_index=1,
            ),
            hrime.VideoWindowRef(
                video_id="video-a",
                window_start_frame=0,
                valid_length=768,
                source_index=0,
                cheap_feature_index=0,
            ),
        )
    )
    assert len(grouped) == 1
    group = grouped[0]
    feasible = tuple(
        hrime.canonicalize_effective_k_options(
            window.valid_length,
            (192, 256, 384, 512),
        )
        for window in group.windows
    )
    return group, feasible


def test_video_group_scan_budget_replay_and_dispatch_contracts_close():
    group, feasible = _group_and_feasible_sets()
    assert tuple(window.window_start_frame for window in group.windows) == (0, 768)
    scan = hrime.build_shared_scan_receipt(
        group,
        scan_version="cheap_scan_v1",
        scan_input_sha256=_sha("scan-input"),
        video_summary_sha256=_sha("video-summary"),
        window_summary_sha256=(_sha("window-0"), _sha("window-1")),
    )
    plan = hrime.build_video_budget_plan(
        group,
        scan,
        planner_version="video_budget_planner_v1",
        raw_budget=430,
        feasible_sets=feasible,
    )
    assert plan.projection.reachable_budget == 416
    windows = (
        _window(
            group.windows[0].window_key,
            tuple(
                hrime.MCKPOption(
                    choice.effective_k,
                    utility=10 if choice.effective_k == 192 else 0,
                    risk=0,
                    nominal_budgets=choice.nominal_budgets,
                )
                for choice in feasible[0].choices
            ),
        ),
        _window(
            group.windows[1].window_key,
            tuple(
                hrime.MCKPOption(
                    choice.effective_k,
                    utility=10 if choice.effective_k == 224 else 0,
                    risk=0,
                    nominal_budgets=choice.nominal_budgets,
                )
                for choice in feasible[1].choices
            ),
        ),
    )
    result = hrime.solve_exact_mckp(
        windows,
        target_total_cost=plan.projection.reachable_budget,
        beta=1,
        allocation_context_sha256=plan.decision_input_sha256,
    )
    assert result.assignment == (192, 224)
    replay = hrime.build_hrime_replay_rows(
        group,
        feasible,
        plan,
        result,
        budget_protocol_sha256=_sha("budget-protocol"),
    )
    assert tuple(row["requested_k"] for row in replay) == (192, 256)
    assert tuple(row["effective_k"] for row in replay) == (192, 224)
    assert all(
        row["provenance"]["role"] == hrime.HRIME_REPLAY_ROLE
        and row["provenance"]["uses_gt"] is False
        and row["provenance"]["solver_unused_budget"] == 0
        for row in replay
    )
    dispatch = hrime.build_homogeneous_k_dispatch(group, result)
    assert tuple(bucket.effective_k for bucket in dispatch.buckets) == (192, 224)
    assert dispatch.heavy_frame_total == 416
    assert dispatch.tail_padding_mode == "none_exact_k_bucket"
    assert dispatch.restore_group_order == (0, 1)


def test_replay_rejects_feasible_aliases_that_differ_from_frozen_plan():
    group, feasible = _group_and_feasible_sets()
    scan = hrime.build_shared_scan_receipt(
        group,
        scan_version="cheap_scan_v1",
        scan_input_sha256=_sha("scan-input"),
        video_summary_sha256=_sha("video-summary"),
        window_summary_sha256=(_sha("window-0"), _sha("window-1")),
    )
    plan = hrime.build_video_budget_plan(
        group,
        scan,
        planner_version="video_budget_planner_v1",
        raw_budget=430,
        feasible_sets=feasible,
    )
    windows = tuple(
        _window(
            window.window_key,
            tuple(
                hrime.MCKPOption(
                    choice.effective_k,
                    utility=0,
                    risk=0,
                    nominal_budgets=choice.nominal_budgets,
                )
                for choice in choices.choices
            ),
        )
        for window, choices in zip(group.windows, feasible)
    )
    result = hrime.solve_exact_mckp(
        windows,
        target_total_cost=416,
        beta=0,
        allocation_context_sha256=plan.decision_input_sha256,
    )
    tampered = (
        feasible[0],
        hrime.WindowFeasibleKSet(
            valid_length=231,
            execution_quantum=16,
            choices=(
                hrime.EffectiveKChoice(192, (192,)),
                hrime.EffectiveKChoice(224, (384, 512)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="differ from the frozen budget plan"):
        hrime.build_hrime_replay_rows(
            group,
            tampered,
            plan,
            result,
            budget_protocol_sha256=_sha("budget-protocol"),
        )


def test_dispatch_records_the_inverse_needed_to_restore_original_window_order():
    group, feasible = _group_and_feasible_sets()
    windows = (
        _window(
            group.windows[0].window_key,
            tuple(
                hrime.MCKPOption(choice.effective_k, utility=0, risk=0)
                for choice in feasible[0].choices
            ),
        ),
        _window(
            group.windows[1].window_key,
            tuple(
                hrime.MCKPOption(choice.effective_k, utility=0, risk=0)
                for choice in feasible[1].choices
            ),
        ),
    )
    result = hrime.solve_exact_mckp(
        windows,
        target_total_cost=448,
        beta=0,
        allocation_context_sha256=_sha("allocation"),
    )
    assert result.assignment == (224, 224) or result.assignment == (256, 192)
    if result.assignment == (256, 192):
        dispatch = hrime.build_homogeneous_k_dispatch(group, result)
        assert dispatch.heavy_execution_group_order == (1, 0)
        assert dispatch.restore_group_order == (1, 0)


def test_video_group_rejects_duplicate_physical_window_identity():
    with pytest.raises(ValueError, match="duplicate window starts"):
        hrime.group_video_windows(
            (
                hrime.VideoWindowRef("v", 0, 768, 0, 0),
                hrime.VideoWindowRef("v", 0, 231, 1, 1),
            )
        )
