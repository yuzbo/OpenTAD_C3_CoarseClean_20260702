import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "audit_sdpq_support_observability.py"
SPEC = importlib.util.spec_from_file_location("support_audit", MODULE_PATH)
support_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support_audit)


def geometry(
    *,
    centers=(0.5, 1.5, 2.5, 3.5),
    coverage=(1.0, 0.0, 1.0, 0.0),
    assignment=(True, True, True, True),
):
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.ones_like(centers)
    intervals = np.stack((centers - 0.5, centers + 0.5), axis=-1)
    return [
        {
            "centers_sec": centers,
            "widths_sec": widths,
            "intervals_sec": intervals,
            "domain_valid_mask": np.ones_like(centers, dtype=np.bool_),
            "valid_mask": np.ones_like(centers, dtype=np.bool_),
            "evidence_mask": np.asarray(coverage) > 0.0,
            "assignment_mask": np.asarray(assignment, dtype=np.bool_),
            "coverage_sec": np.asarray(coverage, dtype=np.float64),
        }
    ]


def run(segments, labels, **geometry_kwargs):
    return support_audit.recompute_sdpq_targets(
        geometry(**geometry_kwargs),
        np.asarray(segments, dtype=np.float64).reshape(-1, 2),
        np.asarray(labels, dtype=np.int64),
        num_classes=2,
        regression_ranges_sec=((0.0, 100.0),),
        center_sample_radius=0.25,
        width_reference_multiplier=2.0,
        max_abs_delta_center=8.0,
        min_log_width=-6.0,
        max_log_width=6.0,
    )


def test_reservation_keeps_gt_representable_without_local_center():
    result = run([[1.9, 2.1]], [1])
    gt = result["per_gt"][0]
    assert gt["local_assignment_eligible_count"] == 0
    assert gt["assignment_eligible_count"] == 4
    assert gt["reserved_query"] is True
    assert gt["assigned_query_count"] == 1
    assert result["cls_target"].sum() == pytest.approx(1.0)


def test_assignment_is_decoupled_from_evidence_and_reports_observability():
    result = run(
        [[1.9, 2.1]],
        [0],
        coverage=(1.0, 0.0, 0.0, 1.0),
    )
    summary = support_audit.summarize_recomputed_sample(result)
    assert summary["zero_evidence_assignment_eligible_query_count"] == 2
    assert summary["positive_query_count"] == 1
    assert summary["positive_uncovered_query_count"] == 1
    assert result["per_gt"][0]["assigned_uncovered_query_count"] == 1


def test_no_assignment_candidate_remains_unassigned():
    result = run(
        [[1.9, 2.1]],
        [0],
        assignment=(False, False, False, False),
    )
    gt = result["per_gt"][0]
    assert gt["assignment_eligible_count"] == 0
    assert gt["assigned_query_count"] == 0
    assert result["cls_target"].sum() == pytest.approx(0.0)


def test_range_fallback_restores_unusual_duration_representability():
    result = support_audit.recompute_sdpq_targets(
        geometry(),
        np.asarray([[0.0, 4.0]], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        num_classes=2,
        regression_ranges_sec=((10.0, 20.0),),
        center_sample_radius=2.0,
        width_reference_multiplier=2.0,
        max_abs_delta_center=8.0,
        min_log_width=-6.0,
        max_log_width=6.0,
    )
    assert result["per_gt"][0]["regression_range_fallback"] is True
    assert result["per_gt"][0]["assignment_eligible_count"] == 4
    assert result["per_gt"][0]["assigned_query_count"] > 0


def test_reservation_collision_is_visible():
    result = run(
        [[0.4, 0.6], [0.45, 0.55]],
        [0, 1],
        assignment=(True, False, False, False),
    )
    assert result["reserved_match_count"] == 1
    assert result["reservation_collision_count"] == 1
    assert sum(gt["assigned_query_count"] == 0 for gt in result["per_gt"]) == 1


def test_endpoint_final_edge_is_inclusive_only_for_last_cell():
    result = run([[0.0, 4.0]], [0])
    endpoint = result["endpoint_target"]
    assert endpoint[0, 0] == pytest.approx(1.0)
    assert endpoint[-1, 1] == pytest.approx(1.0)
    assert endpoint[:-1, 1].sum() == pytest.approx(0.0)


def test_aggregate_rows_separates_evidence_eligibility_and_assignment():
    first = support_audit.summarize_recomputed_sample(
        run([[1.9, 2.1]], [0])
    )
    second = support_audit.summarize_recomputed_sample(
        run(
            [[1.9, 2.1]],
            [0],
            assignment=(False, False, False, False),
        )
    )
    summary = support_audit.aggregate_rows([first, second])
    assert summary["window_count"] == 2
    assert summary["gt_count"] == 2
    assert summary["gt_without_assignment_eligible_query"] == 1
    assert summary["gt_without_assigned_query"] == 1


@pytest.mark.parametrize(
    "segments,labels",
    [
        ([[np.nan, 1.0]], [0]),
        ([[1.0, 1.0]], [0]),
        ([[0.0, 1.0]], [2]),
    ],
)
def test_malformed_gt_fails_closed(segments, labels):
    with pytest.raises(support_audit.SupportAuditError):
        run(segments, labels)


def test_module_does_not_call_loss_backward_or_optimizer():
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        ".backward(",
        "optimizer.step(",
        "forward_train(",
        "._losses(",
    )
    assert not any(token in source for token in forbidden)


def test_checkpoint_selector_accepts_explicit_epoch19_online_state():
    state = {"weight": object()}
    observed_epoch, state_key, selected = support_audit._select_checkpoint_state(
        {"epoch": 19, "state_dict": state},
        "online",
        19,
    )
    assert observed_epoch == 19
    assert state_key == "state_dict"
    assert selected is state


def test_checkpoint_selector_rejects_epoch_mismatch():
    with pytest.raises(
        support_audit.SupportAuditError,
        match="checkpoint epoch mismatch: expected 59, observed 19",
    ):
        support_audit._select_checkpoint_state(
            {"epoch": 19, "state_dict": {"weight": object()}},
            "online",
            59,
        )


def test_checkpoint_selector_does_not_substitute_online_for_missing_ema():
    with pytest.raises(
        support_audit.SupportAuditError,
        match="checkpoint is missing state_dict_ema",
    ):
        support_audit._select_checkpoint_state(
            {"epoch": 19, "state_dict": {"weight": object()}},
            "ema",
            19,
        )
