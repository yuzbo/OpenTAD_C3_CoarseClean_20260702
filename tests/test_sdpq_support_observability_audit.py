import importlib.util
import inspect
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


def test_padded_zero_query_geometry_matches_production_clamp_contract():
    padded_geometry = geometry()
    padded_geometry[0] = {
        key: np.concatenate(
            (
                value,
                (
                    np.zeros((1, 2), dtype=value.dtype)
                    if key == "intervals_sec"
                    else np.zeros(1, dtype=value.dtype)
                ),
            ),
            axis=0,
        )
        for key, value in padded_geometry[0].items()
    }
    for key in (
        "domain_valid_mask",
        "valid_mask",
        "evidence_mask",
        "assignment_mask",
    ):
        padded_geometry[0][key][-1] = False
    result = support_audit.recompute_sdpq_targets(
        padded_geometry,
        np.asarray([[1.9, 2.1]], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        num_classes=2,
        regression_ranges_sec=((0.0, 100.0),),
        center_sample_radius=0.25,
        width_reference_multiplier=2.0,
        max_abs_delta_center=8.0,
        min_log_width=-6.0,
        max_log_width=6.0,
    )
    assert result["geometry"]["widths"][-1] == 0.0
    assert result["geometry"]["width_reference"][-1] == pytest.approx(2.0e-8)
    assert result["assigned_gt"][-1] == -1


def test_domain_valid_zero_width_query_fails_closed():
    invalid_geometry = geometry()
    invalid_geometry[0]["widths_sec"][-1] = 0.0
    invalid_geometry[0]["intervals_sec"][-1] = 0.0
    with pytest.raises(
        support_audit.SupportAuditError,
        match="domain-valid physical query widths must be positive",
    ):
        support_audit.recompute_sdpq_targets(
            invalid_geometry,
            np.asarray([[1.9, 2.1]], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            num_classes=2,
            regression_ranges_sec=((0.0, 100.0),),
            center_sample_radius=0.25,
            width_reference_multiplier=2.0,
            max_abs_delta_center=8.0,
            min_log_width=-6.0,
            max_log_width=6.0,
        )


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


def _fake_sealed_batch():
    return {
        "inputs": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "masks": np.ones((2, 3), dtype=np.bool_),
        "metas": [
            {
                "video_name": "video_a",
                "window_start": 0,
                "duration": 1.0,
                "phystime_native_token_timestamps_sec": [0.25, 0.75],
            },
            {
                "video_name": "video_b",
                "window_start": 2,
                "duration": 2.0,
                "phystime_native_token_timestamps_sec": [0.5, 1.5],
            },
        ],
        "gt_segments": [
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            np.asarray([[0.5, 1.5]], dtype=np.float32),
        ],
        "gt_labels": [np.asarray([0]), np.asarray([1])],
    }


def test_cloned_sealed_batch_is_immutable_after_source_mutation():
    batch = _fake_sealed_batch()
    sealed = support_audit._clone_sealed_batch(batch)
    before = support_audit.canonical_sha256(
        support_audit._canonical_meta_value(sealed)
    )
    batch["inputs"][0, 0, 0] = -100.0
    batch["metas"][0]["video_name"] = "mutated"
    batch["metas"][0]["phystime_native_token_timestamps_sec"][0] = -1.0
    batch["gt_segments"][0][0, 0] = -1.0
    after = support_audit.canonical_sha256(
        support_audit._canonical_meta_value(sealed)
    )
    assert after == before


def test_sealed_sample_validator_reports_exact_changed_fields(monkeypatch):
    batch = _fake_sealed_batch()
    expected = {
        "sequence_index": 0,
        "batch_index": 0,
        "sample_index": 0,
        "inputs_sha256": "before",
        "fingerprint_sha256": "before-fingerprint",
    }
    observed = dict(expected)
    observed["inputs_sha256"] = "after"
    observed["fingerprint_sha256"] = "after-fingerprint"
    monkeypatch.setattr(
        support_audit,
        "_sample_fingerprint",
        lambda *args, **kwargs: observed,
    )
    with pytest.raises(
        support_audit.SupportAuditError,
        match=(
            r"sequence_index=0 batch_index=0 sample_index=0 "
            r"differing_fields=\['inputs_sha256'\]"
        ),
    ):
        support_audit._validate_sealed_sample(
            batch,
            batch_index=0,
            sample_index=0,
            sequence_index=0,
            expected=expected,
            phase="test",
        )


def test_formal_replay_does_not_reexecute_stochastic_training_loader():
    source = inspect.getsource(support_audit._run_formal_audit)
    assert "_build_train_loader" not in source
    assert "sealed_batches" in source
    assert "pre-model replay validation" in source
    assert "post-model replay validation" in source


def test_target_error_contract_allows_only_bounded_offset_roundoff():
    errors = {
        "cls_target": 0.0,
        "offset_target": 3.0517578125e-5,
        "segment_target": 0.0,
        "endpoint_target": 0.0,
    }
    assert support_audit._validate_target_error_contract(errors) is True


def test_target_error_contract_rejects_offset_beyond_bound():
    errors = {
        "cls_target": 0.0,
        "offset_target": 5.0001e-5,
        "segment_target": 0.0,
        "endpoint_target": 0.0,
    }
    with pytest.raises(
        support_audit.SupportAuditError,
        match="offset_target differs from production beyond its numerical tolerance",
    ):
        support_audit._validate_target_error_contract(errors)


@pytest.mark.parametrize(
    "field",
    ("cls_target", "segment_target", "endpoint_target"),
)
def test_target_error_contract_requires_exact_semantic_targets(field):
    errors = {
        "cls_target": 0.0,
        "offset_target": 0.0,
        "segment_target": 0.0,
        "endpoint_target": 0.0,
    }
    errors[field] = 1.0e-12
    with pytest.raises(
        support_audit.SupportAuditError,
        match=rf"{field} must match production exactly",
    ):
        support_audit._validate_target_error_contract(errors)


def test_target_error_contract_rejects_non_finite_error():
    errors = {
        "cls_target": 0.0,
        "offset_target": np.nan,
        "segment_target": 0.0,
        "endpoint_target": 0.0,
    }
    with pytest.raises(
        support_audit.SupportAuditError,
        match="offset_target target error must be finite",
    ):
        support_audit._validate_target_error_contract(errors)


def test_target_error_rejects_non_finite_production_tensor():
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    independent = {
        name: np.zeros((1,), dtype=np.float64)
        for name in support_audit.TARGET_ERROR_ATOL_BY_FIELD
    }
    production = [
        FakeTensor([0.0]),
        FakeTensor([np.nan]),
        FakeTensor([0.0]),
        FakeTensor([0.0]),
    ]
    with pytest.raises(
        support_audit.SupportAuditError,
        match="offset_target production target contains non-finite values",
    ):
        support_audit._target_error(independent, production)
