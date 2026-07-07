from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.bata import train_detector_aware_acquisition_policy as train_detector
from tools.bata import apply_detector_aware_acquisition_policy as detector_apply
from tools.bata import convert_detector_aware_samples_to_value_transport_ledger as detector_convert
from tools.bata import run_detector_aware_ledger_pipeline as detector_pipeline
from tools.bata import validate_detector_aware_policy_ledger as detector_validator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _paction_provenance() -> dict:
    return {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }


def _has_key_recursive(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_has_key_recursive(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key_recursive(item, key) for item in value)
    return False


def _sample(sample_id: str, p_action: list[float], utility: list[float]) -> dict:
    return {
        "sample_id": sample_id,
        "split": "validation",
        "dense_len": len(p_action),
        "valid_len": len(p_action),
        "frame_signals": {"p_action": p_action},
        "paction_positive_provenance": _paction_provenance(),
        "action_target": [1 if value >= 0.5 else 0 for value in utility],
        "gt_boundaries": [1, 4],
        "teacher_utility": {"signed_frame_utility": utility},
        "teacher_utility_provenance": {"split_scope": "train_only"},
    }


def _deploy_policy_metadata(checkpoint: Path, sha: str) -> dict:
    return {
        "source": "learned_detector_aware_policy_checkpoint",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha,
        "policy_checkpoint_sha256": sha,
        "policy_family": "detector_aware_offline_selector",
        "stage_label": "Stage-2 detector-aware offline selector",
        "end_to_end": False,
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "p_action_provenance": _paction_provenance(),
    }


def _deploy_sample(checkpoint: Path, sha: str) -> dict:
    return {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]},
        "paction_positive_provenance": _paction_provenance(),
        "strategy_selected_positions": {"detector_aware_fixed_384": [0, 2]},
        "detector_aware_policy": _deploy_policy_metadata(checkpoint, sha),
    }


def _deploy_ledger_row(checkpoint: Path, sha: str) -> dict:
    return {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": [0, 2],
        "target_len": 2,
        "selected_count": 2,
        "valid_len": 4,
        "dense_len": 4,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "learned_detector_aware_policy_checkpoint",
        "policy_checkpoint_path": str(checkpoint),
        "policy_checkpoint_sha256": sha,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
        "diagnostics": {
            "uniform_visible_fill_count": 0,
            "source_strategy": "detector_aware_fixed_384",
            "policy_source": "learned_detector_aware_policy_checkpoint",
            "policy_checkpoint_path": str(checkpoint),
            "policy_checkpoint_sha256": sha,
            "p_action_provenance": _paction_provenance(),
        },
    }


def test_detector_aware_pipeline_generates_three_ledgers_and_utility_metrics(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "detector_policy.pth"
    checkpoint.write_bytes(b"fake detector aware checkpoint")
    _write_jsonl(
        source,
        [
            _sample("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], [0.0, 1.0, 0.2, 0.9, 0.1, 0.8]),
            _sample("video_test_0002|0", [0.9, 0.1, 0.8, 0.2, 0.7, 0.3], [1.0, 0.0, 0.9, 0.1, 0.8, 0.2]),
        ],
    )

    monkeypatch.setattr(
        detector_pipeline.apply_policy,
        "load_policy_checkpoint",
        lambda *args, **kwargs: (object(), {"dynamic_budget_buckets": [2, 4]}),
    )
    monkeypatch.setattr(
        detector_pipeline.apply_policy,
        "checkpoint_policy_scores",
        lambda _model, p_action, *, valid, target_budget=None, device: detector_pipeline.apply_policy.bootstrap_policy_scores(
            p_action,
            valid=valid,
            dynamic_budget_buckets=[2, 4],
        ),
    )

    summary = detector_pipeline.run_pipeline(
        input_jsonl=source,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        fixed_budgets=(2, 4),
        dynamic_target_len=4,
        dynamic_budget_buckets=[2, 4],
        device="cpu",
        allow_tiny_dynamic_diagnostic=True,
        max_unselected_hole=6,
        max_p95_unselected_hole=6,
        max_uniform_similarity=1.0,
    )

    assert summary["decision"] == "C3_DETECTOR_AWARE_LEDGER_PIPELINE_READY"
    assert summary["stage_label"] == "Stage-2 detector-aware offline selector"
    assert summary["dynamic_budget_diagnostic_allow_constant_tiny"] is True
    assert summary["baseline_comparison"]["matched_budget_baselines"] == ["p_action_only", "GAS-VT"]
    assert sorted(summary["ledgers"]) == [
        "detector_aware_dynamic",
        "detector_aware_fixed_384",
        "detector_aware_fixed_768",
    ]
    for item in summary["ledgers"].values():
        validation = item["validation_summary"]
        assert validation["required_policy_source"] == "learned_detector_aware_policy_checkpoint"
        assert "detector_utility_coverage" in validation
        assert "detector_utility_ndcg" in validation
        assert validation["detector_utility_rows"] == 0
        assert validation["detector_utility_metric_availability"] == "not_available_no_train_only_teacher_utility"
        assert validation["dynamic_gain_calibration"]["score_semantics"] == "calibrated_marginal_gain"
        assert validation["max_unselected_hole"] <= 6
        assert validation["adatad_map"] is None
        assert validation["map_claim_allowed"] is False


def test_detector_aware_pipeline_rejects_collapsed_dynamic_without_diagnostic_optout(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "detector_policy.pth"
    checkpoint.write_bytes(b"fake detector aware checkpoint")
    _write_jsonl(
        source,
        [
            _sample("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], [0.0, 1.0, 0.2, 0.9, 0.1, 0.8]),
            _sample("video_test_0002|0", [0.2, 0.8, 0.3, 0.7, 0.4, 0.6], [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        ],
    )
    monkeypatch.setattr(
        detector_pipeline.apply_policy,
        "load_policy_checkpoint",
        lambda *args, **kwargs: (object(), {"dynamic_budget_buckets": [2, 4]}),
    )
    monkeypatch.setattr(
        detector_pipeline.apply_policy,
        "checkpoint_policy_scores",
        lambda _model, p_action, *, valid, target_budget=None, device: ([float(item) for item in p_action], [1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="dynamic budget ledger is degenerate"):
        detector_pipeline.run_pipeline(
            input_jsonl=source,
            checkpoint_path=checkpoint,
            out_dir=tmp_path / "out_collapsed",
            fixed_budgets=(2, 4),
            dynamic_target_len=4,
            dynamic_budget_buckets=[2, 4],
            device="cpu",
        )


def test_detector_aware_apply_strips_nested_forbidden_payload_from_deploy_output(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "applied.jsonl"
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]},
        "paction_positive_provenance": _paction_provenance(),
        "teacher_utility": {"frame_utility": [1.0, 0.0, 0.8, 0.0]},
        "debug_payload": [
            {"teacher_scores": [0.9, 0.1]},
            {"nested": {"raw_predictions": [{"score": 0.8}]}},
        ],
    }
    _write_jsonl(source, [row])

    detector_apply.run_policy_application(
        input_jsonl=source,
        output_jsonl=output,
        fixed_budgets=(2, 3),
        dynamic_budget_buckets=[2, 3],
        strip_deploy_invisible_payload=True,
        strict_deploy_source=False,
        allow_bootstrap_for_tests=True,
    )

    applied = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert not _has_key_recursive(applied[0], "teacher_utility")
    assert not _has_key_recursive(applied[0], "teacher_scores")
    assert not _has_key_recursive(applied[0], "raw_predictions")


def test_detector_aware_convert_rejects_nested_forbidden_payload_in_deploy_sample(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = detector_validator._sha256_file(checkpoint)
    sample = _deploy_sample(checkpoint, sha)
    sample["audit"] = {"frames": [{"signed_frame_utility": [0.0, 1.0]}]}
    sample["detector_aware_policy"]["debug"] = {"dense_teacher_logits": [[0.1, 0.9]]}
    _write_jsonl(samples, [sample])

    with pytest.raises(ValueError, match="forbidden detector-aware deploy payload key"):
        detector_convert.run_conversion(
            samples,
            ledger,
            strategy="detector_aware_fixed_384",
            target_len=2,
            require_selected_count=2,
            deploy_selection_ledger=True,
        )


def test_detector_aware_validator_rejects_nested_forbidden_payloads_and_accepts_clean_deploy_rows(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = detector_validator._sha256_file(checkpoint)
    sample = _deploy_sample(checkpoint, sha)
    sample["detector_aware_policy"]["audit"] = [{"teacher_proposals": [[0.0, 1.0]]}]
    metric = {
        "sample_id": "video_test_0001|0",
        "split": "training",
        "dense_len": 4,
        "valid_len": 4,
        "teacher_utility": {"frame_utility": [1.0, 0.0, 0.8, 0.0]},
        "teacher_utility_provenance": {"split_scope": "train_only"},
        "action_target": [1, 0, 1, 0],
        "gt_boundaries": [0, 2],
    }
    ledger_row = _deploy_ledger_row(checkpoint, sha)
    ledger_row["diagnostics"]["prediction_cache"] = {"path": "hidden-cache.json"}
    _write_jsonl(samples, [sample])
    _write_jsonl(metrics, [metric])
    _write_jsonl(ledger, [ledger_row])

    with pytest.raises(ValueError, match="forbidden detector-aware deploy payload key"):
        detector_validator.validate_ledger(
            sample_jsonl=samples,
            metric_sample_jsonl=metrics,
            ledger_jsonl=ledger,
            strategy="detector_aware_fixed_384",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            require_policy_source="learned_detector_aware_policy_checkpoint",
            require_checkpoint_path=checkpoint,
            require_checkpoint_sha256=sha,
            require_paction_provenance=True,
        )

    sample["detector_aware_policy"].pop("audit")
    ledger_row["diagnostics"].pop("prediction_cache")
    _write_jsonl(samples, [sample])
    _write_jsonl(ledger, [ledger_row])
    summary = detector_validator.validate_ledger(
        sample_jsonl=samples,
        metric_sample_jsonl=metrics,
        ledger_jsonl=ledger,
        strategy="detector_aware_fixed_384",
        expected_target_len=2,
        require_selected_count=2,
        require_deployable=True,
        require_policy_source="learned_detector_aware_policy_checkpoint",
        require_checkpoint_path=checkpoint,
        require_checkpoint_sha256=sha,
        require_paction_provenance=True,
    )

    assert summary["decision"] == "C3_DETECTOR_AWARE_POLICY_LEDGER_VALIDATION_PASS"


def test_detector_aware_validator_rejects_teacher_payload_in_deploy_sample(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = detector_validator._sha256_file(checkpoint)
    sample = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"detector_aware_fixed_384": [0, 2]},
        "teacher_utility": {"frame_utility": [1.0, 0.0, 0.8, 0.0]},
        "detector_aware_policy": {
            "source": "learned_detector_aware_policy_checkpoint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha,
            "policy_checkpoint_sha256": sha,
            "policy_family": "detector_aware_offline_selector",
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
            "p_action_provenance": _paction_provenance(),
        },
    }
    metric = {
        "sample_id": "video_test_0001|0",
        "split": "training",
        "dense_len": 4,
        "valid_len": 4,
        "teacher_utility": {"frame_utility": [1.0, 0.0, 0.8, 0.0]},
        "teacher_utility_provenance": {"split_scope": "train_only"},
        "action_target": [1, 0, 1, 0],
        "gt_boundaries": [0, 2],
    }
    ledger_row = {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": [0, 2],
        "target_len": 2,
        "selected_count": 2,
        "valid_len": 4,
        "dense_len": 4,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "learned_detector_aware_policy_checkpoint",
        "policy_checkpoint_path": str(checkpoint),
        "policy_checkpoint_sha256": sha,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
        "diagnostics": {
            "uniform_visible_fill_count": 0,
            "source_strategy": "detector_aware_fixed_384",
            "policy_source": "learned_detector_aware_policy_checkpoint",
            "policy_checkpoint_path": str(checkpoint),
            "policy_checkpoint_sha256": sha,
            "p_action_provenance": _paction_provenance(),
        },
    }
    _write_jsonl(samples, [sample])
    _write_jsonl(metrics, [metric])
    _write_jsonl(ledger, [ledger_row])

    with pytest.raises(ValueError, match="forbidden detector-aware deploy payload key at teacher_utility"):
        detector_validator.validate_ledger(
            sample_jsonl=samples,
            metric_sample_jsonl=metrics,
            ledger_jsonl=ledger,
            strategy="detector_aware_fixed_384",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            require_policy_source="learned_detector_aware_policy_checkpoint",
            require_checkpoint_path=checkpoint,
            require_checkpoint_sha256=sha,
            require_paction_provenance=True,
        )

    sample.pop("teacher_utility")
    _write_jsonl(samples, [sample])
    summary = detector_validator.validate_ledger(
        sample_jsonl=samples,
        metric_sample_jsonl=metrics,
        ledger_jsonl=ledger,
        strategy="detector_aware_fixed_384",
        expected_target_len=2,
        require_selected_count=2,
        require_deployable=True,
        require_policy_source="learned_detector_aware_policy_checkpoint",
        require_checkpoint_path=checkpoint,
        require_checkpoint_sha256=sha,
        require_paction_provenance=True,
    )

    assert summary["detector_utility_coverage"] == pytest.approx(1.0)
    assert summary["detector_utility_ndcg"] == pytest.approx(1.0)
    assert summary["boundary_bracket_support@r1"] is not None


def test_detector_aware_validator_ignores_validation_teacher_utility_for_decision_metrics(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = detector_validator._sha256_file(checkpoint)
    sample = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "strategy_selected_positions": {"detector_aware_fixed_384": [0, 2]},
        "detector_aware_policy": {
            "source": "learned_detector_aware_policy_checkpoint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha,
            "policy_checkpoint_sha256": sha,
            "policy_family": "detector_aware_offline_selector",
            "stage_label": "Stage-2 detector-aware offline selector",
            "end_to_end": False,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
            "p_action_provenance": _paction_provenance(),
        },
    }
    metric = {
        "sample_id": "video_test_0001|0",
        "split": "validation",
        "dense_len": 4,
        "valid_len": 4,
        "teacher_utility": {"frame_utility": [1.0, 0.0, 0.8, 0.0]},
        "teacher_utility_provenance": {"split_scope": "train_only"},
    }
    ledger_row = {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": [0, 2],
        "target_len": 2,
        "selected_count": 2,
        "valid_len": 4,
        "dense_len": 4,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "learned_detector_aware_policy_checkpoint",
        "policy_checkpoint_path": str(checkpoint),
        "policy_checkpoint_sha256": sha,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
        "diagnostics": {
            "uniform_visible_fill_count": 0,
            "source_strategy": "detector_aware_fixed_384",
            "policy_source": "learned_detector_aware_policy_checkpoint",
            "policy_checkpoint_path": str(checkpoint),
            "policy_checkpoint_sha256": sha,
            "p_action_provenance": _paction_provenance(),
        },
    }
    _write_jsonl(samples, [sample])
    _write_jsonl(metrics, [metric])
    _write_jsonl(ledger, [ledger_row])

    summary = detector_validator.validate_ledger(
        sample_jsonl=samples,
        metric_sample_jsonl=metrics,
        ledger_jsonl=ledger,
        strategy="detector_aware_fixed_384",
        expected_target_len=2,
        require_selected_count=2,
        require_deployable=True,
        require_policy_source="learned_detector_aware_policy_checkpoint",
        require_checkpoint_path=checkpoint,
        require_checkpoint_sha256=sha,
        require_paction_provenance=True,
    )

    assert summary["detector_utility_rows"] == 0
    assert summary["detector_utility_coverage"] is None
    assert summary["detector_utility_ndcg"] is None
    assert summary["detector_utility_metric_availability"] == "not_available_no_train_only_teacher_utility"


@pytest.mark.skipif(os.name == "nt", reason="torch checkpoint smoke test runs in Linux/remote OpenTAD")
def test_detector_aware_real_cpu_checkpoint_generates_deploy_ledgers_without_monkeypatch(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.samples_with_teacher_utility.jsonl"
    deploy_jsonl = tmp_path / "deploy.samples.jsonl"
    checkpoint = tmp_path / "detector_aware_policy.pth"
    rows = [
        {
            "sample_id": "video_test_0001|0",
            "split": "training",
            "dense_len": 8,
            "valid_len": 8,
            "frame_signals": {"p_action": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]},
            "paction_positive_provenance": _paction_provenance(),
            "teacher_utility": {"signed_frame_utility": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7]},
            "teacher_utility_provenance": {"split_scope": "train_only"},
        },
        {
            "sample_id": "video_test_0002|0",
            "split": "training",
            "dense_len": 8,
            "valid_len": 8,
            "frame_signals": {"p_action": [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]},
            "paction_positive_provenance": _paction_provenance(),
            "teacher_utility": {"signed_frame_utility": [1.0, 0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3]},
            "teacher_utility_provenance": {"split_scope": "train_only"},
        },
    ]
    _write_jsonl(train_jsonl, rows)
    _write_jsonl(deploy_jsonl, [{k: v for k, v in row.items() if k not in {"teacher_utility", "teacher_utility_provenance"}} for row in rows])

    train_detector.run_training(
        train_jsonl,
        out_dir=tmp_path / "policy",
        checkpoint_path=checkpoint,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_layers=1,
        dynamic_budget_buckets=[2, 4, 6],
        device="cpu",
        seed=0,
    )
    summary = detector_pipeline.run_pipeline(
        input_jsonl=deploy_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "ledgers",
        fixed_budgets=(2, 4),
        dynamic_target_len=4,
        dynamic_budget_buckets=[2, 4, 6],
        device="cpu",
        allow_tiny_dynamic_diagnostic=True,
        max_uniform_similarity=1.0,
    )

    assert summary["decision"] == "C3_DETECTOR_AWARE_LEDGER_PIPELINE_READY"
    assert summary["ledgers"]["detector_aware_fixed_384"]["validation_summary"]["required_policy_source"] == "learned_detector_aware_policy_checkpoint"
