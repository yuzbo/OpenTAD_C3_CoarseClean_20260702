import ast
from pathlib import Path

import pytest

import tools.bata.run_phystime_p0_fullprecision_gate as gate_validator
import tools.bata.validate_phystime_p0_fullprecision_replay as replay_validator
import tools.bata.validate_phystime_p0_fullprecision_suite as suite_validator


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_p0_configs_are_frozen_eval_variants_only():
    physical = _read(
        "configs/adatad/thumos/"
        "phystime_g1a_physical_metric_native_j192_p0_replay.py"
    )
    selected = _read(
        "configs/adatad/thumos/"
        "phystime_g1a_selected_axis_native_j192_p0_replay.py"
    )
    for source, base_name in (
        (physical, "phystime_g1a_physical_metric_native_j192.py"),
        (selected, "phystime_g1a_selected_axis_native_j192.py"),
    ):
        tree = ast.parse(source)
        assert tree is not None
        assert base_name in source
        assert "save_pre_cross_window_detections=True" in source
        assert "save_post_processing_audit=True" in source
        assert "save_dict=True" in source
        assert "filter_invalid_proposals=True" in source
        assert "proposal_min_duration=1.0e-6" in source
        assert "round_before_cross_window_nms=False" in source
        assert "round_after_cross_window_nms=False" in source
        assert "scheduler" not in source
        assert "workflow" not in source


def test_p0_contract_does_not_change_the_ordinary_main_config():
    source = _read(
        "configs/adatad/thumos/"
        "phystime_g1a_physical_metric_native_j192.py"
    )
    assert "filter_invalid_proposals" not in source
    assert "proposal_min_duration" not in source
    assert "round_before_cross_window_nms" not in source
    assert "round_after_cross_window_nms" not in source


def test_submitter_uses_one_gate_four_replays_and_one_suite_validator():
    source = _read("scripts/submit_phystime_p0_fullprecision_replay.sh")
    assert "SOURCE_COMMIT=\"${PHYSTIME_SOURCE_COMMIT:-0dc5851" in source
    assert "SOURCE_TREE=\"${PHYSTIME_SOURCE_TREE:-bddc9b9" in source
    assert "gate_job=\"$(submit" in source
    assert 'submit --dependency="afterok:${gate_job}"' in source
    for variant in (
        "selected_online",
        "selected_ema",
        "physical_online",
        "physical_ema",
    ):
        assert variant in source
    assert '"new_training": false' in source
    assert "tools/train.py" not in source
    assert "p0_suite" in source
    assert "run_phystime_p0_fullprecision_suite_slurm.sh" in source
    assert "echo '#SBATCH --gpus=1'" in source
    assert "--gres=gpu:1" not in source
    assert '"suite_scheduler_gpu_allocation": 1' in source
    assert '"suite_validator_uses_cuda": false' in source
    assert 'gate["dataset_manifest"]' in source
    assert 'for key in ("annotation", "class_map", "train_videos", "test_videos")' in source
    assert "${BASE}/raw/Validation Data/validation" not in source
    assert "${BASE}/raw/Test Data/TH14_test_set_mp4" not in source
    assert (
        'afterok:${jobs[selected_online]}:${jobs[selected_ema]}:'
        '${jobs[physical_online]}:${jobs[physical_ema]}'
    ) in source


def test_replay_runner_never_invokes_training_and_requires_validator():
    source = _read("scripts/run_phystime_p0_fullprecision_replay_slurm.sh")
    assert "tools/test.py" in source
    assert "tools/train.py" not in source
    assert "replay_phystime_p0_fullprecision_nms.py" in source
    assert "validate_phystime_p0_fullprecision_replay.py" in source
    assert "DIRECT_INFERENCE_COMPLETE" in source
    assert "P0_COMPLETE.json" in source
    assert '"solver.ema=${USE_EMA}"' in source


def test_direct_test_binds_checkpoint_epoch_to_all_artifacts():
    source = _read("tools/test.py")
    assert 'evaluation_epoch = int(checkpoint["epoch"])' in source
    assert "evaluation_epoch=evaluation_epoch" in source


def test_gate_binds_both_source_weight_sets_and_focused_tests():
    source = _read("tools/bata/run_phystime_p0_fullprecision_gate.py")
    assert 'SOURCE_COMMIT = "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132"' in source
    assert 'SOURCE_TREE = "bddc9b9386604d00d213275a47ce7997b35d3f4c"' in source
    assert 'for key in ("state_dict", "state_dict_ema")' in source
    assert '"focused_tests"' in source
    assert '"new_training": False' in source
    assert "inference_semantic_sha256" in source
    assert "EXPECTED_COORDINATE_MODES" in source


@pytest.mark.parametrize(
    ("arm", "config_name", "coordinate_mode"),
    (
        (
            "selected_axis",
            "phystime_g1a_selected_axis_native_j192_p0_replay.py",
            "uniform_rank_seconds",
        ),
        (
            "physical_metric",
            "phystime_g1a_physical_metric_native_j192_p0_replay.py",
            "physical_time_seconds",
        ),
    ),
)
def test_gate_executes_runtime_config_contract(
    arm,
    config_name,
    coordinate_mode,
):
    report = gate_validator.validate_runtime_config(
        arm,
        ROOT / "configs" / "adatad" / "thumos" / config_name,
    )
    assert set(report["coordinate_modes"]["pipelines"].values()) == {
        coordinate_mode
    }
    assert (
        report["coordinate_modes"]["head_time_contract"]
        == gate_validator.EXPECTED_HEAD_TIME_CONTRACT
    )


def test_gate_rejects_file_hash_in_place_of_canonical_config_hash():
    config_path = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_physical_metric_native_j192.py"
    )
    config = gate_validator.Config.fromfile(config_path, lazy_import=False)
    manifest = {
        "config_sha256": gate_validator.canonical_sha256(config.to_dict())
    }
    _, binding = gate_validator.load_bound_source_config(config_path, manifest)
    assert binding["canonical_sha256"] == manifest["config_sha256"]
    with pytest.raises(ValueError, match="canonical hash"):
        gate_validator.load_bound_source_config(
            config_path,
            {"config_sha256": gate_validator.sha256_file(config_path)},
        )


def _one_prediction_audit():
    reason_counts = {
        "malformed_detection": 0,
        "malformed_segment": 0,
        "non_finite_segment": 0,
        "non_finite_score": 0,
        "non_positive_duration": 0,
        "invalid_label": 0,
    }
    validation = {
        "video_name": "video",
        "input_detections": 1,
        "valid_detections": 1,
        "invalid_detections": 0,
        "invalid_reason_counts": reason_counts,
        "invalid_samples": [],
    }
    video = {
        "video_name": "video",
        "input_detections": 1,
        "valid_detections": 1,
        "invalid_detections": 0,
        "filtered_detections": 0,
        "raw_invalid_detections": 0,
        "raw_filtered_detections": 0,
        "effective_input_detections": 1,
        "effective_invalid_detections": 0,
        "effective_filtered_detections": 0,
        "rounding_induced_invalid_detections": 0,
        "kept_for_nms": 1,
        "post_nms_detections": 1,
        "post_nms_invalid_detections": 0,
        "post_nms_filtered_detections": 0,
        "pre_nms_rounding_changed_segment_values": 0,
        "pre_nms_rounding_changed_scores": 0,
        "invalid_reason_counts": reason_counts,
        "raw_validation": dict(validation),
        "effective_validation": dict(validation),
        "output_validation": dict(validation),
    }
    aggregate = {
        key: value
        for key, value in video.items()
        if isinstance(value, int)
    }
    aggregate.update(
        {
            "videos": 1,
            "videos_with_invalid_detections": 0,
            "invalid_reason_counts": reason_counts,
            "raw_invalid_reason_counts": reason_counts,
            "effective_invalid_reason_counts": reason_counts,
        }
    )
    return {
        "schema_version": "opentad_cross_window_nms_audit_v1",
        "nms_applied": True,
        "policy": {"filter_invalid_proposals": True},
        "aggregate": aggregate,
        "videos": {"video": video},
    }


def test_validator_rejects_wrong_epoch_and_tampered_audit_count():
    with pytest.raises(ValueError, match="epoch mismatch"):
        replay_validator.validate_evaluation_epoch(
            {"evaluation_epoch": 58},
            "tampered metrics",
        )
    input_payload = {"results": {"video": [{}]}}
    output_payload = {"results": {"video": [{}]}}
    audit = _one_prediction_audit()
    replay_validator.validate_audit_count_conservation(
        audit,
        input_payload,
        output_payload,
        "valid audit",
    )
    audit["aggregate"]["input_detections"] = 2
    with pytest.raises(ValueError, match="aggregate input_detections differs"):
        replay_validator.validate_audit_count_conservation(
            audit,
            input_payload,
            output_payload,
            "tampered audit",
        )


def test_suite_rejects_completion_from_another_run_directory(tmp_path):
    expected = tmp_path / "expected"
    wrong = tmp_path / "wrong"
    with pytest.raises(ValueError, match="run_dir mismatch"):
        suite_validator.validate_completion_run_dir(
            {"run_dir": str(wrong)},
            expected,
            "tampered",
        )


def test_replay_tool_defines_the_required_two_by_two_modes():
    source = _read("tools/bata/replay_phystime_p0_fullprecision_nms.py")
    for mode in (
        "legacy_unfiltered",
        "legacy_filtered",
        "fullprecision_unfiltered",
        "fullprecision_filtered",
    ):
        assert f'"{mode}"' in source
    assert "direct_fullprecision_filtered_equivalence" in source
    assert "source_legacy_ema_equivalence" in source
    assert "validity_filter_effect_fullprecision" in source


def test_suite_validator_is_independent_and_closes_cross_arm_contract():
    source = _read(
        "tools/bata/validate_phystime_p0_fullprecision_suite.py"
    )
    assert "EXPECTED_RUNS" in source
    assert "cross_arm_physical_minus_selected" in source
    assert "weight_source_ema_minus_online" in source
    assert "within_run_decision_diagnostics" in source
    assert "proposal_recall_by_duration_and_iou" in source
    assert "P0_COMPLETE.json" in source
    assert "replay_phystime_p0_fullprecision_nms" not in source


def test_single_run_validator_does_not_import_producer_mode_definitions():
    source = _read(
        "tools/bata/validate_phystime_p0_fullprecision_replay.py"
    )
    assert "EXPECTED_MODE_SPECS" in source
    assert "from tools.bata.replay_phystime" not in source
    assert "build_delta_report" in source


def test_suite_metric_delta_uses_physical_minus_selected_direction():
    delta = suite_validator.metric_delta(
        {"average_mAP": 0.6, "mAP@0.7": 0.3},
        {"average_mAP": 0.4, "mAP@0.7": 0.2},
    )
    assert delta["fraction"] == pytest.approx(
        {"average_mAP": 0.2, "mAP@0.7": 0.1}
    )
    assert delta["percentage_points"] == pytest.approx(
        {"average_mAP": 20.0, "mAP@0.7": 10.0}
    )


def test_suite_decision_diagnostics_detect_suppression_and_rank_changes():
    lhs = {
        "results": {
            "video": [
                {
                    "segment": [0.001, 1.001],
                    "label": "action",
                    "score": 0.9,
                },
                {
                    "segment": [2.0, 3.0],
                    "label": "action",
                    "score": 0.8,
                },
            ]
        }
    }
    rhs = {
        "results": {
            "video": [
                {
                    "segment": [2.0, 3.0],
                    "label": "action",
                    "score": 0.7,
                }
            ]
        }
    }
    report = suite_validator.compare_prediction_decisions(lhs, rhs)
    assert report["lhs_prediction_count"] == 2
    assert report["rhs_prediction_count"] == 1
    assert report["matched_prediction_count"] == 1
    assert report["lhs_only_count"] == 1
    assert report["rhs_only_count"] == 0
    assert report["rank_abs_delta"]["mean"] == pytest.approx(1.0)


def test_suite_decision_diagnostics_treat_small_boundary_shift_as_match():
    lhs = {
        "results": {
            "video": [
                {
                    "segment": [1.006, 2.006],
                    "label": "action",
                    "score": 0.9,
                }
            ]
        }
    }
    rhs = {
        "results": {
            "video": [
                {
                    "segment": [1.0, 2.0],
                    "label": "action",
                    "score": 0.9,
                }
            ]
        }
    }
    report = suite_validator.compare_prediction_decisions(lhs, rhs)
    assert report["matched_prediction_count"] == 1
    assert report["lhs_only_count"] == 0
    assert report["rhs_only_count"] == 0
    assert report["boundary_start_abs_seconds"]["mean"] == pytest.approx(
        0.006
    )
    assert report["boundary_end_abs_seconds"]["mean"] == pytest.approx(
        0.006
    )
    assert (
        report["strict_rounded_identity"]["matched_prediction_count"] == 0
    )


def test_suite_proposal_recall_reports_short_action_high_iou():
    prediction = {
        "results": {
            "video": [
                {
                    "segment": [0.0, 1.0],
                    "label": "short",
                    "score": 0.9,
                },
                {
                    "segment": [2.0, 6.0],
                    "label": "long",
                    "score": 0.8,
                },
            ]
        }
    }
    ground_truth = [
        {
            "video": "video",
            "label": "short",
            "start": 0.0,
            "end": 1.0,
            "duration": 1.0,
        },
        {
            "video": "video",
            "label": "middle",
            "start": 1.0,
            "end": 3.0,
            "duration": 2.0,
        },
        {
            "video": "video",
            "label": "long",
            "start": 2.0,
            "end": 6.0,
            "duration": 4.0,
        },
    ]
    report = suite_validator.proposal_recall_diagnostics(
        prediction,
        ground_truth,
    )
    assert (
        report["strata"]["short_q1"]["proposal_recall"]["0.9"]
        == pytest.approx(1.0)
    )
    assert report["strata"]["all"]["proposal_recall"]["0.9"] == pytest.approx(
        2.0 / 3.0
    )
