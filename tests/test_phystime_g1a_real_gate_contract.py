import json
from types import SimpleNamespace

import pytest

from tools.bata.run_phystime_g1a_real_gate import (
    SCHEMA_VERSION,
    _directory_inventory,
    _state_dict_sha256,
    audit_dataset_timebases,
    validate_gate_report,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GIT_SHA = "d" * 40


def test_state_dict_hash_supports_scalar_integer_buffers():
    import torch

    module = torch.nn.Module()
    module.register_buffer("step", torch.tensor(1, dtype=torch.long))
    before = _state_dict_sha256(module)
    module.step.add_(1)
    after = _state_dict_sha256(module)

    assert len(before) == 64
    assert before != after


def _step_reports():
    return [
        {
            "step": step,
            "losses": {"cost": 1.0 + step, "cls_loss": 0.5},
            "amp_scale_before": 1024.0,
            "amp_scale_after": 1024.0,
        }
        for step in range(3)
    ]


def _timebase_audit():
    return {
        "audit_pass": True,
        "audit_scope": "dataset_consumed_videos_only",
        "video_count": 2,
        "missing_consumed_video_count": 0,
        "frame_count_mismatch_count": 0,
        "records_sha256": SHA_A,
        "audited_video_names_sha256": SHA_C,
        "unreferenced_records_sha256": SHA_B,
        "split_counts": {"train": 1, "test": 1},
    }


def _variant():
    gradient = {"all_finite": True, "nonzero": True}
    return {
        "decoded_frame_count": 384,
        "raw_valid_count": 384,
        "backbone_feature_length": 192,
        "inference_backbone_feature_length": 192,
        "finite_loss": True,
        "finite_predictions": True,
        "optimizer_coverage": True,
        "optimizer_steps_requested": 3,
        "optimizer_steps_completed": 3,
        "parameter_state_changed": True,
        "amp_contract_verified": True,
        "train_window_crop_uses_gt": True,
        "train_subsample_uses_gt": False,
        "tail_window_crop_uses_gt": False,
        "tail_subsample_uses_gt": False,
        "adapter_gradient": gradient,
        "projection_gradient": gradient,
        "classification_gradient": gradient,
        "regression_gradient": gradient,
        "native_geometry_audit": {
            "feature_interpolation": False,
            "query_tensor_count": 378,
            "lineage_evidence_level": "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
        },
        "tail_native_geometry_audit": {
            "raw_valid_counts": [127],
            "invalid_native_features_zeroed": True,
            "padding_repeat_counts": [257],
            "valid_tokens_may_depend_on_padding_repeats": [True],
            "valid_tokens_depend_on_padding_after_isolation": False,
            "candidate_mask_policy": "semantic_anchor_prefix",
            "backbone_temporal_padding_isolation": {
                "strict_isolation_verified": True,
                "attention_key_value_masked": True,
                "adapter_convolution_masked": True,
                "output_invalid_features_zeroed": True,
            },
        },
        "head_geometry_debug": {
            "physical_grid_actionformer_enabled": True,
            "physical_grid_actionformer_valid_points": 378,
            "physical_grid_actionformer_axis_start_key": "phystime_g1a_axis_start_sec",
            "physical_grid_actionformer_axis_end_key": "phystime_g1a_axis_end_sec",
        },
        "tail_head_geometry_debug": {
            "physical_grid_actionformer_enabled": True,
            "physical_grid_actionformer_valid_points": 199,
        },
        "production_single_video_eval_executed": True,
        "production_single_video_detection_count": 1,
        "production_single_video_metrics": {"average_mAP": 0.0},
        "full_post_processing_executed": True,
        "prediction_time_unit": "seconds",
        "optimizer_step_reports": _step_reports(),
        "initial_state_sha256": SHA_A,
        "final_state_sha256": SHA_B,
        "canonical_config_sha256": SHA_C,
    }


def test_g1a_real_gate_contract_requires_native_counts_and_both_matched_arms():
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }

    assert validate_gate_report(report) is True


def test_g1a_real_gate_contract_fails_closed_on_provenance_or_seconds_mismatch():
    base = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }

    import copy
    import pytest

    for mutator in (
        lambda report: report.update(initial_state_match=False),
        lambda report: report.update(dataset_manifest_sha256=""),
        lambda report: report.update(target_checksum_match=False),
        lambda report: report.update(real_g0_pass=False),
        lambda report: report["timebase_audit"].update(audit_pass=False),
        lambda report: report["timebase_audit"].update(
            missing_consumed_video_count=1
        ),
        lambda report: report["variants"]["selected_axis"].update(parameter_state_changed=False),
        lambda report: report["variants"]["selected_axis"].update(optimizer_step_reports=[]),
        lambda report: report["variants"]["selected_axis"]["optimizer_step_reports"][1].update(
            amp_scale_after=float("nan")
        ),
        lambda report: report["variants"]["selected_axis"].update(
            final_state_sha256=report["variants"]["selected_axis"]["initial_state_sha256"]
        ),
        lambda report: report.update(checkpoint_sha256="not-a-digest"),
        lambda report: report["variants"]["selected_axis"].update(prediction_time_unit="dense_index"),
        lambda report: report["variants"]["physical_metric"]["tail_native_geometry_audit"].update(
            raw_valid_counts=[384]
        ),
    ):
        report = copy.deepcopy(base)
        mutator(report)
        with pytest.raises(RuntimeError):
            validate_gate_report(report)


def test_dataset_inventory_hashes_file_content_not_only_size(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"same-size-a")
    before = _directory_inventory(tmp_path)

    video.write_bytes(b"same-size-b")
    after = _directory_inventory(tmp_path)

    assert before["inventory_sha256"] != after["inventory_sha256"]
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]
    assert before["hash_scope"] == "full_file_content_sha256_merkle_v1"


def test_full_dataset_timebase_audit_uses_the_same_fail_closed_contract(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "video_validation_1.mp4").write_bytes(b"train")
    (test_dir / "video_test_1.mp4").write_bytes(b"test")
    (test_dir / "video_test_unused.mp4").write_bytes(b"unused")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "video_validation_1": {"frame": 400, "duration": 20.0},
                    "video_test_1": {"frame": 200, "duration": 10.0},
                }
            }
        ),
        encoding="utf-8",
    )
    raw_step = {
        "type": "BuildPhysTimeRawFrameGeometry",
        "fps_relative_tolerance": 0.0125,
        "duration_relative_tolerance": 0.0125,
        "frame_count_relative_tolerance": 0.0001,
    }
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(data_path=str(train_dir), pipeline=[raw_step]),
            test=SimpleNamespace(data_path=str(test_dir), pipeline=[raw_step]),
        )
    )

    consumed_video_names = {
        "train": {"video_validation_1"},
        "test": {"video_test_1"},
    }
    report = audit_dataset_timebases(
        cfg,
        annotation,
        decoder_probe=lambda path: (20.0, 400 if "validation" in path.name else 200),
        dataset_video_names=consumed_video_names,
    )

    assert report["audit_pass"] is True
    assert report["audit_scope"] == "dataset_consumed_videos_only"
    assert report["video_count"] == 2
    assert report["directory_file_counts"] == {"train": 1, "test": 2}
    assert report["unreferenced_file_counts"] == {"train": 0, "test": 1}
    assert report["unreferenced_video_names"] == {
        "train": [],
        "test": ["video_test_unused"],
    }
    assert report["missing_consumed_video_count"] == 0
    assert report["frame_count_mismatch_count"] == 0
    assert report["records_sha256"]

    with pytest.raises(ValueError, match="FPS"):
        audit_dataset_timebases(
            cfg,
            annotation,
            decoder_probe=lambda path: (25.0, 400 if "validation" in path.name else 200),
            dataset_video_names=consumed_video_names,
        )
