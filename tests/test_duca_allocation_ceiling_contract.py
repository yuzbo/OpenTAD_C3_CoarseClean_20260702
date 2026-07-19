from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.authorize_duca_allocation_validation import authorize_validation
from tools.bata.diagnose_duca_allocation_family_ceiling import (
    _binary_ranking_metrics,
    diagnose_record,
    run_diagnostic,
    validate_input_record,
)
from tools.bata.duca_exact_physical_solver import GroundTruthObjectiveSpec
from tools.bata.evaluate_duca_allocation_candidates import (
    _summarize,
    candidate_dataset_indices,
)
from tools.bata.export_duca_allocation_ceiling_inputs import (
    build_record,
    canonical_sha256,
    data_directory_provenance,
    deduplicate_sliding_windows,
    extract_center_frame_indices,
    sha256,
)
from tools.bata.profile_duca_allocation_solver_cost import profile_solver
from tools.bata.finalize_duca_allocation_ceiling_gate import _validate_gt_runtime
from tools.bata.subset_duca_allocation_inputs import _hash_video_round_robin
from tools.bata.validate_duca_allocation_ceiling_artifact import (
    validate_artifact,
    validate_artifact_receipt,
)
from tools.bata.validate_duca_allocation_candidate_loss_artifact import (
    validate_candidate_artifact,
)
from tools.bata.validate_duca_allocation_solver_cost_artifact import (
    validate_solver_cost_artifact,
)


def _source() -> dict:
    fixture = Path(__file__).parent / "fixtures" / "duca_allocation"
    annotation = (fixture / "annotation.json").resolve()
    class_map = (fixture / "class_map.txt").resolve()
    data_path = (fixture / "videos").resolve()
    return {
        "git_commit": "a" * 40,
        "git_clean": True,
        "annotation_path": str(annotation),
        "annotation_sha256": sha256(annotation),
        "class_map_path": str(class_map),
        "class_map_sha256": sha256(class_map),
        "data_path": str(data_path),
        **data_directory_provenance(data_path),
        "dataset_subset_name": "training",
        "dataset_test_mode": False,
        "dataset_filter_gt": False,
        "dataset_ioa_thresh": 1.0e-8,
        "dataset_feature_stride": 4,
        "dataset_sample_stride": 1,
        "dataset_window_size": 768,
        "dataset_window_overlap_ratio": 0.5,
        "dataset_offset_frames": 0,
        "dataset_config_sha256": "f" * 64,
        "dataset_window_manifest_sha256": "1" * 64,
        "dataset_window_count": 10,
        "dataset_window_deduplication": "exact_video_start_identity_keep_first",
        "dataset_duplicate_window_count_removed": 0,
        "config": "/tmp/config.py",
        "config_sha256": "b" * 64,
        "checkpoint": "/tmp/checkpoint.pth",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint_epoch": 131,
        "split": "train",
        "selector_only_inference": True,
        "detector_backbone_executed": False,
        "uses_gt_for_score_generation": False,
        "validation_authorized": False,
    }


def _input_record() -> dict:
    valid_len = 8
    selector_output = {
        "selector_outputs": {
            "p_action": [[0.1 * index for index in range(valid_len)]],
            "actionness_logits": [[float(index - 3) for index in range(valid_len)]],
            "transition_policy_scores": [[0.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0]],
            "transition_score": [[0.0, 0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.4]],
            "abs_delta_p_action": [[0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
            "uncertainty": [[0.5] * valid_len],
        }
    }
    meta = {
        "video_name": "video_test_0000001",
        "window_start_frame": 8,
        "snippet_stride": 4,
        "frame_inds": [
            [max(0, 8 + 4 * index - 2), 8 + 4 * index, 8 + 4 * index + 2]
            for index in range(valid_len)
        ],
        "avg_fps": 25.0,
        "fps": 25.0,
        "total_frames": 100,
    }
    return build_record(
        selector_output=selector_output,
        masks=[[True] * valid_len],
        gt_segments=[[[1.0, 2.0], [5.0, 6.0]]],
        metas=[meta],
        source=_source(),
        split="train",
        requested_budget=4,
        seen_count=0,
        coordinate_tolerance_frames=0.0,
    )[0]


def test_center_frame_extraction_uses_clip_center_not_clip_edges() -> None:
    meta = {"frame_inds": [[1, 2, 3], [5, 6, 7], [9, 10, 11]]}
    assert extract_center_frame_indices(meta, 3) == [2.0, 6.0, 10.0]


def test_export_record_binds_actual_physical_axis_and_no_leak_contract() -> None:
    row = _input_record()
    assert row["physical_axis"]["source_frames"] == [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0]
    assert row["coordinate_audit"]["passed"] is True
    assert row["decision_contract"]["gt_passed_to_selector"] is False
    assert row["gt_role"] == "privileged_diagnostic_only_never_score_generation"
    validate_input_record(row, context="fixture")


def test_input_contract_rejects_gt_leakage_even_with_rehashed_record() -> None:
    row = _input_record()
    row["decision_contract"]["gt_passed_to_selector"] = True
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    with pytest.raises(ValueError, match="leakage"):
        validate_input_record(row, context="fixture")


def test_input_contract_reconstructs_gt_from_bound_annotation() -> None:
    row = _input_record()
    row["gt_segments"] = [[0.5, 1.5], [4.0, 5.0]]
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    with pytest.raises(ValueError, match="bound source annotation"):
        validate_input_record(row, context="fixture")


def test_validation_input_requires_real_test_mode_and_no_gt() -> None:
    row = _input_record()
    row["split"] = "test"
    row["source"]["split"] = "test"
    row["source"]["dataset_subset_name"] = "validation"
    row["source"]["dataset_test_mode"] = True
    row["source"]["validation_authorized"] = True
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    with pytest.raises(ValueError, match="contains runtime GT"):
        validate_input_record(row, context="fixture")


def test_input_contract_rejects_source_frame_outside_video() -> None:
    row = _input_record()
    row["physical_axis"]["total_frames"] = 20
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    with pytest.raises(ValueError, match="outside the video"):
        validate_input_record(row, context="fixture")


def test_export_record_rejects_out_of_prefix_gt_instead_of_clipping() -> None:
    row = _input_record()
    selector_output = {
        "selector_outputs": {
            "p_action": [row["scores"]["p_action"]],
            "actionness_logits": [row["scores"]["actionness_logits"]],
            "transition_policy_scores": [row["scores"]["transition_policy_scores"]],
            "transition_score": [row["scores"]["raw_transition_scores"]],
            "abs_delta_p_action": [row["scores"]["abs_delta_p_action"]],
            "uncertainty": [row["scores"]["uncertainty"]],
        }
    }
    meta = {
        "video_name": row["video_id"],
        "window_start_frame": 8,
        "snippet_stride": 4,
        "frame_inds": [[8 + 4 * index] for index in range(row["valid_len"])],
        "avg_fps": 25.0,
        "fps": 25.0,
        "total_frames": 100,
    }
    with pytest.raises(ValueError, match="outside the exported valid prefix"):
        build_record(
            selector_output=selector_output,
            masks=[[True] * row["valid_len"]],
            gt_segments=[[[-10.0, -5.0]]],
            metas=[meta],
            source=_source(),
            split="train",
            requested_budget=4,
            seen_count=0,
            coordinate_tolerance_frames=0.0,
        )


def test_input_contract_rejects_decoder_annotation_timeline_drift() -> None:
    row = _input_record()
    row["physical_axis"]["annotation_fps"] = 30.0
    row["timeline_audit"] = {
        "decoder_fps": 25.0,
        "annotation_fps": 30.0,
        "absolute_fps_error": 5.0,
        "tolerance_fps": 25.0 / 99.0,
        "cumulative_drift_frames": 19.8,
        "tolerance_frames": 1.0,
        "passed": True,
    }
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    with pytest.raises(ValueError, match="timelines are misaligned"):
        validate_input_record(row, context="fixture")


def test_tied_binary_scores_use_attainable_group_thresholds() -> None:
    first = _binary_ranking_metrics([0.0, 0.0], [1, 0])
    second = _binary_ranking_metrics([0.0, 0.0], [0, 1])
    assert first == second
    assert first == {
        "roc_auc": 0.5,
        "average_precision": 0.5,
        "best_f1": pytest.approx(2.0 / 3.0),
    }


def test_diagnostic_emits_deploy_and_privileged_families_with_separate_roles() -> None:
    row = _input_record()
    result = diagnose_record(
        row,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=True,
    )
    families = {family["family_key"]: family for family in result["families"]}
    assert {
        "A_exact_uniform",
        "B_one_per_uniform_cell",
        "C_uniform_scaffold_residual",
        "D_deploy_score",
        "D_privileged_gt_ceiling",
        "E_privileged_unrestricted_gt",
    } == set(families)
    assert families["D_deploy_score"]["deployable"] is True
    assert families["D_deploy_score"]["privileged"] is False
    assert families["D_privileged_gt_ceiling"]["deployable"] is False
    assert families["D_privileged_gt_ceiling"]["privileged"] is True
    assert families["E_privileged_unrestricted_gt"]["deployable"] is False
    assert result["coarse_signal_metrics"]["action_p_roc_auc"] is not None
    assert result["coarse_signal_metrics"]["transition_policy_average_precision"] is not None


def test_end_to_end_artifact_validator_recomputes_hashes_geometry_and_roles(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    result = validate_artifact(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
    )
    assert result["validation_passed"] is True
    assert result["sample_count"] == 1


def test_validator_rejects_unknown_family_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="none",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    row["families"][0]["silent_extra"] = True
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    output_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    from tools.bata.export_duca_allocation_ceiling_inputs import sha256

    summary["output_jsonl_sha256"] = sha256(output_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict family fields"):
        validate_artifact(
            input_jsonl=input_path,
            output_jsonl=output_path,
            summary_json=summary_path,
        )


def test_validator_resolves_gt_milp_and_rejects_rehashed_nonoptimal_payload(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    family = next(
        item
        for item in row["families"]
        if item["family_key"] == "D_privileged_gt_ceiling"
    )
    family["positions"] = [0, 1, 2, 3]
    family["gt_solver"]["positions"] = [0, 1, 2, 3]
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    output_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_jsonl_sha256"] = sha256(output_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact_replay"):
        validate_artifact(
            input_jsonl=input_path,
            output_jsonl=output_path,
            summary_json=summary_path,
        )


def test_validator_requires_summary_bound_gt_family_set(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    row["families"] = [
        family
        for family in row["families"]
        if family["family_key"] != "E_privileged_unrestricted_gt"
    ]
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    output_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_jsonl_sha256"] = sha256(output_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="family set differs"):
        validate_artifact(
            input_jsonl=input_path,
            output_jsonl=output_path,
            summary_json=summary_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["contract"].__setitem__("offline_full_window", False), "contract"),
        (
            lambda row: next(
                family
                for family in row["families"]
                if family["family_key"] == "D_privileged_gt_ceiling"
            )["gt_solver"].__setitem__("mip_gap", False),
            "numeric",
        ),
    ],
)
def test_validator_rejects_rehashed_contract_and_non_numeric_zero_mip_gap(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    mutation(row)
    row.pop("record_sha256")
    row["record_sha256"] = canonical_sha256(row)
    output_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_jsonl_sha256"] = sha256(output_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_artifact(
            input_jsonl=input_path,
            output_jsonl=output_path,
            summary_json=summary_path,
        )


def test_ceiling_validation_receipt_is_hash_bound(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "ceiling.jsonl"
    summary_path = tmp_path / "summary.json"
    receipt_path = tmp_path / "validation.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="both",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    receipt = validate_artifact(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    assert validate_artifact_receipt(
        validation_json=receipt_path,
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        require_gt_solver_replay=True,
    )["validation_passed"]
    receipt["output_jsonl_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_artifact_receipt(
            validation_json=receipt_path,
            input_jsonl=input_path,
            output_jsonl=output_path,
            summary_json=summary_path,
            require_gt_solver_replay=True,
        )


def test_dataset_data_manifest_detects_content_drift(tmp_path: Path) -> None:
    data = tmp_path / "videos"
    data.mkdir()
    video = data / "sample.mp4"
    video.write_bytes(b"first")
    first = data_directory_provenance(data)
    video.write_bytes(b"second")
    second = data_directory_provenance(data)
    assert first["dataset_data_manifest_sha256"] != second["dataset_data_manifest_sha256"]


def test_gt_runtime_projection_fails_closed_above_bound(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "duca_allocation_gt_runtime_projection_v1",
                "gt_generation_seconds": 10.0,
                "gt_validation_seconds": 10.0,
                "projected_gt32_seconds": 640.0,
                "max_projected_gt32_seconds": 600.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds"):
        _validate_gt_runtime(path, max_projected_gt32_seconds=600.0)


def test_validation_authorization_binds_training_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "go.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "duca_allocation_training_suite_evidence_v1",
                "status": "training_side_ceiling_complete_human_go_kill_required",
                "git_commit": "a" * 40,
                "checkpoint_sha256": "b" * 64,
                "pretrain_sha256": "c" * 64,
                "decision_contract": {
                    "validation_subset_consumed": False,
                    "selector_training_authorized": False,
                    "paper_claim_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = authorize_validation(
        training_suite_evidence_json=evidence,
        expected_commit="a" * 40,
        decision="GO",
        output_json=output,
    )
    assert receipt["decision"] == "GO"
    assert receipt["training_suite_evidence_json_sha256"] == sha256(evidence)
    with pytest.raises(FileExistsError):
        authorize_validation(
            training_suite_evidence_json=evidence,
            expected_commit="a" * 40,
            decision="GO",
            output_json=output,
        )


def test_hash_video_round_robin_spreads_subset_before_reusing_videos() -> None:
    records = [
        {"video_id": video_id, "sample_id": f"{video_id}|{index}"}
        for video_id in ("v1", "v2", "v3")
        for index in range(3)
    ]
    selected = _hash_video_round_robin(
        records,
        first_n=3,
        seed="fixed-test-seed",
    )
    assert len({row["video_id"] for row in selected}) == 3
    assert selected == _hash_video_round_robin(
        records,
        first_n=3,
        seed="fixed-test-seed",
    )


def test_candidate_dataset_is_restricted_before_video_decoding() -> None:
    class Dataset:
        data_list = [
            ["v1", {}, {}, [0, 4, 8]],
            ["v1", {}, {}, [8, 12, 16]],
            ["v2", {}, {}, [0, 4, 8]],
        ]

    assert candidate_dataset_indices(Dataset(), ["v2|0", "v1|8"]) == [2, 1]
    with pytest.raises(ValueError, match="cannot resolve"):
        candidate_dataset_indices(Dataset(), ["v3|0"])


def test_route_local_window_deduplication_removes_only_exact_duplicates() -> None:
    class Dataset:
        data_list = [
            ["v1", {}, {}, [0, 4, 8]],
            ["v1", {}, {}, [8, 12, 16]],
            ["v1", {}, {}, [8, 12, 16]],
        ]

    dataset = Dataset()
    assert deduplicate_sliding_windows(dataset) == 1
    assert len(dataset.data_list) == 2
    assert dataset._duca_duplicate_window_count_removed == 1

    class Conflict:
        data_list = [
            ["v1", {}, {}, [8, 12, 16]],
            ["v1", {}, {}, [8, 13, 16]],
        ]

    with pytest.raises(ValueError, match="conflicting windows"):
        deduplicate_sliding_windows(Conflict())


def test_candidate_validator_rebinds_positions_to_ceiling(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    ceiling_path = tmp_path / "ceiling.jsonl"
    ceiling_summary_path = tmp_path / "ceiling.summary.json"
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_summary_path = tmp_path / "candidate.summary.json"
    input_path.write_text(json.dumps(_input_record(), sort_keys=True) + "\n", encoding="utf-8")
    run_diagnostic(
        input_jsonl=input_path,
        output_jsonl=ceiling_path,
        summary_json=ceiling_summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        gt_families="none",
        objective_spec=GroundTruthObjectiveSpec(lex_block_size=8),
        quantization_scale=1000,
        gt_time_limit_seconds=None,
        compute_upper_envelopes=False,
    )
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))
    family_map = {
        family["family_key"]: family
        for family in ceiling["families"]
    }
    source = {
        **_source(),
        "ceiling_jsonl": str(ceiling_path.resolve()),
        "ceiling_jsonl_sha256": sha256(ceiling_path),
        "ceiling_validation": {"validation_passed": True},
        "frozen_loss_normalizer": 100.0,
    }
    rows = []
    for index, family_key in enumerate(("A_exact_uniform", "D_deploy_score")):
        family = family_map[family_key]
        cls_loss = 1.0 + index
        reg_loss = 0.25
        row = {
            "schema_version": "duca_allocation_candidate_detector_loss_v1",
            "sample_id": ceiling["sample_id"],
            "video_id": ceiling["video_id"],
            "family_key": family_key,
            "selected_positions": list(family["positions"]),
            "selected_count": len(family["positions"]),
            "dense_valid_len": ceiling["valid_len"],
            "privileged": family["privileged"],
            "deployable": family["deployable"],
            "losses": {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss,
                "detector_loss": cls_loss + reg_loss,
                "physical_grid_debug": {
                    "physical_grid_actionformer_enabled": True,
                },
            },
            "source": source,
            "contract": {
                "model_training": False,
                "checkpoint_mutation": False,
                "dense_axis_gt": True,
                "selected_axis_gt_remap": False,
                "physical_grid_actionformer": True,
                "gt_used_for_selection": family["privileged"],
                "detector_loss_is_empirical_not_combinatorial_oracle": True,
            },
        }
        row["record_sha256"] = canonical_sha256(row)
        rows.append(row)
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _summarize(
        rows,
        output_path=candidate_path.resolve(),
        source=source,
        requested_family_keys=("A_exact_uniform", "D_deploy_score"),
    )
    candidate_summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    assert validate_candidate_artifact(
        ceiling_jsonl=ceiling_path,
        candidate_jsonl=candidate_path,
        summary_json=candidate_summary_path,
    )["validation_passed"]

    rows[1]["selected_positions"] = list(family_map["A_exact_uniform"]["positions"])
    rows[1].pop("record_sha256")
    rows[1]["record_sha256"] = canonical_sha256(rows[1])
    candidate_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["output_jsonl_sha256"] = sha256(candidate_path)
    candidate_summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positions differ"):
        validate_candidate_artifact(
            ceiling_jsonl=ceiling_path,
            candidate_jsonl=candidate_path,
            summary_json=candidate_summary_path,
        )


def test_solver_cost_validator_replays_every_selection_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    samples_path = tmp_path / "solver.samples.jsonl"
    summary_path = tmp_path / "solver.summary.json"
    input_path.write_text(
        json.dumps(_input_record(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profile_solver(
        input_jsonl=input_path,
        output_samples_jsonl=samples_path,
        output_summary_json=summary_path,
        score_key="transition_policy_scores",
        cap_policy="uniform_reference",
        cap_value=None,
        quantization_scale=1000,
        warmup_samples=0,
        samples=3,
    )
    assert validate_solver_cost_artifact(
        input_jsonl=input_path,
        samples_jsonl=samples_path,
        summary_json=summary_path,
    )["validation_passed"]

    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["quantized_objective"] += 1
    rows[0].pop("record_sha256")
    rows[0]["record_sha256"] = canonical_sha256(rows[0])
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["samples_jsonl_sha256"] = sha256(samples_path)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="objective mismatch"):
        validate_solver_cost_artifact(
            input_jsonl=input_path,
            samples_jsonl=samples_path,
            summary_json=summary_path,
        )


def test_validation_replay_config_retains_physical_coordinates_without_gt() -> None:
    root = Path(__file__).resolve().parents[1]
    validation_config = (
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "duca_allocation_ceiling_validation_windows.py"
    ).read_text(encoding="utf-8")
    replay_config = (
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "duca_allocation_ceiling_physical_grid_replay.py"
    ).read_text(encoding="utf-8")
    assert 'execution_split="test"' in validation_config
    assert "test_mode=True" in validation_config
    assert '"frame_inds"' in validation_config
    assert "window_overlap_ratio=0.5" in validation_config
    assert "gt_available_to_selector=False" in validation_config
    assert "duca_allocation_ceiling_validation_windows.py" in replay_config


def test_training_submitter_requests_generic_gpu_for_every_dag_node() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_duca_allocation_ceiling_training_suite.sh"
    ).read_text(encoding="utf-8")
    assert (
        'write_header "${DIAGNOSTIC_JOB}" "dac-diag-${SHORT_COMMIT}" 1'
        in script
    )
    assert (
        'write_header "${COMPLETION_JOB}" "dac-done-${SHORT_COMMIT}" 1'
        in script
    )
    assert "SLURM_CLUSTER_NAME" in script
    assert "rollback_partial_submission" in script
    assert 'sbatch --parsable --clusters="${TARGET_CLUSTER}"' in script


def test_validation_export_is_authorized_and_gt_free() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_duca_allocation_validation_export.sh"
    ).read_text(encoding="utf-8")
    assert "DUCA_ALLOCATION_VALIDATION_GO_JSON" in script
    assert "DUCA_ALLOCATION_VALIDATION_GO_SHA256" in script
    assert 'mkdir "${GO_JSON}.consumed"' in script
    assert "--split test" in script
    assert "--validation-authorized" in script
    assert "--gt-families none" in script
    assert '"runtime_gt_input": false' in script
    assert '"selected_axis_gt_remap": false' in script
