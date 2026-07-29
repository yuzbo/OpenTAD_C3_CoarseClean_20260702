from __future__ import annotations

import math

import pytest

from tools.bata.calibrate_duca_numeric_null import calibrate_numeric_null
from tools.bata.duca_acquisition_gate_schema import (
    validate_duca_acquisition_admission_v2,
)
from tools.bata.duca_evidence_io import (
    with_content_sha256,
    write_json_exclusive_atomic,
)
from tools.bata.duca_gate_diagnostics import (
    implemented_uniform_axis_geometry_report,
)
from tools.bata.freeze_duca_acquisition_scientific_protocol import (
    freeze_scientific_protocol,
)


SHA = "a" * 64
COMMIT = "b" * 40
BINDING = {"path": "/immutable/artifact", "sha256": SHA}


def _preregistration_anchor(tmp_path):
    return {
        "schema": "duca_acquisition_preregistration_anchor_v1",
        "repo_root": str(tmp_path / "repo"),
        "remote": "https://github.com/example/repo.git",
        "branch": "codex/test",
        "git_commit": COMMIT,
        "git_tree": "c" * 40,
        "candidate_output_root": str(tmp_path / "candidate"),
        "candidate_output_root_absent": True,
        "candidate_results_observed": False,
        "created_at_utc": "2026-07-29T00:00:00+00:00",
    }


def _receipt():
    windows = [
        {
            "valid_len": 32,
            "selected_positions": [0, 10, 21, 31],
        },
        {
            "valid_len": 17,
            "selected_positions": [0, 5, 11, 16],
        },
    ]
    for row in windows:
        row["implemented_map"] = implemented_uniform_axis_geometry_report(
            valid_len=row["valid_len"],
            positions=row["selected_positions"],
        )
    payload = {
        "schema": "duca_acquisition_admission_v2",
        "status": "passed",
        "admission_effect": True,
        "identity": {
            "remote": "https://github.com/example/repo.git",
            "branch": "codex/test",
            "git_commit": COMMIT,
            "git_tree": "c" * 40,
            "tracked_tree_clean": True,
            "repo_root": "/immutable/repo",
            "config_sha256": SHA,
            "checkpoint_sha256": SHA,
            "data_manifest_sha256": SHA,
            "split_assignment_sha256": SHA,
        },
        "runtime": {
            "python": "3.10",
            "torch": "2.1",
            "cuda_runtime": "11.8",
            "cudnn": "8",
            "gpu_name": "test-gpu",
            "driver": "test-driver",
            "amp_enabled": True,
            "amp_dtype": "float16",
            "deterministic_flags": {"cudnn_benchmark": False},
            "slurm_job_id": "12345",
        },
        "producer": {
            "schema": "duca_acquisition_runtime_producer_v2",
            "module": "tools.bata.run_duca_acquisition_runtime_gate_v2",
            "script": BINDING,
            "launcher": BINDING,
            "slurm_job_id": "12345",
            "git_commit": COMMIT,
            "git_tree": "c" * 40,
            "finalized_in_runtime_producer": True,
            "created_at_utc": "2026-07-29T00:00:00+00:00",
        },
        "artifact_bindings": {
            key: BINDING
            for key in (
                "code_gate_receipt",
                "selected_actionformer_config",
                "standard_actionformer_config",
                "actionformer_checkpoint",
                "selected_tridet_config",
                "standard_tridet_config",
                "tridet_checkpoint",
                "train_block_list",
                "development_block_list",
                "targets_jsonl",
                "budget_protocol",
                "data_manifest",
                "split_assignment",
                "numeric_calibration",
                "scientific_protocol",
            )
        },
        "coordinate_contract": {
            "mode": "selected_axis_plugin",
            "selector_contract": "duca_rime_selected_axis_plugin_v2",
            "detector_output_coordinate_space": "selected_axis_index",
            "inverse_map_before_official_nms": True,
            "mapping_applied_exactly_once": True,
            "physical_head_enabled": False,
            "gt_remapped_to_selected_axis": True,
            "standard_detector_head_unchanged": True,
        },
        "execution": {
            "window_roles": ["full_window", "short_window"],
            "requested_k": [4, 8],
            "effective_k": [4, 4],
            "backbone_input_k": [4, 4],
            "active_mask_count": [4, 4],
            "padded_k": [4, 4],
            "positions_sha256": SHA,
            "bucket_order_sha256": SHA,
        },
        "geometry": {
            "windows": windows,
            "roundtrip_max_abs_error": 0.0,
            "mapping_applied_exactly_once": True,
        },
        "standard_detector_restoration": {
            backend: {
                "status": "passed",
                "physical_head_enabled": False,
                "selector_disabled_null_passed": True,
                "standard_head_state_dict_compatible": True,
                "standard_config_sha256": SHA,
            }
            for backend in ("actionformer", "tridet")
        },
        "numeric": {
            "calibration_manifest_sha256": SHA,
            "calibration_content_sha256": SHA,
            "runtime_fingerprint_sha256": SHA,
            "amp_null_runs": [{"within_frozen_thresholds": True}],
            "autocast_disabled_non_admission_replay": {
                "admission_effect": False,
            },
            "state_before_sha256": SHA,
            "state_after_sha256": SHA,
        },
        "gates": {
            "structural_gate_passed": True,
            "numeric_gate_passed": True,
            "scientific_protocol_preregistered": True,
            "scientific_protocol_sha256": SHA,
            "scientific_protocol_content_sha256": SHA,
            "legacy_scalar_loss_equivalence_required": False,
        },
        "scientific_scope": {
            "uses_official_final": False,
            "paper_claim_allowed": False,
            "phase4_submission_enabled": False,
            "official_final_sealed": True,
            "primary_endpoint": "paired_video_avg_map_same_total_cost",
            "noninferiority_margin": 0.1,
            "multiplicity_procedure": "holm",
        },
        "predecessor_evidence": {
            "recovery_v6_job": "1201417",
            "historical_status": "failed_under_v1",
            "historical_outcome_reclassified": False,
        },
    }
    return with_content_sha256(payload)


def test_admission_v2_accepts_hash_bound_pure_plugin_receipt():
    receipt = _receipt()
    validated = validate_duca_acquisition_admission_v2(
        receipt,
        expected_commit=COMMIT,
    )
    assert validated["status"] == "passed"


def test_admission_v2_rejects_unsigned_or_non_runtime_produced_json():
    unsigned = _receipt()
    unsigned.pop("content_sha256")
    with pytest.raises(ValueError, match="content-bound"):
        validate_duca_acquisition_admission_v2(unsigned, expected_commit=COMMIT)

    forged = _receipt()
    forged.pop("content_sha256")
    forged["producer"]["finalized_in_runtime_producer"] = False
    forged = with_content_sha256(forged)
    with pytest.raises(ValueError, match="producer identity"):
        validate_duca_acquisition_admission_v2(forged, expected_commit=COMMIT)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda row: row["coordinate_contract"].__setitem__(
                "physical_head_enabled",
                True,
            ),
            "coordinate contract",
        ),
        (
            lambda row: row["gates"].__setitem__(
                "legacy_scalar_loss_equivalence_required",
                True,
            ),
            "legacy scalar-loss",
        ),
        (
            lambda row: row["execution"]["padded_k"].__setitem__(1, 8),
            "no-padding",
        ),
        (
            lambda row: row["scientific_scope"].__setitem__(
                "uses_official_final",
                True,
            ),
            "boundary",
        ),
    ],
)
def test_admission_v2_fails_closed_on_contract_drift(mutator, match):
    receipt = _receipt()
    receipt.pop("content_sha256")
    mutator(receipt)
    receipt = with_content_sha256(receipt)
    with pytest.raises(ValueError, match=match):
        validate_duca_acquisition_admission_v2(
            receipt,
            expected_commit=COMMIT,
        )


def test_atomic_writer_is_finite_exclusive_and_cleans_temporary_files(tmp_path):
    target = tmp_path / "receipt.json"
    write_json_exclusive_atomic(target, {"status": "passed", "value": 1.0})
    with pytest.raises(FileExistsError):
        write_json_exclusive_atomic(target, {"status": "different"})
    with pytest.raises(ValueError):
        write_json_exclusive_atomic(
            tmp_path / "nonfinite.json",
            {"value": math.nan},
        )
    assert list(tmp_path.glob(".*.tmp")) == []


def test_numeric_null_calibration_is_train_only_and_separate_from_science():
    manifest = calibrate_numeric_null(
        [
            {
                "run_id": "a",
                "split_scope": "training",
                "uses_official_final": False,
                "metric_errors": {"proposal": 1.0e-6, "score": 2.0e-7},
            },
            {
                "run_id": "b",
                "split_scope": "train_only_calibration",
                "uses_official_final": False,
                "metric_errors": {"proposal": 2.0e-6, "score": 1.0e-7},
            },
        ],
        git_commit=COMMIT,
        safety_multiplier=2.0,
        absolute_floor=1.0e-7,
    )
    assert manifest["thresholds"] == {
        "proposal": 4.0e-6,
        "score": 4.0e-7,
    }
    assert manifest["scientific_noninferiority_margin"] is None
    assert manifest["uses_official_final"] is False


def test_numeric_null_calibration_rejects_official_final():
    with pytest.raises(ValueError, match="split scope"):
        calibrate_numeric_null(
            [
                {
                    "run_id": "bad",
                    "split_scope": "official_final",
                    "uses_official_final": True,
                    "metric_errors": {"proposal": 0.0},
                }
            ],
            git_commit=COMMIT,
            safety_multiplier=2.0,
            absolute_floor=0.0,
        )


def test_scientific_protocol_freeze_requires_a_train_only_margin_source(tmp_path):
    source_path = tmp_path / "margin.json"
    source = with_content_sha256(
        {
            "schema": "duca_acquisition_ni_margin_source_v1",
            "status": "frozen",
            "git_commit": COMMIT,
            "fit_scope": "training_and_calibration",
            "uses_development_results": False,
            "uses_official_final": False,
            "candidate_performance_observed": False,
            "calibration_variability": 0.12,
            "practical_relevance_floor": 0.10,
        }
    )
    write_json_exclusive_atomic(source_path, source)
    protocol = freeze_scientific_protocol(
        expected_commit=COMMIT,
        margin_source=source_path,
        noninferiority_margin=0.12,
        primary_endpoint="paired_video_avg_map_same_total_cost",
        multiplicity_procedure="holm",
        guardrails=("map_0.7", "short_map", "boundary_error", "full_stack_cost"),
        stopping_rules=("stop_on_structural_failure", "stop_before_phase4_on_ni_failure"),
        preregistration_anchor=_preregistration_anchor(tmp_path),
    )
    assert protocol["noninferiority_margin"] == 0.12
    assert protocol["uses_official_final"] is False
    assert protocol["phase4_submission_enabled"] is False


def test_scientific_protocol_rejects_a_post_hoc_margin(tmp_path):
    source_path = tmp_path / "margin.json"
    write_json_exclusive_atomic(
        source_path,
        with_content_sha256(
            {
                "schema": "duca_acquisition_ni_margin_source_v1",
                "status": "frozen",
                "git_commit": COMMIT,
                "fit_scope": "training",
                "uses_development_results": False,
                "uses_official_final": False,
                "candidate_performance_observed": False,
                "calibration_variability": 0.12,
                "practical_relevance_floor": 0.10,
            }
        ),
    )
    with pytest.raises(ValueError, match="must equal"):
        freeze_scientific_protocol(
            expected_commit=COMMIT,
            margin_source=source_path,
            noninferiority_margin=0.05,
            primary_endpoint="paired_video_avg_map_same_total_cost",
            multiplicity_procedure="holm",
            guardrails=("map_0.7",),
            stopping_rules=("stop_on_failure",),
            preregistration_anchor=_preregistration_anchor(tmp_path),
        )
