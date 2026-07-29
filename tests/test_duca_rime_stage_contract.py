from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.bata.create_duca_rime_splits import create_rime_splits
from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    build_profile_summary,
)
from tools.bata.duca_rime_stage_contract import (
    PHASE3_ARMS,
    REQUIRED_PHASE1_CONTROLS,
    authorize_phase4,
    seal_phase1,
    seal_phase2,
    seal_phase3,
    seal_phase4,
)
from tools.bata.duca_gate_diagnostics import (
    implemented_uniform_axis_geometry_report,
)
from tools.bata.finalize_duca_rime_inference_ledger import (
    exact_uniform_positions,
)


COMMIT = "a" * 40


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _with_content_sha(payload):
    output = dict(payload)
    output["content_sha256"] = _canonical_sha(output)
    return output


def _split(tmp_path: Path):
    database = {
        f"train_{index:03d}": {"subset": "training", "annotations": []}
        for index in range(30)
    }
    database.update(
        {
            f"test_{index:03d}": {"subset": "validation", "annotations": []}
            for index in range(5)
        }
    )
    annotation = _write_json(tmp_path / "annotation.json", {"database": database})
    return create_rime_splits(annotation, tmp_path / "split")


def _admission_receipt():
    sha = "b" * 64
    binding = {"path": "/immutable/artifact", "sha256": sha}
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
    return _with_content_sha({
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
            "config_sha256": sha,
            "checkpoint_sha256": sha,
            "data_manifest_sha256": sha,
            "split_assignment_sha256": sha,
        },
        "runtime": {
            "python": "3.10",
            "torch": "2.1",
            "cuda_runtime": "11.8",
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
            "script": binding,
            "launcher": binding,
            "slurm_job_id": "12345",
            "git_commit": COMMIT,
            "git_tree": "c" * 40,
            "finalized_in_runtime_producer": True,
            "created_at_utc": "2026-07-29T00:00:00+00:00",
        },
        "artifact_bindings": {
            key: binding
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
            "positions_sha256": sha,
            "bucket_order_sha256": sha,
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
                "standard_config_sha256": sha,
            }
            for backend in ("actionformer", "tridet")
        },
        "numeric": {
            "calibration_manifest_sha256": sha,
            "calibration_content_sha256": sha,
            "runtime_fingerprint_sha256": sha,
            "amp_null_runs": [{"within_frozen_thresholds": True}],
            "autocast_disabled_non_admission_replay": {
                "admission_effect": False,
            },
            "state_before_sha256": sha,
            "state_after_sha256": sha,
        },
        "gates": {
            "structural_gate_passed": True,
            "numeric_gate_passed": True,
            "scientific_protocol_preregistered": True,
            "scientific_protocol_sha256": sha,
            "scientific_protocol_content_sha256": sha,
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
    })


def _phase1(tmp_path: Path):
    split = _split(tmp_path)
    source_path = Path(split["manifest_path"])
    phase0_metrics = _write_json(
        tmp_path / "phase0_metrics.json",
        _with_content_sha(
            {
                "schema_version": "duca_rime_localization_metrics_v1",
                "phase": 1,
                "split_assignment_sha256": split["assignment_sha256"],
                "uses_official_final": False,
            }
        ),
    )
    phase0_records = tmp_path / "phase0_records.jsonl"
    phase0_rows = [
        {
            "schema_version": "duca_rime_phase0_measurement_v1",
            "video_id": f"train_{video:03d}",
            "replicate_id": f"replicate_{replicate}",
            "replicate_kind": "deterministic_reexecution",
            "metric_name": "avg_map",
            "value": 0.5,
            "source_path": str(phase0_metrics),
            "source_sha256": _sha(phase0_metrics),
            "split_assignment_sha256": split["assignment_sha256"],
            "uses_official_final": False,
        }
        for video in range(3)
        for replicate in range(2)
    ]
    phase0_records.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in phase0_rows
        ),
        encoding="utf-8",
    )
    phase0 = _write_json(
        tmp_path / "phase0.json",
        _with_content_sha(
            {
                "schema_version": "duca_rime_causal_gate_summary_v1",
                "stage": "phase0_variance_power",
                "gate_pass": True,
                "video_count": 3,
                "replicate_count": 6,
                "replicate_kinds": ["deterministic_reexecution"],
                "split_assignment_sha256": split["assignment_sha256"],
                "source_records": {
                    "path": str(phase0_records),
                    "sha256": _sha(phase0_records),
                },
                "rule_derived_thresholds": {
                    "min_o1_headroom": 0.01,
                    "max_o2_decoder_regret": 0.01,
                    "min_o3_spearman": 0.1,
                },
            }
        ),
    )
    code_gate = tmp_path / "code.receipt"
    code_gate.write_text(
        "\n".join(
            (
                "schema=duca_rime_code_gate_v1",
                "status=passed",
                f"commit={COMMIT}",
                "slurm_job_id=123",
                "",
            )
        ),
        encoding="utf-8",
    )
    split_payload = json.loads(source_path.read_text(encoding="utf-8"))
    phase1_role = split_payload["train_roles"]["certification_development"]
    checkpoint = _write_json(tmp_path / "checkpoint.pth", {"epoch": 59})
    geometry_source = _write_json(
        tmp_path / "geometry_source.json",
        _with_content_sha(
            {
                "schema_version": "duca_rime_phase1_geometry_audit_v1",
                "gate_pass": True,
                "git_commit": COMMIT,
                "split_assignment_sha256": split["assignment_sha256"],
            }
        ),
    )
    admission_source = _write_json(
        tmp_path / "admission_source.json",
        _admission_receipt(),
    )

    def metric_sources(name: str, target_cost: int):
        terminal = _write_json(
            tmp_path / f"{name}_terminal.json",
            {
                "git_commit": COMMIT,
                "variant": name,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "runtime_gt_input_to_selector": False,
                "padded_to_kmax": False,
            },
        )
        metrics = _write_json(
            tmp_path / f"{name}_metrics.json",
            _with_content_sha(
                {
                    "schema_version": "duca_rime_localization_metrics_v1",
                    "phase": 1,
                    "git_commit": COMMIT,
                    "variant": name,
                    "target_mean_cost": float(target_cost),
                    "split_role": "certification_development",
                    "evaluation_video_ids": phase1_role["videos"],
                    "split_assignment_sha256": split["assignment_sha256"],
                    "uses_official_final": False,
                    "terminal_evaluation_path": str(terminal),
                    "terminal_evaluation_sha256": _sha(terminal),
                }
            ),
        )
        return [
            {"path": str(metrics), "sha256": _sha(metrics)},
            {"path": str(terminal), "sha256": _sha(terminal)},
            {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        ]

    def cost_profile_source(probe: bool):
        method = (
            "phase1-probe-uniform"
            if probe
            else "phase1-no-probe-uniform"
        )
        sample = {
            "input_pipeline_serial_ms": 1.0,
            "h2d_ms": 1.0,
            "model_forward_ms": 10.0,
            "postprocess_ms": 1.0,
            "frame_selector_total_ms": 1.0 if probe else 0.0,
            "backbone_wrapper_total_ms": 5.0,
            "projection_ms": 1.0,
            "neck_ms": 1.0,
            "head_ms": 1.0,
            "coarse_probe_ms": 1.0 if probe else 0.0,
            "heavy_backbone_ms": 4.0,
            "selected_count": 384.0,
        }
        profile = build_profile_summary(
            [sample for _ in range(30)],
            metadata={
                "method": method,
                "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
                "hardware_fingerprint": "gpu",
                "host_fingerprint": "host",
                "software_fingerprint": "software",
                "config_commit": COMMIT,
                "evidence_git_commit": COMMIT,
                "tracked_tree_clean": True,
                "dataset_fingerprint": "dataset",
                "inference_fingerprint": "inference",
                "detector_stack_fingerprint": "detector",
                "batch_size": 1,
                "loader_workers": 0,
                "warmup_samples": 5,
                "amp": True,
                "uses_ema": True,
                "random_init": False,
                "power_sampling_enabled": False,
                "power_interval_ms": 20,
                "power_gpu_id": None,
                "research_phase": 1,
                "uses_official_final": False,
                "accuracy_claim_allowed": False,
                "profile_session_id": "session",
                "profile_pair_id": "pair",
                "profile_repeat_index": 1,
                "profile_order_position": 2 if probe else 1,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_dropped_prefixes": ["unused"],
                "checkpoint_dropped_key_count": 1,
            },
        )
        profile_path = _write_json(
            tmp_path / f"{method}.summary.json",
            profile,
        )
        return [
            {"path": str(profile_path), "sha256": _sha(profile_path)},
            {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        ]

    controls = []
    for name in REQUIRED_PHASE1_CONTROLS:
        payload = {
            "schema_version": "duca_rime_phase1_control_v2",
            "control": name,
            "gate_pass": True,
            "git_commit": COMMIT,
            "split_assignment_sha256": split["assignment_sha256"],
            "split_role": "certification_development",
            "evaluation_video_ids": phase1_role["videos"],
            "uses_official_final": False,
            "source_artifacts": [],
        }
        if name in {"released_dense", "local_dense"}:
            payload["source_artifacts"] = metric_sources(name, 768)
            payload["measurement"] = {
                "kind": "dense_sanity_control",
                "native_heavy_rgb_frames": 768,
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_compatibility_mode": "strict_exact_v1",
                "checkpoint_sha256": _sha(checkpoint),
                "aggregate_metrics": {"avg_map": 0.5},
            }
        if name in {"uniform_k384", "uniform_k192"}:
            budget = 384 if name.endswith("384") else 192
            ledger = tmp_path / f"{name}_ledger.jsonl"
            ledger_rows = [
                {
                    "schema_version": "duca_rime_inference_ledger_v1",
                    "arm": "exact_uniform",
                    "video_id": video,
                    "window_start_frame": 0,
                    "requested_k": budget,
                    "effective_k": budget,
                    "unique_k": budget,
                    "backbone_input_k": budget,
                    "padded_k": budget,
                    "dense_valid_len": 768,
                    "selected_dense_indices": exact_uniform_positions(768, budget),
                    "observed_max_gap_seconds": 1.0,
                    "max_gap_seconds_cap": 1.0,
                    "provenance": {
                        "uses_gt": False,
                        "uses_teacher": False,
                        "uses_prediction_cache": False,
                        "uses_test_batch_composition": False,
                        "raw_predictions_stored": False,
                    },
                }
                for video in phase1_role["videos"]
            ]
            ledger.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for row in ledger_rows
                ),
                encoding="utf-8",
            )
            ledger_summary = _write_json(
                tmp_path / f"{name}_ledger_summary.json",
                {
                    "schema_version": "duca_rime_inference_ledger_summary_v1",
                    "status": "sealed",
                    "arm": "exact_uniform",
                    "path": str(ledger),
                    "sha256": _sha(ledger),
                    "record_count": len(ledger_rows),
                    "video_count": len(phase1_role["videos"]),
                    "requested_mean_k": float(budget),
                    "effective_mean_k": float(budget),
                    "requested_k_histogram": {
                        str(budget): len(ledger_rows)
                    },
                    "max_observed_gap_seconds": 1.0,
                    "max_gap_seconds_cap": 1.0,
                    "no_padding_ledger": True,
                },
            )
            payload["source_artifacts"] = [
                *metric_sources(name, budget),
                {
                    "path": str(ledger_summary),
                    "sha256": _sha(ledger_summary),
                },
                {"path": str(ledger), "sha256": _sha(ledger)},
            ]
            payload["measurement"] = {
                "kind": "exact_uniform_native_k_control",
                "native_heavy_rgb_frames": budget,
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_sha256": _sha(checkpoint),
                "aggregate_metrics": {"avg_map": 0.5},
            }
            payload["cost_ledger"] = {
                "requested_k": budget,
                "effective_k": budget,
                "unique_k": budget,
                "backbone_input_k": budget,
                "padded_k": budget,
                "record_count": len(ledger_rows),
                "video_count": len(phase1_role["videos"]),
                "max_observed_gap_seconds": 1.0,
                "constant_evidence_exact_uniform_identity": True,
            }
        if name == "acquisition_admission":
            payload["source_artifacts"] = [
                {
                    "path": str(admission_source),
                    "sha256": _sha(admission_source),
                },
                {"path": str(geometry_source), "sha256": _sha(geometry_source)},
            ]
            payload["checks"] = {
                "selected_axis_plugin": True,
                "physical_head_enabled": False,
                "standard_detector_restored": True,
                "gt_remapped_to_selected_axis": True,
                "inverse_map_before_official_nms": True,
                "mapping_applied_exactly_once": True,
                "exact_k_no_padding": True,
                "full_and_short_windows_covered": True,
                "coordinate_roundtrip_max_abs": 0.0,
                "state_neutral": True,
                "legacy_scalar_loss_equivalence_required": False,
            }
        if name == "q_to_t_before_nms":
            payload["source_artifacts"] = [
                {"path": str(geometry_source), "sha256": _sha(geometry_source)}
            ]
            payload["checks"] = {
                "remap_before_official_nms": True,
                "official_nms_call_count": 1,
                "pre_nms_remap_max_abs": 0.0,
                "coordinate_roundtrip_max_abs": 0.0,
                "roundtrip_violation_count": 0,
                "physical_head_passthrough_max_abs": 0.0,
                "physical_head_output_remapped_twice": False,
                "max_gap_violation_count": 0,
            }
        if name in {"no_probe_uniform_cost", "probe_uniform_cost"}:
            probe = name == "probe_uniform_cost"
            payload["source_artifacts"] = cost_profile_source(probe)
            payload["measurement"] = {
                "kind": "real_paired_full_stack_cost",
                "method": (
                    "phase1-probe-uniform"
                    if probe
                    else "phase1-no-probe-uniform"
                ),
                "protocol": "offline_full_window_runtime_selection",
                "profile_session_id": "session",
                "profile_pair_id": "pair",
                "profile_repeat_index": 1,
                "profile_order_position": 2 if probe else 1,
                "sample_count": 30,
                "warmup_samples": 5,
                "hardware_fingerprint": "gpu",
                "coarse_probe_executed": probe,
                "selection_policy": "exact_uniform",
                "selected_count_mean": 384.0,
                "end_to_end_p50_ms": 10.0,
                "frame_selector_p50_ms": 1.0 if probe else 0.0,
                "coarse_probe_p50_ms": 1.0 if probe else 0.0,
                "heavy_backbone_p50_ms": 5.0,
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_dropped_key_count": 1,
                "summary_rebuild_hashes": {
                    "ordered_sha256": "d" * 64,
                    "multiset_sha256": "e" * 64,
                },
            }
        controls.append(
            _write_json(
                tmp_path / f"{name}.json",
                _with_content_sha(payload),
            )
        )
    receipt = tmp_path / "phase1_receipt.json"
    seal_phase1(
        expected_commit=COMMIT,
        split_manifest=split["manifest_path"],
        split_manifest_sha256=split["manifest_sha256"],
        phase0_summary=phase0,
        code_gate_receipt=code_gate,
        controls=controls,
        output=receipt,
    )
    return split, receipt


def _phase2(tmp_path: Path):
    split, phase1 = _phase1(tmp_path)
    stages = (
        "o1_dynamic_budget_headroom",
        "o2_decoder_family_regret",
        "o3_cross_fitted_hard_utility_rank",
        "o4_pair_risk_calibration",
    )
    summaries = []
    evidence = []
    for stage in stages:
        payload = {
            "schema_version": "duca_rime_causal_gate_summary_v1",
            "stage": stage,
            "gate_pass": True,
        }
        if stage == "o2_decoder_family_regret":
            payload["selected_family"] = "independent"
        path = _write_json(tmp_path / f"{stage}.json", payload)
        summaries.append(path)
        evidence.append({"stage": stage, "path": str(path), "sha256": _sha(path)})
    protocols = [
        _write_json(
            tmp_path / f"protocol_{target}.json",
            {
                "schema_version": "duca_rime_budget_protocol_v1",
                "fit_split": "train_only",
                "uses_validation_or_test_labels": False,
                "candidate_budgets": [192, 256, 384, 512],
                "candidate_costs": [192, 256, 384, 512],
                "target_mean_cost": target,
                "realized_calibration_mean_cost": target,
                "allocation_mode": (
                    "frozen_price_dynamic_budget"
                    if target == 384
                    else "fixed_floor_budget_position_only"
                ),
                "forced_budget": None if target == 384 else 192,
                "risk_used_for_allocation": target == 384,
                "dynamic_budget_claim_allowed": target == 384,
                "decoder_family": "independent",
                "gate_pass": True,
                "evidence_summaries": evidence,
            },
        )
        for target in (384, 192)
    ]
    receipt = tmp_path / "phase2_receipt.json"
    seal_phase2(
        phase1_receipt=phase1,
        summaries=summaries,
        budget_protocols=protocols,
        output=receipt,
    )
    return split, receipt


def _phase3_rows(split, *, collapse_shuffle=False):
    split_payload = json.loads(Path(split["manifest_path"]).read_text(encoding="utf-8"))
    videos = split_payload["train_roles"]["certification_development"]["videos"]
    base = {
        "U-fixed": (0.50, 0.45, 0.35, 0.45),
        "U-same-K": (0.54, 0.48, 0.37, 0.48),
        "F-bound": (0.60, 0.53, 0.39, 0.52),
        "D-shuffle": (0.53, 0.47, 0.36, 0.47),
        "D-no-risk": (0.61, 0.50, 0.38, 0.49),
        "AdapTok-TAD": (0.57, 0.50, 0.37, 0.48),
        "RIME-full": (0.70, 0.62, 0.46, 0.61),
    }
    if collapse_shuffle:
        base["D-shuffle"] = base["RIME-full"]
    histogram = {"192": 1, "256": 1, "384": 1}
    cost_path = Path(split["manifest_path"]).parent / "phase3_cost.json"
    cost_payload = {
        "schema_version": "duca_rime_paired_full_stack_cost_v2",
        "research_phase": 3,
        "arm": "RIME-full",
        "seed": 3407,
        "detector_backend": "ActionFormer",
        "target_mean_cost": 384.0,
        "real_full_stack_measurement": True,
        "includes_probe_decoder_solver": True,
        "matched_realized_cost": True,
        "target_budget_respected": True,
        "matched_k_tolerance": 1.0,
        "candidate_effective_mean_k": 384.0,
        "matched_control_arm": "U-same-K",
        "matched_control_effective_mean_k": 384.0,
        "latency_p50_ms": 10.0,
        "matched_control_latency_p50_ms": 11.0,
        "dense_latency_p50_ms": 20.0,
    }
    _write_json(cost_path, cost_payload)
    rows = []
    for arm in PHASE3_ARMS:
        values = base[arm]
        row = {
                "schema_version": "duca_rime_phase3_arm_result_v1",
                "arm": arm,
                "seed": 3407,
                "uses_official_final": False,
                "split_assignment_sha256": split["assignment_sha256"],
                "padded_to_kmax": False,
                "evaluation_video_ids": videos,
                "initialization_sha256": "b" * 64,
                "training_exposure_sha256": "c" * 64,
                "k_histogram": (
                    histogram
                    if arm in {"U-same-K", "D-shuffle", "D-no-risk", "RIME-full"}
                    else {"384": len(videos)}
                ),
                "cost": (
                    {
                        **cost_payload,
                        "artifact_path": str(cost_path),
                        "artifact_sha256": _sha(cost_path),
                    }
                    if arm == "RIME-full"
                    else None
                ),
                "video_metrics": {
                    metric: {video: value for video in videos}
                    for metric, value in zip(
                        ("avg_map", "map_0.7", "short_map", "pair_support"),
                        values,
                    )
                },
            }
        if arm == "U-same-K":
            row.update(
                {
                    "evaluation_only": True,
                    "source_training_arm": "RIME-full",
                    "independent_training_run": False,
                    "successful_detector_updates": 0,
                    "source_successful_detector_updates": 6000,
                    "source_formal_update_audit_passed": True,
                    "source_training_receipt_sha256": f"{len(PHASE3_ARMS):064x}",
                }
            )
        else:
            row.update(
                {
                    "evaluation_only": False,
                    "successful_detector_updates": 6000,
                    "formal_update_audit_passed": True,
                    "training_receipt_sha256": f"{PHASE3_ARMS.index(arm) + 1:064x}",
                }
            )
        rows.append(row)
    return rows


def test_phase_receipts_gate_formal_submission(tmp_path):
    split, phase2 = _phase2(tmp_path)
    results = tmp_path / "phase3.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _phase3_rows(split)),
        encoding="utf-8",
    )
    phase3 = tmp_path / "phase3_receipt.json"
    sealed = seal_phase3(
        phase2_receipt=phase2,
        results_jsonl=results,
        output=phase3,
        expected_seed=3407,
        bootstrap_samples=100,
    )
    assert sealed["payload"]["gate_pass"] is True
    authorization = authorize_phase4(
        phase3_receipt=phase3,
        output=tmp_path / "phase4_authorization.json",
        formal_seeds=(5801, 8123, 12011),
    )
    assert authorization["payload"]["paper_claim_allowed"] is False
    assert authorization["payload"]["required_detectors"] == ["ActionFormer", "TriDet"]


def test_phase3_no_go_when_video_conditioning_has_no_gain(tmp_path):
    split, phase2 = _phase2(tmp_path)
    results = tmp_path / "phase3.jsonl"
    results.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in _phase3_rows(split, collapse_shuffle=True)
        ),
        encoding="utf-8",
    )
    sealed = seal_phase3(
        phase2_receipt=phase2,
        results_jsonl=results,
        output=tmp_path / "phase3_receipt.json",
        expected_seed=3407,
        bootstrap_samples=100,
    )
    assert sealed["payload"]["gate_pass"] is False
    assert sealed["payload"]["phase4_authorized"] is False
    assert sealed["payload"]["contribution_gates"]["content_conditioned_budget"] is False


def test_phase3_rejects_u_same_k_as_an_independent_training_run(tmp_path):
    split, phase2 = _phase2(tmp_path)
    rows = _phase3_rows(split)
    same_k = next(row for row in rows if row["arm"] == "U-same-K")
    same_k["independent_training_run"] = True
    same_k["successful_detector_updates"] = 6000
    results = tmp_path / "phase3.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        seal_phase3(
            phase2_receipt=phase2,
            results_jsonl=results,
            output=tmp_path / "phase3_receipt.json",
            expected_seed=3407,
            bootstrap_samples=100,
        )
    except ValueError as error:
        assert "invalid, incomplete, or contaminated" in str(error)
    else:
        raise AssertionError("U-same-K must inherit RIME-full instead of training")


def test_phase4_requires_complete_cross_detector_budget_seed_evidence(tmp_path):
    split, phase2 = _phase2(tmp_path)
    phase3_rows = tmp_path / "phase3.jsonl"
    phase3_rows.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in _phase3_rows(split)
        ),
        encoding="utf-8",
    )
    phase3 = tmp_path / "phase3_receipt.json"
    seal_phase3(
        phase2_receipt=phase2,
        results_jsonl=phase3_rows,
        output=phase3,
        expected_seed=3407,
        bootstrap_samples=100,
    )
    authorization = tmp_path / "phase4_authorization.json"
    authorize_phase4(
        phase3_receipt=phase3,
        output=authorization,
        formal_seeds=(5801, 8123, 12011),
    )
    split_payload = json.loads(
        Path(split["manifest_path"]).read_text(encoding="utf-8")
    )
    final_videos = split_payload["official_final_evaluation"]["videos"]
    interval = {"mean": 0.02, "ci95_low": 0.01, "ci95_high": 0.03}
    rows = []
    for detector in ("ActionFormer", "TriDet"):
        for budget in (384, 192):
            for seed in (5801, 8123, 12011):
                cell_name = f"{detector.lower()}_{budget}_{seed}"
                comparisons = {
                    name: {
                        "official_map_bootstrap": {
                            "official_evaluator_reexecuted_per_resample": True,
                            "paired_video_cluster_bootstrap": True,
                            "bootstrap_samples": 1000,
                        },
                        "auxiliary_video_bootstrap": {
                            "official_evaluator_reexecuted_per_resample": False,
                            "paired_video_cluster_bootstrap": True,
                            "bootstrap_samples": 1000,
                        },
                        **{
                            metric: dict(interval)
                            for metric in (
                                "avg_map",
                                "map_0.7",
                                "short_map",
                                "pair_support",
                            )
                        },
                    }
                    for name in (
                        "rime_minus_best_fixed",
                        "rime_minus_uniform_same_k",
                    )
                }
                comparison_payload = _with_content_sha(
                    {
                        "schema_version": "duca_rime_phase4_comparisons_v1",
                        "git_commit": COMMIT,
                        "detector_backend": detector,
                        "target_mean_cost": budget,
                        "seed": seed,
                        "evaluation_video_ids": final_videos,
                        "comparisons": comparisons,
                        "official_final_used_for_training_or_selection": False,
                    }
                )
                comparison_path = _write_json(
                    tmp_path / f"{cell_name}_comparisons.json",
                    comparison_payload,
                )
                expected_arm = (
                    "RIME-full-TriDet" if detector == "TriDet" else "RIME-full"
                )
                cost = _with_content_sha(
                    {
                        "schema_version": "duca_rime_paired_full_stack_cost_v2",
                        "research_phase": 4,
                        "arm": expected_arm,
                        "seed": seed,
                        "detector_backend": detector,
                        "target_mean_cost": budget,
                        "real_full_stack_measurement": True,
                        "matched_realized_cost": True,
                        "target_budget_respected": True,
                        "includes_probe_decoder_solver": True,
                        "matched_k_tolerance": 1.0,
                        "candidate_effective_mean_k": float(budget),
                        "matched_control_arm": (
                            "U-same-K-TriDet"
                            if detector == "TriDet"
                            else "U-same-K"
                        ),
                        "matched_control_effective_mean_k": float(budget),
                        "latency_p50_ms": 10.0,
                        "latency_p95_ms": 12.0,
                        "throughput_videos_per_second": 5.0,
                        "energy_joules_per_video": 4.0,
                        "peak_gpu_memory_mb": 1000.0,
                        "matched_control_latency_p50_ms": 11.0,
                        "dense_latency_p50_ms": 20.0,
                        "dense_latency_p95_ms": 22.0,
                        "candidate_below_dense": True,
                        "official_final_labels_used_for_cost_decision": False,
                    }
                )
                cost_path = _write_json(
                    tmp_path / f"{cell_name}_cost.json",
                    cost,
                )
                suffix = "-TriDet" if detector == "TriDet" else ""
                metric_artifacts = {}
                for name, variant in {
                    "rime_metrics": f"RIME-full{suffix}",
                    "fixed_metrics": f"U-fixed{suffix}",
                    "same_k_metrics": f"U-same-K{suffix}",
                }.items():
                    source_arm = (
                        f"RIME-full{suffix}"
                        if name == "same_k_metrics"
                        else variant
                    )
                    terminal = _write_json(
                        tmp_path / f"{cell_name}_{name}_terminal.json",
                        {
                            "schema_version": "duca_rime_terminal_evaluation_v1",
                            "git_commit": COMMIT,
                            "variant": variant,
                            "detector_backend": detector,
                            "target_mean_cost": float(budget),
                            "seed": seed,
                            "padded_to_kmax": False,
                            "metrics": {
                                "average_mAP": 0.6,
                                "mAP@0.6": 0.55,
                                "mAP@0.7": 0.5,
                            },
                            "training_identity": {
                                "evaluation_arm": variant,
                                "source_arm": source_arm,
                                "research_phase": 4,
                                "detector_backend": detector,
                                "target_mean_cost": float(budget),
                                "phase4_authorization_sha256": _sha(authorization),
                                "successful_detector_updates": 6000,
                                "official_final_subset_consumed_during_training": False,
                            },
                        },
                    )
                    video_metrics = {
                        metric: {video: value for video in final_videos}
                        for metric, value in {
                            "short_map": 0.5,
                            "medium_map": 0.6,
                            "long_map": 0.7,
                            "boundary_error": 0.1,
                            "pair_support": 0.7,
                        }.items()
                    }
                    metrics_payload = _with_content_sha(
                        {
                            "schema_version": "duca_rime_localization_metrics_v1",
                            "phase": 4,
                            "git_commit": COMMIT,
                            "variant": variant,
                            "detector_backend": detector,
                            "target_mean_cost": float(budget),
                            "seed": seed,
                            "padded_to_kmax": False,
                            "split_role": "official_final_evaluation",
                            "split_manifest_sha256": split["manifest_sha256"],
                            "split_assignment_sha256": split["assignment_sha256"],
                            "annotation_sha256": "d" * 64,
                            "duration_thresholds_seconds": {
                                "short_max": 4.0,
                                "medium_max": 10.0,
                            },
                            "evaluation_video_ids": final_videos,
                            "terminal_evaluation_path": str(terminal),
                            "terminal_evaluation_sha256": _sha(terminal),
                            "video_metrics": video_metrics,
                            "uses_official_final": True,
                            "official_final_used_for_training_or_selection": False,
                        }
                    )
                    path = _write_json(
                        tmp_path / f"{cell_name}_{name}.json",
                        metrics_payload,
                    )
                    metric_artifacts[name] = {
                        "path": str(path),
                        "sha256": _sha(path),
                    }
                requested_values = (
                    [256, 384, 512, 384, 384]
                    if budget == 384
                    else [192] * len(final_videos)
                )
                ledger_rows = [
                    {
                        "schema_version": "duca_rime_inference_ledger_v1",
                        "arm": "rime_full",
                        "video_id": video,
                        "window_start_frame": 0,
                        "requested_k": requested,
                        "effective_k": requested,
                        "backbone_input_k": requested,
                        "padded_k": requested,
                        "allocation_mode": (
                            "frozen_price_dynamic_budget"
                            if budget == 384
                            else "fixed_floor_budget_position_only"
                        ),
                        "observed_max_gap_seconds": 1.0,
                        "max_gap_seconds_cap": 1.0,
                        "provenance": {
                            "uses_gt": False,
                            "uses_teacher": False,
                            "uses_prediction_cache": False,
                        },
                    }
                    for video, requested in zip(final_videos, requested_values)
                ]
                ledger_data = tmp_path / f"{cell_name}_rime_ledger.jsonl"
                ledger_data.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in ledger_rows
                    ),
                    encoding="utf-8",
                )
                ledger_summary = _write_json(
                    tmp_path / f"{cell_name}_rime_ledger_summary.json",
                    {
                        "schema_version": "duca_rime_inference_ledger_summary_v1",
                        "status": "sealed",
                        "arm": "rime_full",
                        "path": str(ledger_data),
                        "sha256": _sha(ledger_data),
                        "record_count": len(ledger_rows),
                        "requested_mean_k": (
                            sum(requested_values) / len(requested_values)
                        ),
                        "requested_k_histogram": {
                            str(value): requested_values.count(value)
                            for value in sorted(set(requested_values))
                        },
                        "max_observed_gap_seconds": 1.0,
                        "no_padding_ledger": True,
                        "all_observed_gaps_within_cap": True,
                        "official_final_labels_used_for_decision": False,
                    },
                )
                metric_artifacts["rime_ledger_summary"] = {
                    "path": str(ledger_summary),
                    "sha256": _sha(ledger_summary),
                }
                rows.append(
                    {
                        "schema_version": "duca_rime_phase4_result_v1",
                        "git_commit": COMMIT,
                        "detector_backend": detector,
                        "target_mean_cost": budget,
                        "seed": seed,
                        "method_frozen_before_final_evaluation": True,
                        "development_seed_excluded": True,
                        "uses_official_final": True,
                        "official_final_used_for_training_or_selection": False,
                        "rime_successful_detector_updates": 6000,
                        "fixed_successful_detector_updates": 6000,
                        "same_k_successful_detector_updates": 0,
                        "same_k_source_training_arm": "RIME-full",
                        "padded_to_kmax": False,
                        "budget_panel_semantics": (
                            "content_conditioned_dynamic_budget_panel"
                            if budget == 384
                            else "exact_k192_learned_position_stress_panel"
                        ),
                        "dynamic_budget_claim_allowed": budget == 384,
                        "evaluation_video_ids": final_videos,
                            "metrics": {
                            "avg_map": 0.6,
                            "map_0.6": 0.55,
                            "map_0.7": 0.5,
                            "short_map": 0.5,
                            "medium_map": 0.6,
                            "long_map": 0.7,
                            "boundary_error": 0.1,
                            "pair_support": 0.7,
                                "max_gap_seconds": 1.0,
                            },
                            "k_distribution": {
                                str(value): requested_values.count(value)
                                for value in sorted(set(requested_values))
                            },
                        "comparisons": comparisons,
                        "cost": cost,
                        "artifacts": {
                            "authorization": {
                                "path": str(authorization),
                                "sha256": _sha(authorization),
                            },
                            "comparisons": {
                                "path": str(comparison_path),
                                "sha256": _sha(comparison_path),
                            },
                            "cost": {
                                "path": str(cost_path),
                                "sha256": _sha(cost_path),
                            },
                                **metric_artifacts,
                        },
                    }
                )
    results = tmp_path / "phase4.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    sealed = seal_phase4(
        authorization_receipt=authorization,
        results_jsonl=results,
        output=tmp_path / "phase4_receipt.json",
    )
    assert sealed["payload"]["gate_pass"] is True
    assert sealed["payload"]["paper_claim_allowed"] is True
    assert sealed["payload"]["cell_count"] == 12

    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0] = {
        **tampered_rows[0],
        "comparisons": {
            **tampered_rows[0]["comparisons"],
            "rime_minus_best_fixed": {
                **tampered_rows[0]["comparisons"]["rime_minus_best_fixed"],
                "avg_map": {
                    "mean": 0.20,
                    "ci95_low": 0.10,
                    "ci95_high": 0.30,
                },
            },
        },
    }
    tampered_results = tmp_path / "phase4_tampered.jsonl"
    tampered_results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tampered_rows),
        encoding="utf-8",
    )
    try:
        seal_phase4(
            authorization_receipt=authorization,
            results_jsonl=tampered_results,
            output=tmp_path / "phase4_tampered_receipt.json",
        )
    except ValueError as error:
        assert "hash-bound artifact" in str(error)
    else:
        raise AssertionError("Phase-4 must reject hand-edited comparison metadata")
