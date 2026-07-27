from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.bata.create_duca_rime_splits import create_rime_splits
from tools.bata.duca_rime_stage_contract import (
    PHASE3_ARMS,
    REQUIRED_PHASE1_CONTROLS,
    authorize_phase4,
    seal_phase1,
    seal_phase2,
    seal_phase3,
    seal_phase4,
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


def _phase1(tmp_path: Path):
    split = _split(tmp_path)
    phase0 = _write_json(
        tmp_path / "phase0.json",
        {
            "schema_version": "duca_rime_causal_gate_summary_v1",
            "stage": "phase0_variance_power",
            "gate_pass": True,
            "rule_derived_thresholds": {
                "min_o1_headroom": 0.01,
                "max_o2_decoder_regret": 0.01,
                "min_o3_spearman": 0.1,
            },
        },
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
    controls = []
    for name in REQUIRED_PHASE1_CONTROLS:
        payload = {
            "schema_version": "duca_rime_phase1_control_v1",
            "control": name,
            "gate_pass": True,
            "git_commit": COMMIT,
            "split_assignment_sha256": split["assignment_sha256"],
            "uses_official_final": False,
        }
        if name in {"uniform_k384", "uniform_k192"}:
            budget = 384 if name.endswith("384") else 192
            payload["cost_ledger"] = {
                "requested_k": budget,
                "effective_k": budget,
                "unique_k": budget,
                "backbone_input_k": budget,
                "padded_k": budget,
                "constant_evidence_exact_uniform_identity": True,
            }
        if name == "wrapper_parity":
            payload["checks"] = {
                "mask_equal": True,
                "tensor_max_abs": 0.0,
                "raw_proposal_max_abs": 0.0,
                "coordinate_roundtrip_max_abs": 0.0,
                "map_abs_delta": 0.0,
            }
        if name == "q_to_t_before_nms":
            payload["checks"] = {
                "remap_before_official_nms": True,
                "roundtrip_violation_count": 0,
                "max_gap_violation_count": 0,
            }
        controls.append(_write_json(tmp_path / f"{name}.json", payload))
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
                "candidate_budgets": [128, 192, 256, 384, 512],
                "candidate_costs": [128, 192, 256, 384, 512],
                "target_mean_cost": target,
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
        "schema_version": "duca_rime_paired_full_stack_cost_v1",
        "research_phase": 3,
        "arm": "RIME-full",
        "seed": 3407,
        "detector_backend": "ActionFormer",
        "target_mean_cost": 384.0,
        "real_full_stack_measurement": True,
        "includes_probe_decoder_solver": True,
        "matched_realized_cost": True,
        "matched_k_tolerance": 1.0,
        "latency_p50_ms": 10.0,
        "fixed_latency_p50_ms": 11.0,
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
                        "schema_version": "duca_rime_paired_full_stack_cost_v1",
                        "research_phase": 4,
                        "arm": expected_arm,
                        "seed": seed,
                        "detector_backend": detector,
                        "target_mean_cost": budget,
                        "real_full_stack_measurement": True,
                        "matched_realized_cost": True,
                        "includes_probe_decoder_solver": True,
                        "matched_k_tolerance": 1.0,
                        "candidate_effective_mean_k": float(budget),
                        "fixed_effective_mean_k": float(budget),
                        "latency_p50_ms": 10.0,
                        "latency_p95_ms": 12.0,
                        "throughput_videos_per_second": 5.0,
                        "energy_joules_per_video": 4.0,
                        "peak_gpu_memory_mb": 1000.0,
                        "fixed_latency_p50_ms": 11.0,
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
                dummy_artifacts = {}
                for name in (
                    "rime_metrics",
                    "fixed_metrics",
                    "same_k_metrics",
                    "rime_ledger_summary",
                ):
                    path = _write_json(
                        tmp_path / f"{cell_name}_{name}.json",
                        {"cell": cell_name, "artifact": name},
                    )
                    dummy_artifacts[name] = {"path": str(path), "sha256": _sha(path)}
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
                            **dummy_artifacts,
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
