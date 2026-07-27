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
)


COMMIT = "a" * 40


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    protocol = _write_json(
        tmp_path / "protocol.json",
        {
            "schema_version": "duca_rime_budget_protocol_v1",
            "fit_split": "train_only",
            "uses_validation_or_test_labels": False,
            "candidate_budgets": [192, 256, 384, 512],
            "candidate_costs": [192, 256, 384, 512],
            "target_mean_cost": 384,
            "decoder_family": "independent",
            "gate_pass": True,
            "evidence_summaries": evidence,
        },
    )
    receipt = tmp_path / "phase2_receipt.json"
    seal_phase2(
        phase1_receipt=phase1,
        summaries=summaries,
        budget_protocol=protocol,
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
    rows = []
    for arm in PHASE3_ARMS:
        values = base[arm]
        rows.append(
            {
                "schema_version": "duca_rime_phase3_arm_result_v1",
                "arm": arm,
                "seed": 3407,
                "successful_detector_updates": 6000,
                "formal_update_audit_passed": True,
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
                "realized_total_cost": {video: 384.0 for video in videos},
                "dense_reference_mean_cost": 768.0,
                "video_metrics": {
                    metric: {video: value for video in videos}
                    for metric, value in zip(
                        ("avg_map", "map_0.7", "short_map", "pair_support"),
                        values,
                    )
                },
            }
        )
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
