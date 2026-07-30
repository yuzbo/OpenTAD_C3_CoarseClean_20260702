import hashlib
import json
from pathlib import Path

import pytest

from tools.bata import validate_actionformer_s0_screening as screening


METRICS_DENSE = {
    "average_mAP": 0.6658301251307708,
    "mAP@0.3": 0.8190849486121916,
    "mAP@0.4": 0.7795203466370499,
    "mAP@0.5": 0.7128549836803181,
    "mAP@0.6": 0.5825550463357125,
    "mAP@0.7": 0.43513530038858167,
}
METRICS_SPARSE = {
    "average_mAP": 0.4391969933812866,
    "mAP@0.3": 0.6492524848028804,
    "mAP@0.4": 0.5664284467904844,
    "mAP@0.5": 0.4595264064199946,
    "mAP@0.6": 0.32783176623331417,
    "mAP@0.7": 0.19294586265975905,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _build_pair(tmp_path):
    shared = {}
    for name in (
        "source_diff",
        "data_manifest",
        "feature_manifest",
        "evaluator_manifest",
        "environment",
        "nms_extension",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        shared[name] = path

    arm_records = {}
    for arm, metrics in (("dense", METRICS_DENSE), ("sparse", METRICS_SPARSE)):
        arm_root = tmp_path / arm
        arm_root.mkdir()
        artifact_paths = {}
        for name in (
            "config",
            "checkpoint",
            "raw_predictions",
            "train_log",
            "saveonly_log",
            "eval_log",
        ):
            path = arm_root / name
            path.write_bytes(f"{arm}-{name}".encode())
            artifact_paths[name] = path
        attestation = _write_json(
            arm_root / "independent.json",
            {
                "schema_version": screening.ATTESTATION_SCHEMA,
                "validation_pass": True,
                "issues": [],
                "paper_main_table_eligible": False,
                "raw_predictions": {
                    "complete_official_test_coverage": True,
                    "evaluated_video_count": 212,
                    "prediction_video_count": 212,
                    "prediction_count": 42400,
                    "sha256": _sha(artifact_paths["raw_predictions"]),
                },
                "recomputed_metrics": metrics,
            },
        )
        receipts = {
            key: _receipt(value) for key, value in artifact_paths.items()
        }
        receipts["independent_metric_attestation"] = _receipt(attestation)
        intervention = None
        if arm == "sparse":
            intervention = {
                "name": "native_grid_sparse_head_k384",
                "budget": 384,
                "policy": "stratified_uniform",
                "training_loss_support": "selected_native_grid_queries",
                "interpretation": "method_intervention_not_execution_only",
            }
        arm_complete = _write_json(
            arm_root / "ARM_COMPLETE.json",
            {
                "schema_version": screening.ARM_SCHEMA,
                "validation_pass": True,
                "arm": arm,
                "seed": 1234567891,
                "new_training": True,
                "paper_main_table_eligible": False,
                "end_to_end_cost_claim_allowed": False,
                "schedule": {
                    "optimizer_epochs": 30,
                    "warmup_epochs": 5,
                    "executed_epochs": 35,
                    "terminal_checkpoint_epoch": 35,
                    "evaluated_weights": "state_dict_ema",
                    "resume": False,
                },
                "method_intervention": intervention,
                "metrics": metrics,
                "receipts": receipts,
            },
        )
        arm_records[arm] = _receipt(arm_complete)

    pair_path = _write_json(
        tmp_path / "MATCHED_PAIR_COMPLETE.json",
        {
            "schema_version": screening.PAIR_SCHEMA,
            "validation_pass": True,
            "issues": [],
            "experiment_role": "matched_single_seed_screening",
            "paper_main_table_eligible": False,
            "model_route_terminal_claim_allowed": False,
            "end_to_end_cost_claim_allowed": False,
            "comparability": {
                "same_candidate_commit": True,
                "same_seed": True,
                "same_official_data": True,
                "same_official_schedule": True,
                "same_terminal_epoch_35_ema_rule": True,
                "same_official_evaluator": True,
                "sparse_is_method_intervention_not_execution_only": True,
            },
            "source": {
                "source_diff_attestation": _receipt(shared["source_diff"]),
            },
            "data": {
                "data_manifest": _receipt(shared["data_manifest"]),
                "feature_manifest": _receipt(shared["feature_manifest"]),
                "evaluator_manifest": _receipt(shared["evaluator_manifest"]),
            },
            "environment": {
                "official_runtime_environment": _receipt(shared["environment"]),
                "official_nms_extension": _receipt(shared["nms_extension"]),
                "same_environment_for_both_arms": True,
                "official_softnms_7arg_probe": True,
            },
            "arms": arm_records,
            "metrics": {
                "dense": METRICS_DENSE,
                "sparse": METRICS_SPARSE,
                "sparse_minus_dense_average_mAP": (
                    METRICS_SPARSE["average_mAP"]
                    - METRICS_DENSE["average_mAP"]
                ),
            },
        },
    )
    return pair_path


def test_s0_screening_freezes_the_legal_negative_verdict(tmp_path):
    pair = _build_pair(tmp_path)
    verdict = screening.build_verdict(pair, _sha(pair))
    assert verdict["validation_pass"] is True
    assert verdict["legal_model_result"] is True
    assert verdict["engineering_failure"] is False
    assert verdict["screening_pass"] is False
    assert verdict["verdict"] == "KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION"
    assert verdict["metrics"]["sparse_minus_dense_pp"]["average_mAP"] == pytest.approx(
        -22.663313174948418
    )
    assert verdict["metrics"]["sparse_minus_dense_pp"]["mAP@0.6"] == pytest.approx(
        -25.472328010239835
    )
    assert verdict["metrics"]["sparse_minus_dense_pp"]["mAP@0.7"] == pytest.approx(
        -24.218943772882263
    )
    assert verdict["continuation"]["five_seed_main_study_authorized"] is False
    assert verdict["continuation"]["no_retraining_2x2_attribution_authorized"] is True
    assert verdict["paper_main_table_eligible"] is False


def test_s0_screening_rejects_a_tampered_arm_artifact(tmp_path):
    pair = _build_pair(tmp_path)
    payload = json.loads(pair.read_text())
    dense_arm = Path(payload["arms"]["dense"]["path"])
    dense_arm.write_text(dense_arm.read_text() + "\n")
    with pytest.raises(screening.ProtocolError, match="dense.ARM_COMPLETE"):
        screening.build_verdict(pair, _sha(pair))
