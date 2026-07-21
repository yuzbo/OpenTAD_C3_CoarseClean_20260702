from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_p0_evaluation import (
    canonical_sha256,
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)
from tools.bata.select_duca_frontend_checkpoint import sha256_file


EXPECTED_VARIANTS = (
    "two_stage_exact_uniform",
    "gaussian_matched_g0",
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
)


def _average_map(metrics: Mapping[str, Any], *, label: str) -> float:
    value = metrics.get("average_mAP")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} average_mAP is missing or non-numeric")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{label} average_mAP is non-finite")
    return value


def _validate_self_hash(
    payload: Mapping[str, Any], *, hash_key: str, label: str
) -> str:
    expected = str(payload.get(hash_key, ""))
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} self-hash mismatch")
    return expected


def _require_file(path: Any, sha256: Any, *, label: str) -> Path:
    artifact = Path(str(path)).expanduser().resolve()
    if not artifact.is_file() or sha256_file(artifact) != str(sha256):
        raise RuntimeError(f"{label} path/hash drift")
    return artifact


def _same_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        math.isclose(
            float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1.0e-12
        )
        for key in left
    )


def aggregate(
    *,
    expected_commit: str,
    decision_path: str | Path,
    decision_sha256: str,
    gate_path: str | Path,
    gate_sha256: str,
    completion_paths: Sequence[str | Path],
    completion_sha256s: Sequence[str],
    output_path: str | Path,
) -> dict:
    decision = Path(decision_path).expanduser().resolve()
    gate = Path(gate_path).expanduser().resolve()
    if not decision.is_file() or sha256_file(decision) != decision_sha256:
        raise RuntimeError("boundary-burst frontend decision drift")
    if not gate.is_file() or sha256_file(gate) != gate_sha256:
        raise RuntimeError("boundary-burst full-model gate drift")
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    if (
        decision_payload.get("schema") != "duca_boundary_burst_frontend_decision_v1"
        or decision_payload.get("ok") is not True
        or decision_payload.get("git_commit") != expected_commit
    ):
        raise RuntimeError("boundary-burst decision commit/status mismatch")
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("ok") is not True or gate_payload.get("git_commit") != expected_commit:
        raise RuntimeError("boundary-burst gate commit/status mismatch")
    if len(completion_paths) != len(completion_sha256s):
        raise RuntimeError("every completion requires an upstream SHA256 seal")

    rows = []
    for raw, expected_completion_sha256 in zip(
        completion_paths, completion_sha256s
    ):
        path = Path(raw).expanduser().resolve()
        if (
            not path.is_file()
            or len(str(expected_completion_sha256)) != 64
            or sha256_file(path) != str(expected_completion_sha256)
        ):
            raise RuntimeError(f"boundary-burst completion seal drift: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "duca_two_stage_curriculum_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("frontend_decision_sha256") != decision_sha256
            or payload.get("gate_suite_sha256") != gate_sha256
        ):
            raise RuntimeError(f"invalid boundary-burst completion: {path}")
        checkpoint = Path(payload["checkpoint_path"]).resolve()
        evaluation = Path(payload["evaluation_path"]).resolve()
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != payload["checkpoint_sha256"]
            or not evaluation.is_file()
            or sha256_file(evaluation) != payload["evaluation_sha256"]
        ):
            raise RuntimeError(f"boundary-burst completion artifact drift: {path}")
        evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
        if evaluation_payload.get("schema_version") != "duca_selected_axis_terminal_evaluation_v1":
            raise RuntimeError(f"boundary-burst evaluation schema mismatch: {evaluation}")
        evaluation_self_sha256 = _validate_self_hash(
            evaluation_payload,
            hash_key="evaluation_sha256",
            label=f"boundary-burst evaluation {payload['variant']}",
        )
        if payload.get("evaluation_self_sha256") != evaluation_self_sha256:
            raise RuntimeError(f"boundary-burst evaluation self-hash copy mismatch: {path}")
        variant = str(payload.get("variant", ""))
        if (
            variant not in EXPECTED_VARIANTS
            or evaluation_payload.get("git_commit") != expected_commit
            or evaluation_payload.get("task")
            != "offline_temporal_action_detection"
            or evaluation_payload.get("variant") != variant
            or evaluation_payload.get("seed") != 3407
        ):
            raise RuntimeError(f"boundary-burst evaluation run identity mismatch: {path}")
        if (
            Path(str(evaluation_payload.get("checkpoint_path", ""))).resolve()
            != checkpoint
            or evaluation_payload.get("checkpoint_sha256")
            != payload["checkpoint_sha256"]
            or evaluation_payload.get("checkpoint_epoch") != 59
            or evaluation_payload.get("checkpoint_state_key") != "state_dict_ema"
        ):
            raise RuntimeError(f"boundary-burst terminal checkpoint identity mismatch: {path}")

        config = _require_file(
            evaluation_payload.get("config_path"),
            evaluation_payload.get("config_sha256"),
            label=f"boundary-burst source config {variant}",
        )
        launch = _require_file(
            payload.get("launch_manifest_path"),
            payload.get("launch_manifest_sha256"),
            label=f"boundary-burst launch manifest {variant}",
        )
        launch_payload = json.loads(launch.read_text(encoding="utf-8"))
        if (
            launch_payload.get("schema") != "duca_two_stage_curriculum_launch_v1"
            or launch_payload.get("git_commit") != expected_commit
            or launch_payload.get("variant") != variant
            or launch_payload.get("seed") != 3407
            or launch_payload.get("config_sha256") != sha256_file(config)
            or launch_payload.get("gate_suite_sha256") != gate_sha256
            or launch_payload.get("terminal_checkpoint")
            != "epoch_59.pth/state_dict_ema"
            or launch_payload.get("official_training_successful_updates") != 6000
        ):
            raise RuntimeError(f"boundary-burst launch identity mismatch: {launch}")

        identity = evaluation_payload.get("training_identity")
        if not isinstance(identity, Mapping) or payload.get("training_identity") != identity:
            raise RuntimeError(f"boundary-burst training identity mismatch: {path}")
        if (
            identity.get("variant") != variant
            or identity.get("seed") != 3407
            or identity.get("successful_optimizer_updates") != 6000
            or identity.get("gate_suite_sha256") != gate_sha256
        ):
            raise RuntimeError(f"boundary-burst training contract mismatch: {path}")
        sidecar = _require_file(
            identity.get("checkpoint_sidecar_path"),
            identity.get("checkpoint_sidecar_sha256"),
            label=f"boundary-burst checkpoint sidecar {variant}",
        )
        audit = _require_file(
            identity.get("training_audit_path"),
            identity.get("training_audit_sha256"),
            label=f"boundary-burst training audit {variant}",
        )
        audit_payload = json.loads(audit.read_text(encoding="utf-8"))
        audit_self_sha256 = _validate_self_hash(
            audit_payload,
            hash_key="audit_sha256",
            label=f"boundary-burst training audit {variant}",
        )
        if (
            identity.get("training_audit_self_sha256") != audit_self_sha256
            or audit_payload.get("status") != "complete"
            or audit_payload.get("git_commit") != expected_commit
            or audit_payload.get("variant") != variant
            or audit_payload.get("seed") != 3407
            or audit_payload.get("source_config_path") != str(config)
            or audit_payload.get("source_config_sha256") != sha256_file(config)
            or audit_payload.get("resolved_config_sha256")
            != evaluation_payload.get("resolved_config_sha256")
            or audit_payload.get("gate_suite_sha256") != gate_sha256
            or audit_payload.get("full_model_gate_sha256")
            != identity.get("full_model_gate_sha256")
            or audit_payload.get("pretrain_path") != identity.get("pretrain_path")
            or audit_payload.get("pretrain_sha256") != identity.get("pretrain_sha256")
            or audit_payload.get("selector_initialization_contract")
            != identity.get("frontend_initialization")
            or audit_payload.get("evaluation_config_sha256")
            != evaluation_payload.get("evaluation_config_sha256")
            or audit_payload.get("evaluation_annotation_path")
            != evaluation_payload.get("evaluation_annotation_path")
            or audit_payload.get("evaluation_annotation_sha256")
            != evaluation_payload.get("evaluation_annotation_sha256")
            or audit_payload.get("evaluation_class_map_path")
            != evaluation_payload.get("evaluation_class_map_path")
            or audit_payload.get("evaluation_class_map_sha256")
            != evaluation_payload.get("evaluation_class_map_sha256")
        ):
            raise RuntimeError(f"boundary-burst training audit binding mismatch: {audit}")
        counters = audit_payload.get("update_audit")
        if (
            not isinstance(counters, Mapping)
            or audit_payload.get("checkpoint_criterion")
            != "terminal_epoch_59_state_dict_ema"
            or audit_payload.get("primary_checkpoint_epoch") != 59
            or audit_payload.get("primary_checkpoint_state_key")
            != "state_dict_ema"
            or audit_payload.get("expected_successful_optimizer_updates") != 6000
            or audit_payload.get("last_completed_epoch") != 59
            or audit_payload.get("epochs_completed") != 60
            or audit_payload.get("scheduler_last_epoch") != 6000
            or audit_payload.get("selector_schedule_step") != 6000
            or counters.get("successful_optimizer_updates") != 6000
            or counters.get("scheduler_updates") != 6000
            or counters.get("ema_updates") != 6000
            or counters.get("duca_schedule_updates") != 6000
            or counters.get("replay_exhaustions") != 0
        ):
            raise RuntimeError(f"boundary-burst optimizer audit mismatch: {audit}")
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_self_hash(
            sidecar_payload,
            hash_key="sidecar_sha256",
            label=f"boundary-burst checkpoint sidecar {variant}",
        )
        metadata = sidecar_payload.get("experiment_metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("schema_version") != "duca_p0_checkpoint_metadata_v2"
        ):
            raise RuntimeError(
                f"boundary-burst checkpoint metadata schema mismatch: {sidecar}"
            )
        _validate_self_hash(
            metadata,
            hash_key="metadata_sha256",
            label=f"boundary-burst checkpoint metadata {variant}",
        )
        if (
            Path(str(sidecar_payload.get("checkpoint_path", ""))).resolve()
            != checkpoint
            or sidecar_payload.get("checkpoint_sha256")
            != payload["checkpoint_sha256"]
            or metadata.get("training_audit") != audit_payload
        ):
            raise RuntimeError(f"boundary-burst checkpoint sidecar binding mismatch: {sidecar}")

        pretrain = _require_file(
            identity.get("pretrain_path"),
            identity.get("pretrain_sha256"),
            label=f"boundary-burst AdaTAD pretrain {variant}",
        )
        del pretrain
        prediction = _require_file(
            evaluation_payload.get("prediction_path"),
            evaluation_payload.get("prediction_sha256"),
            label=f"boundary-burst prediction {variant}",
        )
        if (
            payload.get("prediction_path") != str(prediction)
            or payload.get("prediction_sha256")
            != evaluation_payload.get("prediction_sha256")
            or evaluation_payload.get("evaluator") != official_evaluator_identity()
        ):
            raise RuntimeError(f"boundary-burst evaluator/prediction mismatch: {path}")
        annotation = _require_file(
            evaluation_payload.get("evaluation_annotation_path"),
            evaluation_payload.get("evaluation_annotation_sha256"),
            label=f"boundary-burst annotation {variant}",
        )
        class_map = _require_file(
            evaluation_payload.get("evaluation_class_map_path"),
            evaluation_payload.get("evaluation_class_map_sha256"),
            label=f"boundary-burst class map {variant}",
        )
        del annotation, class_map
        normalized_evaluation = normalize_evaluation_config(
            evaluation_payload.get("evaluation_config")
        )
        if evaluation_payload.get("evaluation_config_sha256") != canonical_sha256(
            normalized_evaluation
        ):
            raise RuntimeError(f"boundary-burst evaluation config mismatch: {path}")
        evaluation_metrics = evaluation_payload.get("metrics")
        if not isinstance(evaluation_metrics, Mapping):
            raise RuntimeError(f"boundary-burst evaluation metrics missing: {evaluation}")
        if payload.get("metrics") != evaluation_metrics:
            raise RuntimeError(f"boundary-burst copied completion metrics mismatch: {path}")
        recomputed = recompute_official_map(prediction, normalized_evaluation)
        recomputed_metrics = recomputed.get("metrics")
        if (
            not isinstance(recomputed_metrics, Mapping)
            or not _same_metrics(evaluation_metrics, recomputed_metrics)
            or int(evaluation_payload.get("result_count", -1))
            != int(recomputed.get("result_count", -2))
            or int(evaluation_payload.get("video_count", -1))
            != int(recomputed.get("video_count", -2))
        ):
            raise RuntimeError(f"boundary-burst official mAP recomputation mismatch: {path}")
        average_map = _average_map(
            recomputed_metrics, label=f"boundary-burst evaluation {variant}"
        )
        rows.append(
            {
                "variant": variant,
                "terminal_epoch": int(evaluation_payload["checkpoint_epoch"]),
                "checkpoint_state_key": evaluation_payload["checkpoint_state_key"],
                "average_mAP": average_map,
                "metrics": dict(recomputed_metrics),
                "evaluation_path": str(evaluation),
                "evaluation_sha256": payload["evaluation_sha256"],
                "prediction_path": str(prediction),
                "prediction_sha256": evaluation_payload["prediction_sha256"],
                "completion_path": str(path),
                "completion_sha256": expected_completion_sha256,
            }
        )
    if {row["variant"] for row in rows} != set(EXPECTED_VARIANTS):
        raise RuntimeError("result set does not cover uniform/Gaussian/R2Q3/R4Q5")
    rows.sort(key=lambda row: EXPECTED_VARIANTS.index(row["variant"]))
    payload = {
        "schema": "duca_boundary_burst_terminal_suite_v1",
        "ok": True,
        "status": "matched_four_arm_terminal_ema_results_sealed",
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "seed": 3407,
        "successful_optimizer_updates_per_arm": 6000,
        "test_subset_used_once_for_terminal_metrics": True,
        "frontend_decision_path": str(decision),
        "frontend_decision_sha256": decision_sha256,
        "gate_path": str(gate),
        "gate_sha256": gate_sha256,
        "results": rows,
        "paper_claim_allowed": False,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--completion", action="append", required=True)
    parser.add_argument("--completion-sha256", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = aggregate(
        expected_commit=args.expected_commit,
        decision_path=args.decision,
        decision_sha256=args.decision_sha256,
        gate_path=args.gate,
        gate_sha256=args.gate_sha256,
        completion_paths=args.completion,
        completion_sha256s=args.completion_sha256,
        output_path=args.output_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
