from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata.duca_p0_evaluation import (
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)
from tools.bata.duca_protected_physical_training import (
    CHECKPOINT_METADATA_SCHEMA,
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    TRAINING_AUDIT_SCHEMA,
    VARIANTS,
    canonical_sha256,
    sha256_file,
)


EVIDENCE_SCHEMA = "duca_protected_physical_post_run_evidence_v1"
EVALUATION_SCHEMA = "duca_protected_physical_terminal_evaluation_v1"
TERMINAL_EPOCH = 59
END_EPOCH = 60
TERMINAL_STATE_KEY = "state_dict_ema"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(
    payload: Mapping[str, Any],
    hash_key: str,
    label: str,
) -> None:
    expected = payload.get(hash_key)
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    _require(
        isinstance(expected, str) and expected == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _validate_metrics(metrics: Any) -> dict[str, float]:
    _require(isinstance(metrics, Mapping), "terminal metrics are missing")
    keys = [
        "average_mAP",
        *(f"mAP@{threshold}" for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)),
    ]
    output = {}
    for key in keys:
        value = metrics.get(key)
        _require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"terminal metric {key} is invalid",
        )
        output[key] = float(value)
    return output


def _inspect_checkpoint(
    checkpoint_path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    expected_updates: int,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "terminal checkpoint is not a mapping")
    _require(
        int(checkpoint.get("epoch", -1)) == TERMINAL_EPOCH,
        "terminal checkpoint epoch mismatch",
    )
    for key in (
        "state_dict",
        "state_dict_ema",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "rng_state",
        "experiment_metadata",
    ):
        _require(key in checkpoint, f"terminal checkpoint is missing {key}")
    _require(
        checkpoint["experiment_metadata"] == expected_metadata,
        "checkpoint metadata differs from its sidecar",
    )
    _require(
        int(checkpoint["scheduler"].get("last_epoch", -1)) == expected_updates,
        "terminal scheduler update count mismatch",
    )
    rng_state = checkpoint["rng_state"]
    _require(
        isinstance(rng_state, Mapping)
        and set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "terminal RNG state is incomplete",
    )
    del checkpoint
    return {
        "payload_reopened": True,
        "epoch": TERMINAL_EPOCH,
        "scheduler_last_epoch": expected_updates,
        "state_dict_ema_present": True,
        "grad_scaler_present": True,
        "global_rng_state_present": True,
        "selector_schedule_step": 0,
        "embedded_metadata_exact": True,
    }


def finalize_run(
    *,
    variant: str,
    protocol_manifest_path: str | Path,
    protocol_manifest_sha256: str,
    authorization_path: str | Path,
    authorization_sha256: str,
    training_audit_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sidecar_path: str | Path,
    evaluation_path: str | Path,
) -> dict[str, Any]:
    _require(variant in VARIANTS, f"unsupported protected variant: {variant}")
    protocol_file, protocol = _load_json(protocol_manifest_path, "P0 manifest")
    authorization_file, authorization = _load_json(
        authorization_path,
        "P0-P3 authorization",
    )
    audit_file, audit = _load_json(training_audit_path, "training audit")
    sidecar_file, sidecar = _load_json(
        checkpoint_sidecar_path,
        "checkpoint sidecar",
    )
    evaluation_file, evaluation = _load_json(
        evaluation_path,
        "terminal evaluation",
    )
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    _require(checkpoint_file.is_file(), "terminal checkpoint is missing")

    _require(
        sha256_file(protocol_file) == protocol_manifest_sha256,
        "P0 manifest hash drift",
    )
    _require(
        protocol.get("schema") == "duca_protected_physical_protocol_manifest_v1"
        and protocol.get("ok") is True,
        "P0 manifest did not pass",
    )
    _require(
        sha256_file(authorization_file) == authorization_sha256,
        "authorization hash drift",
    )
    _require(
        authorization.get("schema")
        == "duca_protected_physical_authorization_v1"
        and authorization.get("ok") is True,
        "P0-P3 authorization did not pass",
    )
    _require(
        authorization.get("authorized_scope", {}).get(
            "official60_four_arm_training"
        )
        is True
        and authorization.get("paper_claim_allowed") is False,
        "authorization does not unlock official-60 training",
    )
    _require(
        authorization.get("protocol_manifest_sha256")
        == protocol_manifest_sha256,
        "authorization is bound to another P0 manifest",
    )

    _require(audit.get("schema") == TRAINING_AUDIT_SCHEMA, "audit schema mismatch")
    _validate_embedded_hash(audit, "audit_sha256", "training audit")
    _require(audit.get("status") == "complete", "training audit is incomplete")
    _require(audit.get("variant") == variant, "training variant mismatch")
    _require(int(audit.get("seed", -1)) == 3407, "training seed mismatch")
    _require(
        audit.get("git_commit")
        == protocol.get("git_commit")
        == authorization.get("git_commit"),
        "commit binding mismatch",
    )
    _require(
        audit.get("protocol_manifest_sha256") == protocol_manifest_sha256,
        "training used another P0 manifest",
    )
    _require(
        audit.get("authorization_sha256") == authorization_sha256,
        "training used another authorization",
    )
    _require(
        int(audit.get("last_completed_epoch", -1)) == TERMINAL_EPOCH
        and int(audit.get("epochs_completed", -1)) == END_EPOCH,
        "training did not reach the frozen terminal epoch",
    )
    records = audit.get("epoch_records")
    _require(
        isinstance(records, list)
        and [int(record.get("epoch", -1)) for record in records]
        == list(range(END_EPOCH)),
        "training epoch records are incomplete",
    )
    expected_updates = int(audit.get("expected_successful_optimizer_updates", -1))
    loader_contract = audit.get("train_loader_contract")
    _require(
        isinstance(loader_contract, Mapping)
        and expected_updates == int(loader_contract.get("loader_length", -1)) * 60,
        "derived optimizer exposure mismatch",
    )
    counters = audit.get("update_audit")
    _require(isinstance(counters, Mapping), "training update audit is missing")
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
    ):
        _require(
            int(counters.get(key, -1)) == expected_updates,
            f"training counter {key} mismatch",
        )
    _require(
        int(counters.get("duca_schedule_updates", -1)) == 0
        and int(audit.get("selector_schedule_step", -1)) == 0,
        "schedule-free selector advanced a hidden schedule",
    )
    _require(
        int(counters.get("replay_exhaustions", -1)) == 0
        and int(counters.get("forced_amp_overflow_attempts", -1)) == 0,
        "formal training violated the AMP contract",
    )
    _require(
        int(counters.get("optimizer_attempts", -1))
        == expected_updates + int(counters.get("amp_skipped_attempts", -1)),
        "optimizer attempt accounting mismatch",
    )
    _require(
        int(audit.get("scheduler_last_epoch", -1)) == expected_updates,
        "scheduler state mismatch",
    )

    _require(
        sidecar.get("schema_version") == DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint sidecar schema mismatch",
    )
    _validate_embedded_hash(sidecar, "sidecar_sha256", "checkpoint sidecar")
    checkpoint_sha256 = sha256_file(checkpoint_file)
    _require(
        sidecar.get("checkpoint_sha256") == checkpoint_sha256,
        "checkpoint hash differs from sidecar",
    )
    _require(
        Path(str(sidecar.get("checkpoint_path", ""))).resolve() == checkpoint_file,
        "checkpoint path differs from sidecar",
    )
    metadata = sidecar.get("experiment_metadata")
    _require(isinstance(metadata, Mapping), "checkpoint metadata is missing")
    _require(
        metadata.get("schema") == CHECKPOINT_METADATA_SCHEMA,
        "checkpoint metadata schema mismatch",
    )
    _validate_embedded_hash(metadata, "metadata_sha256", "checkpoint metadata")
    _require(
        metadata.get("training_audit") == audit,
        "checkpoint does not embed the terminal audit",
    )
    checkpoint_contract = _inspect_checkpoint(
        checkpoint_file,
        expected_metadata=metadata,
        expected_updates=expected_updates,
    )

    _require(
        evaluation.get("schema_version") == EVALUATION_SCHEMA,
        "terminal evaluation schema mismatch",
    )
    _validate_embedded_hash(evaluation, "evaluation_sha256", "terminal evaluation")
    _require(
        evaluation.get("git_commit") == audit.get("git_commit"),
        "evaluation commit mismatch",
    )
    _require(
        evaluation.get("config_sha256") == audit.get("source_config_sha256"),
        "evaluation source config differs from training",
    )
    _require(
        evaluation.get("resolved_config_sha256")
        == audit.get("resolved_config_sha256"),
        "evaluation resolved config differs from training",
    )
    _require(
        Path(str(evaluation.get("checkpoint_path", ""))).resolve()
        == checkpoint_file
        and evaluation.get("checkpoint_sha256") == checkpoint_sha256,
        "evaluation used another checkpoint",
    )
    _require(
        int(evaluation.get("checkpoint_epoch", -1)) == TERMINAL_EPOCH
        and evaluation.get("checkpoint_state_key") == TERMINAL_STATE_KEY,
        "terminal evaluation did not use epoch-59 EMA",
    )
    prediction_file = Path(
        str(evaluation.get("prediction_path", ""))
    ).expanduser().resolve()
    _require(
        prediction_file.is_file()
        and evaluation.get("prediction_sha256") == sha256_file(prediction_file),
        "terminal predictions are missing or changed",
    )
    _require(
        evaluation.get("evaluator") == official_evaluator_identity(),
        "terminal evaluator is not the official OpenTAD mAP evaluator",
    )
    evaluation_config = normalize_evaluation_config(
        evaluation.get("evaluation_config")
    )
    evaluation_config_hash = canonical_sha256(evaluation_config)
    _require(
        evaluation.get("evaluation_config_sha256")
        == evaluation_config_hash
        == audit.get("evaluation_config_sha256"),
        "evaluation config binding mismatch",
    )
    _require(
        evaluation.get("evaluation_annotation_sha256")
        == audit.get("evaluation_annotation_sha256")
        == protocol.get("data_files", {}).get("annotation_sha256"),
        "evaluation annotation binding mismatch",
    )
    _require(
        evaluation.get("evaluation_class_map_sha256")
        == audit.get("evaluation_class_map_sha256")
        == protocol.get("data_files", {}).get("class_map_sha256"),
        "evaluation class-map binding mismatch",
    )
    reported_metrics = _validate_metrics(evaluation.get("metrics"))
    recomputed = recompute_official_map(prediction_file, evaluation_config)
    metrics = _validate_metrics(recomputed["metrics"])
    for key, value in metrics.items():
        _require(
            math.isclose(
                reported_metrics[key],
                value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"reported {key} differs from official recomputation",
        )
    _require(
        int(evaluation.get("result_count", -1)) == recomputed["result_count"]
        and int(evaluation.get("video_count", -1))
        == recomputed["video_count"],
        "terminal result population mismatch",
    )

    payload = {
        "schema": EVIDENCE_SCHEMA,
        "ok": True,
        "paper_claim_allowed": False,
        "task": "offline_temporal_action_detection",
        "variant": variant,
        "git_commit": audit["git_commit"],
        "seed": 3407,
        "protocol_manifest_path": str(protocol_file),
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "authorization_path": str(authorization_file),
        "authorization_sha256": authorization_sha256,
        "successful_optimizer_updates": expected_updates,
        "checkpoint_epoch": TERMINAL_EPOCH,
        "checkpoint_state_key": TERMINAL_STATE_KEY,
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_path": str(sidecar_file),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_file),
        "checkpoint_payload_contract": checkpoint_contract,
        "training_audit_path": str(audit_file),
        "training_audit_sha256": sha256_file(audit_file),
        "terminal_evaluation_path": str(evaluation_file),
        "terminal_evaluation_sha256": sha256_file(evaluation_file),
        "prediction_path": str(prediction_file),
        "prediction_sha256": evaluation["prediction_sha256"],
        "evaluation_config_sha256": evaluation_config_hash,
        "evaluator": official_evaluator_identity(),
        "metric_recomputation": {
            "performed": True,
            "result_count": recomputed["result_count"],
            "video_count": recomputed["video_count"],
        },
        "metrics": metrics,
        "non_finite_collapse": False,
    }
    payload["artifact_chain_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--protocol-manifest-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sidecar", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).expanduser().resolve()
    if output.exists():
        print(
            json.dumps(
                {
                    "schema": EVIDENCE_SCHEMA,
                    "ok": False,
                    "error_type": "FileExistsError",
                    "error": "refusing to overwrite post-run evidence",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    try:
        payload = finalize_run(
            variant=args.variant,
            protocol_manifest_path=args.protocol_manifest,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
            authorization_path=args.authorization,
            authorization_sha256=args.authorization_sha256,
            training_audit_path=args.training_audit,
            checkpoint_path=args.checkpoint,
            checkpoint_sidecar_path=args.checkpoint_sidecar,
            evaluation_path=args.evaluation_json,
        )
        code = 0
    except Exception as exc:
        payload = {
            "schema": EVIDENCE_SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
