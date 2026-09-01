from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata.duca_cellcf_training import (
    DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    DUCA_P0_TRAINING_AUDIT_SCHEMA,
    VARIANTS,
    canonical_sha256,
    sha256_file,
)
from tools.bata.duca_cellcf_protocol import (
    CellCFTrainingProtocol,
    LEGACY_EXPOSURE132_COMMITS,
    protocol_for_name,
)
from tools.bata.duca_p0_evaluation import (
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)


EVIDENCE_SCHEMA = "duca_cellcf_post_run_evidence_v1"
EVALUATION_SCHEMA = "duca_cellcf_terminal_evaluation_v1"
_DEFAULT_PROTOCOL = protocol_for_name("exposure132")
EXPECTED_UPDATES = _DEFAULT_PROTOCOL.expected_successful_optimizer_updates
TERMINAL_EPOCH = _DEFAULT_PROTOCOL.terminal_epoch
TERMINAL_STATE_KEY = _DEFAULT_PROTOCOL.terminal_state_key


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(payload: Mapping[str, Any], hash_key: str, label: str) -> None:
    expected = payload.get(hash_key)
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    _require(
        isinstance(expected, str) and expected == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _validate_metrics(metrics: Any) -> dict[str, float]:
    _require(isinstance(metrics, Mapping), "terminal metrics are missing")
    required = ["average_mAP", *(f"mAP@{value}" for value in (0.3, 0.4, 0.5, 0.6, 0.7))]
    output = {}
    for key in required:
        value = metrics.get(key)
        _require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"terminal metric {key} is invalid",
        )
        output[key] = float(value)
    return output


def _resolve_artifact_protocol(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> CellCFTrainingProtocol:
    manifest_commit = str(manifest.get("git_commit") or "")
    audit_commit = str(audit.get("git_commit") or "")
    _require(
        manifest_commit and manifest_commit == audit_commit,
        "run manifest and training audit commit mismatch",
    )
    manifest_profile = manifest.get("training_profile")
    audit_profile = audit.get("training_profile")
    if manifest_profile is None or audit_profile is None:
        _require(
            manifest_commit in LEGACY_EXPOSURE132_COMMITS
            and manifest_profile in (None, "exposure132")
            and audit_profile in (None, "exposure132"),
            "training profile may be omitted only by the audited legacy exposure132 commit",
        )
        return protocol_for_name("exposure132")
    _require(
        manifest_profile == audit_profile,
        "run manifest and training audit profile mismatch",
    )
    return protocol_for_name(str(manifest_profile))


def _inspect_checkpoint(
    checkpoint_path: Path,
    expected_metadata: Mapping[str, Any],
    protocol: CellCFTrainingProtocol,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "terminal checkpoint is not a mapping")
    _require(
        int(checkpoint.get("epoch", -1)) == protocol.terminal_epoch,
        "checkpoint epoch mismatch",
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
    _require(checkpoint["experiment_metadata"] == expected_metadata, "embedded checkpoint metadata mismatch")
    _require(
        int(checkpoint["scheduler"].get("last_epoch", -1))
        == protocol.expected_successful_optimizer_updates,
        "scheduler is not at the frozen terminal update",
    )
    rng_state = checkpoint["rng_state"]
    _require(
        isinstance(rng_state, Mapping)
        and set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "checkpoint RNG state is incomplete",
    )
    selector_steps = {}
    for state_key in ("state_dict", "state_dict_ema"):
        matches = [
            value
            for name, value in checkpoint[state_key].items()
            if str(name).endswith("frame_selector._loss_weight_schedule_step")
        ]
        _require(len(matches) == 1, f"{state_key} selector schedule is missing or ambiguous")
        selector_steps[state_key] = int(matches[0].detach().cpu().item())
        _require(
            selector_steps[state_key]
            == protocol.expected_successful_optimizer_updates,
            f"{state_key} selector schedule is not at the frozen terminal update",
        )
    del checkpoint
    return {
        "payload_reopened": True,
        "epoch": protocol.terminal_epoch,
        "scheduler_last_epoch": protocol.expected_successful_optimizer_updates,
        "selector_schedule_steps": selector_steps,
        "grad_scaler_present": True,
        "global_rng_state_present": True,
        "embedded_metadata_exact": True,
    }


def finalize_run(
    *,
    variant: str,
    run_manifest_path: str | Path,
    training_audit_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sidecar_path: str | Path,
    evaluation_path: str | Path,
) -> dict[str, Any]:
    _require(variant in VARIANTS, f"unsupported CellCF variant: {variant}")
    manifest_file, manifest = _load_json(run_manifest_path, "run manifest")
    audit_file, audit = _load_json(training_audit_path, "training audit")
    protocol = _resolve_artifact_protocol(manifest, audit)
    expected_updates = protocol.expected_successful_optimizer_updates
    terminal_epoch = protocol.terminal_epoch
    terminal_state_key = protocol.terminal_state_key
    sidecar_file, sidecar = _load_json(checkpoint_sidecar_path, "checkpoint sidecar")
    evaluation_file, evaluation = _load_json(evaluation_path, "terminal evaluation")
    checkpoint_file = Path(checkpoint_path).resolve()
    _require(checkpoint_file.is_file(), f"terminal checkpoint is missing: {checkpoint_file}")

    _require(manifest.get("variant") == variant, "run manifest variant mismatch")
    _require(audit.get("schema_version") == DUCA_P0_TRAINING_AUDIT_SCHEMA, "training audit schema mismatch")
    _validate_embedded_hash(audit, "audit_sha256", "training audit")
    _require(audit.get("status") == "complete", "training audit is not complete")
    _require(
        audit.get("training_profile", protocol.name) == protocol.name
        and manifest.get("training_profile", protocol.name) == protocol.name,
        "training profile mismatch",
    )
    _require(
        int(audit.get("last_completed_epoch", -1)) == terminal_epoch,
        "training did not reach the frozen terminal epoch",
    )
    _require(
        int(audit.get("epochs_completed", -1)) == protocol.end_epoch,
        "training epoch count mismatch",
    )
    records = audit.get("epoch_records")
    _require(
        isinstance(records, list)
        and [int(record.get("epoch", -1)) for record in records]
        == list(range(protocol.end_epoch)),
        "training epoch records are incomplete",
    )
    _require(
        int(audit.get("expected_successful_optimizer_updates", -1))
        == expected_updates,
        "update contract mismatch",
    )
    counters = audit.get("update_audit")
    _require(isinstance(counters, Mapping), "training update audit is missing")
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        _require(
            int(counters.get(key, -1)) == expected_updates,
            f"training counter {key} mismatch",
        )
    _require(int(counters.get("replay_exhaustions", -1)) == 0, "training exhausted AMP replay")
    _require(int(counters.get("forced_amp_overflow_attempts", -1)) == 0, "formal training injected AMP overflow")
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
        int(audit.get("selector_schedule_step", -1)) == expected_updates,
        "selector schedule mismatch",
    )

    _require(sidecar.get("schema_version") == DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA, "checkpoint sidecar schema mismatch")
    _validate_embedded_hash(sidecar, "sidecar_sha256", "checkpoint sidecar")
    checkpoint_sha256 = sha256_file(checkpoint_file)
    _require(sidecar.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint hash differs from sidecar")
    metadata = sidecar.get("experiment_metadata")
    _require(isinstance(metadata, Mapping), "checkpoint metadata is missing")
    _require(metadata.get("schema_version") == DUCA_P0_CHECKPOINT_METADATA_SCHEMA, "checkpoint metadata schema mismatch")
    _validate_embedded_hash(metadata, "metadata_sha256", "checkpoint metadata")
    _require(metadata.get("training_audit") == audit, "checkpoint does not embed terminal audit")
    checkpoint_contract = _inspect_checkpoint(checkpoint_file, metadata, protocol)

    _require(evaluation.get("schema_version") == EVALUATION_SCHEMA, "terminal evaluation schema mismatch")
    _validate_embedded_hash(evaluation, "evaluation_sha256", "terminal evaluation")
    _require(evaluation.get("git_commit") == audit.get("git_commit"), "evaluation commit mismatch")
    _require(
        evaluation.get("config_sha256")
        == audit.get("source_config_sha256")
        == manifest.get("config_sha256"),
        "terminal config differs from training",
    )
    _require(
        evaluation.get("resolved_config_sha256")
        == audit.get("resolved_config_sha256")
        == manifest.get("resolved_config_sha256"),
        "terminal resolved config differs from training",
    )
    _require(
        evaluation.get("runtime_config_sha256")
        == manifest.get("evaluation_runtime_config_sha256"),
        "terminal evaluation runtime config differs from the frozen launch",
    )
    _require(Path(str(evaluation.get("checkpoint_path", ""))).resolve() == checkpoint_file, "evaluation used another checkpoint")
    _require(evaluation.get("checkpoint_sha256") == checkpoint_sha256, "evaluation checkpoint hash mismatch")
    _require(
        int(evaluation.get("checkpoint_epoch", -1)) == terminal_epoch,
        "evaluation checkpoint epoch mismatch",
    )
    _require(
        evaluation.get("checkpoint_state_key") == terminal_state_key,
        "evaluation did not use EMA",
    )
    prediction_file = Path(str(evaluation.get("prediction_path", ""))).resolve()
    _require(prediction_file.is_file(), "terminal prediction JSON is missing")
    _require(evaluation.get("prediction_sha256") == sha256_file(prediction_file), "prediction hash mismatch")

    frozen_evaluator = official_evaluator_identity()
    _require(evaluation.get("evaluator") == frozen_evaluator, "terminal evaluator is not frozen OpenTAD mAP")
    evaluation_config = normalize_evaluation_config(evaluation.get("evaluation_config"))
    evaluation_config_hash = canonical_sha256(evaluation_config)
    _require(
        evaluation.get("evaluation_config_sha256")
        == evaluation_config_hash
        == audit.get("evaluation_config_sha256"),
        "evaluation config binding mismatch",
    )
    annotation = Path(str(evaluation.get("evaluation_annotation_path", ""))).resolve()
    class_map = Path(str(evaluation.get("evaluation_class_map_path", ""))).resolve()
    _require(annotation.is_file() and class_map.is_file(), "evaluation data artifacts are missing")
    _require(evaluation.get("evaluation_annotation_sha256") == sha256_file(annotation) == audit.get("evaluation_annotation_sha256"), "annotation binding mismatch")
    _require(evaluation.get("evaluation_class_map_sha256") == sha256_file(class_map) == audit.get("evaluation_class_map_sha256"), "class-map binding mismatch")

    reported_metrics = _validate_metrics(evaluation.get("metrics"))
    recomputed = recompute_official_map(prediction_file, evaluation_config)
    metrics = _validate_metrics(recomputed["metrics"])
    for key, value in metrics.items():
        _require(math.isclose(reported_metrics[key], value, rel_tol=0.0, abs_tol=1e-12), f"reported {key} differs from recomputation")
    _require(int(evaluation.get("result_count", -1)) == recomputed["result_count"], "result count mismatch")
    _require(int(evaluation.get("video_count", -1)) == recomputed["video_count"], "video count mismatch")

    binding_map = {
        "git_commit": "git_commit",
        "seed": "seed",
        "source_config_sha256": "config_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "runtime_config_sha256": "runtime_config_sha256",
        "protocol_sha256": "protocol_sha256",
        "ordered_exposure_sha256": "ordered_exposure_sha256",
        "real_loader_gate_sha256": "real_loader_gate_sha256",
        "ddp_pilot_sha256": "ddp_pilot_sha256",
        "evaluation_annotation_sha256": "evaluation_annotation_sha256",
        "evaluation_class_map_sha256": "evaluation_class_map_sha256",
        "evaluation_config_sha256": "evaluation_config_sha256",
    }
    for audit_key, manifest_key in binding_map.items():
        _require(audit.get(audit_key) == manifest.get(manifest_key), f"run binding mismatch: {audit_key}")

    payload = {
        "schema": EVIDENCE_SCHEMA,
        "ok": True,
        "variant": variant,
        "task": "offline_temporal_action_detection",
        "git_commit": audit["git_commit"],
        "seed": int(audit["seed"]),
        "training_profile": protocol.name,
        "training_protocol": protocol.to_dict(),
        "config_sha256": audit["source_config_sha256"],
        "resolved_config_sha256": audit["resolved_config_sha256"],
        "runtime_config_sha256": audit["runtime_config_sha256"],
        "evaluation_runtime_config_sha256": evaluation["runtime_config_sha256"],
        "protocol_sha256": audit["protocol_sha256"],
        "ordered_exposure_sha256": audit["ordered_exposure_sha256"],
        "real_loader_gate_sha256": audit["real_loader_gate_sha256"],
        "ddp_pilot_sha256": audit["ddp_pilot_sha256"],
        "successful_optimizer_updates": expected_updates,
        "checkpoint_epoch": terminal_epoch,
        "checkpoint_state_key": terminal_state_key,
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
        "evaluation_annotation_sha256": evaluation["evaluation_annotation_sha256"],
        "evaluation_class_map_sha256": evaluation["evaluation_class_map_sha256"],
        "evaluation_config_sha256": evaluation_config_hash,
        "evaluator": frozen_evaluator,
        "metric_recomputation": {
            "performed": True,
            "implementation": "opentad.evaluations.mAP.compute_average_precision_detection",
            "result_count": recomputed["result_count"],
            "video_count": recomputed["video_count"],
        },
        "metrics": metrics,
        "non_finite_collapse": False,
        "run_manifest_path": str(manifest_file),
        "run_manifest_sha256": sha256_file(manifest_file),
    }
    payload["artifact_chain_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sidecar", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        failure = {
            "schema": EVIDENCE_SCHEMA,
            "ok": False,
            "error_type": "FileExistsError",
            "error": "refusing to overwrite post-run evidence",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    try:
        payload = finalize_run(
            variant=args.variant,
            run_manifest_path=args.run_manifest,
            training_audit_path=args.training_audit,
            checkpoint_path=args.checkpoint,
            checkpoint_sidecar_path=args.checkpoint_sidecar,
            evaluation_path=args.evaluation_json,
        )
        code = 0
    except Exception as exc:
        payload = {"schema": EVIDENCE_SCHEMA, "ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
