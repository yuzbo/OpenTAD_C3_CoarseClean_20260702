from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tools.bata.duca_p0_training import (
    DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    DUCA_P0_TRAINING_AUDIT_SCHEMA,
    DUCA_P0_VARIANTS,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from tools.bata.duca_p0_evaluation import (
    normalize_evaluation_config,
    official_evaluator_identity,
    recompute_official_map,
)


EVIDENCE_SCHEMA = "duca_p0_post_run_evidence_v3"
EVALUATION_SCHEMA = "duca_p0_terminal_evaluation_v3"
EXPECTED_UPDATES = 13200
TERMINAL_EPOCH = 131
TERMINAL_STATE_KEY = "state_dict_ema"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(
    payload: Mapping[str, Any], hash_key: str, label: str
) -> None:
    expected = payload.get(hash_key)
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    _require(
        isinstance(expected, str) and expected == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _validate_metrics(metrics: Any) -> dict[str, float]:
    _require(isinstance(metrics, Mapping), "terminal evaluation metrics are missing")
    required = ["average_mAP", *(f"mAP@{value}" for value in (0.3, 0.4, 0.5, 0.6, 0.7))]
    out = {}
    for key in required:
        value = metrics.get(key)
        _require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"terminal metric {key} is invalid",
        )
        out[key] = float(value)
    return out


def _inspect_checkpoint_payload(
    checkpoint_path: Path, expected_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "terminal checkpoint is not a mapping")
    _require(int(checkpoint.get("epoch", -1)) == TERMINAL_EPOCH, "checkpoint payload epoch mismatch")
    for key in (
        "state_dict",
        "state_dict_ema",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "rng_state",
        "experiment_metadata",
    ):
        _require(key in checkpoint, f"terminal checkpoint payload is missing {key}")
    _require(checkpoint["experiment_metadata"] == expected_metadata, "checkpoint embedded metadata mismatch")
    _require(int(checkpoint["scheduler"].get("last_epoch", -1)) == EXPECTED_UPDATES, "checkpoint scheduler is not at 13200 updates")
    rng_state = checkpoint["rng_state"]
    _require(
        isinstance(rng_state, Mapping)
        and set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "checkpoint global RNG state is incomplete",
    )

    selector_steps = {}
    for state_key in ("state_dict", "state_dict_ema"):
        state = checkpoint[state_key]
        matches = [
            value
            for name, value in state.items()
            if str(name).endswith("frame_selector._loss_weight_schedule_step")
        ]
        _require(len(matches) == 1, f"{state_key} selector schedule buffer is missing or ambiguous")
        selector_steps[state_key] = int(matches[0].detach().cpu().item())
        _require(selector_steps[state_key] == EXPECTED_UPDATES, f"{state_key} selector schedule is not at 13200")
    del checkpoint
    return {
        "payload_reopened": True,
        "epoch": TERMINAL_EPOCH,
        "scheduler_last_epoch": EXPECTED_UPDATES,
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
    _require(variant in DUCA_P0_VARIANTS, f"unsupported DUCA P0 variant: {variant}")
    run_manifest_file, run_manifest = _load_json(run_manifest_path, "run manifest")
    audit_file, audit = _load_json(training_audit_path, "training audit")
    sidecar_file, sidecar = _load_json(checkpoint_sidecar_path, "checkpoint sidecar")
    evaluation_file, evaluation = _load_json(evaluation_path, "terminal evaluation")
    checkpoint_file = Path(checkpoint_path).resolve()
    _require(checkpoint_file.is_file(), f"terminal checkpoint is missing: {checkpoint_file}")

    _require(run_manifest.get("variant") == variant, "run manifest variant mismatch")
    _require(audit.get("schema_version") == DUCA_P0_TRAINING_AUDIT_SCHEMA, "training audit schema mismatch")
    _validate_embedded_hash(audit, "audit_sha256", "training audit")
    _require(audit.get("status") == "complete", "training audit is not complete")
    _require(int(audit.get("last_completed_epoch", -1)) == TERMINAL_EPOCH, "training did not reach epoch 131")
    _require(int(audit.get("epochs_completed", -1)) == TERMINAL_EPOCH + 1, "training epoch count mismatch")
    epoch_records = audit.get("epoch_records")
    _require(
        isinstance(epoch_records, list)
        and [int(item.get("epoch", -1)) for item in epoch_records]
        == list(range(TERMINAL_EPOCH + 1)),
        "training epoch records are incomplete or non-contiguous",
    )
    _require(int(audit.get("expected_successful_optimizer_updates", -1)) == EXPECTED_UPDATES, "training contract update count mismatch")
    counters = audit.get("update_audit")
    _require(isinstance(counters, Mapping), "training update audit is missing")
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        _require(int(counters.get(key, -1)) == EXPECTED_UPDATES, f"training counter {key} mismatch")
    _require(int(counters.get("replay_exhaustions", -1)) == 0, "training exhausted AMP replay")
    _require(int(counters.get("forced_amp_overflow_attempts", -1)) == 0, "formal training injected a synthetic AMP overflow")
    _require(
        int(counters.get("optimizer_attempts", -1))
        == EXPECTED_UPDATES + int(counters.get("amp_skipped_attempts", -1)),
        "optimizer attempt accounting mismatch",
    )
    _require(int(audit.get("scheduler_last_epoch", -1)) == EXPECTED_UPDATES, "scheduler state mismatch")
    _require(int(audit.get("selector_schedule_step", -1)) == EXPECTED_UPDATES, "selector schedule state mismatch")

    _require(sidecar.get("schema_version") == DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA, "checkpoint sidecar schema mismatch")
    _validate_embedded_hash(sidecar, "sidecar_sha256", "checkpoint sidecar")
    checkpoint_sha256 = sha256_file(checkpoint_file)
    _require(sidecar.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint hash differs from sidecar")
    _require(Path(str(sidecar.get("checkpoint_path", ""))).resolve() == checkpoint_file, "checkpoint path differs from sidecar")
    metadata = sidecar.get("experiment_metadata")
    _require(isinstance(metadata, Mapping), "checkpoint metadata is missing")
    _require(metadata.get("schema_version") == DUCA_P0_CHECKPOINT_METADATA_SCHEMA, "checkpoint metadata schema mismatch")
    _validate_embedded_hash(metadata, "metadata_sha256", "checkpoint metadata")
    _require(metadata.get("training_audit") == audit, "checkpoint does not embed the terminal training audit")
    checkpoint_payload_contract = _inspect_checkpoint_payload(checkpoint_file, metadata)

    _require(evaluation.get("schema_version") == EVALUATION_SCHEMA, "terminal evaluation schema mismatch")
    _validate_embedded_hash(evaluation, "evaluation_sha256", "terminal evaluation")
    _require(evaluation.get("git_commit") == audit.get("git_commit"), "evaluation commit differs from training")
    _require(
        evaluation.get("config_sha256")
        == audit.get("source_config_sha256")
        == run_manifest.get("config_sha256"),
        "terminal detector config differs from training",
    )
    _require(Path(str(evaluation.get("checkpoint_path", ""))).resolve() == checkpoint_file, "evaluation used a different checkpoint")
    _require(evaluation.get("checkpoint_sha256") == checkpoint_sha256, "evaluation checkpoint hash mismatch")
    _require(int(evaluation.get("checkpoint_epoch", -1)) == TERMINAL_EPOCH, "evaluation did not use epoch 131")
    _require(evaluation.get("checkpoint_state_key") == TERMINAL_STATE_KEY, "evaluation did not use state_dict_ema")
    prediction_file = Path(str(evaluation.get("prediction_path", ""))).resolve()
    _require(prediction_file.is_file(), "terminal prediction JSON is missing")
    _require(evaluation.get("prediction_sha256") == sha256_file(prediction_file), "terminal prediction hash mismatch")
    evaluator = evaluation.get("evaluator")
    _require(isinstance(evaluator, Mapping), "evaluator identity is missing")
    frozen_evaluator = official_evaluator_identity()
    _require(dict(evaluator) == frozen_evaluator, "terminal evaluator is not the frozen OpenTAD mAP implementation")
    evaluation_config = normalize_evaluation_config(
        evaluation.get("evaluation_config")
    )
    evaluation_config_sha256 = canonical_sha256(evaluation_config)
    _require(
        evaluation.get("evaluation_config_sha256")
        == evaluation_config_sha256
        == audit.get("evaluation_config_sha256"),
        "terminal evaluation config identity mismatch",
    )
    annotation_file = Path(
        str(evaluation.get("evaluation_annotation_path", ""))
    ).resolve()
    class_map_file = Path(
        str(evaluation.get("evaluation_class_map_path", ""))
    ).resolve()
    _require(annotation_file.is_file(), "evaluation annotation file is missing")
    _require(class_map_file.is_file(), "evaluation class-map file is missing")
    _require(
        Path(evaluation_config["ground_truth_filename"]).resolve()
        == annotation_file,
        "evaluation config uses a different annotation file",
    )
    _require(
        evaluation.get("evaluation_annotation_sha256")
        == sha256_file(annotation_file)
        == audit.get("evaluation_annotation_sha256"),
        "evaluation annotation identity mismatch",
    )
    _require(
        evaluation.get("evaluation_class_map_sha256")
        == sha256_file(class_map_file)
        == audit.get("evaluation_class_map_sha256"),
        "evaluation class-map identity mismatch",
    )
    reported_metrics = _validate_metrics(evaluation.get("metrics"))
    recomputed = recompute_official_map(prediction_file, evaluation_config)
    _require(
        recomputed["evaluator"] == frozen_evaluator,
        "recomputed evaluator identity mismatch",
    )
    _require(
        recomputed["evaluation_config_sha256"] == evaluation_config_sha256,
        "recomputed evaluation config identity mismatch",
    )
    _require(
        int(evaluation.get("result_count", -1)) == recomputed["result_count"],
        "terminal result count differs from the prediction artifact",
    )
    _require(
        int(evaluation.get("video_count", -1)) == recomputed["video_count"],
        "terminal video count differs from the prediction artifact",
    )
    metrics = _validate_metrics(recomputed["metrics"])
    for key, value in metrics.items():
        _require(
            math.isclose(reported_metrics[key], value, rel_tol=0.0, abs_tol=1e-12),
            f"terminal metric {key} differs from official prediction recomputation",
        )

    binding_map = {
        "git_commit": "git_commit",
        "seed": "seed",
        "source_config_sha256": "config_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "variant_contract_sha256": "variant_contract_sha256",
        "shared_protocol_sha256": "shared_protocol_sha256",
        "core_gate_sha256": "core_gate_json_sha256",
        "ddp_pilot_sha256": "ddp_pilot_json_sha256",
        "canonical_env_sha256": "canonical_env_sha256",
        "evaluation_annotation_sha256": "evaluation_annotation_sha256",
        "evaluation_class_map_sha256": "evaluation_class_map_sha256",
        "evaluation_config_sha256": "evaluation_config_sha256",
    }
    for audit_key, manifest_key in binding_map.items():
        _require(audit.get(audit_key) == run_manifest.get(manifest_key), f"run binding mismatch: {audit_key}")

    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "ok": True,
        "variant": variant,
        "git_commit": audit["git_commit"],
        "seed": int(audit["seed"]),
        "config_sha256": audit["source_config_sha256"],
        "resolved_config_sha256": audit["resolved_config_sha256"],
        "variant_contract_sha256": audit["variant_contract_sha256"],
        "shared_protocol_sha256": audit["shared_protocol_sha256"],
        "core_gate_sha256": audit["core_gate_sha256"],
        "ddp_pilot_sha256": audit["ddp_pilot_sha256"],
        "canonical_env_sha256": audit["canonical_env_sha256"],
        "successful_optimizer_updates": EXPECTED_UPDATES,
        "lr_scheduler_successful_update_exposure": EXPECTED_UPDATES,
        "ema_successful_update_exposure": EXPECTED_UPDATES,
        "selector_schedule_successful_update_exposure": EXPECTED_UPDATES,
        "amp_skipped_attempts": int(counters["amp_skipped_attempts"]),
        "max_amp_retries_observed": int(counters["max_amp_retries_observed"]),
        "checkpoint_criterion": "terminal_epoch_131_state_dict_ema",
        "checkpoint_epoch": TERMINAL_EPOCH,
        "checkpoint_state_key": TERMINAL_STATE_KEY,
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_path": str(sidecar_file),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_file),
        "checkpoint_payload_contract": checkpoint_payload_contract,
        "training_audit_path": str(audit_file),
        "training_audit_sha256": sha256_file(audit_file),
        "terminal_evaluation_path": str(evaluation_file),
        "terminal_evaluation_sha256": sha256_file(evaluation_file),
        "prediction_path": str(prediction_file),
        "prediction_sha256": evaluation["prediction_sha256"],
        "evaluation_annotation_path": str(annotation_file),
        "evaluation_annotation_sha256": evaluation[
            "evaluation_annotation_sha256"
        ],
        "evaluation_class_map_path": str(class_map_file),
        "evaluation_class_map_sha256": evaluation[
            "evaluation_class_map_sha256"
        ],
        "evaluation_config": evaluation_config,
        "evaluation_config_sha256": evaluation_config_sha256,
        "evaluator": frozen_evaluator,
        "metric_recomputation": {
            "performed": True,
            "implementation": "opentad.evaluations.mAP.compute_average_precision_detection",
            "result_count": recomputed["result_count"],
            "video_count": recomputed["video_count"],
        },
        "metrics": metrics,
        "non_finite_collapse": False,
        "run_manifest_path": str(run_manifest_file),
        "run_manifest_sha256": sha256_file(run_manifest_file),
    }
    payload["artifact_chain_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=DUCA_P0_VARIANTS)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sidecar", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
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
        payload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    atomic_write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
