from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata import duca_p0_training as legacy


FORMAL_PROTOCOL = "duca_evidence_recovery_full_matrix_v1"
FORMAL_SEEDS = frozenset({8261, 19237, 31153})
ARMS = {
    "C0": "MATCHED_H65_60",
    "F": "FULL",
    "A1": "NO_COVERAGE",
    "A2": "NO_TIME",
    "A3": "NO_ROBUST",
    "A4": "NO_MERGE",
    "A5": "NO_RECOVERY",
    "A6": "H65_SELECTION",
}

DUCA_P0_TRAINING_AUDIT_SCHEMA = "duca_evidence_recovery_training_audit_v1"
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = "duca_evidence_recovery_checkpoint_metadata_v1"
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = "duca_evidence_recovery_checkpoint_sidecar_v1"
DUCA_TRAINING_AUDIT_FILENAME = "duca_evidence_recovery_training_audit.json"

atomic_write_json = legacy.atomic_write_json
capture_global_rng_state = legacy.capture_global_rng_state
canonical_sha256 = legacy.canonical_sha256
restore_global_rng_state = legacy.restore_global_rng_state
sha256_file = legacy.sha256_file


def _flatten_cfg_options(value: Mapping[str, Any], prefix: str = ""):
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            yield from _flatten_cfg_options(item, path)
        else:
            yield path, item


def assert_safe_cfg_options(cfg_options: Mapping[str, Any] | None, *, entrypoint: str) -> None:
    if not cfg_options:
        return
    allowed = {"seed", "work_dir"}
    if entrypoint == "tools/test.py":
        allowed.update(
            {
                "post_processing.save_dict",
                "inference.load_from_raw_predictions",
            }
        )
    rejected = sorted(path for path, _ in _flatten_cfg_options(cfg_options) if path not in allowed)
    if rejected:
        raise RuntimeError(
            f"{entrypoint} rejected Evidence-Recovery cfg overrides: "
            + ", ".join(rejected)
        )
    values = dict(_flatten_cfg_options(cfg_options))
    if "seed" in values and int(values["seed"]) not in FORMAL_SEEDS:
        raise RuntimeError("Evidence-Recovery seed is outside the frozen matrix")
    if "work_dir" in values and not str(values["work_dir"]).strip():
        raise RuntimeError("Evidence-Recovery work_dir override must be non-empty")
    if entrypoint == "tools/test.py":
        if values.get("post_processing.save_dict", True) is not True:
            raise RuntimeError("formal Evidence-Recovery evaluation must save predictions")
        if values.get("inference.load_from_raw_predictions", False) is not False:
            raise RuntimeError("formal Evidence-Recovery evaluation forbids raw predictions")


def formal_training_contract(cfg) -> dict[str, Any] | None:
    workflow = cfg.workflow
    if str(workflow.get("formal_protocol", "")) != FORMAL_PROTOCOL:
        return None
    contract = legacy.formal_training_contract(cfg)
    if contract is None:
        raise ValueError("Evidence-Recovery requires the successful-update contract")
    if (
        int(contract["end_epoch"]) != 60
        or int(contract["expected_train_batches_per_epoch"]) != 100
        or int(contract["expected_successful_optimizer_updates"]) != 6000
    ):
        raise ValueError("Evidence-Recovery is frozen to 60 epochs and 6000 updates")
    arm_id = str(cfg.get("arm_id", ""))
    arm_name = str(cfg.get("arm_name", ""))
    if ARMS.get(arm_id) != arm_name:
        raise ValueError("Evidence-Recovery arm identity is not in the frozen matrix")
    selector = cfg.model.frame_selector
    if int(selector.budget) != 384 or int(selector.window_size) != 768:
        raise ValueError("Evidence-Recovery requires K=384 over T=768")
    contract = dict(contract)
    contract.update(
        {
            "formal_protocol": FORMAL_PROTOCOL,
            "arm_id": arm_id,
            "arm_name": arm_name,
            "selector_schedule_enabled": False,
        }
    )
    return contract


def bind_train_loader_contract(
    contract: Mapping[str, Any],
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = int(contract["expected_train_batches_per_epoch"])
    available = int(len(train_loader))
    if available < expected:
        raise RuntimeError(
            f"Evidence-Recovery loader exposes only {available} batches; "
            f"{expected} are required"
        )
    manifest = {
        "schema": "duca_evidence_recovery_train_loader_contract_v1",
        "dataset_class": train_dataset.__class__.__name__,
        "dataset_length": int(len(train_dataset)),
        "available_batches_per_epoch": available,
        "exposed_batches_per_epoch": expected,
        "exposure": "deterministic_epoch_prefix",
        "batch_size_per_process": int(cfg.solver.train.batch_size),
        "num_workers_per_process": int(cfg.solver.train.num_workers),
        "world_size": int(world_size),
        "drop_last": True,
        "shuffle": True,
        "sampler_class": train_loader.sampler.__class__.__name__,
    }
    manifest["contract_sha256"] = canonical_sha256(manifest)
    bound = dict(contract)
    bound.update(
        {
            "allow_train_loader_prefix": True,
            "available_train_batches_per_epoch": available,
            "train_loader_exposure": "deterministic_epoch_prefix",
        }
    )
    return bound, manifest


def _bound_file(path: str | Path, label: str) -> tuple[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Evidence-Recovery {label} is missing: {resolved}")
    return str(resolved), sha256_file(resolved)


def build_runtime_bindings(
    *,
    git_commit: str,
    variant: str,
    seed: int,
    slurm_job_id: str | None,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
    runtime_config_sha256: str,
    evaluation_annotation_path: str | Path,
    evaluation_class_map_path: str | Path,
    evaluation_config: Mapping[str, Any],
    runtime_pretrain_path: str | Path,
    arm_name: str,
    ledger_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    arm_id = str(variant)
    if ARMS.get(arm_id) != str(arm_name):
        raise RuntimeError("Evidence-Recovery runtime arm identity drift")
    if int(seed) not in FORMAL_SEEDS:
        raise RuntimeError("Evidence-Recovery runtime seed is outside the frozen matrix")
    if os.environ.get("DUCA_EXPECTED_COMMIT") != git_commit:
        raise RuntimeError("Evidence-Recovery checkout differs from DUCA_EXPECTED_COMMIT")
    if not slurm_job_id or not str(slurm_job_id).isdigit():
        raise RuntimeError("formal Evidence-Recovery training requires a Slurm job id")
    source_path, observed_source_sha256 = _bound_file(source_config_path, "source config")
    if observed_source_sha256 != source_config_sha256:
        raise RuntimeError("Evidence-Recovery source config changed during launch")

    pretrain_path, pretrain_sha256 = _bound_file(runtime_pretrain_path, "VideoMAE pretrain")
    annotation_path, annotation_sha256 = _bound_file(
        evaluation_annotation_path, "evaluation annotation"
    )
    class_map_path, class_map_sha256 = _bound_file(
        evaluation_class_map_path, "evaluation class map"
    )
    if set(ledger_paths) != {"train", "val", "test"}:
        raise RuntimeError("Evidence-Recovery requires train/val/test H65 ledger identities")
    ledgers = {}
    for split in ("train", "val", "test"):
        ledger_path, ledger_sha256 = _bound_file(
            ledger_paths[split], f"{split} H65 ledger"
        )
        ledgers[split] = {"path": ledger_path, "sha256": ledger_sha256}

    return {
        "formal_protocol": FORMAL_PROTOCOL,
        "git_commit": str(git_commit),
        "arm_id": arm_id,
        "arm_name": str(arm_name),
        "seed": int(seed),
        "slurm_job_id": str(slurm_job_id),
        "source_config_path": source_path,
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "runtime_pretrain_path": pretrain_path,
        "runtime_pretrain_sha256": pretrain_sha256,
        "evaluation_annotation_path": annotation_path,
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_class_map_path": class_map_path,
        "evaluation_class_map_sha256": class_map_sha256,
        "evaluation_config_sha256": canonical_sha256(dict(evaluation_config)),
        "h65_ledgers": ledgers,
    }


def validate_ledger_coverage(dataset, split_name: str) -> dict[str, int | str]:
    """Validate one ledger row per unique physical dataset window."""

    transforms = list(getattr(getattr(dataset, "pipeline", None), "transforms", ()))
    ledger_transform = next(
        (
            item
            for item in transforms
            if item.__class__.__name__ == "DucaH65PositionsFromLedger"
        ),
        None,
    )
    if ledger_transform is None:
        raise RuntimeError(
            f"formal Evidence {split_name} pipeline has no H65 ledger transform"
        )
    if bool(getattr(ledger_transform, "allow_missing", True)):
        raise RuntimeError(
            f"formal Evidence {split_name} ledger transform must fail on missing rows"
        )

    ledger = ledger_transform._value_transport_ledger()
    exposure_ids = []
    for item in getattr(dataset, "data_list", ()):
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            raise RuntimeError(
                f"formal Evidence {split_name} data_list contains an invalid window record"
            )
        video_name, snippet_centers = str(item[0]), item[3]
        if len(snippet_centers) <= 0:
            raise RuntimeError(
                f"formal Evidence {split_name} has an empty snippet window for {video_name}"
            )
        exposure_ids.append(f"{video_name}|{int(snippet_centers[0])}")

    expected_ids = set(exposure_ids)
    ledger_ids = set(ledger)
    missing = sorted(expected_ids - ledger_ids)
    if missing:
        raise RuntimeError(
            f"formal Evidence {split_name} ledger is missing {len(missing)} windows; "
            f"first: {', '.join(missing[:5])}"
        )
    extra = sorted(ledger_ids - expected_ids)
    if extra:
        raise RuntimeError(
            f"formal Evidence {split_name} ledger contains {len(extra)} unexpected windows; "
            f"first: {', '.join(extra[:5])}"
        )

    expected_target = int(ledger_transform.target_len)
    expected_dense = int(ledger_transform.dense_len)
    positions_key = (
        "expanded_selected_positions"
        if bool(getattr(ledger_transform, "use_expanded_positions", False))
        else "selected_positions"
    )
    for sample_id in sorted(expected_ids):
        row = ledger[sample_id]
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} is not a mapping"
            )
        valid_len = int(row.get("valid_len", -1))
        if (
            int(row.get("dense_len", expected_dense)) != expected_dense
            or int(row.get("target_len", expected_target)) != expected_target
            or valid_len <= 0
            or valid_len > expected_dense
        ):
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} has invalid dimensions"
            )
        positions = row.get(positions_key)
        if positions is None:
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} has no positions"
            )
        try:
            positions = [int(item) for item in positions]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} has invalid positions"
            ) from exc
        if not positions:
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} has no positions"
            )
        required_count = ledger_transform._required_count(valid_len)
        if (
            positions != sorted(set(positions))
            or positions[0] < 0
            or positions[-1] >= valid_len
            or len(positions) > expected_target
            or (
                required_count is not None
                and len(positions) != int(required_count)
            )
        ):
            raise RuntimeError(
                f"formal Evidence {split_name} ledger row {sample_id} violates the selection contract"
            )

    return {
        "split": str(split_name),
        "loader_exposures": len(exposure_ids),
        "unique_physical_windows": len(expected_ids),
        "ledger_rows": len(ledger),
    }


def new_update_audit() -> dict[str, int]:
    audit = legacy.new_update_audit()
    audit.update(
        {
            "nonfinite_loss_attempts": 0,
            "nonfinite_loss_replays": 0,
            "nonfinite_loss_replay_exhaustions": 0,
            "max_nonfinite_loss_retries_observed": 0,
            "replay_state_restorations": 0,
        }
    )
    return audit


def selector_schedule_step(model) -> int:
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    if selector is None or selector.__class__.__name__ != "DucaEvidenceRecoveryFrameSelector":
        raise RuntimeError("formal Evidence-Recovery model lacks its frame selector")
    if hasattr(selector, "_loss_weight_schedule_step"):
        raise RuntimeError("Evidence-Recovery unexpectedly exposes a DUCA-P0 schedule")
    return 0


def validate_update_state(
    *,
    contract: Mapping[str, Any],
    epoch: int,
    train_batches_per_epoch: int,
    update_audit: Mapping[str, Any],
    scheduler_last_epoch: int,
    selector_step: int,
    uses_ema: bool,
) -> None:
    expected_batches = int(contract["expected_train_batches_per_epoch"])
    expected_updates = (int(epoch) + 1) * expected_batches
    successful = int(update_audit.get("successful_optimizer_updates", -1))
    skipped = int(update_audit.get("amp_skipped_attempts", -1))
    if int(train_batches_per_epoch) != expected_batches:
        raise RuntimeError("Evidence-Recovery train-loader exposure changed")
    if successful != expected_updates:
        raise RuntimeError("Evidence-Recovery successful-update count mismatch")
    if int(update_audit.get("attempted_batches", -1)) != expected_updates:
        raise RuntimeError("Evidence-Recovery consumed-batch count mismatch")
    if int(update_audit.get("optimizer_attempts", -1)) != successful + skipped:
        raise RuntimeError("Evidence-Recovery optimizer-attempt accounting mismatch")
    if int(update_audit.get("replay_exhaustions", -1)) != 0:
        raise RuntimeError("Evidence-Recovery AMP replay exhausted")
    if int(update_audit.get("nonfinite_loss_replay_exhaustions", -1)) != 0:
        raise RuntimeError("Evidence-Recovery non-finite-loss replay exhausted")
    if int(update_audit.get("scheduler_updates", -1)) != successful:
        raise RuntimeError("Evidence-Recovery scheduler exposure mismatch")
    if int(update_audit.get("ema_updates", -1)) != (successful if uses_ema else 0):
        raise RuntimeError("Evidence-Recovery EMA exposure mismatch")
    if int(update_audit.get("duca_schedule_updates", -1)) != 0 or int(selector_step) != 0:
        raise RuntimeError("Evidence-Recovery must not advance a DUCA-P0 schedule")
    if int(scheduler_last_epoch) != successful:
        raise RuntimeError("Evidence-Recovery scheduler state differs from successful updates")
    if int(update_audit.get("max_amp_retries_observed", -1)) > int(
        contract["max_amp_retries_per_batch"]
    ):
        raise RuntimeError("Evidence-Recovery AMP retries exceeded the frozen limit")


def build_training_audit(
    *,
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
    epoch: int,
    train_batches_per_epoch: int,
    update_audit: Mapping[str, Any],
    epoch_records: list[Mapping[str, Any]],
    scheduler_last_epoch: int,
    selector_step: int,
    scaler_scale: float | None,
    uses_ema: bool,
    complete: bool,
) -> dict[str, Any]:
    validate_update_state(
        contract=contract,
        epoch=epoch,
        train_batches_per_epoch=train_batches_per_epoch,
        update_audit=update_audit,
        scheduler_last_epoch=scheduler_last_epoch,
        selector_step=selector_step,
        uses_ema=uses_ema,
    )
    if [int(item.get("epoch", -1)) for item in epoch_records] != list(
        range(int(epoch) + 1)
    ):
        raise RuntimeError("Evidence-Recovery epoch records are incomplete")
    payload = {
        "schema_version": DUCA_P0_TRAINING_AUDIT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        **dict(bindings),
        "checkpoint_criterion": contract["checkpoint_criterion"],
        "primary_checkpoint_epoch": int(contract["primary_checkpoint_epoch"]),
        "primary_checkpoint_state_key": contract["primary_checkpoint_state_key"],
        "expected_train_batches_per_epoch": int(contract["expected_train_batches_per_epoch"]),
        "expected_successful_optimizer_updates": int(
            contract["expected_successful_optimizer_updates"]
        ),
        "available_train_batches_per_epoch": int(
            contract["available_train_batches_per_epoch"]
        ),
        "train_loader_exposure": str(contract["train_loader_exposure"]),
        "last_completed_epoch": int(epoch),
        "epochs_completed": int(epoch) + 1,
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "selector_schedule_enabled": False,
        "update_audit": {key: int(value) for key, value in update_audit.items()},
        "scheduler_last_epoch": int(scheduler_last_epoch),
        "selector_schedule_step": int(selector_step),
        "grad_scaler_scale": None if scaler_scale is None else float(scaler_scale),
        "epoch_records": [dict(item) for item in epoch_records],
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


def build_checkpoint_metadata(training_audit: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "schema_version": DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
        "training_audit": dict(training_audit),
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def _validated_checkpoint_audit(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != DUCA_P0_CHECKPOINT_METADATA_SCHEMA:
        raise RuntimeError("Evidence-Recovery checkpoint metadata schema mismatch")
    unsigned_metadata = dict(metadata)
    metadata_sha256 = unsigned_metadata.pop("metadata_sha256", None)
    if metadata_sha256 != canonical_sha256(unsigned_metadata):
        raise RuntimeError("Evidence-Recovery checkpoint metadata hash mismatch")
    audit = metadata.get("training_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != DUCA_P0_TRAINING_AUDIT_SCHEMA:
        raise RuntimeError("Evidence-Recovery checkpoint training audit schema mismatch")
    unsigned_audit = dict(audit)
    audit_sha256 = unsigned_audit.pop("audit_sha256", None)
    if audit_sha256 != canonical_sha256(unsigned_audit):
        raise RuntimeError("Evidence-Recovery checkpoint training audit hash mismatch")
    return audit


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    audit = _validated_checkpoint_audit(checkpoint)
    for key, expected in bindings.items():
        if key != "slurm_job_id" and audit.get(key) != expected:
            raise RuntimeError(f"Evidence-Recovery resume binding mismatch: {key}")
    if audit.get("checkpoint_criterion") != contract["checkpoint_criterion"]:
        raise RuntimeError("Evidence-Recovery checkpoint criterion mismatch")
    counters = audit.get("update_audit")
    records = audit.get("epoch_records")
    if not isinstance(counters, Mapping) or not isinstance(records, list):
        raise RuntimeError("Evidence-Recovery resume state is incomplete")
    restored = {str(key): int(value) for key, value in counters.items()}
    if set(restored) != set(new_update_audit()):
        raise RuntimeError("Evidence-Recovery resume counters are incomplete")
    return restored, [dict(item) for item in records]


validate_checkpoint_successful_optimizer_updates = (
    legacy.validate_checkpoint_successful_optimizer_updates
)


def validate_terminal_checkpoint_binding(
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    git_commit: str,
    arm_id: str,
    arm_name: str,
    seed: int,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
    checkpoint_epoch: int,
    checkpoint_state_key: str,
    evaluation_annotation_path: str | Path,
    evaluation_class_map_path: str | Path,
    runtime_pretrain_path: str | Path,
) -> dict[str, Any]:
    audit = _validated_checkpoint_audit(checkpoint)
    source_path, observed_source_sha256 = _bound_file(source_config_path, "evaluation config")
    pretrain_path, pretrain_sha256 = _bound_file(runtime_pretrain_path, "VideoMAE pretrain")
    annotation_path, annotation_sha256 = _bound_file(
        evaluation_annotation_path, "evaluation annotation"
    )
    class_map_path, class_map_sha256 = _bound_file(
        evaluation_class_map_path, "evaluation class map"
    )
    counters = audit.get("update_audit", {})
    expected = {
        "formal_protocol": FORMAL_PROTOCOL,
        "git_commit": git_commit,
        "arm_id": arm_id,
        "arm_name": arm_name,
        "seed": int(seed),
        "source_config_path": source_path,
        "source_config_sha256": source_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "runtime_pretrain_path": pretrain_path,
        "runtime_pretrain_sha256": pretrain_sha256,
        "evaluation_annotation_path": annotation_path,
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_class_map_path": class_map_path,
        "evaluation_class_map_sha256": class_map_sha256,
    }
    if observed_source_sha256 != source_config_sha256:
        raise RuntimeError("Evidence-Recovery evaluation config hash mismatch")
    training_config_name = Path(str(audit.get("source_config_path", ""))).name
    evaluation_config_name = Path(source_path).name
    if training_config_name != evaluation_config_name:
        raise RuntimeError(
            "Evidence-Recovery terminal checkpoint binding mismatch: source_config_path"
        )
    for key, value in expected.items():
        if key == "source_config_path":
            continue
        if audit.get(key) != value:
            raise RuntimeError(f"Evidence-Recovery terminal checkpoint binding mismatch: {key}")
    if (
        audit.get("status") != "complete"
        or int(audit.get("last_completed_epoch", -1)) != 59
        or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
        or int(counters.get("successful_optimizer_updates", -1)) != 6000
        or int(counters.get("duca_schedule_updates", -1)) != 0
        or int(checkpoint.get("successful_optimizer_updates", -1)) != 6000
        or int(checkpoint_epoch) != 59
        or checkpoint_state_key != "state_dict_ema"
    ):
        raise RuntimeError("Evidence-Recovery checkpoint is not a complete terminal run")
    checkpoint_resolved, checkpoint_sha256 = _bound_file(
        checkpoint_path, "terminal checkpoint"
    )
    return {
        "checkpoint_path": checkpoint_resolved,
        "checkpoint_sha256": checkpoint_sha256,
        "training_audit_sha256": audit["audit_sha256"],
        "training_slurm_job_id": audit["slurm_job_id"],
    }


__all__ = [
    "ARMS",
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_P0_TRAINING_AUDIT_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "FORMAL_PROTOCOL",
    "FORMAL_SEEDS",
    "assert_safe_cfg_options",
    "atomic_write_json",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "formal_training_contract",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "validate_checkpoint_successful_optimizer_updates",
    "validate_ledger_coverage",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
