from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata import duca_p0_training


FORMAL_PROTOCOL = "duca_protected_physical_v1"
TRAINING_AUDIT_SCHEMA = "duca_protected_physical_training_audit_v1"
CHECKPOINT_METADATA_SCHEMA = "duca_protected_physical_checkpoint_metadata_v1"
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = "duca_protected_physical_checkpoint_sidecar_v1"
DUCA_TRAINING_AUDIT_FILENAME = "duca_protected_physical_training_audit.json"
VARIANTS = (
    "exact_uniform",
    "transition_no_bridge",
    "protected_e2e",
    "protected_e2e_bridge025",
    "protected_e2e_homotopy025",
    "protected_e2e_uni_companion",
    "protected_e2e_rho001",
)
HOMOTOPY_VARIANT = "protected_e2e_homotopy025"
_UPDATE_KEYS = (
    "attempted_batches",
    "optimizer_attempts",
    "successful_optimizer_updates",
    "amp_skipped_attempts",
    "replayed_batches",
    "replay_exhaustions",
    "scheduler_updates",
    "ema_updates",
    "duca_schedule_updates",
    "forced_amp_overflow_attempts",
    "max_amp_retries_observed",
)

canonical_sha256 = duca_p0_training.canonical_sha256
sha256_file = duca_p0_training.sha256_file
atomic_write_json = duca_p0_training.atomic_write_json
capture_global_rng_state = duca_p0_training.capture_global_rng_state
restore_global_rng_state = duca_p0_training.restore_global_rng_state


def formal_training_contract(cfg) -> dict[str, Any] | None:
    workflow = cfg.workflow
    if str(workflow.get("formal_protocol", "")) != FORMAL_PROTOCOL:
        return None
    if int(workflow.end_epoch) != 60:
        raise ValueError("protected physical DUCA is frozen to 60 epochs")
    if int(workflow.checkpoint_interval) != 5:
        raise ValueError("protected physical DUCA checkpoint interval must be five")
    if int(workflow.val_eval_interval) >= 0:
        raise ValueError("protected physical DUCA forbids intermediate test evaluation")
    if cfg.dataset.val is not None:
        raise ValueError("protected physical DUCA forbids a validation dataset")
    if not bool(workflow.get("seal_eval_dataloaders_during_training", False)):
        raise ValueError("protected physical DUCA must seal evaluation dataloaders")
    if not bool(workflow.get("derive_train_loader_contract", False)):
        raise ValueError("protected physical DUCA must derive loader exposure")
    if workflow.get("max_train_iters", None) is not None:
        raise ValueError("protected physical DUCA cannot truncate an epoch")
    if int(workflow.get("force_amp_overflow_attempts", 0)) != 0:
        raise ValueError("formal training cannot inject AMP overflow")
    if int(workflow.max_amp_retries_per_batch) <= 0:
        raise ValueError("formal training requires bounded AMP replay")
    if not bool(workflow.fail_on_amp_replay_exhaustion):
        raise ValueError("formal training must fail on AMP replay exhaustion")
    if not bool(workflow.require_finite_train_loss):
        raise ValueError("formal training must reject non-finite loss")
    if int(workflow.primary_checkpoint_epoch) != 59:
        raise ValueError("primary checkpoint must be terminal epoch 59")
    if str(workflow.primary_checkpoint_state_key) != "state_dict_ema":
        raise ValueError("primary checkpoint must use terminal EMA")
    if str(workflow.checkpoint_criterion) != "terminal_epoch_59_state_dict_ema":
        raise ValueError("terminal checkpoint criterion drift")
    variant = str(cfg.model.frame_selector.arm)
    if variant not in VARIANTS:
        raise ValueError(f"unsupported protected physical variant: {variant}")
    selector_schedule_enabled = variant == HOMOTOPY_VARIANT
    homotopy_total_steps = int(
        cfg.model.frame_selector.get("homotopy_total_steps", 0)
    )
    if selector_schedule_enabled and homotopy_total_steps <= 0:
        raise ValueError("homotopy arm requires a positive total-step contract")
    if not selector_schedule_enabled and homotopy_total_steps != 0:
        raise ValueError("schedule-free arm cannot declare homotopy_total_steps")
    return {
        "protocol": FORMAL_PROTOCOL,
        "variant": variant,
        "expected_train_batches_per_epoch": None,
        "expected_successful_optimizer_updates": None,
        "selector_schedule_enabled": selector_schedule_enabled,
        "expected_selector_schedule_updates": None,
        "homotopy_total_steps": homotopy_total_steps,
        "end_epoch": 60,
        "max_amp_retries_per_batch": int(workflow.max_amp_retries_per_batch),
        "primary_checkpoint_epoch": 59,
        "primary_checkpoint_state_key": "state_dict_ema",
        "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
        "checkpoint_interval": 5,
        "train_loader_contract": None,
    }


def _dataset_identity(dataset) -> dict[str, Any]:
    rows = []
    for index, item in enumerate(dataset.data_list):
        rows.append(
            {
                "index": int(index),
                "video_name": str(item[0]),
            }
        )
    return {
        "dataset_class": dataset.__class__.__name__,
        "dataset_length": int(len(dataset)),
        "subset_name": str(dataset.subset_name),
        "ann_file": str(Path(dataset.ann_file).expanduser().resolve()),
        "ann_file_sha256": sha256_file(dataset.ann_file),
        "class_map_path": str(Path(dataset.class_map_path).expanduser().resolve())
        if hasattr(dataset, "class_map_path")
        else None,
        "sample_identity_sha256": canonical_sha256(rows),
        "stateless_seed": int(dataset.stateless_seed),
    }


def derive_train_loader_contract(
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> dict[str, Any]:
    if train_dataset.__class__.__name__ != "DucaStatelessThumosPaddingDataset":
        raise RuntimeError("formal DUCA requires the stateless THUMOS dataset")
    if int(train_dataset.stateless_seed) != 3407:
        raise RuntimeError("formal DUCA stateless data seed must be 3407")
    if int(world_size) != 1:
        raise RuntimeError("formal DUCA is frozen to one Slurm process")
    loader_length = int(len(train_loader))
    if loader_length <= 0:
        raise RuntimeError("formal DUCA train loader is empty")
    dataset_identity = _dataset_identity(train_dataset)
    loader_manifest = {
        "schema": "duca_protected_physical_train_loader_contract_v1",
        "dataset": dataset_identity,
        "loader_length": loader_length,
        "batch_size": int(cfg.solver.train.batch_size),
        "num_workers": int(cfg.solver.train.num_workers),
        "world_size": int(world_size),
        "drop_last": True,
        "shuffle": True,
        "sampler_class": train_loader.sampler.__class__.__name__,
        "dataset_config_sha256": canonical_sha256(cfg.dataset.train.to_dict()),
    }
    loader_manifest["contract_sha256"] = canonical_sha256(loader_manifest)
    return loader_manifest


def bind_train_loader_contract(
    contract: Mapping[str, Any],
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader_manifest = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=train_dataset,
        train_loader=train_loader,
        world_size=world_size,
    )
    expected_manifest = _load_bound_json(
        os.environ.get("DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON"),
        os.environ.get("DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256"),
        "duca_protected_physical_protocol_manifest_v1",
        "P0 protocol manifest",
    )
    expected_loader = expected_manifest.get("train_loader_contract")
    if not isinstance(expected_loader, Mapping):
        raise RuntimeError("P0 manifest lacks train-loader contract")
    if dict(expected_loader) != loader_manifest:
        raise RuntimeError("runtime train-loader contract differs from P0")

    loader_length = int(loader_manifest["loader_length"])
    expected_updates = loader_length * 60
    bound = dict(contract)
    bound.setdefault("selector_schedule_enabled", False)
    bound.setdefault("homotopy_total_steps", 0)
    bound["expected_train_batches_per_epoch"] = loader_length
    bound["expected_successful_optimizer_updates"] = expected_updates
    bound["expected_selector_schedule_updates"] = (
        expected_updates if bool(bound["selector_schedule_enabled"]) else 0
    )
    if bool(bound["selector_schedule_enabled"]):
        if int(bound["homotopy_total_steps"]) != expected_updates:
            raise RuntimeError(
                "homotopy total steps differ from the frozen optimizer exposure"
            )
    bound["train_loader_contract"] = loader_manifest
    return bound, loader_manifest


def assert_safe_cfg_options(
    cfg_options: Mapping[str, Any] | None,
    *,
    entrypoint: str,
) -> None:
    if not cfg_options:
        return
    allowed = {
        "work_dir": None,
        "model.backbone.custom.pretrain": None,
    }
    if entrypoint == "tools/test.py":
        allowed.update(
            {
                "post_processing.save_dict": True,
                "inference.load_from_raw_predictions": False,
            }
        )

    def flatten(node: Mapping[str, Any], prefix: str = ""):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                yield from flatten(value, path)
            else:
                yield path, value

    rejected = []
    for path, value in flatten(cfg_options):
        if path not in allowed:
            rejected.append(path)
            continue
        expected = allowed[path]
        if expected is not None and value is not expected:
            rejected.append(path)
    if rejected:
        raise RuntimeError(
            f"{entrypoint} rejected protected-DUCA cfg overrides: "
            + ", ".join(sorted(rejected))
        )


def _load_bound_json(
    path: str | None,
    expected_sha256: str | None,
    schema: str,
    label: str,
) -> dict[str, Any]:
    if not path:
        raise RuntimeError(f"{label} path is missing")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise RuntimeError(f"{label} schema mismatch")
    return payload


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
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"invalid protected physical variant: {variant}")
    if int(seed) != 3407:
        raise ValueError("protected physical DUCA seed must be 3407")
    if resolved_config_sha256 != os.environ.get("DUCA_RESOLVED_CONFIG_SHA256"):
        raise RuntimeError("resolved config hash differs from suite manifest")
    protocol = _load_bound_json(
        os.environ.get("DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON"),
        os.environ.get("DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256"),
        "duca_protected_physical_protocol_manifest_v1",
        "P0 protocol manifest",
    )
    authorization = _load_bound_json(
        os.environ.get("DUCA_PROTECTED_AUTHORIZATION_JSON"),
        os.environ.get("DUCA_PROTECTED_AUTHORIZATION_SHA256"),
        "duca_protected_physical_authorization_v1",
        "P0-P3 authorization",
    )
    if authorization.get("ok") is not True:
        raise RuntimeError("P0-P3 authorization did not pass")
    if (
        authorization.get("authorized_scope", {}).get("official60_four_arm_training")
        is not True
        or authorization.get("paper_claim_allowed") is not False
    ):
        raise RuntimeError("authorization does not unlock official-60 training")
    if (
        variant
        in {
            "protected_e2e_bridge025",
            "protected_e2e_uni_companion",
        }
        and authorization.get("authorized_scope", {}).get(
            "official60_uni_companion_training"
        )
        is not True
    ):
        raise RuntimeError(
            "authorization does not permit the Uni companion optimization suite"
        )
    if (
        variant == HOMOTOPY_VARIANT
        and authorization.get("authorized_scope", {}).get(
            "official60_homotopy_training"
        )
        is not True
    ):
        raise RuntimeError("authorization does not permit homotopy training")
    if protocol.get("git_commit") != git_commit:
        raise RuntimeError("P0 protocol commit drift")
    if authorization.get("git_commit") != git_commit:
        raise RuntimeError("authorization commit drift")
    if authorization.get("protocol_manifest_sha256") != os.environ.get(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256"
    ):
        raise RuntimeError("authorization is bound to another P0 manifest")
    authorization_config_hashes = authorization.get("config_hashes")
    if not isinstance(authorization_config_hashes, Mapping):
        raise RuntimeError("authorization lacks config-hash bindings")
    if authorization_config_hashes.get(variant) != source_config_sha256:
        raise RuntimeError("runtime source config differs from authorization")
    protocol_arm = protocol.get("configs", {}).get("arms", {}).get(variant)
    if not isinstance(protocol_arm, Mapping):
        raise RuntimeError(f"P0 manifest has no config evidence for {variant}")
    if protocol_arm.get("source_sha256") != source_config_sha256:
        raise RuntimeError("runtime source config differs from P0")
    if protocol_arm.get("resolved_sha256") != resolved_config_sha256:
        raise RuntimeError("runtime resolved config differs from P0")
    pretrain = Path(runtime_pretrain_path).expanduser().resolve()
    annotation = Path(evaluation_annotation_path).expanduser().resolve()
    class_map = Path(evaluation_class_map_path).expanduser().resolve()
    if not pretrain.is_file():
        raise RuntimeError("runtime VideoMAE-S pretrain is missing")
    if not annotation.is_file():
        raise RuntimeError("runtime evaluation annotation is missing")
    if not class_map.is_file():
        raise RuntimeError("runtime evaluation class map is missing")
    pretrain_sha256 = sha256_file(pretrain)
    annotation_sha256 = sha256_file(annotation)
    class_map_sha256 = sha256_file(class_map)
    if protocol.get("videomae_pretrain", {}).get("sha256") != pretrain_sha256:
        raise RuntimeError("runtime VideoMAE-S pretrain differs from P0")
    protocol_data = protocol.get("data_files", {})
    if protocol_data.get("annotation_sha256") != annotation_sha256:
        raise RuntimeError("runtime evaluation annotation differs from P0")
    if protocol_data.get("class_map_sha256") != class_map_sha256:
        raise RuntimeError("runtime evaluation class map differs from P0")
    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "protocol_manifest_sha256": str(
            os.environ["DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256"]
        ),
        "authorization_sha256": str(os.environ["DUCA_PROTECTED_AUTHORIZATION_SHA256"]),
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha256,
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": class_map_sha256,
        "evaluation_config_sha256": canonical_sha256(dict(evaluation_config)),
    }
    return bindings


def new_update_audit() -> dict[str, int]:
    return {key: 0 for key in _UPDATE_KEYS}


def selector_schedule_step(model) -> int:
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    if selector is None or selector.selector_variant != "protected_e2e_physical":
        raise RuntimeError("formal model lacks protected physical selector")
    arm = str(getattr(selector, "arm", ""))
    if arm == HOMOTOPY_VARIANT:
        schedule_step = getattr(selector, "schedule_step", None)
        if schedule_step is None or getattr(schedule_step, "numel", lambda: 0)() != 1:
            raise RuntimeError("homotopy selector lacks its persistent schedule step")
        return int(schedule_step.detach().item())
    if hasattr(selector, "schedule_step"):
        raise RuntimeError("schedule-free selector exposes a hidden schedule buffer")
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
    if int(train_batches_per_epoch) != expected_batches:
        raise RuntimeError("formal loader exposure changed")
    expected_updates = (int(epoch) + 1) * expected_batches
    successful = int(update_audit["successful_optimizer_updates"])
    if successful != expected_updates:
        raise RuntimeError("successful optimizer update count mismatch")
    if int(update_audit["attempted_batches"]) != expected_updates:
        raise RuntimeError("consumed batch count mismatch")
    skipped = int(update_audit["amp_skipped_attempts"])
    if int(update_audit["optimizer_attempts"]) != successful + skipped:
        raise RuntimeError("optimizer attempt accounting mismatch")
    if int(update_audit["replay_exhaustions"]) != 0:
        raise RuntimeError("AMP replay exhausted")
    if int(update_audit["scheduler_updates"]) != successful:
        raise RuntimeError("scheduler exposure mismatch")
    if int(update_audit["ema_updates"]) != (successful if uses_ema else 0):
        raise RuntimeError("EMA exposure mismatch")
    schedule_enabled = bool(contract["selector_schedule_enabled"])
    full_selector_updates = int(contract["expected_selector_schedule_updates"])
    full_optimizer_updates = int(contract["expected_successful_optimizer_updates"])
    if schedule_enabled:
        if full_selector_updates != full_optimizer_updates:
            raise RuntimeError(
                "homotopy full-run schedule exposure differs from optimizer exposure"
            )
        expected_selector_updates = expected_updates
    else:
        if full_selector_updates != 0:
            raise RuntimeError("schedule-free arm declares selector schedule exposure")
        expected_selector_updates = 0
    if (
        int(update_audit["duca_schedule_updates"]) != expected_selector_updates
        or int(selector_step) != expected_selector_updates
    ):
        label = (
            "homotopy selector schedule exposure mismatch"
            if schedule_enabled
            else "schedule-free selector advanced a hidden schedule"
        )
        raise RuntimeError(label)
    if int(scheduler_last_epoch) != successful:
        raise RuntimeError("scheduler state differs from successful updates")
    if int(update_audit["max_amp_retries_observed"]) > int(
        contract["max_amp_retries_per_batch"]
    ):
        raise RuntimeError("AMP retries exceeded the frozen limit")


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
    if len(epoch_records) != int(epoch) + 1:
        raise RuntimeError("epoch records are incomplete")
    payload = {
        "schema": TRAINING_AUDIT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        **dict(bindings),
        "train_loader_contract": contract["train_loader_contract"],
        "checkpoint_criterion": contract["checkpoint_criterion"],
        "primary_checkpoint_epoch": int(contract["primary_checkpoint_epoch"]),
        "primary_checkpoint_state_key": contract["primary_checkpoint_state_key"],
        "expected_train_batches_per_epoch": int(
            contract["expected_train_batches_per_epoch"]
        ),
        "expected_successful_optimizer_updates": int(
            contract["expected_successful_optimizer_updates"]
        ),
        "selector_schedule_enabled": bool(contract["selector_schedule_enabled"]),
        "expected_selector_schedule_updates": int(
            contract["expected_selector_schedule_updates"]
        ),
        "homotopy_total_steps": int(contract["homotopy_total_steps"]),
        "last_completed_epoch": int(epoch),
        "epochs_completed": int(epoch) + 1,
        "update_audit": {key: int(value) for key, value in update_audit.items()},
        "scheduler_last_epoch": int(scheduler_last_epoch),
        "selector_schedule_step": int(selector_step),
        "grad_scaler_scale": (None if scaler_scale is None else float(scaler_scale)),
        "epoch_records": [dict(item) for item in epoch_records],
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


def build_checkpoint_metadata(
    training_audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_METADATA_SCHEMA,
        "training_audit": dict(training_audit),
    }
    payload["metadata_sha256"] = canonical_sha256(payload)
    return payload


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("resume checkpoint lacks protected metadata")
    if metadata.get("schema") != CHECKPOINT_METADATA_SCHEMA:
        raise RuntimeError("resume metadata schema mismatch")
    expected_hash = metadata.get("metadata_sha256")
    unsigned = dict(metadata)
    unsigned.pop("metadata_sha256", None)
    if expected_hash != canonical_sha256(unsigned):
        raise RuntimeError("resume metadata hash mismatch")
    audit = metadata.get("training_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError("resume checkpoint lacks training audit")
    audit_hash = audit.get("audit_sha256")
    unsigned_audit = dict(audit)
    unsigned_audit.pop("audit_sha256", None)
    if audit_hash != canonical_sha256(unsigned_audit):
        raise RuntimeError("resume training-audit hash mismatch")
    for key, expected in bindings.items():
        if key != "slurm_job_id" and audit.get(key) != expected:
            raise RuntimeError(f"resume binding mismatch: {key}")
    if audit.get("train_loader_contract") != contract["train_loader_contract"]:
        raise RuntimeError("resume loader contract mismatch")
    counters = audit.get("update_audit")
    records = audit.get("epoch_records")
    if not isinstance(counters, Mapping) or not isinstance(records, list):
        raise RuntimeError("resume training state is incomplete")
    restored = {str(key): int(value) for key, value in counters.items()}
    if set(restored) != set(_UPDATE_KEYS):
        raise RuntimeError("resume update counters are incomplete")
    return restored, [dict(item) for item in records]


__all__ = [
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "FORMAL_PROTOCOL",
    "HOMOTOPY_VARIANT",
    "atomic_write_json",
    "bind_train_loader_contract",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "derive_train_loader_contract",
    "formal_training_contract",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "validate_update_state",
]
