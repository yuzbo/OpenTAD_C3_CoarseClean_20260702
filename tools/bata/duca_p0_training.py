from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping


DUCA_P0_TRAINING_AUDIT_SCHEMA = "duca_p0_training_audit_v2"
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = "duca_p0_checkpoint_metadata_v2"
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = "duca_p0_checkpoint_sidecar_v2"
DUCA_P0_VARIANTS = (
    "uniform",
    "direct",
    "transition_beta0",
    "transition_counterfactual",
)
_UPDATE_AUDIT_KEYS = (
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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def formal_training_contract(
    cfg,
    *,
    expected_checkpoint_criterion: str | None = None,
) -> dict[str, Any] | None:
    workflow = cfg.workflow
    if not bool(workflow.get("formal_successful_update_contract", False)):
        return None
    contract = {
        "expected_train_batches_per_epoch": int(
            workflow.expected_train_batches_per_epoch
        ),
        "expected_successful_optimizer_updates": int(
            workflow.expected_successful_optimizer_updates
        ),
        "end_epoch": int(workflow.end_epoch),
        "max_amp_retries_per_batch": int(workflow.max_amp_retries_per_batch),
        "primary_checkpoint_epoch": int(workflow.primary_checkpoint_epoch),
        "primary_checkpoint_state_key": str(workflow.primary_checkpoint_state_key),
        "checkpoint_criterion": str(workflow.checkpoint_criterion),
        "checkpoint_interval": int(workflow.checkpoint_interval),
    }
    if contract["expected_train_batches_per_epoch"] <= 0:
        raise ValueError("formal DUCA train batches per epoch must be positive")
    if contract["end_epoch"] <= 0:
        raise ValueError("formal DUCA end_epoch must be positive")
    expected_total = (
        contract["expected_train_batches_per_epoch"] * contract["end_epoch"]
    )
    if contract["expected_successful_optimizer_updates"] != expected_total:
        raise ValueError("formal DUCA successful-update total is inconsistent")
    if contract["max_amp_retries_per_batch"] <= 0:
        raise ValueError("formal DUCA requires a positive AMP replay limit")
    if contract["primary_checkpoint_epoch"] != contract["end_epoch"] - 1:
        raise ValueError("formal DUCA primary checkpoint must be the terminal epoch")
    if contract["primary_checkpoint_state_key"] != "state_dict_ema":
        raise ValueError("formal DUCA primary state must be state_dict_ema")
    expected_checkpoint_criterion = (
        f"terminal_epoch_{contract['primary_checkpoint_epoch']}_"
        f"{contract['primary_checkpoint_state_key']}"
        if expected_checkpoint_criterion is None
        else str(expected_checkpoint_criterion)
    )
    if contract["checkpoint_criterion"] != expected_checkpoint_criterion:
        raise ValueError("formal DUCA checkpoint criterion is not frozen")
    if contract["checkpoint_interval"] != 5:
        raise ValueError("formal DUCA checkpoint interval must remain five epochs")
    if not bool(workflow.get("fail_on_amp_replay_exhaustion", False)):
        raise ValueError("formal DUCA must fail closed when AMP replay is exhausted")
    if not bool(workflow.get("require_finite_train_loss", False)):
        raise ValueError("formal DUCA must reject non-finite train losses")
    if workflow.get("max_train_iters", None) is not None:
        raise ValueError("formal DUCA cannot truncate an epoch with max_train_iters")
    if int(workflow.get("force_amp_overflow_attempts", 0)) != 0:
        raise ValueError("formal DUCA training cannot inject a synthetic AMP overflow")
    val_eval_interval = int(workflow.get("val_eval_interval", 0))
    intermediate_validation = val_eval_interval > 0
    if intermediate_validation:
        if val_eval_interval != 5:
            raise ValueError("formal DUCA performance-curve evaluation must run every five epochs")
        if int(workflow.get("val_eval_interval_anchor_epoch", -1)) != 5:
            raise ValueError("formal DUCA performance-curve evaluation must anchor at epoch five")
        if int(workflow.get("val_start_epoch", -1)) != 4:
            raise ValueError("formal DUCA performance-curve evaluation must start after epoch four")
        if str(workflow.get("intermediate_validation_role", "")) != (
            "full_curve_and_best_validation_checkpoint"
        ):
            raise ValueError("formal DUCA intermediate validation role is not explicit")
        if workflow.get("intermediate_validation_selects_checkpoint", None) is not True:
            raise ValueError("formal DUCA performance curve must retain the best validation checkpoint")
    elif workflow.get("intermediate_validation_selects_checkpoint", False):
        raise ValueError("checkpoint selection requires an enabled intermediate validation schedule")
    contract["intermediate_validation"] = intermediate_validation
    contract["intermediate_validation_interval"] = val_eval_interval
    contract["intermediate_validation_role"] = str(
        workflow.get("intermediate_validation_role", "disabled")
    )
    return contract


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
) -> dict[str, Any]:
    from tools.bata.duca_p0_evaluation import evaluation_config_sha256

    if variant not in DUCA_P0_VARIANTS:
        raise ValueError(f"invalid formal DUCA variant: {variant}")
    expected_resolved = os.environ.get("DUCA_RESOLVED_CONFIG_SHA256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError("formal DUCA resolved config hash differs from suite manifest")
    canonical_env_file = os.environ.get("DUCA_CANONICAL_ENV_FILE")
    canonical_env_sha256 = os.environ.get("DUCA_CANONICAL_ENV_SHA256")
    if _optional_file_sha256(canonical_env_file) != canonical_env_sha256:
        raise ValueError("formal DUCA canonical environment file hash mismatch")
    core_gate_sha256 = _validated_gate_sha256(
        os.environ.get("DUCA_CORE_GATE_JSON"), git_commit
    )
    ddp_pilot_sha256 = _validated_pilot_sha256(
        os.environ.get("DUCA_DDP_PILOT_JSON"), git_commit
    )
    annotation_sha256 = _bound_data_file_sha256(
        evaluation_annotation_path,
        os.environ.get("DUCA_EVALUATION_ANNOTATION_SHA256"),
        "evaluation annotation",
    )
    class_map_sha256 = _bound_data_file_sha256(
        evaluation_class_map_path,
        os.environ.get("DUCA_EVALUATION_CLASS_MAP_SHA256"),
        "evaluation class map",
    )
    eval_config_sha256 = evaluation_config_sha256(evaluation_config)
    if eval_config_sha256 != os.environ.get("DUCA_EVALUATION_CONFIG_SHA256"):
        raise ValueError("formal DUCA evaluation config differs from suite manifest")
    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "shared_protocol_sha256": os.environ.get("DUCA_SHARED_PROTOCOL_SHA256"),
        "variant_contract_sha256": os.environ.get("DUCA_VARIANT_CONTRACT_SHA256"),
        "core_gate_sha256": core_gate_sha256,
        "ddp_pilot_sha256": ddp_pilot_sha256,
        "canonical_env_sha256": canonical_env_sha256,
        "evaluation_annotation_path": str(
            Path(evaluation_annotation_path).expanduser().resolve()
        ),
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_class_map_path": str(
            Path(evaluation_class_map_path).expanduser().resolve()
        ),
        "evaluation_class_map_sha256": class_map_sha256,
        "evaluation_config_sha256": eval_config_sha256,
    }
    if re.fullmatch(r"[0-9a-f]{40}", bindings["git_commit"]) is None:
        raise ValueError("formal DUCA runtime binding has an invalid git commit")
    for key in (
        "source_config_sha256",
        "resolved_config_sha256",
        "runtime_config_sha256",
        "shared_protocol_sha256",
        "variant_contract_sha256",
        "core_gate_sha256",
        "ddp_pilot_sha256",
        "canonical_env_sha256",
        "evaluation_annotation_sha256",
        "evaluation_class_map_sha256",
        "evaluation_config_sha256",
    ):
        value = bindings.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"formal DUCA runtime binding is missing {key}")
    return bindings


def _optional_file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"formal DUCA bound artifact is missing: {resolved}")
    return sha256_file(resolved)


def _bound_data_file_sha256(
    path: str | Path, expected_sha256: str | None, label: str
) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"formal DUCA {label} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(f"formal DUCA {label} hash differs from suite manifest")
    return actual


def _validated_gate_sha256(path: str | None, git_commit: str) -> str:
    payload, artifact_sha256 = _load_bound_json(path, "formal core gate")
    if payload.get("formal_proof_ok") is not True or payload.get("git_commit") != git_commit:
        raise ValueError("formal DUCA core gate is stale or did not pass")
    return artifact_sha256


def _validated_pilot_sha256(path: str | None, git_commit: str) -> str:
    payload, artifact_sha256 = _load_bound_json(path, "DDP pilot")
    if (
        payload.get("schema_version") != "duca_p0_ddp_pilot_suite_v1"
        or payload.get("ok") is not True
        or payload.get("git_commit") != git_commit
    ):
        raise ValueError("formal DUCA DDP pilot is stale or did not pass")
    return artifact_sha256


def _load_bound_json(path: str | None, label: str) -> tuple[dict[str, Any], str]:
    if not path:
        raise ValueError(f"formal DUCA {label} path is missing")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"formal DUCA {label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"formal DUCA {label} is not a JSON object")
    return payload, sha256_file(resolved)


def new_update_audit() -> dict[str, int]:
    return {key: 0 for key in _UPDATE_AUDIT_KEYS}


def capture_global_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_global_rng_state(snapshot: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(snapshot) != required:
        raise RuntimeError("formal DUCA checkpoint RNG state is incomplete")
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch_cpu_state = snapshot["torch_cpu"]
    if hasattr(torch_cpu_state, "cpu"):
        torch_cpu_state = torch_cpu_state.cpu()
    torch.set_rng_state(torch_cpu_state)
    cuda_states = [
        state.cpu() if hasattr(state, "cpu") else state
        for state in snapshot["torch_cuda"]
    ]
    torch.cuda.set_rng_state_all(cuda_states)


def selector_schedule_step(model) -> int:
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    step = getattr(selector, "_loss_weight_schedule_step", None)
    if step is None:
        raise RuntimeError("formal DUCA model has no selector schedule buffer")
    return int(step.detach().cpu().item())


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
        raise RuntimeError(
            f"formal DUCA loader has {train_batches_per_epoch} batches, expected {expected_batches}"
        )
    expected_updates = (int(epoch) + 1) * expected_batches
    successful = int(update_audit.get("successful_optimizer_updates", -1))
    if successful != expected_updates:
        raise RuntimeError(
            f"formal DUCA epoch {epoch} has {successful} successful updates, expected {expected_updates}"
        )
    attempted_batches = int(update_audit.get("attempted_batches", -1))
    if attempted_batches != expected_updates:
        raise RuntimeError("formal DUCA consumed-batch count differs from successful updates")
    skipped = int(update_audit.get("amp_skipped_attempts", -1))
    attempts = int(update_audit.get("optimizer_attempts", -1))
    if skipped < 0 or attempts != successful + skipped:
        raise RuntimeError("formal DUCA optimizer-attempt accounting is inconsistent")
    if int(update_audit.get("replay_exhaustions", -1)) != 0:
        raise RuntimeError("formal DUCA exhausted an AMP replay")
    if int(update_audit.get("scheduler_updates", -1)) != successful:
        raise RuntimeError("formal DUCA scheduler exposure differs from optimizer exposure")
    expected_ema = successful if uses_ema else 0
    if int(update_audit.get("ema_updates", -1)) != expected_ema:
        raise RuntimeError("formal DUCA EMA exposure differs from optimizer exposure")
    if int(update_audit.get("duca_schedule_updates", -1)) != successful:
        raise RuntimeError("formal DUCA selector schedule exposure differs from optimizer exposure")
    if int(scheduler_last_epoch) != successful:
        raise RuntimeError("formal DUCA scheduler state does not match successful updates")
    if int(selector_step) != successful:
        raise RuntimeError("formal DUCA selector schedule buffer does not match successful updates")
    if int(update_audit.get("max_amp_retries_observed", -1)) > int(
        contract["max_amp_retries_per_batch"]
    ):
        raise RuntimeError("formal DUCA observed more AMP retries than the frozen limit")


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
        raise RuntimeError("formal DUCA epoch-record count is incomplete")
    if [int(item.get("epoch", -1)) for item in epoch_records] != list(
        range(int(epoch) + 1)
    ):
        raise RuntimeError("formal DUCA epoch records are not contiguous")
    payload = {
        "schema_version": DUCA_P0_TRAINING_AUDIT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        **dict(bindings),
        "checkpoint_criterion": contract["checkpoint_criterion"],
        "primary_checkpoint_epoch": int(contract["primary_checkpoint_epoch"]),
        "primary_checkpoint_state_key": contract["primary_checkpoint_state_key"],
        "expected_train_batches_per_epoch": int(
            contract["expected_train_batches_per_epoch"]
        ),
        "expected_successful_optimizer_updates": int(
            contract["expected_successful_optimizer_updates"]
        ),
        "last_completed_epoch": int(epoch),
        "epochs_completed": int(epoch) + 1,
        "train_batches_per_epoch": int(train_batches_per_epoch),
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


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    metadata = checkpoint.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("formal DUCA resume checkpoint lacks experiment metadata")
    if metadata.get("schema_version") != DUCA_P0_CHECKPOINT_METADATA_SCHEMA:
        raise RuntimeError("formal DUCA resume checkpoint metadata schema mismatch")
    expected_hash = metadata.get("metadata_sha256")
    unsigned = dict(metadata)
    unsigned.pop("metadata_sha256", None)
    if expected_hash != canonical_sha256(unsigned):
        raise RuntimeError("formal DUCA resume checkpoint metadata hash mismatch")
    audit = metadata.get("training_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError("formal DUCA resume checkpoint lacks training audit")
    audit_hash = audit.get("audit_sha256")
    unsigned_audit = dict(audit)
    unsigned_audit.pop("audit_sha256", None)
    if audit_hash != canonical_sha256(unsigned_audit):
        raise RuntimeError("formal DUCA resume training audit hash mismatch")
    for key, expected in bindings.items():
        if key == "slurm_job_id":
            continue
        if audit.get(key) != expected:
            raise RuntimeError(f"formal DUCA resume binding mismatch: {key}")
    if audit.get("checkpoint_criterion") != contract["checkpoint_criterion"]:
        raise RuntimeError("formal DUCA resume checkpoint criterion mismatch")
    counters = audit.get("update_audit")
    records = audit.get("epoch_records")
    if not isinstance(counters, Mapping) or not isinstance(records, list):
        raise RuntimeError("formal DUCA resume training state is incomplete")
    restored = {str(key): int(value) for key, value in counters.items()}
    if set(restored) != set(_UPDATE_AUDIT_KEYS):
        raise RuntimeError("formal DUCA resume update counters are incomplete")
    return (restored, [dict(x) for x in records])


__all__ = [
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_P0_TRAINING_AUDIT_SCHEMA",
    "atomic_write_json",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "canonical_sha256",
    "formal_training_contract",
    "new_update_audit",
    "restore_training_state",
    "restore_global_rng_state",
    "selector_schedule_step",
    "sha256_file",
    "validate_update_state",
]
