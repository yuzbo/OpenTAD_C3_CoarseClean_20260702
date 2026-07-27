from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from tools.bata import duca_p0_training


FORMAL_PROTOCOLS = {
    "duca_rime_uniform_control_v1",
    "duca_rime_physical_dynamic_k_v1",
}
TRAIN_ARMS = {
    "U-fixed",
    "F-bound",
    "D-shuffle",
    "D-no-risk",
    "AdapTok-TAD",
    "RIME-full",
}
DUCA_TRAINING_AUDIT_FILENAME = "duca_rime_training_audit.json"
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = duca_p0_training.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = duca_p0_training.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA

atomic_write_json = duca_p0_training.atomic_write_json
build_checkpoint_metadata = duca_p0_training.build_checkpoint_metadata
build_training_audit = duca_p0_training.build_training_audit
capture_global_rng_state = duca_p0_training.capture_global_rng_state
new_update_audit = duca_p0_training.new_update_audit
restore_global_rng_state = duca_p0_training.restore_global_rng_state
restore_training_state = duca_p0_training.restore_training_state
selector_schedule_step = duca_p0_training.selector_schedule_step
validate_update_state = duca_p0_training.validate_update_state


def is_formal_protocol(value: str) -> bool:
    return str(value) in FORMAL_PROTOCOLS


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _bound_file(path: str | Path | None, expected: str | None, label: str) -> tuple[str, str]:
    if not path:
        raise ValueError(f"RIME {label} path is missing")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = _sha256_file(resolved)
    if not expected or actual != str(expected).lower():
        raise ValueError(f"RIME {label} SHA-256 drift")
    return str(resolved), actual


def _optional_bound_file(
    path: str | Path | None,
    expected: str | None,
    label: str,
) -> tuple[str | None, str | None]:
    if not path and not expected:
        return None, None
    return _bound_file(path, expected, label)


def formal_training_contract(cfg) -> dict[str, Any] | None:
    formal_protocol = str(cfg.workflow.get("formal_protocol", ""))
    if not is_formal_protocol(formal_protocol):
        return None
    contract = duca_p0_training.formal_training_contract(cfg)
    if contract is None:
        raise ValueError("RIME formal training contract was not activated")
    arm = str(cfg.duca_rime_variant.arm)
    if arm not in TRAIN_ARMS:
        raise ValueError(f"RIME config has an unregistered train arm: {arm}")
    if (
        int(cfg.solver.train.batch_size) != 1
        or int(contract["expected_successful_optimizer_updates"]) != 6000
        or int(contract["end_epoch"]) * int(contract["expected_train_batches_per_epoch"])
        != 6000
    ):
        raise ValueError("RIME Phase-3 requires batch_size=1 and exactly 6000 updates")
    if arm != "U-fixed":
        rime_contract = cfg.duca_rime_contract
        if (
            rime_contract.task != "offline_temporal_action_detection"
            or rime_contract.online_tad is not False
            or rime_contract.pad_to_kmax is not False
            or int(rime_contract.execution_quantum) != 16
            or rime_contract.official_final_subset_consumed is not False
        ):
            raise ValueError("RIME dynamic exact-K/physical-time contract drift")
    contract = dict(contract)
    contract["formal_protocol"] = formal_protocol
    contract["rime_arm"] = arm
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
    runtime_pretrain_path: str | Path,
) -> dict[str, Any]:
    from tools.bata.duca_p0_evaluation import evaluation_config_sha256

    if variant not in TRAIN_ARMS:
        raise ValueError(f"invalid RIME train arm: {variant}")
    if re.fullmatch(r"[0-9a-f]{40}", str(git_commit)) is None:
        raise ValueError("RIME runtime binding requires an exact Git commit")
    if int(seed) != 3407:
        raise ValueError("Phase-3 development training seed is frozen to 3407")
    expected_resolved = os.environ.get("DUCA_RESOLVED_CONFIG_SHA256")
    if expected_resolved != str(resolved_config_sha256):
        raise ValueError("RIME resolved config differs from the sealed launch")

    phase2_path, phase2_sha = _bound_file(
        os.environ.get("DUCA_RIME_PHASE2_RECEIPT"),
        os.environ.get("DUCA_RIME_PHASE2_RECEIPT_SHA256"),
        "Phase-2 receipt",
    )
    phase2 = json.loads(Path(phase2_path).read_text(encoding="utf-8"))
    if (
        phase2.get("schema_version") != "duca_rime_stage_receipt_v1"
        or phase2.get("phase") != "phase2"
        or phase2.get("gate_pass") is not True
        or phase2.get("phase3_training_authorized") is not True
        or phase2.get("official_final_subset_consumed") is not False
        or phase2.get("git_commit") != str(git_commit)
    ):
        raise ValueError("Phase-2 receipt does not authorize RIME training")
    exposure_path, exposure_sha = _bound_file(
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_JSON"),
        os.environ.get("DUCA_RIME_TRAINING_EXPOSURE_SHA256"),
        "training exposure",
    )
    exposure = json.loads(Path(exposure_path).read_text(encoding="utf-8"))
    if (
        exposure.get("schema_version") != "duca_rime_phase3_training_exposure_v1"
        or int(exposure.get("successful_detector_updates", -1)) != 6000
        or exposure.get("split_assignment_sha256")
        != phase2.get("split_assignment_sha256")
        or exposure.get("official_final_subset_consumed") is not False
    ):
        raise ValueError("RIME training exposure artifact is invalid")
    pretrain_path, pretrain_sha = _bound_file(
        runtime_pretrain_path,
        os.environ.get("DUCA_RIME_PRETRAIN_SHA256"),
        "VideoMAE initialization",
    )
    annotation_path, annotation_sha = _bound_file(
        evaluation_annotation_path,
        os.environ.get("DUCA_RIME_EVALUATION_ANNOTATION_SHA256"),
        "evaluation annotation",
    )
    class_map_path, class_map_sha = _bound_file(
        evaluation_class_map_path,
        os.environ.get("DUCA_RIME_EVALUATION_CLASS_MAP_SHA256"),
        "evaluation class map",
    )
    targets_path, targets_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_TARGETS_JSONL"),
        os.environ.get("DUCA_RIME_TARGETS_SHA256"),
        "cross-fitted targets",
    )
    protocol_path, protocol_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_BUDGET_PROTOCOL_JSON"),
        os.environ.get("DUCA_RIME_BUDGET_PROTOCOL_SHA256"),
        "budget protocol",
    )
    replay_path, replay_sha = _optional_bound_file(
        os.environ.get("DUCA_RIME_REPLAY_JSONL"),
        os.environ.get("DUCA_RIME_REPLAY_SHA256"),
        "budget replay",
    )
    if variant != "U-fixed" and (targets_path is None or protocol_path is None):
        raise ValueError("trainable RIME selector arms require targets and frozen protocol")
    if variant in {"D-shuffle", "AdapTok-TAD"} and replay_path is None:
        raise ValueError(f"{variant} requires its immutable budget replay")
    if variant not in {"D-shuffle", "AdapTok-TAD"} and replay_path is not None:
        raise ValueError(f"{variant} must not consume a budget replay during training")
    actual_eval_hash = evaluation_config_sha256(
        evaluation_config,
        expected_subset=str(evaluation_config["subset"]),
    )
    if actual_eval_hash != os.environ.get("DUCA_RIME_EVALUATION_CONFIG_SHA256"):
        raise ValueError("RIME evaluation config differs from the sealed launch")

    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "phase2_receipt_path": phase2_path,
        "phase2_receipt_sha256": phase2_sha,
        "split_assignment_sha256": str(phase2["split_assignment_sha256"]),
        "training_exposure_path": exposure_path,
        "training_exposure_sha256": exposure_sha,
        "initialization_path": pretrain_path,
        "initialization_sha256": pretrain_sha,
        "targets_path": targets_path,
        "targets_sha256": targets_sha,
        "budget_protocol_path": protocol_path,
        "budget_protocol_sha256": protocol_sha,
        "budget_replay_path": replay_path,
        "budget_replay_sha256": replay_sha,
        "evaluation_annotation_path": annotation_path,
        "evaluation_annotation_sha256": annotation_sha,
        "evaluation_class_map_path": class_map_path,
        "evaluation_class_map_sha256": class_map_sha,
        "evaluation_config_sha256": actual_eval_hash,
        "official_final_subset_consumed": False,
    }
    bindings["binding_sha256"] = _canonical_sha256(bindings)
    return bindings


def validate_terminal_checkpoint_binding(
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    git_commit: str,
    evaluation_arm: str,
    seed: int,
) -> dict[str, Any]:
    receipt_path, _receipt_sha = _bound_file(
        os.environ.get("DUCA_RIME_TRAINING_RECEIPT"),
        os.environ.get("DUCA_RIME_TRAINING_RECEIPT_SHA256"),
        "Phase-3 training receipt",
    )
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    checkpoint_resolved = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint_sha = _sha256_file(checkpoint_resolved)
    source_arm = str(receipt.get("arm"))
    if (
        receipt.get("schema_version") != "duca_rime_phase3_training_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != str(git_commit)
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("successful_detector_updates", -1)) != 6000
        or receipt.get("formal_update_audit_passed") is not True
        or receipt.get("uses_official_final") is not False
        or str(Path(receipt.get("checkpoint_path", "")).resolve()) != checkpoint_resolved
        or receipt.get("checkpoint_sha256") != checkpoint_sha
        or int(receipt.get("checkpoint_epoch", -1)) != 59
        or receipt.get("checkpoint_state_key") != "state_dict_ema"
        or (
            str(evaluation_arm) != source_arm
            and not (str(evaluation_arm) == "U-same-K" and source_arm == "RIME-full")
        )
    ):
        raise ValueError("RIME terminal checkpoint/training receipt binding mismatch")
    metadata = checkpoint.get("experiment_metadata")
    audit = None if not isinstance(metadata, Mapping) else metadata.get("training_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "complete"
        or audit.get("git_commit") != str(git_commit)
        or audit.get("variant") != source_arm
        or int(audit.get("seed", -1)) != int(seed)
        or int(audit.get("expected_successful_optimizer_updates", -1)) != 6000
        or int(audit.get("update_audit", {}).get("successful_optimizer_updates", -1))
        != 6000
    ):
        raise ValueError("RIME terminal checkpoint contains an invalid training audit")
    return {
        "training_receipt_path": receipt_path,
        "training_receipt_sha256": _sha256_file(receipt_path),
        "source_arm": source_arm,
        "evaluation_arm": str(evaluation_arm),
        "checkpoint_path": checkpoint_resolved,
        "checkpoint_sha256": checkpoint_sha,
        "successful_detector_updates": 6000,
        "split_assignment_sha256": audit["split_assignment_sha256"],
        "training_exposure_sha256": audit["training_exposure_sha256"],
        "initialization_sha256": audit["initialization_sha256"],
        "official_final_subset_consumed_during_training": False,
    }


__all__ = [
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "FORMAL_PROTOCOLS",
    "TRAIN_ARMS",
    "atomic_write_json",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "formal_training_contract",
    "is_formal_protocol",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
