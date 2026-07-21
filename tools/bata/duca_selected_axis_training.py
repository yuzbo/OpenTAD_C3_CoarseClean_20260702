from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.bata import duca_p0_training as legacy
from tools.bata.duca_p0_evaluation import evaluation_config_sha256


FORMAL_PROTOCOL = "duca_selected_axis_optimization_v1"
BOUNDARY_BURST_GATE_SCHEMA = "duca_boundary_burst_full_model_gate_v1"
LOCKED_ALIGNMENT_VARIANTS = frozenset(
    {"global_curriculum_g1", "global_curriculum_g2"}
)
VARIANT_CONFIGS = {
    "exact_uniform": "duca_exact_uniform_fixed384_official60.py",
    "direct025": "duca_protected_e2e_direct025_fixed384_official60.py",
    "homotopy025": "duca_protected_e2e_homotopy025_fixed384_official60.py",
    "homotopy_uni_companion025": (
        "duca_protected_e2e_homotopy_uni_companion025_fixed384_official60.py"
    ),
    "two_stage_exact_uniform": (
        "duca_two_stage_exact_uniform_fixed384_official60.py"
    ),
    "two_stage_scratch": "duca_two_stage_scratch_fixed384_official60.py",
    "two_stage_pretrained_joint": (
        "duca_two_stage_pretrained_joint_fixed384_official60.py"
    ),
    "two_stage_pretrained_frozen": (
        "duca_two_stage_pretrained_frozen_fixed384_official60.py"
    ),
    "global_curriculum_g0": (
        "duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    ),
    "global_curriculum_g1": (
        "duca_global_curriculum_g1_protected_fixed384_official60.py"
    ),
    "global_curriculum_g2": (
        "duca_global_curriculum_g2_uni_companion_fixed384_official60.py"
    ),
    "gaussian_matched_g0": (
        "duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    ),
    "boundary_burst_r2q3_g0": (
        "duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    ),
    "boundary_burst_r4q5_g0": (
        "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
    ),
}

DUCA_P0_TRAINING_AUDIT_SCHEMA = legacy.DUCA_P0_TRAINING_AUDIT_SCHEMA
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = legacy.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = legacy.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA
DUCA_TRAINING_AUDIT_FILENAME = "duca_selected_axis_training_audit.json"

atomic_write_json = legacy.atomic_write_json
build_checkpoint_metadata = legacy.build_checkpoint_metadata
build_training_audit = legacy.build_training_audit
capture_global_rng_state = legacy.capture_global_rng_state
canonical_sha256 = legacy.canonical_sha256
new_update_audit = legacy.new_update_audit
restore_global_rng_state = legacy.restore_global_rng_state
restore_training_state = legacy.restore_training_state
selector_schedule_step = legacy.selector_schedule_step
sha256_file = legacy.sha256_file
validate_update_state = legacy.validate_update_state


def _validate_embedded_hash(
    payload: Mapping[str, Any], *, hash_key: str, label: str
) -> str:
    expected = str(payload.get(hash_key, ""))
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    observed = canonical_sha256(unsigned)
    if expected != observed:
        raise RuntimeError(f"selected-axis {label} self-hash mismatch")
    return observed


def validate_frozen_pretrain_binding(
    *,
    runtime_path: str | Path,
    expected_path: str | Path,
    expected_sha256: str,
) -> dict[str, str]:
    runtime = Path(runtime_path).expanduser().resolve()
    expected = Path(expected_path).expanduser().resolve()
    digest = str(expected_sha256).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("submit-frozen AdaTAD pretrain SHA256 is invalid")
    if runtime != expected:
        raise RuntimeError("AdaTAD pretrain path drifted after submission")
    if not runtime.is_file():
        raise RuntimeError(f"AdaTAD pretrain is missing: {runtime}")
    observed = sha256_file(runtime)
    if observed != digest:
        raise RuntimeError("AdaTAD pretrain content drifted after submission")
    return {"path": str(runtime), "sha256": observed}


def formal_training_contract(cfg) -> dict[str, Any] | None:
    workflow = cfg.workflow
    if str(workflow.get("formal_protocol", "")) != FORMAL_PROTOCOL:
        return None
    contract = legacy.formal_training_contract(
        cfg,
        expected_checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    )
    if contract is None:
        raise ValueError(
            "selected-axis official-60 requires formal_successful_update_contract"
        )
    if str(workflow.get("training_profile", "")) != "official60":
        raise ValueError("selected-axis training profile must be official60")
    if int(contract["end_epoch"]) != 60:
        raise ValueError("selected-axis official training must use 60 epochs")
    if int(contract["expected_train_batches_per_epoch"]) != 100:
        raise ValueError("selected-axis official training must use 100 batches per epoch")
    if int(contract["expected_successful_optimizer_updates"]) != 6000:
        raise ValueError("selected-axis official training must use 6000 updates")
    selector = cfg.model.frame_selector
    if int(selector.budget) != 384 or int(selector.dense_window_size) != 768:
        raise ValueError("selected-axis official training requires T=768 and K=384")
    if str(selector.detector_output_coordinate_space) != "selected_axis_index":
        raise ValueError("selected-axis detector geometry is not frozen")
    if not bool(selector.remap_gt_to_selected_axis):
        raise ValueError("selected-axis GT remapping is required")
    contract = dict(contract)
    contract["formal_protocol"] = FORMAL_PROTOCOL
    contract["training_profile"] = "official60"
    return contract


def assert_safe_cfg_options(
    cfg_options: Mapping[str, Any] | None,
    *,
    entrypoint: str,
) -> None:
    if not cfg_options:
        return
    allowed: dict[str, Any] = {
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
            f"{entrypoint} rejected selected-axis DUCA cfg overrides: "
            + ", ".join(sorted(rejected))
        )


def _load_gate_suite(git_commit: str) -> tuple[dict[str, Any], str]:
    raw_path = os.environ.get("DUCA_SELECTED_OPT_GATE_SUITE", "")
    expected_sha256 = os.environ.get("DUCA_SELECTED_OPT_GATE_SUITE_SHA256", "")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("selected-axis gate suite is missing")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("selected-axis gate suite hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        not in {
            "duca_selected_axis_optimization_gate_v1",
            BOUNDARY_BURST_GATE_SCHEMA,
        }
        or payload.get("ok") is not True
        or payload.get("formal_training_unlocked") is not True
        or payload.get("git_commit") != git_commit
    ):
        raise RuntimeError("selected-axis gate suite did not authorize this commit")
    return payload, observed_sha256


def _load_full_model_gate(
    suite: Mapping[str, Any],
    *,
    config_name: str,
) -> tuple[dict[str, Any], str]:
    suffix = f"/full_model/{Path(config_name).stem}.json"
    matches = [
        item
        for item in suite.get("artifacts", [])
        if isinstance(item, Mapping)
        and str(item.get("path", "")).replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError("selected-axis suite lacks one matching full-model gate")
    record = matches[0]
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RuntimeError("selected-axis full-model gate hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("selected-axis full-model gate did not pass")
    return payload, str(record["sha256"])


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
    selector_initialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if variant in LOCKED_ALIGNMENT_VARIANTS:
        raise RuntimeError(
            f"selected-axis variant {variant} requires real legal hard-swap alignment"
        )
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"invalid selected-axis variant: {variant}")
    if int(seed) != 3407:
        raise ValueError("selected-axis official training seed must be 3407")
    source_config = Path(source_config_path).resolve()
    if source_config.name != VARIANT_CONFIGS[variant]:
        raise RuntimeError("selected-axis variant/config mismatch")
    suite, suite_sha256 = _load_gate_suite(git_commit)
    full_gate, full_gate_sha256 = _load_full_model_gate(
        suite,
        config_name=source_config.name,
    )
    if full_gate.get("config_sha256") != source_config_sha256:
        raise RuntimeError("runtime config source differs from the full-model gate")
    runtime = full_gate.get("runtime", {})
    if runtime.get("git_commit") != git_commit:
        raise RuntimeError("full-model gate commit drift")

    pretrain = Path(runtime_pretrain_path).expanduser().resolve()
    annotation = Path(evaluation_annotation_path).expanduser().resolve()
    class_map = Path(evaluation_class_map_path).expanduser().resolve()
    for path, label in (
        (pretrain, "VideoMAE-S pretrain"),
        (annotation, "evaluation annotation"),
        (class_map, "evaluation class map"),
    ):
        if not path.is_file():
            raise RuntimeError(f"selected-axis {label} is missing: {path}")
    pretrain_sha256 = sha256_file(pretrain)
    if full_gate.get("adatad_pretrain", {}).get("sha256") != pretrain_sha256:
        raise RuntimeError("runtime pretrain differs from the full-model gate")

    initialization_cfg = dict(selector_initialization or {})
    initialization_enabled = bool(
        initialization_cfg and initialization_cfg.get("enabled", True)
    )
    gate_initialization = full_gate.get("selector_initialization")
    initialization_binding = None
    if initialization_enabled:
        if not isinstance(gate_initialization, Mapping):
            raise RuntimeError("full-model gate lacks selector initialization evidence")
        checkpoint = Path(
            str(initialization_cfg.get("checkpoint_path", ""))
        ).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError("runtime selector initialization checkpoint is missing")
        checkpoint_sha256 = sha256_file(checkpoint)
        expected_sha256 = str(
            initialization_cfg.get("checkpoint_sha256", "")
        ).lower()
        if checkpoint_sha256 != expected_sha256:
            raise RuntimeError("runtime selector initialization checkpoint hash drift")
        expected_epoch = int(initialization_cfg["expected_checkpoint_epoch"])
        expected_state_key = str(initialization_cfg.get("state_key", "state_dict_ema"))
        if (
            gate_initialization.get("checkpoint_sha256") != checkpoint_sha256
            or int(gate_initialization.get("checkpoint_epoch", -1)) != expected_epoch
            or gate_initialization.get("checkpoint_state_key") != expected_state_key
            or gate_initialization.get("detector_state_loaded") is not False
            or gate_initialization.get("optimizer_state_loaded") is not False
            or gate_initialization.get("scheduler_state_loaded") is not False
        ):
            raise RuntimeError("runtime selector initialization differs from the full-model gate")
        initialization_binding = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": expected_epoch,
            "checkpoint_state_key": expected_state_key,
            "reset_state_keys": list(initialization_cfg.get("reset_state_keys", [])),
            "gate_receipt_sha256": str(gate_initialization.get("receipt_sha256", "")),
        }
    elif gate_initialization is not None:
        raise RuntimeError("full-model gate unexpectedly initialized a selector checkpoint")

    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "source_config_path": str(source_config),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "gate_suite_sha256": suite_sha256,
        "full_model_gate_sha256": full_gate_sha256,
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha256,
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": sha256_file(annotation),
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": sha256_file(class_map),
        "evaluation_config_sha256": evaluation_config_sha256(evaluation_config),
    }
    if initialization_binding is not None:
        bindings["selector_initialization_contract"] = initialization_binding
    return bindings


def validate_terminal_checkpoint_binding(
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    git_commit: str,
    variant: str,
    seed: int,
    slurm_job_id: str | None,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
    checkpoint_epoch: int,
    checkpoint_state_key: str,
    evaluation_annotation_path: str | Path,
    evaluation_class_map_path: str | Path,
    evaluation_config: Mapping[str, Any],
    runtime_pretrain_path: str | Path,
    frozen_pretrain_path: str | Path,
    frozen_pretrain_sha256: str,
    selector_initialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the terminal checkpoint and its complete training identity chain."""

    if int(seed) != 3407:
        raise RuntimeError("selected-axis terminal evaluation seed must be 3407")
    if int(checkpoint_epoch) != 59 or checkpoint_state_key != "state_dict_ema":
        raise RuntimeError(
            "selected-axis terminal evaluation requires epoch-59 EMA"
        )
    if checkpoint_state_key not in checkpoint:
        raise RuntimeError("selected-axis checkpoint lacks terminal EMA weights")

    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise RuntimeError("selected-axis terminal checkpoint is missing")
    sidecar_file = Path(f"{checkpoint_file}.metadata.json").resolve()
    if not sidecar_file.is_file():
        raise RuntimeError("selected-axis terminal checkpoint sidecar is missing")
    sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("schema_version") != DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA
    ):
        raise RuntimeError("selected-axis terminal checkpoint sidecar schema mismatch")
    _validate_embedded_hash(
        sidecar, hash_key="sidecar_sha256", label="checkpoint sidecar"
    )
    checkpoint_sha256 = sha256_file(checkpoint_file)
    if (
        Path(str(sidecar.get("checkpoint_path", ""))).expanduser().resolve()
        != checkpoint_file
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise RuntimeError("selected-axis terminal checkpoint/sidecar drift")

    metadata = sidecar.get("experiment_metadata")
    embedded_metadata = checkpoint.get("experiment_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != DUCA_P0_CHECKPOINT_METADATA_SCHEMA
        or embedded_metadata != metadata
    ):
        raise RuntimeError("selected-axis terminal checkpoint metadata mismatch")
    _validate_embedded_hash(
        metadata, hash_key="metadata_sha256", label="checkpoint metadata"
    )
    audit = metadata.get("training_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != DUCA_P0_TRAINING_AUDIT_SCHEMA
    ):
        raise RuntimeError("selected-axis terminal training audit is missing")
    _validate_embedded_hash(audit, hash_key="audit_sha256", label="training audit")

    audit_file = checkpoint_file.parent.parent / DUCA_TRAINING_AUDIT_FILENAME
    if not audit_file.is_file():
        raise RuntimeError("selected-axis terminal training audit file is missing")
    persisted_audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if persisted_audit != audit:
        raise RuntimeError("selected-axis persisted training audit differs from checkpoint")

    counters = audit.get("update_audit")
    if not isinstance(counters, Mapping):
        raise RuntimeError("selected-axis terminal update audit is missing")
    expected_updates = 6000
    required_identity = {
        "status": "complete",
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": 3407,
        "formal_protocol": FORMAL_PROTOCOL,
        "training_profile": "official60",
        "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
        "primary_checkpoint_epoch": 59,
        "primary_checkpoint_state_key": "state_dict_ema",
        "expected_train_batches_per_epoch": 100,
        "expected_successful_optimizer_updates": expected_updates,
        "last_completed_epoch": 59,
        "epochs_completed": 60,
        "train_batches_per_epoch": 100,
        "scheduler_last_epoch": expected_updates,
        "selector_schedule_step": expected_updates,
    }
    for key, expected in required_identity.items():
        if audit.get(key) != expected:
            raise RuntimeError(
                f"selected-axis terminal training identity mismatch: {key}"
            )
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        if int(counters.get(key, -1)) != expected_updates:
            raise RuntimeError(
                f"selected-axis terminal update accounting mismatch: {key}"
            )
    if (
        int(counters.get("replay_exhaustions", -1)) != 0
        or int(counters.get("forced_amp_overflow_attempts", -1)) != 0
    ):
        raise RuntimeError("selected-axis terminal AMP contract was violated")
    if int(counters.get("optimizer_attempts", -1)) != expected_updates + int(
        counters.get("amp_skipped_attempts", -1)
    ):
        raise RuntimeError("selected-axis terminal optimizer accounting mismatch")
    records = audit.get("epoch_records")
    if (
        not isinstance(records, list)
        or len(records) != 60
        or [int(item.get("epoch", -1)) for item in records] != list(range(60))
    ):
        raise RuntimeError("selected-axis terminal epoch audit is incomplete")

    pretrain = validate_frozen_pretrain_binding(
        runtime_path=runtime_pretrain_path,
        expected_path=frozen_pretrain_path,
        expected_sha256=frozen_pretrain_sha256,
    )
    expected_bindings = build_runtime_bindings(
        git_commit=git_commit,
        variant=variant,
        seed=seed,
        slurm_job_id=slurm_job_id,
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        resolved_config_sha256=resolved_config_sha256,
        runtime_config_sha256=str(audit.get("runtime_config_sha256", "")),
        evaluation_annotation_path=evaluation_annotation_path,
        evaluation_class_map_path=evaluation_class_map_path,
        evaluation_config=evaluation_config,
        runtime_pretrain_path=runtime_pretrain_path,
        selector_initialization=selector_initialization,
    )
    for key, expected in expected_bindings.items():
        if audit.get(key) != expected:
            raise RuntimeError(
                f"selected-axis terminal training binding mismatch: {key}"
            )

    return {
        "variant": str(variant),
        "seed": 3407,
        "successful_optimizer_updates": expected_updates,
        "checkpoint_sidecar_path": str(sidecar_file),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_file),
        "training_audit_path": str(audit_file.resolve()),
        "training_audit_sha256": sha256_file(audit_file),
        "training_audit_self_sha256": str(audit["audit_sha256"]),
        "gate_suite_sha256": str(audit["gate_suite_sha256"]),
        "full_model_gate_sha256": str(audit["full_model_gate_sha256"]),
        "pretrain_path": pretrain["path"],
        "pretrain_sha256": pretrain["sha256"],
        "frontend_initialization": audit.get("selector_initialization_contract"),
    }


__all__ = [
    "BOUNDARY_BURST_GATE_SCHEMA",
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_P0_TRAINING_AUDIT_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "FORMAL_PROTOCOL",
    "LOCKED_ALIGNMENT_VARIANTS",
    "VARIANT_CONFIGS",
    "assert_safe_cfg_options",
    "atomic_write_json",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "canonical_sha256",
    "formal_training_contract",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "sha256_file",
    "validate_frozen_pretrain_binding",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
