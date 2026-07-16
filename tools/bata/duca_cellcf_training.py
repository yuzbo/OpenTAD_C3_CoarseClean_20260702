from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from tools.bata import duca_p0_training as legacy
from tools.bata.duca_p0_evaluation import evaluation_config_sha256


DUCA_P0_TRAINING_AUDIT_SCHEMA = "duca_cellcf_training_audit_v1"
DUCA_P0_CHECKPOINT_METADATA_SCHEMA = "duca_cellcf_checkpoint_metadata_v1"
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = "duca_cellcf_checkpoint_sidecar_v1"
DUCA_TRAINING_AUDIT_FILENAME = "duca_cellcf_training_audit.json"
VARIANTS = ("uniform", "transition_beta0", "cellcf")
ROOT = Path(__file__).resolve().parents[2]

atomic_write_json = legacy.atomic_write_json
capture_global_rng_state = legacy.capture_global_rng_state
canonical_sha256 = legacy.canonical_sha256
new_update_audit = legacy.new_update_audit
restore_global_rng_state = legacy.restore_global_rng_state
selector_schedule_step = legacy.selector_schedule_step
sha256_file = legacy.sha256_file
validate_update_state = legacy.validate_update_state


def _flatten_cfg_options(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(_flatten_cfg_options(item, path))
        else:
            output[path] = item
    return output


def assert_safe_cfg_options(cfg, cfg_options, *, entrypoint: str) -> None:
    """Freeze CellCF semantics while permitting only audited runtime paths."""
    if str(cfg.workflow.get("formal_protocol", "")) != "duca_cellcf_v1":
        return
    if not cfg_options:
        return
    allowed = {
        "tools/train.py": {
            "work_dir",
            "model.backbone.custom.pretrain",
        },
        "tools/test.py": {
            "work_dir",
            "model.backbone.custom.pretrain",
            "post_processing.save_dict",
            "inference.load_from_raw_predictions",
        },
    }
    if entrypoint not in allowed:
        raise RuntimeError(f"CellCF has no cfg-options allowlist for {entrypoint}")
    flattened = _flatten_cfg_options(cfg_options)
    rejected = sorted(set(flattened) - allowed[entrypoint])
    if rejected:
        raise RuntimeError(
            "formal CellCF rejected semantic --cfg-options: " + ", ".join(rejected)
        )
    work_dir = flattened.get("work_dir")
    if work_dir is not None and (not isinstance(work_dir, str) or not work_dir.strip()):
        raise RuntimeError("formal CellCF work_dir override must be a non-empty path")
    pretrain = flattened.get("model.backbone.custom.pretrain")
    if pretrain is not None and (not isinstance(pretrain, str) or not pretrain.strip()):
        raise RuntimeError("formal CellCF pretrain override must be a non-empty path")
    if entrypoint == "tools/test.py":
        if flattened.get("post_processing.save_dict", True) is not True:
            raise RuntimeError("formal CellCF evaluation must save the prediction dictionary")
        if flattened.get("inference.load_from_raw_predictions", False) is not False:
            raise RuntimeError("formal CellCF evaluation forbids raw-prediction loading")


def expected_runtime_config_sha256(
    config_path: str | Path,
    cfg_options: Mapping[str, Any],
    *,
    experiment_id: int,
    gpu_num: int,
    entrypoint: str,
) -> str:
    """Resolve the exact effective config before launching a formal process."""
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    assert_safe_cfg_options(cfg, cfg_options, entrypoint=entrypoint)
    cfg.merge_from_dict(dict(cfg_options))
    cfg.work_dir = os.path.join(cfg.work_dir, f"gpu{int(gpu_num)}_id{int(experiment_id)}/")
    return canonical_sha256(cfg.to_dict())


def formal_training_contract(cfg) -> dict[str, Any] | None:
    if str(cfg.workflow.get("formal_protocol", "")) != "duca_cellcf_v1":
        return None
    contract = legacy.formal_training_contract(cfg)
    if contract is None:
        raise ValueError("CellCF formal protocol requires formal_successful_update_contract=True")
    contract = dict(contract)
    contract["formal_protocol"] = "duca_cellcf_v1"
    return contract


def _bound_file(path: str | Path, expected_sha256: str | None, label: str) -> tuple[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"CellCF {label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if expected_sha256 != observed:
        raise ValueError(f"CellCF {label} hash mismatch")
    return str(resolved), observed


def _bound_artifact(
    env_name: str,
    *,
    schema: str,
    git_commit: str,
    label: str,
) -> tuple[str, str, dict[str, Any]]:
    raw_path = os.environ.get(env_name, "")
    if not raw_path:
        raise ValueError(f"{env_name} is required for formal CellCF training")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"CellCF {label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("ok") is not True:
        raise ValueError(f"CellCF {label} has an incompatible schema/status")
    if payload.get("git_commit") != git_commit:
        raise ValueError(f"CellCF {label} is bound to another commit")
    return str(path), sha256_file(path), payload


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
    runtime_pretrain_path: str | Path,
    evaluation_annotation_path: str | Path,
    evaluation_class_map_path: str | Path,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    variant = os.environ.get("DUCA_CELLCF_VARIANT", variant)
    if variant not in VARIANTS:
        raise ValueError(f"invalid formal CellCF variant: {variant!r}")
    if os.environ.get("DUCA_EXPECTED_COMMIT") != git_commit:
        raise ValueError("formal CellCF checkout differs from DUCA_EXPECTED_COMMIT")
    expected_resolved = os.environ.get("DUCA_CELLCF_RESOLVED_CONFIG_SHA256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError("formal CellCF resolved config differs from the frozen suite")
    expected_runtime = os.environ.get("DUCA_CELLCF_RUNTIME_CONFIG_SHA256")
    if expected_runtime != runtime_config_sha256:
        raise ValueError("formal CellCF effective runtime config differs from the frozen launch")
    protocol_sha = os.environ.get("DUCA_CELLCF_PROTOCOL_SHA256", "")
    order_sha = os.environ.get("DUCA_CELLCF_ORDER_SHA256", "")
    if len(protocol_sha) != 64 or len(order_sha) != 64:
        raise ValueError("formal CellCF protocol/order SHA256 bindings are missing")

    gate_path, gate_sha, gate = _bound_artifact(
        "DUCA_CELLCF_GATE_JSON",
        schema="duca_cellcf_real_loader_cuda_gate_v1",
        git_commit=git_commit,
        label="real-loader CUDA gate",
    )
    pilot_path, pilot_sha, pilot = _bound_artifact(
        "DUCA_CELLCF_DDP_PILOT_JSON",
        schema="duca_cellcf_ddp_pilot_suite_v1",
        git_commit=git_commit,
        label="forced-overflow DDP pilot",
    )
    if gate_sha != os.environ.get("DUCA_CELLCF_GATE_SHA256"):
        raise ValueError("CellCF real-loader gate hash differs from the frozen suite")
    if pilot_sha != os.environ.get("DUCA_CELLCF_DDP_PILOT_SHA256"):
        raise ValueError("CellCF DDP pilot hash differs from the frozen suite")
    from tools.bata.validate_duca_cellcf_real_loader_gate import (
        validate_real_loader_gate_artifact,
    )
    from tools.bata.validate_duca_cellcf_ddp_pilot import validate_pilot_artifact

    validate_real_loader_gate_artifact(
        gate_path,
        expected_commit=git_commit,
        expected_sha256=gate_sha,
        require_clean=True,
    )
    validate_pilot_artifact(
        pilot_path,
        repo_root=ROOT,
        expected_commit=git_commit,
        expected_real_loader_gate_sha256=gate_sha,
        require_clean=True,
    )
    if pilot.get("real_loader_gate_sha256") != gate_sha:
        raise ValueError("CellCF pilot is not bound to the selected real-loader gate")
    if tuple(pilot.get("variant_order", ())) != VARIANTS:
        raise ValueError("CellCF pilot arm order differs from formal training")
    gate_pretrain = gate.get("assets", {}).get("videomae_checkpoint", {})
    pretrain_path, pretrain_sha = _bound_file(
        runtime_pretrain_path,
        gate_pretrain.get("sha256"),
        "VideoMAE pretrain",
    )
    if pretrain_path != str(Path(str(gate_pretrain.get("path", ""))).expanduser().resolve()):
        raise ValueError("formal CellCF VideoMAE pretrain path differs from the real-loader gate")

    annotation_path, annotation_sha = _bound_file(
        evaluation_annotation_path,
        os.environ.get("DUCA_CELLCF_ANNOTATION_SHA256"),
        "evaluation annotation",
    )
    class_map_path, class_map_sha = _bound_file(
        evaluation_class_map_path,
        os.environ.get("DUCA_CELLCF_CLASS_MAP_SHA256"),
        "evaluation class map",
    )
    observed_eval_sha = evaluation_config_sha256(evaluation_config)
    if observed_eval_sha != os.environ.get("DUCA_CELLCF_EVALUATION_CONFIG_SHA256"):
        raise ValueError("formal CellCF evaluation config hash mismatch")
    gate_dataset = gate.get("dataset", {})
    if gate_dataset.get("annotation_sha256") != annotation_sha:
        raise ValueError("CellCF real-loader gate annotation differs from training")
    if gate_dataset.get("class_map_sha256") != class_map_sha:
        raise ValueError("CellCF real-loader gate class map differs from training")
    if not slurm_job_id or not str(slurm_job_id).isdigit():
        raise ValueError("formal CellCF training requires a numeric Slurm job id")

    return {
        "formal_protocol": "duca_cellcf_v1",
        "git_commit": git_commit,
        "variant": variant,
        "seed": int(seed),
        "slurm_job_id": str(slurm_job_id),
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": source_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "runtime_pretrain_path": pretrain_path,
        "runtime_pretrain_sha256": pretrain_sha,
        "protocol_sha256": protocol_sha,
        "ordered_exposure_sha256": order_sha,
        "real_loader_gate_json": gate_path,
        "real_loader_gate_sha256": gate_sha,
        "ddp_pilot_json": pilot_path,
        "ddp_pilot_sha256": pilot_sha,
        "synthetic_gate_sha256": gate.get("synthetic_gate_sha256"),
        "evaluation_annotation_path": annotation_path,
        "evaluation_annotation_sha256": annotation_sha,
        "evaluation_class_map_path": class_map_path,
        "evaluation_class_map_sha256": class_map_sha,
        "evaluation_config_sha256": observed_eval_sha,
    }


def build_training_audit(**kwargs: Any) -> dict[str, Any]:
    payload = legacy.build_training_audit(**kwargs)
    payload["schema_version"] = DUCA_P0_TRAINING_AUDIT_SCHEMA
    payload.pop("audit_sha256", None)
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
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != DUCA_P0_CHECKPOINT_METADATA_SCHEMA:
        raise RuntimeError("formal CellCF resume checkpoint metadata schema mismatch")
    unsigned_metadata = dict(metadata)
    observed_metadata_hash = unsigned_metadata.pop("metadata_sha256", None)
    if observed_metadata_hash != canonical_sha256(unsigned_metadata):
        raise RuntimeError("formal CellCF resume checkpoint metadata hash mismatch")
    audit = metadata.get("training_audit")
    if not isinstance(audit, Mapping) or audit.get("schema_version") != DUCA_P0_TRAINING_AUDIT_SCHEMA:
        raise RuntimeError("formal CellCF resume training audit schema mismatch")
    unsigned_audit = dict(audit)
    observed_audit_hash = unsigned_audit.pop("audit_sha256", None)
    if observed_audit_hash != canonical_sha256(unsigned_audit):
        raise RuntimeError("formal CellCF resume training audit hash mismatch")
    for key, expected in bindings.items():
        if key != "slurm_job_id" and audit.get(key) != expected:
            raise RuntimeError(f"formal CellCF resume binding mismatch: {key}")
    if audit.get("checkpoint_criterion") != contract["checkpoint_criterion"]:
        raise RuntimeError("formal CellCF checkpoint criterion mismatch")
    counters = audit.get("update_audit")
    records = audit.get("epoch_records")
    if not isinstance(counters, Mapping) or not isinstance(records, list):
        raise RuntimeError("formal CellCF resume state is incomplete")
    restored = {str(key): int(value) for key, value in counters.items()}
    if set(restored) != set(legacy.new_update_audit()):
        raise RuntimeError("formal CellCF resume update counters are incomplete")
    return restored, [dict(item) for item in records]


__all__ = [
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_P0_TRAINING_AUDIT_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "atomic_write_json",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "capture_global_rng_state",
    "formal_training_contract",
    "new_update_audit",
    "restore_training_state",
    "restore_global_rng_state",
    "selector_schedule_step",
    "validate_update_state",
]
