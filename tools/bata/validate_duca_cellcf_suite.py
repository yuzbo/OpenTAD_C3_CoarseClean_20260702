from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config

from tools.bata.duca_cellcf_protocol import (
    protocol_for_name,
    protocol_from_environment,
)
from tools.bata.duca_p0_evaluation import evaluation_config_sha256
from tools.bata.finalize_duca_cellcf_run import (
    EVIDENCE_SCHEMA as POST_RUN_SCHEMA,
    finalize_run as finalize_cellcf_run,
)
from tools.bata.summarize_duca_cellcf_cost import (
    SCHEMA as COST_EVIDENCE_SCHEMA,
    summarize as summarize_cost_evidence,
)
from tools.bata.validate_duca_cellcf_real_loader_gate import (
    validate_real_loader_gate_artifact,
)
from tools.bata.validate_duca_cellcf_ddp_pilot import validate_pilot_artifact
from tools.bata.validate_duca_cellcf_fixed384 import VARIANTS, validate_config


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "duca_cellcf_suite_manifest_v1"
VARIANT_ORDER = ("uniform", "transition_beta0", "cellcf")
BARE_COST_CONFIG = (
    "configs/adatad/thumos/duca_cellcf_bare_exact_uniform_fixed384_cost.py"
)
_LEGACY_EXPOSURE132_PROTOCOL = protocol_for_name("exposure132")
EXPECTED_UPDATES = (
    _LEGACY_EXPOSURE132_PROTOCOL.expected_successful_optimizer_updates
)
TERMINAL_EPOCH = _LEGACY_EXPOSURE132_PROTOCOL.terminal_epoch
TERMINAL_STATE_KEY = _LEGACY_EXPOSURE132_PROTOCOL.terminal_state_key


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _resolve_referenced_path(
    value: Any, *, label: str, base: Path | None = None
) -> Path:
    _require(isinstance(value, (str, Path)) and bool(str(value)), f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    resolved = path.resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    return resolved


def _require_hashed_file(
    payload: Mapping[str, Any],
    path_key: str,
    hash_key: str,
    label: str,
    *,
    base: Path | None = None,
    hash_cache: dict[Path, str] | None = None,
) -> tuple[Path, str]:
    resolved = _resolve_referenced_path(payload.get(path_key), label=label, base=base)
    if hash_cache is not None and resolved in hash_cache:
        observed = hash_cache[resolved]
    else:
        observed = _sha256(resolved)
        if hash_cache is not None:
            hash_cache[resolved] = observed
    _require(
        payload.get(hash_key) == observed,
        f"{label} hash mismatch: {resolved}",
    )
    return resolved, observed


def _require_evaluator_source(
    value: Any,
    *,
    label: str,
    hash_cache: dict[Path, str],
) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{label} identity is missing")
    source, source_sha256 = _require_hashed_file(
        value,
        "source_path",
        "source_sha256",
        f"{label} source",
        hash_cache=hash_cache,
    )
    _require(value.get("class_name") == "mAP", f"{label} is not OpenTAD mAP")
    return {"path": str(source), "sha256": source_sha256}


def _shared_protocol(cfg: Config) -> dict[str, Any]:
    model = _plain(cfg.model)
    selector = dict(model["frame_selector"])
    for key in (
        "local_cell_force_exact_uniform",
        "counterfactual_utility_distillation_weight",
        "require_counterfactual_utility_teacher",
    ):
        selector.pop(key, None)
    model["frame_selector"] = selector
    return {
        "task": "offline_temporal_action_detection",
        "dense_window_size": int(cfg.dense_window_size),
        "budget": int(cfg.window_size),
        "model": model,
        "dataset": _plain(cfg.dataset),
        "solver": _plain(cfg.solver),
        "optimizer": _plain(cfg.optimizer),
        "scheduler": _plain(cfg.scheduler),
        "workflow": _plain(cfg.workflow),
        "evaluation": _plain(cfg.evaluation),
        "inference": _plain(cfg.inference),
        "post_processing": _plain(cfg.post_processing),
    }


def _variant_contract(cfg: Config, variant: str) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    return {
        "variant": variant,
        "force_exact_uniform": bool(selector.local_cell_force_exact_uniform),
        "counterfactual_weight": float(
            selector.counterfactual_utility_distillation_weight
        ),
        "requires_counterfactual_teacher": bool(
            selector.require_counterfactual_utility_teacher
        ),
        "acquisition_policy": str(selector.acquisition_policy),
        "counterfactual_objective": str(selector.counterfactual_objective),
        "detector_gradient_mode": str(selector.detector_gradient_mode),
    }


def _reference_data(cfg: Config) -> dict[str, Any]:
    annotation = Path(cfg.evaluation.ground_truth_filename).expanduser().resolve()
    class_map = Path(cfg.dataset.test.class_map).expanduser().resolve()
    _require(annotation.is_file(), f"evaluation annotation is missing: {annotation}")
    _require(class_map.is_file(), f"evaluation class map is missing: {class_map}")
    return {
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config_sha256": evaluation_config_sha256(cfg.evaluation),
    }


def _validate_gate(path: str | Path, commit: str) -> dict[str, Any]:
    resolved, payload = _load_json(path, "CellCF real-loader CUDA gate")
    validated = validate_real_loader_gate_artifact(
        resolved,
        expected_commit=commit,
        expected_sha256=_sha256(resolved),
        require_clean=False,
    )
    protocol = protocol_from_environment()
    _require(
        validated.get("training_profile", "exposure132") == protocol.name,
        "CellCF gate training profile drift",
    )
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "synthetic_gate_sha256": validated["synthetic_gate_sha256"],
        "payload": payload,
    }


def _validate_pilot(
    path: str | Path,
    *,
    commit: str,
    gate_sha256: str,
    protocol_sha256: str,
    order_sha256: str,
) -> dict[str, Any]:
    resolved, payload = _load_json(path, "CellCF DDP pilot")
    validate_pilot_artifact(
        resolved,
        repo_root=ROOT,
        expected_commit=commit,
        expected_real_loader_gate_sha256=gate_sha256,
        require_clean=False,
    )
    _require(
        payload.get("training_profile", "exposure132")
        == protocol_from_environment().name,
        "CellCF pilot training profile drift",
    )
    _require(tuple(payload.get("variant_order", ())) == VARIANT_ORDER, "CellCF pilot arm order drift")
    return {"path": str(resolved), "sha256": _sha256(resolved), "payload": payload}


def _validate_post_run_artifacts(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    variant: str,
    commit: str,
    seed: int,
    config_path: Path,
    gate_path: Path,
    pilot_path: Path,
    annotation_path: Path,
    class_map_path: Path,
) -> dict[str, Any]:
    training_protocol = protocol_from_environment()
    hash_cache: dict[Path, str] = {}
    terminal_specs = {
        "checkpoint": ("checkpoint_path", "checkpoint_sha256"),
        "checkpoint_sidecar": (
            "checkpoint_sidecar_path",
            "checkpoint_sidecar_sha256",
        ),
        "training_audit": ("training_audit_path", "training_audit_sha256"),
        "terminal_evaluation": (
            "terminal_evaluation_path",
            "terminal_evaluation_sha256",
        ),
        "prediction": ("prediction_path", "prediction_sha256"),
        "run_manifest": ("run_manifest_path", "run_manifest_sha256"),
    }
    terminal: dict[str, dict[str, str]] = {}
    terminal_paths: dict[str, Path] = {}
    for name, (path_key, hash_key) in terminal_specs.items():
        artifact_path, artifact_sha256 = _require_hashed_file(
            payload,
            path_key,
            hash_key,
            f"{variant} {name.replace('_', ' ')}",
            hash_cache=hash_cache,
        )
        terminal_paths[name] = artifact_path
        terminal[name] = {"path": str(artifact_path), "sha256": artifact_sha256}

    _, manifest = _load_json(terminal_paths["run_manifest"], f"{variant} run manifest")
    _, audit = _load_json(terminal_paths["training_audit"], f"{variant} training audit")
    _, sidecar = _load_json(
        terminal_paths["checkpoint_sidecar"], f"{variant} checkpoint sidecar"
    )
    _, evaluation = _load_json(
        terminal_paths["terminal_evaluation"], f"{variant} terminal evaluation"
    )

    for label, document in (("run manifest", manifest), ("training audit", audit)):
        _require(document.get("variant") == variant, f"{variant}: {label} variant mismatch")
        _require(document.get("git_commit") == commit, f"{variant}: {label} commit mismatch")
        _require(document.get("seed") == seed, f"{variant}: {label} seed mismatch")
    _require(
        evaluation.get("git_commit") == commit,
        f"{variant}: terminal evaluation commit mismatch",
    )

    checkpoint_path = terminal_paths["checkpoint"]
    sidecar_checkpoint, _ = _require_hashed_file(
        sidecar,
        "checkpoint_path",
        "checkpoint_sha256",
        f"{variant} sidecar checkpoint",
        hash_cache=hash_cache,
    )
    evaluation_checkpoint, _ = _require_hashed_file(
        evaluation,
        "checkpoint_path",
        "checkpoint_sha256",
        f"{variant} evaluation checkpoint",
        hash_cache=hash_cache,
    )
    _require(
        sidecar_checkpoint == checkpoint_path == evaluation_checkpoint,
        f"{variant}: checkpoint path binding mismatch",
    )
    _require(
        int(evaluation.get("checkpoint_epoch", -1))
        == training_protocol.terminal_epoch,
        f"{variant}: terminal evaluation checkpoint epoch mismatch",
    )
    _require(
        evaluation.get("checkpoint_state_key")
        == training_protocol.terminal_state_key,
        f"{variant}: terminal evaluation checkpoint state mismatch",
    )
    evaluation_prediction, _ = _require_hashed_file(
        evaluation,
        "prediction_path",
        "prediction_sha256",
        f"{variant} evaluation prediction",
        hash_cache=hash_cache,
    )
    _require(
        evaluation_prediction == terminal_paths["prediction"],
        f"{variant}: prediction path binding mismatch",
    )

    metadata = sidecar.get("experiment_metadata")
    _require(isinstance(metadata, Mapping), f"{variant}: checkpoint metadata is missing")
    _require(
        metadata.get("training_audit") == audit,
        f"{variant}: checkpoint metadata audit mismatch",
    )

    expected_config = config_path.resolve()
    manifest_config, manifest_config_sha256 = _require_hashed_file(
        manifest,
        "config",
        "config_sha256",
        f"{variant} manifest config",
        base=repo_root,
        hash_cache=hash_cache,
    )
    audit_config, audit_config_sha256 = _require_hashed_file(
        audit,
        "source_config_path",
        "source_config_sha256",
        f"{variant} training config",
        hash_cache=hash_cache,
    )
    evaluation_config, evaluation_config_sha256 = _require_hashed_file(
        evaluation,
        "config_path",
        "config_sha256",
        f"{variant} evaluation config",
        hash_cache=hash_cache,
    )
    _require(
        manifest_config == audit_config == evaluation_config == expected_config,
        f"{variant}: source config path binding mismatch",
    )

    pretrain, pretrain_sha256 = _require_hashed_file(
        audit,
        "runtime_pretrain_path",
        "runtime_pretrain_sha256",
        f"{variant} runtime pretrain",
        hash_cache=hash_cache,
    )
    audit_gate, audit_gate_sha256 = _require_hashed_file(
        audit,
        "real_loader_gate_json",
        "real_loader_gate_sha256",
        f"{variant} real-loader gate",
        hash_cache=hash_cache,
    )
    audit_pilot, audit_pilot_sha256 = _require_hashed_file(
        audit,
        "ddp_pilot_json",
        "ddp_pilot_sha256",
        f"{variant} DDP pilot",
        hash_cache=hash_cache,
    )
    _require(audit_gate == gate_path.resolve(), f"{variant}: real-loader gate path mismatch")
    _require(audit_pilot == pilot_path.resolve(), f"{variant}: DDP pilot path mismatch")

    expected_annotation = annotation_path.resolve()
    expected_class_map = class_map_path.resolve()
    audit_annotation, audit_annotation_sha256 = _require_hashed_file(
        audit,
        "evaluation_annotation_path",
        "evaluation_annotation_sha256",
        f"{variant} audit annotation",
        hash_cache=hash_cache,
    )
    audit_class_map, audit_class_map_sha256 = _require_hashed_file(
        audit,
        "evaluation_class_map_path",
        "evaluation_class_map_sha256",
        f"{variant} audit class map",
        hash_cache=hash_cache,
    )
    evaluation_annotation, evaluation_annotation_sha256 = _require_hashed_file(
        evaluation,
        "evaluation_annotation_path",
        "evaluation_annotation_sha256",
        f"{variant} evaluation annotation",
        hash_cache=hash_cache,
    )
    evaluation_class_map, evaluation_class_map_sha256 = _require_hashed_file(
        evaluation,
        "evaluation_class_map_path",
        "evaluation_class_map_sha256",
        f"{variant} evaluation class map",
        hash_cache=hash_cache,
    )
    _require(
        audit_annotation == evaluation_annotation == expected_annotation,
        f"{variant}: evaluation annotation path mismatch",
    )
    _require(
        audit_class_map == evaluation_class_map == expected_class_map,
        f"{variant}: evaluation class-map path mismatch",
    )

    evaluation_config_payload = evaluation.get("evaluation_config")
    _require(
        isinstance(evaluation_config_payload, Mapping),
        f"{variant}: evaluation config payload is missing",
    )
    config_annotation = _resolve_referenced_path(
        evaluation_config_payload.get("ground_truth_filename"),
        label=f"{variant} evaluation-config annotation",
    )
    _require(
        config_annotation == expected_annotation,
        f"{variant}: evaluation-config annotation path mismatch",
    )
    optional_data: dict[str, str] | None = None
    blocked_videos = evaluation_config_payload.get("blocked_videos")
    if blocked_videos is not None:
        blocked_path = _resolve_referenced_path(
            blocked_videos, label=f"{variant} blocked-videos data"
        )
        blocked_sha256 = hash_cache.setdefault(blocked_path, _sha256(blocked_path))
        optional_data = {"path": str(blocked_path), "sha256": blocked_sha256}

    payload_evaluator = _require_evaluator_source(
        payload.get("evaluator"),
        label=f"{variant} post-run evaluator",
        hash_cache=hash_cache,
    )
    evaluation_evaluator = _require_evaluator_source(
        evaluation.get("evaluator"),
        label=f"{variant} terminal evaluator",
        hash_cache=hash_cache,
    )
    _require(
        payload.get("evaluator") == evaluation.get("evaluator"),
        f"{variant}: evaluator identity binding mismatch",
    )

    exposed_assets: dict[str, Any] = {
        "source_config": {
            "path": str(expected_config),
            "sha256": manifest_config_sha256,
        },
        "runtime_pretrain": {"path": str(pretrain), "sha256": pretrain_sha256},
        "real_loader_gate": {
            "path": str(audit_gate),
            "sha256": audit_gate_sha256,
        },
        "ddp_pilot": {"path": str(audit_pilot), "sha256": audit_pilot_sha256},
        "evaluation_annotation": {
            "path": str(expected_annotation),
            "sha256": audit_annotation_sha256,
        },
        "evaluation_class_map": {
            "path": str(expected_class_map),
            "sha256": audit_class_map_sha256,
        },
        "evaluator_source": payload_evaluator,
    }
    if optional_data is not None:
        exposed_assets["blocked_videos"] = optional_data
    _require(
        manifest_config_sha256 == audit_config_sha256 == evaluation_config_sha256,
        f"{variant}: source config hash binding mismatch",
    )
    _require(
        audit_annotation_sha256 == evaluation_annotation_sha256,
        f"{variant}: annotation hash binding mismatch",
    )
    _require(
        audit_class_map_sha256 == evaluation_class_map_sha256,
        f"{variant}: class-map hash binding mismatch",
    )
    _require(
        payload_evaluator == evaluation_evaluator,
        f"{variant}: evaluator source hash binding mismatch",
    )
    return {"terminal_artifacts": terminal, "exposed_assets": exposed_assets}


def _validate_post_run(
    path: str | Path,
    *,
    repo_root: Path,
    variant: str,
    commit: str,
    seed: int,
    config_path: Path,
    config_sha256: str,
    resolved_config_sha256: str,
    protocol_sha256: str,
    order_sha256: str,
    gate_sha256: str,
    pilot_sha256: str,
    gate_path: Path,
    pilot_path: Path,
    annotation_path: Path,
    annotation_sha256: str,
    class_map_path: Path,
    class_map_sha256: str,
    evaluation_config_sha256: str,
) -> dict[str, Any]:
    training_protocol = protocol_from_environment()
    resolved, payload = _load_json(path, f"{variant} post-run evidence")
    _require(payload.get("schema") == POST_RUN_SCHEMA, f"{variant}: bad post-run schema")
    _require(payload.get("ok") is True, f"{variant}: post-run evidence did not pass")
    expected = {
        "variant": variant,
        "git_commit": commit,
        "seed": seed,
        "config_sha256": config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "protocol_sha256": protocol_sha256,
        "ordered_exposure_sha256": order_sha256,
        "real_loader_gate_sha256": gate_sha256,
        "ddp_pilot_sha256": pilot_sha256,
        "evaluation_annotation_sha256": annotation_sha256,
        "evaluation_class_map_sha256": class_map_sha256,
        "evaluation_config_sha256": evaluation_config_sha256,
        "checkpoint_epoch": training_protocol.terminal_epoch,
        "checkpoint_state_key": training_protocol.terminal_state_key,
        "successful_optimizer_updates": (
            training_protocol.expected_successful_optimizer_updates
        ),
        "training_profile": training_protocol.name,
    }
    for key, value in expected.items():
        observed = (
            payload.get(key, "exposure132")
            if key == "training_profile"
            else payload.get(key)
        )
        _require(observed == value, f"{variant}: post-run {key} mismatch")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), f"{variant}: post-run metrics are missing")
    for key in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"):
        value = metrics.get(key)
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{variant}: invalid {key}")
    artifact_hash = payload.get("artifact_chain_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_chain_sha256", None)
    _require(artifact_hash == _canonical_sha256(unsigned), f"{variant}: post-run artifact hash mismatch")
    artifact_validation = _validate_post_run_artifacts(
        payload,
        repo_root=repo_root,
        variant=variant,
        commit=commit,
        seed=seed,
        config_path=config_path,
        gate_path=gate_path,
        pilot_path=pilot_path,
        annotation_path=annotation_path,
        class_map_path=class_map_path,
    )
    regenerated = finalize_cellcf_run(
        variant=variant,
        run_manifest_path=payload["run_manifest_path"],
        training_audit_path=payload["training_audit_path"],
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_sidecar_path=payload["checkpoint_sidecar_path"],
        evaluation_path=payload["terminal_evaluation_path"],
    )
    _require(regenerated == payload, f"{variant}: post-run evidence is not reproducible")
    checkpoint_contract = regenerated.get("checkpoint_payload_contract")
    _require(
        isinstance(checkpoint_contract, Mapping)
        and checkpoint_contract.get("payload_reopened") is True
        and checkpoint_contract.get("epoch")
        == training_protocol.terminal_epoch,
        f"{variant}: terminal checkpoint payload was not revalidated",
    )
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "metrics": dict(regenerated["metrics"]),
        "checkpoint_path": str(Path(regenerated["checkpoint_path"]).resolve()),
        "checkpoint_sha256": regenerated["checkpoint_sha256"],
        "artifact_revalidation": artifact_validation,
        "reproduced_from_terminal_artifacts": True,
    }


def _cost_profile_paths(payload: Mapping[str, Any], key: str) -> list[str]:
    values = payload.get(key)
    _require(
        isinstance(values, list)
        and len(values) >= 3
        and all(isinstance(item, str) and item for item in values),
        f"cost evidence {key} must contain at least three profile paths",
    )
    _require(len(set(values)) == len(values), f"cost evidence {key} contains duplicates")
    return list(values)


def _validate_cost_profile_artifacts(
    payload: Mapping[str, Any],
    *,
    group: str,
    paths: list[str],
) -> None:
    profile_artifacts = payload.get("profile_artifacts")
    _require(isinstance(profile_artifacts, Mapping), "cost profile artifact index is missing")
    records = profile_artifacts.get(group)
    _require(
        isinstance(records, list) and len(records) == len(paths),
        f"cost profile artifact index is incomplete for {group}",
    )
    for index, (record, value) in enumerate(zip(records, paths)):
        _require(
            isinstance(record, Mapping),
            f"cost profile artifact record is invalid for {group}[{index}]",
        )
        profile_path, _ = _require_hashed_file(
            record,
            "path",
            "sha256",
            f"cost profile artifact {group}[{index}]",
        )
        _require(
            profile_path == Path(value).resolve(),
            f"cost profile artifact path mismatch for {group}[{index}]",
        )


def _validate_cost_profile(
    path: str | Path,
    *,
    method: str,
    commit: str,
    expected_checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_config_path: Path,
    hash_cache: dict[Path, str],
) -> dict[str, str]:
    training_protocol = protocol_from_environment()
    profile_path, profile = _load_json(path, f"{method} cost profile")
    _require(profile.get("method") == method, f"unexpected cost method: {profile_path}")
    _require(profile.get("config_commit") == commit, f"cost profile commit mismatch: {profile_path}")
    _require(profile.get("uses_ema") is True, f"cost profile did not use EMA: {profile_path}")
    _require(profile.get("random_init") is False, f"cost profile used random init: {profile_path}")
    _require(
        int(profile.get("checkpoint_epoch", -1))
        == training_protocol.terminal_epoch,
        f"cost profile checkpoint epoch mismatch: {profile_path}",
    )
    _require(
        profile.get("checkpoint_state_key")
        == training_protocol.terminal_state_key,
        f"cost profile checkpoint state mismatch: {profile_path}",
    )
    checkpoint_path, checkpoint_sha256 = _require_hashed_file(
        profile,
        "checkpoint_path",
        "checkpoint_sha256",
        f"{method} cost checkpoint",
        hash_cache=hash_cache,
    )
    _require(
        checkpoint_path == expected_checkpoint_path.resolve()
        and checkpoint_sha256 == expected_checkpoint_sha256,
        f"cost profile used another CellCF checkpoint: {profile_path}",
    )
    config_path = _resolve_referenced_path(
        profile.get("config_path"), label=f"{method} cost config"
    )
    _require(
        config_path == expected_config_path.resolve(),
        f"cost profile used another config: {profile_path}",
    )
    if config_path in hash_cache:
        config_sha256 = hash_cache[config_path]
    else:
        config_sha256 = _sha256(config_path)
        hash_cache[config_path] = config_sha256
    _require(
        profile.get("profile_config_sha256") == config_sha256,
        f"cost profile config hash mismatch: {profile_path}",
    )
    return {
        "path": str(profile_path),
        "sha256": _sha256(profile_path),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
    }


def validate_cost_evidence(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_commit: str,
    expected_checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_post_run_evidence_path: str | Path,
    expected_post_run_evidence_sha256: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    resolved, payload = _load_json(path, "CellCF cost evidence")
    _require(payload.get("schema") == COST_EVIDENCE_SCHEMA, "bad CellCF cost evidence schema")
    _require(payload.get("ok") is True, "CellCF cost evidence did not pass")
    _require(
        payload.get("status") == "complete" and payload.get("pass") is True,
        "CellCF cost evidence is not a complete passing artifact",
    )
    _require(payload.get("config_commit") == expected_commit, "CellCF cost evidence commit mismatch")
    _require(
        payload.get("checkpoint_sha256") == expected_checkpoint_sha256,
        "CellCF cost evidence checkpoint mismatch",
    )
    evidence_hash = payload.get("evidence_sha256")
    unsigned = dict(payload)
    unsigned.pop("evidence_sha256", None)
    _require(
        evidence_hash == _canonical_sha256(unsigned),
        "CellCF cost evidence canonical hash mismatch",
    )
    post_run_path = Path(expected_post_run_evidence_path).resolve()
    _require(post_run_path.is_file(), f"CellCF post-run evidence is missing: {post_run_path}")
    _require(
        _sha256(post_run_path) == expected_post_run_evidence_sha256,
        "CellCF post-run evidence hash changed before cost validation",
    )
    cost_binding = payload.get("cellcf_cost_binding")
    _require(isinstance(cost_binding, Mapping), "CellCF cost binding is missing")
    _require(
        Path(str(cost_binding.get("post_run_evidence_path", ""))).resolve()
        == post_run_path,
        "CellCF cost binding uses another post-run evidence artifact",
    )
    _require(
        cost_binding.get("post_run_evidence_sha256")
        == expected_post_run_evidence_sha256,
        "CellCF cost binding post-run evidence hash mismatch",
    )
    cellcf_paths = _cost_profile_paths(payload, "cellcf_profile_paths")
    bare_paths = _cost_profile_paths(payload, "bare_uniform_profile_paths")
    _validate_cost_profile_artifacts(payload, group="cellcf", paths=cellcf_paths)
    _validate_cost_profile_artifacts(
        payload, group="bare_uniform", paths=bare_paths
    )
    hash_cache: dict[Path, str] = {}
    checkpoint_path = Path(expected_checkpoint_path).resolve()
    cellcf_profiles = [
        _validate_cost_profile(
            profile_path,
            method="cellcf-fixed384",
            commit=expected_commit,
            expected_checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_config_path=root / VARIANTS["cellcf"],
            hash_cache=hash_cache,
        )
        for profile_path in cellcf_paths
    ]
    bare_profiles = [
        _validate_cost_profile(
            profile_path,
            method="bare-uniform384",
            commit=expected_commit,
            expected_checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_config_path=root / BARE_COST_CONFIG,
            hash_cache=hash_cache,
        )
        for profile_path in bare_paths
    ]
    regenerated = summarize_cost_evidence(
        cellcf_paths,
        bare_paths,
        post_run_evidence_path=post_run_path,
        post_run_evidence_sha256=expected_post_run_evidence_sha256,
    )
    _require(regenerated == payload, "CellCF cost evidence is not reproducible")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "validated": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "post_run_evidence_path": str(post_run_path),
        "post_run_evidence_sha256": expected_post_run_evidence_sha256,
        "cellcf_profiles": cellcf_profiles,
        "bare_uniform_profiles": bare_profiles,
    }


def _suite_status(
    completed_runs: Mapping[str, Any],
    validated_cost: Mapping[str, Any],
) -> str:
    if validated_cost:
        return "complete"
    if completed_runs:
        return "runs_complete_cost_pending"
    return "deployable_not_submitted"


def validate_suite(
    *,
    repo_root: str | Path,
    seed: int,
    expected_commit: str | None,
    require_clean: bool,
    gate_json: str | Path,
    pilot_json: str | Path,
    post_run_evidence: Mapping[str, str | Path] | None = None,
    cost_evidence: str | Path | None = None,
    require_cost_evidence: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None:
        _require(commit == expected_commit, "HEAD differs from expected CellCF commit")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if require_clean:
        _require(not dirty, "CellCF suite requires a clean exact-commit tree")
    _require(seed >= 0, "seed must be non-negative")
    training_protocol = protocol_from_environment()

    configs = {
        variant: Config.fromfile(str(root / VARIANTS[variant]))
        for variant in VARIANT_ORDER
    }
    reference_protocol = _shared_protocol(configs[VARIANT_ORDER[0]])
    protocol_sha256 = _canonical_sha256(reference_protocol)
    order_sha256 = _canonical_sha256(list(VARIANT_ORDER))
    variants = []
    for variant in VARIANT_ORDER:
        cfg = configs[variant]
        validation = validate_config(variant, VARIANTS[variant])
        _require(_shared_protocol(cfg) == reference_protocol, f"{variant}: shared protocol drift")
        config_path = root / VARIANTS[variant]
        contract = _variant_contract(cfg, variant)
        variants.append(
            {
                "name": variant,
                "config": VARIANTS[variant].replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "resolved_config_sha256": _canonical_sha256(cfg.to_dict()),
                "variant_contract": contract,
                "variant_contract_sha256": _canonical_sha256(contract),
                "validation": validation,
            }
        )

    data = _reference_data(configs[VARIANT_ORDER[0]])
    gate = _validate_gate(gate_json, commit)
    gate_dataset = gate["payload"].get("dataset", {})
    _require(
        gate_dataset.get("annotation_sha256") == data["evaluation_annotation_sha256"],
        "CellCF gate annotation differs from the suite",
    )
    _require(
        gate_dataset.get("class_map_sha256") == data["evaluation_class_map_sha256"],
        "CellCF gate class map differs from the suite",
    )
    pilot = _validate_pilot(
        pilot_json,
        commit=commit,
        gate_sha256=gate["sha256"],
        protocol_sha256=protocol_sha256,
        order_sha256=order_sha256,
    )

    completed = {}
    if post_run_evidence is not None:
        _require(set(post_run_evidence) == set(VARIANT_ORDER), "post-run evidence must cover exactly three CellCF arms")
        completed = {
            variant: _validate_post_run(
                post_run_evidence[variant],
                repo_root=root,
                variant=variant,
                commit=commit,
                seed=seed,
                config_path=root / VARIANTS[variant],
                config_sha256=next(
                    item["config_sha256"]
                    for item in variants
                    if item["name"] == variant
                ),
                resolved_config_sha256=next(
                    item["resolved_config_sha256"]
                    for item in variants
                    if item["name"] == variant
                ),
                protocol_sha256=protocol_sha256,
                order_sha256=order_sha256,
                gate_sha256=gate["sha256"],
                pilot_sha256=pilot["sha256"],
                gate_path=Path(gate["path"]),
                pilot_path=Path(pilot["path"]),
                annotation_path=Path(data["evaluation_annotation_path"]),
                annotation_sha256=data["evaluation_annotation_sha256"],
                class_map_path=Path(data["evaluation_class_map_path"]),
                class_map_sha256=data["evaluation_class_map_sha256"],
                evaluation_config_sha256=data["evaluation_config_sha256"],
            )
            for variant in VARIANT_ORDER
        }

    _require(
        not require_cost_evidence or cost_evidence is not None,
        "required CellCF cost evidence is missing",
    )
    validated_cost: dict[str, Any] = {}
    if cost_evidence is not None:
        _require(
            bool(completed),
            "CellCF cost evidence requires all three validated post-run artifacts",
        )
        validated_cost = validate_cost_evidence(
            cost_evidence,
            repo_root=root,
            expected_commit=commit,
            expected_checkpoint_path=completed["cellcf"]["checkpoint_path"],
            expected_checkpoint_sha256=completed["cellcf"]["checkpoint_sha256"],
            expected_post_run_evidence_path=completed["cellcf"]["path"],
            expected_post_run_evidence_sha256=completed["cellcf"]["sha256"],
        )

    return {
        "schema": SCHEMA,
        "ok": True,
        "status": _suite_status(completed, validated_cost),
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "git_tree_clean": not bool(dirty),
        "seed": int(seed),
        "training_profile": training_protocol.name,
        "training_protocol": training_protocol.to_dict(),
        "variant_order": list(VARIANT_ORDER),
        "ordered_exposure_sha256": order_sha256,
        "shared_protocol": reference_protocol,
        "shared_protocol_sha256": protocol_sha256,
        "variants": variants,
        "reference_data_artifacts": data,
        "real_loader_gate": {key: value for key, value in gate.items() if key != "payload"},
        "ddp_pilot": {key: value for key, value in pilot.items() if key != "payload"},
        "completed_runs": completed,
        "cost_evidence_required": bool(require_cost_evidence),
        "cost_evidence": validated_cost,
        "submission_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--pilot-json", required=True)
    parser.add_argument("--post-run-evidence", action="append", default=[])
    parser.add_argument("--cost-evidence")
    parser.add_argument("--require-cost-evidence", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": "FileExistsError",
            "error": "refusing to overwrite suite evidence",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    try:
        post_runs = None
        if args.post_run_evidence:
            post_runs = {}
            for item in args.post_run_evidence:
                name, separator, path = item.partition("=")
                _require(bool(separator and name and path), "post-run evidence must be VARIANT=JSON")
                _require(name not in post_runs, f"duplicate post-run evidence for {name}")
                post_runs[name] = path
        payload = validate_suite(
            repo_root=args.repo_root,
            seed=args.seed,
            expected_commit=args.expected_commit,
            require_clean=args.require_clean,
            gate_json=args.gate_json,
            pilot_json=args.pilot_json,
            post_run_evidence=post_runs,
            cost_evidence=args.cost_evidence,
            require_cost_evidence=args.require_cost_evidence,
        )
        code = 0
    except Exception as exc:
        payload = {"schema": SCHEMA, "ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(output + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
