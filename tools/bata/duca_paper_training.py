from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.bata import duca_p0_training as legacy


FORMAL_PROTOCOL = "duca_paper_full200_actionformer_v1"
MATRIX_SCHEMA = "duca_paper_full200_matrix_v1"
TRAINING_RECEIPT_SCHEMA = "duca_paper_full200_training_receipt_v1"
EVALUATION_SCHEMA = "duca_paper_full211_terminal_evaluation_v1"
DUCA_TRAINING_AUDIT_FILENAME = "duca_paper_training_audit.json"
ARMS = (
    "dense",
    "uniform_fixed_k384",
    "uniform_mixed_train_k384_eval",
    "duca_fixed_k384",
)
SEEDS = (5801, 8123, 12011)
TRAIN_VIDEO_COUNT = 200
EVALUATION_VIDEO_COUNT = 211
WORLD_SIZE = 2
GLOBAL_BATCH_SIZE = 2
PER_RANK_BATCH_SIZE = 1
EPOCHS = 60
UPDATES_PER_EPOCH = 100
SUCCESSFUL_UPDATES = EPOCHS * UPDATES_PER_EPOCH

DUCA_P0_CHECKPOINT_METADATA_SCHEMA = legacy.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA = legacy.DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA
atomic_write_json = legacy.atomic_write_json
build_checkpoint_metadata = legacy.build_checkpoint_metadata
capture_global_rng_state = legacy.capture_global_rng_state
new_update_audit = legacy.new_update_audit
restore_global_rng_state = legacy.restore_global_rng_state
restore_training_state = legacy.restore_training_state


def is_formal_protocol(value: Any) -> bool:
    return str(value) == FORMAL_PROTOCOL


def _canonical_sha256(value: Any) -> str:
    return legacy.canonical_sha256(value)


canonical_sha256 = legacy.canonical_sha256


def _sha256_file(path: str | Path) -> str:
    return legacy.sha256_file(Path(path).expanduser().resolve())


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"DUCA paper {label} SHA-256 is invalid")
    return digest


def _bound_file(path_value: Any, digest_value: Any, label: str) -> tuple[Path, str]:
    path = Path(str(path_value or "")).expanduser().resolve()
    digest = _require_sha256(digest_value, label)
    if not path.is_file() or _sha256_file(path) != digest:
        raise RuntimeError(f"DUCA paper {label} artifact drift")
    return path, digest


def _load_bound_json(
    path_value: Any,
    digest_value: Any,
    *,
    schema: str,
    label: str,
) -> tuple[dict[str, Any], Path, str]:
    path, digest = _bound_file(path_value, digest_value, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise RuntimeError(f"DUCA paper {label} schema drift")
    return payload, path, digest


def _config_value(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def validate_static_config(cfg) -> dict[str, Any]:
    workflow = cfg.workflow
    if str(workflow.get("formal_protocol", "")) != FORMAL_PROTOCOL:
        raise ValueError("DUCA paper formal protocol is not active")
    cell = cfg.get("duca_paper_cell", None)
    if not isinstance(cell, Mapping):
        raise ValueError("DUCA paper config lacks its cell identity")
    arm = str(cell.get("arm", ""))
    if arm not in ARMS:
        raise ValueError(f"DUCA paper arm is not registered: {arm}")

    train = cfg.dataset.train
    test = cfg.dataset.test
    common_invalid = (
        str(cfg.model.type) != "ActionFormer"
        or str(train.type) != "DucaStatelessThumosPaddingDataset"
        or int(train.stateless_seed) != 3407
        or str(train.subset_name) != "training"
        or train.get("block_list", None) is not None
        or cfg.dataset.val is not None
        or str(test.subset_name) != "validation"
        or test.get("block_list", None) is not None
        or test.get("test_mode", None) is not True
        or int(test.window_size) != 768
        or str(cfg.evaluation.subset) != "validation"
        or cfg.evaluation.get("blocked_videos", None) is not None
        or cfg.post_processing.get("save_dict", None) is not True
        or int(cfg.solver.train.batch_size) != GLOBAL_BATCH_SIZE
        or int(cfg.solver.val.batch_size) != 1
        or int(cfg.solver.test.batch_size) != 1
        or int(workflow.end_epoch) != EPOCHS
        or int(workflow.expected_train_batches_per_epoch) != UPDATES_PER_EPOCH
        or int(workflow.expected_successful_optimizer_updates) != SUCCESSFUL_UPDATES
        or int(workflow.checkpoint_interval) != 5
        or int(workflow.get("checkpoint_retention", 0)) != 1
        or int(workflow.val_loss_interval) >= 0
        or int(workflow.val_eval_interval) >= 0
        or int(workflow.val_start_epoch) < 9999
        or workflow.get("seal_eval_dataloaders_during_training", None) is not True
        or workflow.get("derive_train_loader_contract", None) is not True
        or workflow.get("formal_successful_update_contract", None) is not True
        or int(workflow.primary_checkpoint_epoch) != 59
        or str(workflow.primary_checkpoint_state_key) != "state_dict_ema"
        or str(workflow.checkpoint_criterion)
        != "terminal_epoch_59_state_dict_ema"
        or workflow.get("max_train_iters", None) is not None
        or int(cell.get("train_video_count", -1)) != TRAIN_VIDEO_COUNT
        or int(cell.get("evaluation_video_count", -1))
        != EVALUATION_VIDEO_COUNT
        or int(cell.get("world_size", -1)) != WORLD_SIZE
        or int(cell.get("global_batch_size", -1)) != GLOBAL_BATCH_SIZE
        or int(cell.get("successful_updates", -1)) != SUCCESSFUL_UPDATES
        or str(cell.get("detector_backend", "")) != "ActionFormer"
    )
    if common_invalid:
        raise ValueError("DUCA paper full-200/exact-211 common contract drift")

    selector = cfg.model.get("frame_selector", None)
    backbone = cfg.model.backbone
    projection = cfg.model.projection
    physical_head = cfg.model.rpn_head.get("physical_grid_actionformer", None)
    dynamic_backbone = bool(backbone.custom.get("dynamic_temporal_bucket", False))
    if arm == "dense":
        invalid = (
            selector is not None
            or dynamic_backbone
            or int(backbone.backbone.total_frames) != 768
            or int(projection.max_seq_len) != 768
            or physical_head is not None
            or int(cell.get("evaluation_heavy_k", -1)) != 768
        )
        selector_schedule_enabled = False
    elif arm == "uniform_fixed_k384":
        invalid = (
            str(_config_value(selector, "type", ""))
            != "DucaProtectedE2EFrameSelector"
            or str(_config_value(selector, "arm", "")) != "exact_uniform"
            or _config_value(selector, "actionness_source_cfg", object()) is not None
            or int(_config_value(selector, "budget", -1)) != 384
            or str(_config_value(selector, "detector_coordinate_mode", ""))
            != "selected_axis_plugin"
            or not dynamic_backbone
            or int(backbone.backbone.total_frames) != 384
            or int(projection.max_seq_len) != 512
            or physical_head is not None
            or int(cell.get("evaluation_heavy_k", -1)) != 384
        )
        selector_schedule_enabled = False
    elif arm == "uniform_mixed_train_k384_eval":
        invalid = (
            str(_config_value(selector, "type", "")) != "DucaRimeFrameSelector"
            or str(_config_value(selector, "rime_arm", "")) != "uniform_mixed_k"
            or tuple(int(v) for v in _config_value(selector, "candidate_budgets", ()))
            != (192, 256, 384, 512)
            or tuple(int(v) for v in _config_value(selector, "mixed_k_schedule_counts", ()))
            != (8, 12, 16, 24)
            or int(_config_value(selector, "mixed_k_schedule_seed", -1)) != 3407
            or int(_config_value(selector, "fixed_budget", -1)) != 384
            or _config_value(selector, "actionness_source_cfg", object()) is not None
            or str(_config_value(selector, "detector_coordinate_mode", ""))
            != "selected_axis_plugin"
            or not dynamic_backbone
            or int(backbone.backbone.total_frames) != 512
            or int(projection.max_seq_len) != 512
            or physical_head is not None
            or float(cell.get("training_mean_heavy_k", -1)) != 384.0
            or int(cell.get("evaluation_heavy_k", -1)) != 384
        )
        selector_schedule_enabled = True
    else:
        source = _config_value(selector, "actionness_source_cfg", None)
        invalid = (
            str(_config_value(selector, "type", "")) != "DucaRimeFrameSelector"
            or str(_config_value(selector, "rime_arm", "")) != "fixed_bound"
            or int(_config_value(selector, "fixed_budget", -1)) != 384
            or _config_value(selector, "require_frozen_protocol", None) is not False
            or str(_config_value(selector, "detector_coordinate_mode", ""))
            != "selected_axis_plugin"
            or not isinstance(source, Mapping)
            or str(source.get("probe_model", "")) != "official-action-seg"
            or str(source.get("official_action_seg_backend", ""))
            != "official_asformer"
            or source.get("frozen", None) is not False
            or source.get("trainable", None) is not True
            or source.get("require_checkpoint", None) is not False
            or str(source.get("training_supervision_scope", "")) != "train_only"
            or not dynamic_backbone
            or int(backbone.backbone.total_frames) != 512
            or int(projection.max_seq_len) != 512
            or physical_head is not None
            or int(cell.get("evaluation_heavy_k", -1)) != 384
        )
        selector_schedule_enabled = True
    if invalid:
        raise ValueError(f"DUCA paper {arm} model contract drift")

    return {
        "formal_protocol": FORMAL_PROTOCOL,
        "variant": arm,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "per_rank_batch_size": PER_RANK_BATCH_SIZE,
        "train_video_count": TRAIN_VIDEO_COUNT,
        "evaluation_video_count": EVALUATION_VIDEO_COUNT,
        "expected_train_batches_per_epoch": UPDATES_PER_EPOCH,
        "expected_successful_optimizer_updates": SUCCESSFUL_UPDATES,
        "end_epoch": EPOCHS,
        "selector_schedule_enabled": selector_schedule_enabled,
        "expected_selector_schedule_updates": (
            SUCCESSFUL_UPDATES if selector_schedule_enabled else 0
        ),
        "checkpoint_retention": 1,
    }


def formal_training_contract(cfg) -> dict[str, Any] | None:
    if not is_formal_protocol(cfg.workflow.get("formal_protocol", "")):
        return None
    base = legacy.formal_training_contract(
        cfg,
        expected_checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    )
    if base is None:
        raise ValueError("DUCA paper successful-update contract is not active")
    contract = validate_static_config(cfg)
    contract.update(base)
    contract["train_loader_contract"] = None
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
        if path not in allowed or (
            allowed[path] is not None and value is not allowed[path]
        ):
            rejected.append(path)
    if rejected:
        raise RuntimeError(
            f"{entrypoint} rejected DUCA paper cfg overrides: "
            + ", ".join(sorted(rejected))
        )


def validate_evaluation_request(
    cfg,
    *,
    arm: str,
    seed: int,
    expected_checkpoint_epoch: int | None,
    checkpoint_state_key: str,
    metrics_json: str | None,
) -> dict[str, Any]:
    contract = validate_static_config(cfg)
    if (
        str(arm) != contract["variant"]
        or int(seed) not in SEEDS
        or expected_checkpoint_epoch != 59
        or checkpoint_state_key != "state_dict_ema"
        or not metrics_json
    ):
        raise RuntimeError(
            "DUCA paper evaluation requires a registered arm/seed, exact "
            "211-video validation, and terminal epoch-59 EMA"
        )
    return {
        "arm": str(arm),
        "seed": int(seed),
        "evaluation_heavy_k": int(cfg.duca_paper_cell.evaluation_heavy_k),
        "evaluation_video_count": EVALUATION_VIDEO_COUNT,
    }


def _dataset_identity(dataset) -> dict[str, Any]:
    video_ids = tuple(str(row[0]) for row in dataset.data_list)
    if len(video_ids) != TRAIN_VIDEO_COUNT or len(set(video_ids)) != TRAIN_VIDEO_COUNT:
        raise RuntimeError("DUCA paper training dataset is not the exact 200-video set")
    return {
        "dataset_class": dataset.__class__.__name__,
        "subset_name": str(dataset.subset_name),
        "video_count": len(video_ids),
        "ordered_video_ids": list(video_ids),
        "ordered_video_ids_sha256": _canonical_sha256(video_ids),
        "annotation_path": str(Path(dataset.ann_file).expanduser().resolve()),
        "annotation_sha256": _sha256_file(dataset.ann_file),
        "stateless_seed": int(dataset.stateless_seed),
    }


def derive_train_loader_contract(
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> dict[str, Any]:
    import torch

    if train_dataset.__class__.__name__ != "DucaStatelessThumosPaddingDataset":
        raise RuntimeError("DUCA paper training requires the stateless THUMOS dataset")
    if int(world_size) != WORLD_SIZE or int(train_dataset.stateless_seed) != 3407:
        raise RuntimeError("DUCA paper training requires world-size two and data seed 3407")
    sampler = getattr(train_loader, "sampler", None)
    if (
        sampler is None
        or sampler.__class__.__name__ != "DistributedSampler"
        or int(getattr(sampler, "num_replicas", -1)) != WORLD_SIZE
        or int(getattr(sampler, "rank", -1)) not in {0, 1}
        or getattr(sampler, "shuffle", None) is not True
        or getattr(sampler, "drop_last", None) is not True
        or int(len(train_loader)) != UPDATES_PER_EPOCH
        or int(cfg.solver.train.batch_size) != GLOBAL_BATCH_SIZE
    ):
        raise RuntimeError("DUCA paper two-rank loader contract drift")

    identity = _dataset_identity(train_dataset)
    video_ids = tuple(identity["ordered_video_ids"])
    sampler_seed = int(sampler.seed)
    epoch_rank_index_sha256 = []
    exposure_counts = {video_id: 0 for video_id in video_ids}
    for epoch in range(EPOCHS):
        rank_indices = []
        for rank in range(WORLD_SIZE):
            replay = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=WORLD_SIZE,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=sampler_seed,
            )
            replay.set_epoch(epoch)
            indices = tuple(int(index) for index in replay)
            if len(indices) != UPDATES_PER_EPOCH or len(set(indices)) != len(indices):
                raise RuntimeError("DUCA paper rank exposure is not one hundred unique videos")
            rank_indices.append(indices)
        merged = rank_indices[0] + rank_indices[1]
        if len(merged) != TRAIN_VIDEO_COUNT or set(merged) != set(range(TRAIN_VIDEO_COUNT)):
            raise RuntimeError("DUCA paper two ranks do not cover every train video once per epoch")
        epoch_rank_index_sha256.append(
            [_canonical_sha256(indices) for indices in rank_indices]
        )
        for index in merged:
            exposure_counts[video_ids[index]] += 1
    if set(exposure_counts.values()) != {EPOCHS}:
        raise RuntimeError("DUCA paper global per-video exposure is not exactly sixty")

    payload = {
        "schema_version": "duca_paper_full200_train_loader_v1",
        "dataset": identity,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "per_rank_batch_size": PER_RANK_BATCH_SIZE,
        "batches_per_rank_per_epoch": UPDATES_PER_EPOCH,
        "epochs": EPOCHS,
        "successful_optimizer_updates": SUCCESSFUL_UPDATES,
        "global_video_exposures": TRAIN_VIDEO_COUNT * EPOCHS,
        "per_video_exposure_count": EPOCHS,
        "sampler_seed": sampler_seed,
        "sampler_drop_last": True,
        "sampler_shuffle": True,
        "epoch_rank_index_sha256": epoch_rank_index_sha256,
        "dataset_config_sha256": _canonical_sha256(cfg.dataset.train.to_dict()),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def bind_train_loader_contract(
    contract: Mapping[str, Any],
    *,
    cfg,
    train_dataset,
    train_loader,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader_contract = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=train_dataset,
        train_loader=train_loader,
        world_size=world_size,
    )
    bound = dict(contract)
    bound["train_loader_contract"] = loader_contract
    return bound, loader_contract


def _validate_matrix_manifest(
    *,
    git_commit: str,
    arm: str,
    seed: int,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    manifest, path, digest = _load_bound_json(
        os.environ.get("DUCA_PAPER_MATRIX_MANIFEST"),
        os.environ.get("DUCA_PAPER_MATRIX_MANIFEST_SHA256"),
        schema=MATRIX_SCHEMA,
        label="matrix manifest",
    )
    if (
        manifest.get("git_commit") != git_commit
        or manifest.get("task") != "offline_temporal_action_detection"
        or tuple(manifest.get("arms", ())) != ARMS
        or tuple(int(v) for v in manifest.get("seeds", ())) != SEEDS
        or int(manifest.get("train_video_count", -1)) != TRAIN_VIDEO_COUNT
        or int(manifest.get("evaluation_video_count", -1))
        != EVALUATION_VIDEO_COUNT
        or int(manifest.get("world_size", -1)) != WORLD_SIZE
        or int(manifest.get("global_batch_size", -1)) != GLOBAL_BATCH_SIZE
        or int(manifest.get("successful_updates", -1)) != SUCCESSFUL_UPDATES
        or manifest.get("training_consumes_validation", None) is not False
    ):
        raise RuntimeError("DUCA paper matrix manifest contract drift")
    record = manifest.get("configs", {}).get(arm)
    repo_root = Path(__file__).resolve().parents[2]
    source = Path(source_config_path).expanduser().resolve()
    try:
        relative = source.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise RuntimeError("DUCA paper source config is outside the exact checkout") from exc
    if (
        not isinstance(record, Mapping)
        or record.get("path") != relative
        or record.get("sha256") != source_config_sha256
        or record.get("resolved_sha256") != resolved_config_sha256
        or _sha256_file(source) != source_config_sha256
        or {"arm": arm, "seed": int(seed)} not in manifest.get("cells", ())
    ):
        raise RuntimeError("DUCA paper runtime cell differs from the matrix manifest")
    return manifest, path, digest


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
    if variant not in ARMS or int(seed) not in SEEDS:
        raise ValueError("DUCA paper runtime arm/seed is outside the frozen matrix")
    if re.fullmatch(r"[0-9a-f]{40}", str(git_commit)) is None:
        raise ValueError("DUCA paper runtime requires an exact Git commit")
    if resolved_config_sha256 != os.environ.get("DUCA_PAPER_RESOLVED_CONFIG_SHA256"):
        raise RuntimeError("DUCA paper resolved config differs from the sealed launch")
    manifest, manifest_path, manifest_sha = _validate_matrix_manifest(
        git_commit=str(git_commit),
        arm=str(variant),
        seed=int(seed),
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        resolved_config_sha256=resolved_config_sha256,
    )
    pretrain, pretrain_sha = _bound_file(
        runtime_pretrain_path,
        os.environ.get("DUCA_PAPER_PRETRAIN_SHA256"),
        "VideoMAE initialization",
    )
    annotation, annotation_sha = _bound_file(
        evaluation_annotation_path,
        os.environ.get("DUCA_PAPER_ANNOTATION_SHA256"),
        "THUMOS14 annotation",
    )
    class_map, class_map_sha = _bound_file(
        evaluation_class_map_path,
        os.environ.get("DUCA_PAPER_CLASS_MAP_SHA256"),
        "THUMOS14 class map",
    )
    assets = manifest.get("assets", {})
    if (
        assets.get("pretrain_sha256") != pretrain_sha
        or assets.get("annotation_sha256") != annotation_sha
        or assets.get("class_map_sha256") != class_map_sha
        or str(evaluation_config.get("subset", "")) != "validation"
        or evaluation_config.get("blocked_videos", None) is not None
    ):
        raise RuntimeError("DUCA paper asset/evaluation binding drift")
    bindings = {
        "git_commit": str(git_commit),
        "variant": str(variant),
        "seed": int(seed),
        "slurm_job_id": None if slurm_job_id is None else str(slurm_job_id),
        "formal_protocol": FORMAL_PROTOCOL,
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": str(source_config_sha256),
        "resolved_config_sha256": str(resolved_config_sha256),
        "runtime_config_sha256": str(runtime_config_sha256),
        "matrix_manifest_path": str(manifest_path),
        "matrix_manifest_sha256": manifest_sha,
        "pretrain_path": str(pretrain),
        "pretrain_sha256": pretrain_sha,
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": annotation_sha,
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": class_map_sha,
        "evaluation_config_sha256": _canonical_sha256(dict(evaluation_config)),
        "train_video_count": TRAIN_VIDEO_COUNT,
        "evaluation_video_count": EVALUATION_VIDEO_COUNT,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "training_consumes_validation": False,
    }
    bindings["binding_sha256"] = _canonical_sha256(bindings)
    return bindings


def selector_schedule_step(model) -> int:
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    if selector is None:
        return 0
    variant = str(getattr(selector, "selector_variant", ""))
    if variant == "duca_rime_selected_axis":
        step = getattr(selector, "_loss_weight_schedule_step", None)
        if step is None or int(step.numel()) != 1:
            raise RuntimeError("DUCA paper RIME selector lacks its schedule buffer")
        return int(step.detach().item())
    if variant == "protected_e2e_selected_axis":
        if str(getattr(selector, "arm", "")) != "exact_uniform":
            raise RuntimeError("DUCA paper protected selector is not exact uniform")
        return 0
    raise RuntimeError("DUCA paper model exposes an unregistered selector")


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
    expected_batches = UPDATES_PER_EPOCH
    if int(train_batches_per_epoch) != expected_batches:
        raise RuntimeError("DUCA paper loader no longer has 100 updates per rank/epoch")
    expected = (int(epoch) + 1) * expected_batches
    successful = int(update_audit.get("successful_optimizer_updates", -1))
    if successful != expected or int(update_audit.get("attempted_batches", -1)) != expected:
        raise RuntimeError("DUCA paper successful-update exposure drift")
    skipped = int(update_audit.get("amp_skipped_attempts", -1))
    if int(update_audit.get("optimizer_attempts", -1)) != successful + skipped:
        raise RuntimeError("DUCA paper optimizer-attempt accounting drift")
    if int(update_audit.get("replay_exhaustions", -1)) != 0:
        raise RuntimeError("DUCA paper AMP replay was exhausted")
    if int(update_audit.get("scheduler_updates", -1)) != successful:
        raise RuntimeError("DUCA paper scheduler exposure drift")
    if int(update_audit.get("ema_updates", -1)) != (successful if uses_ema else 0):
        raise RuntimeError("DUCA paper EMA exposure drift")
    expected_selector = expected if contract["selector_schedule_enabled"] else 0
    if (
        int(update_audit.get("duca_schedule_updates", -1)) != expected_selector
        or int(selector_step) != expected_selector
    ):
        raise RuntimeError("DUCA paper selector schedule exposure drift")
    if int(scheduler_last_epoch) != successful:
        raise RuntimeError("DUCA paper scheduler state drift")
    if int(update_audit.get("max_amp_retries_observed", -1)) > int(
        contract["max_amp_retries_per_batch"]
    ):
        raise RuntimeError("DUCA paper AMP retry bound was exceeded")


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
        raise RuntimeError("DUCA paper epoch records are incomplete")
    payload = {
        "schema_version": legacy.DUCA_P0_TRAINING_AUDIT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        **dict(bindings),
        "train_loader_contract": contract["train_loader_contract"],
        "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
        "primary_checkpoint_epoch": 59,
        "primary_checkpoint_state_key": "state_dict_ema",
        "expected_train_batches_per_epoch": UPDATES_PER_EPOCH,
        "expected_successful_optimizer_updates": SUCCESSFUL_UPDATES,
        "selector_schedule_enabled": bool(contract["selector_schedule_enabled"]),
        "expected_selector_schedule_updates": int(
            contract["expected_selector_schedule_updates"]
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
    payload["audit_sha256"] = _canonical_sha256(payload)
    return payload


def after_checkpoint_saved(
    *,
    checkpoint_path: str | Path,
    work_dir: str | Path,
    epoch: int,
    contract: Mapping[str, Any],
) -> list[str]:
    if int(contract.get("checkpoint_retention", 0)) != 1:
        raise RuntimeError("DUCA paper rolling checkpoint retention is not active")
    checkpoint_dir = (Path(work_dir).expanduser().resolve() / "checkpoint").resolve()
    current = Path(checkpoint_path).expanduser().resolve()
    if (
        current.parent != checkpoint_dir
        or current.name != f"epoch_{int(epoch)}.pth"
        or not current.is_file()
        or current.is_symlink()
    ):
        raise RuntimeError("DUCA paper checkpoint retention received an unsafe path")
    removed = []
    for candidate in sorted(checkpoint_dir.glob("epoch_*.pth")):
        match = re.fullmatch(r"epoch_([0-9]+)\.pth", candidate.name)
        if candidate.resolve() == current or candidate.is_symlink() or match is None:
            continue
        if int(match.group(1)) >= int(epoch):
            raise RuntimeError("DUCA paper checkpoint directory contains a future epoch")
        candidate.unlink()
        removed.append(str(candidate.resolve()))
    remaining = sorted(
        path.resolve() for path in checkpoint_dir.glob("epoch_*.pth") if path.is_file()
    )
    if remaining != [current]:
        raise RuntimeError("DUCA paper checkpoint retention failed closed")
    return removed


def _validate_embedded_hash(payload: Mapping[str, Any], key: str, label: str) -> None:
    unsigned = dict(payload)
    expected = unsigned.pop(key, None)
    if expected != _canonical_sha256(unsigned):
        raise RuntimeError(f"DUCA paper {label} self-hash drift")


def validate_terminal_checkpoint_binding(
    *,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    git_commit: str,
    arm: str,
    seed: int,
    source_config_path: str | Path,
    source_config_sha256: str,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    receipt, receipt_path, receipt_sha = _load_bound_json(
        os.environ.get("DUCA_PAPER_TRAINING_RECEIPT"),
        os.environ.get("DUCA_PAPER_TRAINING_RECEIPT_SHA256"),
        schema=TRAINING_RECEIPT_SCHEMA,
        label="training receipt",
    )
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha = _sha256_file(checkpoint_file)
    if (
        receipt.get("status") != "passed"
        or receipt.get("git_commit") != git_commit
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("train_video_count", -1)) != TRAIN_VIDEO_COUNT
        or int(receipt.get("world_size", -1)) != WORLD_SIZE
        or int(receipt.get("global_batch_size", -1)) != GLOBAL_BATCH_SIZE
        or int(receipt.get("successful_optimizer_updates", -1))
        != SUCCESSFUL_UPDATES
        or receipt.get("training_consumed_validation", None) is not False
        or Path(str(receipt.get("checkpoint_path", ""))).resolve()
        != checkpoint_file
        or receipt.get("checkpoint_sha256") != checkpoint_sha
        or int(receipt.get("checkpoint_epoch", -1)) != 59
        or receipt.get("checkpoint_state_key") != "state_dict_ema"
    ):
        raise RuntimeError("DUCA paper training receipt/checkpoint binding drift")
    compaction, compaction_path, compaction_sha = _load_bound_json(
        receipt.get("checkpoint_compaction_receipt_path"),
        receipt.get("checkpoint_compaction_receipt_sha256"),
        schema="duca_rime_compact_checkpoint_receipt_v1",
        label="checkpoint compaction receipt",
    )
    if (
        compaction.get("status") != "passed"
        or compaction.get("git_commit") != git_commit
        or compaction.get("variant") != arm
        or int(compaction.get("seed", -1)) != int(seed)
        or Path(str(compaction.get("compact_checkpoint_path", ""))).resolve()
        != checkpoint_file
        or compaction.get("compact_checkpoint_sha256") != checkpoint_sha
        or compaction.get("evaluation_equivalent") is not True
    ):
        raise RuntimeError("DUCA paper compact checkpoint binding drift")
    metadata = checkpoint.get("experiment_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version")
        != legacy.DUCA_P0_CHECKPOINT_METADATA_SCHEMA
    ):
        raise RuntimeError("DUCA paper terminal checkpoint metadata is missing")
    _validate_embedded_hash(metadata, "metadata_sha256", "checkpoint metadata")
    audit = metadata.get("training_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != legacy.DUCA_P0_TRAINING_AUDIT_SCHEMA
    ):
        raise RuntimeError("DUCA paper terminal training audit is missing")
    _validate_embedded_hash(audit, "audit_sha256", "training audit")
    expected = {
        "status": "complete",
        "git_commit": git_commit,
        "variant": arm,
        "seed": int(seed),
        "formal_protocol": FORMAL_PROTOCOL,
        "source_config_path": str(Path(source_config_path).resolve()),
        "source_config_sha256": source_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "train_video_count": TRAIN_VIDEO_COUNT,
        "evaluation_video_count": EVALUATION_VIDEO_COUNT,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "expected_train_batches_per_epoch": UPDATES_PER_EPOCH,
        "expected_successful_optimizer_updates": SUCCESSFUL_UPDATES,
        "last_completed_epoch": 59,
        "epochs_completed": EPOCHS,
        "train_batches_per_epoch": UPDATES_PER_EPOCH,
        "scheduler_last_epoch": SUCCESSFUL_UPDATES,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(f"DUCA paper terminal identity drift: {key}")
    counters = audit.get("update_audit", {})
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
    ):
        if int(counters.get(key, -1)) != SUCCESSFUL_UPDATES:
            raise RuntimeError(f"DUCA paper terminal counter drift: {key}")
    expected_selector = (
        SUCCESSFUL_UPDATES if bool(audit.get("selector_schedule_enabled")) else 0
    )
    if (
        int(counters.get("duca_schedule_updates", -1)) != expected_selector
        or int(audit.get("selector_schedule_step", -1)) != expected_selector
    ):
        raise RuntimeError("DUCA paper terminal selector schedule drift")
    return {
        "training_receipt_path": str(receipt_path),
        "training_receipt_sha256": receipt_sha,
        "checkpoint_compaction_receipt_path": str(compaction_path),
        "checkpoint_compaction_receipt_sha256": compaction_sha,
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha,
        "arm": arm,
        "seed": int(seed),
        "successful_optimizer_updates": SUCCESSFUL_UPDATES,
        "train_video_count": TRAIN_VIDEO_COUNT,
        "world_size": WORLD_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "training_consumed_validation": False,
    }


def validate_official_evaluation_execution(
    *,
    evaluation_summary: Mapping[str, Any],
    annotation_path: str | Path,
    prediction_path: str | Path,
) -> dict[str, Any]:
    annotation = Path(annotation_path).expanduser().resolve()
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    database = payload.get("database", {})
    expected = {
        str(video_id)
        for video_id, row in database.items()
        if str(row.get("subset", "")) == "validation"
    }
    execution = evaluation_summary.get("post_processing_execution", {})
    observed = set(execution.get("window_counts", {}))
    prediction = Path(prediction_path).expanduser().resolve()
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    predicted = set(prediction_payload.get("results", {}))
    if (
        len(expected) != EVALUATION_VIDEO_COUNT
        or observed != expected
        or predicted != expected
        or int(evaluation_summary.get("video_count", -1)) != EVALUATION_VIDEO_COUNT
        or int(execution.get("world_size", -1)) != 1
        or execution.get("dataset_is_sliding_window", None) is not True
        or execution.get(
            "full_detector_window_merge_nms_evaluation_completed", None
        )
        is not True
        or execution.get("evaluator_evaluate_succeeded", None) is not True
        or str(execution.get("evaluation_config", {}).get("subset", ""))
        != "validation"
        or execution.get("evaluation_config", {}).get("blocked_videos", None)
        is not None
    ):
        raise RuntimeError(
            "DUCA paper evaluation did not execute the exact 211-video official set"
        )
    return {
        "schema_version": "duca_paper_exact211_execution_v1",
        "evaluation_video_count": EVALUATION_VIDEO_COUNT,
        "evaluation_video_ids_sha256": _canonical_sha256(sorted(expected)),
        "annotation_path": str(annotation),
        "annotation_sha256": _sha256_file(annotation),
        "prediction_path": str(prediction),
        "prediction_sha256": _sha256_file(prediction),
        "official_open_tad_pipeline_completed": True,
    }


__all__ = [
    "ARMS",
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "EVALUATION_SCHEMA",
    "FORMAL_PROTOCOL",
    "MATRIX_SCHEMA",
    "SEEDS",
    "TRAINING_RECEIPT_SCHEMA",
    "after_checkpoint_saved",
    "assert_safe_cfg_options",
    "atomic_write_json",
    "bind_train_loader_contract",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_training_audit",
    "canonical_sha256",
    "capture_global_rng_state",
    "derive_train_loader_contract",
    "formal_training_contract",
    "is_formal_protocol",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "validate_official_evaluation_execution",
    "validate_evaluation_request",
    "validate_static_config",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
