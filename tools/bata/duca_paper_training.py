from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.bata import duca_p0_training as legacy


FORMAL_PROTOCOL = "duca_paper_full200_actionformer_v2"
MATRIX_SCHEMA = "duca_paper_full200_matrix_v2"
TRAINING_RECEIPT_SCHEMA = "duca_paper_full200_training_receipt_v2"
EVALUATION_SCHEMA = "duca_paper_full211_terminal_evaluation_v2"
DUCA_TRAINING_AUDIT_FILENAME = "duca_paper_training_audit.json"
BUDGET_SEMANTICS = "requested_then_deterministic_feasibility_shrink_v1"
BUDGET_EPOCH_SCHEMA = "duca_paper_committed_budget_epoch_v1"
BUDGET_SUMMARY_SCHEMA = "duca_paper_committed_budget_summary_v1"
EVALUATION_BUDGET_SCHEMA = "duca_paper_exact211_budget_execution_v1"
EXECUTION_QUANTUM = 16
MIXED_K_CANDIDATES = (192, 256, 384, 512)
MIXED_K_COUNTS = (8, 12, 16, 24)
MIXED_K_SEED = 3407
MIXED_K_NOMINAL_REQUESTED_MEAN = 384.0
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


def mixed_k_requested_schedule() -> tuple[int, ...]:
    tokens = [
        (
            hashlib.sha256(
                f"{MIXED_K_SEED}|{budget}|{occurrence}".encode("utf-8")
            ).digest(),
            budget,
        )
        for budget, count in zip(MIXED_K_CANDIDATES, MIXED_K_COUNTS)
        for occurrence in range(count)
    ]
    tokens.sort(key=lambda item: item[0])
    cycle = tuple(int(budget) for _digest, budget in tokens)
    if (
        len(cycle) != sum(MIXED_K_COUNTS)
        or tuple(cycle.count(value) for value in MIXED_K_CANDIDATES)
        != MIXED_K_COUNTS
        or sum(cycle) / len(cycle) != MIXED_K_NOMINAL_REQUESTED_MEAN
    ):
        raise RuntimeError("DUCA paper mixed-K requested schedule drift")
    return cycle


def mixed_k_requested_schedule_sha256() -> str:
    return _canonical_sha256(
        {
            "candidate_budgets": MIXED_K_CANDIDATES,
            "candidate_costs": tuple(float(value) for value in MIXED_K_CANDIDATES),
            "schedule_counts": MIXED_K_COUNTS,
            "schedule_seed": MIXED_K_SEED,
            "cycle": mixed_k_requested_schedule(),
            "target_mean_cost": MIXED_K_NOMINAL_REQUESTED_MEAN,
        }
    )


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
    budget_contract = cfg.get("duca_paper_budget_contract", None)
    if (
        not isinstance(budget_contract, Mapping)
        or str(budget_contract.get("semantics", "")) != BUDGET_SEMANTICS
        or int(budget_contract.get("execution_quantum", -1)) != EXECUTION_QUANTUM
        or str(budget_contract.get("valid_length_definition", ""))
        != "contiguous_true_dense_candidate_prefix"
        or str(budget_contract.get("subquantum_policy", ""))
        != "fail_closed_below_one_quantum"
        or budget_contract.get("padding_or_repetition_allowed", None) is not False
        or budget_contract.get("length_conditioned_requested_schedule", None)
        is not False
        or budget_contract.get(
            "fixed_requested_k384_evaluation_is_dynamic",
            None,
        )
        is not False
    ):
        raise ValueError("DUCA paper requested/effective budget semantics drift")

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
            != MIXED_K_CANDIDATES
            or tuple(int(v) for v in _config_value(selector, "mixed_k_schedule_counts", ()))
            != MIXED_K_COUNTS
            or int(_config_value(selector, "mixed_k_schedule_seed", -1))
            != MIXED_K_SEED
            or int(_config_value(selector, "fixed_budget", -1)) != 384
            or int(_config_value(selector, "execution_quantum", -1))
            != EXECUTION_QUANTUM
            or _config_value(selector, "actionness_source_cfg", object()) is not None
            or str(_config_value(selector, "detector_coordinate_mode", ""))
            != "selected_axis_plugin"
            or not dynamic_backbone
            or int(backbone.backbone.total_frames) != 512
            or int(projection.max_seq_len) != 512
            or physical_head is not None
            or float(cell.get("nominal_requested_mean_k", -1))
            != MIXED_K_NOMINAL_REQUESTED_MEAN
            or str(cell.get("realized_training_mean_k", ""))
            != "measured_from_committed_backbone_rows"
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
        "budget_semantics": BUDGET_SEMANTICS,
        "execution_quantum": EXECUTION_QUANTUM,
        "mixed_k_requested_schedule_sha256": (
            mixed_k_requested_schedule_sha256()
            if arm == "uniform_mixed_train_k384_eval"
            else None
        ),
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
    budget = manifest.get("budget_semantics", {})
    mixed = budget.get("mixed_k", {}) if isinstance(budget, Mapping) else {}
    if (
        budget.get("version") != BUDGET_SEMANTICS
        or int(budget.get("execution_quantum", -1)) != EXECUTION_QUANTUM
        or budget.get("padding_or_repetition_allowed", None) is not False
        or budget.get("length_conditioned_requested_schedule", None) is not False
        or budget.get("fixed_requested_k384_evaluation_is_dynamic", None)
        is not False
        or tuple(int(value) for value in mixed.get("candidate_budgets", ()))
        != MIXED_K_CANDIDATES
        or tuple(int(value) for value in mixed.get("schedule_counts", ()))
        != MIXED_K_COUNTS
        or int(mixed.get("schedule_seed", -1)) != MIXED_K_SEED
        or tuple(int(value) for value in mixed.get("cycle", ()))
        != mixed_k_requested_schedule()
        or mixed.get("schedule_sha256") != mixed_k_requested_schedule_sha256()
        or float(mixed.get("nominal_requested_mean_k", -1))
        != MIXED_K_NOMINAL_REQUESTED_MEAN
    ):
        raise RuntimeError("DUCA paper matrix budget semantics drift")
    prerequisite_gates = manifest.get("prerequisite_gates", {})
    code_gate = prerequisite_gates.get("clean_linux_pytorch_code")
    code_gate_path = (
        Path(str(code_gate.get("path", ""))).expanduser().resolve()
        if isinstance(code_gate, Mapping)
        else None
    )
    if (
        not isinstance(code_gate, Mapping)
        or code_gate.get("schema_version") != "duca_paper_clean_linux_code_gate_v2"
        or code_gate.get("status") != "passed"
        or code_gate.get("git_commit") != git_commit
        or code_gate.get("claim_scope")
        != "engineering_clean_linux_pytorch_code_only"
        or code_gate.get("performance_evidence") is not False
        or code_gate_path is None
        or not code_gate_path.is_file()
        or code_gate.get("sha256") != _sha256_file(code_gate_path)
        or str(code_gate_path)
        != str(
            Path(os.environ.get("DUCA_PAPER_CODE_GATE_RECEIPT", ""))
            .expanduser()
            .resolve()
        )
        or code_gate.get("sha256")
        != os.environ.get("DUCA_PAPER_CODE_GATE_RECEIPT_SHA256")
    ):
        raise RuntimeError("DUCA paper clean Linux prerequisite gate drift")
    gate = prerequisite_gates.get(
        "real_natural_short_window_heavy_backbone"
    )
    gate_path = Path(str(gate.get("path", ""))).expanduser().resolve() if isinstance(gate, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema_version")
        != "duca_paper_real_short_window_heavy_backbone_gate_v1"
        or gate.get("status") != "passed"
        or gate.get("git_commit") != git_commit
        or gate.get("claim_scope") != "engineering_short_window_execution_only"
        or gate.get("performance_evidence") is not False
        or gate_path is None
        or not gate_path.is_file()
        or gate.get("sha256") != _sha256_file(gate_path)
        or str(gate_path)
        != str(Path(os.environ.get("DUCA_PAPER_SHORT_WINDOW_GATE_JSON", "")).expanduser().resolve())
        or gate.get("sha256")
        != os.environ.get("DUCA_PAPER_SHORT_WINDOW_GATE_SHA256")
    ):
        raise RuntimeError("DUCA paper short-window prerequisite gate drift")
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


def _expected_effective_k(requested_k: int, dense_valid_len: int) -> int:
    requested = int(requested_k)
    valid_len = int(dense_valid_len)
    available = valid_len - valid_len % EXECUTION_QUANTUM
    effective = min(requested, available)
    effective -= effective % EXECUTION_QUANTUM
    if effective <= 0:
        raise RuntimeError(
            "DUCA paper natural window is shorter than one execution quantum"
        )
    return effective


def build_epoch_budget_audit(
    *,
    arm: str,
    epoch: int,
    rows: list[Mapping[str, Any]],
    ordered_video_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    arm = str(arm)
    epoch = int(epoch)
    if arm == "dense":
        if rows:
            raise RuntimeError("dense Stage-A arm cannot expose selector budget rows")
        return {
            "schema_version": BUDGET_EPOCH_SCHEMA,
            "arm": arm,
            "epoch": epoch,
            "mode": "dense_temporal_axis",
            "row_count": 0,
            "nominal_backbone_input_k": 768,
            "budget_semantics": "dense_t768_reference",
            "rows": [],
            "rows_sha256": _canonical_sha256([]),
        }
    if arm not in ARMS:
        raise RuntimeError("DUCA paper budget audit received an unknown arm")
    if len(rows) != TRAIN_VIDEO_COUNT or len(ordered_video_ids) != TRAIN_VIDEO_COUNT:
        raise RuntimeError(
            "DUCA paper epoch budget audit must cover exactly 200 training videos"
        )
    schedule = mixed_k_requested_schedule()
    normalized = []
    observed_indices = set()
    for source in rows:
        row = dict(source)
        if row.get("schema_version") != "duca_paper_committed_budget_row_v1":
            raise RuntimeError("DUCA paper committed budget row schema drift")
        row_epoch = int(row.get("duca_stateless_epoch", -1))
        sample_index = int(row.get("duca_stateless_sample_index", -1))
        video_id = str(row.get("video_id", ""))
        valid_len = int(row.get("dense_valid_len", -1))
        requested = int(row.get("requested_k", -1))
        effective = int(row.get("effective_k", -1))
        unique = int(row.get("unique_k", -1))
        backbone = int(row.get("backbone_input_k", -1))
        backbone_source = str(
            row.get("backbone_input_measurement_source", "")
        )
        backbone_contract_sha256 = str(
            row.get("backbone_input_contract_sha256", "")
        )
        padded = int(row.get("padded_k", -1))
        quantum = int(row.get("execution_quantum", -1))
        selected = [int(value) for value in row.get("selected_dense_indices", ())]
        expected_requested = (
            int(schedule[(epoch + sample_index) % len(schedule)])
            if arm == "uniform_mixed_train_k384_eval"
            else 384
        )
        expected_effective = _expected_effective_k(expected_requested, valid_len)
        if (
            row_epoch != epoch
            or sample_index < 0
            or sample_index >= TRAIN_VIDEO_COUNT
            or sample_index in observed_indices
            or video_id != str(ordered_video_ids[sample_index])
            or quantum != EXECUTION_QUANTUM
            or row.get("budget_semantics") != BUDGET_SEMANTICS
            or requested != expected_requested
            or effective != expected_effective
            or unique != expected_effective
            or backbone != expected_effective
            or backbone_source
            != "actual_backbone_wrapper_and_videomae_input_tensors"
            or re.fullmatch(r"[0-9a-f]{64}", backbone_contract_sha256) is None
            or padded != expected_effective
            or len(selected) != expected_effective
            or selected != sorted(set(selected))
            or any(value < 0 or value >= valid_len for value in selected)
        ):
            raise RuntimeError(
                "DUCA paper committed budget row violates its frozen natural-window "
                "requested/effective/backbone contract"
            )
        observed_indices.add(sample_index)
        normalized.append(
            {
                "rank": int(row.get("rank", -1)),
                "video_id": video_id,
                "window_start_frame": int(row.get("window_start_frame", 0)),
                "duca_stateless_epoch": epoch,
                "duca_stateless_sample_index": sample_index,
                "dense_valid_len": valid_len,
                "execution_quantum": quantum,
                "requested_k": requested,
                "effective_k": effective,
                "unique_k": unique,
                "backbone_input_k": backbone,
                "backbone_input_measurement_source": backbone_source,
                "backbone_input_contract_sha256": backbone_contract_sha256,
                "padded_k": padded,
                "selected_dense_indices_sha256": _canonical_sha256(selected),
            }
        )
    if observed_indices != set(range(TRAIN_VIDEO_COUNT)):
        raise RuntimeError("DUCA paper epoch budget rows do not cover indices 0..199")
    normalized.sort(key=lambda row: int(row["duca_stateless_sample_index"]))
    requested_histogram = Counter(int(row["requested_k"]) for row in normalized)
    effective_histogram = Counter(int(row["effective_k"]) for row in normalized)
    return {
        "schema_version": BUDGET_EPOCH_SCHEMA,
        "arm": arm,
        "epoch": epoch,
        "mode": "selector_no_padding_backbone_execution",
        "budget_semantics": BUDGET_SEMANTICS,
        "execution_quantum": EXECUTION_QUANTUM,
        "row_count": len(normalized),
        "requested_histogram": {
            str(key): requested_histogram[key] for key in sorted(requested_histogram)
        },
        "effective_histogram": {
            str(key): effective_histogram[key] for key in sorted(effective_histogram)
        },
        "observed_requested_mean_k": sum(requested_histogram.elements())
        / len(normalized),
        "realized_backbone_mean_k": sum(
            int(row["backbone_input_k"]) for row in normalized
        )
        / len(normalized),
        "rows": normalized,
        "rows_sha256": _canonical_sha256(normalized),
    }


def collect_epoch_budget_audit(
    *,
    model,
    contract: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    import torch.distributed as dist

    arm = str(contract.get("variant", ""))
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    local_rows: list[dict[str, Any]] = []
    if selector is not None:
        drain = getattr(selector, "drain_committed_budget_rows", None)
        if not callable(drain):
            raise RuntimeError("DUCA paper selector lacks committed budget rows")
        local_rows = [dict(row) for row in drain()]
    if not dist.is_available() or not dist.is_initialized():
        if int(contract.get("world_size", -1)) != 1:
            raise RuntimeError("DUCA paper budget audit requires initialized DDP")
        gathered = [local_rows]
    else:
        if dist.get_world_size() != WORLD_SIZE:
            raise RuntimeError("DUCA paper budget audit requires two DDP ranks")
        gathered = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(gathered, local_rows)
    rows = []
    for rank, rank_rows in enumerate(gathered):
        if not isinstance(rank_rows, list):
            raise RuntimeError("DUCA paper gathered budget rows are malformed")
        for source in rank_rows:
            row = dict(source)
            row["rank"] = rank
            rows.append(row)
    loader = contract.get("train_loader_contract")
    ordered_video_ids = (
        loader.get("dataset", {}).get("ordered_video_ids", ())
        if isinstance(loader, Mapping)
        else ()
    )
    return build_epoch_budget_audit(
        arm=arm,
        epoch=epoch,
        rows=rows,
        ordered_video_ids=ordered_video_ids,
    )


def summarize_budget_epoch_records(
    *,
    arm: str,
    epoch_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    audits = [dict(record.get("budget_audit", {})) for record in epoch_records]
    if any(audit.get("schema_version") != BUDGET_EPOCH_SCHEMA for audit in audits):
        raise RuntimeError("DUCA paper epoch budget evidence is incomplete")
    if arm == "dense":
        if any(
            audit.get("mode") != "dense_temporal_axis"
            or int(audit.get("nominal_backbone_input_k", -1)) != 768
            for audit in audits
        ):
            raise RuntimeError("DUCA paper dense budget evidence drift")
        payload = {
            "schema_version": BUDGET_SUMMARY_SCHEMA,
            "arm": arm,
            "mode": "dense_temporal_axis",
            "epochs": len(audits),
            "row_count": 0,
            "nominal_backbone_input_k": 768,
            "budget_semantics": "dense_t768_reference",
            "epoch_rows_sha256": [audit["rows_sha256"] for audit in audits],
        }
    else:
        rows = [row for audit in audits for row in audit.get("rows", ())]
        expected_count = len(audits) * TRAIN_VIDEO_COUNT
        if len(rows) != expected_count:
            raise RuntimeError("DUCA paper aggregate budget row count drift")
        requested_histogram = Counter(int(row["requested_k"]) for row in rows)
        effective_histogram = Counter(int(row["effective_k"]) for row in rows)
        cross = Counter(
            (int(row["requested_k"]), int(row["effective_k"])) for row in rows
        )
        shrink_count = sum(
            1 for row in rows if int(row["effective_k"]) < int(row["requested_k"])
        )
        payload = {
            "schema_version": BUDGET_SUMMARY_SCHEMA,
            "arm": arm,
            "mode": "selector_no_padding_backbone_execution",
            "budget_semantics": BUDGET_SEMANTICS,
            "execution_quantum": EXECUTION_QUANTUM,
            "epochs": len(audits),
            "row_count": len(rows),
            "nominal_requested_mean_k": MIXED_K_NOMINAL_REQUESTED_MEAN,
            "observed_requested_mean_k": sum(
                int(row["requested_k"]) for row in rows
            )
            / len(rows),
            "realized_effective_mean_k": sum(
                int(row["effective_k"]) for row in rows
            )
            / len(rows),
            "realized_backbone_mean_k": sum(
                int(row["backbone_input_k"]) for row in rows
            )
            / len(rows),
            "requested_histogram": {
                str(key): requested_histogram[key]
                for key in sorted(requested_histogram)
            },
            "effective_histogram": {
                str(key): effective_histogram[key]
                for key in sorted(effective_histogram)
            },
            "requested_effective_crosstab": {
                f"{requested}->{effective}": cross[(requested, effective)]
                for requested, effective in sorted(cross)
            },
            "feasibility_shrink_count": shrink_count,
            "feasibility_shrink_rate": shrink_count / len(rows),
            "mixed_k_requested_schedule_sha256": (
                mixed_k_requested_schedule_sha256()
                if arm == "uniform_mixed_train_k384_eval"
                else None
            ),
            "epoch_rows_sha256": [audit["rows_sha256"] for audit in audits],
            "all_rows_sha256": _canonical_sha256(rows),
        }
    payload["budget_summary_sha256"] = _canonical_sha256(payload)
    return payload


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
    budget_ledger_summary = summarize_budget_epoch_records(
        arm=str(contract["variant"]),
        epoch_records=epoch_records,
    )
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
        "budget_ledger_summary": budget_ledger_summary,
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
    budget_summary = audit.get("budget_ledger_summary")
    if not isinstance(budget_summary, Mapping):
        raise RuntimeError("DUCA paper terminal budget ledger summary is missing")
    unsigned_budget = dict(budget_summary)
    budget_summary_sha256 = unsigned_budget.pop("budget_summary_sha256", None)
    expected_budget_rows = 0 if arm == "dense" else TRAIN_VIDEO_COUNT * EPOCHS
    if (
        budget_summary.get("schema_version") != BUDGET_SUMMARY_SCHEMA
        or budget_summary.get("arm") != arm
        or int(budget_summary.get("epochs", -1)) != EPOCHS
        or int(budget_summary.get("row_count", -1)) != expected_budget_rows
        or budget_summary_sha256 != _canonical_sha256(unsigned_budget)
        or receipt.get("budget_summary_sha256") != budget_summary_sha256
    ):
        raise RuntimeError("DUCA paper terminal budget ledger summary drift")
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
        "budget_ledger_summary": dict(budget_summary),
        "budget_summary_sha256": str(budget_summary_sha256),
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


def validate_evaluation_budget_execution(
    *,
    arm: str,
    evaluation_summary: Mapping[str, Any],
    ledger_root: str | Path | None,
    protocol_sha256: str,
) -> dict[str, Any]:
    arm = str(arm)
    protocol_sha = _require_sha256(protocol_sha256, "evaluation budget protocol")
    execution = evaluation_summary.get("post_processing_execution", {})
    window_counts = execution.get("window_counts", {})
    if not isinstance(window_counts, Mapping) or len(window_counts) != EVALUATION_VIDEO_COUNT:
        raise RuntimeError("DUCA paper evaluation budget lacks exact window counts")
    if arm == "dense":
        if ledger_root:
            root = Path(ledger_root).expanduser().resolve()
            if root.exists() and any(root.iterdir()):
                raise RuntimeError("dense Stage-A evaluation cannot expose selector ledgers")
        payload = {
            "schema_version": EVALUATION_BUDGET_SCHEMA,
            "arm": arm,
            "mode": "dense_t768_reference",
            "requested_budget_is_dynamic": False,
            "nominal_backbone_input_k": 768,
            "window_count": sum(int(value) for value in window_counts.values()),
            "protocol_sha256": protocol_sha,
            "window_budget_vector_sha256": None,
        }
        payload["content_sha256"] = _canonical_sha256(payload)
        return payload

    root = Path(str(ledger_root or "")).expanduser().resolve()
    files = sorted(root.glob("inference_ledger.rank*.jsonl")) if root.is_dir() else []
    if not files:
        raise RuntimeError("DUCA paper selector evaluation budget ledger is missing")
    expected_selector_arm = {
        "uniform_fixed_k384": "exact_uniform",
        "uniform_mixed_train_k384_eval": "uniform_mixed_k",
        "duca_fixed_k384": "fixed_bound",
    }.get(arm)
    if expected_selector_arm is None:
        raise RuntimeError("DUCA paper evaluation budget received an unknown arm")
    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"DUCA paper budget ledger JSON drift: {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError("DUCA paper budget ledger row is not a mapping")
                rows.append(row)
    expected_row_count = sum(int(value) for value in window_counts.values())
    if len(rows) != expected_row_count:
        raise RuntimeError("DUCA paper evaluation budget row count drift")
    keys = set()
    per_video = Counter()
    vectors = []
    effective_histogram = Counter()
    for row in rows:
        video_id = str(row.get("video_id", ""))
        window_start = int(row.get("window_start_frame", -1))
        key = (video_id, window_start)
        requested = int(row.get("requested_k", -1))
        valid_len = int(row.get("dense_valid_len", -1))
        effective = int(row.get("effective_k", -1))
        unique = int(row.get("unique_k", -1))
        backbone = int(row.get("backbone_input_k", -1))
        padded = int(row.get("padded_k", -1))
        selected = [int(value) for value in row.get("selected_dense_indices", ())]
        expected_effective = _expected_effective_k(384, valid_len)
        if (
            row.get("schema_version") != "duca_rime_inference_ledger_v1"
            or str(row.get("arm", "")) != expected_selector_arm
            or row.get("budget_protocol_sha256") != protocol_sha
            or row.get("backbone_input_measurement_source")
            != "actual_backbone_wrapper_and_videomae_input_tensors"
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(row.get("backbone_input_contract_sha256", "")),
            )
            or not video_id
            or video_id not in window_counts
            or window_start < 0
            or key in keys
            or requested != 384
            or effective != expected_effective
            or unique != expected_effective
            or backbone != expected_effective
            or padded != expected_effective
            or len(selected) != expected_effective
            or selected != sorted(set(selected))
            or any(value < 0 or value >= valid_len for value in selected)
        ):
            raise RuntimeError(
                "DUCA paper evaluation budget violates fixed-requested-K384 "
                "natural-window no-padding execution"
            )
        keys.add(key)
        per_video[video_id] += 1
        effective_histogram[effective] += 1
        vectors.append(
            {
                "video_id": video_id,
                "window_start_frame": window_start,
                "effective_k": effective,
                "backbone_input_k": backbone,
            }
        )
    if {
        str(video_id): int(count) for video_id, count in per_video.items()
    } != {str(video_id): int(count) for video_id, count in window_counts.items()}:
        raise RuntimeError("DUCA paper evaluation budget/video window coverage drift")
    vectors.sort(key=lambda row: (row["video_id"], row["window_start_frame"]))
    payload = {
        "schema_version": EVALUATION_BUDGET_SCHEMA,
        "arm": arm,
        "mode": "fixed_requested_k384_natural_window_feasibility",
        "budget_semantics": BUDGET_SEMANTICS,
        "execution_quantum": EXECUTION_QUANTUM,
        "requested_budget_is_dynamic": False,
        "requested_k": 384,
        "window_count": len(vectors),
        "evaluation_video_count": len(per_video),
        "realized_backbone_mean_k": sum(
            int(row["backbone_input_k"]) for row in vectors
        )
        / len(vectors),
        "effective_histogram": {
            str(key): effective_histogram[key] for key in sorted(effective_histogram)
        },
        "feasibility_shrink_count": sum(
            1 for row in vectors if int(row["effective_k"]) < 384
        ),
        "window_budget_vector_sha256": _canonical_sha256(vectors),
        "ledger_files": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in files
        ],
        "protocol_sha256": protocol_sha,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ARMS",
    "DUCA_P0_CHECKPOINT_METADATA_SCHEMA",
    "DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA",
    "DUCA_TRAINING_AUDIT_FILENAME",
    "EVALUATION_SCHEMA",
    "EVALUATION_BUDGET_SCHEMA",
    "FORMAL_PROTOCOL",
    "MATRIX_SCHEMA",
    "SEEDS",
    "TRAINING_RECEIPT_SCHEMA",
    "BUDGET_SEMANTICS",
    "BUDGET_SUMMARY_SCHEMA",
    "after_checkpoint_saved",
    "assert_safe_cfg_options",
    "atomic_write_json",
    "bind_train_loader_contract",
    "build_checkpoint_metadata",
    "build_runtime_bindings",
    "build_epoch_budget_audit",
    "build_training_audit",
    "canonical_sha256",
    "capture_global_rng_state",
    "derive_train_loader_contract",
    "collect_epoch_budget_audit",
    "formal_training_contract",
    "is_formal_protocol",
    "new_update_audit",
    "restore_global_rng_state",
    "restore_training_state",
    "selector_schedule_step",
    "summarize_budget_epoch_records",
    "mixed_k_requested_schedule",
    "mixed_k_requested_schedule_sha256",
    "validate_evaluation_budget_execution",
    "validate_official_evaluation_execution",
    "validate_evaluation_request",
    "validate_static_config",
    "validate_terminal_checkpoint_binding",
    "validate_update_state",
]
