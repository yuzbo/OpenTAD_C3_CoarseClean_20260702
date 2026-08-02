"""No-performance real-data health contract for dynamic SCNR Stage-1.

This module deliberately stops before validation, inference, checkpointing, or
metric computation.  It admits the dynamic ROI + TokenSelect implementation to
matched performance experiments only after 64 successful optimizer updates keep
the exact global token budget, true ragged executor, AMP, losses, and gradients
healthy on the frozen development-fit population.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.georoute_experiment_contract import (
    assert_development_annotation,
    canonical_sha256,
    load_development_manifest,
    sha256_file,
)


DYNAMIC_POLICY_HEALTH_SCHEMA = "georoute_dynamic_policy_health_v1"
DYNAMIC_POLICY_HEALTH_BINDING_SCHEMA = "georoute_dynamic_policy_health_binding_v1"
DYNAMIC_POLICY_HEALTH_STUDY_ID = "scnr-dynamic-stage1-policy-health-v1"
DYNAMIC_POLICY_HEALTH_PASS = "PASS_NO_PERFORMANCE_DYNAMIC_POLICY_HEALTH"
DYNAMIC_POLICY_HEALTH_HOLD = "HOLD_DYNAMIC_POLICY_HEALTH"
DYNAMIC_POLICY_HEALTH_FAIL = "FAIL_DYNAMIC_POLICY_HEALTH_EXECUTION"
DYNAMIC_ROUTING_SCHEMA = "georoute_dynamic_global_routing_v2"
RAGGED_EXECUTOR_SCHEMA = "videomae_native_ragged_v1"

HEALTH_SEED = 4423
TARGET_SUCCESSFUL_UPDATES = 64
WINDOW_TUBELETS = 384
NATIVE_SPATIAL_CANDIDATES = 11 * 20
WINDOW_TOKEN_BUDGET = WINDOW_TUBELETS * 64
INITIAL_LOSS_SCALE = 65536.0
MINIMUM_LOSS_SCALE = 1024.0
MAX_AMP_RETRIES_PER_BATCH = 4
MAX_TOTAL_SKIPPED_ATTEMPTS = 16
MIN_COMPONENT_HEALTHY_UPDATES = 60
MAX_DOMINANT_ROLE_FRACTION = 0.995
SOFT_BUDGET_ABSOLUTE_TOLERANCE = 0.05

ROLE_NAMES = ("context", "roi", "residual")
REQUIRED_LOSS_KEYS = {
    "cls_loss",
    "reg_loss",
    "georoute_geometry_regularization_loss",
    "georoute_dynamic_auxiliary_loss",
    "georoute_dynamic_soft_proxy_loss",
    "cost",
}
REQUIRED_GRADIENT_COMPONENTS = {
    "dynamic_aux_head",
    "projection",
    "rpn_head",
    "scout_base_utility",
    "scout_geometry",
    "scout_residual",
    "scout_stem",
    "sparse_adapter",
    "videomae_adapter",
}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _full_hex(value: Any, *, length: int, name: str) -> str:
    text = str(value).lower()
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return text


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _finite_number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_clean_git_checkout(*, expected_commit: str, root: Path) -> dict[str, Any]:
    import subprocess

    expected_commit = _full_hex(expected_commit, length=40, name="expected_commit")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD").lower()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    branch = git("branch", "--show-current")
    if head != expected_commit or status:
        raise RuntimeError(
            "dynamic policy health requires its exact clean runtime commit"
        )
    if branch:
        origin_ref_name = f"origin/{branch}"
        origin_candidates = [origin_ref_name]
    else:
        origin_candidates = [
            line.strip()
            for line in git(
                "for-each-ref",
                "--contains",
                "HEAD",
                "--format=%(refname:short)",
                "refs/remotes/origin",
            ).splitlines()
            if line.strip() and line.strip() != "origin/HEAD"
        ]
        origin_ref_name = next(
            (
                candidate
                for candidate in origin_candidates
                if git("rev-parse", candidate).lower() == expected_commit
            ),
            "",
        )
    origin_ref = git("rev-parse", origin_ref_name).lower() if origin_ref_name else ""
    if origin_ref != expected_commit:
        raise RuntimeError("dynamic policy health requires origin-ref parity")
    return {
        "commit": head,
        "expected_commit": expected_commit,
        "branch": branch,
        "detached_head": not bool(branch),
        "origin_ref_name": origin_ref_name,
        "origin_candidates": origin_candidates,
        "origin_ref": origin_ref,
        "head_matches_expected": True,
        "origin_ref_matches_expected": True,
        "tree_clean": True,
    }


def require_slurm_single_gpu() -> dict[str, Any]:
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not job_id.isdigit():
        raise RuntimeError("dynamic policy health requires a numeric Slurm Job ID")
    if not visible or "," in visible:
        raise RuntimeError("dynamic policy health requires one Slurm-visible GPU")
    return {
        "job_id": job_id,
        "logical_device": "cuda:0",
        "visible_device_count": 1,
        "cuda_visible_devices_sha256": canonical_sha256({"value": visible}),
    }


def bind_dynamic_policy_health_config(
    *,
    source_config_path: str | Path,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    runtime_commit: str,
    seed: int = HEALTH_SEED,
):
    """Bind the approved dynamic route to one immutable fit-only health run."""

    from mmengine.config import Config

    if int(seed) != HEALTH_SEED:
        raise ValueError("dynamic policy-health seed is frozen")
    runtime_commit = _full_hex(runtime_commit, length=40, name="runtime_commit")
    source_config = Path(source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    annotation = assert_development_annotation(development_annotation_path)
    manifest = load_development_manifest(manifest_path)
    fit_ids = list(manifest["splits"]["fit"])
    gate_ids = list(manifest["splits"]["gate"])
    if not (set(fit_ids) | set(gate_ids)) <= set(annotation["video_ids"]):
        raise ValueError("policy-health manifest names videos absent from annotation")
    class_map = Path(class_map_path).resolve()
    video_root = Path(development_video_root).resolve()
    pretrained = Path(pretrained_checkpoint_path).resolve()
    for file_path in (source_config, class_map, pretrained):
        if not file_path.is_file() or file_path.is_symlink():
            raise FileNotFoundError(file_path)
    if not video_root.is_dir() or "test" in video_root.name.lower():
        raise ValueError(
            "policy-health video root must exist and must not be a test root"
        )
    work_dir = Path(work_dir).resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if (
        os.name != "nt"
        and str(os.environ.get("SLURM_JOB_ID", "")).isdigit()
        and not _inside(work_dir, write_boundary)
    ):
        raise ValueError(
            "policy-health work directory leaves the remote write boundary"
        )

    cfg = Config.fromfile(str(source_config))
    custom = cfg.model.backbone.custom
    if (
        custom.get("georoute_route_mode") != "dynamic_scnr"
        or custom.get("georoute_policy_estimator") != "straight_through"
        or int(custom.get("georoute_window_token_budget", -1)) != WINDOW_TOKEN_BUDGET
        or custom.get("georoute_zero_carrier_mode") != "masked_zero"
        or custom.get("georoute_roi_extent_floor_mode") != "native_cells"
        or int(custom.get("georoute_roi_extent_floor_cells", -1)) != 1
    ):
        raise ValueError("source config is not the approved dynamic Stage-1 route")
    if any(
        bool(custom.get(field, False))
        for field in (
            "georoute_absolute_coordinates_enabled",
            "georoute_roi_relative_coordinates_enabled",
            "georoute_geometry_projection_enabled",
            "georoute_geometry_side_channel",
            "georoute_diagnostic_telemetry_enabled",
            "georoute_amp_diagnostic_enabled",
            "georoute_gradient_decomposition_enabled",
            "georoute_p0_dense_reference_check",
        )
    ):
        raise ValueError("policy-health source config left support-only Stage-1")

    train_cfg = cfg.dataset.train
    train_cfg.ann_file = annotation["path"]
    train_cfg.class_map = str(class_map)
    train_cfg.data_path = str(video_root)
    train_cfg.subset_name = "training"
    train_cfg.block_list = gate_ids
    cfg.dataset = dict(train=train_cfg)
    cfg.pop("evaluation", None)
    cfg.model.backbone.custom.pretrain = str(pretrained)
    cfg.model.backbone.custom.georoute_random_seed = int(seed)
    cfg.solver.fp16_compress = False
    cfg.solver.static_graph = False
    cfg.solver.amp = True
    cfg.solver.ema = True
    cfg.workflow.end_epoch = 1
    cfg.workflow.val_start_epoch = 1
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.disable_checkpoint = True
    cfg.workflow.max_train_iters = TARGET_SUCCESSFUL_UPDATES
    cfg.workflow.max_amp_retries_per_batch = MAX_AMP_RETRIES_PER_BATCH
    cfg.workflow.fail_on_skipped_update = True
    cfg.workflow.require_successful_update_hook = True
    cfg.workflow.schedule_and_ema_on_success_only = True
    cfg.workflow.capture_amp_rng_state = True
    cfg.workflow.fail_on_nonfinite_loss = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.post_processing.save_dict = False
    cfg.work_dir = str(work_dir)

    binding: dict[str, Any] = {
        "schema_version": DYNAMIC_POLICY_HEALTH_BINDING_SCHEMA,
        "study_id": DYNAMIC_POLICY_HEALTH_STUDY_ID,
        "runtime_commit": runtime_commit,
        "world_size": 1,
        "seed": HEALTH_SEED,
        "target_successful_updates": TARGET_SUCCESSFUL_UPDATES,
        "max_amp_retries_per_batch": MAX_AMP_RETRIES_PER_BATCH,
        "max_total_skipped_attempts": MAX_TOTAL_SKIPPED_ATTEMPTS,
        "initial_loss_scale": INITIAL_LOSS_SCALE,
        "minimum_loss_scale": MINIMUM_LOSS_SCALE,
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "fit_video_ids": fit_ids,
        "gate_video_ids": gate_ids,
        "training_video_ids": fit_ids,
        "training_block_list_video_ids": gate_ids,
        "development_annotation": annotation,
        "class_map_path": str(class_map),
        "class_map_sha256": sha256_file(class_map),
        "development_video_root": str(video_root),
        "pretrained_checkpoint_path": str(pretrained),
        "pretrained_checkpoint_sha256": sha256_file(pretrained),
        "work_dir": str(work_dir),
        "report_path": str((work_dir / "policy_health_report.json").resolve()),
        "dataset_split_built": "train_only",
        "development_fit_annotations_used": True,
        "gt_used_for_detector_and_auxiliary_fit_only": True,
        "gt_used_for_route": False,
        "validation_loader_built": False,
        "test_loader_built": False,
        "metric_computed": False,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_dynamic_policy_health_binding = binding
    cfg.georoute_runtime_binding = binding
    return cfg


def validate_dynamic_policy_health_binding(
    binding: Mapping[str, Any], *, seed: int | None = None
) -> dict[str, Any]:
    binding = dict(_mapping(binding, name="dynamic policy-health binding"))
    if not _self_hash_matches(binding, field="binding_sha256"):
        raise ValueError("dynamic policy-health binding self-hash mismatch")
    if (
        binding.get("schema_version") != DYNAMIC_POLICY_HEALTH_BINDING_SCHEMA
        or binding.get("study_id") != DYNAMIC_POLICY_HEALTH_STUDY_ID
        or int(binding.get("world_size", -1)) != 1
        or int(binding.get("seed", -1)) != HEALTH_SEED
        or int(binding.get("target_successful_updates", -1))
        != TARGET_SUCCESSFUL_UPDATES
        or int(binding.get("max_amp_retries_per_batch", -1))
        != MAX_AMP_RETRIES_PER_BATCH
        or int(binding.get("max_total_skipped_attempts", -1))
        != MAX_TOTAL_SKIPPED_ATTEMPTS
        or float(binding.get("initial_loss_scale", -1.0)) != INITIAL_LOSS_SCALE
        or float(binding.get("minimum_loss_scale", -1.0)) != MINIMUM_LOSS_SCALE
        or binding.get("dataset_split_built") != "train_only"
        or binding.get("development_fit_annotations_used") is not True
        or binding.get("gt_used_for_detector_and_auxiliary_fit_only") is not True
        or binding.get("gt_used_for_route") is not False
        or any(
            binding.get(field) is not False
            for field in (
                "validation_loader_built",
                "test_loader_built",
                "metric_computed",
                "checkpoint_emitted",
                "prediction_emitted",
                "evaluator_invoked",
                "official_test_opened",
                "performance_inference_allowed",
                "paper_claim_allowed",
            )
        )
    ):
        raise ValueError("dynamic policy-health binding contract is invalid")
    if seed is not None and int(seed) != int(binding["seed"]):
        raise ValueError("policy-health CLI seed differs from binding")
    _full_hex(binding.get("runtime_commit"), length=40, name="runtime_commit")
    for field in (
        "source_config_sha256",
        "manifest_file_sha256",
        "class_map_sha256",
        "pretrained_checkpoint_sha256",
    ):
        _full_hex(binding.get(field), length=64, name=field)
    annotation = _mapping(
        binding.get("development_annotation"), name="development annotation"
    )
    _full_hex(annotation.get("sha256"), length=64, name="annotation sha256")
    if (
        list(binding.get("training_video_ids", []))
        != list(binding.get("fit_video_ids", []))
        or list(binding.get("training_block_list_video_ids", []))
        != list(binding.get("gate_video_ids", []))
        or set(binding.get("fit_video_ids", []))
        & set(binding.get("gate_video_ids", []))
    ):
        raise ValueError("dynamic policy-health population binding changed")
    work_dir = Path(str(binding.get("work_dir", ""))).resolve()
    if (
        Path(str(binding.get("report_path", ""))).resolve()
        != work_dir / "policy_health_report.json"
    ):
        raise ValueError("dynamic policy-health report is not work-dir bound")
    return binding


def validate_dynamic_policy_health_config(cfg: Any, *, seed: int) -> dict[str, Any]:
    if "georoute_dynamic_policy_health_binding" not in cfg:
        raise ValueError("config lacks dynamic policy-health binding")
    binding = validate_dynamic_policy_health_binding(
        cfg.georoute_dynamic_policy_health_binding, seed=seed
    )
    custom = cfg.model.backbone.custom
    workflow = cfg.workflow
    if (
        set(cfg.dataset.keys()) != {"train"}
        or cfg.dataset.train.subset_name != "training"
        or list(cfg.dataset.train.get("block_list", []))
        != list(binding["training_block_list_video_ids"])
        or custom.georoute_route_mode != "dynamic_scnr"
        or custom.georoute_policy_estimator != "straight_through"
        or int(custom.georoute_window_token_budget) != WINDOW_TOKEN_BUDGET
        or custom.georoute_zero_carrier_mode != "masked_zero"
        or cfg.solver.amp is not True
        or cfg.solver.ema is not True
        or cfg.solver.fp16_compress is not False
        or cfg.solver.static_graph is not False
        or float(cfg.solver.clip_grad_norm) <= 0.0
        or int(workflow.end_epoch) != 1
        or int(workflow.max_train_iters) != TARGET_SUCCESSFUL_UPDATES
        or int(workflow.max_amp_retries_per_batch) != MAX_AMP_RETRIES_PER_BATCH
        or workflow.fail_on_skipped_update is not True
        or workflow.require_successful_update_hook is not True
        or workflow.schedule_and_ema_on_success_only is not True
        or workflow.disable_checkpoint is not True
        or int(workflow.val_loss_interval) != -1
        or int(workflow.val_eval_interval) != -1
        or cfg.inference.load_from_raw_predictions is not False
        or cfg.inference.save_raw_prediction is not False
        or cfg.post_processing.save_dict is not False
    ):
        raise ValueError(
            "dynamic policy-health config violates its train-only contract"
        )
    return binding


def _component_for_parameter(name: str) -> str | None:
    while name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("backbone.scout.stem"):
        return "scout_stem"
    if name.startswith("backbone.scout.geometry_head"):
        return "scout_geometry"
    if name.startswith("backbone.scout.base_utility_head"):
        return "scout_base_utility"
    if name.startswith("backbone.scout.residual_head"):
        return "scout_residual"
    if name.startswith("backbone.dynamic_aux_head"):
        return "dynamic_aux_head"
    if name.startswith("backbone.sparse_adapter"):
        return "sparse_adapter"
    if ".adapter." in f".{name}":
        return "videomae_adapter"
    if name.startswith("projection"):
        return "projection"
    if name.startswith("rpn_head"):
        return "rpn_head"
    if name.startswith("neck"):
        return "neck"
    return None


def gradient_component_summary(model: Any) -> dict[str, Any]:
    """Return compact post-unscale gradient health without changing the graph."""

    import torch

    component_stats: dict[str, dict[str, Any]] = {}
    nonfinite_tensors: list[str] = []
    for name, parameter in model.named_parameters():
        component = _component_for_parameter(str(name))
        if component is None or not bool(parameter.requires_grad):
            continue
        stats = component_stats.setdefault(
            component,
            {
                "trainable_parameter_tensors": 0,
                "gradient_parameter_tensors": 0,
                "nonzero_parameter_tensors": 0,
                "nonfinite_elements": 0,
                "max_abs": 0.0,
                "l2_norm": 0.0,
            },
        )
        stats["trainable_parameter_tensors"] += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        stats["gradient_parameter_tensors"] += 1
        detached = gradient.detach()
        finite = torch.isfinite(detached)
        nonfinite_count = int((~finite).sum().item())
        stats["nonfinite_elements"] += nonfinite_count
        if nonfinite_count:
            nonfinite_tensors.append(str(name))
        finite_values = detached.float().masked_select(finite)
        if finite_values.numel() == 0:
            continue
        if bool(torch.count_nonzero(finite_values).item()):
            stats["nonzero_parameter_tensors"] += 1
        stats["max_abs"] = max(
            float(stats["max_abs"]), float(finite_values.abs().max().item())
        )
        norm = float(torch.linalg.vector_norm(finite_values).item())
        stats["l2_norm"] += norm * norm
    for stats in component_stats.values():
        stats["l2_norm"] = math.sqrt(float(stats["l2_norm"]))
        stats["gradient_nonzero"] = bool(stats["nonzero_parameter_tensors"] > 0)
        stats["all_finite"] = int(stats["nonfinite_elements"]) == 0
    return {
        "components": component_stats,
        "nonzero_components": sorted(
            name for name, stats in component_stats.items() if stats["gradient_nonzero"]
        ),
        "all_gradients_finite": not nonfinite_tensors,
        "nonfinite_gradient_tensors": sorted(nonfinite_tensors),
    }


def _loss_summary(losses: Mapping[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name, value in losses.items():
        if hasattr(value, "detach"):
            detached = value.detach()
            if int(detached.numel()) != 1:
                continue
            summary[str(name)] = _finite_number(
                float(detached.item()), name=f"loss {name}"
            )
    missing = REQUIRED_LOSS_KEYS - set(summary)
    if missing:
        raise ValueError(f"policy-health forward lacks losses: {sorted(missing)}")
    if summary["georoute_geometry_regularization_loss"] != 0.0:
        raise ValueError("dynamic main route must keep geometry regularization zero")
    return summary


def _input_summary(data_dict: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name, value in sorted(data_dict.items()):
        if torch.is_tensor(value):
            result[str(name)] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result[str(name)] = {
                "kind": "sequence",
                "length": len(value),
                "tensor_shapes": [
                    list(item.shape) for item in value if torch.is_tensor(item)
                ],
            }
        elif isinstance(value, Mapping):
            result[str(name)] = {"kind": "mapping", "keys": sorted(map(str, value))}
        else:
            result[str(name)] = {"kind": type(value).__name__}
    return result


def summarize_dynamic_route_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    audit = _mapping(audit, name="GeoRoute audit")
    if (
        audit.get("routing_schema") != DYNAMIC_ROUTING_SCHEMA
        or audit.get("route_mode") != "dynamic_scnr"
        or audit.get("policy_estimator") != "straight_through"
        or audit.get("roi_modifier_geometry")
        != "signed_ellipse_with_semiaxes_half_decoded_full_extent"
        or audit.get("window_budget_is_global") is not True
        or audit.get("independent_count_head") is not False
        or audit.get("fixed_context_quota") is not False
        or audit.get("fixed_per_tubelet_k") is not False
        or audit.get("k_t_allows_zero") is not True
        or audit.get("zero_carrier_mode") != "masked_zero"
        or audit.get("heavy_valid_mask_matches_k_t") is not True
        or audit.get("scout_policy_stop_gradient") is not True
        or audit.get("proxy_inference_enabled") is not False
        or audit.get("proxy_updates_scout_stem") is not False
        or audit.get("proxy_updates_heavy_backbone") is not False
        or audit.get("uses_gt_for_route") is not False
        or audit.get("uses_gt_for_auxiliary_fit_only") is not True
        or audit.get("uses_teacher") is not False
        or audit.get("uses_oracle") is not False
        or audit.get("uses_test_evidence") is not False
    ):
        raise ValueError("dynamic policy-health route isolation failed")
    budget = int(audit.get("window_token_budget", -1))
    if budget != WINDOW_TOKEN_BUDGET or any(
        int(audit.get(field, -1)) != budget
        for field in (
            "requested_physical_tokens_per_window",
            "unique_physical_tokens_per_window",
            "executed_patch_tokens_per_window",
        )
    ):
        raise ValueError("dynamic policy-health violated exact global B")
    if (
        int(audit.get("padded_heavy_tokens_per_window", -1)) != 0
        or int(audit.get("heavy_backbone_forward_count", -1)) != 1
    ):
        raise ValueError(
            "dynamic policy-health used padding, dummy tokens, or extra heavy forwards"
        )

    k_rows = audit.get("k_per_tubelet")
    if not isinstance(k_rows, list) or not k_rows:
        raise ValueError("dynamic policy-health lacks K_t rows")
    histogram: Counter[int] = Counter()
    for row in k_rows:
        if not isinstance(row, list) or len(row) != WINDOW_TUBELETS:
            raise ValueError(
                "dynamic policy-health K_t row has the wrong tubelet count"
            )
        values = [int(value) for value in row]
        if any(value < 0 or value > NATIVE_SPATIAL_CANDIDATES for value in values):
            raise ValueError("dynamic policy-health K_t leaves native support")
        if sum(values) != budget:
            raise ValueError("dynamic policy-health K_t does not sum to exact B")
        histogram.update(values)

    role_counts = _mapping(audit.get("role_counts"), name="role counts")
    roles = {name: int(role_counts.get(name, -1)) for name in ROLE_NAMES}
    if any(value < 0 for value in roles.values()) or sum(
        roles.values()
    ) != budget * len(k_rows):
        raise ValueError("dynamic policy-health role counts do not partition B")
    role_rows = audit.get("role_counts_per_window")
    if (
        not isinstance(role_rows, list)
        or len(role_rows) != len(k_rows)
        or any(
            not isinstance(row, list)
            or len(row) != len(ROLE_NAMES)
            or sum(map(int, row)) != budget
            for row in role_rows
        )
    ):
        raise ValueError("dynamic policy-health per-window roles do not partition B")
    soft_sums = audit.get("proxy_soft_budget_sum")
    if (
        not isinstance(soft_sums, list)
        or len(soft_sums) != len(k_rows)
        or any(
            abs(_finite_number(value, name="soft budget") - budget)
            > SOFT_BUDGET_ABSOLUTE_TOLERANCE
            for value in soft_sums
        )
    ):
        raise ValueError("dynamic policy-health soft proxy violates exact budget")

    packed = _mapping(audit.get("packed"), name="ragged executor audit")
    clip_rows = packed.get("clip_token_counts")
    pair_rows = packed.get("attention_pairs_per_window")
    if (
        packed.get("schema_version") != RAGGED_EXECUTOR_SCHEMA
        or packed.get("execution_mode") != "true_clip_ragged_no_padding"
        or packed.get("adapter_execution") != "coordinate_lineage_true_ragged"
        or int(packed.get("padded_heavy_tokens_per_window", -1)) != 0
        or int(packed.get("executed_patch_tokens_per_window", -1)) != budget
        or int(packed.get("dense_adapter_forward_count", -1)) != 0
        or not isinstance(clip_rows, list)
        or len(clip_rows) != len(k_rows)
        or not isinstance(pair_rows, list)
        or len(pair_rows) != len(k_rows)
    ):
        raise ValueError("dynamic policy-health ragged executor contract failed")
    for clip_counts, attention_pairs in zip(clip_rows, pair_rows):
        if (
            not isinstance(clip_counts, list)
            or any(int(count) <= 0 for count in clip_counts)
            or sum(map(int, clip_counts)) != budget
            or sum(int(count) ** 2 for count in clip_counts) != int(attention_pairs)
        ):
            raise ValueError("dynamic policy-health ragged attention ledger is invalid")

    statistics = {}
    for name in (
        "hard_utility",
        "soft_utility",
        "q_base",
        "delta_roi",
        "delta_residual",
    ):
        values = dict(_mapping(audit.get(name), name=name))
        for key, value in values.items():
            if isinstance(value, (int, float)):
                _finite_number(value, name=f"{name}.{key}")
        statistics[name] = values
    return {
        "successful_update": int(audit.get("successful_update", -1)),
        "window_token_budget": budget,
        "k_t_min": min(histogram),
        "k_t_max": max(histogram),
        "k_t_zero_count": int(histogram.get(0, 0)),
        "k_t_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "role_counts": roles,
        "soft_budget_sums": [float(value) for value in soft_sums],
        "soft_budget_max_abs_residual": max(
            abs(float(value) - budget) for value in soft_sums
        ),
        "attention_pairs_per_window": [int(value) for value in pair_rows],
        "statistics": statistics,
        "geometry_min_extent_wh": list(audit.get("geometry_min_extent_wh", [])),
        "geometry_extent_floor_cells": int(
            audit.get("geometry_extent_floor_cells", -1)
        ),
        "source_grid_hw": list(audit.get("source_grid_hw", [])),
        "dynamic_auxiliary_raw": _finite_number(
            audit.get("dynamic_auxiliary_raw"), name="dynamic auxiliary raw"
        ),
        "dynamic_proxy_raw": _finite_number(
            audit.get("dynamic_proxy_raw"), name="dynamic proxy raw"
        ),
        "dynamic_proxy_weight": _finite_number(
            audit.get("dynamic_proxy_weight"), name="dynamic proxy weight"
        ),
    }


def audit_no_performance_artifacts(work_dir: Path) -> dict[str, Any]:
    checkpoints = sorted(
        {*work_dir.rglob("*.pth"), *work_dir.rglob("*.pt"), *work_dir.rglob("*.ckpt")}
    )
    forbidden_names = {
        "result_detection.json",
        "georoute_development_profile.json",
        "georoute_diagnostic_telemetry.json",
        "test.out",
    }
    forbidden = sorted(
        path
        for path in work_dir.rglob("*")
        if path.is_file() and path.name in forbidden_names
    )
    temporaries = sorted(work_dir.rglob("*.tmp*"))
    if checkpoints or forbidden or temporaries:
        raise RuntimeError(
            "policy-health emitted a forbidden artifact: "
            f"checkpoints={checkpoints}, forbidden={forbidden}, temporaries={temporaries}"
        )
    return {
        "checkpoint_payload_count": 0,
        "temporary_payload_count": 0,
        "prediction_payload_count": 0,
        "evaluator_output_count": 0,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


class DynamicPolicyHealthObserver:
    """Collect compact per-attempt and per-success route/gradient health."""

    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []
        self.successful_updates: list[dict[str, Any]] = []
        self.batch_inputs: list[dict[str, Any]] = []
        self._pending_forward: dict[str, Any] | None = None
        self._pending_gradient: dict[str, Any] | None = None

    def __call__(self, event: str, **payload: Any) -> None:
        if event == "batch_start":
            self.batch_inputs.append(
                {
                    "iter_idx": int(payload["iter_idx"]),
                    "successful_update_index": int(payload["successful_update_index"]),
                    "scale": float(payload["scale"]),
                    "input_summary": _input_summary(payload["data_dict"]),
                }
            )
            return
        if event == "forward_complete":
            unwrapped = getattr(payload["model"], "module", payload["model"])
            backbone = getattr(unwrapped, "backbone", None)
            audit = getattr(backbone, "latest_georoute_audit", None)
            if not isinstance(audit, Mapping):
                raise RuntimeError("dynamic policy-health forward lacks GeoRoute audit")
            self._pending_forward = {
                "iter_idx": int(payload["iter_idx"]),
                "retry_count": int(payload["retry_count"]),
                "scale": float(payload["scale"]),
                "losses": _loss_summary(payload["losses"]),
                "route": json.loads(json.dumps(dict(audit), allow_nan=False)),
            }
            return
        if event == "post_clip":
            self._pending_gradient = gradient_component_summary(payload["model"])
            return
        if event == "scaler_result":
            if self._pending_forward is None or self._pending_gradient is None:
                raise RuntimeError(
                    "dynamic policy-health attempt telemetry is incomplete"
                )
            attempt = {
                "attempt_index": len(self.attempts),
                "iter_idx": int(payload["iter_idx"]),
                "retry_count": int(payload["retry_count"]),
                "scale_before": float(payload["scale_before"]),
                "scale_after": float(payload["scale_after"]),
                "update_succeeded": bool(payload["update_succeeded"]),
                "losses": self._pending_forward["losses"],
                "gradient": self._pending_gradient,
            }
            self.attempts.append(attempt)
            if attempt["update_succeeded"]:
                route = summarize_dynamic_route_audit(self._pending_forward["route"])
                expected_index = len(self.successful_updates)
                if route["successful_update"] != expected_index:
                    raise RuntimeError(
                        "dynamic proxy schedule is not successful-update indexed"
                    )
                self.successful_updates.append(
                    {
                        "update_index": expected_index,
                        **attempt,
                        "route": route,
                    }
                )
            self._pending_forward = None
            self._pending_gradient = None
            return
        if event in {"scaled_backward", "unscaled", "pre_clip", "batch_complete"}:
            return
        raise ValueError(f"unsupported dynamic policy-health event {event!r}")

    def build_report(
        self,
        *,
        binding: Mapping[str, Any],
        source: Mapping[str, Any],
        slurm: Mapping[str, Any],
        update_audit: Mapping[str, Any],
        successful_updates: int,
        artifact_audit: Mapping[str, Any],
        peak_cuda_allocated_bytes: int,
        execution_error: BaseException | None = None,
        execution_traceback: str | None = None,
    ) -> dict[str, Any]:
        binding = validate_dynamic_policy_health_binding(binding)
        component_hits = {name: 0 for name in REQUIRED_GRADIENT_COMPONENTS}
        aggregate_roles = Counter({name: 0 for name in ROLE_NAMES})
        aggregate_k = Counter()
        all_losses_finite = True
        all_gradients_finite = True
        for update in self.successful_updates:
            gradient = update["gradient"]
            all_gradients_finite &= gradient["all_gradients_finite"] is True
            for component in set(gradient["nonzero_components"]):
                if component in component_hits:
                    component_hits[component] += 1
            route = update["route"]
            aggregate_roles.update(route["role_counts"])
            aggregate_k.update(
                {int(key): int(value) for key, value in route["k_t_histogram"].items()}
            )
            all_losses_finite &= all(
                math.isfinite(float(value)) for value in update["losses"].values()
            )
        skipped_attempts = sum(
            attempt["update_succeeded"] is False for attempt in self.attempts
        )
        scales = [
            float(value)
            for attempt in self.attempts
            for value in (attempt["scale_before"], attempt["scale_after"])
        ]
        total_roles = sum(aggregate_roles.values())
        dominant_role_fraction = (
            max(aggregate_roles.values()) / total_roles if total_roles else 1.0
        )
        hold_reasons: list[str] = []
        if execution_error is not None:
            hold_reasons.append(f"execution_error:{type(execution_error).__name__}")
        if int(successful_updates) != TARGET_SUCCESSFUL_UPDATES:
            hold_reasons.append("successful_update_count")
        if len(self.successful_updates) != TARGET_SUCCESSFUL_UPDATES:
            hold_reasons.append("successful_update_telemetry_count")
        if len(self.batch_inputs) != TARGET_SUCCESSFUL_UPDATES:
            hold_reasons.append("consumed_batch_count")
        if skipped_attempts > MAX_TOTAL_SKIPPED_ATTEMPTS:
            hold_reasons.append("amp_skip_budget")
        if not scales or min(scales) < MINIMUM_LOSS_SCALE:
            hold_reasons.append("amp_scale_floor")
        if not all_losses_finite:
            hold_reasons.append("nonfinite_loss")
        if not all_gradients_finite:
            hold_reasons.append("nonfinite_gradient")
        weak_components = sorted(
            name
            for name, count in component_hits.items()
            if count < MIN_COMPONENT_HEALTHY_UPDATES
        )
        if weak_components:
            hold_reasons.append("insufficient_sustained_component_gradient")
        missing_roles = sorted(
            name for name in ROLE_NAMES if aggregate_roles[name] <= 0
        )
        if missing_roles:
            hold_reasons.append("missing_dynamic_role")
        if dominant_role_fraction > MAX_DOMINANT_ROLE_FRACTION:
            hold_reasons.append("role_collapse")
        if len(aggregate_k) < 2:
            hold_reasons.append("fixed_k_t_collapse")
        if int(update_audit.get("scheduler_advances", -1)) != int(successful_updates):
            hold_reasons.append("scheduler_success_cadence")
        if int(update_audit.get("ema_updates", -1)) != int(successful_updates):
            hold_reasons.append("ema_success_cadence")
        status = (
            DYNAMIC_POLICY_HEALTH_PASS
            if not hold_reasons
            else DYNAMIC_POLICY_HEALTH_FAIL
            if execution_error is not None
            else DYNAMIC_POLICY_HEALTH_HOLD
        )
        report: dict[str, Any] = {
            "schema_version": DYNAMIC_POLICY_HEALTH_SCHEMA,
            "status": status,
            "study_id": DYNAMIC_POLICY_HEALTH_STUDY_ID,
            "binding": dict(binding),
            "source": dict(source),
            "slurm": dict(slurm),
            "successful_updates": int(successful_updates),
            "attempts": copy.deepcopy(self.attempts),
            "updates": copy.deepcopy(self.successful_updates),
            "summary": {
                "target_successful_updates": TARGET_SUCCESSFUL_UPDATES,
                "successful_update_count": int(successful_updates),
                "consumed_batch_count": len(self.batch_inputs),
                "optimizer_attempt_count": len(self.attempts),
                "amp_skipped_attempt_count": skipped_attempts,
                "minimum_observed_scale": min(scales) if scales else None,
                "final_scale": scales[-1] if scales else None,
                "all_losses_finite": all_losses_finite,
                "all_gradients_finite": all_gradients_finite,
                "component_nonzero_update_counts": component_hits,
                "minimum_component_healthy_updates": MIN_COMPONENT_HEALTHY_UPDATES,
                "weak_gradient_components": weak_components,
                "aggregate_role_counts": dict(aggregate_roles),
                "missing_roles": missing_roles,
                "dominant_role_fraction": dominant_role_fraction,
                "max_dominant_role_fraction": MAX_DOMINANT_ROLE_FRACTION,
                "aggregate_k_t_histogram": {
                    str(key): aggregate_k[key] for key in sorted(aggregate_k)
                },
                "k_t_min": min(aggregate_k) if aggregate_k else None,
                "k_t_max": max(aggregate_k) if aggregate_k else None,
                "k_t_zero_count": int(aggregate_k.get(0, 0)),
                "k_t_zero_is_capability_not_pass_requirement": True,
                "peak_cuda_allocated_bytes": int(peak_cuda_allocated_bytes),
                "hold_reasons": sorted(set(hold_reasons)),
                "policy_health_gate_passed": not hold_reasons,
            },
            "update_audit": dict(update_audit),
            "artifact_audit": dict(artifact_audit),
            "scope": {
                "real_development_fit_data_loaded": True,
                "fit_labels_used_for_detector_and_auxiliary_only": True,
                "gt_used_for_route": False,
                "validation_loader_built": False,
                "test_loader_built": False,
                "metric_computed": False,
                "prediction_written": False,
                "checkpoint_written": False,
                "evaluator_invoked": False,
                "official_test_opened": False,
                "performance_inference_allowed": False,
                "paper_claim_allowed": False,
            },
            "execution_error": (
                {
                    "exception_type": type(execution_error).__name__,
                    "exception_message": str(execution_error)[:2000],
                    "traceback_sha256": canonical_sha256(
                        {"traceback": str(execution_traceback or "unavailable")}
                    ),
                }
                if execution_error is not None
                else None
            ),
        }
        report["report_sha256"] = canonical_sha256(report)
        return validate_dynamic_policy_health_report(report)


def validate_dynamic_policy_health_report(report: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(_mapping(report, name="dynamic policy-health report"))
    if not _self_hash_matches(report, field="report_sha256"):
        raise ValueError("dynamic policy-health report self-hash mismatch")
    status = report.get("status")
    if (
        report.get("schema_version") != DYNAMIC_POLICY_HEALTH_SCHEMA
        or report.get("study_id") != DYNAMIC_POLICY_HEALTH_STUDY_ID
        or status
        not in {
            DYNAMIC_POLICY_HEALTH_PASS,
            DYNAMIC_POLICY_HEALTH_HOLD,
            DYNAMIC_POLICY_HEALTH_FAIL,
        }
    ):
        raise ValueError("dynamic policy-health report contract is invalid")
    binding = validate_dynamic_policy_health_binding(
        _mapping(report.get("binding"), name="report binding")
    )
    if report.get("source", {}).get("commit") != binding["runtime_commit"]:
        raise ValueError("dynamic policy-health report/source commit mismatch")
    slurm = _mapping(report.get("slurm"), name="Slurm receipt")
    if (
        not str(slurm.get("job_id", "")).isdigit()
        or slurm.get("logical_device") != "cuda:0"
        or int(slurm.get("visible_device_count", -1)) != 1
    ):
        raise ValueError("dynamic policy-health Slurm receipt is invalid")
    scope = _mapping(report.get("scope"), name="no-performance scope")
    if (
        scope.get("real_development_fit_data_loaded") is not True
        or scope.get("fit_labels_used_for_detector_and_auxiliary_only") is not True
        or scope.get("gt_used_for_route") is not False
        or any(
            scope.get(field) is not False
            for field in (
                "validation_loader_built",
                "test_loader_built",
                "metric_computed",
                "prediction_written",
                "checkpoint_written",
                "evaluator_invoked",
                "official_test_opened",
                "performance_inference_allowed",
                "paper_claim_allowed",
            )
        )
    ):
        raise ValueError("dynamic policy-health left its no-performance scope")
    artifact = _mapping(report.get("artifact_audit"), name="artifact audit")
    if any(
        int(artifact.get(field, -1)) != 0
        for field in (
            "checkpoint_payload_count",
            "temporary_payload_count",
            "prediction_payload_count",
            "evaluator_output_count",
        )
    ):
        raise ValueError("dynamic policy-health emitted a forbidden artifact")
    summary = _mapping(report.get("summary"), name="policy-health summary")
    updates = report.get("updates")
    attempts = report.get("attempts")
    if not isinstance(updates, list) or not isinstance(attempts, list):
        raise ValueError("dynamic policy-health ordered telemetry is missing")
    successful_attempts = [
        attempt for attempt in attempts if attempt.get("update_succeeded") is True
    ]
    failed_attempts = [
        attempt for attempt in attempts if attempt.get("update_succeeded") is False
    ]
    if (
        len(successful_attempts) != len(updates)
        or len(successful_attempts) != int(report.get("successful_updates", -1))
        or int(summary.get("successful_update_count", -1))
        != int(report.get("successful_updates", -2))
        or len(attempts) != int(summary.get("optimizer_attempt_count", -1))
        or len(failed_attempts) != int(summary.get("amp_skipped_attempt_count", -1))
    ):
        raise ValueError("dynamic policy-health attempt/update accounting is invalid")
    update_audit = _mapping(report.get("update_audit"), name="update audit")
    if (
        int(update_audit.get("optimizer_attempts", -1)) != len(attempts)
        or int(update_audit.get("amp_skipped_attempts", -1)) != len(failed_attempts)
        or int(update_audit.get("consumed_batches", -1))
        != int(summary.get("consumed_batch_count", -2))
        or int(update_audit.get("scheduler_advances", -1))
        != int(report.get("successful_updates", -2))
        or int(update_audit.get("ema_updates", -1))
        != int(report.get("successful_updates", -2))
    ):
        raise ValueError("dynamic policy-health transition audit is inconsistent")
    recomputed_components = Counter({name: 0 for name in REQUIRED_GRADIENT_COMPONENTS})
    recomputed_roles = Counter({name: 0 for name in ROLE_NAMES})
    recomputed_k = Counter()
    for expected_index, update in enumerate(updates):
        update = _mapping(update, name="successful update")
        if (
            int(update.get("update_index", -1)) != expected_index
            or update.get("update_succeeded") is not True
        ):
            raise ValueError(
                "dynamic policy-health successful updates are out of order"
            )
        losses = _mapping(update.get("losses"), name="successful-update losses")
        if set(losses) != REQUIRED_LOSS_KEYS or any(
            not math.isfinite(float(value)) for value in losses.values()
        ):
            raise ValueError(
                "dynamic policy-health successful update has invalid losses"
            )
        if float(losses["georoute_geometry_regularization_loss"]) != 0.0:
            raise ValueError("dynamic policy-health enabled geometry regularization")
        gradient = _mapping(update.get("gradient"), name="successful-update gradient")
        if gradient.get("all_gradients_finite") is not True:
            raise ValueError("dynamic policy-health successful gradient is nonfinite")
        for component in set(gradient.get("nonzero_components", [])):
            if component in recomputed_components:
                recomputed_components[component] += 1
        route = _mapping(update.get("route"), name="successful-update route")
        histogram = {
            int(key): int(value)
            for key, value in _mapping(
                route.get("k_t_histogram"), name="K_t histogram"
            ).items()
        }
        if (
            int(route.get("successful_update", -1)) != expected_index
            or int(route.get("window_token_budget", -1)) != WINDOW_TOKEN_BUDGET
            or sum(histogram.values()) != WINDOW_TUBELETS
            or sum(key * value for key, value in histogram.items())
            != WINDOW_TOKEN_BUDGET
            or min(histogram, default=-1) != int(route.get("k_t_min", -2))
            or max(histogram, default=-1) != int(route.get("k_t_max", -2))
            or int(histogram.get(0, 0)) != int(route.get("k_t_zero_count", -1))
            or float(route.get("soft_budget_max_abs_residual", math.inf))
            > SOFT_BUDGET_ABSOLUTE_TOLERANCE
        ):
            raise ValueError("dynamic policy-health successful update violates exact B")
        roles = _mapping(route.get("role_counts"), name="successful-update roles")
        if (
            set(roles) != set(ROLE_NAMES)
            or sum(map(int, roles.values())) != WINDOW_TOKEN_BUDGET
        ):
            raise ValueError(
                "dynamic policy-health successful roles do not partition B"
            )
        recomputed_roles.update({name: int(roles[name]) for name in ROLE_NAMES})
        recomputed_k.update(histogram)
    observed_component_hits = _mapping(
        summary.get("component_nonzero_update_counts"), name="gradient hit counts"
    )
    if {
        name: int(observed_component_hits.get(name, -1))
        for name in REQUIRED_GRADIENT_COMPONENTS
    } != dict(recomputed_components):
        raise ValueError("dynamic policy-health gradient hit summary is inconsistent")
    if {
        name: int(
            _mapping(summary.get("aggregate_role_counts"), name="aggregate roles").get(
                name, -1
            )
        )
        for name in ROLE_NAMES
    } != dict(recomputed_roles):
        raise ValueError("dynamic policy-health aggregate roles are inconsistent")
    observed_k = {
        int(key): int(value)
        for key, value in _mapping(
            summary.get("aggregate_k_t_histogram"), name="aggregate K_t histogram"
        ).items()
    }
    if observed_k != dict(recomputed_k):
        raise ValueError("dynamic policy-health aggregate K_t summary is inconsistent")
    if status == DYNAMIC_POLICY_HEALTH_PASS:
        if (
            int(report.get("successful_updates", -1)) != TARGET_SUCCESSFUL_UPDATES
            or len(updates) != TARGET_SUCCESSFUL_UPDATES
            or int(summary.get("successful_update_count", -1))
            != TARGET_SUCCESSFUL_UPDATES
            or int(summary.get("consumed_batch_count", -1)) != TARGET_SUCCESSFUL_UPDATES
            or int(summary.get("amp_skipped_attempt_count", -1))
            > MAX_TOTAL_SKIPPED_ATTEMPTS
            or float(summary.get("minimum_observed_scale", 0.0)) < MINIMUM_LOSS_SCALE
            or summary.get("all_losses_finite") is not True
            or summary.get("all_gradients_finite") is not True
            or summary.get("weak_gradient_components") != []
            or summary.get("missing_roles") != []
            or float(summary.get("dominant_role_fraction", 1.0))
            > MAX_DOMINANT_ROLE_FRACTION
            or len(summary.get("aggregate_k_t_histogram", {})) < 2
            or summary.get("policy_health_gate_passed") is not True
            or summary.get("hold_reasons") != []
            or report.get("execution_error") is not None
        ):
            raise ValueError("passing dynamic policy-health report violates its gate")
        if set(observed_component_hits) != REQUIRED_GRADIENT_COMPONENTS or any(
            int(observed_component_hits[name]) < MIN_COMPONENT_HEALTHY_UPDATES
            for name in REQUIRED_GRADIENT_COMPONENTS
        ):
            raise ValueError("passing policy-health report lacks sustained gradients")
    elif summary.get("policy_health_gate_passed") is not False or not summary.get(
        "hold_reasons"
    ):
        raise ValueError("non-passing policy-health report lacks a hold reason")
    return report


def publish_dynamic_policy_health_report(
    path: str | Path, report: Mapping[str, Any]
) -> None:
    path = Path(path).resolve()
    validated = validate_dynamic_policy_health_report(report)
    if path.exists():
        raise FileExistsError("dynamic policy-health report already exists")
    _atomic_write_json(path, validated)
