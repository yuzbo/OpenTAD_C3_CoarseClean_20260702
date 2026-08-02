"""No-performance CUDA P0 for the dynamic exact-budget GeoRoute Stage-1 path.

The gate builds the real AdaTAD detector, executes one synthetic production-
shape training forward and one GradScaler-backed backward on a Slurm GPU, and
validates the exact-B/ragged/masked-zero ledgers.  It never loads a dataset,
evaluates mAP, writes predictions, or saves a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DYNAMIC_STAGE1_P0_SCHEMA = "georoute_dynamic_stage1_cuda_p0_v1"
DYNAMIC_ROUTING_SCHEMA = "georoute_dynamic_global_routing_v2"
RAGGED_EXECUTOR_SCHEMA = "videomae_native_ragged_v1"
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
REQUIRED_LOSS_KEYS = {
    "cls_loss",
    "reg_loss",
    "georoute_geometry_regularization_loss",
    "georoute_dynamic_auxiliary_loss",
    "georoute_dynamic_soft_proxy_loss",
    "cost",
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_dynamic_stage1_p0_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(payload)
    report.pop("report_sha256", None)
    report["report_sha256"] = _payload_sha256(report)
    return report


def _require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _require_int(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)
    return int(value)


def validate_dynamic_stage1_p0_report(report: Mapping[str, Any]) -> None:
    """Fail closed on any overclaim or missing dynamic-executor invariant."""

    if report.get("schema_version") != DYNAMIC_STAGE1_P0_SCHEMA:
        raise ValueError("unexpected dynamic Stage-1 P0 schema")
    unhashed = dict(report)
    observed_hash = unhashed.pop("report_sha256", None)
    if observed_hash != _payload_sha256(unhashed):
        raise ValueError("dynamic Stage-1 P0 report hash mismatch")
    if report.get("status") != "PASS_NO_PERFORMANCE_P0":
        raise ValueError("dynamic Stage-1 P0 report is not PASS")
    scope = _require_mapping(report.get("scope"), "P0 scope is missing")
    expected_scope = {
        "synthetic_inputs_only": True,
        "dataset_loaded": False,
        "metric_computed": False,
        "prediction_written": False,
        "checkpoint_written": False,
        "performance_claim_allowed": False,
        "official_test_opened": False,
    }
    if {key: scope.get(key) for key in expected_scope} != expected_scope:
        raise ValueError("dynamic Stage-1 P0 exceeded its no-performance scope")

    source = _require_mapping(report.get("source"), "P0 source receipt is missing")
    if (
        source.get("tree_clean") is not True
        or source.get("head_matches_expected") is not True
        or source.get("origin_ref_matches_expected") is not True
        or not isinstance(source.get("commit"), str)
        or len(str(source["commit"])) != 40
    ):
        raise ValueError("P0 source is not one exact clean remote commit")
    slurm = _require_mapping(report.get("slurm"), "P0 Slurm receipt is missing")
    if not str(slurm.get("job_id", "")) or slurm.get("logical_device") != "cuda:0":
        raise ValueError("P0 did not run on Slurm logical cuda:0")
    if _require_int(slurm.get("visible_device_count"), "visible GPU count missing") != 1:
        raise ValueError("P0 requires exactly one Slurm-visible GPU")

    audit = _require_mapping(
        report.get("backbone_audit"),
        "dynamic backbone audit is missing",
    )
    if (
        audit.get("routing_schema") != DYNAMIC_ROUTING_SCHEMA
        or audit.get("route_mode") != "dynamic_scnr"
        or audit.get("policy_estimator") != "straight_through"
        or audit.get("roi_modifier_geometry")
        != "signed_ellipse_with_semiaxes_half_decoded_full_extent"
        or audit.get("scout_policy_stop_gradient") is not True
        or audit.get("proxy_inference_enabled") is not False
        or audit.get("proxy_updates_scout_stem") is not False
        or audit.get("proxy_updates_heavy_backbone") is not False
        or audit.get("window_budget_is_global") is not True
        or audit.get("independent_count_head") is not False
        or audit.get("fixed_context_quota") is not False
        or audit.get("fixed_per_tubelet_k") is not False
        or audit.get("k_t_allows_zero") is not True
        or audit.get("zero_carrier_mode") != "masked_zero"
        or audit.get("heavy_valid_mask_matches_k_t") is not True
        or audit.get("uses_gt_for_route") is not False
        or audit.get("uses_teacher") is not False
        or audit.get("uses_oracle") is not False
        or audit.get("uses_test_evidence") is not False
    ):
        raise ValueError("dynamic route/isolation audit violated the frozen contract")
    budget = _require_int(audit.get("window_token_budget"), "window budget missing")
    if budget <= 0:
        raise ValueError("dynamic Stage-1 P0 budget must be positive")
    for key in (
        "requested_physical_tokens_per_window",
        "unique_physical_tokens_per_window",
        "executed_patch_tokens_per_window",
    ):
        if _require_int(audit.get(key), f"{key} missing") != budget:
            raise ValueError(f"{key} differs from exact global B")
    if _require_int(audit.get("padded_heavy_tokens_per_window"), "padding missing") != 0:
        raise ValueError("dynamic ragged P0 executed padding or dummy tokens")
    if _require_int(audit.get("heavy_backbone_forward_count"), "forward count missing") != 1:
        raise ValueError("dynamic P0 requires one heavy ragged forward")

    k_per_tubelet = audit.get("k_per_tubelet")
    role_counts = audit.get("role_counts_per_window")
    if not isinstance(k_per_tubelet, list) or not k_per_tubelet:
        raise ValueError("dynamic K_t ledger is missing")
    if not isinstance(role_counts, list) or len(role_counts) != len(k_per_tubelet):
        raise ValueError("dynamic role-count ledger is missing")
    item_count = _require_int(
        _require_mapping(report.get("config"), "P0 config receipt missing").get(
            "native_spatial_candidates"
        ),
        "native spatial capacity missing",
    )
    for k_row, role_row in zip(k_per_tubelet, role_counts):
        if (
            not isinstance(k_row, list)
            or not k_row
            or any(not isinstance(value, int) or value < 0 or value > item_count for value in k_row)
            or sum(k_row) != budget
        ):
            raise ValueError("per-window K_t ledger does not sum to exact B")
        if (
            not isinstance(role_row, list)
            or len(role_row) != 3
            or any(not isinstance(value, int) or value < 0 for value in role_row)
            or sum(role_row) != budget
        ):
            raise ValueError("dynamic operational roles do not partition exact B")
    soft_sums = audit.get("proxy_soft_budget_sum")
    if (
        not isinstance(soft_sums, list)
        or len(soft_sums) != len(k_per_tubelet)
        # Production has 84,480 candidates.  Match the allocator's FP32
        # accumulation tolerance while remaining far below one physical token.
        or any(abs(float(value) - budget) > 5e-2 for value in soft_sums)
    ):
        raise ValueError("backward-only soft projection does not preserve exact B")

    packed = _require_mapping(audit.get("packed"), "ragged executor ledger missing")
    if (
        packed.get("schema_version") != RAGGED_EXECUTOR_SCHEMA
        or packed.get("execution_mode") != "true_clip_ragged_no_padding"
        or packed.get("adapter_execution") != "coordinate_lineage_true_ragged"
        or _require_int(packed.get("padded_heavy_tokens_per_window"), "packed padding missing") != 0
        or _require_int(packed.get("executed_patch_tokens_per_window"), "packed execution missing") != budget
        or _require_int(packed.get("dense_adapter_forward_count"), "dense adapter count missing") != 0
    ):
        raise ValueError("ragged VideoMAE executor violated its no-padding ledger")
    clip_counts = packed.get("clip_token_counts")
    attention_pairs = packed.get("attention_pairs_per_window")
    if (
        not isinstance(clip_counts, list)
        or len(clip_counts) != len(k_per_tubelet)
        or not isinstance(attention_pairs, list)
        or len(attention_pairs) != len(clip_counts)
    ):
        raise ValueError("ragged clip cost ledger is missing")
    for counts, pairs in zip(clip_counts, attention_pairs):
        if not isinstance(counts, list) or sum(counts) != budget:
            raise ValueError("per-clip b_c ledger does not sum to B")
        if sum(int(value) ** 2 for value in counts) != int(pairs):
            raise ValueError("ragged attention ledger is not sum_c b_c^2")

    output_shape = audit.get("output_shape")
    if output_shape != [1, 384, 768]:
        raise ValueError("dynamic P0 lost the AdaTAD [1,384,768] contract")
    losses = _require_mapping(report.get("losses"), "P0 loss receipt missing")
    if set(losses) != REQUIRED_LOSS_KEYS or any(
        not isinstance(value, (int, float)) for value in losses.values()
    ):
        raise ValueError("P0 did not execute the complete detector/auxiliary loss graph")
    if float(losses["georoute_geometry_regularization_loss"]) != 0.0:
        raise ValueError("dynamic main route must have zero geometry regularization")

    gradient = _require_mapping(report.get("gradient"), "gradient receipt missing")
    if (
        gradient.get("all_gradients_finite") is not True
        or set(gradient.get("required_components", [])) != REQUIRED_GRADIENT_COMPONENTS
        or gradient.get("missing_required_components") != []
        or not REQUIRED_GRADIENT_COMPONENTS
        <= set(gradient.get("nonzero_components", []))
    ):
        raise ValueError("real detector backward did not reach every required component")
    amp = _require_mapping(report.get("amp"), "AMP receipt missing")
    if (
        amp.get("autocast_dtype") != "torch.float16"
        or amp.get("optimizer_update_succeeded") is not True
        or float(amp.get("loss_scale_after", 0.0))
        < float(amp.get("loss_scale_before", 1.0))
    ):
        raise ValueError("dynamic Stage-1 GradScaler P0 did not pass")


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _gradient_summary(model) -> dict[str, Any]:
    import torch

    components: set[str] = set()
    nonfinite: list[str] = []
    missing_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_trainable.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
            continue
        if not bool(torch.count_nonzero(parameter.grad).item()):
            continue
        if name.startswith("backbone.scout.stem"):
            components.add("scout_stem")
        elif name.startswith("backbone.scout.geometry_head"):
            components.add("scout_geometry")
        elif name.startswith("backbone.scout.base_utility_head"):
            components.add("scout_base_utility")
        elif name.startswith("backbone.scout.residual_head"):
            components.add("scout_residual")
        elif name.startswith("backbone.dynamic_aux_head"):
            components.add("dynamic_aux_head")
        elif name.startswith("backbone.sparse_adapter"):
            components.add("sparse_adapter")
        elif ".adapter." in f".{name}":
            components.add("videomae_adapter")
        elif name.startswith("projection"):
            components.add("projection")
        elif name.startswith("rpn_head"):
            components.add("rpn_head")
        elif name.startswith("neck"):
            components.add("neck")
    missing_required = sorted(REQUIRED_GRADIENT_COMPONENTS - components)
    return {
        "all_gradients_finite": not nonfinite,
        "required_components": sorted(REQUIRED_GRADIENT_COMPONENTS),
        "nonzero_components": sorted(components),
        "missing_required_components": missing_required,
        "nonfinite_gradient_tensors": sorted(nonfinite),
        "missing_trainable_gradient_tensors": sorted(missing_trainable),
    }


def _configure_model(config_path: Path, pretrained: Path | None):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    custom = cfg.model.backbone.custom
    if str(custom.georoute_route_mode) != "dynamic_scnr":
        raise ValueError("dynamic Stage-1 P0 requires dynamic_scnr config")
    if pretrained is not None:
        custom.pretrain = str(pretrained.resolve())
    checkpoint = Path(str(custom.pretrain))
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
        custom.pretrain = str(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VideoMAE checkpoint is missing: {checkpoint}")
    return cfg, checkpoint


def _run_cuda_p0(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dynamic Stage-1 CUDA P0 must run inside Slurm")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("dynamic Stage-1 P0 requires Slurm logical cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("dynamic Stage-1 P0 requires one visible Slurm GPU")
    expected_commit = str(args.expected_commit).lower()
    if len(expected_commit) != 40:
        raise ValueError("expected commit must be a full 40-character SHA")
    head = _git("rev-parse", "HEAD").lower()
    origin = _git(
        "rev-parse",
        "refs/remotes/origin/codex/spatial-zoom-s1-audit-fix-20260715",
    ).lower()
    status = _git("status", "--porcelain")
    if head != expected_commit or origin != expected_commit or status:
        raise RuntimeError("dynamic Stage-1 P0 source is not exact and clean")

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    pretrained = None if args.pretrained is None else Path(args.pretrained)
    cfg, checkpoint = _configure_model(config_path, pretrained)
    custom = cfg.model.backbone.custom
    budget = int(custom.georoute_window_token_budget)
    if budget <= 0:
        raise ValueError("dynamic Stage-1 config has an invalid window budget")

    from opentad.models import build_detector

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = build_detector(cfg.model).to(device).train()
    model.backbone.set_successful_update_index(0)
    torch.cuda.reset_peak_memory_stats(device)

    source = torch.randint(
        0,
        256,
        (1, 1, 3, 768, 180, 320),
        dtype=torch.uint8,
        device=device,
    )
    scout = torch.randint(
        0,
        256,
        (1, 1, 3, 768, 96, 96),
        dtype=torch.uint8,
        device=device,
    )
    masks = torch.ones((1, 768), dtype=torch.bool, device=device)
    gt_segments = [
        torch.tensor([[96.0, 240.0]], dtype=torch.float32, device=device)
    ]
    gt_labels = [torch.tensor([0], dtype=torch.long, device=device)]
    metas = [
        {
            "video_name": "dynamic_stage1_p0_synthetic",
            "fps": 30.0,
            "duration": 25.6,
        }
    ]

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=256.0, growth_interval=2**30)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
        losses = model.forward_train(
            {"source": source, "scout": scout},
            masks=masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
        )
    if set(losses) != REQUIRED_LOSS_KEYS:
        raise RuntimeError(
            "dynamic Stage-1 P0 observed unexpected loss keys: "
            + ",".join(sorted(losses))
        )
    if any(
        not torch.is_tensor(value)
        or value.ndim != 0
        or not bool(torch.isfinite(value).item())
        for value in losses.values()
    ):
        raise FloatingPointError("dynamic Stage-1 P0 produced a nonfinite loss")
    scale_before = float(scaler.get_scale())
    scaler.scale(losses["cost"]).backward()
    scaler.unscale_(optimizer)
    gradient = _gradient_summary(model)
    if gradient["nonfinite_gradient_tensors"]:
        raise FloatingPointError("dynamic Stage-1 P0 produced nonfinite gradients")
    if gradient["missing_required_components"]:
        raise RuntimeError(
            "dynamic Stage-1 backward missed components: "
            + ",".join(gradient["missing_required_components"])
        )
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after < scale_before:
        raise FloatingPointError("dynamic Stage-1 GradScaler update overflowed")
    torch.cuda.synchronize(device)

    audit = dict(model.backbone.latest_georoute_audit or {})
    if not audit:
        raise RuntimeError("dynamic Stage-1 backbone audit is missing")
    report = {
        "schema_version": DYNAMIC_STAGE1_P0_SCHEMA,
        "status": "PASS_NO_PERFORMANCE_P0",
        "source": {
            "commit": head,
            "expected_commit": expected_commit,
            "origin_ref": origin,
            "head_matches_expected": head == expected_commit,
            "origin_ref_matches_expected": origin == expected_commit,
            "tree_clean": status == "",
        },
        "slurm": {
            "job_id": str(os.environ["SLURM_JOB_ID"]),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "visible_device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(device),
        },
        "config": {
            "path": str(config_path),
            "pretrained": str(checkpoint),
            "window_token_budget": budget,
            "native_tubelets": 384,
            "native_grid_hw": [11, 20],
            "native_spatial_candidates": 220,
            "source_shape": list(source.shape),
            "scout_shape": list(scout.shape),
            "successful_update": 0,
        },
        "losses": {
            name: float(value.detach().item()) for name, value in losses.items()
        },
        "gradient": gradient,
        "amp": {
            "autocast_dtype": "torch.float16",
            "loss_scale_before": scale_before,
            "loss_scale_after": scale_after,
            "optimizer": "sgd_lr_zero_overflow_probe",
            "optimizer_update_succeeded": scale_after >= scale_before,
        },
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "backbone_audit": audit,
        "scope": {
            "synthetic_inputs_only": True,
            "dataset_loaded": False,
            "metric_computed": False,
            "prediction_written": False,
            "checkpoint_written": False,
            "performance_claim_allowed": False,
            "official_test_opened": False,
        },
    }
    sealed = build_dynamic_stage1_p0_report(report)
    validate_dynamic_stage1_p0_report(sealed)
    return sealed


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _run_cuda_p0(args)
    _atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
