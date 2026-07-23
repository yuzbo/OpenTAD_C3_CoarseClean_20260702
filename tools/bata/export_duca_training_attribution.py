"""Export one fixed training batch's DUCA selection and detector attribution.

This is a post-training diagnostic.  It replays an existing checkpoint on a
deterministic THUMOS training batch, performs no optimizer update, and never
changes the selector or detector implementation.  GT is used to form the
*training* detector loss, while the exported GT boundaries are an overlay for
visualisation only and are never an inference-time selector input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RECORD_SCHEMA_VERSION = "duca_training_attribution_record_v1"
SUMMARY_SCHEMA_VERSION = "duca_training_attribution_summary_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_git_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tensor_row(value: Any, index: int, valid_len: int) -> list[float] | None:
    if value is None:
        return None
    row = value[index].detach().float().cpu()
    if row.ndim != 1 or int(row.shape[0]) < valid_len:
        raise ValueError("expected a [B,T] tensor aligned to the dense window")
    return [round(float(item), 8) for item in row[:valid_len].tolist()]


def _tensor_last_channel_matrix(
    value: Any, index: int, valid_len: int
) -> list[list[float]] | None:
    if value is None:
        return None
    matrix = value[index].detach().float().cpu()
    if matrix.ndim != 2 or int(matrix.shape[0]) < valid_len:
        raise ValueError("expected a [B,T,C] tensor aligned to the dense window")
    return [
        [round(float(item), 8) for item in row.tolist()]
        for row in matrix[:valid_len]
    ]


def _tensor_scalar(value: Any, index: int) -> float | None:
    if value is None:
        return None
    row = value[index].detach().float().cpu()
    if row.numel() != 1:
        raise ValueError("expected one scalar per batch row")
    return round(float(row.item()), 8)


def _selected_tensor_row(value: Any, index: int) -> list[float] | None:
    if value is None:
        return None
    row = value[index].detach().float().cpu()
    if row.ndim != 1:
        raise ValueError("expected a [B,K] selected-frame contribution tensor")
    return [round(float(item), 8) for item in row.tolist()]


def _sample_id(meta: Mapping[str, Any], fallback: int) -> str:
    video_id = str(meta.get("video_name") or meta.get("video_id") or f"sample_{fallback:06d}")
    return f"{video_id}|{int(meta.get('window_start_frame', 0))}"


def _selected_input_time_dim(inputs: Any) -> int:
    if inputs.ndim in {3, 5}:
        return 2
    if inputs.ndim == 6:
        return 3
    raise ValueError(f"unsupported selected detector input shape: {tuple(inputs.shape)}")


def _input_x_gradient(selected_inputs: Any, objective: Any) -> Any:
    """Return ``|input * d objective / d input|`` reduced to [B,K]."""

    import torch

    if selected_inputs is None or not bool(selected_inputs.requires_grad):
        return None
    gradient = torch.autograd.grad(
        objective,
        selected_inputs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return None
    time_dim = _selected_input_time_dim(selected_inputs)
    reduce_dims = tuple(
        axis for axis in range(selected_inputs.ndim) if axis not in {0, time_dim}
    )
    return (selected_inputs.detach() * gradient.detach()).abs().mean(dim=reduce_dims)


def _gradient_abs(tensor: Any, objective: Any) -> Any:
    """Return the actual magnitude ``|d objective / d tensor|`` without scaling it."""

    import torch

    if tensor is None or not bool(tensor.requires_grad):
        return None
    gradient = torch.autograd.grad(
        objective,
        tensor,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    return None if gradient is None else gradient.detach().abs()


def _loss_terms(losses: Mapping[str, Any], component: str) -> list[Any]:
    terms = []
    for name, value in losses.items():
        key = str(name).lower()
        if (
            "contribution" not in key
            and "loss" in key
            and component in key
            and getattr(value, "ndim", None) == 0
        ):
            terms.append(value)
    if not terms:
        raise RuntimeError(f"detector did not expose a scalar {component} loss")
    return terms


def _dense_contribution(selector: Any, selected: Any, scores: Mapping[str, Any]) -> Any:
    if selected is None:
        return None
    grid = scores.get("grid")
    valid = scores.get("valid_mask")
    if grid is None or valid is None:
        raise RuntimeError("selector output lacks grid or dense valid_mask")
    return selector._interpolate_selected_contribution(
        selected,
        grid.selected_positions,
        valid,
    ).detach()


def _select_fixed_training_batch(
    cfg: Any,
    *,
    batch_index: int,
    batch_size: int | None,
    device: Any,
) -> dict[str, Any]:
    """Pick a deterministic training batch without changing dataset order."""

    from opentad.datasets import build_dataloader, build_dataset

    dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": None})
    loader_cfg = copy.deepcopy(cfg.solver.train)
    if batch_size is not None:
        loader_cfg["batch_size"] = int(batch_size)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    wanted = int(batch_index)
    if wanted < 0:
        raise ValueError("batch_index must be non-negative")
    for index, raw in enumerate(loader):
        if index != wanted:
            continue
        return {
            "inputs": raw["inputs"].to(device, non_blocking=True),
            "masks": raw["masks"].to(device, dtype=__import__("torch").bool, non_blocking=True),
            "metas": [dict(meta) for meta in raw["metas"]],
            "gt_segments": [item.to(device, non_blocking=True) for item in raw["gt_segments"]],
            "gt_labels": [item.to(device, non_blocking=True) for item in raw["gt_labels"]],
            "gt_boundary_validity": raw.get("gt_boundary_validity"),
        }
    raise ValueError(f"training batch {wanted} does not exist")


def _checkpoint_model_state(checkpoint: Mapping[str, Any], use_ema: str) -> tuple[str, Mapping[str, Any]]:
    if use_ema == "state_dict":
        state = checkpoint.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint is missing state_dict")
        return "state_dict", state
    if use_ema == "state_dict_ema":
        state = checkpoint.get("state_dict_ema")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint is missing state_dict_ema")
        return "state_dict_ema", state
    state = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    key = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is missing mapping {key}")
    return key, state


def normalize_model_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict non-DDP state dict for a fresh detector instance.

    Training checkpoints are emitted from ``DistributedDataParallel`` while
    this diagnostic intentionally rebuilds one plain detector.  Strip the
    uniform DDP prefix, but fail closed on a mixed namespace rather than
    silently loading a partial model.
    """

    keys = [str(key) for key in state_dict]
    has_ddp_prefix = [key.startswith("module.") for key in keys]
    if any(has_ddp_prefix) and not all(has_ddp_prefix):
        raise ValueError("checkpoint state dict mixes DDP and non-DDP keys")
    if not any(has_ddp_prefix):
        return {str(key): value for key, value in state_dict.items()}
    return {str(key)[len("module.") :]: value for key, value in state_dict.items()}


def _record_rows(
    *,
    batch: Mapping[str, Any],
    scores: Mapping[str, Any],
    cls_selected: Any,
    reg_selected: Any,
    cls_target: Any,
    reg_target: Any,
    sampling_logit_gradient: Any,
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grid = scores.get("grid")
    if grid is None:
        raise RuntimeError("selector output lacks SparseTemporalGrid")
    positions = grid.selected_positions.detach().cpu().long()
    predicted = scores.get("detector_contribution_logits")
    contribution_targets = scores.get("detector_contribution_targets", {})
    valid_mask = scores.get("valid_mask", batch["masks"])
    records: list[dict[str, Any]] = []
    for row, meta in enumerate(batch["metas"]):
        valid_len = int(valid_mask[row].detach().long().sum().cpu().item())
        selected = [
            int(item)
            for item in positions[row].tolist()
            if 0 <= int(item) < valid_len
        ]
        gt_segments = batch["gt_segments"][row].detach().float().cpu().tolist()
        validity = batch.get("gt_boundary_validity")
        if validity is None:
            gt_validity = [[True, True] for _ in gt_segments]
        else:
            raw_validity = validity[row]
            gt_validity = (
                raw_validity.detach().cpu().tolist()
                if hasattr(raw_validity, "detach")
                else raw_validity
            )
        records.append(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "sample_id": _sample_id(meta, row),
                "video_id": str(meta.get("video_name") or meta.get("video_id") or f"sample_{row:06d}"),
                "window_start_frame": int(meta.get("window_start_frame", 0)),
                "snippet_stride": int(meta.get("snippet_stride", 1)),
                "valid_len": valid_len,
                "selected_positions": selected,
                "p_action": _tensor_row(scores.get("p_action"), row, valid_len),
                "abs_delta_p_action": _tensor_row(scores.get("abs_delta_p_action"), row, valid_len),
                "transition_policy_scores": _tensor_row(
                    scores.get("transition_policy_scores", scores.get("center_scores")), row, valid_len
                ),
                "sampling_rate_logits": _tensor_row(scores.get("decode_policy_logits"), row, valid_len),
                "sampling_rates": _tensor_row(scores.get("sampling_rates"), row, valid_len),
                "sampling_density": _tensor_row(scores.get("sampling_density"), row, valid_len),
                "sampling_cumulative_rates": _tensor_row(scores.get("sampling_cumulative_rates"), row, valid_len),
                "sampling_rate_logit_gradient_abs": _tensor_row(
                    sampling_logit_gradient, row, valid_len
                ),
                "detector_contribution_logits": _tensor_last_channel_matrix(
                    predicted, row, valid_len
                ),
                "detector_contribution_prediction_distribution": _tensor_last_channel_matrix(
                    scores.get("detector_contribution_prediction_distribution"), row, valid_len
                ),
                # The first pair is measured only on the K detector inputs.  The
                # dense pair is their train-only linear reconstruction for an
                # aligned 768-point plot, not a new per-frame measurement.
                "detector_cls_selected_input_x_gradient": _selected_tensor_row(cls_selected, row),
                "detector_reg_selected_input_x_gradient": _selected_tensor_row(reg_selected, row),
                "detector_cls_input_x_gradient_dense_interpolated": _tensor_row(
                    cls_target, row, valid_len
                ),
                "detector_reg_input_x_gradient_dense_interpolated": _tensor_row(
                    reg_target, row, valid_len
                ),
                "detector_cls_contribution_target": _tensor_row(
                    contribution_targets.get("cls") if isinstance(contribution_targets, Mapping) else None,
                    row,
                    valid_len,
                ),
                "detector_reg_contribution_target": _tensor_row(
                    contribution_targets.get("reg") if isinstance(contribution_targets, Mapping) else None,
                    row,
                    valid_len,
                ),
                "sampling_calibration_residual": _tensor_scalar(
                    scores.get("sampling_calibration_residual"), row
                ),
                "gt_segments": [
                    [round(float(start), 7), round(float(end), 7)]
                    for start, end in gt_segments
                ],
                "gt_boundary_validity": [[bool(pair[0]), bool(pair[1])] for pair in gt_validity],
                "gt_role": "visualization_overlay_and_train_loss_only_not_inference_decision",
                "source": dict(source),
            }
        )
    return records


def export_training_attribution(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    device: str = "cuda:0",
    checkpoint_state: str = "state_dict",
    batch_index: int = 0,
    batch_size: int | None = 2,
    seed: int = 3407,
) -> dict[str, Any]:
    """Replay one real training batch and export train-only attribution evidence."""

    import numpy as np
    import torch
    from mmengine.config import Config
    from opentad.models import build_detector

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    summary_path = Path(summary_json).expanduser().resolve()
    if output_path == summary_path:
        raise ValueError("output_jsonl and summary_json must be different paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite training attribution evidence")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = Config.fromfile(str(config_path))
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_key, state_dict = _checkpoint_model_state(checkpoint, checkpoint_state)
    model = build_detector(copy.deepcopy(cfg.model))
    incompatible = model.load_state_dict(normalize_model_state_dict(state_dict), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("full checkpoint did not strictly match the configured detector")
    model = model.to(torch_device).train()
    selector = getattr(model, "frame_selector", None)
    if selector is None:
        raise ValueError("training attribution requires a model.frame_selector")
    batch = _select_fixed_training_batch(
        cfg,
        batch_index=int(batch_index),
        batch_size=batch_size,
        device=torch_device,
    )
    if batch.get("gt_boundary_validity") is None:
        raise ValueError("real training loader omitted gt_boundary_validity")

    captured: dict[str, Any] = {}
    original_forward_train = selector.forward_train

    def capture_forward_train(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        output = original_forward_train(*args, **kwargs)
        captured["output"] = output
        return output

    selector.forward_train = capture_forward_train
    try:
        model.zero_grad(set_to_none=True)
        losses = model(
            batch["inputs"],
            batch["masks"],
            batch["metas"],
            gt_segments=batch["gt_segments"],
            gt_labels=batch["gt_labels"],
            gt_boundary_validity=batch["gt_boundary_validity"],
            return_loss=True,
        )
    finally:
        selector.forward_train = original_forward_train
    if "output" not in captured:
        raise RuntimeError("selector.forward_train was not called by the full detector")
    selector_output = captured["output"]
    scores = selector_output.get("selector_outputs")
    if not isinstance(scores, Mapping):
        raise RuntimeError("selector did not expose selector_outputs")

    cls_objective = sum(_loss_terms(losses, "cls"))
    reg_objective = sum(_loss_terms(losses, "reg"))
    detector_objective = cls_objective + reg_objective
    selected_inputs = scores.get("detector_contribution_teacher_inputs")
    cls_selected = _input_x_gradient(selected_inputs, cls_objective)
    reg_selected = _input_x_gradient(selected_inputs, reg_objective)
    cls_target = _dense_contribution(selector, cls_selected, scores)
    reg_target = _dense_contribution(selector, reg_selected, scores)
    sampling_logits = scores.get("decode_policy_logits")
    sampling_logit_gradient = _gradient_abs(sampling_logits, detector_objective)

    logits = scores.get("detector_contribution_logits")
    valid = scores.get("valid_mask")
    if logits is not None and valid is not None:
        masked = logits.masked_fill(~valid[:, :, None].bool(), float("-inf"))
        scores = dict(scores)
        scores["detector_contribution_prediction_distribution"] = torch.softmax(masked, dim=1)

    repo_root = _find_git_root(config_path.parent) or Path.cwd().resolve()
    source = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_epoch_one_based": (
            None
            if checkpoint.get("epoch") is None
            else int(checkpoint["epoch"]) + 1
        ),
        "git_commit": _git_commit(repo_root),
        "split": "train",
        "fixed_batch_index": int(batch_index),
        "batch_size": int(batch["inputs"].shape[0]),
        "seed": int(seed),
        "optimizer_step_executed": False,
        "detector_executed": True,
        "gt_used_for_train_loss": True,
        "gt_used_for_inference_decision": False,
        "train_only_attribution": True,
    }
    records = _record_rows(
        batch=batch,
        scores=scores,
        cls_selected=cls_selected,
        reg_selected=reg_selected,
        cls_target=cls_target,
        reg_target=reg_target,
        sampling_logit_gradient=sampling_logit_gradient,
        source=source,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "output_jsonl": str(output_path),
        "record_count": len(records),
        "source": source,
        "losses": {
            "detector_cls": float(cls_objective.detach().float().cpu().item()),
            "detector_reg": float(reg_objective.detach().float().cpu().item()),
            "detector_cls_reg": float(detector_objective.detach().float().cpu().item()),
        },
        "evidence_contract": {
            "selector_or_decoder_modified": False,
            "optimizer_step_executed": False,
            "gt_overlay_not_inference_input": True,
            "input_x_gradient_is_train_only": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export train-only DUCA detector contribution and sampling-gradient evidence."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-state",
        choices=("state_dict", "state_dict_ema", "auto"),
        default="state_dict",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = export_training_attribution(
        config=args.config,
        checkpoint=args.checkpoint,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        device=args.device,
        checkpoint_state=args.checkpoint_state,
        batch_index=args.batch_index,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
