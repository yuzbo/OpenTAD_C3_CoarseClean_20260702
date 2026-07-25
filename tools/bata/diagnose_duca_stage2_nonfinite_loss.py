from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "duca_stage2_nonfinite_loss_diagnostic_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Stage-2 checkpoint must be a mapping")
    return payload


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).removeprefix("module."): value for key, value in state_dict.items()
    }


def _finite_tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value):
        return {"tensor": False, "type": type(value).__name__}
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    element_count = int(detached.numel())
    summary: dict[str, Any] = {
        "tensor": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "requires_grad": bool(value.requires_grad),
        "element_count": element_count,
        "finite": bool(finite.all().item()),
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(detached).sum().item()),
        "posinf_count": int(torch.isposinf(detached).sum().item()),
        "neginf_count": int(torch.isneginf(detached).sum().item()),
    }
    if finite_count:
        finite_values = detached[finite].float()
        summary["finite_min"] = float(finite_values.min().cpu().item())
        summary["finite_max"] = float(finite_values.max().cpu().item())
        if detached.ndim == 0:
            summary["scalar_value"] = float(finite_values.cpu().item())
    else:
        summary["finite_min"] = None
        summary["finite_max"] = None
        if detached.ndim == 0:
            summary["scalar_value"] = None
    return summary


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _batch_at_index(loader: Any, batch_index: int) -> Any:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    for observed_index, batch in enumerate(loader):
        if observed_index == batch_index:
            return batch
    raise RuntimeError(f"training loader ended before batch_index={batch_index}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_diagnostic(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_checkpoint_epoch: int,
    expected_commit: str,
    pretrain_path: str | Path,
    stage1_checkpoint: str | Path,
    stage1_checkpoint_sha256: str,
    stage1_checkpoint_epoch: int,
    epoch: int,
    batch_index: int,
    selector_schedule_step: int | None,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Replay exactly one training forward without backward or optimizer mutation."""

    import torch
    from mmengine.config import Config

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed
    from tools.bata.duca_frontend_initialization import initialize_model_from_checkpoint

    repo_root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if git_commit != expected_commit:
        raise RuntimeError("checkout commit differs from --expected-commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        text=True,
    ).strip():
        raise RuntimeError("read-only diagnostic requires a clean exact checkout")

    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    pretrain = Path(pretrain_path).expanduser().resolve()
    stage1 = Path(stage1_checkpoint).expanduser().resolve()
    if not config.is_file() or not pretrain.is_file() or not stage1.is_file():
        raise FileNotFoundError("config, pretrain, and Stage-1 checkpoint must exist")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint is missing: {checkpoint}")
    observed_checkpoint_sha256 = sha256_file(checkpoint)
    if observed_checkpoint_sha256 != checkpoint_sha256.lower():
        raise RuntimeError("Stage-2 checkpoint SHA256 mismatch")

    os.environ["DUCA_STAGE1_CHECKPOINT"] = str(stage1)
    os.environ["DUCA_STAGE1_CHECKPOINT_SHA256"] = stage1_checkpoint_sha256.lower()
    os.environ["DUCA_STAGE1_CHECKPOINT_EPOCH"] = str(int(stage1_checkpoint_epoch))

    set_seed(int(seed), False)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("one CUDA device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one Slurm-visible GPU")
    torch.cuda.set_device(device)

    cfg = Config.fromfile(str(config))
    cfg.model.backbone.custom.pretrain = str(pretrain)
    logger = logging.getLogger("duca-stage2-nonfinite-diagnostic")
    dataset = build_dataset(cfg.dataset.train, default_args={"logger": logger})
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )
    set_dataset_epoch = getattr(dataset, "set_epoch", None)
    if bool(cfg.workflow.get("derive_train_loader_contract", False)):
        if not callable(set_dataset_epoch):
            raise RuntimeError("Stage-2 training dataset lacks set_epoch")
        set_dataset_epoch(int(epoch))
    loader.sampler.set_epoch(int(epoch))

    # Match tools/train.py exactly so the diagnostic cannot change model parsing.
    model = build_detector(cfg.model)
    stage1_receipt = initialize_model_from_checkpoint(
        model, cfg.workflow.get("model_initialization"), logger=logger
    )
    if stage1_receipt is None:
        raise RuntimeError("Stage-2 diagnostic requires strict Stage-1 model initialization")
    model = model.to(device).train()

    checkpoint_payload = _load_checkpoint(checkpoint)
    if int(checkpoint_payload.get("epoch", -1)) != int(expected_checkpoint_epoch):
        raise RuntimeError("Stage-2 checkpoint epoch mismatch")
    state = checkpoint_payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks state_dict")
    incompatible = model.load_state_dict(_normalize_state_dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-2 checkpoint strict load did not close")

    selector = getattr(model, "frame_selector", None)
    if selector is None:
        raise RuntimeError("Stage-2 model lacks a DUCA frame selector")
    checkpoint_selector_step = int(
        selector._loss_weight_schedule_step.detach().cpu().item()
    )
    schedule_override_applied = False
    if selector_schedule_step is not None:
        allowed_steps = {checkpoint_selector_step, checkpoint_selector_step + 1}
        if int(selector_schedule_step) not in allowed_steps:
            raise RuntimeError(
                "selector schedule override must equal the checkpoint step or its next "
                "post-optimizer step"
            )
        if int(selector_schedule_step) != checkpoint_selector_step:
            # This is a strictly in-memory gate-state probe, never an optimizer update.
            selector._loss_weight_schedule_step.fill_(int(selector_schedule_step))
            selector._pending_loss_schedule_advance = False
            schedule_override_applied = True
    schedule_before = selector._loss_schedule_state()
    selector_step = int(selector._loss_weight_schedule_step.detach().cpu().item())

    batch = _move_to_device(_batch_at_index(loader, int(batch_index)), device)
    amp_enabled = bool(cfg.solver.get("amp", False))
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp_enabled):
        losses = model(**batch, return_loss=True)
    if not isinstance(losses, Mapping):
        raise RuntimeError("model did not return a loss mapping")

    loss_summary = {str(name): _finite_tensor_summary(value) for name, value in losses.items()}
    schedule_after = selector._loss_schedule_state()
    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != observed_checkpoint_sha256:
        raise RuntimeError("read-only diagnostic changed its input checkpoint")
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(loss_summary.get("cost", {}).get("finite", False)),
        "task": "offline_temporal_action_detection",
        "mode": "read_only_single_training_forward_no_backward_no_optimizer",
        "git_commit": git_commit,
        "config_path": str(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "checkpoint_state_key": "state_dict",
        "checkpoint_sha256_after": checkpoint_sha_after,
        "stage1_initialization": stage1_receipt,
        "replayed_epoch": int(epoch),
        "replayed_batch_index": int(batch_index),
        "seed": int(seed),
        "amp_enabled": amp_enabled,
        "backward_performed": False,
        "optimizer_constructed": False,
        "scheduler_constructed": False,
        "ema_constructed": False,
        "checkpoint_selector_schedule_step": checkpoint_selector_step,
        "selector_schedule_step": selector_step,
        "schedule_override_applied": schedule_override_applied,
        "in_memory_selector_schedule_step_override": selector_schedule_step,
        "schedule_before_forward": schedule_before,
        "schedule_after_forward": schedule_after,
        "losses": loss_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only one-batch Stage-2 non-finite loss diagnosis."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-epoch", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint-sha256", required=True)
    parser.add_argument("--stage1-checkpoint-epoch", required=True, type=int)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--selector-schedule-step", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).expanduser().resolve()
    try:
        report = run_diagnostic(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            expected_checkpoint_epoch=args.expected_checkpoint_epoch,
            expected_commit=args.expected_commit,
            pretrain_path=args.pretrain,
            stage1_checkpoint=args.stage1_checkpoint,
            stage1_checkpoint_sha256=args.stage1_checkpoint_sha256,
            stage1_checkpoint_epoch=args.stage1_checkpoint_epoch,
            epoch=args.epoch,
            batch_index=args.batch_index,
            selector_schedule_step=args.selector_schedule_step,
            seed=args.seed,
            device_name=args.device,
        )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "task": "offline_temporal_action_detection",
            "mode": "read_only_single_training_forward_no_backward_no_optimizer",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_write_json(output, report)
        raise
    _atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
