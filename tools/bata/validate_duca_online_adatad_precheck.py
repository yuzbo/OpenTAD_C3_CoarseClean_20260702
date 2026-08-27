from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmengine.config import Config, ConfigDict


CONFIG_DEFAULT = "configs/adatad/thumos/duca_online_adatad_precheck.py"
FORBIDDEN_MODEL_TOKENS = (
    "ledger_path",
    "value_transport",
    "load_from_raw_predictions': true",
    "save_raw_prediction': true",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _metadata_keys(cfg: Config) -> Mapping[str, str]:
    keys = cfg.duca_online_precheck_contract.metadata_keys
    required = (
        "selected_positions",
        "selected_positions_unit",
        "selected_mask",
        "selected_count",
        "remap",
        "source",
    )
    for key in required:
        _require(key in keys, f"missing metadata key declaration: {key}")
    return keys


def _validate_static_config(cfg: Config, config_path: str) -> dict[str, Any]:
    contract = cfg.duca_online_precheck_contract
    selector = cfg.model.frame_selector
    head = cfg.model.rpn_head
    teacher_cfg = head.get("teacher_cfg", {})
    source_cfg = selector.get("actionness_source_cfg", {})
    keys = _metadata_keys(cfg)

    _require(cfg.model.type == "SingleStageDetector", "model.type must be SingleStageDetector")
    _require(selector.type == "DucaOnlineFrameSelector", "frame_selector must be DucaOnlineFrameSelector")
    _require(head.type == "DucaOnlinePrecheckHead", "rpn_head must be DucaOnlinePrecheckHead")
    _require(contract.no_ledger_decision is True, "contract must declare no_ledger_decision")
    _require(selector.get("no_ledger_decision") is True, "selector must declare no_ledger_decision")
    _require(selector.get("forbid_ledger") is True, "selector must forbid ledgers")
    _require(int(selector.budget) <= 384, "selector budget must be <=384")
    _require(int(contract.budget_max) <= 384, "contract budget_max must be <=384")
    _require(contract.coordinate_space == "original_time", "contract coordinate_space must be original_time")
    _require(selector.coordinate_space == "original_time", "selector coordinate_space must be original_time")
    _require(contract.selected_positions_unit == "original_time_index", "positions unit must be original_time_index")
    _require(selector.selected_positions_unit == "original_time_index", "selector positions unit must be original_time_index")
    _require(contract.teacher_free_eval is True, "eval must be teacher-free")
    _require(teacher_cfg.get("train_only") is True, "teacher config must be train-only")
    _require(teacher_cfg.get("enabled_for_inference") is False, "teacher must be disabled for inference")
    _require(teacher_cfg.get("forbid_inference") is True, "teacher inference must be forbidden")
    _require(contract.selected_axis_remap_required is True, "selected-axis remap must be required")
    _require(selector.get("remap_gt_to_selected_axis") is True, "selector must remap GT to selected axis")
    _require(head.get("require_selected_axis_remap") is True, "head must require selected-axis remap")
    _require(head.get("require_gt_in_train") is True, "head must require train GT")
    _require(not _as_bool(cfg.inference.load_from_raw_predictions), "raw prediction loading must be off")
    _require(not _as_bool(cfg.inference.save_raw_prediction), "raw prediction saving must be off")
    _require(source_cfg.get("no_teacher") is True, "actionness source must not use teacher")
    _require(source_cfg.get("no_oracle") is True, "actionness source must not use oracle")
    _require(source_cfg.get("no_raw_prediction_cache") is True, "actionness source must not use raw prediction cache")
    _require(source_cfg.get("no_gt_generation") is True, "actionness source must not generate from GT")
    if contract.actionness_source == "zero_shot_motion":
        _require(source_cfg.type == "ZeroShotMotionActionnessSource", "zero-shot config must use zero-shot source")
        _require(source_cfg.get("no_train_gt") is True, "zero-shot source must not use train GT")

    model_text = repr(cfg.model).lower()
    for forbidden in FORBIDDEN_MODEL_TOKENS:
        _require(forbidden not in model_text, f"forbidden model token present: {forbidden}")

    return {
        "config_path": str(config_path),
        "config_import": True,
        "model_type": str(cfg.model.type),
        "frame_selector_type": str(selector.type),
        "rpn_head_type": str(head.type),
        "actionness_source": str(contract.actionness_source),
        "actionness_source_type": str(source_cfg.type),
        "no_ledger": True,
        "teacher_only_train_loss": True,
        "no_teacher_in_inference": True,
        "budget_lte_384": True,
        "budget": int(selector.budget),
        "coordinate_space": str(contract.coordinate_space),
        "selected_positions_unit": str(contract.selected_positions_unit),
        "selected_axis_remap_required": True,
        "raw_prediction_cache_forbidden": True,
        "metadata_keys": dict(keys),
    }


def _toy_batch(cfg: Config):
    import torch

    batch = 2
    channels = int(cfg.get("toy_input_channels", 8))
    dense_len = int(cfg.get("dense_window_size", 768))
    inputs = torch.linspace(0.0, 1.0, steps=batch * channels * dense_len, dtype=torch.float32).reshape(
        batch,
        channels,
        dense_len,
    )
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [
        {
            "video_name": f"duca_online_precheck_{idx}",
            "fps": 30.0,
            "duration": float(dense_len) / 30.0,
            "snippet_stride": 1,
            "window_start_frame": 100 * idx,
            "window_size": dense_len,
            "original_time_axis": True,
        }
        for idx in range(batch)
    ]
    gt_segments = [
        torch.tensor([[8.0, 32.0], [128.0, 180.0]], dtype=torch.float32),
        torch.tensor([[16.0, 64.0]], dtype=torch.float32),
    ]
    gt_labels = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([3], dtype=torch.long),
    ]
    return inputs, masks, metas, gt_segments, gt_labels


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().reshape(-1).tolist()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _selected_count(value: Any) -> int | None:
    try:
        import torch

        if torch.is_tensor(value):
            if value.dtype == torch.bool:
                return int(value.detach().cpu().bool().sum().item())
            return int(value.detach().cpu().reshape(-1).numel())
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return len(value)
    if value is None:
        return None
    return int(value)


def _summary_from_runtime_objects(model: Any, metas: list[dict[str, Any]], keys: Mapping[str, str]) -> dict[str, Any]:
    runtime = {}
    for holder in (
        getattr(model, "frame_selector", None),
        getattr(model, "rpn_head", None),
        model,
    ):
        if holder is None:
            continue
        for attr in ("last_precheck_summary", "last_forward_summary", "debug_last_forward", "last_contract"):
            value = getattr(holder, attr, None)
            if isinstance(value, Mapping):
                runtime.update(value)

    first_meta = metas[0] if metas else {}
    positions = _to_list(first_meta.get(keys["selected_positions"], runtime.get(keys["selected_positions"])))
    selected_mask = first_meta.get(keys["selected_mask"], runtime.get(keys["selected_mask"]))
    selected_count = first_meta.get(keys["selected_count"], runtime.get(keys["selected_count"]))
    remap = first_meta.get(keys["remap"], runtime.get(keys["remap"]))
    unit = first_meta.get(keys["selected_positions_unit"], runtime.get(keys["selected_positions_unit"]))
    mask_count = _selected_count(selected_mask)
    if mask_count is None and selected_count is not None:
        mask_count = int(selected_count)

    return {
        "positions": positions,
        "positions_unit": unit,
        "selected_mask_count": mask_count,
        "selected_count": int(selected_count) if selected_count is not None else None,
        "remap": remap,
        "runtime_debug": runtime,
    }


def _validate_runtime(cfg: Config) -> dict[str, Any]:
    import torch

    from opentad.models import build_detector

    model = build_detector(cfg.model)
    model.cpu()
    keys = _metadata_keys(cfg)
    inputs, masks, metas, gt_segments, gt_labels = _toy_batch(cfg)

    model.train()
    train_losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    _require(isinstance(train_losses, Mapping), "train forward must return a loss mapping")
    _require(len(train_losses) > 0, "train forward must return at least one loss")
    _require("cost" in train_losses, "train losses must include cost")

    model.eval()
    _, _, test_metas, _, _ = _toy_batch(cfg)
    with torch.no_grad():
        model(
            inputs,
            masks,
            test_metas,
            gt_segments=None,
            gt_labels=None,
            return_loss=False,
            infer_cfg=ConfigDict(load_from_raw_predictions=False, save_raw_prediction=False),
            post_cfg=ConfigDict(cfg.get("post_processing", dict(sliding_window=True, nms=None))),
            ext_cls=[f"class_{idx}" for idx in range(int(cfg.get("num_classes", 20)))],
        )

    runtime = _summary_from_runtime_objects(model, test_metas, keys)
    positions = [int(item) for item in runtime["positions"]]
    budget = int(cfg.model.frame_selector.budget)
    dense_len = int(cfg.dense_window_size)
    _require(positions, "selected positions metadata missing after inference")
    _require(positions == sorted(set(positions)), "selected positions must be sorted unique")
    _require(all(0 <= pos < dense_len for pos in positions), "selected positions must be original-time indices")
    _require(runtime["positions_unit"] == "original_time_index", "selected positions unit must be original_time_index")
    _require(len(positions) <= budget, "selected count must be <= budget")
    _require(runtime["selected_mask_count"] == len(positions), "selected mask count must match selected positions")
    _require(runtime["selected_count"] in (None, len(positions)), "selected_count metadata must match positions")
    _require(isinstance(runtime["remap"], Mapping), "selected-axis remap metadata must be present")
    _require(
        any(key in runtime["remap"] for key in ("selected_to_original", "original_to_selected", "gt_segments_selected_axis")),
        "remap metadata must expose selected/original mapping",
    )

    train_loss_keys = sorted(str(key) for key in train_losses.keys())
    inference_text = repr(test_metas).lower() + repr(runtime["runtime_debug"]).lower()
    _require("teacher" not in inference_text, "teacher metadata/debug output must not appear in inference")
    _require("raw_prediction_cache" not in inference_text, "raw prediction cache must not appear in inference metadata")

    return {
        "build_detector": True,
        "standard_forward_train": True,
        "standard_forward_test": True,
        "gt_reaches_detector_train": True,
        "selected_positions_original_time": True,
        "masks_selected_count": True,
        "remap_metadata_present": True,
        "train_loss_keys": train_loss_keys,
        "runtime_selected_count": len(positions),
    }


def validate_config(config_path: str = CONFIG_DEFAULT, *, run_runtime: bool | None = None) -> dict[str, Any]:
    cfg = Config.fromfile(str(config_path))
    summary = _validate_static_config(cfg, config_path)

    if run_runtime is None:
        run_runtime = os.environ.get("DUCA_ONLINE_PRECHECK_RUNTIME", "1") != "0"
    if not run_runtime:
        summary.update(
            {
                "build_detector": "skipped",
                "standard_forward_train": "skipped",
                "standard_forward_test": "skipped",
                "gt_reaches_detector_train": "runtime_required",
                "selected_positions_original_time": "runtime_required",
                "masks_selected_count": "runtime_required",
                "remap_metadata_present": "runtime_required",
            }
        )
        return summary

    summary.update(_validate_runtime(cfg))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--no-runtime", action="store_true")
    args = parser.parse_args(argv)

    summary: dict[str, Any]
    try:
        summary = validate_config(args.config, run_runtime=not bool(args.no_runtime))
    except Exception as exc:
        summary = {
            "config_path": str(args.config),
            "ok": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    summary["ok"] = True
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
