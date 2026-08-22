#!/usr/bin/env python3
"""Render on-policy ZoomToken native-token selections for one THUMOS14 window.

Each row is evaluated with that arm's own checkpoint.  The script invokes the
same pre-backbone routing function used by the production forward path, but it
stops before the heavy VideoMAE execution.  Consequently the artifact is
qualitative routing evidence, not an accuracy or cost measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.datasets import build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.models.backbones.georoute_wrapper import (  # noqa: E402
    extract_native_tubelets,
)


@dataclass(frozen=True)
class ArmDefinition:
    key: str
    row_label: str
    config_name: str


ARMS: tuple[ArmDefinition, ...] = (
    ArmDefinition("B", "B: ALL-100", "georoute_official_b_alltoken_prebackbone_seed42_v001.py"),
    ArmDefinition("C", "C: ROI-64", "georoute_official_c_roi_k64_prebackbone_seed42_v001.py"),
    ArmDefinition("R1", "R1: RECT-8x8", "georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"),
    ArmDefinition("R2", "R2: RECT-8x8 / Q48", "georoute_official_r2_strict_rect8x8_q48_prebackbone_seed42_v001.py"),
    ArmDefinition("R2-SHUF48", "R2-SHUF48", "georoute_official_r2_shuf48_prebackbone_seed42_v001.py"),
    ArmDefinition("Q48-GLOBAL", "Q48-GLOBAL", "georoute_official_q48_global_prebackbone_seed42_v001.py"),
    ArmDefinition("R3", "R3: DYNAMIC-RECT", "georoute_official_r3_continuous_rect_prebackbone_seed42_v001.py"),
    ArmDefinition("R3-AREA-SHIFT", "R3-AREA-SHIFT97", "georoute_official_r3_area_shift97_prebackbone_seed42_v001.py"),
    ArmDefinition("R4", "R4: RECT-7x7 + Q15", "georoute_official_r4_core49_q15_prebackbone_seed42_v001.py"),
    ArmDefinition("R4-SHUF15", "R4-SHUF15", "georoute_official_r4_shuf15_prebackbone_seed42_v001.py"),
    ArmDefinition("Q64-GLOBAL", "Q64-GLOBAL", "georoute_official_q64_global_prebackbone_seed42_v001.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help="checkpoint binding; provide exactly once for every frozen arm",
    )
    parser.add_argument(
        "--qualitative",
        action="append",
        default=[],
        metavar="ARM",
        help="mark an arm as recovery-checkpoint qualitative evidence only",
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-name", default="video_test_0001339")
    parser.add_argument("--video-window-index", type=int, default=0)
    parser.add_argument(
        "--tubelets",
        default="32,96,160,224,288,352",
        help="comma-separated tubelet ordinals in the 384-tubelet window",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _parse_checkpoint_bindings(raw: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    allowed = {arm.key for arm in ARMS}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"checkpoint binding must be ARM=PATH, got {item!r}")
        key, path = item.split("=", 1)
        if key not in allowed:
            raise ValueError(f"unknown arm in checkpoint binding: {key!r}")
        if key in bindings:
            raise ValueError(f"duplicate checkpoint binding for {key}")
        bindings[key] = Path(path).resolve()
    missing = allowed - bindings.keys()
    if missing:
        raise ValueError(f"missing checkpoint bindings: {sorted(missing)}")
    for key, path in bindings.items():
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint for {key} does not exist: {path}")
    return bindings


def _parse_tubelets(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("tubelet list must be non-empty and unique")
    if min(values) < 0 or max(values) >= 384:
        raise ValueError("tubelet ordinals must lie in [0, 383]")
    return values


def _configure_test_dataset(cfg: Config, args: argparse.Namespace) -> None:
    cfg.dataset.test.ann_file = str(Path(args.annotation).resolve())
    cfg.dataset.test.class_map = str(Path(args.class_map).resolve())
    cfg.dataset.test.data_path = str(Path(args.video_root).resolve())
    cfg.dataset.test.subset_name = "validation"
    cfg.dataset.test.test_mode = True


def _select_sample(dataset: Any, video_name: str, video_window_index: int) -> tuple[int, dict[str, Any]]:
    matches = [
        index
        for index, entry in enumerate(dataset.data_list)
        if str(entry[0]) == str(video_name)
    ]
    if not matches:
        raise ValueError(f"video {video_name!r} has no validation windows")
    if video_window_index < 0 or video_window_index >= len(matches):
        raise ValueError(
            f"video-window index {video_window_index} is outside [0, {len(matches) - 1}]"
        )
    sample_index = matches[video_window_index]
    return sample_index, dataset[sample_index]


def _checkpoint_state(checkpoint: dict[str, Any], use_ema: bool) -> tuple[str, dict[str, torch.Tensor]]:
    if use_ema and "state_dict_ema" in checkpoint:
        return "state_dict_ema", checkpoint["state_dict_ema"]
    if "state_dict" in checkpoint:
        return "state_dict", checkpoint["state_dict"]
    raise KeyError("checkpoint contains neither state_dict_ema nor state_dict")


def _load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path, use_ema: bool) -> tuple[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_key, state = _checkpoint_state(checkpoint, use_ema)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as original_error:
        if state and all(str(key).startswith("module.") for key in state):
            stripped = {str(key)[7:]: value for key, value in state.items()}
            try:
                model.load_state_dict(stripped, strict=True)
            except RuntimeError:
                raise original_error
        else:
            raise
    return state_key, checkpoint.get("epoch")


def _route_mask(
    route: dict[str, Any],
    *,
    tubelet_count: int,
    spatial_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "physical_indices" in route:
        physical = route["physical_indices"].detach().to(device="cpu", dtype=torch.long)
        if physical.ndim != 2 or physical.shape[0] != 1:
            raise ValueError("dynamic route physical_indices must be [1,S]")
        flat_mask = torch.zeros(tubelet_count * spatial_tokens, dtype=torch.bool)
        flat_mask[physical[0]] = True
        mask = flat_mask.reshape(tubelet_count, spatial_tokens)
        k_per_tubelet = mask.sum(dim=1)
        return mask, k_per_tubelet, physical[0]

    spatial = route["spatial_indices"].detach().to(device="cpu", dtype=torch.long)
    if spatial.ndim != 3 or tuple(spatial.shape[:2]) != (1, tubelet_count):
        raise ValueError("fixed route spatial_indices must be [1,T,K]")
    mask = torch.zeros((tubelet_count, spatial_tokens), dtype=torch.bool)
    mask.scatter_(1, spatial[0], True)
    offsets = torch.arange(tubelet_count, dtype=torch.long).view(-1, 1) * spatial_tokens
    physical = (spatial[0] + offsets).reshape(-1)
    return mask, mask.sum(dim=1), physical


@torch.no_grad()
def _capture_arm(
    arm: ArmDefinition,
    checkpoint_path: Path,
    sample_inputs: torch.Tensor,
    sample_meta: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    config_path = REPO_ROOT / "configs" / "adatad" / "thumos" / arm.config_name
    cfg = Config.fromfile(str(config_path))
    cfg.model.backbone.custom.pretrain = None
    model = build_detector(cfg.model)
    state_key, checkpoint_epoch = _load_checkpoint_strict(
        model,
        checkpoint_path,
        bool(getattr(cfg.solver, "ema", False)),
    )
    backbone = model.backbone.to(device)
    backbone.eval()

    frames = sample_inputs.unsqueeze(0).to(device=device, non_blocking=True)
    source = backbone._validate_official_fixed_support_input(frames)
    native, source_grid_hw, ignored_border_hw, valid_patch_mask = extract_native_tubelets(
        source,
        patch_size=backbone.patch_size,
        tubelet_size=backbone.tubelet_size,
    )
    window_ordinals = torch.tensor(
        [int(sample_meta["window_ordinal"])],
        device=device,
        dtype=torch.long,
    )
    route = backbone._official_fixed_support_route(
        source,
        source_grid_hw=source_grid_hw,
        valid_patch_mask=valid_patch_mask,
        window_ordinals=(
            window_ordinals if backbone.requires_route_window_ordinals else None
        ),
    )
    tubelet_count, spatial_tokens = map(int, native.shape[1:3])
    mask, k_per_tubelet, physical_indices = _route_mask(
        route,
        tubelet_count=tubelet_count,
        spatial_tokens=spatial_tokens,
    )
    geometry = route.get("geometry")
    result = {
        "arm": arm.key,
        "row_label": arm.row_label,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": checkpoint_epoch,
        "official_support": backbone.official_support,
        "source_grid_hw": list(map(int, source_grid_hw)),
        "ignored_border_hw": list(map(int, ignored_border_hw)),
        "mask": mask,
        "k_per_tubelet": k_per_tubelet,
        "physical_indices": physical_indices,
        "geometry": None
        if geometry is None
        else geometry.detach().to(device="cpu", dtype=torch.float32),
    }

    backbone.cpu()
    del route, native, valid_patch_mask, source, frames, backbone, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _frame_rgb(sample_inputs: torch.Tensor, input_frame_index: int) -> np.ndarray:
    if sample_inputs.ndim != 5 or tuple(sample_inputs.shape[:2]) != (1, 3):
        raise ValueError("sample inputs must be [1,3,T,H,W]")
    frame = sample_inputs[0, :, input_frame_index].permute(1, 2, 0).cpu().numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _mask_frame(frame: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    if frame.shape[:2] != (160, 160) or token_mask.shape != (10, 10):
        raise ValueError("visualization requires a 160x160 frame and a 10x10 token mask")
    gray = np.dot(frame[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1)
    gray_rgb = np.clip(gray_rgb * 0.72 + 28.0, 0, 255)
    pixel_mask = np.repeat(np.repeat(token_mask, 16, axis=0), 16, axis=1)
    rendered = np.where(pixel_mask[..., None], frame, gray_rgb)
    return np.clip(rendered, 0, 255).astype(np.uint8)


def _draw_grid(ax: Any) -> None:
    for coordinate in range(0, 161, 16):
        ax.axhline(coordinate - 0.5, color="white", linewidth=0.30, alpha=0.58)
        ax.axvline(coordinate - 0.5, color="white", linewidth=0.30, alpha=0.58)


def _render_figure(
    captures: list[dict[str, Any]],
    sample_inputs: torch.Tensor,
    tubelets: list[int],
    qualitative: set[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    columns = len(tubelets)
    rows = len(captures)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.25 * columns, 1.68 * rows),
        squeeze=False,
        constrained_layout=False,
    )
    for row_index, capture in enumerate(captures):
        mask = capture["mask"].numpy().reshape(384, 10, 10)
        k_values = capture["k_per_tubelet"].tolist()
        checkpoint_name = Path(capture["checkpoint"]).name
        row_label = capture["row_label"]
        if capture["arm"] in qualitative:
            row_label += f" †\n{checkpoint_name}"
        for column_index, tubelet in enumerate(tubelets):
            ax = axes[row_index, column_index]
            input_frame = 2 * tubelet
            frame = _frame_rgb(sample_inputs, input_frame)
            ax.imshow(_mask_frame(frame, mask[tubelet]), interpolation="nearest")
            _draw_grid(ax)
            ax.text(
                0.025,
                0.965,
                f"K={int(k_values[tubelet])}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="white",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.62, "linewidth": 0},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(
                    f"tubelet {tubelet}\ninput frames {input_frame}–{input_frame + 1}",
                    fontsize=8.8,
                    pad=5,
                )
            if column_index == 0:
                ax.set_ylabel(row_label, fontsize=8.4, rotation=0, ha="right", va="center", labelpad=54)

    figure.subplots_adjust(left=0.185, right=0.995, top=0.965, bottom=0.035, hspace=0.055, wspace=0.035)
    if qualitative:
        figure.text(
            0.005,
            0.008,
            "† current recovery checkpoint; qualitative observation only",
            ha="left",
            va="bottom",
            fontsize=8.3,
        )
    png_path = output_dir / "zoomtoken_on_policy_token_selection.png"
    pdf_path = output_dir / "zoomtoken_on_policy_token_selection.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _jsonable_capture(capture: dict[str, Any], tubelets: list[int], qualitative: set[str]) -> dict[str, Any]:
    mask = capture["mask"]
    geometry = capture["geometry"]
    displayed = {}
    for tubelet in tubelets:
        displayed[str(tubelet)] = {
            "input_frame_pair": [2 * tubelet, 2 * tubelet + 1],
            "selected_spatial_indices": torch.nonzero(mask[tubelet], as_tuple=False).flatten().tolist(),
            "selected_count": int(capture["k_per_tubelet"][tubelet].item()),
            "geometry_cxcywh": None
            if geometry is None
            else [float(value) for value in geometry[0, tubelet].tolist()],
        }
    return {
        "arm": capture["arm"],
        "row_label": capture["row_label"],
        "evidence_status": "recovery_checkpoint_qualitative_only"
        if capture["arm"] in qualitative
        else "arm_own_final_checkpoint_qualitative",
        "config": capture["config"],
        "checkpoint": capture["checkpoint"],
        "checkpoint_state_key": capture["checkpoint_state_key"],
        "checkpoint_epoch": capture["checkpoint_epoch"],
        "official_support": capture["official_support"],
        "source_grid_hw": capture["source_grid_hw"],
        "ignored_border_hw": capture["ignored_border_hw"],
        "k_per_tubelet": capture["k_per_tubelet"].tolist(),
        "selected_physical_indices": capture["physical_indices"].tolist(),
        "displayed_tubelets": displayed,
    }


def main() -> None:
    args = parse_args()
    checkpoints = _parse_checkpoint_bindings(args.checkpoint)
    qualitative = set(args.qualitative)
    allowed = {arm.key for arm in ARMS}
    if not qualitative <= allowed:
        raise ValueError(f"unknown qualitative arms: {sorted(qualitative - allowed)}")
    tubelets = _parse_tubelets(args.tubelets)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_config = Config.fromfile(
        str(REPO_ROOT / "configs" / "adatad" / "thumos" / ARMS[0].config_name)
    )
    _configure_test_dataset(reference_config, args)
    dataset = build_dataset(reference_config.dataset.test)
    sample_index, sample = _select_sample(dataset, args.video_name, args.video_window_index)
    sample_inputs = sample["inputs"].contiguous()
    sample_meta = dict(sample["metas"])
    if sample_inputs.dtype != torch.uint8 or tuple(sample_inputs.shape) != (1, 3, 768, 160, 160):
        raise ValueError(
            f"expected augmented uint8 [1,3,768,160,160], got {sample_inputs.dtype} {tuple(sample_inputs.shape)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA visualization requested but CUDA is unavailable")
    captures = [
        _capture_arm(arm, checkpoints[arm.key], sample_inputs, sample_meta, device)
        for arm in ARMS
    ]
    png_path, pdf_path = _render_figure(captures, sample_inputs, tubelets, qualitative, output_dir)

    payload = {
        "schema_version": "zoomtoken_on_policy_token_selection_visualization_v001",
        "evidence_boundary": (
            "Each row uses its own checkpoint and is an on-policy qualitative selection. "
            "Rows are not a common-checkpoint counterfactual comparison and provide no accuracy or cost claim."
        ),
        "sample": {
            "dataset": "THUMOS14 validation",
            "video_name": sample_meta["video_name"],
            "dataset_sample_index": sample_index,
            "video_window_index": args.video_window_index,
            "window_ordinal": int(sample_meta["window_ordinal"]),
            "window_start_frame": int(sample_meta["window_start_frame"]),
            "snippet_stride": int(sample_meta["snippet_stride"]),
            "input_shape": list(map(int, sample_inputs.shape)),
            "routing_gt_used": False,
            "routing_teacher_used": False,
            "routing_prediction_cache_used": False,
        },
        "displayed_tubelets": tubelets,
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "arms": [_jsonable_capture(capture, tubelets, qualitative) for capture in captures],
    }
    json_path = output_dir / "zoomtoken_on_policy_token_selection.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    caption = (
        "**图注。** 同一 THUMOS14 validation 窗口中，不同 ZoomToken 方法在 VideoMAE "
        "主干计算前选择的原生 10×10 空间 token。每列对应一个 tubelet，路由决定共同作用于其两帧；"
        "保留原始颜色的网格为实际选中 token，灰色网格为未选中 token，K 为该 tubelet 的实际选择数。"
        "每行使用该方法自己的 checkpoint，因此这是部署式（on-policy）定性观察，而不是固定公共 "
        "checkpoint 下的反事实因果比较。R4-SHUF15 与 Q64-GLOBAL 标记为 †，使用当前恢复点，"
        "尚未终态，仅作定性观察。该图不提供精度或成本结论。\n"
    )
    (output_dir / "caption_zh.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"status": "PASS", "png": str(png_path), "pdf": str(pdf_path), "json": str(json_path)}))


if __name__ == "__main__":
    main()
