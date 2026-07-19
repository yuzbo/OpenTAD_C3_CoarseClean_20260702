from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..builder import PIPELINES


NATIVE_CROP_INPUT_SCHEMA = "native_crop_source_views_v1"


def _as_uint8_rgb(image: np.ndarray, *, frame_index: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"decoded frame {frame_index} must be HWC RGB; got shape={image.shape}"
        )
    if image.dtype != np.uint8:
        raise TypeError(
            "Native-Crop must run on decoded uint8 source frames before float "
            f"materialization; frame {frame_index} has dtype={image.dtype}"
        )
    return image


def center_crop_box(source_height: int, source_width: int, crop_size: int) -> tuple[int, int, int, int]:
    """Return an in-bounds source-coordinate square whenever it is feasible."""

    if min(source_height, source_width, crop_size) <= 0:
        raise ValueError("source geometry and crop_size must be positive")
    crop_height = min(int(crop_size), int(source_height))
    crop_width = min(int(crop_size), int(source_width))
    y0 = max((int(source_height) - crop_height) // 2, 0)
    x0 = max((int(source_width) - crop_width) // 2, 0)
    return x0, y0, x0 + crop_width, y0 + crop_height


def crop_source_native_uint8(
    image: np.ndarray,
    *,
    crop_size: int,
    allow_padding: bool,
    frame_index: int = 0,
) -> tuple[np.ndarray, dict]:
    """Crop source pixels 1:1; pad only when the source itself is too small."""

    image = _as_uint8_rgb(image, frame_index=frame_index)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = center_crop_box(height, width, crop_size)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    pad_bottom = max(int(crop_size) - crop.shape[0], 0)
    pad_right = max(int(crop_size) - crop.shape[1], 0)
    padding = (0, 0, pad_right, pad_bottom)
    if pad_bottom or pad_right:
        if not allow_padding:
            raise ValueError(
                "source frame is smaller than the requested source-native crop: "
                f"source_hw=({height},{width}) crop_size={crop_size}"
            )
        crop = np.pad(
            crop,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="edge",
        )
    if crop.shape != (crop_size, crop_size, 3):
        raise RuntimeError(f"unexpected local crop shape {crop.shape}")
    return np.ascontiguousarray(crop), {
        "source_box_xyxy": [x0, y0, x1, y1],
        "padding_ltrb": list(padding),
        "valid_pixel_fraction": float((x1 - x0) * (y1 - y0))
        / float(crop_size * crop_size),
        "local_interpolation": False,
    }


def letterbox_global_uint8(
    image: np.ndarray,
    *,
    output_size: int,
    frame_index: int = 0,
) -> tuple[np.ndarray, dict]:
    """Resize the whole frame into a square low-cost context view."""

    image = _as_uint8_rgb(image, frame_index=frame_index)
    height, width = image.shape[:2]
    scale = min(float(output_size) / float(width), float(output_size) / float(height))
    resized_width = max(1, min(output_size, int(round(width * scale))))
    resized_height = max(1, min(output_size, int(round(height * scale))))
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
    ).copy()
    pad_left = (output_size - resized_width) // 2
    pad_top = (output_size - resized_height) // 2
    pad_right = output_size - resized_width - pad_left
    pad_bottom = output_size - resized_height - pad_top
    view = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return np.ascontiguousarray(view), {
        "content_box_xyxy": [
            pad_left,
            pad_top,
            pad_left + resized_width,
            pad_top + resized_height,
        ],
        "source_to_global_scale": scale,
        "global_interpolation": "bilinear",
    }


def _pack_ncthw(frames: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(frames, axis=0)
    return np.ascontiguousarray(stacked.transpose(3, 0, 1, 2)[None])


@PIPELINES.register_module()
class NativeCropSourceViews:
    """Create low-cost global and source-native local views before float/H2D."""

    def __init__(
        self,
        global_size: int = 96,
        local_size: int = 128,
        output_key: str = "native_crop_inputs",
        allow_local_padding: bool = False,
        require_constant_geometry: bool = True,
    ):
        if min(global_size, local_size) <= 0:
            raise ValueError("global_size and local_size must be positive")
        if global_size % 16 or local_size % 16:
            raise ValueError("Native-Crop view sizes must be divisible by patch size 16")
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.output_key = str(output_key)
        self.allow_local_padding = bool(allow_local_padding)
        self.require_constant_geometry = bool(require_constant_geometry)

    def __call__(self, results):
        images = results.get("imgs")
        if not isinstance(images, (list, tuple, np.ndarray)) or len(images) == 0:
            raise TypeError("NativeCropSourceViews requires decoded non-empty results['imgs']")

        global_frames = []
        local_frames = []
        source_geometries = []
        local_records = []
        global_records = []
        for frame_index, raw_image in enumerate(images):
            image = _as_uint8_rgb(raw_image, frame_index=frame_index)
            source_geometries.append([int(image.shape[0]), int(image.shape[1])])
            local_view, local_record = crop_source_native_uint8(
                image,
                crop_size=self.local_size,
                allow_padding=self.allow_local_padding,
                frame_index=frame_index,
            )
            global_view, global_record = letterbox_global_uint8(
                image,
                output_size=self.global_size,
                frame_index=frame_index,
            )
            local_frames.append(local_view)
            global_frames.append(global_view)
            local_records.append(local_record)
            global_records.append(global_record)

        unique_geometries = {tuple(item) for item in source_geometries}
        if self.require_constant_geometry and len(unique_geometries) != 1:
            raise ValueError(
                "one decoded video window changed source geometry across frames: "
                f"{sorted(unique_geometries)}"
            )
        if len({tuple(item["source_box_xyxy"]) for item in local_records}) != 1:
            raise ValueError("fixed center crop unexpectedly changed within one video window")
        if len({tuple(item["padding_ltrb"]) for item in local_records}) != 1:
            raise ValueError("local padding unexpectedly changed within one video window")

        global_tensor = _pack_ncthw(global_frames)
        local_tensor = _pack_ncthw(local_frames)
        if global_tensor.dtype != np.uint8 or local_tensor.dtype != np.uint8:
            raise RuntimeError("Native-Crop views must remain uint8 until data preprocessing")

        results[self.output_key] = {
            "global": global_tensor,
            "local": local_tensor,
        }
        local_record = local_records[0]
        global_record = global_records[0]
        results["native_crop_geometry"] = {
            "schema_version": NATIVE_CROP_INPUT_SCHEMA,
            "policy": "fixed_center",
            "decision_inputs": ["source_height", "source_width"],
            "source_hw": source_geometries[0],
            "source_frame_count": len(source_geometries),
            "global_hw": [self.global_size, self.global_size],
            "global_content_box_xyxy": global_record["content_box_xyxy"],
            "global_interpolation": global_record["global_interpolation"],
            "local_hw": [self.local_size, self.local_size],
            "local_source_box_xyxy": local_record["source_box_xyxy"],
            "local_padding_ltrb": local_record["padding_ltrb"],
            "local_valid_pixel_fraction": local_record["valid_pixel_fraction"],
            "local_interpolation": False,
            "source_float_video_materialized": False,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(global_size={self.global_size}, "
            f"local_size={self.local_size}, allow_local_padding="
            f"{self.allow_local_padding})"
        )
