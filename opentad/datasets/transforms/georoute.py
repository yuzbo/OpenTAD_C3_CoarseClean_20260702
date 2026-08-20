from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..builder import PIPELINES


GEOROUTE_INPUT_SCHEMA = "georoute_native_source_scout_v1"


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
class GeoRouteSourceViews:
    """Keep native source pixels and add only a low-cost global scout view.

    Unlike the historical Continuous-RoI transform, this class has no frozen
    source resolution and never creates a local crop.  The heavy path receives
    source-coordinate uint8 pixels; boundary padding to the native patch grid is
    deferred to the backbone and recorded there.
    """

    def __init__(
        self,
        scout_size: int = 96,
        output_key: str = "georoute_inputs",
        require_constant_geometry: bool = True,
    ):
        if scout_size <= 0 or scout_size % 16:
            raise ValueError("GeoRoute scout_size must be a positive multiple of patch size 16")
        self.scout_size = int(scout_size)
        self.output_key = str(output_key)
        self.require_constant_geometry = bool(require_constant_geometry)

    def __call__(self, results):
        images = results.get("imgs")
        if not isinstance(images, (list, tuple, np.ndarray)) or len(images) == 0:
            raise TypeError("GeoRouteSourceViews requires decoded non-empty results['imgs']")
        source_frames = []
        scout_frames = []
        source_geometries = []
        scout_records = []
        for frame_index, raw_image in enumerate(images):
            image = _as_uint8_rgb(raw_image, frame_index=frame_index)
            source_geometries.append((int(image.shape[0]), int(image.shape[1])))
            source_frames.append(np.ascontiguousarray(image))
            scout, record = letterbox_global_uint8(
                image,
                output_size=self.scout_size,
                frame_index=frame_index,
            )
            scout_frames.append(scout)
            scout_records.append(record)
        unique_geometry = set(source_geometries)
        if self.require_constant_geometry and len(unique_geometry) != 1:
            raise ValueError(
                "one GeoRoute video window changed source geometry across frames: "
                f"{sorted(unique_geometry)}"
            )
        if len({tuple(record["content_box_xyxy"]) for record in scout_records}) != 1:
            raise ValueError("GeoRoute scout letterbox geometry changed within one video window")
        source_tensor = _pack_ncthw(source_frames)
        scout_tensor = _pack_ncthw(scout_frames)
        if source_tensor.dtype != np.uint8 or scout_tensor.dtype != np.uint8:
            raise RuntimeError("GeoRoute source/scout tensors must remain uint8")
        source_height, source_width = source_geometries[0]
        patch_padding = ((16 - source_height % 16) % 16, (16 - source_width % 16) % 16)
        results[self.output_key] = {"source": source_tensor, "scout": scout_tensor}
        results["georoute_geometry"] = {
            "schema_version": GEOROUTE_INPUT_SCHEMA,
            "policy": "source_native_continuous_geometry_plus_residual_tokens",
            "source_hw": [source_height, source_width],
            "source_frame_count": len(source_frames),
            "native_patch_padding_bottom_right": list(patch_padding),
            "scout_hw": [self.scout_size, self.scout_size],
            "scout_content_box_xyxy": scout_records[0]["content_box_xyxy"],
            "scout_interpolation": scout_records[0]["global_interpolation"],
            "source_resized_before_native_patch_gather": False,
            "local_crop_resized": False,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(scout_size={self.scout_size})"
