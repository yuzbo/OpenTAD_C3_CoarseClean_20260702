from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..builder import PIPELINES


NATIVE_CROP_INPUT_SCHEMA = "native_crop_source_views_v1"
CONTINUOUS_ROI_INPUT_SCHEMA = "continuous_roi_source_global_v2_1"
CONTINUOUS_ROI_DENSE_INPUT_SCHEMA = "continuous_roi_full_frame_letterbox_v2_1"
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


def stable_video_key(video_name: str) -> np.int64:
    if not isinstance(video_name, str) or not video_name:
        raise ValueError("continuous ROI inputs require a non-empty video_name")
    digest = hashlib.sha256(video_name.encode("utf-8")).digest()
    return np.int64(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))


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


@PIPELINES.register_module()
class ContinuousRoiSourceViews:
    """Keep source RGB and add only the registered low-cost global view."""

    def __init__(
        self,
        global_size: int = 96,
        output_key: str = "continuous_roi_inputs",
        required_source_height: int = 180,
        required_source_width: int = 320,
        require_constant_geometry: bool = True,
    ):
        if global_size <= 0 or global_size % 16:
            raise ValueError("continuous ROI global_size must be positive and divisible by 16")
        if min(required_source_height, required_source_width) <= 0:
            raise ValueError("required source geometry must be positive")
        self.global_size = int(global_size)
        self.output_key = str(output_key)
        self.required_source_height = int(required_source_height)
        self.required_source_width = int(required_source_width)
        self.require_constant_geometry = bool(require_constant_geometry)

    def __call__(self, results):
        images = results.get("imgs")
        if not isinstance(images, (list, tuple, np.ndarray)) or len(images) == 0:
            raise TypeError(
                "ContinuousRoiSourceViews requires decoded non-empty results['imgs']"
            )
        video_name = results.get("video_name")
        window_start = int(results.get("window_start_frame", 0))
        source_frames = []
        global_frames = []
        source_geometries = []
        global_records = []
        for frame_index, raw_image in enumerate(images):
            image = _as_uint8_rgb(raw_image, frame_index=frame_index)
            height, width = image.shape[:2]
            source_geometries.append([int(height), int(width)])
            source_frames.append(np.ascontiguousarray(image))
            global_view, global_record = letterbox_global_uint8(
                image,
                output_size=self.global_size,
                frame_index=frame_index,
            )
            global_frames.append(global_view)
            global_records.append(global_record)

        unique_geometries = {tuple(value) for value in source_geometries}
        if self.require_constant_geometry and len(unique_geometries) != 1:
            raise ValueError(
                "one continuous ROI window changed source geometry across frames: "
                f"{sorted(unique_geometries)}"
            )
        expected_geometry = (
            self.required_source_height,
            self.required_source_width,
        )
        if unique_geometries != {expected_geometry}:
            raise ValueError(
                "continuous ROI v2.1 is frozen to source geometry "
                f"{expected_geometry}; got {sorted(unique_geometries)}"
            )
        if len(
            {
                tuple(record["content_box_xyxy"])
                for record in global_records
            }
        ) != 1:
            raise ValueError("global letterbox geometry changed within one video window")

        source_tensor = _pack_ncthw(source_frames)
        global_tensor = _pack_ncthw(global_frames)
        if source_tensor.dtype != np.uint8 or global_tensor.dtype != np.uint8:
            raise RuntimeError("continuous ROI source/global tensors must remain uint8")
        results[self.output_key] = {
            "global": global_tensor,
            "source": source_tensor,
            "sample_key": stable_video_key(video_name),
            "window_start": np.int64(window_start),
        }
        global_record = global_records[0]
        results["continuous_roi_geometry"] = {
            "schema_version": CONTINUOUS_ROI_INPUT_SCHEMA,
            "policy": "none_pre_policy_source",
            "decision_inputs": [],
            "video_key": int(stable_video_key(video_name)),
            "window_start_frame": window_start,
            "source_hw": list(expected_geometry),
            "source_frame_count": len(source_frames),
            "global_hw": [self.global_size, self.global_size],
            "global_content_box_xyxy": global_record["content_box_xyxy"],
            "global_interpolation": global_record["global_interpolation"],
            "source_float_video_materialized": False,
            "source_resized_before_crop": False,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(global_size={self.global_size}, "
            f"required_source_height={self.required_source_height}, "
            f"required_source_width={self.required_source_width})"
        )


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


@PIPELINES.register_module()
class FullFrameLetterboxView:
    """Create a registered dense comparator without spatial cropping."""

    def __init__(
        self,
        output_size: int,
        output_key: str = "imgs",
        required_source_height: int = 180,
        required_source_width: int = 320,
    ):
        if output_size <= 0 or output_size % 16:
            raise ValueError("letterbox output_size must be positive and divisible by 16")
        self.output_size = int(output_size)
        self.output_key = str(output_key)
        self.required_source_height = int(required_source_height)
        self.required_source_width = int(required_source_width)

    def __call__(self, results):
        images = results.get("imgs")
        if not isinstance(images, (list, tuple, np.ndarray)) or len(images) == 0:
            raise TypeError(
                "FullFrameLetterboxView requires decoded non-empty results['imgs']"
            )
        views = []
        records = []
        source_geometries = []
        for frame_index, raw_image in enumerate(images):
            image = _as_uint8_rgb(raw_image, frame_index=frame_index)
            source_geometries.append(tuple(map(int, image.shape[:2])))
            view, record = letterbox_global_uint8(
                image,
                output_size=self.output_size,
                frame_index=frame_index,
            )
            views.append(view)
            records.append(record)
        expected = (self.required_source_height, self.required_source_width)
        if set(source_geometries) != {expected}:
            raise ValueError(
                f"full-frame comparator expects source geometry {expected}; "
                f"got {sorted(set(source_geometries))}"
            )
        if len({tuple(record["content_box_xyxy"]) for record in records}) != 1:
            raise ValueError("letterbox geometry changed inside one video window")
        tensor = _pack_ncthw(views)
        if tensor.dtype != np.uint8:
            raise RuntimeError("full-frame letterbox must remain uint8")
        results[self.output_key] = tensor
        results["continuous_roi_geometry"] = {
            "schema_version": CONTINUOUS_ROI_DENSE_INPUT_SCHEMA,
            "policy": "full_frame_letterbox",
            "source_hw": list(expected),
            "source_frame_count": len(views),
            "output_hw": [self.output_size, self.output_size],
            "content_box_xyxy": records[0]["content_box_xyxy"],
            "source_resized_before_crop": False,
            "crop_applied": False,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(output_size={self.output_size}, "
            f"required_source_height={self.required_source_height}, "
            f"required_source_width={self.required_source_width})"
        )
