# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..builder import PIPELINES
from .native_crop import _as_uint8_rgb, _pack_ncthw, letterbox_global_uint8, stable_video_key

BAFDR_INPUT_SCHEMA = "bafdr_source_global_v1"


@PIPELINES.register_module()
class BAFDRSourceViews:
    """Prepare G96 global view and retain CPU uint8 source frames for fixed-K U128 refresh."""

    def __init__(
        self,
        global_size: int = 96,
        output_key: str = "bafdr_inputs",
        required_source_height: int = 180,
        required_source_width: int = 320,
        require_constant_geometry: bool = True,
    ):
        if global_size <= 0 or global_size % 16 != 0:
            raise ValueError("BA-FDR global_size must be positive and divisible by 16")
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
            raise TypeError("BAFDRSourceViews requires decoded non-empty results['imgs']")
        video_name = results.get("video_name", "unknown")
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

        unique_geometries = {tuple(v) for v in source_geometries}
        if self.require_constant_geometry and len(unique_geometries) != 1:
            raise ValueError(f"BA-FDR window changed source geometry across frames: {sorted(unique_geometries)}")

        expected_geometry = (self.required_source_height, self.required_source_width)
        if unique_geometries != {expected_geometry}:
            raise ValueError(f"BA-FDR is frozen to source geometry {expected_geometry}; got {sorted(unique_geometries)}")

        source_tensor = _pack_ncthw(source_frames)
        global_tensor = _pack_ncthw(global_frames)

        if source_tensor.dtype != np.uint8 or global_tensor.dtype != np.uint8:
            raise RuntimeError("BA-FDR source and global tensors must remain uint8")

        results[self.output_key] = {
            "global": global_tensor,
            "source": source_tensor,
            "sample_key": stable_video_key(video_name),
            "window_start": np.int64(window_start),
        }
        global_record = global_records[0]
        results["bafdr_geometry"] = {
            "schema_version": BAFDR_INPUT_SCHEMA,
            "policy": "bafdr_g96_carrier_plus_k16_u128_refresh",
            "video_key": int(stable_video_key(video_name)),
            "window_start_frame": window_start,
            "source_hw": list(expected_geometry),
            "source_frame_count": len(source_frames),
            "global_hw": [self.global_size, self.global_size],
            "global_content_box_xyxy": global_record["content_box_xyxy"],
            "global_interpolation": global_record["global_interpolation"],
            "k_chunks": 16,
            "total_chunks": 48,
            "local_crop_size": 128,
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
            f"required_source_height={self.required_source_height}, "
            f"required_source_width={self.required_source_width})"
        )
