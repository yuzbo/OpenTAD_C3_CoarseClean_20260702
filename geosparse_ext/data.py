"""Capture decoder PTS before DecordDecode discards its source reader."""
import numpy as np
import torch
from opentad.datasets.builder import PIPELINES
from opentad.datasets.transforms.formatting import Collect
from .contracts import VideoBatch


def nearest_physical_frames(starts_s, target_s):
    """Choose source frames before defining tubelets; never compress selected rank."""
    right = np.searchsorted(starts_s, target_s).clip(0, len(starts_s) - 1)
    left = np.maximum(0, right - 1)
    return np.where(np.abs(starts_s[left] - target_s) <= np.abs(starts_s[right] - target_s), left, right)


@PIPELINES.register_module()
class CaptureGeoSparseSupport:
    def __call__(self, results):
        ids = np.asarray(results["frame_inds"], dtype=np.int64).reshape(-1)
        intervals = np.asarray(results["video_reader"].get_frame_timestamp(ids), dtype=np.float64)
        if intervals.shape != (len(ids), 2) or not np.isfinite(intervals).all():
            raise ValueError("decoder did not provide actual display intervals")
        fps = float(results["fps"] if results.get("fps", -1) > 0 else results["avg_fps"])
        step = float(results.get("snippet_stride", 1))
        if results.get("fps") == -1:
            regular = (np.arange(len(ids)) + 0.5) * float(results["duration"]) / len(ids)
        else:
            regular = (float(ids[0]) + np.arange(len(ids)) * step) / fps
        # Observe actual PTS without changing the official LoadFrames indices.
        results["geosparse_support"] = dict(source_frame_id=ids, intervals_s=intervals,
                                             regular_s=regular, pts_status="DECODED",
                                             remapped_source_frames=0,
                                             sampling_policy="official_loadframes_indices_with_observed_pts",
                                             source_path=results["filename"])
        return results


@PIPELINES.register_module()
class GeoSparseCollect(Collect):
    def __call__(self, results):
        data = super().__call__(results)
        geo = dict(results["geosparse_support"])
        x, y, w, h = results.get("crop_quadruple", [0., 0., 1., 1.])
        transform = np.array([[1 / w, 0, -x / w], [0, 1 / h, -y / h], [0, 0, 1]], dtype=np.float32)
        if results.get("flip", False):
            if results.get("flip_direction", "horizontal") != "horizontal":
                raise ValueError("GeoSparse recipe uses horizontal flips only")
            transform = np.array([[-1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.float32) @ transform
        geo["spatial_transform"] = transform
        data["metas"]["geosparse"] = geo
        return data


def video_batch(inputs, masks, metas, device):
    if inputs.ndim == 6:
        if inputs.shape[1] != 1:
            raise ValueError("GeoSparse takes one full native window per example")
        inputs = inputs[:, 0]
    frames = inputs.to(device=device, dtype=torch.float32)
    mean = frames.new_tensor([123.675, 116.28, 103.53])[None, :, None, None, None]
    std = frames.new_tensor([58.395, 57.12, 57.375])[None, :, None, None, None]
    frames = (frames - mean) / std
    stack = lambda key, dtype: torch.as_tensor(np.stack([m["geosparse"][key] for m in metas]), dtype=dtype, device=device)
    intervals = stack("intervals_s", torch.float64)
    return VideoBatch(frames, intervals[..., 0], stack("regular_s", torch.float64), masks.to(device).bool(),
                      stack("source_frame_id", torch.long), intervals,
                      [m["video_name"] for m in metas],
                      [f"{m['video_name']}:{int(m['geosparse']['source_frame_id'][0])}" for m in metas],
                      stack("spatial_transform", torch.float32))
