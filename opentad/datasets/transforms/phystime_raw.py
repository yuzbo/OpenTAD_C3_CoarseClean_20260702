import numpy as np
import torch

from ..builder import PIPELINES


def _valid_prefix_count(mask):
    mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    expected = torch.arange(mask.numel(), device=mask.device) < count
    if count <= 0 or not torch.equal(mask, expected):
        raise ValueError("PhysTime raw masks must contain a non-empty valid prefix")
    return count


@PIPELINES.register_module()
class BuildPhysTimeRawFrameGeometry:
    """Build physical-time ownership cells from selected original RGB frames."""

    def __init__(self, convert_gt_to_seconds=True):
        self.convert_gt_to_seconds = bool(convert_gt_to_seconds)

    def __call__(self, results):
        if results.get("remap_gt_to_selected_axis") or results.get("gt_remapped_to_selected_axis"):
            raise ValueError("PhysTime raw geometry forbids selected-axis ground truth")
        if results.get("gt_time_unit") == "seconds" and self.convert_gt_to_seconds:
            raise ValueError("PhysTime raw ground truth was already converted to seconds")

        frame_indices = np.asarray(results["frame_inds"], dtype=np.float64).reshape(-1)
        valid_count = _valid_prefix_count(results["masks"])
        if frame_indices.size < valid_count:
            raise ValueError("PhysTime raw frame indices must cover every valid observation")
        selected_frames = frame_indices[:valid_count]

        selected_dense = np.asarray(results["selected_dense_indices"], dtype=np.float64).reshape(-1)
        if selected_dense.size != valid_count:
            raise ValueError("PhysTime raw frame and dense indices must be aligned")
        if valid_count > 1:
            if np.any(np.diff(selected_frames) <= 0) or np.any(np.diff(selected_dense) <= 0):
                raise ValueError("PhysTime raw indices must be strictly increasing")

        fps = float(results.get("avg_fps", results.get("fps", 0.0)))
        stride = float(results["snippet_stride"])
        duration = float(results["duration"])
        dense_valid_len = int(round(float(results["irregular_dense_valid_len"])))
        if fps <= 0 or stride <= 0 or duration <= 0 or dense_valid_len <= 0:
            raise ValueError(
                "PhysTime raw geometry requires positive fps, stride, duration, and dense valid length"
            )
        if selected_dense[0] < 0 or selected_dense[-1] >= dense_valid_len:
            raise ValueError("PhysTime selected dense indices exceed the dense window")

        dense_origin_frame = float(selected_frames[0] - selected_dense[0] * stride)
        expected_frames = dense_origin_frame + selected_dense * stride
        if not np.allclose(selected_frames, expected_frames, atol=1.0e-4, rtol=0.0):
            raise ValueError("PhysTime raw frame and dense indices are not aligned")

        centers = selected_frames / fps
        half_width = 0.5 * stride / fps
        supports = np.stack(
            (
                np.maximum(centers - half_width, 0.0),
                np.minimum(centers + half_width, duration),
            ),
            axis=-1,
        )
        if np.any(supports[:, 1] <= supports[:, 0]):
            raise ValueError("PhysTime raw ownership produced an empty support interval")

        domain_start = max(dense_origin_frame / fps - half_width, 0.0)
        domain_last_center = dense_origin_frame + (dense_valid_len - 1) * stride
        domain_end = min(domain_last_center / fps + half_width, duration)
        if domain_end <= domain_start:
            raise ValueError("PhysTime raw domain must be non-empty")

        if self.convert_gt_to_seconds and "gt_segments" in results:
            gt_segments = torch.as_tensor(results["gt_segments"], dtype=torch.float32)
            results["gt_segments"] = (gt_segments * stride + dense_origin_frame) / fps
            results["gt_time_unit"] = "seconds"

        results.update(
            phystime_timestamps_sec=centers.astype(np.float32).tolist(),
            phystime_support_intervals_sec=supports.astype(np.float32).tolist(),
            phystime_duration_sec=duration,
            phystime_domain_start_sec=float(domain_start),
            phystime_domain_end_sec=float(domain_end),
            phystime_support_provenance="original_raw_dense_cells",
            phystime_selected_raw_frame_indices=selected_frames.astype(np.int64).tolist(),
            phystime_sampling_strategy="random_fixed_subsample",
            phystime_sampling_uses_gt=False,
            irregular_native_axis=True,
            remap_gt_to_selected_axis=False,
            gt_remapped_to_selected_axis=False,
            prediction_time_unit="seconds",
        )
        return results
