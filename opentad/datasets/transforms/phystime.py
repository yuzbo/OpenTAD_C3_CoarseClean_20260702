import copy
import zlib

import numpy as np
import torch

from ..builder import PIPELINES


def _valid_prefix_count(mask):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    count = int(mask.sum().item())
    expected = torch.arange(mask.numel(), device=mask.device) < count
    if not torch.equal(mask.reshape(-1), expected):
        raise ValueError("PhysTime feature masks must use prefix padding")
    return count


@PIPELINES.register_module()
class SampleIrregularFeatureObservations:
    """Drop feature tokens without changing the physical support of retained tokens."""

    def __init__(
        self,
        num_observations,
        strategy="uniform",
        seed=0,
        stochastic=True,
        burst_count=3,
        gap_fraction=0.2,
    ):
        self.num_observations = int(num_observations)
        self.strategy = str(strategy)
        self.seed = int(seed)
        self.stochastic = bool(stochastic)
        self.burst_count = int(burst_count)
        self.gap_fraction = float(gap_fraction)
        if self.num_observations <= 0:
            raise ValueError("num_observations must be positive")
        if self.strategy not in {"uniform", "random", "bursty", "contiguous_gap"}:
            raise ValueError(f"unsupported PhysTime observation strategy: {self.strategy}")

    def _rng(self, results):
        if self.stochastic:
            return np.random
        sample_name = str(results.get("video_name", "unknown"))
        sample_seed = (self.seed + zlib.crc32(sample_name.encode("utf-8"))) % (2**32)
        return np.random.RandomState(sample_seed)

    @staticmethod
    def _uniform_indices(valid_count, count):
        if count >= valid_count:
            return np.arange(valid_count, dtype=np.int64)
        selected = np.rint(np.linspace(0, valid_count - 1, count)).astype(np.int64)
        selected = np.unique(selected)
        if selected.size < count:
            missing = np.setdiff1d(np.arange(valid_count, dtype=np.int64), selected, assume_unique=True)
            selected = np.sort(np.concatenate((selected, missing[: count - selected.size])))
        return selected

    def _select_indices(self, valid_count, count, rng):
        if self.strategy == "uniform":
            return self._uniform_indices(valid_count, count)
        if self.strategy == "random":
            return np.sort(rng.choice(valid_count, size=count, replace=False)).astype(np.int64)
        if self.strategy == "bursty":
            positions = np.arange(valid_count, dtype=np.float64)
            centers = rng.uniform(0, max(valid_count - 1, 1), size=max(self.burst_count, 1))
            sigma = max(valid_count / (4.0 * max(self.burst_count, 1)), 1.0)
            weights = sum(np.exp(-0.5 * ((positions - center) / sigma) ** 2) for center in centers)
            weights = weights / weights.sum()
            return np.sort(rng.choice(valid_count, size=count, replace=False, p=weights)).astype(np.int64)

        gap_length = min(max(int(round(valid_count * self.gap_fraction)), 1), max(valid_count - count, 1))
        max_start = max(valid_count - gap_length, 0)
        gap_start = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
        candidates = np.concatenate(
            (
                np.arange(0, gap_start, dtype=np.int64),
                np.arange(gap_start + gap_length, valid_count, dtype=np.int64),
            )
        )
        if candidates.size < count:
            return self._uniform_indices(valid_count, count)
        candidate_slots = self._uniform_indices(candidates.size, count)
        return candidates[candidate_slots]

    def __call__(self, results):
        features = results.get("feats")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("SampleIrregularFeatureObservations requires tensor features shaped [T, C]")
        mask = results.get("masks", torch.ones(features.shape[0], dtype=torch.bool))
        valid_count = _valid_prefix_count(mask)
        if valid_count <= 0:
            raise ValueError("PhysTime sampling requires at least one valid feature token")
        count = min(self.num_observations, valid_count)
        selected_local = self._select_indices(valid_count, count, self._rng(results))
        selected_tensor = torch.as_tensor(selected_local, dtype=torch.long, device=features.device)
        selected_features = features.index_select(0, selected_tensor)

        if count < self.num_observations:
            padding = features.new_zeros((self.num_observations - count, features.shape[1]))
            selected_features = torch.cat((selected_features, padding), dim=0)
        selected_mask = torch.arange(self.num_observations, device=features.device) < count

        window_start = int(
            results.get(
                "phystime_window_start_feature_idx",
                results.get("feature_start_idx", 0),
            )
        )
        results["phystime_window_start_feature_idx"] = window_start
        results["phystime_window_valid_feature_count"] = valid_count
        results["phystime_selected_feature_indices"] = (window_start + selected_local).astype(int).tolist()
        results["phystime_sampling_strategy"] = self.strategy
        results["phystime_sampling_uses_gt"] = False
        results["feats"] = selected_features
        results["masks"] = selected_mask.bool()
        return results


@PIPELINES.register_module()
class BuildPhysTimeFeatureGeometry:
    """Attach absolute-seconds ownership cells for pre-extracted feature tokens."""

    def __init__(self, convert_gt_to_seconds=True):
        self.convert_gt_to_seconds = bool(convert_gt_to_seconds)

    def __call__(self, results):
        provenance = results.get("phystime_input_support_provenance", "preextracted_feature_tokens")
        if provenance not in {"preextracted_feature_tokens", "original_feature_ownership_cells"}:
            raise ValueError(
                "PhysTime feature geometry requires contiguous feature-token ownership; "
                f"got {provenance!r}"
            )
        features = results.get("feats")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("BuildPhysTimeFeatureGeometry requires tensor features shaped [T, C]")
        mask = results.get("masks", torch.ones(features.shape[0], dtype=torch.bool))
        valid_count = _valid_prefix_count(mask)
        fps = float(results["fps"])
        snippet_stride = float(results["snippet_stride"])
        offset_frames = float(results.get("offset_frames", 0.0))
        duration = float(results["duration"])
        if fps <= 0 or snippet_stride <= 0 or duration <= 0:
            raise ValueError("PhysTime feature geometry requires positive fps, snippet_stride, and duration")

        window_start = int(
            results.get(
                "phystime_window_start_feature_idx",
                results.get("feature_start_idx", 0),
            )
        )
        window_valid_count = int(results.get("phystime_window_valid_feature_count", valid_count))
        selected_indices = results.get(
            "phystime_selected_feature_indices",
            list(range(window_start, window_start + valid_count)),
        )
        if len(selected_indices) != valid_count:
            raise ValueError("PhysTime selected feature indices must match the valid feature count")
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        if selected_indices.size > 1 and np.any(selected_indices[1:] <= selected_indices[:-1]):
            raise ValueError("PhysTime selected feature indices must be strictly increasing")

        centers = (selected_indices.astype(np.float64) * snippet_stride + offset_frames) / fps
        support_half_width = 0.5 * snippet_stride / fps
        supports = np.stack(
            (
                np.maximum(centers - support_half_width, 0.0),
                np.minimum(centers + support_half_width, duration),
            ),
            axis=-1,
        )
        if np.any(supports[:, 1] <= supports[:, 0]):
            raise ValueError("PhysTime feature ownership produced an empty support interval")

        domain_last_index = window_start + max(window_valid_count - 1, 0)
        domain_start = max((window_start * snippet_stride + offset_frames) / fps - support_half_width, 0.0)
        domain_end = min((domain_last_index * snippet_stride + offset_frames) / fps + support_half_width, duration)
        if domain_end <= domain_start:
            raise ValueError("PhysTime feature domain must be non-empty")

        if self.convert_gt_to_seconds and "gt_segments" in results:
            if results.get("gt_time_unit") == "seconds":
                raise ValueError("PhysTime ground truth was already converted to seconds")
            gt_segments = torch.as_tensor(results["gt_segments"], dtype=torch.float32)
            results["gt_segments"] = (
                (gt_segments + float(window_start)) * snippet_stride + offset_frames
            ) / fps
            results["gt_time_unit"] = "seconds"

        results["phystime_timestamps_sec"] = centers.astype(np.float32).tolist()
        results["phystime_support_intervals_sec"] = supports.astype(np.float32).tolist()
        results["phystime_duration_sec"] = duration
        results["phystime_domain_start_sec"] = float(domain_start)
        results["phystime_domain_end_sec"] = float(domain_end)
        results["phystime_support_provenance"] = "original_feature_ownership_cells"
        results["prediction_time_unit"] = "seconds"
        results["irregular_native_axis"] = True
        results["remap_gt_to_selected_axis"] = False
        results["gt_remapped_to_selected_axis"] = False
        return results


@PIPELINES.register_module()
class BuildPairedPhysTimeFeatureViews:
    """Build two GT-independent irregular views for discretization-consistent training."""

    _META_KEYS = (
        "video_name",
        "duration",
        "fps",
        "snippet_stride",
        "offset_frames",
        "window_start_frame",
        "gt_time_unit",
        "prediction_time_unit",
        "irregular_native_axis",
        "remap_gt_to_selected_axis",
        "gt_remapped_to_selected_axis",
        "phystime_timestamps_sec",
        "phystime_support_intervals_sec",
        "phystime_duration_sec",
        "phystime_domain_start_sec",
        "phystime_domain_end_sec",
        "phystime_support_provenance",
        "phystime_selected_feature_indices",
        "phystime_window_start_feature_idx",
        "phystime_window_valid_feature_count",
        "phystime_sampling_strategy",
        "phystime_sampling_uses_gt",
    )

    def __init__(self, first_view, second_view):
        self.first_sampler = SampleIrregularFeatureObservations(**dict(first_view))
        self.second_sampler = SampleIrregularFeatureObservations(**dict(second_view))
        self.geometry = BuildPhysTimeFeatureGeometry(convert_gt_to_seconds=True)

    def __call__(self, results):
        first = self.geometry(self.first_sampler(copy.deepcopy(results)))
        second = self.geometry(self.second_sampler(copy.deepcopy(results)))
        if not torch.allclose(first["gt_segments"], second["gt_segments"]):
            raise RuntimeError("paired PhysTime views must preserve identical absolute-seconds ground truth")

        results.update(first)
        results["paired_feats"] = second["feats"]
        results["paired_masks"] = second["masks"]
        results["paired_metas"] = {key: second[key] for key in self._META_KEYS if key in second}
        return results
