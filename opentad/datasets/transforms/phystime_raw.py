import numpy as np
import torch

from ..builder import PIPELINES


def validate_raw_video_timebase(
    *,
    annotation_fps,
    decoder_avg_fps,
    total_frames,
    duration,
    fps_relative_tolerance,
    duration_relative_tolerance,
    frame_count_relative_tolerance,
):
    annotation_fps = float(annotation_fps)
    decoder_avg_fps = float(decoder_avg_fps)
    total_frames = int(total_frames)
    duration = float(duration)
    if (
        not np.isfinite(annotation_fps)
        or annotation_fps <= 0.0
        or not np.isfinite(decoder_avg_fps)
        or decoder_avg_fps <= 0.0
        or total_frames <= 0
        or not np.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError("PhysTime raw geometry requires a positive audited video timebase")
    fps_relative_error = abs(annotation_fps - decoder_avg_fps) / max(
        annotation_fps, decoder_avg_fps
    )
    if fps_relative_error > float(fps_relative_tolerance):
        raise ValueError("PhysTime annotation and decoder FPS differ beyond the audited tolerance")
    decoder_duration = total_frames / decoder_avg_fps
    duration_relative_error = abs(duration - decoder_duration) / max(duration, decoder_duration)
    if duration_relative_error > float(duration_relative_tolerance):
        raise ValueError(
            "PhysTime annotation and decoder duration differ beyond the audited tolerance"
        )
    annotation_frame_count = duration * annotation_fps
    frame_count_relative_error = abs(annotation_frame_count - total_frames) / max(
        annotation_frame_count, float(total_frames)
    )
    if frame_count_relative_error > float(frame_count_relative_tolerance):
        raise ValueError(
            "PhysTime annotation and decoder frame count differ beyond the audited tolerance"
        )
    return {
        "fps_relative_error": float(fps_relative_error),
        "duration_relative_error": float(duration_relative_error),
        "frame_count_relative_error": float(frame_count_relative_error),
        "decoder_duration": float(decoder_duration),
    }


def _valid_prefix_count(mask):
    mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    count = int(mask.sum().item())
    expected = torch.arange(mask.numel(), device=mask.device) < count
    if count <= 0 or not torch.equal(mask, expected):
        raise ValueError("PhysTime raw masks must contain a non-empty valid prefix")
    return count


def _structural_token_dependency_ranges(
    token_count,
    chunk_token_count,
    transformer_depth,
    adapter_indices,
    adapter_radius,
):
    """Return half-open token ranges that can structurally affect each output token."""
    token_count = int(token_count)
    chunk_token_count = int(chunk_token_count)
    transformer_depth = int(transformer_depth)
    adapter_radius = int(adapter_radius)
    adapter_indices = {int(index) for index in adapter_indices}
    if token_count <= 0 or chunk_token_count <= 0 or transformer_depth <= 0:
        raise ValueError("structural lineage dimensions must be positive")
    if token_count % chunk_token_count != 0:
        raise ValueError("token count must be divisible by the per-chunk token count")
    if adapter_radius < 0:
        raise ValueError("adapter radius must be non-negative")
    if any(index < 0 or index >= transformer_depth for index in adapter_indices):
        raise ValueError("adapter indices must lie inside the transformer depth")

    starts = np.arange(token_count, dtype=np.int64)
    ends = starts + 1
    for block_idx in range(transformer_depth):
        attended_starts = starts.copy()
        attended_ends = ends.copy()
        for chunk_start in range(0, token_count, chunk_token_count):
            chunk_end = chunk_start + chunk_token_count
            attended_starts[chunk_start:chunk_end] = starts[chunk_start:chunk_end].min()
            attended_ends[chunk_start:chunk_end] = ends[chunk_start:chunk_end].max()
        starts, ends = attended_starts, attended_ends

        if block_idx in adapter_indices and adapter_radius > 0:
            mixed_starts = starts.copy()
            mixed_ends = ends.copy()
            for token_idx in range(token_count):
                left = max(0, token_idx - adapter_radius)
                right = min(token_count, token_idx + adapter_radius + 1)
                mixed_starts[token_idx] = starts[left:right].min()
                mixed_ends[token_idx] = ends[left:right].max()
            starts, ends = mixed_starts, mixed_ends

    return np.stack((starts, ends), axis=-1)


@PIPELINES.register_module()
class BuildPhysTimeRawFrameGeometry:
    """Build physical-time ownership cells from selected original RGB frames."""

    def __init__(
        self,
        convert_gt_to_seconds=True,
        fps_relative_tolerance=0.0125,
        duration_relative_tolerance=0.0125,
        frame_count_relative_tolerance=0.0001,
    ):
        self.convert_gt_to_seconds = bool(convert_gt_to_seconds)
        self.fps_relative_tolerance = float(fps_relative_tolerance)
        self.duration_relative_tolerance = float(duration_relative_tolerance)
        self.frame_count_relative_tolerance = float(frame_count_relative_tolerance)
        if min(
            self.fps_relative_tolerance,
            self.duration_relative_tolerance,
            self.frame_count_relative_tolerance,
        ) < 0.0:
            raise ValueError("PhysTime timebase tolerances must be non-negative")

    def __call__(self, results):
        if results.get("remap_gt_to_selected_axis") or results.get("gt_remapped_to_selected_axis"):
            raise ValueError("PhysTime raw geometry forbids selected-axis ground truth")
        if results.get("gt_time_unit") == "seconds" and self.convert_gt_to_seconds:
            raise ValueError("PhysTime raw ground truth was already converted to seconds")
        if results.get("phystime_support_provenance") is not None:
            raise ValueError("PhysTime raw geometry must not be applied more than once")
        if "selected_raw_frame_indices" not in results:
            raise ValueError("PhysTime raw geometry requires an upstream selected-frame audit")
        if results.get("irregular_sampling_strategy") != "random_fixed_subsample":
            raise ValueError("PhysTime raw geometry requires explicit random-fixed sampling provenance")
        if results.get("irregular_sampling_scope") != "within_accepted_window":
            raise ValueError("PhysTime raw geometry requires explicit within-window sampling provenance")
        if results.get("irregular_subsample_uses_gt") is not False:
            raise ValueError("PhysTime raw geometry requires explicitly GT-independent within-window subsampling")
        if "irregular_window_crop_uses_gt" not in results:
            raise ValueError("PhysTime raw geometry requires an explicit window-crop provenance flag")

        frame_indices = np.asarray(results["frame_inds"], dtype=np.float64).reshape(-1)
        valid_count = _valid_prefix_count(results["masks"])
        if frame_indices.size < valid_count:
            raise ValueError("PhysTime raw frame indices must cover every valid observation")
        selected_frames = frame_indices[:valid_count]
        audited_frames = np.asarray(results["selected_raw_frame_indices"], dtype=np.float64).reshape(-1)
        if audited_frames.size != valid_count or not np.array_equal(audited_frames, selected_frames):
            raise ValueError("PhysTime raw-frame audit metadata does not match decoded frame indices")

        selected_dense = np.asarray(results["selected_dense_indices"], dtype=np.float64).reshape(-1)
        if selected_dense.size != valid_count:
            raise ValueError("PhysTime raw frame and dense indices must be aligned")
        if valid_count > 1:
            if np.any(np.diff(selected_frames) <= 0) or np.any(np.diff(selected_dense) <= 0):
                raise ValueError("PhysTime raw indices must be strictly increasing")

        if "fps" not in results:
            raise ValueError("PhysTime raw geometry requires the dataset annotation FPS")
        fps = float(results["fps"])
        decoder_avg_fps = float(results.get("avg_fps", float("nan")))
        total_frames = int(results.get("total_frames", 0))
        stride = float(results["snippet_stride"])
        duration = float(results["duration"])
        dense_valid_len = int(round(float(results["irregular_dense_valid_len"])))
        if fps <= 0 or not np.isfinite(fps) or stride <= 0 or duration <= 0 or dense_valid_len <= 0:
            raise ValueError(
                "PhysTime raw geometry requires positive fps, stride, duration, and dense valid length"
            )
        timebase = validate_raw_video_timebase(
            annotation_fps=fps,
            decoder_avg_fps=decoder_avg_fps,
            total_frames=total_frames,
            duration=duration,
            fps_relative_tolerance=self.fps_relative_tolerance,
            duration_relative_tolerance=self.duration_relative_tolerance,
            frame_count_relative_tolerance=self.frame_count_relative_tolerance,
        )
        fps_relative_error = timebase["fps_relative_error"]
        duration_relative_error = timebase["duration_relative_error"]
        frame_count_relative_error = timebase["frame_count_relative_error"]
        if selected_frames[0] < 0 or selected_frames[-1] >= total_frames:
            raise ValueError("PhysTime selected frames exceed the decoded video frame range")
        if selected_dense[0] < 0 or selected_dense[-1] >= dense_valid_len:
            raise ValueError("PhysTime selected dense indices exceed the dense window")

        dense_origin_frame = float(selected_frames[0] - selected_dense[0] * stride)
        expected_frames = dense_origin_frame + selected_dense * stride
        if not np.allclose(selected_frames, expected_frames, atol=1.0e-4, rtol=0.0):
            raise ValueError("PhysTime raw frame and dense indices are not aligned")

        centers = selected_frames / fps
        half_width = 0.5 * stride / fps
        domain_start = max(dense_origin_frame / fps, 0.0)
        domain_end = min((dense_origin_frame + dense_valid_len * stride) / fps, duration)
        if domain_end <= domain_start:
            raise ValueError("PhysTime raw domain must be non-empty")
        supports = np.stack(
            (
                np.maximum(centers - half_width, domain_start),
                np.minimum(centers + half_width, domain_end),
            ),
            axis=-1,
        )
        if np.any(supports[:, 1] <= supports[:, 0]):
            raise ValueError("PhysTime raw ownership produced an empty support interval")

        if self.convert_gt_to_seconds and "gt_segments" in results:
            gt_segments = torch.as_tensor(results["gt_segments"], dtype=torch.float32)
            gt_seconds = (gt_segments * stride + dense_origin_frame) / fps
            eps = max(1.0e-5, stride / fps * 1.0e-4)
            if not torch.isfinite(gt_seconds).all():
                raise ValueError("PhysTime ground truth seconds must be finite")
            if bool((gt_seconds[:, 0] < domain_start - eps).any()) or bool(
                (gt_seconds[:, 1] > domain_end + eps).any()
            ):
                raise ValueError("PhysTime ground truth lies outside the end-exclusive window domain")
            results["gt_segments"] = gt_seconds
            results["gt_time_unit"] = "seconds"

        results.update(
            phystime_timestamps_sec=centers.astype(np.float32).tolist(),
            phystime_support_intervals_sec=supports.astype(np.float32).tolist(),
            phystime_duration_sec=duration,
            phystime_canonical_fps=fps,
            phystime_canonical_fps_source="dataset_annotation_frame_over_duration",
            phystime_decoder_avg_fps=decoder_avg_fps,
            phystime_decoder_total_frames=total_frames,
            phystime_fps_relative_error=float(fps_relative_error),
            phystime_duration_relative_error=float(duration_relative_error),
            phystime_frame_count_relative_error=float(frame_count_relative_error),
            phystime_timebase_tolerance_policy=dict(
                fps_relative_tolerance=self.fps_relative_tolerance,
                duration_relative_tolerance=self.duration_relative_tolerance,
                frame_count_relative_tolerance=self.frame_count_relative_tolerance,
            ),
            phystime_domain_start_sec=float(domain_start),
            phystime_domain_end_sec=float(domain_end),
            phystime_support_provenance="original_raw_dense_cells",
            phystime_selected_raw_frame_indices=selected_frames.astype(np.int64).tolist(),
            phystime_sampling_strategy=str(results["irregular_sampling_strategy"]),
            phystime_sampling_uses_gt=False,
            phystime_sampling_scope=str(results["irregular_sampling_scope"]),
            phystime_window_crop_uses_gt=bool(results["irregular_window_crop_uses_gt"]),
            phystime_subsample_uses_gt=False,
            irregular_native_axis=True,
            remap_gt_to_selected_axis=False,
            gt_remapped_to_selected_axis=False,
            prediction_time_unit="seconds",
        )
        return results


@PIPELINES.register_module()
class BuildPhysTimeNativeTubeletGeometry:
    """Audit VideoMAE patch atoms and construct one explicit seconds axis.

    The two-frame groups are exact inputs to patch embedding, not a claim about
    the final token support after chunk-wide ViT/adapter mixing.
    """

    _VALID_COORDINATE_MODES = {"uniform_rank_seconds", "physical_time_seconds"}

    def __init__(
        self,
        tubelet_size=2,
        chunk_size=16,
        transformer_depth=12,
        adapter_indices=None,
        adapter_kernel_size=3,
        adapter_dilation=1,
        coordinate_mode="physical_time_seconds",
    ):
        self.tubelet_size = int(tubelet_size)
        self.chunk_size = int(chunk_size)
        self.transformer_depth = int(transformer_depth)
        if adapter_indices is None:
            adapter_indices = list(range(self.transformer_depth))
        self.adapter_indices = tuple(int(index) for index in adapter_indices)
        self.adapter_kernel_size = int(adapter_kernel_size)
        self.adapter_dilation = int(adapter_dilation)
        self.coordinate_mode = str(coordinate_mode)
        if self.tubelet_size <= 0:
            raise ValueError("tubelet_size must be positive")
        if self.chunk_size <= 0 or self.chunk_size % self.tubelet_size != 0:
            raise ValueError("chunk_size must be positive and divisible by tubelet_size")
        if self.transformer_depth <= 0:
            raise ValueError("transformer_depth must be positive")
        if len(set(self.adapter_indices)) != len(self.adapter_indices):
            raise ValueError("adapter_indices must be unique")
        if any(index < 0 or index >= self.transformer_depth for index in self.adapter_indices):
            raise ValueError("adapter_indices must lie inside transformer_depth")
        if self.adapter_kernel_size <= 0 or self.adapter_kernel_size % 2 == 0:
            raise ValueError("adapter_kernel_size must be a positive odd integer")
        if self.adapter_dilation <= 0:
            raise ValueError("adapter_dilation must be positive")
        if self.coordinate_mode not in self._VALID_COORDINATE_MODES:
            raise ValueError(f"unsupported native tubelet coordinate mode: {self.coordinate_mode}")

    def __call__(self, results):
        if results.get("phystime_support_provenance") != "original_raw_dense_cells":
            raise ValueError("native tubelet geometry requires audited raw support provenance")
        if results.get("phystime_subsample_uses_gt") is not False:
            raise ValueError("native tubelet geometry requires explicitly GT-independent within-window subsampling")

        raw_mask = torch.as_tensor(results["masks"], dtype=torch.bool).reshape(-1)
        raw_valid_count = _valid_prefix_count(raw_mask)
        raw_observation_count = int(raw_mask.numel())
        if raw_observation_count % self.tubelet_size != 0:
            raise ValueError("padded raw observation count must be divisible by tubelet_size")
        if raw_observation_count % self.chunk_size != 0:
            raise ValueError("padded raw observation count must be divisible by chunk_size")

        dense_positions = np.asarray(results["selected_dense_indices"], dtype=np.float64).reshape(-1)
        timestamps = np.asarray(results["phystime_timestamps_sec"], dtype=np.float64).reshape(-1)
        supports = np.asarray(results["phystime_support_intervals_sec"], dtype=np.float64).reshape(-1, 2)
        raw_frames = np.asarray(results["phystime_selected_raw_frame_indices"], dtype=np.float64).reshape(-1)
        padded_raw_frames = np.asarray(results["frame_inds"], dtype=np.float64).reshape(-1)
        for name, values in (
            ("selected_dense_indices", dense_positions),
            ("phystime_timestamps_sec", timestamps),
            ("phystime_support_intervals_sec", supports),
            ("phystime_selected_raw_frame_indices", raw_frames),
        ):
            if values.shape[0] != raw_valid_count:
                raise ValueError(f"{name} count must equal the valid raw-frame prefix")
        if padded_raw_frames.shape[0] != raw_observation_count:
            raise ValueError("frame_inds must expose every raw input slot consumed by the backbone")
        if not np.array_equal(padded_raw_frames[:raw_valid_count], raw_frames):
            raise ValueError("valid raw-frame input slots disagree with audited selected frames")
        if raw_valid_count < raw_observation_count and not np.all(
            padded_raw_frames[raw_valid_count:] == raw_frames[-1]
        ):
            raise ValueError("invalid raw-frame input slots must be explicit edge-padding repeats")
        if raw_valid_count > 1:
            if np.any(np.diff(dense_positions) <= 0) or np.any(np.diff(timestamps) <= 0):
                raise ValueError("native tubelet atoms must be strictly ordered")

        native_token_count = raw_observation_count // self.tubelet_size
        native_valid_count = (raw_valid_count + self.tubelet_size - 1) // self.tubelet_size
        padding_repeat_count = raw_observation_count - raw_valid_count
        chunk_token_count = self.chunk_size // self.tubelet_size
        adapter_radius = (self.adapter_kernel_size // 2) * self.adapter_dilation
        dependency_token_ranges = _structural_token_dependency_ranges(
            native_token_count,
            chunk_token_count,
            self.transformer_depth,
            self.adapter_indices,
            adapter_radius,
        )
        dependency_raw_ranges = dependency_token_ranges * self.tubelet_size
        padding_dependency = dependency_raw_ranges[:, 1] > raw_valid_count
        final_feature_raw_slot_upper_bound = int(
            (dependency_raw_ranges[:, 1] - dependency_raw_ranges[:, 0]).max()
        )
        padded_dense_positions = np.pad(
            dense_positions,
            (0, padding_repeat_count),
            mode="edge",
        )
        padded_timestamps = np.pad(timestamps, (0, padding_repeat_count), mode="edge")
        padded_supports = np.pad(supports, ((0, padding_repeat_count), (0, 0)), mode="edge")
        token_dense_positions = np.zeros(native_valid_count, dtype=np.float64)
        token_timestamps = np.zeros(native_valid_count, dtype=np.float64)
        patch_support_atoms = np.zeros((native_token_count, self.tubelet_size, 2), dtype=np.float64)
        patch_dense_atoms = np.zeros((native_token_count, self.tubelet_size), dtype=np.float64)
        patch_raw_frame_atoms = np.zeros((native_token_count, self.tubelet_size), dtype=np.float64)
        semantic_atom_mask = np.zeros((native_token_count, self.tubelet_size), dtype=np.bool_)
        compute_atom_mask = np.ones((native_token_count, self.tubelet_size), dtype=np.bool_)
        atom_kinds = []
        envelopes = np.zeros((native_token_count, 2), dtype=np.float64)
        envelope_inflation = np.zeros(native_token_count, dtype=np.float64)
        atom_gap = np.zeros(native_token_count, dtype=np.float64)

        for token_idx in range(native_token_count):
            start = token_idx * self.tubelet_size
            end = start + self.tubelet_size
            observed_end = min(end, raw_valid_count)
            observed_count = max(observed_end - start, 0)
            semantic_atom_mask[token_idx, :observed_count] = True
            patch_support_atoms[token_idx] = padded_supports[start:end]
            patch_dense_atoms[token_idx] = padded_dense_positions[start:end]
            patch_raw_frame_atoms[token_idx] = padded_raw_frames[start:end]
            kinds = [
                "observed" if slot_idx < raw_valid_count else "padding_repeat"
                for slot_idx in range(start, end)
            ]
            atom_kinds.append(kinds)
            if observed_count > 0:
                valid_supports = supports[start:observed_end]
                token_dense_positions[token_idx] = dense_positions[start:observed_end].mean()
                token_timestamps[token_idx] = timestamps[start:observed_end].mean()
                envelopes[token_idx] = [valid_supports[:, 0].min(), valid_supports[:, 1].max()]
                atom_mass = np.maximum(valid_supports[:, 1] - valid_supports[:, 0], 0.0).sum()
                envelope_mass = envelopes[token_idx, 1] - envelopes[token_idx, 0]
                envelope_inflation[token_idx] = max(float(envelope_mass - atom_mass), 0.0)
                if observed_count > 1:
                    atom_gap[token_idx] = np.maximum(
                        valid_supports[1:, 0] - valid_supports[:-1, 1], 0.0
                    ).sum()
            else:
                repeated_supports = padded_supports[start:end]
                envelopes[token_idx] = [
                    repeated_supports[:, 0].min(),
                    repeated_supports[:, 1].max(),
                ]

        domain_start = float(results["phystime_domain_start_sec"])
        domain_end = float(results["phystime_domain_end_sec"])
        uniform_step = (domain_end - domain_start) / float(native_valid_count)
        uniform_positions = domain_start + (
            np.arange(native_valid_count, dtype=np.float64) + 0.5
        ) * uniform_step
        if self.coordinate_mode == "physical_time_seconds":
            axis_positions = token_timestamps
        else:
            axis_positions = uniform_positions
        results.update(
            phystime_raw_observation_count=raw_observation_count,
            phystime_raw_valid_count=raw_valid_count,
            phystime_raw_selected_dense_indices=dense_positions.astype(np.float32).tolist(),
            phystime_native_token_count=native_token_count,
            phystime_native_valid_count=native_valid_count,
            phystime_native_tubelet_size=self.tubelet_size,
            phystime_native_chunk_size=self.chunk_size,
            phystime_native_attention_chunk_token_count=chunk_token_count,
            phystime_native_transformer_depth=self.transformer_depth,
            phystime_native_adapter_indices=list(self.adapter_indices),
            phystime_native_adapter_kernel_size=self.adapter_kernel_size,
            phystime_native_adapter_dilation=self.adapter_dilation,
            phystime_native_coordinate_mode=self.coordinate_mode,
            phystime_native_token_dense_positions=token_dense_positions.astype(np.float32).tolist(),
            phystime_native_token_timestamps_sec=token_timestamps.astype(np.float32).tolist(),
            phystime_patch_embed_support_atoms_sec=patch_support_atoms.astype(np.float32).tolist(),
            phystime_patch_embed_dense_atoms=patch_dense_atoms.astype(np.float32).tolist(),
            phystime_patch_embed_raw_frame_atoms=patch_raw_frame_atoms.astype(np.int64).tolist(),
            phystime_patch_embed_semantic_atom_mask=semantic_atom_mask.tolist(),
            phystime_patch_embed_compute_atom_mask=compute_atom_mask.tolist(),
            phystime_patch_embed_atom_kind=atom_kinds,
            phystime_patch_embed_support_envelopes_sec=envelopes.astype(np.float32).tolist(),
            phystime_patch_embed_envelope_inflation_sec=envelope_inflation.astype(np.float32).tolist(),
            phystime_patch_embed_atom_gap_sec=atom_gap.astype(np.float32).tolist(),
            phystime_patch_embed_padding_repeat_count=padding_repeat_count,
            phystime_patch_embed_lineage_provenance="raw_atoms_exact_at_patch_embed_input",
            phystime_native_final_feature_lineage=(
                "structural_upper_bound_chunk_attention_global_temporal_adapter"
            ),
            phystime_native_final_feature_token_ranges_exclusive=dependency_token_ranges.tolist(),
            phystime_native_final_feature_raw_slot_ranges_exclusive=dependency_raw_ranges.tolist(),
            phystime_native_final_feature_raw_slot_upper_bound=final_feature_raw_slot_upper_bound,
            phystime_native_final_feature_support_is_exact=False,
            phystime_native_final_feature_valid_tokens_may_depend_on_padding_repeats=bool(
                padding_dependency[:native_valid_count].any()
            ),
            phystime_native_final_feature_padding_dependency_upper_bound_mask=padding_dependency.tolist(),
            phystime_uniform_rank_timestamps_sec=uniform_positions.astype(np.float32).tolist(),
            phystime_g1a_axis_positions_sec=axis_positions.astype(np.float32).tolist(),
            phystime_g1a_axis_start_sec=domain_start,
            phystime_g1a_axis_end_sec=domain_end,
            irregular_native_axis=True,
            remap_gt_to_selected_axis=False,
            gt_remapped_to_selected_axis=False,
            prediction_time_unit="seconds",
        )
        if "gt_segments" in results:
            if results.get("gt_time_unit") != "seconds":
                raise ValueError("G1a requires GT to remain in canonical seconds")
        return results
