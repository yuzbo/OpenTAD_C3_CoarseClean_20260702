#!/usr/bin/env python3
import glob
import os
import sys

import numpy as np


def map_axis(coords, positions, start, end, dtype):
    coords = np.asarray(coords, dtype=dtype)
    positions = np.asarray(positions, dtype=dtype)
    count = positions.size
    ranks = np.arange(count, dtype=dtype)
    xp = np.concatenate(
        (np.asarray([-0.5], dtype=dtype), ranks, np.asarray([count - 0.5], dtype=dtype))
    )
    fp = np.concatenate(
        (
            np.asarray([start], dtype=dtype),
            positions,
            np.asarray([end], dtype=dtype),
        )
    )
    flat = np.clip(coords.reshape(-1), dtype(-0.5), dtype(count - 0.5))
    upper = np.searchsorted(xp, flat, side="right")
    upper = np.clip(upper, 1, xp.size - 1)
    lower = upper - 1
    denominator = np.maximum(xp[upper] - xp[lower], dtype(1.0e-6))
    fraction = (flat - xp[lower]) / denominator
    mapped = fp[lower] + fraction * (fp[upper] - fp[lower])
    return mapped.reshape(coords.shape)


def decode(arrays, axis_key, dtype):
    base = np.asarray(arrays["base_points"], dtype=dtype)
    reg = np.asarray(arrays["reg_distances"], dtype=dtype)
    counts = arrays["native_valid_count"]
    domains = arrays["domain_sec"]
    axes = arrays[axis_key]
    proposals = np.empty(reg.shape[:2] + (2,), dtype=dtype)
    points_all = np.empty(reg.shape[:2] + (4,), dtype=dtype)
    for index in range(reg.shape[0]):
        count = int(counts[index])
        positions = np.asarray(axes[index, :count], dtype=dtype)
        start = dtype(domains[index, 0])
        end = dtype(domains[index, 1])
        center = base[:, 0]
        nominal_stride = np.maximum(base[:, 3], dtype(1.0e-6))
        mapped_center = map_axis(center, positions, start, end, dtype)
        mapped_left = map_axis(
            center - dtype(0.5) * nominal_stride,
            positions,
            start,
            end,
            dtype,
        )
        mapped_right = map_axis(
            center + dtype(0.5) * nominal_stride,
            positions,
            start,
            end,
            dtype,
        )
        stride = np.maximum(mapped_right - mapped_left, dtype(1.0e-6))
        scale = stride / nominal_stride
        points = base.copy()
        points[:, 0] = mapped_center
        points[:, 1] = base[:, 1] * scale
        points[:, 2] = base[:, 2] * scale
        points[:, 3] = stride
        decoded = np.stack(
            (
                mapped_center - reg[index, :, 0] * stride,
                mapped_center + reg[index, :, 1] * stride,
            ),
            axis=-1,
        )
        decoded[:, 0] = np.clip(decoded[:, 0], start, end)
        decoded[:, 1] = np.clip(decoded[:, 1], start, end)
        proposals[index] = decoded
        points_all[index] = points
    return proposals, points_all


root = sys.argv[1]
for capture_path in sorted(
    glob.glob(os.path.join(root, "*", "direct_work", "gpu1_id0", "decode_replay_inputs.npz"))
):
    condition = capture_path.split(os.sep)[-4]
    with np.load(capture_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    print(f"CONDITION {condition}")
    native_axis = (
        "physical_axis_sec" if condition.startswith("physical") else "uniform_axis_sec"
    )
    for axis_key in ("uniform_axis_sec", "physical_axis_sec"):
        candidate_path = os.path.join(
            root,
            condition,
            "replay",
            "modes",
            "physical_time_seconds"
            if axis_key == "physical_axis_sec"
            else "uniform_rank_seconds",
            "decoded_candidates.npz",
        )
        with np.load(candidate_path, allow_pickle=False) as candidate:
            observed = candidate["proposals"]
            valid = candidate["valid_mask"]
        for dtype in (np.float32, np.float64):
            proposals, points = decode(arrays, axis_key, dtype)
            proposal_diff = np.abs(proposals.astype(np.float64) - observed.astype(np.float64))
            point_diff = np.abs(
                points.astype(np.float64) - arrays["native_points"].astype(np.float64)
            )
            print(
                f"{axis_key} {dtype.__name__} proposal_all={proposal_diff.max():.12g} "
                f"proposal_valid={proposal_diff[valid].max():.12g} "
                f"point_all={point_diff.max():.12g} "
                f"point_valid={point_diff[valid].max():.12g} "
                f"native_axis={axis_key == native_axis}"
            )
            if axis_key == native_axis:
                point_arg = np.unravel_index(np.argmax(point_diff), point_diff.shape)
                proposal_arg = np.unravel_index(
                    np.argmax(proposal_diff), proposal_diff.shape
                )
                print(
                    f"MAX_POINT index={point_arg} expected={points[point_arg]} "
                    f"observed={arrays['native_points'][point_arg]}"
                )
                print(
                    f"MAX_PROPOSAL index={proposal_arg} expected={proposals[proposal_arg]} "
                    f"observed={observed[proposal_arg]}"
                )
