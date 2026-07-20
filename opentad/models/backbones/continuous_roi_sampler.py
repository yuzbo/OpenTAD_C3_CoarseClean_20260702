from __future__ import annotations

import torch
import torch.nn.functional as F


CONTINUOUS_ROI_SAMPLER_SCHEMA = "pixel_center_bilinear_align_corners_false_v1"


def continuous_roi_grid(
    boxes: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Build an align_corners=False grid from normalized `(cx,cy,w,h)` boxes."""

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("continuous_roi_grid expects [N,4] boxes")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output geometry must be positive")
    if not bool(torch.isfinite(boxes).all().item()):
        raise ValueError("boxes must be finite")
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    if bool(((width <= 0.0) | (height <= 0.0)).any().item()):
        raise ValueError("box width and height must be positive")
    left = center_x - 0.5 * width
    right = center_x + 0.5 * width
    top = center_y - 0.5 * height
    bottom = center_y + 0.5 * height
    tolerance = 1e-6
    if bool(
        (
            (left < -tolerance)
            | (top < -tolerance)
            | (right > 1.0 + tolerance)
            | (bottom > 1.0 + tolerance)
        ).any().item()
    ):
        raise ValueError("boxes must stay inside normalized source coordinates")

    x_fraction = (
        torch.arange(output_width, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / float(output_width)
    y_fraction = (
        torch.arange(output_height, device=boxes.device, dtype=boxes.dtype) + 0.5
    ) / float(output_height)
    x = left[:, None] + width[:, None] * x_fraction[None, :]
    y = top[:, None] + height[:, None] * y_fraction[None, :]
    grid_x = (2.0 * x - 1.0)[:, None, :].expand(
        boxes.shape[0], output_height, output_width
    )
    grid_y = (2.0 * y - 1.0)[:, :, None].expand(
        boxes.shape[0], output_height, output_width
    )
    return torch.stack((grid_x, grid_y), dim=-1)


def sample_continuous_roi(
    source: torch.Tensor,
    clip_boxes: torch.Tensor,
    *,
    output_height: int = 128,
    output_width: int = 128,
    frames_per_clip: int = 16,
    clips_per_call: int | None = None,
) -> torch.Tensor:
    """Sample a source video with one continuous box per temporal clip.

    Args:
        source: `[B,1,3,T,H,W]` or `[B,3,T,H,W]`, uint8 or floating point.
        clip_boxes: `[B,K,4]` normalized `(cx,cy,w,h)` boxes.

    Returns:
        A floating-point tensor `[B,1,3,T,output_height,output_width]`.
    """

    if source.ndim == 6:
        if source.shape[1] != 1:
            raise ValueError("six-dimensional source requires N=1")
        source = source[:, 0]
    if source.ndim != 5 or source.shape[1] != 3:
        raise ValueError("source must be [B,3,T,H,W] or [B,1,3,T,H,W]")
    if clip_boxes.ndim != 3 or clip_boxes.shape[0] != source.shape[0]:
        raise ValueError("clip_boxes must be [B,K,4] and align with source batch")
    if clip_boxes.shape[-1] != 4:
        raise ValueError("clip_boxes must have four geometry channels")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    batch, _, time, _, _ = source.shape
    clips = int(clip_boxes.shape[1])
    if time != clips * int(frames_per_clip):
        raise ValueError(
            f"source T={time} does not equal clips*frames_per_clip="
            f"{clips * int(frames_per_clip)}"
        )
    if clips_per_call is None:
        clips_per_call = clips
    clips_per_call = int(clips_per_call)
    if clips_per_call <= 0:
        raise ValueError("clips_per_call must be positive")

    output_chunks = []
    for clip_start in range(0, clips, clips_per_call):
        clip_end = min(clip_start + clips_per_call, clips)
        frame_start = clip_start * frames_per_clip
        frame_end = clip_end * frames_per_clip
        source_chunk = source[:, :, frame_start:frame_end]
        chunk_clips = clip_end - clip_start
        source_chunk = (
            source_chunk.permute(0, 2, 1, 3, 4)
            .reshape(batch * chunk_clips * frames_per_clip, 3, source.shape[-2], source.shape[-1])
            .to(device=clip_boxes.device, dtype=clip_boxes.dtype, non_blocking=True)
        )
        boxes = (
            clip_boxes[:, clip_start:clip_end]
            .unsqueeze(2)
            .expand(batch, chunk_clips, frames_per_clip, 4)
            .reshape(batch * chunk_clips * frames_per_clip, 4)
        )
        grid = continuous_roi_grid(
            boxes,
            output_height=output_height,
            output_width=output_width,
        )
        sampled = F.grid_sample(
            source_chunk,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = (
            sampled.reshape(
                batch,
                chunk_clips * frames_per_clip,
                3,
                output_height,
                output_width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        output_chunks.append(sampled)
    output = torch.cat(output_chunks, dim=2)
    return output.unsqueeze(1)


@torch.no_grad()
def sample_continuous_roi_runtime(
    source: torch.Tensor,
    clip_boxes: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Inference entry point sharing the exact formal sampling implementation."""

    return sample_continuous_roi(source, clip_boxes, **kwargs)
