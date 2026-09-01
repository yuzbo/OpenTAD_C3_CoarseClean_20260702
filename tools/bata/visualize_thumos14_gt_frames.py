from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ANNOTATION_ZIP = (
    Path("analysis_outputs")
    / "thumos14_test_gt_visualization_20260709"
    / "TH14_Temporal_annotations_test.zip"
)
DEFAULT_OUTPUT_DIR = Path("analysis_outputs") / "thumos14_test_gt_visualization_20260709"

CLASS_CN = {
    "BaseballPitch": "棒球投球",
    "BasketballDunk": "篮球扣篮",
    "Billiards": "台球",
    "CleanAndJerk": "挺举",
    "CliffDiving": "悬崖跳水",
    "CricketBowling": "板球投球",
    "CricketShot": "板球击球",
    "Diving": "跳水",
    "FrisbeeCatch": "飞盘接球",
    "GolfSwing": "高尔夫挥杆",
    "HammerThrow": "链球",
    "HighJump": "跳高",
    "JavelinThrow": "标枪",
    "LongJump": "跳远",
    "PoleVault": "撑杆跳",
    "Shotput": "铅球",
    "SoccerPenalty": "足球点球",
    "TennisSwing": "网球挥拍",
    "ThrowDiscus": "铁饼",
    "VolleyballSpiking": "排球扣球",
    "Hurdles": "跨栏",
}


@dataclass(frozen=True)
class Segment:
    video_id: str
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class FrameSpec:
    time_sec: float
    state: str
    active_labels: tuple[str, ...]


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _parse_annotations(annotation_zip: Path) -> tuple[dict[str, list[Segment]], dict[str, list[Segment]]]:
    by_class: dict[str, list[Segment]] = defaultdict(list)
    by_video: dict[str, list[Segment]] = defaultdict(list)
    with zipfile.ZipFile(annotation_zip) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if not lower.endswith("_test.txt"):
                continue
            label = Path(name).stem
            if label.endswith("_test"):
                label = label[: -len("_test")]
            if label == "Ambiguous":
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            for raw in text.splitlines():
                parts = raw.split()
                if len(parts) < 3:
                    continue
                video_id = parts[0]
                try:
                    start = float(parts[1])
                    end = float(parts[2])
                except ValueError:
                    continue
                if end <= start:
                    continue
                segment = Segment(video_id=video_id, label=label, start=start, end=end)
                by_class[label].append(segment)
                by_video[video_id].append(segment)
    return dict(by_class), dict(by_video)


def _video_meta(video_path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return fps, frames, duration


def _read_frame(video_path: Path, time_sec: float, *, fps: float, duration: float) -> Image.Image:
    safe_time = max(0.0, min(float(time_sec), max(0.0, duration - 1e-3)))
    offsets = [0.0, -0.2, 0.2, -0.5, 0.5, -1.0, 1.0]
    last_error = ""
    for offset in offsets:
        attempt = max(0.0, min(safe_time + offset, max(0.0, duration - 1e-3)))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            last_error = "failed to open"
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, attempt * 1000.0)
        ok, frame = cap.read()
        if not ok and fps > 0:
            frame_idx = max(0, int(round(attempt * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        last_error = f"failed at {attempt:.3f}s"
    raise RuntimeError(f"failed to read frame: {video_path} at {safe_time:.3f}s ({last_error})")


def _active_labels_at(segments: Sequence[Segment], time_sec: float) -> tuple[str, ...]:
    labels = sorted({seg.label for seg in segments if seg.start <= time_sec <= seg.end})
    return tuple(labels)


def _merge_intervals(segments: Sequence[Segment], duration: float) -> list[tuple[float, float]]:
    intervals = sorted((max(0.0, seg.start), min(duration, seg.end)) for seg in segments if seg.end > 0)
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 1e-3:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _even_subset(items: Sequence[float], limit: int) -> list[float]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[len(items) // 2]]
    out: list[float] = []
    for idx in range(limit):
        src = round(idx * (len(items) - 1) / (limit - 1))
        out.append(float(items[int(src)]))
    return out


def _select_frame_times(
    *,
    target_segments: Sequence[Segment],
    all_video_segments: Sequence[Segment],
    duration: float,
    max_action_frames: int,
    max_background_frames: int,
) -> list[FrameSpec]:
    action_midpoints = sorted((seg.start + seg.end) * 0.5 for seg in target_segments)
    action_times = _even_subset(action_midpoints, max_action_frames)

    union = _merge_intervals(all_video_segments, duration)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in union:
        if start - cursor >= 0.35:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= 0.35:
        gaps.append((cursor, duration))

    gap_midpoints = sorted(((start + end) * 0.5, end - start) for start, end in gaps)
    # Prefer sizable background intervals, while keeping chronological diversity.
    ranked = sorted(gap_midpoints, key=lambda item: item[1], reverse=True)[: max_background_frames * 2]
    bg_times = sorted(item[0] for item in ranked)[:max_background_frames]

    specs: list[FrameSpec] = []
    for time_sec in action_times:
        active = _active_labels_at(all_video_segments, time_sec)
        specs.append(FrameSpec(time_sec=float(time_sec), state="action", active_labels=active))
    for time_sec in bg_times:
        active = _active_labels_at(all_video_segments, time_sec)
        if active:
            continue
        specs.append(FrameSpec(time_sec=float(time_sec), state="background", active_labels=()))
    return sorted(specs, key=lambda item: item.time_sec)


def _resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    resized = image.resize((int(round(src_w * scale)), int(round(src_h * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - dst_w) // 2)
    top = max(0, (resized.height - dst_h) // 2)
    return resized.crop((left, top, left + dst_w, top + dst_h))


def _label_text(labels: Sequence[str]) -> str:
    if not labels:
        return "背景"
    return " + ".join(f"{CLASS_CN.get(label, label)} / {label}" for label in labels)


def _draw_annotated_frame(
    image: Image.Image,
    spec: FrameSpec,
    *,
    thumb_size: tuple[int, int],
    border_width: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    image = _resize_cover(image, thumb_size).convert("RGB")
    draw = ImageDraw.Draw(image)
    if spec.active_labels:
        label = _label_text(spec.active_labels)
        bbox = draw.textbbox((0, 0), label, font=font)
        pad_x, pad_y = 8, 5
        box_w = bbox[2] - bbox[0] + 2 * pad_x
        box_h = bbox[3] - bbox[1] + 2 * pad_y
        x0 = image.width - box_w - 8
        y0 = 8
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=4, fill=(190, 18, 60))
        draw.text((x0 + pad_x, y0 + pad_y - 1), label, fill=(255, 255, 255), font=font)
        border_color = (220, 38, 38)
    else:
        border_color = (255, 255, 255)

    time_label = f"{spec.time_sec:.1f}s  {'ACTION' if spec.active_labels else 'BG'}"
    bbox = draw.textbbox((0, 0), time_label, font=small_font)
    box_w = bbox[2] - bbox[0] + 14
    box_h = bbox[3] - bbox[1] + 8
    draw.rectangle((7, image.height - box_h - 7, 7 + box_w, image.height - 7), fill=(0, 0, 0))
    draw.text((14, image.height - box_h - 4), time_label, fill=(255, 255, 255), font=small_font)

    framed = ImageOps.expand(image, border=border_width, fill=border_color)
    # Thin neutral outside stroke keeps white background frames visible on a white page.
    return ImageOps.expand(framed, border=1, fill=(120, 120, 120))


def _draw_overlay_on_full_frame(
    frame_bgr,
    *,
    time_sec: float,
    active_labels: Sequence[str],
    border_width: int,
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    action = bool(active_labels)
    border_color = (220, 38, 38) if action else (255, 255, 255)
    for offset in range(border_width):
        draw.rectangle((offset, offset, width - 1 - offset, height - 1 - offset), outline=border_color)

    if action:
        label = _label_text(active_labels)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        pad_x = max(8, width // 160)
        pad_y = max(5, height // 180)
        box_w = bbox[2] - bbox[0] + 2 * pad_x
        box_h = bbox[3] - bbox[1] + 2 * pad_y
        x0 = max(border_width + 4, width - box_w - border_width - 8)
        y0 = border_width + 8
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=5, fill=(190, 18, 60))
        draw.text((x0 + pad_x, y0 + pad_y - 1), label, fill=(255, 255, 255), font=label_font)

    time_label = f"{time_sec:.1f}s  {'ACTION' if action else 'BG'}"
    bbox = draw.textbbox((0, 0), time_label, font=small_font)
    pad_x = 8
    pad_y = 5
    box_w = bbox[2] - bbox[0] + 2 * pad_x
    box_h = bbox[3] - bbox[1] + 2 * pad_y
    x0 = border_width + 8
    y0 = height - box_h - border_width - 8
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    draw.text((x0 + pad_x, y0 + pad_y - 1), time_label, fill=(255, 255, 255), font=small_font)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _make_strip(
    *,
    video_path: Path,
    video_id: str,
    target_class: str,
    frame_specs: Sequence[FrameSpec],
    output_path: Path,
    fps: float,
    duration: float,
    thumb_size: tuple[int, int],
    border_width: int,
) -> dict[str, object]:
    title_font = _load_font(28)
    label_font = _load_font(18)
    small_font = _load_font(15)
    footer_font = _load_font(16)

    thumbs: list[Image.Image] = []
    kept_specs: list[FrameSpec] = []
    skipped_specs: list[dict[str, object]] = []
    for spec in frame_specs:
        try:
            frame = _read_frame(video_path, spec.time_sec, fps=fps, duration=duration)
            thumbs.append(
                _draw_annotated_frame(
                    frame,
                    spec,
                    thumb_size=thumb_size,
                    border_width=border_width,
                    font=label_font,
                    small_font=small_font,
                )
            )
            kept_specs.append(spec)
        except RuntimeError as exc:
            skipped_specs.append(
                {
                    "time_sec": spec.time_sec,
                    "state": "action" if spec.active_labels else "background",
                    "active_labels": list(spec.active_labels),
                    "error": str(exc),
                }
            )
            continue
    if not thumbs:
        raise RuntimeError(f"all requested frames failed for {video_path}")

    gap = 14
    margin = 24
    title_h = 58
    footer_h = 34
    strip_w = margin * 2 + sum(im.width for im in thumbs) + gap * max(0, len(thumbs) - 1)
    strip_h = margin + title_h + (max((im.height for im in thumbs), default=0)) + footer_h + margin
    canvas = Image.new("RGB", (strip_w, strip_h), (236, 239, 244))
    draw = ImageDraw.Draw(canvas)

    target_name = f"{CLASS_CN.get(target_class, target_class)} / {target_class}"
    title = f"THUMOS14 TAD GT visualization | {video_id} | target: {target_name}"
    draw.text((margin, margin), title, fill=(15, 23, 42), font=title_font)

    x = margin
    y = margin + title_h
    for im in thumbs:
        canvas.paste(im, (x, y))
        x += im.width + gap

    footer = "Red border = GT action frame; white border = background frame; action label is drawn at top-right."
    draw.text((margin, strip_h - margin - 20), footer, fill=(71, 85, 105), font=footer_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return {
        "video_id": video_id,
        "target_class": target_class,
        "target_name": target_name,
        "video_path": str(video_path),
        "fps": fps,
        "duration": duration,
        "frame_count": len(frame_specs),
        "frames": [
            {
                "time_sec": spec.time_sec,
                "state": "action" if spec.active_labels else "background",
                "active_labels": list(spec.active_labels),
                "active_names": [_label_text([label]) for label in spec.active_labels],
            }
            for spec in kept_specs
        ],
        "skipped_frames": skipped_specs,
        "strip_png": str(output_path),
    }


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"failed to create video writer: {path}")


def _make_full_video(
    *,
    video_path: Path,
    video_id: str,
    target_class: str,
    all_video_segments: Sequence[Segment],
    output_path: Path,
    border_width: int,
    max_seconds: float | None = None,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")

    keep_total = frame_total
    if max_seconds is not None and max_seconds > 0:
        keep_total = min(frame_total, int(round(max_seconds * fps)))
    writer = _video_writer(output_path, fps, (width, height))
    label_font = _load_font(max(18, int(round(height / 22))))
    small_font = _load_font(max(14, int(round(height / 34))))

    written = 0
    action_frames = 0
    background_frames = 0
    active_label_counts: dict[str, int] = defaultdict(int)
    while written < keep_total:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        time_sec = written / fps
        active = _active_labels_at(all_video_segments, time_sec)
        if active:
            action_frames += 1
            for label in active:
                active_label_counts[label] += 1
        else:
            background_frames += 1
        rendered = _draw_overlay_on_full_frame(
            frame,
            time_sec=time_sec,
            active_labels=active,
            border_width=border_width,
            label_font=label_font,
            small_font=small_font,
        )
        writer.write(rendered)
        written += 1

    writer.release()
    cap.release()
    duration = written / fps if fps > 0 else 0.0
    return {
        "video_id": video_id,
        "target_class": target_class,
        "target_name": f"{CLASS_CN.get(target_class, target_class)} / {target_class}",
        "video_path": str(video_path),
        "output_video": str(output_path),
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": frame_total,
        "written_frame_count": written,
        "duration": duration,
        "is_full_length": written == frame_total,
        "action_frames": action_frames,
        "background_frames": background_frames,
        "active_label_frame_counts": dict(sorted(active_label_counts.items())),
        "segments": [
            {"start": seg.start, "end": seg.end, "label": seg.label}
            for seg in all_video_segments
        ],
    }


def _make_contact_sheet(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    title_font = _load_font(24)
    label_font = _load_font(16)
    thumb_w = 560
    gap = 18
    margin = 24
    cols = 2
    loaded: list[tuple[dict[str, object], Image.Image]] = []
    for row in rows:
        img = Image.open(str(row["strip_png"])).convert("RGB")
        scale = thumb_w / img.width
        thumb_h = int(round(img.height * scale))
        loaded.append((row, img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)))
    cell_h = max(img.height for _, img in loaded) + 42
    rows_count = int(math.ceil(len(loaded) / cols))
    canvas_w = margin * 2 + cols * thumb_w + (cols - 1) * gap
    canvas_h = margin * 2 + 46 + rows_count * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "THUMOS14 GT frame visualization samples", fill=(15, 23, 42), font=title_font)
    for idx, (row, img) in enumerate(loaded):
        col = idx % cols
        rr = idx // cols
        x = margin + col * (thumb_w + gap)
        y = margin + 48 + rr * cell_h
        label = f"{row['target_name']} | {row['video_id']}"
        draw.text((x, y), label, fill=(30, 41, 59), font=label_font)
        canvas.paste(img, (x, y + 24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _parse_class_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def build_visualizations(
    *,
    annotation_zip: Path,
    video_dir: Path,
    output_dir: Path,
    classes: Sequence[str],
    videos_per_class: int,
    max_action_frames: int,
    max_background_frames: int,
    thumb_width: int,
    border_width: int,
) -> dict[str, object]:
    by_class, by_video = _parse_annotations(annotation_zip)
    strips_dir = output_dir / "strips"
    rows: list[dict[str, object]] = []
    missing_classes: list[str] = []
    available_classes = sorted(by_class)

    for target_class in classes:
        if target_class not in by_class:
            missing_classes.append(target_class)
            continue
        grouped: dict[str, list[Segment]] = defaultdict(list)
        for seg in by_class[target_class]:
            grouped[seg.video_id].append(seg)
        selected = sorted(grouped)[:videos_per_class]
        for video_id in selected:
            video_path = video_dir / f"{video_id}.mp4"
            if not video_path.exists():
                continue
            fps, _frames, duration = _video_meta(video_path)
            specs = _select_frame_times(
                target_segments=grouped[video_id],
                all_video_segments=by_video.get(video_id, []),
                duration=duration,
                max_action_frames=max_action_frames,
                max_background_frames=max_background_frames,
            )
            if not specs:
                continue
            out_name = f"{target_class}_{video_id}_gt_red_action_white_bg_strip.png"
            row = _make_strip(
                video_path=video_path,
                video_id=video_id,
                target_class=target_class,
                frame_specs=specs,
                output_path=strips_dir / out_name,
                fps=fps,
                duration=duration,
                thumb_size=(thumb_width, int(round(thumb_width * 9 / 16))),
                border_width=border_width,
            )
            row["segments"] = [
                {"start": seg.start, "end": seg.end, "label": seg.label}
                for seg in grouped[video_id]
            ]
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / "thumos14_gt_red_action_white_bg_contact_sheet.png"
    _make_contact_sheet(rows, contact_sheet)

    csv_path = output_dir / "thumos14_gt_visualization_index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_class",
                "target_name",
                "video_id",
                "duration",
                "frame_count",
                "strip_png",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    manifest = {
        "schema_version": "thumos14_gt_frame_visualization_v1",
        "annotation_zip": str(annotation_zip),
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "requested_classes": list(classes),
        "missing_requested_classes": missing_classes,
        "available_detection_classes": available_classes,
        "note": "Hurdles is not a THUMOS14 temporal detection class in the official test annotations.",
        "rows": rows,
        "contact_sheet": str(contact_sheet),
        "index_csv": str(csv_path),
    }
    manifest_path = output_dir / "thumos14_gt_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_full_video_visualizations(
    *,
    annotation_zip: Path,
    video_dir: Path,
    output_dir: Path,
    classes: Sequence[str],
    videos_per_class: int,
    border_width: int,
    max_seconds: float | None,
) -> dict[str, object]:
    by_class, by_video = _parse_annotations(annotation_zip)
    videos_dir = output_dir / "full_videos"
    rows: list[dict[str, object]] = []
    missing_classes: list[str] = []
    available_classes = sorted(by_class)

    for target_class in classes:
        if target_class not in by_class:
            missing_classes.append(target_class)
            continue
        grouped: dict[str, list[Segment]] = defaultdict(list)
        for seg in by_class[target_class]:
            grouped[seg.video_id].append(seg)
        selected = sorted(grouped)[:videos_per_class]
        for video_id in selected:
            video_path = video_dir / f"{video_id}.mp4"
            if not video_path.exists():
                continue
            suffix = "full" if max_seconds is None or max_seconds <= 0 else f"first_{max_seconds:g}s"
            out_name = f"{target_class}_{video_id}_gt_red_action_white_bg_{suffix}.mp4"
            row = _make_full_video(
                video_path=video_path,
                video_id=video_id,
                target_class=target_class,
                all_video_segments=by_video.get(video_id, []),
                output_path=videos_dir / out_name,
                border_width=border_width,
                max_seconds=max_seconds,
            )
            row["target_segments"] = [
                {"start": seg.start, "end": seg.end, "label": seg.label}
                for seg in grouped[video_id]
            ]
            rows.append(row)
            print(
                f"generated full video: {row['output_video']} "
                f"frames={row['written_frame_count']}/{row['source_frame_count']} "
                f"duration={row['duration']:.2f}s",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "thumos14_gt_full_video_index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_class",
                "target_name",
                "video_id",
                "duration",
                "fps",
                "width",
                "height",
                "source_frame_count",
                "written_frame_count",
                "is_full_length",
                "action_frames",
                "background_frames",
                "output_video",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    manifest = {
        "schema_version": "thumos14_gt_full_video_visualization_v1",
        "annotation_zip": str(annotation_zip),
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "requested_classes": list(classes),
        "missing_requested_classes": missing_classes,
        "available_detection_classes": available_classes,
        "note": "Hurdles is not a THUMOS14 temporal detection class in the official test annotations.",
        "rows": rows,
        "index_csv": str(csv_path),
    }
    manifest_path = output_dir / "thumos14_gt_full_video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize THUMOS14 TAD GT frames with red action borders and white background borders.")
    parser.add_argument("--annotation-zip", default=str(DEFAULT_ANNOTATION_ZIP))
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=("strip", "full-video", "both"), default="strip")
    parser.add_argument(
        "--classes",
        default="Hurdles Shotput HighJump LongJump PoleVault JavelinThrow HammerThrow ThrowDiscus",
        help="Space/comma separated THUMOS14 class names. Hurdles is kept here to report that it is absent.",
    )
    parser.add_argument("--videos-per-class", type=int, default=2)
    parser.add_argument("--max-action-frames", type=int, default=6)
    parser.add_argument("--max-background-frames", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--border-width", type=int, default=9)
    parser.add_argument(
        "--max-video-seconds",
        type=float,
        default=None,
        help="Debug/smoke option. Omit for original full-length videos.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifests: dict[str, object] = {}
    if args.mode in {"strip", "both"}:
        manifests["strip"] = build_visualizations(
            annotation_zip=Path(args.annotation_zip),
            video_dir=Path(args.video_dir),
            output_dir=Path(args.output_dir),
            classes=_parse_class_list(args.classes),
            videos_per_class=max(1, int(args.videos_per_class)),
            max_action_frames=max(1, int(args.max_action_frames)),
            max_background_frames=max(1, int(args.max_background_frames)),
            thumb_width=max(160, int(args.thumb_width)),
            border_width=max(2, int(args.border_width)),
        )
    if args.mode in {"full-video", "both"}:
        manifests["full_video"] = build_full_video_visualizations(
            annotation_zip=Path(args.annotation_zip),
            video_dir=Path(args.video_dir),
            output_dir=Path(args.output_dir),
            classes=_parse_class_list(args.classes),
            videos_per_class=max(1, int(args.videos_per_class)),
            border_width=max(2, int(args.border_width)),
            max_seconds=args.max_video_seconds,
        )
    primary = manifests.get("full_video") or manifests.get("strip")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": len(primary["rows"]) if isinstance(primary, dict) else None,
                "missing_requested_classes": primary["missing_requested_classes"] if isinstance(primary, dict) else [],
                "contact_sheet": primary.get("contact_sheet") if isinstance(primary, dict) else None,
                "index_csv": primary.get("index_csv") if isinstance(primary, dict) else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
