from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata import eval_zero_shot_actionness as actionness_eval


OUTPUT_SCHEMA_VERSION = actionness_eval.OUTPUT_SCHEMA_VERSION
SUMMARY_SCHEMA_VERSION = "frozen_kinetics_actionness_export_summary_v1"
READY = "FROZEN_KINETICS_ACTIONNESS_EXPORT_READY"

LIGHTWEIGHT_PROVIDERS = {
    "efficient_x3d_xs": ("pytorchvideo", "efficient_x3d_xs"),
    "efficient_x3d_s": ("pytorchvideo", "efficient_x3d_s"),
    "x3d_xs": ("pytorchvideo", "x3d_xs"),
    "x3d_s": ("pytorchvideo", "x3d_s"),
    "torchvision_r3d_18": ("torchvision", "r3d_18"),
    "torchvision_mc3_18": ("torchvision", "mc3_18"),
    "torchvision_r2plus1d_18": ("torchvision", "r2plus1d_18"),
}
HEAVY_OR_UPPER_BOUND_PROVIDERS = {
    "videomae_s",
    "videomae_b",
    "slowfast_r50",
    "xclip",
    "actionclip",
    "internvideo",
}
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def _logit(prob: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
    return math.log(clipped / (1.0 - clipped))


def _video_database(annotation: Mapping[str, Any], *, subset: str | None) -> list[tuple[str, Mapping[str, Any]]]:
    database = annotation.get("database", annotation)
    if not isinstance(database, Mapping):
        raise ValueError("annotation must contain a database object or be a mapping of video ids")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for video_id, payload in database.items():
        if not isinstance(payload, Mapping):
            continue
        if subset and str(payload.get("subset", "")).lower() != str(subset).lower():
            continue
        rows.append((str(video_id), payload))
    if not rows:
        raise ValueError(f"no videos found for subset={subset!r}")
    return rows


def _build_video_index(video_roots: Sequence[str | Path]) -> dict[str, Path]:
    exts = {".mp4", ".avi", ".mpg", ".mpeg", ".mkv"}
    out: dict[str, Path] = {}
    for root in video_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                out.setdefault(path.stem, path)
    return out


def _sample_times(duration: float, dense_window_size: int) -> list[float]:
    dense = int(dense_window_size)
    if dense <= 0:
        raise ValueError("dense_window_size must be positive")
    duration = max(0.0, float(duration))
    if duration <= 0.0:
        return [float(idx) for idx in range(dense)]
    if dense == 1:
        return [duration * 0.5]
    # Use bin centers so the first/last observations do not force clips past the file edge.
    return [duration * (idx + 0.5) / float(dense) for idx in range(dense)]


def _load_model(provider: str, *, device: Any, pretrained: bool = True) -> Any:
    import torch

    if provider in HEAVY_OR_UPPER_BOUND_PROVIDERS:
        raise ValueError(
            f"{provider} is intentionally not a lightweight pre-backbone provider; "
            "use it only as teacher/upper-bound outside this exporter"
        )
    if provider not in LIGHTWEIGHT_PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(LIGHTWEIGHT_PROVIDERS)}")
    backend, name = LIGHTWEIGHT_PROVIDERS[provider]
    if backend == "pytorchvideo":
        from pytorchvideo.models import hub

        model = getattr(hub, name)(pretrained=bool(pretrained))
    elif backend == "torchvision":
        from torchvision.models import video as tv_video

        weights_name = {
            "r3d_18": "R3D_18_Weights",
            "mc3_18": "MC3_18_Weights",
            "r2plus1d_18": "R2Plus1D_18_Weights",
        }[name]
        weights_enum = getattr(tv_video, weights_name)
        weights = weights_enum.DEFAULT if pretrained else None
        model = getattr(tv_video, name)(weights=weights)
    else:  # pragma: no cover
        raise ValueError(f"unsupported backend: {backend}")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _decode_clip_decord(
    video_path: Path,
    *,
    center_sec: float,
    clip_frames: int,
    frame_interval: int,
    crop_size: int,
) -> Any:
    import decord
    import torch
    import torch.nn.functional as F

    reader = decord.VideoReader(str(video_path), num_threads=1)
    frame_count = len(reader)
    if frame_count <= 0:
        raise ValueError(f"video has no frames: {video_path}")
    fps = float(reader.get_avg_fps() or 30.0)
    center_frame = int(round(float(center_sec) * fps))
    half = (int(clip_frames) - 1) / 2.0
    frame_ids = [
        max(0, min(frame_count - 1, int(round(center_frame + (idx - half) * int(frame_interval)))))
        for idx in range(int(clip_frames))
    ]
    frames = reader.get_batch(frame_ids).asnumpy()
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # T,C,H,W
    tensor = F.interpolate(tensor, size=(int(crop_size), int(crop_size)), mode="bilinear", align_corners=False)
    tensor = tensor.permute(1, 0, 2, 3).contiguous()  # C,T,H,W
    mean = torch.tensor(KINETICS_MEAN, dtype=tensor.dtype)[:, None, None, None]
    std = torch.tensor(KINETICS_STD, dtype=tensor.dtype)[:, None, None, None]
    return (tensor - mean) / std


def _confidence_actionness(logits: Any, *, mode: str = "entropy_mix") -> Any:
    import torch

    if logits.ndim != 2:
        raise ValueError("logits must be [B,num_classes]")
    probs = logits.softmax(dim=1)
    max_prob = probs.max(dim=1).values
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(dim=1)
    norm_entropy = entropy / math.log(float(probs.shape[1]))
    confidence = (1.0 - norm_entropy).clamp(0.0, 1.0)
    if mode == "max_prob":
        return max_prob.clamp(0.0, 1.0)
    if mode == "inverse_entropy":
        return confidence
    if mode == "entropy_mix":
        return (0.5 * max_prob + 0.5 * confidence).clamp(0.0, 1.0)
    raise ValueError("score_mode must be one of max_prob, inverse_entropy, entropy_mix")


def _source_provenance(
    *,
    provider: str,
    pretrained: bool,
    score_mode: str,
    clip_frames: int,
    frame_interval: int,
    crop_size: int,
) -> dict[str, Any]:
    return {
        "source_name": f"frozen_kinetics_{provider}_actionness",
        "source_mode": "frozen_kinetics_classifier_confidence",
        "provider": provider,
        "pretrained": bool(pretrained),
        "training_dataset": "Kinetics",
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "calibration_split": "none",
        "checkpoint_hash": f"pytorch_provider:{provider}:pretrained={bool(pretrained)}",
        "prompt_hash": None,
        "score_mode": score_mode,
        "clip_frames": int(clip_frames),
        "frame_interval": int(frame_interval),
        "crop_size": int(crop_size),
        "weights_downloaded": bool(pretrained),
        "efficiency_claim_role": "lightweight_train_free_prebackbone_candidate",
    }


def export_actionness(
    *,
    annotation_json: str | Path,
    video_roots: Sequence[str | Path],
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    provider: str = "efficient_x3d_xs",
    subset: str | None = "validation",
    dense_window_size: int = 768,
    clip_frames: int = 4,
    frame_interval: int = 12,
    crop_size: int = 160,
    batch_size: int = 16,
    device: str = "cuda",
    pretrained: bool = True,
    max_videos: int = 0,
    score_mode: str = "entropy_mix",
) -> dict[str, Any]:
    if provider in HEAVY_OR_UPPER_BOUND_PROVIDERS:
        raise ValueError(f"{provider} is too heavy for the train-free pre-backbone branch")
    import torch

    annotation = _read_json(annotation_json)
    videos = _video_database(annotation, subset=subset)
    if int(max_videos) > 0:
        videos = videos[: int(max_videos)]
    video_index = _build_video_index(video_roots)
    missing = [video_id for video_id, _payload in videos if video_id not in video_index]
    if missing:
        raise ValueError(f"missing video files for {len(missing)} videos; first missing={missing[0]}")
    device_obj = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    model = _load_model(provider, device=device_obj, pretrained=bool(pretrained))
    provenance = _source_provenance(
        provider=provider,
        pretrained=bool(pretrained),
        score_mode=score_mode,
        clip_frames=clip_frames,
        frame_interval=frame_interval,
        crop_size=crop_size,
    )
    out = Path(output_jsonl).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    start_time = time.time()
    with out.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for video_offset, (video_id, payload) in enumerate(videos, start=1):
                duration = float(payload.get("duration", 0.0))
                times = _sample_times(duration, int(dense_window_size))
                video_path = video_index[video_id]
                batch_clips: list[torch.Tensor] = []
                batch_meta: list[tuple[int, float]] = []
                video_rows = 0
                for time_index, center_sec in enumerate(times):
                    clip = _decode_clip_decord(
                        video_path,
                        center_sec=center_sec,
                        clip_frames=int(clip_frames),
                        frame_interval=int(frame_interval),
                        crop_size=int(crop_size),
                    )
                    batch_clips.append(clip)
                    batch_meta.append((time_index, center_sec))
                    if len(batch_clips) >= int(batch_size) or time_index == len(times) - 1:
                        inputs = torch.stack(batch_clips, dim=0).to(device_obj, non_blocking=True)
                        logits = model(inputs)
                        if isinstance(logits, Mapping):
                            logits = logits.get("logits", next(iter(logits.values())))
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                        logits = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
                        p_action = _confidence_actionness(logits.detach().float().cpu(), mode=score_mode)
                        max_logits = logits.detach().float().cpu().max(dim=1).values
                        for (idx, original_time), score, raw_logit in zip(
                            batch_meta, p_action.tolist(), max_logits.tolist()
                        ):
                            prob = float(score)
                            row = {
                                "schema_version": OUTPUT_SCHEMA_VERSION,
                                "video_id": video_id,
                                "window_id": f"{video_id}_{idx:04d}",
                                "time_index": int(idx),
                                "original_time": float(original_time),
                                "p_action": prob,
                                "logit": _logit(prob),
                                "raw_classifier_logit": float(raw_logit),
                                "valid": True,
                                "source_name": provenance["source_name"],
                                "source_provenance": dict(provenance),
                                "prompt_hash": None,
                                "checkpoint_hash": provenance["checkpoint_hash"],
                                "thumos_trained": False,
                                "uses_labels": False,
                                "uses_teacher": False,
                                "calibration_split": "none",
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            row_count += 1
                            video_rows += 1
                        handle.flush()
                        batch_clips.clear()
                        batch_meta.clear()
                elapsed = time.time() - start_time
                print(
                    "[FROZEN_KINETICS_ACTIONNESS] "
                    f"video={video_offset}/{len(videos)} video_id={video_id} "
                    f"rows={video_rows} total_rows={row_count} elapsed_sec={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
    elapsed = time.time() - start_time
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "annotation_json": str(annotation_json),
        "video_roots": [str(item) for item in video_roots],
        "output_jsonl": str(output_jsonl),
        "row_count": int(row_count),
        "video_count": len(videos),
        "provider": provider,
        "source_provenance": dict(provenance),
        "dense_window_size": int(dense_window_size),
        "clip_frames": int(clip_frames),
        "frame_interval": int(frame_interval),
        "crop_size": int(crop_size),
        "batch_size": int(batch_size),
        "device": str(device_obj),
        "elapsed_sec": elapsed,
        "score_generation_cost": {
            "classifier_forwards": int(math.ceil(row_count / max(1, int(batch_size)))),
            "clips_scored": int(row_count),
            "raw_video_decode": True,
            "detector_forward": False,
        },
        "source_scoring_reads_gt": False,
        "gt_labels_eval_only": True,
        "not_a_detector_mAP_result": True,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def _split_video_roots(value: str) -> list[str]:
    return [item for item in value.replace(";", ":").split(":") if item]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export lightweight frozen Kinetics classifier actionness.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--video-roots", required=True, help="Colon-separated roots containing THUMOS mp4 files.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--provider", default="efficient_x3d_xs", choices=sorted(LIGHTWEIGHT_PROVIDERS))
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--dense-window-size", type=int, default=768)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--frame-interval", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--score-mode", default="entropy_mix", choices=("max_prob", "inverse_entropy", "entropy_mix"))
    args = parser.parse_args(argv)
    summary = export_actionness(
        annotation_json=args.annotation_json,
        video_roots=_split_video_roots(args.video_roots),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        provider=args.provider,
        subset=args.subset if args.subset else None,
        dense_window_size=int(args.dense_window_size),
        clip_frames=int(args.clip_frames),
        frame_interval=int(args.frame_interval),
        crop_size=int(args.crop_size),
        batch_size=int(args.batch_size),
        device=args.device,
        pretrained=bool(args.pretrained),
        max_videos=int(args.max_videos),
        score_mode=args.score_mode,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
