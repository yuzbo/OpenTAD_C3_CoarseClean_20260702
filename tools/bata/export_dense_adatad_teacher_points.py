from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROW_SCHEMA_VERSION = "c3_dense_adatad_teacher_points_row_v1"
SUMMARY_SCHEMA_VERSION = "c3_dense_adatad_teacher_points_export_v1"
GENERATOR_MANIFEST_SCHEMA_VERSION = "c3_detector_teacher_utility_generator_manifest_v1"
GENERATOR_MANIFEST_READY = "C3_DETECTOR_TEACHER_UTILITY_GENERATOR_MANIFEST_READY"
TEACHER_SIGNAL_SOURCE = "adatad_dense_teacher"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_gt_for_selection",
    "uses_val_gt",
    "uses_test_gt",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
    "uses_evaluator_outputs",
    "load_from_raw_predictions",
    "uses_val_or_test_gt_for_selection",
)


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    video_name: str
    window_start_frame: int
    dense_len: int
    valid_len: int
    row: Mapping[str, Any]


class _PrintLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _finite_float(value: Any, *, default: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return float(default)
    if not math.isfinite(out):
        if default is None:
            raise ValueError(f"non-finite numeric value: {value!r}")
        return float(default)
    return out


def _int_value(value: Any, *, default: int | None = None) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        if default is None:
            raise
        return int(default)


def _nested_get(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    for container_key in ("frame_signals", "paction_positive_provenance", "source_provenance", "provenance", "meta"):
        container = row.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
    return None


def _source_or_meta_float(
    sample: SourceSample,
    meta: Mapping[str, Any],
    source_keys: Sequence[str],
    meta_key: str,
    *,
    label: str,
) -> float:
    value = _nested_get(sample.row, source_keys)
    if value is None:
        value = meta.get(meta_key)
    if value is None:
        raise ValueError(f"{sample.sample_id}: missing {label} provenance")
    return _finite_float(value)


def _paction_len(row: Mapping[str, Any]) -> int | None:
    value = _nested_get(row, ("p_action",))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return None


def _parse_sample_id(sample_id: str) -> tuple[str, int]:
    if "|" not in sample_id:
        raise ValueError(f"sample_id must be video_name|window_start_frame, got {sample_id!r}")
    video_name, raw_start = sample_id.rsplit("|", 1)
    if not video_name:
        raise ValueError(f"sample_id video_name is empty: {sample_id!r}")
    return video_name, _int_value(raw_start)


def _sample_from_row(row: Mapping[str, Any], *, line_no: int, source_name: str) -> SourceSample:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{source_name}:{line_no}: sample_id is required")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"{source_name}:{line_no}: forbidden source flag {key}=true")
    video_name, start_from_id = _parse_sample_id(sample_id)
    dense_len = _int_value(row.get("dense_len") or row.get("source_dense_len") or _paction_len(row), default=0)
    valid_len = _int_value(row.get("valid_len") or row.get("source_valid_len") or dense_len, default=0)
    if dense_len <= 0:
        raise ValueError(f"{source_name}:{line_no}: dense_len is required and must be positive")
    if valid_len <= 0 or valid_len > dense_len:
        raise ValueError(f"{source_name}:{line_no}: valid_len must be in [1, dense_len]")
    return SourceSample(
        sample_id=sample_id,
        video_name=str(row.get("video_name") or video_name),
        window_start_frame=_int_value(row.get("window_start_frame"), default=start_from_id),
        dense_len=dense_len,
        valid_len=valid_len,
        row=row,
    )


def read_source_samples(path: str | Path) -> dict[str, SourceSample]:
    samples: dict[str, SourceSample] = {}
    for line_no, row in enumerate(_read_jsonl(path), start=1):
        sample = _sample_from_row(row, line_no=line_no, source_name=str(path))
        if sample.sample_id in samples:
            raise ValueError(f"{path}:{line_no}: duplicate sample_id {sample.sample_id}")
        samples[sample.sample_id] = sample
    return samples


def sample_id_from_meta(meta: Mapping[str, Any]) -> str:
    video_name = meta.get("video_name")
    if not isinstance(video_name, str) or not video_name:
        raise ValueError("dataset meta missing video_name")
    window_start = _int_value(meta.get("window_start_frame"))
    return f"{video_name}|{window_start}"


def dense_teacher_row_from_predictions(
    *,
    sample: SourceSample,
    meta: Mapping[str, Any],
    proposals: Sequence[Any],
    scores: Sequence[Any],
    topk: int | None,
    teacher_checkpoint_path: str,
    teacher_checkpoint_sha256: str,
    teacher_config_path: str,
    teacher_config_sha256: str,
) -> dict[str, Any]:
    from tools.bata.detector_teacher_utility import actionformer_predictions_to_dense_points

    dense_points = actionformer_predictions_to_dense_points(
        proposals,
        scores,
        dense_len=sample.dense_len,
        valid_len=sample.valid_len,
        topk=topk,
    )
    fps = _source_or_meta_float(
        sample,
        meta,
        ("fps", "source_fps", "video_fps"),
        "fps",
        label="fps",
    )
    snippet_stride = _source_or_meta_float(
        sample,
        meta,
        ("snippet_stride", "feature_stride", "sample_stride"),
        "snippet_stride",
        label="snippet_stride",
    )
    window_size = _int_value(_nested_get(sample.row, ("window_size", "dense_window_size")), default=sample.dense_len)
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "sample_id": sample.sample_id,
        "video_name": sample.video_name,
        "split": "training",
        "dense_len": sample.dense_len,
        "valid_len": sample.valid_len,
        "teacher_dense_points": dense_points,
        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
        "teacher_axis": "dense_frame_index",
        "fps": fps,
        "snippet_stride": snippet_stride,
        "window_start_frame": sample.window_start_frame,
        "window_size": window_size,
        "teacher_checkpoint_path": teacher_checkpoint_path,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_config_path": teacher_config_path,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_forward_sample_id": sample_id_from_meta(meta),
        "teacher_point_count": len(dense_points),
        "teacher_topk_points": topk,
        "teacher_utility_provenance": {
            "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
            "teacher_axis": "dense_frame_index",
            "fps": fps,
            "snippet_stride": snippet_stride,
            "window_start_frame": sample.window_start_frame,
            "window_size": window_size,
            "teacher_checkpoint_path": teacher_checkpoint_path,
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "teacher_config_path": teacher_config_path,
            "teacher_config_sha256": teacher_config_sha256,
            "split_scope": "train_only",
            "generator_source": "dense_detector_forward_test_proposal_score_surrogate",
        },
        "uses_evaluator_outputs": False,
        "uses_raw_prediction": False,
        "uses_prediction_cache": False,
        "load_from_raw_predictions": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_gt": False,
        "uses_oracle": False,
        "prediction_uses_gt": False,
        "training_only": True,
        "end_to_end": False,
    }
    return row


def make_train_sliding_dataset_cfg(cfg: Any, *, window_overlap_ratio: float | None = None) -> Any:
    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.ann_file = cfg.dataset.train.ann_file
    dataset_cfg.class_map = cfg.dataset.train.class_map
    dataset_cfg.data_path = cfg.dataset.train.data_path
    dataset_cfg.subset_name = cfg.dataset.train.subset_name
    dataset_cfg.test_mode = True
    if hasattr(cfg.dataset, "val") and hasattr(cfg.dataset.val, "window_size"):
        dataset_cfg.window_size = cfg.dataset.val.window_size
    if hasattr(cfg.dataset, "val") and hasattr(cfg.dataset.val, "window_overlap_ratio"):
        dataset_cfg.window_overlap_ratio = cfg.dataset.val.window_overlap_ratio
    if window_overlap_ratio is not None:
        dataset_cfg.window_overlap_ratio = float(window_overlap_ratio)
    return dataset_cfg


def load_teacher_checkpoint(model: Any, checkpoint_path: str | Path, *, device: Any, use_ema: bool = True) -> tuple[int | None, str]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    key = "state_dict_ema" if use_ema and "state_dict_ema" in checkpoint else "state_dict"
    if key not in checkpoint:
        raise ValueError(f"checkpoint missing state_dict/state_dict_ema: {checkpoint_path}")
    state_dict = checkpoint[key]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{key} must be a state dict mapping")
    normalized = dict(state_dict)
    if normalized and all(str(item).startswith("module.") for item in normalized):
        normalized = {str(item)[7:]: value for item, value in normalized.items()}
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"teacher checkpoint load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    epoch = checkpoint.get("epoch")
    return (None if epoch is None else int(epoch)), key


def _tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().tolist()
    return list(value)


def _device_batch(data: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {key: _device_batch(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_device_batch(value, device) for value in data)
    if isinstance(data, list):
        return [_device_batch(value, device) for value in data]
    return data


def export_dense_teacher_points(
    *,
    config: str | Path,
    checkpoint: str | Path,
    source_samples_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    manifest_json: str | Path | None = None,
    pretrain_path: str | Path | None = None,
    device_name: str = "cuda",
    batch_size: int | None = None,
    num_workers: int | None = None,
    topk: int | None = 2000,
    window_overlap_ratio: float | None = None,
    allow_missing: bool = False,
    use_ema: bool = True,
    amp: bool | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector

    cfg = Config.fromfile(str(config))
    if pretrain_path is not None and hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "custom"):
        cfg.model.backbone.custom.pretrain = str(pretrain_path)
    if hasattr(cfg, "inference"):
        cfg.inference.load_from_raw_predictions = False
        cfg.inference.save_raw_prediction = False
    samples = read_source_samples(source_samples_jsonl)
    dataset_cfg = make_train_sliding_dataset_cfg(cfg, window_overlap_ratio=window_overlap_ratio)
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=_PrintLogger()))
    loader_kwargs = dict(cfg.solver.test)
    if batch_size is not None:
        loader_kwargs["batch_size"] = int(batch_size)
    if num_workers is not None:
        loader_kwargs["num_workers"] = int(num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    device = torch.device(device_name if torch.cuda.is_available() or not str(device_name).startswith("cuda") else "cpu")
    model = build_detector(cfg.model).to(device)
    model.eval()
    checkpoint_sha256 = _sha256_file(checkpoint)
    config_sha256 = _sha256_file(config)
    checkpoint_epoch, checkpoint_state_key = load_teacher_checkpoint(model, checkpoint, device=device, use_ema=use_ema)
    use_amp = bool(getattr(cfg.solver, "amp", False) if amp is None else amp)

    rows_by_id: dict[str, dict[str, Any]] = {}
    seen_dataset_ids = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            metas = data.get("metas")
            if not isinstance(metas, list):
                raise ValueError("teacher export dataloader must provide list metas")
            batch_sample_ids = [sample_id_from_meta(meta) for meta in metas]
            if not any(sample_id in samples for sample_id in batch_sample_ids):
                if max_batches is not None and batch_idx + 1 >= int(max_batches):
                    break
                continue
            inputs = _device_batch(data["inputs"], device)
            masks = _device_batch(data["masks"], device)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                proposals, scores = model.forward_test(inputs=inputs, masks=masks, metas=metas, infer_cfg=cfg.inference)
            for meta, sample_id, proposal, score in zip(metas, batch_sample_ids, proposals, scores):
                seen_dataset_ids += 1
                sample = samples.get(sample_id)
                if sample is None:
                    continue
                if sample.sample_id in rows_by_id:
                    raise ValueError(f"duplicate teacher forward sample_id: {sample.sample_id}")
                rows_by_id[sample.sample_id] = dense_teacher_row_from_predictions(
                    sample=sample,
                    meta=meta,
                    proposals=_tensor_to_list(proposal),
                    scores=_tensor_to_list(score),
                    topk=topk,
                    teacher_checkpoint_path=str(checkpoint),
                    teacher_checkpoint_sha256=checkpoint_sha256,
                    teacher_config_path=str(config),
                    teacher_config_sha256=config_sha256,
                )
            if max_batches is not None and batch_idx + 1 >= int(max_batches):
                break

    missing = sorted(set(samples) - set(rows_by_id))
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"missing dense teacher forward rows for {len(missing)} source samples; first: {preview}")
    rows = [rows_by_id[sample_id] for sample_id in samples if sample_id in rows_by_id]
    _write_jsonl(output_jsonl, rows)
    output_sha256 = _sha256_file(output_jsonl)
    manifest_payload = {
        "schema_version": GENERATOR_MANIFEST_SCHEMA_VERSION,
        "decision": GENERATOR_MANIFEST_READY,
        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
        "generator_source": "dense_detector_forward_test_proposal_score_surrogate",
        "split_scope": "train_only",
        "input_split": "training",
        "input_config": str(config),
        "input_config_sha256": config_sha256,
        "teacher_checkpoint_path": str(checkpoint),
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint_epoch": checkpoint_epoch,
        "teacher_checkpoint_state_key": checkpoint_state_key,
        "source_samples_jsonl": str(source_samples_jsonl),
        "source_samples_jsonl_sha256": _sha256_file(source_samples_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": output_sha256,
        "row_count": len(rows),
        "source_row_count": len(samples),
        "missing_source_row_count": len(missing),
        "missing_source_sample_ids_preview": missing[:10],
        "dataset_window_count_seen": seen_dataset_ids,
        "topk": topk,
        "window_overlap_ratio": getattr(dataset_cfg, "window_overlap_ratio", None),
        "uses_evaluator_outputs": False,
        "uses_raw_prediction": False,
        "uses_prediction_cache": False,
        "load_from_raw_predictions": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_gt": False,
        "uses_oracle": False,
        "prediction_uses_gt": False,
        "training_only": True,
        "end_to_end": False,
    }
    if manifest_json is not None:
        _write_json(manifest_json, manifest_payload)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": "C3_DENSE_ADATAD_TEACHER_POINTS_EXPORT_READY",
        "teacher_signal_source": TEACHER_SIGNAL_SOURCE,
        "generator_source": "dense_detector_forward_test_proposal_score_surrogate",
        "split_scope": "train_only",
        "input_config": str(config),
        "input_config_sha256": config_sha256,
        "teacher_checkpoint_path": str(checkpoint),
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint_epoch": checkpoint_epoch,
        "teacher_checkpoint_state_key": checkpoint_state_key,
        "source_samples_jsonl": str(source_samples_jsonl),
        "source_samples_jsonl_sha256": _sha256_file(source_samples_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_jsonl_sha256": output_sha256,
        "manifest_json": None if manifest_json is None else str(manifest_json),
        "manifest_sha256": None if manifest_json is None else _sha256_file(manifest_json),
        "row_count": len(rows),
        "source_row_count": len(samples),
        "missing_source_row_count": len(missing),
        "uses_evaluator_outputs": False,
        "uses_raw_prediction": False,
        "uses_prediction_cache": False,
        "load_from_raw_predictions": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_gt_for_selection": False,
        "uses_gt": False,
        "uses_oracle": False,
        "prediction_uses_gt": False,
        "training_only": True,
        "end_to_end": False,
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forward dense AdaTAD teacher over train-only windows to teacher points JSONL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-samples-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--manifest-json")
    parser.add_argument("--pretrain-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument("--window-overlap-ratio", type=float)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    amp = None if args.amp == "auto" else args.amp == "on"
    summary = export_dense_teacher_points(
        config=args.config,
        checkpoint=args.checkpoint,
        source_samples_jsonl=args.source_samples_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        manifest_json=args.manifest_json,
        pretrain_path=args.pretrain_path,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=None if args.topk <= 0 else args.topk,
        window_overlap_ratio=args.window_overlap_ratio,
        allow_missing=args.allow_missing,
        use_ema=not args.no_ema,
        amp=amp,
        max_batches=args.max_batches,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
