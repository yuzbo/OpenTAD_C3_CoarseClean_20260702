from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BUDGETS = (256, 384, 512)
BASELINE_BUDGET = 384
DETECTOR_LENGTH = 384
PACKET_SIZE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return count


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty stage artifact: {path}")
    return rows


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _resolved_file(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _resolved_dir(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is missing: {path}")
    return path


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sample_id(meta: Mapping[str, Any]) -> str:
    video_id = str(meta.get("video_name") or meta.get("video_id") or "")
    if not video_id:
        raise ValueError("window metadata is missing video_name")
    start = int(round(float(meta.get("window_start_frame", 0))))
    return f"{video_id}|{start}"


def _stage_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "selection": output_dir / "selection_k384.jsonl.gz",
        "k256": output_dir / "counterfactual_k256.jsonl.gz",
        "k512": output_dir / "counterfactual_k512.jsonl.gz",
        "selection_receipt": output_dir / "selection_k384_receipt.json",
        "k256_receipt": output_dir / "counterfactual_k256_receipt.json",
        "k512_receipt": output_dir / "counterfactual_k512_receipt.json",
        "result": output_dir / "probe_result.json",
        "pre_run": output_dir / "pre_run_receipt.json",
        "split_dir": output_dir / "controller_split",
    }


def _load_config_and_paths(args: argparse.Namespace):
    from mmengine.config import Config

    repo_root = Path(__file__).resolve().parents[2]
    config_path = _resolved_file(args.config, name="probe config")
    cfg = Config.fromfile(str(config_path))
    checkpoint_value = args.checkpoint or cfg.duca_marginal_probe.h65_checkpoint_path
    checkpoint = _resolved_file(checkpoint_value, name="H65 checkpoint")
    expected_sha = str(
        args.expected_checkpoint_sha256
        or cfg.duca_marginal_probe.h65_checkpoint_sha256
    )
    actual_sha = _sha256(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            f"H65 checkpoint SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    annotation = _resolved_file(
        args.annotation or cfg.dataset.train.ann_file,
        name="THUMOS14 annotation",
    )
    class_map = _resolved_file(
        args.class_map or cfg.dataset.train.class_map,
        name="THUMOS14 class map",
    )
    train_data = _resolved_dir(
        args.train_data or cfg.dataset.train.data_path,
        name="THUMOS14 training videos",
    )
    pretrain = _resolved_file(
        args.pretrain or cfg.model.backbone.custom.pretrain,
        name="VideoMAE-S pretrain",
    )
    return (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        actual_sha,
        annotation,
        class_map,
        train_data,
        pretrain,
    )


def _build_training_window_dataset(
    cfg,
    *,
    annotation: Path,
    class_map: Path,
    train_data: Path,
):
    from opentad.datasets import build_dataset

    dataset_cfg = copy.deepcopy(cfg.dataset.val)
    dataset_cfg.ann_file = str(annotation)
    dataset_cfg.class_map = str(class_map)
    dataset_cfg.data_path = str(train_data)
    dataset_cfg.subset_name = "training"
    dataset_cfg.block_list = None
    dataset_cfg.test_mode = False
    dataset_cfg.include_background_windows = True
    dataset_cfg.ioa_thresh = 0.75
    dataset_cfg.window_overlap_ratio = 0.5
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=None))
    videos = sorted({str(item[0]) for item in dataset.data_list})
    if len(videos) != 200:
        raise ValueError(
            f"frozen controller probe requires 200 training-side videos, got {len(videos)}"
        )
    return dataset, dataset_cfg, videos


def _build_loader(dataset, *, num_workers: int):
    import torch
    from opentad.datasets.builder import collate

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate,
    )


def _configure_real_budget_shape(model_cfg, *, execution_slots: int, pretrain: Path):
    model_cfg = copy.deepcopy(model_cfg)
    execution_slots = int(execution_slots)
    if execution_slots <= 0 or execution_slots > 512 or execution_slots % PACKET_SIZE != 0:
        raise ValueError("heavy execution length must be packet aligned and at most 512")
    model_cfg.backbone.custom.pretrain = str(pretrain)
    chunk_count = execution_slots // PACKET_SIZE
    pre_hits = 0
    for operation in model_cfg.backbone.custom.pre_processing_pipeline:
        if str(operation.get("type")) == "Rearrange" and "t1" in operation:
            operation["t1"] = chunk_count
            pre_hits += 1
    post_hits = 0
    for operation in model_cfg.backbone.custom.post_processing_pipeline:
        if str(operation.get("type")) == "Rearrange" and "t1" in operation:
            operation["t1"] = chunk_count
            post_hits += 1
    if pre_hits != 1 or post_hits != 1:
        raise ValueError(
            "VideoMAE wrapper must expose exactly one pre/post tubelet rearrange"
        )
    if int(model_cfg.projection.max_seq_len) != DETECTOR_LENGTH:
        raise ValueError("frozen detector length must remain 384")
    return model_cfg


def _load_frozen_model(
    cfg,
    *,
    checkpoint: Path,
    pretrain: Path,
    execution_slots: int,
    device,
):
    import torch
    from opentad.models.builder import build_detector

    model_cfg = _configure_real_budget_shape(
        cfg.model,
        execution_slots=execution_slots,
        pretrain=pretrain,
    )
    model = build_detector(model_cfg)
    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("H65 checkpoint must be a mapping")
    if int(payload.get("epoch", -1)) != 59:
        raise ValueError(
            f"frozen H65 checkpoint must report epoch 59, got {payload.get('epoch')!r}"
        )
    state = payload.get("state_dict_ema")
    if not isinstance(state, Mapping):
        raise ValueError("frozen H65 checkpoint has no state_dict_ema")
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"H65 checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.frame_selector is None:
        raise RuntimeError("frozen H65 model has no frame selector")
    model.frame_selector.inference_policy_alpha = 1.0
    frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
    model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)
    return model, payload, float(frozen_normalizer.detach().cpu().item())


def _move_batch(data: Mapping[str, Any], *, device):
    import torch

    inputs = data["inputs"].to(device, non_blocking=True)
    masks = data["masks"].to(device, non_blocking=True).bool()
    metas = [dict(item) for item in data["metas"]]
    gt_segments = [
        item.to(device, non_blocking=True)
        if torch.is_tensor(item)
        else torch.as_tensor(item, device=device, dtype=torch.float32)
        for item in data["gt_segments"]
    ]
    gt_labels = [
        item.to(device, non_blocking=True)
        if torch.is_tensor(item)
        else torch.as_tensor(item, device=device, dtype=torch.long)
        for item in data["gt_labels"]
    ]
    return inputs, masks, metas, gt_segments, gt_labels


@contextmanager
def _selector_disabled(model):
    selector = model.frame_selector
    model.frame_selector = None
    try:
        yield
    finally:
        model.frame_selector = selector


def _autocast(device, enabled: bool):
    import torch

    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    )


def _inference_settings(cfg):
    inference = copy.deepcopy(cfg.inference)
    inference.load_from_raw_predictions = False
    inference.save_raw_prediction = False
    post = copy.deepcopy(cfg.post_processing)
    post.sliding_window = True
    return inference, post


def _one_window_predictions(
    model,
    *,
    inputs,
    masks,
    metas,
    class_map: Sequence[str],
    inference,
    post,
    amp: bool,
    disable_selector: bool,
):
    import torch

    context = _selector_disabled(model) if disable_selector else nullcontext()
    with context, torch.no_grad(), _autocast(inputs.device, amp):
        results = model(
            inputs=inputs,
            masks=masks,
            metas=metas,
            return_loss=False,
            infer_cfg=inference,
            post_cfg=post,
            ext_cls=list(class_map),
        )
    return results


def _explicit_meta(
    raw_meta: Mapping[str, Any],
    *,
    acquisition_positions: Sequence[int],
    detector_positions: Sequence[float],
    dense_len: int,
    valid_len: int,
) -> dict[str, Any]:
    if len(detector_positions) != DETECTOR_LENGTH:
        raise ValueError("detector position map must contain 384 coordinates")
    if not 0 < len(acquisition_positions) <= max(BUDGETS):
        raise ValueError("acquisition positions must contain at most 512 real observations")
    out = dict(raw_meta)
    detector_values = [float(value) for value in detector_positions]
    acquisition_values = [int(value) for value in acquisition_positions]
    out.update(
        {
            "duca_acquisition_positions": acquisition_values,
            "duca_detector_grid_positions": detector_values,
            "selected_axis_to_true_time_dense_index": detector_values,
            "truetime_selected_positions": detector_values,
            "truetime_dense_len": int(dense_len),
            "truetime_dense_valid_len": int(valid_len),
            "irregular_selected_positions": detector_values,
            "irregular_native_axis": True,
            "irregular_selected_count": DETECTOR_LENGTH,
            "irregular_selected_valid_len": DETECTOR_LENGTH,
            "irregular_dense_valid_len": int(valid_len),
            "detector_output_coordinate_space": "selected_axis_index",
            "detector_prediction_inverse_map_required": True,
            "duca_online_selected_axis_remap": {
                "source": "selected_axis_index",
                "target": "true_time_dense_index",
                "selected_to_original": {
                    int(index): float(value)
                    for index, value in enumerate(detector_values)
                },
                "selected_axis_to_true_time_dense_index": detector_values,
                "acquisition_positions": acquisition_values,
            },
        }
    )
    return out


def _prepare_explicit_window(
    model,
    *,
    raw_inputs,
    raw_masks,
    raw_meta: Mapping[str, Any],
    gt_segments,
    gt_labels,
    positions: Sequence[int],
    detector_positions: Sequence[float],
    actual_count: int,
    execution_slots: int,
    baseline_execution: bool = False,
):
    import torch
    from opentad.models.duca import validate_real_heavy_observation_tensor

    stored_positions = torch.as_tensor(
        positions,
        device=raw_inputs.device,
        dtype=torch.long,
    ).reshape(1, -1)
    actual_count = int(actual_count)
    execution_slots = int(execution_slots)
    if not 0 < actual_count <= int(stored_positions.shape[1]):
        raise ValueError("actual_count must identify a non-empty stored active prefix")
    active = stored_positions[:, :actual_count]
    inactive = stored_positions[:, actual_count:]
    if torch.any(active < 0) or torch.any(active[:, 1:] <= active[:, :-1]):
        raise ValueError("active marginal positions must be ordered and non-negative")
    if inactive.numel() and torch.any(inactive != -1):
        raise ValueError("inactive stored marginal positions must use trailing -1 padding")
    if execution_slots < actual_count:
        raise ValueError("execution_slots cannot be smaller than actual_count")
    execution_positions = torch.full(
        (1, execution_slots),
        -1,
        device=raw_inputs.device,
        dtype=torch.long,
    )
    execution_positions[:, :actual_count] = active
    acquisition_mask = execution_positions >= 0
    selected = model._duca_gather_raw(raw_inputs, execution_positions)
    validate_real_heavy_observation_tensor(
        selected,
        actual_observations=torch.tensor([actual_count], device=raw_inputs.device),
        execution_slots=execution_slots,
        acquisition_mask=acquisition_mask,
        baseline_execution=baseline_execution,
    )
    detector_masks = torch.ones(
        1,
        DETECTOR_LENGTH,
        device=raw_masks.device,
        dtype=torch.bool,
    )
    meta = _explicit_meta(
        raw_meta,
        acquisition_positions=[int(value) for value in active[0].detach().cpu().tolist()],
        detector_positions=detector_positions,
        dense_len=int(raw_masks.shape[1]),
        valid_len=int(raw_masks[0].long().sum().item()),
    )
    remapped_segments, remapped_labels, remapped_metas = (
        model.frame_selector._remap_train_targets_to_selected_axis(
            gt_segments,
            gt_labels,
            [meta],
        )
    )
    return selected, detector_masks, remapped_metas, remapped_segments, remapped_labels


def _historical_k384_loss(
    model,
    *,
    raw_inputs,
    raw_masks,
    metas,
    gt_segments,
    gt_labels,
    amp: bool,
) -> float:
    import torch

    with torch.no_grad(), _autocast(raw_inputs.device, amp):
        losses = model.forward_train(
            raw_inputs,
            raw_masks,
            metas,
            gt_segments,
            gt_labels,
            _duca_counterfactual_eval=True,
        )
        objective = model._duca_detector_objective(losses)
    value = float(objective.detach().float().cpu().item())
    if not math.isfinite(value):
        raise RuntimeError("historical K384 detector loss is not finite")
    return value


def _explicit_loss_and_predictions(
    model,
    *,
    prepared,
    class_map: Sequence[str],
    inference,
    post,
    amp: bool,
    emit_predictions: bool,
):
    import torch

    inputs, masks, metas, gt_segments, gt_labels = prepared
    with torch.no_grad(), _autocast(inputs.device, amp):
        losses = model.forward_train(
            inputs,
            masks,
            metas,
            gt_segments,
            gt_labels,
            _duca_skip_frame_selector=True,
            _duca_counterfactual_eval=True,
        )
        objective = model._duca_detector_objective(losses)
    value = float(objective.detach().float().cpu().item())
    if not math.isfinite(value):
        raise RuntimeError("counterfactual detector loss is not finite")
    predictions = None
    if emit_predictions:
        predictions = _one_window_predictions(
            model,
            inputs=inputs,
            masks=masks,
            metas=metas,
            class_map=class_map,
            inference=inference,
            post=post,
            amp=amp,
            disable_selector=True,
        )
    return value, predictions


def _create_or_validate_split(annotation: Path, split_dir: Path) -> dict[str, Any]:
    from tools.bata.create_duca_frontend_split import create_split, validate_split_manifest

    manifest_path = split_dir / "frontend_split_manifest.json"
    if not manifest_path.exists():
        create_split(annotation, split_dir, seed=3407, holdout_fraction=0.20)
    validated = validate_split_manifest(manifest_path, annotation_path=annotation)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("train_video_count") != 160 or manifest.get("holdout_video_count") != 40:
        raise ValueError("frozen controller split must contain 160 fit and 40 holdout videos")
    manifest["validated"] = validated
    return manifest


def _stage_source(
    *,
    repo_root: Path,
    config_path: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    annotation: Path,
    class_map: Path,
    train_data: Path,
    pretrain: Path,
) -> dict[str, Any]:
    return {
        "git": _git_identity(repo_root),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "annotation": str(annotation),
        "annotation_sha256": _sha256(annotation),
        "class_map": str(class_map),
        "class_map_sha256": _sha256(class_map),
        "train_data": str(train_data),
        "videomae_pretrain": str(pretrain),
        "videomae_pretrain_sha256": _sha256(pretrain),
    }


def run_selection_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from opentad.models.duca import build_frozen_scout_marginal_features

    (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map_path,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = _stage_paths(output_dir)
    if paths["selection"].exists() and paths["selection_receipt"].exists():
        return json.loads(paths["selection_receipt"].read_text(encoding="utf-8"))
    split = _create_or_validate_split(annotation, paths["split_dir"])
    holdout = set(str(value) for value in split["holdout_videos"])
    dataset, _dataset_cfg, videos = _build_training_window_dataset(
        cfg,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
    )
    loader = _build_loader(dataset, num_workers=args.num_workers)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, payload, frozen_normalizer = _load_frozen_model(
        cfg,
        checkpoint=checkpoint,
        pretrain=pretrain,
        execution_slots=BASELINE_BUDGET,
        device=device,
    )
    inference, post = _inference_settings(cfg)
    class_map = list(dataset.class_map)
    rows: list[dict[str, Any]] = []
    seen = set()
    try:
        for index, data in enumerate(loader):
            raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(
                data,
                device=device,
            )
            if len(metas) != 1:
                raise RuntimeError("selection stage requires batch size one")
            meta = metas[0]
            sample_id = _sample_id(meta)
            if sample_id in seen:
                raise ValueError(f"duplicate training window identity: {sample_id}")
            seen.add(sample_id)
            video_id = str(meta["video_name"])

            normal = _one_window_predictions(
                model,
                inputs=raw_inputs,
                masks=raw_masks,
                metas=metas,
                class_map=class_map,
                inference=inference,
                post=post,
                amp=args.amp,
                disable_selector=False,
            )
            normal_positions = model.frame_selector._last_selected_positions.detach().clone()
            loss_k384 = _historical_k384_loss(
                model,
                raw_inputs=raw_inputs,
                raw_masks=raw_masks,
                metas=metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                amp=args.amp,
            )
            with torch.no_grad(), _autocast(device, args.amp):
                prefixes = model.frame_selector.forward_marginal_prefixes(
                    raw_inputs,
                    raw_masks,
                    metas,
                    budgets=BUDGETS,
                    detector_length=DETECTOR_LENGTH,
                )
            baseline_positions = prefixes["positions_by_budget"][BASELINE_BUDGET]
            if not torch.equal(normal_positions, baseline_positions):
                raise RuntimeError(f"{sample_id}: nested K384 is not bit-exact H65 selection")
            valid_count = int(prefixes["valid_count"][0].detach().cpu().item())
            features = build_frozen_scout_marginal_features(
                prefixes["selector_outputs"],
                baseline_positions,
            )
            if features.shape[0] != 1 or features.requires_grad:
                raise RuntimeError("frozen Scout feature extraction violated detach contract")

            selected_positions = {
                str(budget): [
                    int(value)
                    for value in prefixes["positions_by_budget"][budget][0]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                for budget in BUDGETS
            }
            detector_positions = {
                str(budget): [
                    float(value)
                    for value in prefixes["detector_grid_by_budget"][budget][0]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                ]
                for budget in BUDGETS
            }
            normal_hash = _canonical_sha256(normal)
            explicit_hash = None
            parity_path = "historical_padded_k384_short_window"
            if valid_count >= BASELINE_BUDGET:
                prepared = _prepare_explicit_window(
                    model,
                    raw_inputs=raw_inputs,
                    raw_masks=raw_masks,
                    raw_meta=meta,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    positions=selected_positions[str(BASELINE_BUDGET)],
                    detector_positions=detector_positions[str(BASELINE_BUDGET)],
                    actual_count=BASELINE_BUDGET,
                    execution_slots=BASELINE_BUDGET,
                    baseline_execution=True,
                )
                _explicit_loss, explicit = _explicit_loss_and_predictions(
                    model,
                    prepared=prepared,
                    class_map=class_map,
                    inference=inference,
                    post=post,
                    amp=args.amp,
                    emit_predictions=True,
                )
                explicit_hash = _canonical_sha256(explicit)
                if normal_hash != explicit_hash:
                    raise RuntimeError(
                        f"{sample_id}: explicit K384 prediction differs from frozen H65"
                    )
                parity_path = "full_window_explicit_k384_matches_historical"
            accounting = {
                str(budget): {
                    "actual_cost": int(prefixes["actual_count_by_budget"][budget][0].item()),
                    "effective_tier": int(prefixes["effective_budget_by_requested"][budget][0].item()),
                    "execution_slots": int(prefixes["execution_slots_by_budget"][budget][0].item()),
                    "padding_slots": int(prefixes["padding_slots_by_budget"][budget][0].item()),
                    "collapsed_to_k384": bool(prefixes["collapsed_to_baseline_by_budget"][budget][0].item()),
                }
                for budget in BUDGETS
            }
            prediction = normal.get(video_id, []) if video_id in holdout else None
            row = {
                "schema": "duca_marginal_selection_k384_v1",
                "sample_id": sample_id,
                "video_id": video_id,
                "window_start_frame": int(round(float(meta.get("window_start_frame", 0)))),
                "valid_observations": valid_count,
                "controller_partition": "holdout" if video_id in holdout else "fit",
                "positions": selected_positions,
                "budget_accounting": accounting,
                "detector_grid_positions": detector_positions,
                "scout_features": [
                    float(value) for value in features[0].detach().float().cpu().tolist()
                ],
                "loss_k384": loss_k384,
                "prediction_k384": prediction,
                "normal_h65_prediction_sha256": normal_hash,
                "explicit_k384_prediction_sha256": explicit_hash,
                "k384_parity_path": parity_path,
                "k384_selection_bit_exact": True,
                "k384_prediction_exact": True,
            }
            rows.append(row)
            if (index + 1) % 25 == 0:
                print(f"selection_k384 windows={index + 1}/{len(dataset)}", flush=True)
    finally:
        model.rpn_head.duca_set_frozen_loss_normalizer(None)

    if {row["video_id"] for row in rows} != set(videos):
        raise RuntimeError("selection stage did not cover every training-side video")
    row_count = _write_jsonl_gz(paths["selection"], rows)
    source = _stage_source(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
        pretrain=pretrain,
    )
    receipt = {
        "status": "SELECTION_K384_COMPLETE",
        "stage": "select-k384",
        "artifact": str(paths["selection"]),
        "artifact_sha256": _sha256(paths["selection"]),
        "window_count": row_count,
        "video_count": len(videos),
        "fit_video_count": 160,
        "holdout_video_count": 40,
        "checkpoint_payload_epoch": int(payload["epoch"]),
        "frozen_loss_normalizer": frozen_normalizer,
        "k384_selection_bit_exact_all_windows": True,
        "k384_prediction_exact_all_windows": True,
        "detector_frozen": True,
        "scout_frozen": True,
        "source": source,
    }
    _write_json(paths["selection_receipt"], receipt)
    return receipt


def run_counterfactual_stage(args: argparse.Namespace, *, budget: int) -> dict[str, Any]:
    import torch

    if budget not in {256, 512}:
        raise ValueError("counterfactual stage budget must be 256 or 512")
    (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map_path,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = _stage_paths(output_dir)
    artifact_key = "k256" if budget == 256 else "k512"
    receipt_key = f"{artifact_key}_receipt"
    if paths[artifact_key].exists() and paths[receipt_key].exists():
        return json.loads(paths[receipt_key].read_text(encoding="utf-8"))
    selection_rows = _read_jsonl_gz(paths["selection"])
    selection_by_id = {str(row["sample_id"]): row for row in selection_rows}
    if len(selection_by_id) != len(selection_rows):
        raise ValueError("selection artifact contains duplicate sample identities")
    split = _create_or_validate_split(annotation, paths["split_dir"])
    holdout = set(str(value) for value in split["holdout_videos"])
    dataset, _dataset_cfg, videos = _build_training_window_dataset(
        cfg,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    inference, post = _inference_settings(cfg)
    class_map = list(dataset.class_map)
    dataset_index_by_id: dict[str, int] = {}
    for dataset_index, item in enumerate(dataset.data_list):
        video_id = str(item[0])
        window_positions = item[3]
        sample_id = f"{video_id}|{int(round(float(window_positions[0])))}"
        if sample_id in dataset_index_by_id:
            raise ValueError(f"duplicate dataset sample identity: {sample_id}")
        dataset_index_by_id[sample_id] = dataset_index
    if set(dataset_index_by_id) != set(selection_by_id):
        raise RuntimeError("counterfactual dataset identities differ from selection stage")

    row_by_id: dict[str, dict[str, Any]] = {}
    groups: dict[int, list[int]] = {}
    collapsed_count = 0
    for source_row in selection_rows:
        sample_id = str(source_row["sample_id"])
        accounting = source_row["budget_accounting"][str(budget)]
        if bool(accounting["collapsed_to_k384"]):
            collapsed_count += 1
            row_by_id[sample_id] = {
                "schema": f"duca_marginal_counterfactual_k{budget}_v2",
                "sample_id": sample_id,
                "video_id": str(source_row["video_id"]),
                "requested_budget": budget,
                "effective_tier": BASELINE_BUDGET,
                "actual_heavy_observations": int(accounting["actual_cost"]),
                "execution_slots": BASELINE_BUDGET,
                "padding_slots": int(accounting["padding_slots"]),
                "collapsed_to_k384": True,
                "detector_forward_executed": False,
                "loss": float(source_row["loss_k384"]),
                "prediction": source_row["prediction_k384"],
                "controller_partition": source_row["controller_partition"],
                "detector_frozen": True,
                "scout_not_executed_in_counterfactual_stage": True,
            }
            continue
        execution_slots = int(accounting["execution_slots"])
        groups.setdefault(execution_slots, []).append(dataset_index_by_id[sample_id])

    payload_epoch = None
    frozen_normalizers = set()
    for execution_slots, dataset_indices in sorted(groups.items()):
        model, payload, frozen_normalizer = _load_frozen_model(
            cfg,
            checkpoint=checkpoint,
            pretrain=pretrain,
            execution_slots=execution_slots,
            device=device,
        )
        payload_epoch = int(payload["epoch"])
        frozen_normalizers.add(float(frozen_normalizer))
        loader = _build_loader(
            torch.utils.data.Subset(dataset, dataset_indices),
            num_workers=args.num_workers,
        )
        try:
            for data in loader:
                raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(
                    data,
                    device=device,
                )
                meta = metas[0]
                sample_id = _sample_id(meta)
                source_row = selection_by_id[sample_id]
                video_id = str(meta["video_name"])
                accounting = source_row["budget_accounting"][str(budget)]
                if bool(accounting["collapsed_to_k384"]):
                    raise RuntimeError("collapsed counterfactual entered a detector execution group")
                if int(accounting["execution_slots"]) != execution_slots:
                    raise RuntimeError("counterfactual execution group does not match saved accounting")
                positions = source_row["positions"][str(budget)]
                detector_positions = source_row["detector_grid_positions"][str(budget)]
                prepared = _prepare_explicit_window(
                    model,
                    raw_inputs=raw_inputs,
                    raw_masks=raw_masks,
                    raw_meta=meta,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    positions=positions,
                    detector_positions=detector_positions,
                    actual_count=int(accounting["actual_cost"]),
                    execution_slots=execution_slots,
                )
                loss, predictions = _explicit_loss_and_predictions(
                    model,
                    prepared=prepared,
                    class_map=class_map,
                    inference=inference,
                    post=post,
                    amp=args.amp,
                    emit_predictions=video_id in holdout,
                )
                row_by_id[sample_id] = {
                    "schema": f"duca_marginal_counterfactual_k{budget}_v2",
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "requested_budget": budget,
                    "effective_tier": budget,
                    "actual_heavy_observations": int(accounting["actual_cost"]),
                    "execution_slots": execution_slots,
                    "padding_slots": int(accounting["padding_slots"]),
                    "collapsed_to_k384": False,
                    "detector_forward_executed": True,
                    "loss": loss,
                    "prediction": None if predictions is None else predictions.get(video_id, []),
                    "controller_partition": "holdout" if video_id in holdout else "fit",
                    "detector_frozen": True,
                    "scout_not_executed_in_counterfactual_stage": True,
                }
        finally:
            model.rpn_head.duca_set_frozen_loss_normalizer(None)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = [row_by_id[str(row["sample_id"])] for row in selection_rows]
    seen = set(row_by_id)
    if seen != set(selection_by_id):
        missing = sorted(set(selection_by_id) - seen)
        raise RuntimeError(f"counterfactual K{budget} missed windows: {missing[:5]}")
    row_count = _write_jsonl_gz(paths[artifact_key], rows)
    source = _stage_source(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
        pretrain=pretrain,
    )
    receipt = {
        "status": f"COUNTERFACTUAL_K{budget}_COMPLETE",
        "stage": f"counterfactual-k{budget}",
        "artifact": str(paths[artifact_key]),
        "artifact_sha256": _sha256(paths[artifact_key]),
        "window_count": row_count,
        "video_count": len(videos),
        "requested_budget": budget,
        "distinct_forward_count": len(rows) - collapsed_count,
        "collapsed_alias_count": collapsed_count,
        "observed_execution_slot_classes": sorted(groups),
        "padded_to_k512": False,
        "checkpoint_payload_epoch": 59 if payload_epoch is None else payload_epoch,
        "frozen_loss_normalizers": sorted(frozen_normalizers),
        "detector_frozen": True,
        "source": source,
    }
    _write_json(paths[receipt_key], receipt)
    return receipt


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = _rankdata(np.asarray(actual, dtype=np.float64))
    right = _rankdata(np.asarray(predicted, dtype=np.float64))
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _sign_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = np.sign(np.asarray(actual, dtype=np.float64))
    right = np.sign(np.asarray(predicted, dtype=np.float64))
    if left.size == 0:
        return 0.0
    return float(np.mean(left == right))


def _apply_sliding_window_nms(raw_results: Mapping[str, list], *, nms_cfg) -> dict[str, list]:
    import torch
    from opentad.models.utils.post_processing import batched_nms

    output: dict[str, list] = {}
    for video_id, proposals in raw_results.items():
        if not proposals:
            output[str(video_id)] = []
            continue
        segments = torch.tensor([item["segment"] for item in proposals], dtype=torch.float32)
        scores = torch.tensor([item["score"] for item in proposals], dtype=torch.float32)
        class_names: list[str] = []
        labels = []
        for item in proposals:
            label = str(item["label"])
            if label not in class_names:
                class_names.append(label)
            labels.append(class_names.index(label))
        label_tensor = torch.tensor(labels, dtype=torch.long)
        segments, scores, label_tensor = batched_nms(
            segments,
            scores,
            label_tensor,
            **nms_cfg,
        )
        output[str(video_id)] = [
            {
                "segment": [round(float(value), 2) for value in segment.tolist()],
                "label": class_names[int(label.item())],
                "score": round(float(score.item()), 4),
            }
            for segment, label, score in zip(segments, label_tensor, scores)
        ]
    return output


def _json_block_list_for_evaluator(block_list: str | Path) -> Path:
    source = Path(block_list).expanduser().resolve()
    blocked_videos = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = source.with_suffix(".evaluator.json")
    payload = json.dumps(blocked_videos, indent=2, sort_keys=False) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                f"refusing to overwrite a different evaluator block list: {target}"
            )
    else:
        target.write_text(payload, encoding="utf-8")
    return target


def _official_holdout_metrics(
    cfg,
    *,
    raw_results: Mapping[str, list],
    annotation: Path,
    holdout_block_list: str,
    evaluator_threads: int,
):
    from opentad.evaluations import build_evaluator

    final_results = _apply_sliding_window_nms(raw_results, nms_cfg=cfg.post_processing.nms)
    evaluation = copy.deepcopy(cfg.evaluation)
    evaluation.ground_truth_filename = str(annotation)
    evaluation.subset = "training"
    evaluation.blocked_videos = str(
        _json_block_list_for_evaluator(holdout_block_list)
    )
    evaluation.thread = int(evaluator_threads)
    evaluator = build_evaluator(
        dict(prediction_filename={"results": final_results}, **evaluation)
    )
    metrics = {
        key: float(value) for key, value in evaluator.evaluate().items()
    }
    return metrics, final_results


def _allocate_rows_by_video(
    rows: Sequence[Mapping[str, Any]],
    *,
    downgrade: Sequence[float],
    upgrade: Sequence[float],
):
    import torch
    from opentad.models.duca import allocate_equal_budget_marginal_reallocation

    by_video: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_video.setdefault(str(row["video_id"]), []).append(index)
    budgets = [BASELINE_BUDGET] * len(rows)
    allocation_summaries = {}
    for video_id, indices in sorted(by_video.items()):
        indices = sorted(indices, key=lambda index: str(rows[index]["sample_id"]))
        decision = allocate_equal_budget_marginal_reallocation(
            torch.tensor([downgrade[index] for index in indices], dtype=torch.float32),
            torch.tensor([upgrade[index] for index in indices], dtype=torch.float32),
            torch.tensor(
                [int(rows[index]["valid_observations"]) for index in indices],
                dtype=torch.long,
            ),
            max_changed_fraction=0.5,
        )
        if not decision.feasible:
            raise RuntimeError(f"{video_id}: exact equal-budget allocation failed: {decision.reason}")
        values = [int(value) for value in decision.effective_budget.cpu().tolist()]
        for row_index, budget in zip(indices, values):
            budgets[row_index] = budget
        allocation_summaries[video_id] = {
            "window_count": len(indices),
            "budgets": values,
            "requested_budgets": [int(value) for value in decision.budget.cpu().tolist()],
            "actual_cost": [int(value) for value in decision.actual_cost.cpu().tolist()],
            "execution_slots": [int(value) for value in decision.execution_slots.cpu().tolist()],
            "padding_slots": [int(value) for value in decision.padding_slots.cpu().tolist()],
            "collapsed_to_k384": [bool(value) for value in decision.collapsed_to_baseline.cpu().tolist()],
            "target_actual_cost": int(decision.target_actual_cost),
            "actual_budget_error": int(decision.actual_cost.sum().item())
            - int(decision.target_actual_cost),
            "predicted_total_utility": float(decision.predicted_total_utility.item()),
        }
    return budgets, allocation_summaries


def _raw_results_for_budgets(
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
) -> dict[str, list]:
    results: dict[str, list] = {}
    for row, budget in zip(rows, budgets):
        prediction = row["predictions"].get(str(int(budget)))
        if prediction is None:
            raise RuntimeError(
                f"{row['sample_id']}: holdout prediction for K{budget} is missing"
            )
        results.setdefault(str(row["video_id"]), []).extend(prediction)
    return results


def _fit_utility_head(
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
):
    import torch
    from opentad.models.duca import SignedTwoSidedMarginalUtilityHead

    features = torch.tensor(
        [row["scout_features"] for row in fit_rows],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[row["downgrade_penalty"], row["upgrade_gain"]] for row in fit_rows],
        dtype=torch.float32,
    )
    target_valid = torch.tensor(
        [[row["downgrade_target_valid"], row["upgrade_target_valid"]] for row in fit_rows],
        dtype=torch.bool,
    )
    head = SignedTwoSidedMarginalUtilityHead(
        input_dim=int(features.shape[1]),
        hidden_dim=128,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets, target_valid),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    terminal_loss = None
    head.train()
    for _epoch in range(20):
        epoch_loss = 0.0
        sample_count = 0
        for feature_batch, target_batch, valid_batch in loader:
            valid_count = int(valid_batch.long().sum().item())
            if valid_count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = head(feature_batch)
            values = torch.stack(
                (prediction["downgrade_penalty"], prediction["upgrade_gain"]),
                dim=1,
            )
            squared = (values - target_batch).square()
            loss = squared.masked_select(valid_batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * valid_count
            sample_count += valid_count
        terminal_loss = epoch_loss / max(1, sample_count)
    return head.eval(), float(terminal_loss)


def run_summary_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map_path,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = _stage_paths(output_dir)
    if paths["result"].exists():
        return json.loads(paths["result"].read_text(encoding="utf-8"))
    selection_rows = _read_jsonl_gz(paths["selection"])
    k256_rows = {row["sample_id"]: row for row in _read_jsonl_gz(paths["k256"])}
    k512_rows = {row["sample_id"]: row for row in _read_jsonl_gz(paths["k512"])}
    if set(k256_rows) != {row["sample_id"] for row in selection_rows}:
        raise ValueError("K256 counterfactual sample set differs from selection stage")
    if set(k512_rows) != {row["sample_id"] for row in selection_rows}:
        raise ValueError("K512 counterfactual sample set differs from selection stage")
    split = _create_or_validate_split(annotation, paths["split_dir"])
    fit_videos = set(str(value) for value in split["train_videos"])
    holdout_videos = set(str(value) for value in split["holdout_videos"])
    merged = []
    for selected in selection_rows:
        sample_id = str(selected["sample_id"])
        loss256 = float(k256_rows[sample_id]["loss"])
        loss384 = float(selected["loss_k384"])
        loss512 = float(k512_rows[sample_id]["loss"])
        row = dict(selected)
        row.update(
            {
                "loss_k256": loss256,
                "loss_k512": loss512,
                "downgrade_penalty": loss256 - loss384,
                "upgrade_gain": loss384 - loss512,
                "downgrade_target_valid": not bool(k256_rows[sample_id]["collapsed_to_k384"]),
                "upgrade_target_valid": not bool(k512_rows[sample_id]["collapsed_to_k384"]),
                "predictions": {
                    "256": k256_rows[sample_id]["prediction"],
                    "384": selected["prediction_k384"],
                    "512": k512_rows[sample_id]["prediction"],
                },
            }
        )
        merged.append(row)
    fit_rows = [row for row in merged if row["video_id"] in fit_videos]
    holdout_rows = [row for row in merged if row["video_id"] in holdout_videos]
    if {row["video_id"] for row in fit_rows} != fit_videos:
        raise RuntimeError("fit artifact does not cover every frozen fit video")
    if {row["video_id"] for row in holdout_rows} != holdout_videos:
        raise RuntimeError("holdout artifact does not cover every frozen holdout video")

    fixed_budgets = [BASELINE_BUDGET] * len(holdout_rows)
    fixed_raw = _raw_results_for_budgets(holdout_rows, fixed_budgets)
    fixed_metrics, fixed_predictions = _official_holdout_metrics(
        cfg,
        raw_results=fixed_raw,
        annotation=annotation,
        holdout_block_list=split["holdout_block_list"],
        evaluator_threads=args.evaluator_threads,
    )
    actual_downgrade = [float(row["downgrade_penalty"]) for row in holdout_rows]
    actual_upgrade = [float(row["upgrade_gain"]) for row in holdout_rows]
    downgrade_valid = [bool(row["downgrade_target_valid"]) for row in holdout_rows]
    upgrade_valid = [bool(row["upgrade_target_valid"]) for row in holdout_rows]
    oracle_budgets, oracle_allocations = _allocate_rows_by_video(
        holdout_rows,
        downgrade=actual_downgrade,
        upgrade=actual_upgrade,
    )
    oracle_raw = _raw_results_for_budgets(holdout_rows, oracle_budgets)
    oracle_metrics, oracle_predictions = _official_holdout_metrics(
        cfg,
        raw_results=oracle_raw,
        annotation=annotation,
        holdout_block_list=split["holdout_block_list"],
        evaluator_threads=args.evaluator_threads,
    )
    delta_avg_pp = 100.0 * (
        oracle_metrics["average_mAP"] - fixed_metrics["average_mAP"]
    )
    delta_07_pp = 100.0 * (
        oracle_metrics["mAP@0.7"] - fixed_metrics["mAP@0.7"]
    )
    headroom_pass = delta_avg_pp >= 0.8 and delta_07_pp >= 1.0
    no_headroom = delta_avg_pp < 0.3 and delta_07_pp < 0.5
    result: dict[str, Any] = {
        "schema": "duca_marginal_frozen_h65_probe_result_v1",
        "method": "DUCA-Marginal-v1",
        "status": (
            "ORACLE_HEADROOM_PASS"
            if headroom_pass
            else "NO_ORACLE_HEADROOM"
            if no_headroom
            else "ORACLE_HEADROOM_GRAY_ZONE_RETURN_TO_PRO"
        ),
        "paper_claim_allowed": False,
        "official_test_consumed": False,
        "fit_video_count": len(fit_videos),
        "holdout_video_count": len(holdout_videos),
        "fit_window_count": len(fit_rows),
        "holdout_window_count": len(holdout_rows),
        "implementation_gate": {
            "k384_selection_bit_exact_all_windows": all(
                bool(row["k384_selection_bit_exact"]) for row in selection_rows
            ),
            "k384_prediction_exact_all_windows": all(
                bool(row["k384_prediction_exact"]) for row in selection_rows
            ),
            "observed_distinct_execution_slot_classes": sorted(
                {
                    int(source["execution_slots"])
                    for rows_by_budget in (k256_rows, k512_rows)
                    for source in rows_by_budget.values()
                    if not bool(source["collapsed_to_k384"])
                }
            ),
            "padded_to_upper_budget": False,
            "detector_frozen": True,
            "scout_frozen": True,
            "utility_targets_detached": True,
        },
        "fixed_arm_name": "Fixed-H65-384",
        "fixed_h65_384": fixed_metrics,
        "oracle_reallocate_384": oracle_metrics,
        "oracle_headroom": {
            "delta_average_mAP_pp": delta_avg_pp,
            "delta_mAP_at_0.7_pp": delta_07_pp,
            "strong_gate_pass": headroom_pass,
            "no_headroom_boundary": no_headroom,
            "gray_zone_requires_pro": not headroom_pass and not no_headroom,
        },
        "oracle_allocation": oracle_allocations,
        "secondary_mean_k320": {
            "status": "NOT_RUN_UNRESOLVED_EXACT_VIDEO_BUDGET",
            "reason": (
                "K in {256,384} cannot guarantee exact mean K=320 for videos "
                "with an odd number of windows without an additional frozen rule"
            ),
        },
        "source": _stage_source(
            repo_root=repo_root,
            config_path=config_path,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            annotation=annotation,
            class_map=class_map_path,
            train_data=train_data,
            pretrain=pretrain,
        ),
        "stage_artifacts": {
            "selection": {
                "path": str(paths["selection"]),
                "sha256": _sha256(paths["selection"]),
            },
            "k256": {"path": str(paths["k256"]), "sha256": _sha256(paths["k256"])},
            "k512": {"path": str(paths["k512"]), "sha256": _sha256(paths["k512"])},
        },
    }
    _write_json(output_dir / "holdout_fixed_h65_384_predictions.json", {"results": fixed_predictions})
    _write_json(output_dir / "holdout_oracle_reallocate_384_predictions.json", {"results": oracle_predictions})

    if headroom_pass:
        head, terminal_loss = _fit_utility_head(fit_rows, seed=3407)
        holdout_features = torch.tensor(
            [row["scout_features"] for row in holdout_rows],
            dtype=torch.float32,
        )
        with torch.no_grad():
            predicted = head(holdout_features)
        predicted_downgrade = [
            float(value) for value in predicted["downgrade_penalty"].cpu().tolist()
        ]
        predicted_upgrade = [
            float(value) for value in predicted["upgrade_gain"].cpu().tolist()
        ]
        learned_budgets, learned_allocations = _allocate_rows_by_video(
            holdout_rows,
            downgrade=predicted_downgrade,
            upgrade=predicted_upgrade,
        )
        learned_raw = _raw_results_for_budgets(holdout_rows, learned_budgets)
        learned_metrics, learned_predictions = _official_holdout_metrics(
            cfg,
            raw_results=learned_raw,
            annotation=annotation,
            holdout_block_list=split["holdout_block_list"],
            evaluator_threads=args.evaluator_threads,
        )
        oracle_gain = oracle_metrics["average_mAP"] - fixed_metrics["average_mAP"]
        learned_gain = learned_metrics["average_mAP"] - fixed_metrics["average_mAP"]
        recovered = float(learned_gain / oracle_gain) if oracle_gain > 0 else 0.0
        fraction_256 = sum(value == 256 for value in learned_budgets) / len(learned_budgets)
        fraction_512 = sum(value == 512 for value in learned_budgets) / len(learned_budgets)
        budget_errors = [
            int(value["actual_budget_error"]) for value in learned_allocations.values()
        ]
        actual_downgrade_eligible = [
            value for value, eligible in zip(actual_downgrade, downgrade_valid) if eligible
        ]
        predicted_downgrade_eligible = [
            value for value, eligible in zip(predicted_downgrade, downgrade_valid) if eligible
        ]
        actual_upgrade_eligible = [
            value for value, eligible in zip(actual_upgrade, upgrade_valid) if eligible
        ]
        predicted_upgrade_eligible = [
            value for value, eligible in zip(predicted_upgrade, upgrade_valid) if eligible
        ]
        predictability = {
            "downgrade_spearman": _spearman(
                actual_downgrade_eligible,
                predicted_downgrade_eligible,
            ),
            "upgrade_spearman": _spearman(
                actual_upgrade_eligible,
                predicted_upgrade_eligible,
            ),
            "downgrade_sign_accuracy": _sign_accuracy(
                actual_downgrade_eligible,
                predicted_downgrade_eligible,
            ),
            "upgrade_sign_accuracy": _sign_accuracy(
                actual_upgrade_eligible,
                predicted_upgrade_eligible,
            ),
            "downgrade_eligible_window_count": len(actual_downgrade_eligible),
            "upgrade_eligible_window_count": len(actual_upgrade_eligible),
            "learned_oracle_gain_fraction": recovered,
            "k256_window_fraction": fraction_256,
            "k512_window_fraction": fraction_512,
            "max_abs_video_budget_error": max(abs(value) for value in budget_errors),
        }
        predictability["gate_pass"] = bool(
            predictability["downgrade_spearman"] >= 0.25
            and predictability["upgrade_spearman"] >= 0.25
            and predictability["downgrade_sign_accuracy"] >= 0.60
            and predictability["upgrade_sign_accuracy"] >= 0.60
            and recovered >= 0.40
            and fraction_256 >= 0.10
            and fraction_512 >= 0.10
            and predictability["max_abs_video_budget_error"] == 0
        )
        result.update(
            {
                "status": (
                    "MECHANISM_GATE_PASS"
                    if predictability["gate_pass"]
                    else "ORACLE_POSITIVE_PREDICTOR_FAILED"
                ),
                "utility_training": {
                    "epochs": 20,
                    "terminal_mse": terminal_loss,
                    "fit_window_count": len(fit_rows),
                    "only_trainable_module": "SignedTwoSidedMarginalUtilityHead",
                },
                "predictability": predictability,
                "learned_reallocate_384": learned_metrics,
                "learned_allocation": learned_allocations,
            }
        )
        head_path = output_dir / "utility_head_epoch20.pth"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "input_dim": int(holdout_features.shape[1]),
                "hidden_dim": 128,
                "epoch": 20,
                "seed": 3407,
            },
            head_path,
        )
        result["utility_training"]["checkpoint"] = str(head_path)
        result["utility_training"]["checkpoint_sha256"] = _sha256(head_path)
        _write_json(
            output_dir / "holdout_learned_reallocate_384_predictions.json",
            {"results": learned_predictions},
        )

    _write_json(paths["result"], result)
    return result


def _single_dataset_batch(dataset, index: int, *, num_workers: int):
    import torch

    loader = _build_loader(
        torch.utils.data.Subset(dataset, [int(index)]),
        num_workers=int(num_workers),
    )
    return next(iter(loader))


def run_pre_run_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Bounded scientific preflight for the frozen short-window contract."""

    import torch

    (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map_path,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = _stage_paths(output_dir)
    identity = _git_identity(repo_root)
    if identity["dirty"]:
        raise RuntimeError("PRE_RUN requires one clean Git commit")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("PRE_RUN must execute in the bound Linux runtime")

    focused_files = [
        "opentad/models/duca/dynamic_budget.py",
        "opentad/models/duca/acquisition.py",
        "opentad/models/duca/counterfactual_utility.py",
        "opentad/models/selectors/duca_online_frame_selector.py",
        "tools/bata/run_duca_marginal_frozen_h65_probe.py",
        "tools/bata/train_lowres_action_probe.py",
        "tools/train.py",
        "tests/test_duca_marginal_budget.py",
    ]
    subprocess.run(["git", "diff", "--check"], cwd=repo_root, check=True)
    subprocess.run(
        [sys.executable, "-m", "py_compile", *focused_files],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_duca_marginal_budget.py",
            "tests/test_c3_coarse_classifier_model_matrix.py",
            "tests/test_c3_asformer_delta_ledger_full_train.py",
            "-q",
        ],
        cwd=repo_root,
        check=True,
    )

    split = _create_or_validate_split(annotation, paths["split_dir"])
    fit_videos = set(str(value) for value in split["train_videos"])
    holdout_videos = set(str(value) for value in split["holdout_videos"])
    if fit_videos & holdout_videos or len(fit_videos) != 160 or len(holdout_videos) != 40:
        raise RuntimeError("PRE_RUN split must be disjoint 160/40")
    dataset, _dataset_cfg, videos = _build_training_window_dataset(
        cfg,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
    )
    if fit_videos | holdout_videos != set(videos):
        raise RuntimeError("PRE_RUN split union must equal all 200 training-side videos")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, payload, frozen_normalizer = _load_frozen_model(
        cfg,
        checkpoint=checkpoint,
        pretrain=pretrain,
        execution_slots=BASELINE_BUDGET,
        device=device,
    )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("PRE_RUN detector and Scout must have no trainable parameters")

    loader = _build_loader(dataset, num_workers=args.num_workers)
    baseline_cost_by_video: dict[str, int] = {}
    expected_baseline_cost_by_video: dict[str, int] = {}
    scanned_videos = set()
    scanned_windows = 0
    collapsed_alias_count = 0
    representatives: dict[tuple[int, int], dict[str, Any]] = {}
    short_k384_index = None
    full_k384_example = None
    try:
        for dataset_index, data in enumerate(loader):
            raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(
                data,
                device=device,
            )
            meta = metas[0]
            video_id = str(meta["video_name"])
            sample_id = _sample_id(meta)
            raw_valid_count = int(raw_masks[0].long().sum().item())
            with torch.no_grad(), _autocast(device, args.amp):
                normal_outputs = model.frame_selector.forward_test(raw_inputs, raw_masks, metas)
                normal_positions = model.frame_selector._last_selected_positions.detach().clone()
                prefixes = model.frame_selector.forward_marginal_prefixes(
                    raw_inputs,
                    raw_masks,
                    metas,
                    budgets=BUDGETS,
                    detector_length=DETECTOR_LENGTH,
                )
            if not torch.equal(normal_positions, prefixes["positions_by_budget"][BASELINE_BUDGET]):
                raise RuntimeError(f"{sample_id}: full K384 tensor differs from normal H65")
            if not torch.equal(normal_outputs["inputs"], prefixes["historical_k384_inputs"]):
                raise RuntimeError(f"{sample_id}: K384 detector input differs from normal H65")
            if not torch.equal(normal_outputs["masks"], prefixes["historical_k384_masks"]):
                raise RuntimeError(f"{sample_id}: K384 detector mask differs from normal H65")
            valid_count = int(prefixes["valid_count"][0].item())
            if valid_count != raw_valid_count:
                raise RuntimeError(f"{sample_id}: selector valid count differs from the raw mask")
            baseline_actual = min(valid_count, BASELINE_BUDGET)
            baseline_cost_by_video[video_id] = baseline_cost_by_video.get(video_id, 0) + baseline_actual
            expected_baseline_cost_by_video[video_id] = (
                expected_baseline_cost_by_video.get(video_id, 0)
                + min(raw_valid_count, BASELINE_BUDGET)
            )
            scanned_videos.add(video_id)
            scanned_windows += 1
            if valid_count < BASELINE_BUDGET and short_k384_index is None:
                short_k384_index = dataset_index
            if valid_count >= BASELINE_BUDGET and full_k384_example is None:
                full_k384_example = {
                    "dataset_index": dataset_index,
                    "positions": prefixes["positions_by_budget"][384][0].detach().cpu().tolist(),
                    "detector_positions": prefixes["detector_grid_by_budget"][384][0].detach().cpu().tolist(),
                }
            for budget in BUDGETS:
                accounting = prefixes["effective_budget_by_requested"][budget]
                actual = int(prefixes["actual_count_by_budget"][budget][0].item())
                effective = int(accounting[0].item())
                execution_slots = int(prefixes["execution_slots_by_budget"][budget][0].item())
                padding_slots = int(prefixes["padding_slots_by_budget"][budget][0].item())
                collapsed = bool(prefixes["collapsed_to_baseline_by_budget"][budget][0].item())
                expected_actual = min(valid_count, budget)
                expected_baseline = min(valid_count, BASELINE_BUDGET)
                expected_effective = BASELINE_BUDGET if expected_actual == expected_baseline else budget
                expected_execution = (
                    BASELINE_BUDGET
                    if expected_effective == BASELINE_BUDGET
                    else PACKET_SIZE * ((expected_actual + PACKET_SIZE - 1) // PACKET_SIZE)
                )
                if (actual, effective, execution_slots, padding_slots) != (
                    expected_actual,
                    expected_effective,
                    expected_execution,
                    expected_execution - expected_actual,
                ):
                    raise RuntimeError(f"{sample_id}: short-window budget accounting mismatch at K{budget}")
                if collapsed:
                    collapsed_alias_count += 1
                    if effective != BASELINE_BUDGET:
                        raise RuntimeError("collapsed arm did not canonicalize to K384")
                    continue
                if budget == BASELINE_BUDGET:
                    continue
                representatives.setdefault(
                    (budget, execution_slots),
                    {
                        "dataset_index": dataset_index,
                        "sample_id": sample_id,
                        "positions": prefixes["positions_by_budget"][budget][0].detach().cpu().tolist(),
                        "detector_positions": prefixes["detector_grid_by_budget"][budget][0].detach().cpu().tolist(),
                        "actual_count": actual,
                        "execution_slots": execution_slots,
                    },
                )
    finally:
        # Keep the frozen normalizer active for the bounded representative
        # K384 forwards below. The model is released immediately afterwards.
        pass

    if scanned_windows != len(dataset) or scanned_videos != set(videos):
        raise RuntimeError("PRE_RUN metadata scan excluded training-side windows or videos")
    if baseline_cost_by_video != expected_baseline_cost_by_video:
        raise RuntimeError("PRE_RUN per-video K384 actual cost differs from the raw-mask target")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("PRE_RUN created detector or Scout gradients")

    inference, post = _inference_settings(cfg)
    class_map = list(dataset.class_map)
    real_forward_classes = []
    if short_k384_index is not None:
        data = _single_dataset_batch(dataset, short_k384_index, num_workers=args.num_workers)
        raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(data, device=device)
        _historical_k384_loss(
            model,
            raw_inputs=raw_inputs,
            raw_masks=raw_masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            amp=args.amp,
        )
        real_forward_classes.append("historical_k384_short")

    if full_k384_example is not None:
        data = _single_dataset_batch(
            dataset,
            int(full_k384_example["dataset_index"]),
            num_workers=args.num_workers,
        )
        raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(data, device=device)
        normal = _one_window_predictions(
            model,
            inputs=raw_inputs,
            masks=raw_masks,
            metas=metas,
            class_map=class_map,
            inference=inference,
            post=post,
            amp=args.amp,
            disable_selector=False,
        )
        prepared = _prepare_explicit_window(
            model,
            raw_inputs=raw_inputs,
            raw_masks=raw_masks,
            raw_meta=metas[0],
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            positions=full_k384_example["positions"],
            detector_positions=full_k384_example["detector_positions"],
            actual_count=BASELINE_BUDGET,
            execution_slots=BASELINE_BUDGET,
            baseline_execution=True,
        )
        _loss, explicit = _explicit_loss_and_predictions(
            model,
            prepared=prepared,
            class_map=class_map,
            inference=inference,
            post=post,
            amp=args.amp,
            emit_predictions=True,
        )
        if _canonical_sha256(normal) != _canonical_sha256(explicit):
            raise RuntimeError("PRE_RUN full-window explicit K384 prediction is not historical parity")
        real_forward_classes.append("explicit_k384_full")
    model.rpn_head.duca_set_frozen_loss_normalizer(None)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for (budget, execution_slots), example in sorted(representatives.items()):
        model, _payload, _normalizer = _load_frozen_model(
            cfg,
            checkpoint=checkpoint,
            pretrain=pretrain,
            execution_slots=execution_slots,
            device=device,
        )
        try:
            data = _single_dataset_batch(
                dataset,
                int(example["dataset_index"]),
                num_workers=args.num_workers,
            )
            raw_inputs, raw_masks, metas, gt_segments, gt_labels = _move_batch(data, device=device)
            prepared = _prepare_explicit_window(
                model,
                raw_inputs=raw_inputs,
                raw_masks=raw_masks,
                raw_meta=metas[0],
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                positions=example["positions"],
                detector_positions=example["detector_positions"],
                actual_count=int(example["actual_count"]),
                execution_slots=execution_slots,
            )
            _explicit_loss_and_predictions(
                model,
                prepared=prepared,
                class_map=class_map,
                inference=inference,
                post=post,
                amp=args.amp,
                emit_predictions=False,
            )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("PRE_RUN distinct frozen forward created gradients")
            real_forward_classes.append(f"k{budget}_exec{execution_slots}")
        finally:
            model.rpn_head.duca_set_frozen_loss_normalizer(None)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    receipt = {
        "status": "PRE_RUN_PASS",
        "stage": "pre-run",
        "source": _stage_source(
            repo_root=repo_root,
            config_path=config_path,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            annotation=annotation,
            class_map=class_map_path,
            train_data=train_data,
            pretrain=pretrain,
        ),
        "checkpoint_payload_epoch": int(payload["epoch"]),
        "checkpoint_state_key": "state_dict_ema",
        "frozen_loss_normalizer": frozen_normalizer,
        "fit_video_count": len(fit_videos),
        "holdout_video_count": len(holdout_videos),
        "training_video_count": len(videos),
        "training_window_count": scanned_windows,
        "short_windows_included": short_k384_index is not None,
        "collapsed_alias_count": collapsed_alias_count,
        "real_forward_execution_classes": real_forward_classes,
        "all_k384_video_actual_cost": baseline_cost_by_video,
        "all_k384_video_expected_target": expected_baseline_cost_by_video,
        "all_k384_video_target_exact": True,
        "k384_full_tensor_equal_all_windows": True,
        "utility_head_fit_performed": False,
        "official_evaluator_called": False,
        "official_test_consumed": False,
        "detector_training_performed": False,
        "detector_or_scout_gradients_created": False,
    }
    _write_json(paths["pre_run"], receipt)
    return receipt


def _require_pre_run_pass(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    receipt_path = _stage_paths(output_dir)["pre_run"]
    if not receipt_path.is_file():
        raise RuntimeError("the frozen probe requires PRE_RUN_PASS on the same output root")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PRE_RUN_PASS":
        raise RuntimeError("the frozen probe PRE_RUN receipt is not a pass")
    current = _git_identity(Path(__file__).resolve().parents[2])
    if receipt.get("source", {}).get("git", {}).get("head") != current["head"] or current["dirty"]:
        raise RuntimeError("PRE_RUN_PASS does not bind the current clean Git commit")
    (
        repo_root,
        config_path,
        _cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map_path,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    current_source = _stage_source(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        annotation=annotation,
        class_map=class_map_path,
        train_data=train_data,
        pretrain=pretrain,
    )
    receipt_source = receipt.get("source", {})
    identity_keys = (
        "config",
        "config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
        "annotation",
        "annotation_sha256",
        "class_map",
        "class_map_sha256",
        "train_data",
        "videomae_pretrain",
        "videomae_pretrain_sha256",
    )
    mismatched = [
        key for key in identity_keys if receipt_source.get(key) != current_source.get(key)
    ]
    if mismatched:
        raise RuntimeError(
            "PRE_RUN_PASS does not bind the current frozen inputs: " + ", ".join(mismatched)
        )
    if receipt.get("checkpoint_payload_epoch") != 59 or receipt.get("checkpoint_state_key") != "state_dict_ema":
        raise RuntimeError("PRE_RUN_PASS does not bind epoch-59 state_dict_ema")


def _build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen H65 three-budget counterfactual marginal-utility probe "
            "without detector training or official-test access."
        )
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "pre-run",
            "select-k384",
            "counterfactual-k256",
            "counterfactual-k512",
            "summarize",
            "all",
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            repo_root
            / "configs/adatad/thumos/duca_marginal_frozen_h65_probe.py"
        ),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--annotation")
    parser.add_argument("--class-map")
    parser.add_argument("--train-data")
    parser.add_argument("--pretrain")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--evaluator-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe_stages = (
            (
                "select-k384",
                lambda: run_selection_stage(args),
            ),
            (
                "counterfactual-k256",
                lambda: run_counterfactual_stage(args, budget=256),
            ),
            (
                "counterfactual-k512",
                lambda: run_counterfactual_stage(args, budget=512),
            ),
            (
                "summarize",
                lambda: run_summary_stage(args),
            ),
        )
        if args.stage == "pre-run":
            selected = [("pre-run", lambda: run_pre_run_stage(args))]
        else:
            _require_pre_run_pass(args)
            selected = (
                probe_stages
                if args.stage == "all"
                else [item for item in probe_stages if item[0] == args.stage]
            )
        if not selected:
            raise ValueError(f"unsupported stage {args.stage}")
        final = None
        for name, operation in selected:
            print(f"DUCA_MARGINAL_STAGE_BEGIN {name}", flush=True)
            final = operation()
            print(
                f"DUCA_MARGINAL_STAGE_COMPLETE {name} status={final.get('status')}",
                flush=True,
            )
        print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as exc:
        failure = {
            "status": "PRE_RUN_FAIL" if args.stage == "pre-run" else "DUCA_MARGINAL_PROBE_FAILED",
            "stage": args.stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / f"failure_{args.stage}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
