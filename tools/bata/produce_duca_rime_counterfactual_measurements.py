from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.bata.create_duca_rime_splits import (
    TRAIN_ROLES,
    validate_rime_splits,
)


MEASUREMENT_SCHEMA = "duca_rime_counterfactual_measurement_v1"
SUMMARY_SCHEMA = "duca_rime_counterfactual_measurement_producer_v1"
FEATURE_SCHEMA = "duca_rime_cheap_rgb_statistics_v1"
UTILITY_METRIC = "kmax_detector_loss_minus_candidate_detector_loss"
FRAME_UTILITY_METRIC = "uniform_detector_loss_minus_legal_single_swap_loss"
DEFAULT_BUDGETS = (192, 256, 384, 512)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"counterfactual measurements never overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count <= 0:
            raise ValueError("counterfactual measurement output is empty")
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": count,
    }


def cheap_frame_features(inputs: Any, valid_len: int) -> list[list[float]]:
    """Return deterministic label-free per-frame statistics from dense RGB."""

    import torch

    if not torch.is_tensor(inputs) or int(inputs.shape[0]) != 1:
        raise ValueError("cheap feature extraction requires one tensor sample")
    if inputs.ndim not in {3, 5, 6}:
        raise ValueError("cheap RGB tensor must be [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")
    temporal_dim = 2 if inputs.ndim in {3, 5} else 3
    temporal_len = int(inputs.shape[temporal_dim])
    valid_len = int(valid_len)
    if not 0 < valid_len <= temporal_len:
        raise ValueError("cheap feature valid length is invalid")
    order = [0, temporal_dim] + [
        index for index in range(inputs.ndim) if index not in {0, temporal_dim}
    ]
    frames = inputs.permute(order)[0, :valid_len].detach().float().reshape(valid_len, -1)
    if inputs.dtype == torch.uint8:
        frames = frames / 255.0
    level = frames.mean(dim=1)
    spread = frames.std(dim=1, unbiased=False)
    motion_prev = torch.zeros_like(level)
    if valid_len > 1:
        motion_prev[1:] = (frames[1:] - frames[:-1]).abs().mean(dim=1)
    motion_next = torch.zeros_like(level)
    if valid_len > 1:
        motion_next[:-1] = motion_prev[1:]
    curvature = (motion_next - motion_prev).abs()
    relative = torch.linspace(
        0.0,
        1.0,
        valid_len,
        device=frames.device,
        dtype=torch.float32,
    )
    features = torch.stack(
        (
            level,
            spread,
            motion_prev,
            motion_next,
            curvature,
            relative,
            torch.sin(relative * math.pi),
            torch.cos(relative * math.pi),
        ),
        dim=1,
    )
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("cheap frame features became non-finite")
    return [
        [round(float(value), 8) for value in row]
        for row in features.cpu().tolist()
    ]


def budget_features(
    frame_features: Sequence[Sequence[float]],
    *,
    budget: int,
    maximum_budget: int,
) -> list[float]:
    if not frame_features:
        raise ValueError("budget features require dense cheap features")
    width = len(frame_features[0])
    if width <= 0 or any(len(row) != width for row in frame_features):
        raise ValueError("cheap frame feature width drift")
    columns = list(zip(*frame_features))
    output = [
        float(budget) / float(maximum_budget),
        float(budget) / float(len(frame_features)),
        float(len(frame_features) - budget) / float(len(frame_features)),
    ]
    output.extend(statistics.fmean(map(float, column)) for column in columns)
    output.extend(
        statistics.pstdev(map(float, column)) if len(column) > 1 else 0.0
        for column in columns
    )
    if not all(math.isfinite(value) for value in output):
        raise RuntimeError("budget features became non-finite")
    return [round(float(value), 8) for value in output]


def observed_pair_failures(
    detector_losses: Sequence[float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> list[int]:
    losses = [float(value) for value in detector_losses]
    if (
        len(losses) < 2
        or not all(math.isfinite(value) and value >= 0.0 for value in losses)
        or not math.isfinite(float(relative_tolerance))
        or float(relative_tolerance) < 0.0
        or not math.isfinite(float(absolute_tolerance))
        or float(absolute_tolerance) < 0.0
    ):
        raise ValueError("pair-risk inputs are invalid")
    output = []
    for left, right in zip(losses[:-1], losses[1:]):
        tolerance = max(
            float(absolute_tolerance),
            float(relative_tolerance) * max(abs(right), 1.0e-12),
        )
        output.append(int(left > right + tolerance))
    output.append(0)
    return output


def legal_local_swaps(
    baseline_positions: Sequence[int],
    physical_seconds: Sequence[float],
    *,
    max_gap_seconds: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Enumerate deterministic legal ±1 substitutions around a uniform path."""

    baseline = tuple(int(value) for value in baseline_positions)
    seconds = tuple(float(value) for value in physical_seconds)
    if (
        not baseline
        or baseline != tuple(sorted(set(baseline)))
        or not seconds
        or baseline[0] < 0
        or baseline[-1] >= len(seconds)
        or any(
            not math.isfinite(value)
            for value in (*seconds, float(max_gap_seconds))
        )
        or any(right <= left for left, right in zip(seconds[:-1], seconds[1:]))
        or int(max_candidates) <= 0
    ):
        raise ValueError("local-swap inputs are invalid")
    baseline_set = set(baseline)
    anchor_order = sorted(
        range(len(baseline)),
        key=lambda index: (abs(index - 0.5 * (len(baseline) - 1)), index),
    )
    candidates = []
    seen_added = set()
    for anchor_index in anchor_order:
        removed = baseline[anchor_index]
        for delta in (-1, 1):
            added = removed + delta
            if (
                added < 0
                or added >= len(seconds)
                or added in baseline_set
                or added in seen_added
            ):
                continue
            positions = sorted((baseline_set - {removed}) | {added})
            intervals = [
                seconds[positions[0]] - seconds[0],
                seconds[-1] - seconds[positions[-1]],
                *(
                    seconds[right] - seconds[left]
                    for left, right in zip(positions[:-1], positions[1:])
                ),
            ]
            observed = max(intervals)
            tolerance = max(1.0e-9, 8.0 * math.ulp(max(abs(observed), 1.0)))
            if observed > float(max_gap_seconds) + tolerance:
                continue
            candidates.append(
                {
                    "added_position": int(added),
                    "removed_position": int(removed),
                    "selected_positions": positions,
                    "observed_max_gap_seconds": float(observed),
                }
            )
            seen_added.add(added)
            if len(candidates) >= int(max_candidates):
                return candidates
    return candidates


def _sample_identity(meta: Mapping[str, Any]) -> tuple[str, int]:
    video = str(meta.get("video_name") or meta.get("video_id") or "")
    start = int(meta.get("window_start_frame", -1))
    if not video or start < 0:
        raise ValueError("counterfactual sample metadata lacks video/window identity")
    return video, start


def _load_ema(model: Any, checkpoint_path: Path) -> tuple[dict[str, Any], str]:
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("counterfactual checkpoint must be a mapping")
    state = checkpoint.get("state_dict_ema")
    if not isinstance(state, Mapping):
        raise ValueError("counterfactual checkpoint requires state_dict_ema")
    normalized = {
        (
            str(key)[len("module.") :]
            if str(key).startswith("module.")
            else str(key)
        ): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "counterfactual checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return dict(checkpoint), "state_dict_ema"


def _detector_objective(
    model: Any,
    materialized: Mapping[str, Any],
    *,
    gt_segments: Sequence[Any],
    gt_labels: Sequence[Any],
    use_amp: bool,
) -> float:
    import torch

    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(use_amp),
        cache_enabled=False,
    ):
        losses = model.forward_train(
            materialized["inputs"],
            materialized["masks"],
            materialized["metas"],
            gt_segments,
            gt_labels,
            _duca_skip_frame_selector=True,
            _duca_counterfactual_eval=True,
        )
        objective = model._duca_detector_objective(losses)
    value = float(objective.detach().float().cpu().item())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("counterfactual detector objective is invalid")
    return value


def produce_measurements(
    *,
    config: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    output_jsonl: str | Path,
    summary_json: str | Path,
    candidate_budgets: Sequence[int] = DEFAULT_BUDGETS,
    frame_swap_budget: int = 384,
    max_frame_counterfactuals: int = 16,
    relative_pair_tolerance: float = 0.05,
    absolute_pair_tolerance: float = 0.01,
    seed: int = 3407,
    device: str = "cuda:0",
    num_workers: int = 2,
    backbone_pretrain: str | Path | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.duca.rime import decode_rime_exact_k

    budgets = tuple(int(value) for value in candidate_budgets)
    if (
        len(budgets) < 3
        or tuple(sorted(set(budgets))) != budgets
        or any(value <= 0 or value % 16 for value in budgets)
        or int(frame_swap_budget) not in budgets
        or int(max_frame_counterfactuals) < 3
    ):
        raise ValueError("counterfactual budget/candidate contract is invalid")
    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    split_path = Path(split_manifest).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    summary_path = Path(summary_json).expanduser().resolve()
    for target in (output_path, summary_path):
        if target.exists():
            raise FileExistsError(target)
    if _sha256_file(checkpoint_path) != str(checkpoint_sha256):
        raise ValueError("counterfactual checkpoint SHA-256 drift")
    split_validation = validate_rime_splits(
        split_path,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    role_by_video = {
        str(video): role
        for role in TRAIN_ROLES
        for video in split["train_roles"][role]["videos"]
    }
    all_train_videos = set(role_by_video)
    if len(all_train_videos) != int(split["train_video_count"]):
        raise RuntimeError("counterfactual split role union drift")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(str(config_path))
    if (
        cfg.workflow.formal_protocol
        != "duca_rime_phase2_mixed_k_baseline_v1"
        or cfg.duca_rime_variant.arm != "U-mixed-K"
        or tuple(int(value) for value in cfg.duca_rime_variant.candidate_budgets)
        != budgets
        or cfg.model.frame_selector.rime_arm != "uniform_mixed_k"
        or cfg.model.frame_selector.actionness_source_cfg is not None
    ):
        raise ValueError("counterfactual producer requires the frozen U-mixed-K model")
    if backbone_pretrain is not None:
        pretrain = Path(backbone_pretrain).expanduser().resolve()
        if not pretrain.is_file():
            raise FileNotFoundError(pretrain)
        cfg.model.backbone.custom.pretrain = str(pretrain)
    else:
        pretrain = Path(str(cfg.model.backbone.custom.pretrain)).expanduser().resolve()
    if not pretrain.is_file():
        raise FileNotFoundError(f"counterfactual VideoMAE initialization missing: {pretrain}")

    dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    if callable(getattr(dataset, "set_epoch", None)):
        dataset.set_epoch(0)
    dataset_videos = [str(row[0]) for row in getattr(dataset, "data_list", ())]
    if (
        len(dataset_videos) != len(set(dataset_videos))
        or set(dataset_videos) != all_train_videos
    ):
        raise RuntimeError(
            "counterfactual dataset must expose every train-role video exactly once"
        )
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=int(num_workers),
    )
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal detector counterfactual production requires CUDA")
    torch.cuda.set_device(torch_device)
    model = build_detector(cfg.model).to(torch_device)
    checkpoint_payload, state_key = _load_ema(model, checkpoint_path)
    model.eval()
    selector = getattr(model, "frame_selector", None)
    if (
        selector is None
        or getattr(selector, "selector_variant", None) != "duca_rime_physical"
        or not callable(getattr(selector, "materialize_counterfactual_positions", None))
    ):
        raise RuntimeError("counterfactual model lacks the RIME materialization contract")
    frozen_normalizer = None
    if callable(getattr(model.rpn_head, "duca_set_frozen_loss_normalizer", None)):
        frozen_normalizer = model.rpn_head.loss_normalizer.detach().clone()
        model.rpn_head.duca_set_frozen_loss_normalizer(frozen_normalizer)

    source_identity = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_validation["manifest_sha256"],
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "backbone_pretrain_path": str(pretrain),
        "backbone_pretrain_sha256": _sha256_file(pretrain),
        "candidate_budgets": list(budgets),
        "frame_swap_budget": int(frame_swap_budget),
        "max_frame_counterfactuals": int(max_frame_counterfactuals),
        "feature_schema": FEATURE_SCHEMA,
        "utility_metric": UTILITY_METRIC,
        "frame_utility_metric": FRAME_UTILITY_METRIC,
        "seed": int(seed),
        "stateless_dataset_epoch": 0,
    }
    source_artifact_sha256 = _canonical_sha256(source_identity)
    rows = []
    identities = set()
    role_counts: Counter[str] = Counter()
    try:
        for data in loader:
            inputs = data["inputs"].to(torch_device, non_blocking=True)
            masks = data["masks"].to(torch_device, non_blocking=True)
            metas = [dict(value) for value in data["metas"]]
            gt_segments = [
                value.to(torch_device, non_blocking=True)
                for value in data["gt_segments"]
            ]
            gt_labels = [
                value.to(torch_device, non_blocking=True)
                for value in data["gt_labels"]
            ]
            if int(inputs.shape[0]) != 1 or len(metas) != 1:
                raise RuntimeError("counterfactual production requires batch_size=1")
            video, window_start = _sample_identity(metas[0])
            identity = (video, window_start)
            if identity in identities or video not in role_by_video:
                raise RuntimeError("counterfactual dataset identity is duplicate or unregistered")
            valid = masks.to(dtype=torch.bool)
            valid_len = int(valid[0].sum().item())
            if valid_len < budgets[-1]:
                raise RuntimeError(
                    f"{video} cannot realize the frozen maximum K without padding"
                )
            features = cheap_frame_features(inputs, valid_len)
            physical_seconds, _source_frames = selector._physical_axes(
                metas,
                valid,
                torch_device,
            )
            zero_potential = torch.zeros_like(physical_seconds, dtype=torch.float32)
            losses = []
            ledgers = []
            uniform_positions: dict[int, list[int]] = {}
            gap_caps: dict[int, float] = {}
            for budget in budgets:
                decoded = decode_rime_exact_k(
                    zero_potential,
                    physical_seconds,
                    valid,
                    budget,
                    candidate_budgets=budgets,
                    decoder_family="independent",
                    training=False,
                    force_uniform=True,
                    require_homogeneous_execution=True,
                    execution_quantum=16,
                )
                decoded.ledger.validate(require_no_padding=True)
                positions = [
                    int(value)
                    for value in decoded.hard_positions[0, :budget].cpu().tolist()
                ]
                materialized = selector.materialize_counterfactual_positions(
                    inputs,
                    valid,
                    metas,
                    decoded.hard_positions[:, :budget],
                    requested_k=budget,
                )
                loss = _detector_objective(
                    model,
                    materialized,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    use_amp=bool(use_amp),
                )
                losses.append(loss)
                uniform_positions[budget] = positions
                gap_caps[budget] = float(decoded.max_gap_seconds[0].item())
                ledgers.append(
                    {
                        "requested_k": budget,
                        "effective_k": budget,
                        "unique_k": budget,
                        "backbone_input_k": budget,
                        "padded_k": budget,
                        "max_gap_seconds_cap": gap_caps[budget],
                        "max_gap_violation": False,
                    }
                )

            baseline_index = budgets.index(int(frame_swap_budget))
            baseline_loss = losses[baseline_index]
            seconds = [
                float(value)
                for value in physical_seconds[0, :valid_len].cpu().tolist()
            ]
            swaps = legal_local_swaps(
                uniform_positions[int(frame_swap_budget)],
                seconds,
                max_gap_seconds=gap_caps[int(frame_swap_budget)],
                max_candidates=int(max_frame_counterfactuals),
            )
            if len(swaps) < 3:
                raise RuntimeError(
                    f"{video} produced fewer than three legal detector counterfactuals"
                )
            frame_fit_features = []
            frame_utility = []
            counterfactual_rows = []
            for swap in swaps:
                swap_positions = torch.as_tensor(
                    [swap["selected_positions"]],
                    device=torch_device,
                    dtype=torch.long,
                )
                materialized = selector.materialize_counterfactual_positions(
                    inputs,
                    valid,
                    metas,
                    swap_positions,
                    requested_k=int(frame_swap_budget),
                )
                candidate_loss = _detector_objective(
                    model,
                    materialized,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    use_amp=bool(use_amp),
                )
                utility = baseline_loss - candidate_loss
                added = int(swap["added_position"])
                frame_fit_features.append(features[added])
                frame_utility.append(float(utility))
                counterfactual_rows.append(
                    {
                        "added_position": added,
                        "removed_position": int(swap["removed_position"]),
                        "baseline_detector_loss": float(baseline_loss),
                        "candidate_detector_loss": float(candidate_loss),
                        "utility": float(utility),
                        "observed_max_gap_seconds": float(
                            swap["observed_max_gap_seconds"]
                        ),
                        "max_gap_seconds_cap": gap_caps[int(frame_swap_budget)],
                    }
                )

            kmax_loss = losses[-1]
            actual_utility = [float(kmax_loss - value) for value in losses]
            provenance = {
                "measurement_kind": "measured_detector_counterfactual",
                "fit_split": "train_only",
                "uses_official_final": False,
                "uses_gt_for_supervision": True,
                "uses_gt_at_deployment": False,
                "uses_teacher_at_deployment": False,
                "uses_prediction_cache_at_deployment": False,
                "cheap_features_only_at_deployment": True,
                "counterfactual_utility": True,
                "proposal_score_surrogate_utility": False,
                "pad_to_kmax": False,
                "detector_checkpoint_sha256": str(checkpoint_sha256),
                "source_artifact_sha256": source_artifact_sha256,
                "utility_metric": UTILITY_METRIC,
                "frame_utility_metric": FRAME_UTILITY_METRIC,
                "feature_schema": FEATURE_SCHEMA,
                "detector_objective": "official_actionformer_cls_plus_reg",
                "model_training": False,
                "checkpoint_mutation": False,
            }
            row = {
                "schema_version": MEASUREMENT_SCHEMA,
                "video_id": video,
                "window_start_frame": window_start,
                "split_role": role_by_video[video],
                "split_assignment_sha256": split_validation["assignment_sha256"],
                "candidate_budgets": list(budgets),
                "budget_features": [
                    budget_features(
                        features,
                        budget=budget,
                        maximum_budget=budgets[-1],
                    )
                    for budget in budgets
                ],
                "actual_utility": actual_utility,
                "observed_pair_failure": observed_pair_failures(
                    losses,
                    relative_tolerance=float(relative_pair_tolerance),
                    absolute_tolerance=float(absolute_pair_tolerance),
                ),
                "frame_features": features,
                "frame_fit_features": frame_fit_features,
                "actual_frame_utility": frame_utility,
                "frame_counterfactuals": counterfactual_rows,
                "detector_losses": [float(value) for value in losses],
                "uniform_selected_positions": {
                    str(budget): uniform_positions[budget] for budget in budgets
                },
                "cost_ledger": ledgers,
                "provenance": provenance,
            }
            row["record_sha256"] = _canonical_sha256(row)
            rows.append(row)
            identities.add(identity)
            role_counts[role_by_video[video]] += 1
            print(
                json.dumps(
                    {
                        "counterfactual_video": video,
                        "role": role_by_video[video],
                        "completed": len(rows),
                        "total": len(dataset_videos),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if frozen_normalizer is not None:
            model.rpn_head.duca_set_frozen_loss_normalizer(None)

    if {video for video, _start in identities} != all_train_videos:
        missing = sorted(all_train_videos - {video for video, _start in identities})
        raise RuntimeError(
            f"counterfactual production did not cover every train video: {missing[:5]}"
        )
    expected_role_counts = {
        role: len(split["train_roles"][role]["videos"])
        for role in TRAIN_ROLES
    }
    if dict(role_counts) != expected_role_counts:
        raise RuntimeError("counterfactual role coverage differs from the frozen split")
    output_artifact = _write_jsonl_atomic(output_path, rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "produced",
        "claim_scope": (
            "train_only_measured_detector_loss_counterfactuals_"
            "not_tad_map_not_gate_result"
        ),
        "source_identity": source_identity,
        "source_artifact_sha256": source_artifact_sha256,
        "output": output_artifact,
        "role_counts": expected_role_counts,
        "candidate_budgets": list(budgets),
        "frame_swap_budget": int(frame_swap_budget),
        "max_frame_counterfactuals": int(max_frame_counterfactuals),
        "relative_pair_tolerance": float(relative_pair_tolerance),
        "absolute_pair_tolerance": float(absolute_pair_tolerance),
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "official_final_subset_consumed": False,
        "uses_gt_at_deployment": False,
        "uses_teacher_at_deployment": False,
        "uses_prediction_cache_at_deployment": False,
        "proposal_score_surrogate_utility": False,
    }
    summary["content_sha256"] = _canonical_sha256(summary)
    _write_json_exclusive(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = _sha256_file(summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure train-only exact-K ActionFormer detector-loss "
            "counterfactuals for DUCA-RIME O2/O3/O4 supervision."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--candidate-budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--frame-swap-budget", type=int, default=384)
    parser.add_argument("--max-frame-counterfactuals", type=int, default=16)
    parser.add_argument("--relative-pair-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-pair-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-pretrain")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    result = produce_measurements(
        config=args.config,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        candidate_budgets=args.candidate_budgets,
        frame_swap_budget=args.frame_swap_budget,
        max_frame_counterfactuals=args.max_frame_counterfactuals,
        relative_pair_tolerance=args.relative_pair_tolerance,
        absolute_pair_tolerance=args.absolute_pair_tolerance,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        backbone_pretrain=args.backbone_pretrain,
        use_amp=not args.no_amp,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
