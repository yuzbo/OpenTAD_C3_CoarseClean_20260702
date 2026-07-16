#!/usr/bin/env python3
"""Repository-owned, non-substitutable OpenTAD Stage-B construction."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mmengine.config import Config

from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.datasets.base.sliding_dataset import compute_gt_completeness
from opentad.models import build_detector
from opentad.models.chronotransport.formal_stage_b import StageBReplayOutput
from opentad.models.chronotransport.protocol import canonical_sha256
from opentad.models.chronotransport.replay import (
    RNGSnapshot,
    materialized_batch_sha256,
    paired_detector_losses,
    validate_candidate_order_invariance,
)
from opentad.models.chronotransport.registration import validate_pre_gate1_registration
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from opentad.utils import set_seed
from tools.bata.build_chronotransport_r2_manifest import load_manifest_file


ROOT = Path(__file__).resolve().parents[2]
R2_STAGE_B_CONFIG = (
    ROOT / "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py"
)


def move_batch_to_device(value: Any, device: torch.device) -> Any:
    """Move a nested formal batch without importing the superseded v1 factory."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {
            key: move_batch_to_device(item, device) for key, item in value.items()
        }
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime(model: torch.nn.Module):
    runtimes = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "ChronoTransportRuntime"
    ]
    if len(runtimes) != 1:
        raise RuntimeError("formal r2 Stage B requires exactly one ChronoTransportRuntime")
    return runtimes[0]


def _candidate_rows(registration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["name"]): row for row in registration["candidate_library"]["candidates"]
    }


def _model_forward_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "video_id",
        "window_id",
        "manifest_window_sha256",
        "manifest_sampled_frame_indices_sha256",
        "augmentation_sha256",
        "sample_id",
        "split",
    }
    forward = {key: value for key, value in batch.items() if key not in ignored}
    forward["return_loss"] = True
    return forward


def sealed_stage_b_replay(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    schedule: str,
    *,
    registration: Mapping[str, Any],
    track_grad: bool,
) -> StageBReplayOutput:
    """The only replay callable reachable from the formal CLI."""

    if schedule not in R2_NON_DENSE_NAMES:
        raise ValueError("formal Stage-B replay requires one frozen non-dense schedule")
    candidates = _candidate_rows(registration)
    if schedule not in candidates:
        raise ValueError("registered candidate library is missing the requested schedule")
    forward = _model_forward_batch(batch)
    floating_inputs = [
        tensor
        for tensor in _walk_tensors(forward)
        if tensor.is_floating_point()
    ]
    if any(tensor.dtype != torch.float32 for tensor in floating_inputs):
        raise RuntimeError("formal Stage-B OpenTAD forward requires FP32 tensors")
    augmentation_sha256 = materialized_batch_sha256({"inputs": forward["inputs"]})
    device = forward["inputs"].device
    context = nullcontext() if track_grad else torch.no_grad()
    with torch.autocast(device_type=device.type, enabled=False), context:
        replay = paired_detector_losses(
            model,
            forward,
            counterfactual_schedule=schedule,
            track_counterfactual_grad=track_grad,
            augmentation_sha256=augmentation_sha256,
        )
        runtime = _runtime(model)
        executed = runtime.latest_schedule
        summary = runtime.latest_summary
        signals = runtime.latest_signals
        if executed is None or signals is None or not isinstance(summary, Mapping):
            raise RuntimeError("formal replay is missing executed schedule/signals/summary")
        if tuple(executed.actions.shape) != (1, 48, 3):
            raise RuntimeError("formal replay executed action tensor shape mismatch")
        registered_action_sha256 = str(candidates[schedule]["action_sha256"])
        actual_action_sha256 = canonical_sha256(
            executed.actions[0].detach().cpu().to(torch.long).tolist()
        )
        if actual_action_sha256 != registered_action_sha256:
            raise RuntimeError("executed action tensor does not match the registered candidate")
        if (
            summary.get("executed_schedule_name") != schedule
            or summary.get("selected_schedule_names") != [schedule]
            or summary.get("evidence_valid") is not True
            or summary.get("requested_action_sha256")
            != summary.get("executed_action_sha256")
        ):
            raise RuntimeError("formal replay requested/executed schedule evidence is invalid")
        predicted = runtime.risk_predictor(
            signals.float(), executed.actions.unsqueeze(1)
        ).squeeze(1)
        target = replay.regret.detach().reshape(1).expand_as(predicted)
    if replay.dense_features is None or replay.counterfactual_features is None:
        raise RuntimeError("formal replay requires dense and counterfactual runtime features")
    return StageBReplayOutput(
        counterfactual_task_loss=replay.counterfactual_total,
        counterfactual_features=replay.counterfactual_features,
        dense_features=replay.dense_features,
        predicted_quantile=predicted,
        regret_target=target,
        materialized_window_sha256=str(replay.materialized_window_sha256),
        counterfactual_window_sha256=str(replay.counterfactual_window_sha256),
        augmentation_sha256=augmentation_sha256,
        requested_action_sha256=registered_action_sha256,
        executed_action_sha256=actual_action_sha256,
        amp_skipped=False,
    )


def _walk_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


class ManifestFitBatchSequence(Sequence[Mapping[str, Any]]):
    def __init__(self, dataset, windows: Sequence[Mapping[str, Any]], device: torch.device):
        self.dataset = dataset
        self.windows = tuple(dict(row) for row in windows)
        self.device = device
        self.window_ids = tuple(str(row["window_id"]) for row in self.windows)
        self.video_ids = tuple(str(row["video_id"]) for row in self.windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        sample = self.dataset[int(index)]
        batch = move_batch_to_device(collate([sample]), self.device)
        window = self.windows[int(index)]
        batch["video_id"] = str(window["video_id"])
        batch["window_id"] = str(window["window_id"])
        batch["manifest_window_sha256"] = str(window["window_sha256"])
        batch["manifest_sampled_frame_indices_sha256"] = canonical_sha256(
            window["sampled_frame_indices"]
        )
        batch["augmentation_sha256"] = materialized_batch_sha256(
            {"inputs": batch["inputs"]}
        )
        return batch


def _manifest_dataset(cfg: Config, manifest: Mapping[str, Any]):
    dataset_cfg = deepcopy(cfg.dataset.val)
    dataset_cfg.ioa_thresh = 0.0
    dataset_cfg.window_size = 768
    train_pipeline = deepcopy(cfg.dataset.train.pipeline)
    for transform in train_pipeline:
        if str(transform.get("type")) == "LoadFrames":
            for key in ("trunc_len", "trunc_thresh", "crop_ratio"):
                transform.pop(key, None)
            transform["method"] = "sliding_window"
    dataset_cfg.pipeline = train_pipeline
    dataset = build_dataset(dataset_cfg)
    source_by_video = {}
    for row in dataset.data_list:
        source_by_video.setdefault(str(row[0]), row)
    ordered_rows = []
    manifest_windows = {str(row["window_id"]): row for row in manifest["windows"]}
    fit_windows = [manifest_windows[window_id] for window_id in manifest["splits"]["fit"]]
    for window in fit_windows:
        video_id = str(window["video_id"])
        if video_id not in source_by_video:
            raise ValueError(f"manifest video is absent from the OpenTAD dataset: {video_id}")
        _, video_info, full_anno, _ = source_by_video[video_id]
        valid_mask = window["valid_mask"]
        valid_count = sum(value is True for value in valid_mask)
        if valid_count <= 0 or valid_mask != [True] * valid_count + [False] * (768 - valid_count):
            raise ValueError("manifest valid mask must be one true prefix followed by padding")
        sampled = np.asarray(window["sampled_frame_indices"], dtype=np.int64)
        centers = sampled[:valid_count]
        if centers.size == 0 or np.any(np.diff(centers) <= 0):
            raise ValueError("manifest valid sampled frame indices must be strictly increasing")
        annotation = deepcopy(full_anno)
        if annotation and len(annotation.get("gt_segments", ())) > 0:
            completeness, truncated = compute_gt_completeness(
                annotation["gt_segments"], np.asarray([centers[0], centers[-1]])
            )
            keep = completeness > 0.75
            annotation = {
                "gt_segments": truncated[keep].astype(np.float32),
                "gt_labels": annotation["gt_labels"][keep].astype(np.int32),
            }
        ordered_rows.append([video_id, video_info, annotation, centers])
    dataset.data_list = ordered_rows
    return dataset, fit_windows


@dataclass
class RepositoryStageBComponents:
    model: torch.nn.Module
    batches: ManifestFitBatchSequence
    registration: dict[str, Any]
    manifest: dict[str, Any]
    exposure_artifact: dict[str, Any]
    config_sha256: str
    dense_checkpoint_use_ema: bool

    def replay_step(self, model, batch, schedule):
        return sealed_stage_b_replay(
            model,
            batch,
            schedule,
            registration=self.registration,
            track_grad=True,
        )

    def candidate_order_probe(self) -> None:
        outer_rng = RNGSnapshot.capture()
        try:
            batch = self.batches[0]
            canonical = {}
            for name in R2_NON_DENSE_NAMES:
                canonical[name] = float(
                    sealed_stage_b_replay(
                        self.model,
                        batch,
                        name,
                        registration=self.registration,
                        track_grad=False,
                    ).regret_target.flatten()[0].cpu()
                )
            permuted = {}
            for name in reversed(R2_NON_DENSE_NAMES):
                permuted[name] = float(
                    sealed_stage_b_replay(
                        self.model,
                        batch,
                        name,
                        registration=self.registration,
                        track_grad=False,
                    ).regret_target.flatten()[0].cpu()
                )
            validate_candidate_order_invariance(canonical, permuted)
        finally:
            outer_rng.restore()

    def fit_baseline_rows(self) -> list[dict[str, Any]]:
        rows = []
        for batch in self.batches:
            for candidate_index, schedule in enumerate(R2_NON_DENSE_NAMES):
                replay = sealed_stage_b_replay(
                    self.model,
                    batch,
                    schedule,
                    registration=self.registration,
                    track_grad=False,
                )
                rows.append(
                    {
                        "seed": int(self.registration.get("active_stage_b_seed", -1)),
                        "window_id": str(batch["window_id"]),
                        "candidate_index": candidate_index,
                        "schedule": schedule,
                        "regret": float(replay.regret_target.flatten()[0].cpu()),
                        "materialized_window_sha256": replay.materialized_window_sha256,
                        "augmentation_sha256": replay.augmentation_sha256,
                        "requested_action_sha256": replay.requested_action_sha256,
                        "executed_action_sha256": replay.executed_action_sha256,
                    }
                )
        return rows


def _validate_registered_stage_b_inputs(
    *,
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    exposure_artifact: Mapping[str, Any],
) -> None:
    """Bind the live Stage-B artifacts to the validated nested registration.

    The registration schema deliberately embeds the canonical manifest and
    exposure artifacts.  Reading historical flat digest aliases here would
    make the formal CLI statically unreachable after successful registration.
    """

    try:
        registered_manifest = registration["window_manifest"]["artifact"]
        registered_exposure = registration["exposures"]["stage_b"]
        expected_manifest_sha256 = registered_manifest["manifest_sha256"]
        expected_exposure_sha256 = registered_exposure["artifact_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "formal Stage B requires nested registered Stage-B artifacts"
        ) from error
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("manifest does not match the immutable registration")
    if exposure_artifact.get("artifact_sha256") != expected_exposure_sha256:
        raise ValueError(
            "Stage-B exposure artifact does not match the immutable registration"
        )


def build_repository_stage_b_components(
    *,
    registration: Mapping[str, Any],
    manifest_path: Path,
    media_registry_path: Path,
    config_identity_path: Path,
    exposure_artifact: Mapping[str, Any],
    seed: int,
    registration_commit: str,
    registration_relpath: str,
) -> RepositoryStageBComponents:
    registered = validate_pre_gate1_registration(
        registration,
        repository_root=ROOT,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    manifest = load_manifest_file(
        manifest_path,
        registry_path=media_registry_path,
        config_identity_path=config_identity_path,
    )
    _validate_registered_stage_b_inputs(
        registration=registered,
        manifest=manifest,
        exposure_artifact=exposure_artifact,
    )
    config_sha256 = _file_sha256(R2_STAGE_B_CONFIG)
    matching_sources = [
        digest
        for name, digest in registered["source_files"].items()
        if str(name).replace("\\", "/").endswith(
            "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py"
        )
    ]
    if matching_sources != [config_sha256]:
        raise ValueError("fixed r2 Stage-B config is not hash-bound by the registration")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage B requires one protected visible CUDA device")
    cfg = Config.fromfile(str(R2_STAGE_B_CONFIG))
    cfg.model.backbone.custom.pretrain = None
    model = build_detector(cfg.model).to(torch.device("cuda:0"))
    runtime = _runtime(model)
    runtime.capture_replay_signals = True
    dataset, fit_windows = _manifest_dataset(cfg, manifest)
    set_seed(seed)
    batches = ManifestFitBatchSequence(dataset, fit_windows, torch.device("cuda:0"))
    registered_with_seed = dict(registered)
    registered_with_seed["active_stage_b_seed"] = seed
    return RepositoryStageBComponents(
        model=model,
        batches=batches,
        registration=registered_with_seed,
        manifest=dict(manifest),
        exposure_artifact=dict(exposure_artifact),
        config_sha256=config_sha256,
        dense_checkpoint_use_ema=bool(getattr(cfg.solver, "ema", False)),
    )
