"""Repository-owned construction for formal paired ChronoTransport Stage C."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from opentad.datasets.builder import collate
from opentad.models import build_detector
from opentad.models.chronotransport.filesystem import (
    load_bound_torch,
    load_registered_python_config,
    read_bound_bytes,
)
from opentad.models.chronotransport.protocol import canonical_sha256
from opentad.models.chronotransport.registration import validate_pre_gate1_registration
from opentad.models.chronotransport.replay import materialized_batch_sha256
from opentad.models.chronotransport.runtime import ChronoTransportRuntime
from opentad.utils import set_seed
from tools.bata.build_chronotransport_r2_manifest import load_manifest_file
from tools.bata.chronotransport_r2_stage_b_factory import (
    RegisteredManifestFitDataset,
    move_batch_to_device,
)


ROOT = Path(os.path.abspath(__file__)).parents[2]
R2_STAGE_C_CONFIG_RELATIVE = (
    "configs/adatad/thumos/c3_chronotransport_r2_stage_c.py"
)
R2_STAGE_C_CONFIG = ROOT / R2_STAGE_C_CONFIG_RELATIVE
_CONFIG_OVERRIDE_ENV = {
    "CHRONOTRANSPORT_MODE",
    "CHRONOTRANSPORT_COST_JSON",
    "CHRONOTRANSPORT_SCHEDULE_COST_JSON",
    "CHRONOTRANSPORT_RISK_READY",
    "CHRONOTRANSPORT_ALLOW_UNMEASURED_DEBUG",
    "CHRONOTRANSPORT_MAX_CACHE_AGE",
    "CHRONOTRANSPORT_RISK_QUANTILE",
    "CHRONOTRANSPORT_RISK_EPSILON",
    "CHRONOTRANSPORT_PROFILE_SYNC_CUDA",
    "CHRONOTRANSPORT_COST_HARDWARE",
    "CHRONOTRANSPORT_COST_PRECISION",
    "CHRONOTRANSPORT_COST_STATISTIC",
}
_STAGE_C_CONFIG_CONTRACT = {
    "protocol_id": "CT-P3R-3S-r2",
    "stage": "C",
    "seeds": (3407, 3408, 3409),
    "successful_updates": 4200,
    "epochs": 60,
    "batch_size": 2,
    "world_size": 1,
    "amp_fp16": True,
    "gradient_accumulation": False,
    "max_overflow_retries": 3,
}


def _runtime(model: torch.nn.Module) -> ChronoTransportRuntime:
    runtimes = [
        module for module in model.modules() if isinstance(module, ChronoTransportRuntime)
    ]
    if len(runtimes) != 1:
        raise RuntimeError("formal Stage C requires exactly one ChronoTransportRuntime")
    return runtimes[0]


def _strip_uniform_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    names = tuple(map(str, state))
    prefixed = tuple(name.startswith("module.") for name in names)
    if names and all(prefixed):
        return {name[7:]: value for name, value in state.items()}
    if any(prefixed):
        raise ValueError("Stage-C checkpoint has a mixed DDP prefix")
    return dict(state)


def _load_exact_model_state(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    *,
    expected_file_sha256: str,
    state_key: str,
    label: str,
) -> dict[str, Any]:
    exact, checkpoint, payload, digest = load_bound_torch(checkpoint_path, label=label)
    if digest != expected_file_sha256 or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label} exact bytes differ from registered provenance")
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get(state_key), Mapping
    ):
        raise ValueError(f"{label} lacks required {state_key}")
    state = _strip_uniform_ddp_prefix(checkpoint[state_key])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{label} does not strictly match the Stage-C model") from error
    return {
        "path": str(exact),
        "bytes": len(payload),
        "sha256": digest,
        "state_key": state_key,
        "top_level_keys_sha256": canonical_sha256(sorted(map(str, checkpoint))),
    }


class ManifestStageCBatchMaterializer:
    """Materialize one canonical ordered pair exactly once per success index."""

    def __init__(
        self,
        dataset: RegisteredManifestFitDataset,
        windows: Sequence[Mapping[str, Any]],
        device: torch.device,
    ) -> None:
        self.dataset = dataset
        self.windows = tuple(dict(row) for row in windows)
        self.device = device
        self.fit_window_ids = tuple(str(row["window_id"]) for row in self.windows)
        if len(self.windows) != 140 or len(set(self.fit_window_ids)) != 140:
            raise ValueError("formal Stage C requires 140 unique ordered fit windows")

    def __call__(self, successful_update: int) -> Mapping[str, Any]:
        if (
            isinstance(successful_update, bool)
            or not isinstance(successful_update, int)
            or not 0 <= successful_update < 4200
        ):
            raise ValueError("formal Stage-C successful update is outside [0, 4199]")
        pair = successful_update % 70
        indices = (2 * pair, 2 * pair + 1)
        samples = [self.dataset[index] for index in indices]
        batch = move_batch_to_device(collate(samples), self.device)
        windows = [self.windows[index] for index in indices]
        batch["video_id"] = [str(row["video_id"]) for row in windows]
        batch["window_id"] = [str(row["window_id"]) for row in windows]
        batch["manifest_window_sha256"] = [
            str(row["window_sha256"]) for row in windows
        ]
        batch["manifest_sampled_frame_indices_sha256"] = [
            canonical_sha256(row["sampled_frame_indices"]) for row in windows
        ]
        batch["augmentation_sha256"] = materialized_batch_sha256(
            {"inputs": batch["inputs"]}
        )
        batch["sample_id"] = list(batch["window_id"])
        batch["split"] = "fit"
        return batch

    def close(self) -> None:
        self.dataset.close()


@dataclass
class RepositoryStageCComponents:
    ct_model: torch.nn.Module
    matched_model: torch.nn.Module
    materialize_batch: ManifestStageCBatchMaterializer
    fit_window_ids: tuple[str, ...]
    manifest: dict[str, Any]
    config_sha256: str
    config_source_identity: dict[str, str]
    stage_b_checkpoint_identity: dict[str, Any]
    dense_checkpoint_identity: dict[str, Any]
    cost_profile_sha256: str

    def close(self) -> None:
        self.materialize_batch.close()


def build_repository_stage_c_components(
    *,
    registration: Mapping[str, Any],
    manifest_path: Path,
    media_registry_path: Path,
    config_identity_path: Path,
    seed: int,
    registration_commit: str,
    registration_relpath: str,
    gate1_unlock: Mapping[str, Any],
    stage_b_phase_marker: Mapping[str, Any],
) -> RepositoryStageCComponents:
    registered = validate_pre_gate1_registration(
        registration,
        repository_root=ROOT,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if seed not in (3407, 3408, 3409) or isinstance(seed, bool):
        raise ValueError("formal Stage-C seed must be 3407, 3408, or 3409")
    if (
        stage_b_phase_marker.get("status") != "PHASE_COMPLETE"
        or stage_b_phase_marker.get("seed") != seed
        or stage_b_phase_marker.get("registration_sha256")
        != registered["registration_sha256"]
    ):
        raise ValueError("Stage-C Stage-B phase marker identity mismatch")
    manifest = load_manifest_file(
        manifest_path,
        registry_path=media_registry_path,
        config_identity_path=config_identity_path,
    )
    if (
        manifest.get("manifest_sha256")
        != registered["window_manifest"]["artifact"]["manifest_sha256"]
        or len(manifest["splits"]["fit"]) != 140
    ):
        raise ValueError("Stage-C manifest differs from immutable registration")
    overrides = sorted(name for name in _CONFIG_OVERRIDE_ENV if name in os.environ)
    if overrides:
        raise RuntimeError(
            f"formal Stage C forbids environment-driven config overrides: {overrides}"
        )
    config_sha256 = read_bound_bytes(
        R2_STAGE_C_CONFIG, label="formal Stage-C config"
    )[2]
    if registered["source_files"].get(R2_STAGE_C_CONFIG_RELATIVE) != config_sha256:
        raise ValueError("formal Stage-C config is not hash-bound by registration")
    cfg, config_sources = load_registered_python_config(
        repository_root=ROOT,
        config_relative=R2_STAGE_C_CONFIG_RELATIVE,
        registered_sources=registered["source_files"],
    )
    actual_contract = dict(cfg.chronotransport_r2)
    actual_contract["seeds"] = tuple(actual_contract["seeds"])
    if actual_contract != _STAGE_C_CONFIG_CONTRACT:
        raise ValueError("formal Stage-C config contract differs from the frozen protocol")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage C requires one protected visible CUDA device")
    device = torch.device("cuda:0")
    cfg.model.backbone.custom.pretrain = None
    set_seed(seed)
    ct_model = build_detector(cfg.model).to(device)
    matched_model = deepcopy(ct_model)
    if any(
        left is right or left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        for left, right in zip(ct_model.parameters(), matched_model.parameters())
    ):
        raise RuntimeError("Stage-C model arms share Parameter objects or storage")

    phase_checkpoint = stage_b_phase_marker.get("trained_checkpoint")
    if not isinstance(phase_checkpoint, Mapping):
        raise ValueError("Stage-C phase marker lacks trained checkpoint identity")
    stage_b_identity = _load_exact_model_state(
        ct_model,
        phase_checkpoint["path"],
        expected_file_sha256=phase_checkpoint["exact_bytes_sha256"],
        state_key="state_dict_ema",
        label="formal Stage-C Stage-B EMA checkpoint",
    )
    dense = registered["dense_checkpoint"]
    dense_state_key = "state_dict_ema" if bool(getattr(cfg.solver, "ema", False)) else "state_dict"
    dense_identity = _load_exact_model_state(
        matched_model,
        dense["content_addressed_path"],
        expected_file_sha256=dense["sha256"],
        state_key=dense_state_key,
        label="formal Stage-C registered dense checkpoint",
    )

    result = gate1_unlock.get("gate1_result")
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise ValueError("formal Stage C requires a PASS Gate-1 result")
    raw_costs = result.get("candidate_cost_p50")
    profile_sha256 = gate1_unlock.get("profile_sha256")
    if not isinstance(raw_costs, Mapping) or not isinstance(profile_sha256, str):
        raise ValueError("formal Stage C requires direct Gate-1 candidate costs")
    for model in (ct_model, matched_model):
        runtime = _runtime(model)
        costs = {name: raw_costs[name] for name in runtime.schedule_library.names}
        runtime.install_registered_candidate_costs(
            costs,
            profile_sha256=profile_sha256,
        )
        runtime.set_checkpoint_loaded(model is ct_model)

    dataset = RegisteredManifestFitDataset(
        cfg,
        manifest,
        registered,
        split="fit",
        augment=True,
    )
    windows_by_id = {str(row["window_id"]): row for row in manifest["windows"]}
    windows = [windows_by_id[str(item)] for item in manifest["splits"]["fit"]]
    materializer = ManifestStageCBatchMaterializer(dataset, windows, device)
    # Model construction/checkpoint loading must not define the augmentation
    # stream.  The formal stream starts here and is checkpointed after each epoch.
    set_seed(seed)
    return RepositoryStageCComponents(
        ct_model=ct_model,
        matched_model=matched_model,
        materialize_batch=materializer,
        fit_window_ids=materializer.fit_window_ids,
        manifest=dict(manifest),
        config_sha256=config_sha256,
        config_source_identity=config_sources,
        stage_b_checkpoint_identity=stage_b_identity,
        dense_checkpoint_identity=dense_identity,
        cost_profile_sha256=profile_sha256,
    )


__all__ = [
    "ManifestStageCBatchMaterializer",
    "R2_STAGE_C_CONFIG_RELATIVE",
    "RepositoryStageCComponents",
    "build_repository_stage_c_components",
]
