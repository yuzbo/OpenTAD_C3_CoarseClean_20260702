#!/usr/bin/env python3
"""Fixed, registration-owned OpenTAD backend for formal r2 profiling.

There is intentionally no import-string or caller-supplied construction hook in
this module.  The resolved config, detector class, checkpoint, manifest, media
windows, action bytes, and post-processing path are all selected from the
validated registration and repository-owned constants.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose

import opentad.datasets  # noqa: F401 - register OpenTAD and mmaction transforms
from opentad.datasets.builder import collate
from opentad.models import build_detector
from opentad.models.chronotransport.actions import ChronoSchedule, LayerGroup
from opentad.models.chronotransport.controls import (
    motion_topk_actions,
    random_exact_count_actions,
)
from opentad.models.chronotransport.profiler import REQUIRED_STAGE_FIELDS
from opentad.models.chronotransport.protocol import (
    canonical_json_bytes,
    canonical_sha256,
    manifest_exact_bytes,
    validate_r2_manifest,
)
from opentad.models.chronotransport.registration import (
    REGISTERED_PROFILE_BACKEND_IDENTITY,
    REGISTERED_PROFILE_BACKEND_SOURCE,
    REQUIRED_REGISTRATION_SOURCE_PATHS,
    validate_checkpoint_registry_receipt,
    validate_pre_gate1_registration,
)


ROOT = Path(__file__).resolve().parents[2]
R2_PROFILE_CONFIG_RELATIVE = (
    "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py"
)
R2_PROFILE_CONFIG = ROOT / R2_PROFILE_CONFIG_RELATIVE
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


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def observed_registered_environment() -> dict[str, object]:
    """Read the actual one-GPU execution identity used by profile and replay."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or "," not in rows[0]:
        raise RuntimeError("formal backend requires one observable GPU UUID/driver row")
    gpu_uuid, driver = (item.strip() for item in rows[0].split(",", 1))
    environment: dict[str, object] = {
        "gpu_model": torch.cuda.get_device_name(0),
        "gpu_uuid": gpu_uuid,
        "driver": driver,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "cudnn": str(torch.backends.cudnn.version()),
        "precision": "amp_fp16",
        "batch_size": 1,
    }
    environment["environment_sha256"] = canonical_sha256(environment)
    return environment


def preverify_registered_media(
    registration: Mapping[str, object],
    windows: object,
) -> dict[str, Path]:
    """Hash every registered media object before any warmup or timed call."""

    if not isinstance(windows, list) or len(windows) != 200:
        raise RuntimeError("formal profile requires exactly 200 media windows")
    root = Path(registration["data"]["root_path"]).resolve()
    verified: dict[str, Path] = {}
    for window in windows:
        if not isinstance(window, Mapping):
            raise TypeError("registered media window must be a mapping")
        window_id = window.get("window_id")
        if not isinstance(window_id, str) or not window_id or window_id in verified:
            raise RuntimeError("registered media window IDs must be unique strings")
        media_path = (root / str(window.get("media_path", ""))).resolve()
        try:
            media_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("registered media path escapes data root") from exc
        if not media_path.is_file():
            raise RuntimeError("registered media file does not exist")
        if _file_sha256(media_path)[1] != window.get("media_sha256"):
            raise RuntimeError("registered media bytes differ from manifest")
        verified[window_id] = media_path
    return verified


def _load_json(path: Path) -> object:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _strip_ddp_prefix(state: Mapping[str, object]) -> dict[str, object]:
    keys = tuple(str(key) for key in state)
    prefixed = tuple(key.startswith("module.") for key in keys)
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("dense checkpoint mixes DDP-prefixed and unprefixed keys")
    if keys and all(prefixed):
        return {str(key)[7:]: value for key, value in state.items()}
    return {str(key): value for key, value in state.items()}


def audit_load_dense_checkpoint(
    model: torch.nn.Module,
    checkpoint_source: str | Path | bytes,
    *,
    use_ema: bool,
    allow_chronotransport_missing: bool,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, object]:
    """Load the exact dense state and reject every unaudited incompatibility."""

    if isinstance(checkpoint_source, bytes):
        stable_bytes = checkpoint_source
    else:
        path = Path(checkpoint_source)
        if not path.is_file():
            raise RuntimeError("registered dense checkpoint does not exist")
        stable_bytes = path.read_bytes()
    stable_sha256 = hashlib.sha256(stable_bytes).hexdigest()
    if expected_bytes is not None and len(stable_bytes) != expected_bytes:
        raise RuntimeError("dense checkpoint stable bytes size differs from registration")
    if expected_sha256 is not None and stable_sha256 != expected_sha256:
        raise RuntimeError("dense checkpoint stable bytes hash differs from registration")
    buffer = io.BytesIO(stable_bytes)
    try:
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("dense checkpoint root must be a mapping")
    state_key = "state_dict_ema" if use_ema else "state_dict"
    state = checkpoint.get(state_key)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"dense checkpoint requires mapping key {state_key!r}")
    incompatible = model.load_state_dict(_strip_ddp_prefix(state), strict=False)
    missing = sorted(str(key) for key in incompatible.missing_keys)
    unexpected = sorted(str(key) for key in incompatible.unexpected_keys)
    allowed_missing = (
        [key for key in missing if "chronotransport" in key.split(".")]
        if allow_chronotransport_missing
        else []
    )
    forbidden_missing = [key for key in missing if key not in allowed_missing]
    if forbidden_missing or unexpected:
        raise RuntimeError(
            "dense checkpoint has incompatible keys: "
            f"forbidden_missing={forbidden_missing}, unexpected={unexpected}"
        )
    return {
        "status": "PASS",
        "state_key": state_key,
        "allowed_chronotransport_missing": allowed_missing,
        "forbidden_missing": forbidden_missing,
        "unexpected": unexpected,
        "checkpoint_sha256": stable_sha256,
        "checkpoint_bytes": len(stable_bytes),
    }


def _candidate_plan(registration: Mapping[str, Any], candidate_name: str) -> Mapping[str, Any]:
    plans = registration.get("profiler", {}).get("candidate_plan", ())
    matches = [plan for plan in plans if plan.get("candidate_name") == candidate_name]
    if len(matches) != 1:
        raise RuntimeError("registration must contain one exact candidate plan")
    return matches[0]


def _period(candidate_name: str, prefix: str) -> int:
    if not candidate_name.startswith(prefix):
        raise RuntimeError("control candidate name is invalid")
    try:
        period = int(candidate_name[len(prefix) :])
    except ValueError as exc:
        raise RuntimeError("control candidate period is invalid") from exc
    if period not in (2, 4, 8):
        raise RuntimeError("control candidate period is not frozen")
    return period


def _validate_action_payload(payload: object, *, candidate_name: str) -> list[list[int]]:
    tensor = torch.as_tensor(payload)
    if tensor.dtype == torch.bool or tuple(tensor.shape) != (48, 3):
        raise RuntimeError("registered profile action payload must have exact [48,3] shape")
    if tensor.is_floating_point() or tensor.is_complex():
        raise RuntimeError("registered profile actions must use integer values")
    tensor = tensor.to(torch.long)
    ChronoSchedule(
        actions=tensor.unsqueeze(0),
        layer_groups=(
            LayerGroup(0, 4),
            LayerGroup(4, 8),
            LayerGroup(8, 12),
        ),
        name=candidate_name,
    )
    return tensor.tolist()


def resolve_registered_action_payload(
    registration: Mapping[str, Any],
    *,
    window_id: str,
    candidate_name: str,
    deploy_visible_motion: torch.Tensor | None = None,
) -> list[list[int]]:
    """Rebuild one window's actions and require its pre-registered exact hash."""

    invocation_ids = list(registration.get("profiler", {}).get("invocation_ids", ()))
    if invocation_ids.count(window_id) != 1:
        raise RuntimeError("profile window ID is not uniquely registered")
    invocation_index = invocation_ids.index(window_id)
    plan = _candidate_plan(registration, candidate_name)

    library_rows = {
        row.get("name"): row
        for row in registration.get("candidate_library", {}).get("candidates", ())
        if isinstance(row, Mapping)
    }
    if candidate_name in library_rows:
        payload = library_rows[candidate_name].get("actions")
    elif candidate_name.startswith("motion_topk_p"):
        if deploy_visible_motion is None:
            raise RuntimeError("motion control requires deploy-visible motion from this window")
        period = _period(candidate_name, "motion_topk_p")
        payload = motion_topk_actions(
            torch.as_tensor(deploy_visible_motion).unsqueeze(0),
            period=period,
        )[0].cpu().tolist()
    elif candidate_name.startswith("random_p"):
        period = _period(candidate_name, "random_p")
        factory_config = plan.get("factory_config")
        if not isinstance(factory_config, Mapping) or "control_seed" not in factory_config:
            raise RuntimeError(
                "random control_seed is not frozen in registered factory_config"
            )
        payload = random_exact_count_actions(
            window_id,
            seed=factory_config["control_seed"],
            num_groups=3,
            period=period,
        ).cpu().tolist()
    else:
        raise RuntimeError("candidate is absent from the frozen library and controls")

    validated = _validate_action_payload(payload, candidate_name=candidate_name)
    expected_hashes = plan.get("requested_action_sha256_by_invocation")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(invocation_ids):
        raise RuntimeError("candidate plan action hash sequence is incomplete")
    if canonical_sha256(validated) != expected_hashes[invocation_index]:
        raise RuntimeError("rebuilt action bytes do not match registered action hash")
    return validated


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _runtime(model: torch.nn.Module):
    runtimes = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "ChronoTransportRuntime"
    ]
    if len(runtimes) != 1:
        raise RuntimeError("formal profile requires exactly one ChronoTransportRuntime")
    return runtimes[0]


class OpenTADRegisteredProfileBackend:
    """Construct and invoke the one fixed OpenTAD full-stack profile path."""

    def __init__(self, registration: Mapping[str, object]) -> None:
        self.registration = validate_pre_gate1_registration(registration)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("registered OpenTAD profile backend requires one visible CUDA GPU")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("registered OpenTAD profile backend requires physical GPU1")
        if observed_registered_environment() != self.registration["environment"]:
            raise RuntimeError("observed backend environment differs from registration")
        config_overrides = sorted(name for name in _CONFIG_OVERRIDE_ENV if name in os.environ)
        if config_overrides:
            raise RuntimeError(
                "formal profile forbids environment-driven ChronoTransport config overrides: "
                f"{config_overrides}"
            )
        if R2_PROFILE_CONFIG_RELATIVE not in REQUIRED_REGISTRATION_SOURCE_PATHS:
            raise RuntimeError("fixed profile config is absent from required registration sources")
        actual_sources = {
            relative: _file_sha256(ROOT / relative)[1]
            for relative in REQUIRED_REGISTRATION_SOURCE_PATHS
        }
        if actual_sources != self.registration["source_files"]:
            raise RuntimeError("formal profile source bytes differ from registration")

        self._load_external_inputs()
        self._verified_media = preverify_registered_media(
            self.registration,
            self.manifest["windows"],
        )
        self.cfg = Config.fromfile(str(R2_PROFILE_CONFIG))
        self.cfg.model.backbone.custom.pretrain = None
        runtime_cfg = self.cfg.model.backbone.backbone.chronotransport
        allow_legacy_checkpoint = bool(runtime_cfg.allow_legacy_checkpoint)
        runtime_cfg.forced_schedule = None
        runtime_cfg.forced_actions = None
        runtime_cfg.profile_sync_cuda = True
        self.model = build_detector(self.cfg.model)
        self.checkpoint_audit = audit_load_dense_checkpoint(
            self.model,
            self._checkpoint_bytes,
            use_ema=bool(getattr(self.cfg.solver, "ema", False)),
            allow_chronotransport_missing=allow_legacy_checkpoint,
            expected_sha256=self.registration["dense_checkpoint"]["sha256"],
            expected_bytes=self.registration["dense_checkpoint"]["bytes"],
        )
        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device).eval()
        self.runtime = _runtime(self.model)
        self.runtime.capture_replay_signals = False
        self.runtime.profile_sync_cuda = True
        self._build_pipelines()
        self.latest_action_provenance: dict[str, object] | None = None

    def _load_external_inputs(self) -> None:
        manifest_identity = self.registration["window_manifest"]
        manifest_path = Path(manifest_identity["source_path"])
        registry_path = Path(manifest_identity["registry_path"])
        config_path = Path(manifest_identity["config_identity_path"])
        for path in (manifest_path, registry_path, config_path):
            if not path.is_file():
                raise RuntimeError(f"registered profile input does not exist: {path}")
        raw_manifest = manifest_path.read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != manifest_identity["exact_bytes_sha256"]:
            raise RuntimeError("manifest exact bytes differ from registration")
        manifest = _load_json(manifest_path)
        registry = _load_json(registry_path)
        config = _load_json(config_path)
        if raw_manifest != manifest_exact_bytes(manifest):
            raise RuntimeError("registered manifest file is not exact canonical bytes")
        rebuilt = validate_r2_manifest(manifest, registry=registry, config_identity=config)
        if rebuilt != manifest_identity["artifact"]:
            raise RuntimeError("manifest/registry/config payload differs from registration")
        self.manifest = rebuilt
        self.windows = {row["window_id"]: row for row in rebuilt["windows"]}

        checkpoint = self.registration["dense_checkpoint"]
        self._checkpoint_bytes = Path(checkpoint["content_addressed_path"]).read_bytes()
        size = len(self._checkpoint_bytes)
        digest = hashlib.sha256(self._checkpoint_bytes).hexdigest()
        if size != checkpoint["bytes"] or digest != checkpoint["sha256"]:
            raise RuntimeError("dense checkpoint bytes differ from registration")
        receipt = checkpoint["registry_receipt"]
        receipt_path = Path(receipt["source_path"])
        raw_receipt = receipt_path.read_bytes()
        receipt_artifact = _load_json(receipt_path)
        if (
            raw_receipt != canonical_json_bytes(receipt_artifact) + b"\n"
            or hashlib.sha256(raw_receipt).hexdigest() != receipt["exact_bytes_sha256"]
            or receipt_artifact != receipt["artifact"]
        ):
            raise RuntimeError("checkpoint receipt bytes differ from registration")
        validate_checkpoint_registry_receipt(
            receipt_artifact,
            provider_receipt_path=receipt["provider_receipt_path"],
            registry_id=checkpoint["registry_id"],
            authenticated_uri=checkpoint["authenticated_uri"],
            content_sha256=digest,
            content_bytes=size,
        )

    def _build_pipelines(self) -> None:
        pipeline = [deepcopy(dict(item)) for item in self.cfg.dataset.test.pipeline]
        decode_positions = [
            index
            for index, item in enumerate(pipeline)
            if str(item.get("type")) == "mmaction.DecordDecode"
        ]
        if decode_positions != [3]:
            raise RuntimeError("fixed OpenTAD test pipeline decode boundary changed")
        boundary = decode_positions[0]
        self.decode_pipeline = Compose(pipeline[: boundary + 1])
        self.preprocess_pipeline = Compose(pipeline[boundary + 1 :])
        replay_pipeline = [
            deepcopy(dict(item)) for item in self.cfg.dataset.val.pipeline
        ]
        replay_decode_positions = [
            index
            for index, item in enumerate(replay_pipeline)
            if str(item.get("type")) == "mmaction.DecordDecode"
        ]
        if replay_decode_positions != [3]:
            raise RuntimeError("fixed OpenTAD replay pipeline decode boundary changed")
        self.replay_preprocess_pipeline = Compose(
            replay_pipeline[replay_decode_positions[0] + 1 :]
        )
        self.post_cfg = deepcopy(self.cfg.post_processing)
        # Each manifested video contributes exactly one window; execute NMS in
        # this invocation so the measured boundary truly ends after NMS.
        self.post_cfg.sliding_window = False
        self.external_classes = [str(index) for index in range(20)]

    def _sync(self) -> None:
        torch.cuda.synchronize(self.device)

    def _timed(self, callback, *, sync_cuda: bool) -> tuple[Any, float]:
        if sync_cuda:
            self._sync()
        start = perf_counter()
        result = callback()
        if sync_cuda:
            self._sync()
        return result, (perf_counter() - start) * 1000.0

    def _verify_media(self, window: Mapping[str, Any]) -> Path:
        try:
            return self._verified_media[window["window_id"]]
        except KeyError as exc:
            raise RuntimeError("media window was not preverified") from exc

    def _base_results(self, window: Mapping[str, Any]) -> dict[str, object]:
        media_path = self._verify_media(window)
        if media_path.suffix.lower() != ".mp4":
            raise RuntimeError("fixed OpenTAD profile requires registered mp4 media")
        if media_path.stem != window["video_id"]:
            raise RuntimeError("registered video_id must equal the OpenTAD media stem")
        valid_count = sum(value is True for value in window["valid_mask"])
        if valid_count <= 0:
            raise RuntimeError("registered window contains no valid frames")
        sampled = window["sampled_frame_indices"]
        return {
            "video_name": media_path.stem,
            "data_path": str(media_path.parent),
            "window_size": 768,
            "feature_start_idx": int(window["window_start"]),
            "feature_end_idx": int(window["window_start"] + valid_count - 1),
            "sample_stride": 1,
            "fps": float(window["fps"]),
            "snippet_stride": int(window["snippet_stride"]),
            "window_start_frame": int(sampled[0]),
            "duration": float(window["source_total_frames"]) / float(window["fps"]),
            "offset_frames": 0,
        }

    def _decode_and_preprocess(
        self, window: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float, float]:
        decoded, decode_ms = self._timed(
            lambda: self.decode_pipeline(self._base_results(window)),
            sync_cuda=False,
        )
        if int(decoded.get("total_frames", -1)) != int(window["source_total_frames"]):
            raise RuntimeError("decoded source frame count differs from registered media metadata")
        observed_fps = float(decoded.get("avg_fps", float("nan")))
        if not np.isfinite(observed_fps) or not np.isclose(
            observed_fps,
            float(window["fps"]),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError("decoded FPS differs from registered media metadata")
        observed_indices = np.asarray(decoded.get("frame_inds")).reshape(-1).tolist()
        if observed_indices != window["sampled_frame_indices"]:
            raise RuntimeError("OpenTAD decoded frame indices differ from registered window")
        sample, preprocess_ms = self._timed(
            lambda: self.preprocess_pipeline(decoded),
            sync_cuda=False,
        )
        if sample["masks"].tolist() != window["valid_mask"]:
            raise RuntimeError("OpenTAD valid mask differs from registered window")
        return sample, decode_ms, preprocess_ms

    def _model_forward(self, batch: Mapping[str, Any]) -> tuple[object, float]:
        inputs = batch["inputs"]
        masks = batch["masks"]
        metas = batch["metas"]
        # Use the detector's official test path verbatim.  The timer label is a
        # diagnostic bucket only; the formal budget is the outer direct total.
        return self._timed(
            lambda: self.model.forward_test(inputs, masks, metas=metas, infer_cfg=None),
            sync_cuda=True,
        )

    def _official_postprocess(
        self,
        predictions: object,
        metas: object,
    ) -> tuple[object, float]:
        return self._timed(
            lambda: self.model.post_processing(
                predictions,
                metas,
                self.post_cfg,
                self.external_classes,
            ),
            sync_cuda=True,
        )

    def invoke_registered_window(
        self,
        *,
        window_id: str,
        candidate_name: str,
    ) -> Mapping[str, object]:
        if window_id not in self.windows:
            raise RuntimeError("requested profile window is not in registered manifest")
        window = self.windows[window_id]
        sample, data_decode_ms, preprocess_ms = self._decode_and_preprocess(window)
        batch, h2d_ms = self._timed(
            lambda: _move_to_device(collate([sample]), self.device),
            sync_cuda=True,
        )

        control_timing = {"innovation": 0.0, "scheduler": 0.0}
        motion_sha256 = None
        resolved_motion_payload = None
        hook = None
        if candidate_name.startswith("motion_topk_p"):
            def bind_motion_actions(module, args):
                nonlocal motion_sha256, resolved_motion_payload
                x = args[0]
                batch_size = int(x.shape[0]) // module.chunks_per_window
                state = x.reshape(
                    batch_size,
                    module.chunks_per_window,
                    int(x.shape[1]),
                    int(x.shape[2]),
                )
                signals, innovation_ms = self._timed(
                    lambda: module._signals(state).detach(),
                    sync_cuda=True,
                )
                motion = signals[..., 3]
                motion_sha256 = canonical_sha256(
                    motion[0].float().cpu().tolist()
                )
                actions, scheduler_ms = self._timed(
                    lambda: resolve_registered_action_payload(
                        self.registration,
                        window_id=window_id,
                        candidate_name=candidate_name,
                        deploy_visible_motion=motion[0],
                    ),
                    sync_cuda=True,
                )
                control_timing["innovation"] += innovation_ms
                control_timing["scheduler"] += scheduler_ms
                resolved_motion_payload = actions
                module.set_registered_forced_actions(
                    torch.as_tensor(actions, device=x.device),
                    candidate_name=candidate_name,
                )

            hook = self.runtime.register_forward_pre_hook(bind_motion_actions)
            requested_payload = None
        else:
            requested_payload, scheduler_ms = self._timed(
                lambda: resolve_registered_action_payload(
                    self.registration,
                    window_id=window_id,
                    candidate_name=candidate_name,
                ),
                sync_cuda=True,
            )
            if candidate_name.startswith("random_p"):
                control_timing["scheduler"] += scheduler_ms
            self.runtime.set_registered_forced_actions(
                torch.as_tensor(requested_payload, device=self.device),
                candidate_name=candidate_name,
            )

        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                predictions, neck_head_ms = self._model_forward(batch)
        finally:
            if hook is not None:
                hook.remove()

        if requested_payload is None:
            if resolved_motion_payload is None:
                raise RuntimeError("motion control did not produce an executed schedule")
            requested_payload = resolved_motion_payload
        executed = self.runtime.latest_schedule
        summary = self.runtime.latest_summary
        if executed is None or not isinstance(summary, Mapping):
            raise RuntimeError("OpenTAD runtime did not expose schedule evidence")
        executed_payload = executed.actions[0].detach().cpu().to(torch.long).tolist()

        _, postprocess_ms = self._official_postprocess(predictions, batch["metas"])
        runtime_latency = summary.get("profile", {}).get("latency_ms", {})
        diagnostics = {name: 0.0 for name in REQUIRED_STAGE_FIELDS}
        diagnostics.update(
            data_decode=float(data_decode_ms),
            preprocess=float(preprocess_ms),
            h2d=float(h2d_ms),
            neck_head=float(neck_head_ms),
            postprocess=float(postprocess_ms),
        )
        for name in (
            "innovation",
            "scheduler",
            "recompute",
            "transport",
            "cache_movement",
            "dense_adatad_adapter",
        ):
            measurement = runtime_latency.get(name, {})
            diagnostics[name] += float(measurement.get("total", 0.0) or 0.0)
        diagnostics["innovation"] += control_timing["innovation"]
        diagnostics["scheduler"] += control_timing["scheduler"]

        repair_count = int(summary.get("schedule_repair_count", 0))
        whole_window_dense_fallback = bool(summary.get("whole_window_dense_fallback", False))
        nan_fallback = int(summary.get("runtime_fail_closed_repairs", 0)) > 0
        self.latest_action_provenance = {
            "window_id": window_id,
            "candidate_name": candidate_name,
            "manifest_window_sha256": window["window_sha256"],
            "checkpoint_sha256": self.registration["dense_checkpoint"]["sha256"],
            "config_sha256": self.registration["profiler"]["model_config_sha256"],
            "requested_action_sha256": canonical_sha256(requested_payload),
            "executed_action_sha256": canonical_sha256(executed_payload),
            "deploy_visible_motion_sha256": motion_sha256,
        }
        execution_provenance = {
            "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
            "backend_source_sha256": self.registration["source_files"][
                REGISTERED_PROFILE_BACKEND_SOURCE
            ],
            "deploy_visible_signal_sha256": motion_sha256,
            "requested_action_sha256": canonical_sha256(requested_payload),
            "executed_action_sha256": canonical_sha256(executed_payload),
        }
        return {
            "diagnostic_ms": diagnostics,
            "requested_action_payload": requested_payload,
            "executed_schedule_name": executed.name,
            "executed_action_payload": executed_payload,
            "repair_count": repair_count,
            "nan_fallback": nan_fallback,
            "whole_window_dense_fallback": whole_window_dense_fallback,
            "safety_override_budget_violation": False,
            "execution_provenance": execution_provenance,
        }


class OpenTADRegisteredGate1ReplayBackend(OpenTADRegisteredProfileBackend):
    """Materialize deterministic manifested windows and registered GT losses."""

    def __init__(self, registration: Mapping[str, object]) -> None:
        super().__init__(registration)
        annotation_path = Path(str(self.cfg.dataset.val.ann_file))
        if not annotation_path.is_absolute():
            annotation_path = (ROOT / annotation_path).resolve()
        if (
            not annotation_path.is_file()
            or _file_sha256(annotation_path)[1]
            != self.registration["data"]["annotation_sha256"]
        ):
            raise RuntimeError("fixed Gate 1 annotation bytes differ from registration")
        annotation = _load_json(annotation_path)
        if not isinstance(annotation, Mapping) or not isinstance(
            annotation.get("database"), Mapping
        ):
            raise RuntimeError("fixed Gate 1 annotation database is invalid")
        self._annotation_database = annotation["database"]

        class_map_path = Path(str(self.cfg.dataset.val.class_map))
        if not class_map_path.is_absolute():
            class_map_path = (ROOT / class_map_path).resolve()
        if not class_map_path.is_file():
            raise RuntimeError("fixed Gate 1 class map does not exist")
        self._class_map = [
            line.strip()
            for line in class_map_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self._class_map:
            raise RuntimeError("fixed Gate 1 class map is empty")

    def _window_ground_truth(
        self, window: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        video_id = str(window["video_id"])
        info = self._annotation_database.get(video_id)
        if not isinstance(info, Mapping) or not isinstance(info.get("annotations"), list):
            raise RuntimeError("manifest video is absent from registered annotations")
        start_frame = int(window["sampled_frame_indices"][0])
        valid_count = sum(value is True for value in window["valid_mask"])
        end_frame = int(window["sampled_frame_indices"][valid_count - 1])
        snippet_stride = float(window["snippet_stride"])
        segments = []
        labels = []
        for item in info["annotations"]:
            if not isinstance(item, Mapping) or item.get("label") == "Ambiguous":
                continue
            label = str(item.get("label"))
            if label not in self._class_map:
                raise RuntimeError("registered annotation label is absent from class map")
            raw = item.get("segment")
            if not isinstance(raw, list) or len(raw) != 2:
                raise RuntimeError("registered annotation segment is invalid")
            begin = int(float(raw[0]) * float(window["fps"]))
            end = int(float(raw[1]) * float(window["fps"]))
            clipped_begin = max(begin, start_frame)
            clipped_end = min(end, end_frame)
            if clipped_end <= clipped_begin:
                continue
            segments.append(
                [
                    (clipped_begin - start_frame) / snippet_stride,
                    (clipped_end - start_frame) / snippet_stride,
                ]
            )
            labels.append(self._class_map.index(label))
        return (
            np.asarray(segments, dtype=np.float32).reshape(-1, 2),
            np.asarray(labels, dtype=np.int32),
        )

    def materialize_registered_replay_window(
        self, window_id: str
    ) -> Mapping[str, object]:
        if window_id not in self.windows:
            raise RuntimeError("requested replay window is not registered")
        window = self.windows[window_id]
        gt_segments, gt_labels = self._window_ground_truth(window)
        base = self._base_results(window)
        base["gt_segments"] = gt_segments
        base["gt_labels"] = gt_labels
        decoded = self.decode_pipeline(base)
        observed_indices = np.asarray(decoded.get("frame_inds")).reshape(-1).tolist()
        if observed_indices != window["sampled_frame_indices"]:
            raise RuntimeError("Gate 1 replay decoded indices differ from manifest")
        sample = self.replay_preprocess_pipeline(decoded)
        if sample["masks"].tolist() != window["valid_mask"]:
            raise RuntimeError("Gate 1 replay valid mask differs from manifest")
        batch = _move_to_device(collate([sample]), self.device)
        forward_kwargs = dict(batch)
        forward_kwargs["return_loss"] = True
        from opentad.models.chronotransport.replay import materialized_batch_sha256

        return {
            "forward_kwargs": forward_kwargs,
            "augmentation_sha256": materialized_batch_sha256(
                {"inputs": forward_kwargs["inputs"]}
            ),
        }
