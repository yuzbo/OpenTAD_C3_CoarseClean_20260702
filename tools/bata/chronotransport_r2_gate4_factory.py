#!/usr/bin/env python3
"""Repository-owned official-population execution factory for formal Gate 4."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gc
import math
import os
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mmengine.dataset import Compose

import opentad.datasets  # noqa: F401 - register pipeline transforms
from opentad.datasets.builder import collate
from opentad.models.chronotransport.environment import (
    observe_formal_slurm_environment,
)
from opentad.models.chronotransport.filesystem import (
    BoundRegularFile,
    open_bound_directory,
    read_bound_bytes,
)
from opentad.models.chronotransport.formal_gate4 import (
    GATE4_ENERGY_ARM_ORDER_BY_SEED,
    _single_invocation_action_sha256,
    build_gate4_metric_evidence,
    build_gate4_regret_evidence,
    build_gate4_seed_shard,
    build_gate4_timing_evidence,
    integrate_power_trace_j,
    validate_gate4_seed_shard,
)
from opentad.models.chronotransport.gate4 import ARMS
from opentad.models.chronotransport.gate4_population import (
    validate_gate4_population_artifact,
)
from opentad.models.chronotransport.gates23 import SCHEDULER_EPSILON
from opentad.models.chronotransport.post_stage_c import (
    validate_post_stage_c_gate3_unlock,
)
from opentad.models.chronotransport.protocol import canonical_sha256
from opentad.models.chronotransport.registration import (
    validate_formal_gate1_context,
)
from opentad.models.chronotransport.runtime import ChronoTransportRuntime
from opentad.models.detectors.actionformer import ActionFormerPerWindowTrainOutput
from opentad.models.utils.post_processing import batched_nms
from tools.bata.chronotransport_r2_post_stage_c_factory import (
    _load_validated_stage_c_seed,
)


ROOT = Path(os.path.abspath(__file__)).parents[2]
GATE4_RUNNER_ENTRYPOINT = "tools/bata/run_chronotransport_r2_gate4.py"
_POWER_SAMPLING_HZ = 10.0
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


def _reject_config_overrides() -> None:
    """Keep Gate-4 precheck and evidence production on the registered config."""

    overrides = sorted(name for name in _CONFIG_OVERRIDE_ENV if name in os.environ)
    if overrides:
        raise RuntimeError(f"formal Gate4 forbids config overrides: {overrides}")


def _runtime(model: torch.nn.Module) -> ChronoTransportRuntime:
    matches = [module for module in model.modules() if isinstance(module, ChronoTransportRuntime)]
    if len(matches) != 1:
        raise RuntimeError("formal Gate4 requires exactly one runtime per arm")
    return matches[0]


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


class _NvmlPowerSampler:
    def __init__(self, gpu_uuid: str) -> None:
        try:
            import pynvml
        except ImportError as error:  # pragma: no cover - formal GPU environment only
            raise RuntimeError("formal Gate4 requires pynvml for 10-Hz energy evidence") from error
        self._pynvml = pynvml
        pynvml.nvmlInit()
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode("ascii"))
        except TypeError:
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._origin = 0.0
        self._samples: list[dict[str, float]] = []

    def _sample(self) -> None:
        moment = perf_counter()
        power_w = float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        row = {"offset_ms": (moment - self._origin) * 1000.0, "power_w": power_w}
        with self._lock:
            if not self._samples or row["offset_ms"] > self._samples[-1]["offset_ms"]:
                self._samples.append(row)

    def _run(self) -> None:
        interval = 1.0 / _POWER_SAMPLING_HZ
        deadline = perf_counter() + interval
        while not self._stop.wait(max(0.0, deadline - perf_counter())):
            self._sample()
            deadline += interval

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("formal Gate4 power sampler is single-use")
        self._origin = perf_counter()
        self._sample()
        self._thread = Thread(target=self._run, name="ct-gate4-nvml", daemon=True)
        self._thread.start()

    def offset_ms(self) -> float:
        return (perf_counter() - self._origin) * 1000.0

    def stop(self) -> list[dict[str, float]]:
        if self._thread is None:
            raise RuntimeError("formal Gate4 power sampler was not started")
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("formal Gate4 power sampler failed to stop")
        self._sample()
        self._pynvml.nvmlShutdown()
        with self._lock:
            return [dict(row) for row in self._samples]


class _CudaModuleTimer:
    def __init__(self, modules: Sequence[torch.nn.Module]) -> None:
        unique = []
        seen = set()
        for module in modules:
            if id(module) not in seen:
                seen.add(id(module))
                unique.append(module)
        self.modules = tuple(unique)
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._active: dict[int, list[torch.cuda.Event]] = {}
        self._handles = []

    def reset(self) -> None:
        self._events.clear()
        self._active.clear()

    def __enter__(self):
        for module in self.modules:
            module_id = id(module)

            def before(_module, _inputs, *, key=module_id):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                self._active.setdefault(key, []).append(start)

            def after(_module, _inputs, _output, *, key=module_id):
                stack = self._active.get(key)
                if not stack:
                    raise RuntimeError("formal Gate4 CUDA timer hook is unbalanced")
                start = stack.pop()
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._events.append((start, end))

            self._handles.append(module.register_forward_pre_hook(before))
            self._handles.append(module.register_forward_hook(after))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def elapsed_ms(self) -> float:
        if any(self._active.values()):
            raise RuntimeError("formal Gate4 CUDA timer has unfinished events")
        return float(sum(start.elapsed_time(end) for start, end in self._events))


class _OfficialPopulationPipeline:
    def __init__(self, cfg: Any, population: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.population = validate_gate4_population_artifact(population)
        self.records = {
            str(row["official_video_id"]): dict(row) for row in self.population["videos"]
        }
        self.invocations = {
            str(row["invocation_id"]): dict(row)
            for row in self.population["unique_invocations"]
        }
        _, class_bytes, class_sha = read_bound_bytes(
            self.population["class_map"]["path"], label="formal Gate4 class map"
        )
        if class_sha != self.population["class_map"]["sha256"]:
            raise ValueError("formal Gate4 class-map bytes differ from population")
        self.class_names = [
            line.strip()
            for line in class_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not self.class_names:
            raise ValueError("formal Gate4 class map is empty")
        self.ground_truth_by_video: dict[str, list[dict[str, Any]]] = {
            video: [] for video in self.population["official_video_ids"]
        }
        for row in self.population["ground_truth"]:
            self.ground_truth_by_video[row["official_video_id"]].append(dict(row))
        self._media: dict[str, BoundRegularFile] = {}
        try:
            with open_bound_directory(
                self.population["data_root"], label="formal Gate4 media root"
            ) as root:
                for video, record in self.records.items():
                    media = root.open_regular(
                        record["media_path"], label=f"formal Gate4 media {video}"
                    )
                    size, digest = media.size_and_sha256()
                    if size != record["media_bytes"] or digest != record["media_sha256"]:
                        media.close()
                        raise ValueError(f"formal Gate4 media bytes differ: {video}")
                    self._media[video] = media
        except BaseException:
            self.close()
            raise
        self.decode_pipeline, self.test_preprocess = self._split_pipeline(cfg.dataset.test.pipeline)
        _, self.regret_preprocess = self._split_pipeline(cfg.dataset.val.pipeline)
        self.post_cfg = deepcopy(cfg.post_processing)
        self.post_cfg.sliding_window = True

    @staticmethod
    def _split_pipeline(pipeline: Sequence[Mapping[str, Any]]):
        rows = [deepcopy(dict(row)) for row in pipeline]
        positions = [
            index for index, row in enumerate(rows) if str(row.get("type")) == "mmaction.DecordDecode"
        ]
        if positions != [3]:
            raise RuntimeError("formal Gate4 decode boundary changed")
        boundary = positions[0]
        return Compose(rows[: boundary + 1]), Compose(rows[boundary + 1 :])

    def close(self) -> None:
        for media in getattr(self, "_media", {}).values():
            media.close()
        self._media = {}

    def _base(self, invocation: Mapping[str, Any], *, regret: bool) -> dict[str, Any]:
        video = str(invocation["official_video_id"])
        record = self.records[video]
        media = self._media[video]
        media.assert_stable()
        sampled = list(invocation["sampled_frame_indices"])
        snippet_stride = int(self.population["dataset_contract"]["feature_stride"])
        result: dict[str, Any] = {
            "video_name": video,
            "data_path": str(Path(media.path).parent),
            "descriptor_filename": media.proc_path,
            "window_size": 768,
            "feature_start_idx": int(sampled[0] / snippet_stride),
            "feature_end_idx": int(sampled[-1] / snippet_stride),
            "sample_stride": 1,
            "fps": float(record["frame"]) / float(record["duration"]),
            "snippet_stride": snippet_stride,
            "window_start_frame": int(sampled[0]),
            "duration": float(record["duration"]),
            "offset_frames": 0,
        }
        if regret:
            anchor_start = int(sampled[0])
            anchor_end = int(sampled[-1])
            segments = []
            labels = []
            for row in self.ground_truth_by_video[video]:
                raw_start = int(float(row["segment"][0]) / record["duration"] * record["frame"])
                raw_end = int(float(row["segment"][1]) / record["duration"] * record["frame"])
                length = max(raw_end - raw_start, 1)
                clipped_start = max(raw_start, anchor_start)
                clipped_end = min(raw_end, anchor_end)
                completeness = max(0, clipped_end - clipped_start) / length
                if completeness <= float(getattr(self.cfg.dataset.val, "ioa_thresh", 0.75)):
                    continue
                segments.append(
                    [
                        (clipped_start - anchor_start) / snippet_stride,
                        (clipped_end - anchor_start) / snippet_stride,
                    ]
                )
                labels.append(self.class_names.index(str(row["label"])))
            result["gt_segments"] = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
            result["gt_labels"] = np.asarray(labels, dtype=np.int32)
        return result

    def materialize(self, invocation_id: str, *, regret: bool) -> tuple[dict[str, Any], float, float]:
        invocation = self.invocations[invocation_id]
        video = str(invocation["official_video_id"])
        start = perf_counter()
        decoded = self.decode_pipeline(self._base(invocation, regret=regret))
        self._media[video].assert_stable()
        decode_ms = (perf_counter() - start) * 1000.0
        record = self.records[video]
        observed_frames = int(decoded.get("total_frames", -1))
        observed_fps = float(decoded.get("avg_fps", float("nan")))
        expected_fps = float(record["frame"]) / float(record["duration"])
        if observed_frames != int(record["frame"]) or not np.isclose(
            observed_fps, expected_fps, rtol=1e-6, atol=1e-6
        ):
            raise RuntimeError("formal Gate4 decoded media metadata differs from population")
        observed = np.asarray(decoded.get("frame_inds")).reshape(-1).tolist()
        if observed != invocation["sampled_frame_indices"]:
            raise RuntimeError("formal Gate4 decoded indices differ from population")
        start = perf_counter()
        sample = (self.regret_preprocess if regret else self.test_preprocess)(decoded)
        preprocess_ms = (perf_counter() - start) * 1000.0
        expected_mask = [True] * int(invocation["valid_count"]) + [False] * (
            768 - int(invocation["valid_count"])
        )
        if sample["masks"].tolist() != expected_mask:
            raise RuntimeError("formal Gate4 valid mask differs from population")
        return sample, decode_ms, preprocess_ms


@dataclass
class _Arm:
    name: str
    model: torch.nn.Module
    runtime: ChronoTransportRuntime
    patch_timer: _CudaModuleTimer
    heavy_timer: _CudaModuleTimer
    adapter_timer: _CudaModuleTimer
    head_timer: _CudaModuleTimer

    def reset_timers(self) -> None:
        self.patch_timer.reset()
        self.heavy_timer.reset()
        self.adapter_timer.reset()
        self.head_timer.reset()

    def timer_contexts(self):
        return (
            self.patch_timer,
            self.heavy_timer,
            self.adapter_timer,
            self.head_timer,
        )


def _build_arm(name: str, model: torch.nn.Module) -> _Arm:
    runtime = _runtime(model)
    backbone = model.backbone.model.backbone
    blocks = tuple(backbone.blocks)
    patch = _CudaModuleTimer((backbone.patch_embed,))
    heavy = _CudaModuleTimer(
        tuple(module for block in blocks for module in (block.attn, block.mlp))
    )
    adapter = _CudaModuleTimer(
        tuple(block.adapter for block in blocks if bool(getattr(block, "use_adapter", False)))
    )
    head_modules = tuple(
        module
        for module in (
            getattr(model, "projection", None),
            getattr(model, "neck", None),
            getattr(model, "rpn_head", None),
        )
        if module is not None
    )
    return _Arm(
        name=name,
        model=model,
        runtime=runtime,
        patch_timer=patch,
        heavy_timer=heavy,
        adapter_timer=adapter,
        head_timer=_CudaModuleTimer(head_modules),
    )


def _action_payload(registration: Mapping[str, Any], name: str) -> torch.Tensor:
    rows = {
        str(row["name"]): row for row in registration["candidate_library"]["candidates"]
    }
    if name not in rows:
        raise ValueError(f"formal Gate4 schedule is absent from registration: {name}")
    return torch.as_tensor(rows[name]["actions"], dtype=torch.long, device="cuda:0")


def _scheduler_contract(
    gate1_unlock: Mapping[str, Any], post_unlock: Mapping[str, Any]
) -> dict[str, Any]:
    result = gate1_unlock.get("gate1_result")
    if not isinstance(result, Mapping) or result.get("status") != "PASS":
        raise ValueError("formal Gate4 requires a PASS Gate1 result")
    return {
        "budget": float(result["budget"]),
        "epsilon": float(SCHEDULER_EPSILON),
        "calibration_frozen_static": str(result["calibration_frozen_static"]),
        "q_conf_by_seed": {
            str(seed): float(post_unlock["q_conf_by_seed"][str(seed)])
            for seed in (3407, 3408, 3409)
        },
        "gate1_unlock_artifact_sha256": gate1_unlock["artifact_sha256"],
        "calibration_sha256": post_unlock["post_stage_c_gate3_report_sha256"],
    }


def _configure_arms(
    context: Any,
    *,
    registration: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    seed: int,
) -> dict[str, _Arm]:
    context.state.matched_objects["ema"].copy_to(context.components.matched_model)
    ct_model = context.components.ct_model.eval()
    dense_model = context.components.matched_model.eval()
    static_model = deepcopy(ct_model).to(torch.device("cuda:0")).eval()
    arms = {
        "dense": _build_arm("dense", dense_model),
        "chronotransport": _build_arm("chronotransport", ct_model),
        "static": _build_arm("static", static_model),
    }
    dense = arms["dense"].runtime
    static = arms["static"].runtime
    learned = arms["chronotransport"].runtime
    dense.set_registered_forced_actions(_action_payload(registration, "dense"), candidate_name="dense")
    static_name = str(scheduler["calibration_frozen_static"])
    static.set_registered_forced_actions(
        _action_payload(registration, static_name), candidate_name=static_name
    )
    learned.clear_registered_forced_actions()
    learned.install_registered_gate3_calibration(
        q_conf=float(scheduler["q_conf_by_seed"][str(seed)]),
        budget=float(scheduler["budget"]),
        calibration_sha256=str(scheduler["calibration_sha256"]),
    )
    for arm in arms.values():
        arm.runtime.profile_sync_cuda = False
        arm.runtime.profile_deferred_cuda_events = True
        arm.runtime.capture_replay_signals = False
        arm.runtime.set_checkpoint_loaded(True)
    return arms


def _official_sliding_nms(
    rows_by_video: Mapping[str, Sequence[Mapping[str, Any]]], post_cfg: Any
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for video, raw in rows_by_video.items():
        rows = [dict(row) for row in raw]
        if not rows or post_cfg.nms is None:
            result[video] = rows
            continue
        segments = torch.tensor([row["segment"] for row in rows], dtype=torch.float32)
        scores = torch.tensor([row["score"] for row in rows], dtype=torch.float32)
        class_names = []
        labels = []
        for row in rows:
            label = str(row["label"])
            if label not in class_names:
                class_names.append(label)
            labels.append(class_names.index(label))
        nms_segments, nms_scores, nms_labels = batched_nms(
            segments,
            scores,
            torch.tensor(labels, dtype=torch.float32),
            **post_cfg.nms,
        )
        result[video] = [
            {
                "segment": [round(value.item(), 2) for value in segment],
                "label": class_names[int(label.item())],
                "score": round(score.item(), 4),
            }
            for segment, label, score in zip(nms_segments, nms_labels, nms_scores)
        ]
    return result


def _invoke_timed_arm(
    arm: _Arm,
    pipeline: _OfficialPopulationPipeline,
    *,
    invocation_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    total_start = perf_counter()
    sample, decode_ms, preprocess_ms = pipeline.materialize(invocation_id, regret=False)
    torch.cuda.synchronize()
    h2d_start = perf_counter()
    batch = _move_to_device(collate([sample]), torch.device("cuda:0"))
    torch.cuda.synchronize()
    h2d_ms = (perf_counter() - h2d_start) * 1000.0
    arm.reset_timers()
    contexts = arm.timer_contexts()
    with contexts[0], contexts[1], contexts[2], contexts[3]:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            predictions = arm.model.forward_test(
                batch["inputs"], batch["masks"], metas=batch["metas"], infer_cfg=None
            )
    torch.cuda.synchronize()
    arm.runtime.finalize_deferred_profile(outer_cuda_synchronized=True)
    post_start = perf_counter()
    output = arm.model.post_processing(
        predictions,
        batch["metas"],
        pipeline.post_cfg,
        pipeline.class_names,
    )
    torch.cuda.synchronize()
    postprocess_ms = (perf_counter() - post_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    summary = arm.runtime.latest_summary
    schedule = arm.runtime.latest_schedule
    if not isinstance(summary, Mapping) or schedule is None:
        raise RuntimeError("formal Gate4 runtime omitted schedule evidence")
    actions = schedule.actions.detach().cpu().to(torch.long).tolist()
    batched_action_sha = canonical_sha256(actions)
    action_sha = _single_invocation_action_sha256(actions)
    for field in ("requested_action_sha256", "executed_action_sha256"):
        runtime_sha = summary.get(field)
        if runtime_sha is not None and runtime_sha != batched_action_sha:
            raise RuntimeError(
                "formal Gate4 runtime action hash differs from its executed tensor"
            )
    names = summary.get("selected_schedule_names")
    if not isinstance(names, list) or len(names) != 1:
        raise RuntimeError("formal Gate4 runtime selected-name evidence is invalid")
    upper = summary.get("upper_risk")
    cost = summary.get("estimated_cost")
    fail_closed = summary.get("fail_closed")
    if (
        not isinstance(upper, list)
        or len(upper) != 1
        or not isinstance(cost, list)
        or len(cost) != 1
        or not isinstance(fail_closed, list)
        or len(fail_closed) != 1
    ):
        raise RuntimeError("formal Gate4 runtime risk/cost evidence is invalid")
    latency = summary.get("profile", {}).get("latency_ms", {})
    stage = {
        "decode_ms": float(decode_ms),
        "preprocess_ms": float(preprocess_ms),
        "h2d_ms": float(h2d_ms),
        "patch_embed_ms": arm.patch_timer.elapsed_ms(),
        "heavy_ms": arm.heavy_timer.elapsed_ms(),
        "innovation_ms": float(latency.get("innovation", {}).get("total", 0.0) or 0.0),
        "scheduler_ms": float(latency.get("scheduler", {}).get("total", 0.0) or 0.0),
        "transport_ms": float(latency.get("transport", {}).get("total", 0.0) or 0.0),
        "cache_movement_ms": float(
            latency.get("cache_movement", {}).get("total", 0.0) or 0.0
        ),
        "adapter_ms": arm.adapter_timer.elapsed_ms(),
        "head_ms": arm.head_timer.elapsed_ms(),
        "postprocess_ms": float(postprocess_ms),
    }
    audit = {
        "selected_schedule": str(names[0]),
        "requested_action_sha256": action_sha,
        "executed_action_sha256": action_sha,
        "recompute_rows": int(summary.get("recompute_rows", -1)),
        "transport_rows": int(summary.get("transport_rows", -1)),
        "hold_rows": int(summary.get("hold_rows", -1)),
        "schedule_repair_count": int(summary.get("schedule_repair_count", -1)),
        "runtime_fail_closed_repairs": int(summary.get("runtime_fail_closed_repairs", -1)),
        "whole_window_dense_fallback": bool(summary.get("whole_window_dense_fallback", False)),
        "upper_risk": float(upper[0]),
        "estimated_cost": float(cost[0]),
        "registered_gate3_calibration_sha256": summary.get(
            "registered_gate3_calibration_sha256"
        ),
        "registered_q_conf": summary.get("registered_q_conf"),
        "registered_budget": summary.get("registered_budget"),
        "evidence_valid": bool(summary.get("evidence_valid", True)),
        "fail_closed": bool(fail_closed[0]),
    }
    arm_payload = {
        "total_ms": float(total_ms),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "nvml_energy_j": 0.0,
        "stage_ms": stage,
    }
    video = pipeline.invocations[invocation_id]["official_video_id"]
    rows = output.get(video)
    if not isinstance(rows, list):
        raise RuntimeError("formal Gate4 postprocess omitted official video output")
    return arm_payload, audit, {video: [dict(row) for row in rows]}


def _invoke_energy_block(
    arm: _Arm,
    pipeline: _OfficialPopulationPipeline,
    *,
    invocations: Sequence[Mapping[str, Any]],
    sampler: _NvmlPowerSampler,
) -> dict[str, Any]:
    """Run one contiguous full-population block for 10-Hz energy diagnostics."""

    if len(invocations) < 200:
        raise ValueError("formal Gate4 energy block requires the official population")
    deferred = arm.runtime.profile_deferred_cuda_events
    arm.runtime.profile_deferred_cuda_events = False
    rows_by_video = {video: [] for video in pipeline.population["official_video_ids"]}
    torch.cuda.synchronize()
    start_ms = sampler.offset_ms()
    try:
        for invocation in invocations:
            sample, _, _ = pipeline.materialize(
                str(invocation["invocation_id"]), regret=False
            )
            batch = _move_to_device(collate([sample]), torch.device("cuda:0"))
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                predictions = arm.model.forward_test(
                    batch["inputs"],
                    batch["masks"],
                    metas=batch["metas"],
                    infer_cfg=None,
                )
            output = arm.model.post_processing(
                predictions,
                batch["metas"],
                pipeline.post_cfg,
                pipeline.class_names,
            )
            video = str(invocation["official_video_id"])
            current = output.get(video)
            if not isinstance(current, list):
                raise RuntimeError("formal Gate4 energy block omitted official output")
            rows_by_video[video].extend(dict(row) for row in current)
        finalized = _official_sliding_nms(rows_by_video, pipeline.post_cfg)
        torch.cuda.synchronize()
        end_ms = sampler.offset_ms()
    finally:
        arm.runtime.profile_deferred_cuda_events = deferred
    flattened = [
        {
            "official_video_id": video,
            "label": str(item["label"]),
            "segment": [float(value) for value in item["segment"]],
            "score": float(item["score"]),
        }
        for video in pipeline.population["official_video_ids"]
        for item in finalized[video]
    ]
    return {
        "arm": arm.name,
        "invocation_count": len(invocations),
        "invocation_order_sha256": canonical_sha256(
            [str(row["invocation_id"]) for row in invocations]
        ),
        "start_ms": float(start_ms),
        "end_ms": float(end_ms),
        "duration_ms": float(end_ms - start_ms),
        "energy_j": 0.0,
        "post_nms_prediction_sha256": canonical_sha256(flattened),
    }


def _regret_loss(
    arm: _Arm, pipeline: _OfficialPopulationPipeline, *, invocation_id: str
) -> float:
    sample, _, _ = pipeline.materialize(invocation_id, regret=True)
    batch = _move_to_device(collate([sample]), torch.device("cuda:0"))
    deferred = arm.runtime.profile_deferred_cuda_events
    arm.runtime.profile_deferred_cuda_events = False
    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            output = arm.model(
                **batch,
                return_loss=True,
                chronotransport_per_window_output=True,
            )
    finally:
        arm.runtime.profile_deferred_cuda_events = deferred
    if not isinstance(output, ActionFormerPerWindowTrainOutput):
        raise RuntimeError("formal Gate4 regret path lacks per-window ActionFormer output")
    value = float(output.per_window_task_loss.detach().float().item())
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("formal Gate4 detector regret loss is invalid")
    return value


def build_formal_gate4_seed_shard(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    registration_relpath: str,
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path,
    pre_stage_c_gates23_replay_path: Path,
    pre_stage_c_gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
    post_stage_c_replay: Mapping[str, Any],
    post_stage_c_report: Mapping[str, Any],
    post_stage_c_unlock: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Execute one resumable seed shard on the registered official population."""

    _reject_config_overrides()
    registered = validate_formal_gate1_context(
        registration,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    population = validate_gate4_population_artifact(
        registered["gate4_population"]["artifact"]
    )
    post_unlock = validate_post_stage_c_gate3_unlock(
        post_stage_c_unlock,
        report=post_stage_c_report,
        replay=post_stage_c_replay,
    )
    if seed not in (3407, 3408, 3409) or isinstance(seed, bool):
        raise ValueError("formal Gate4 seed must be 3407, 3408, or 3409")
    scheduler = _scheduler_contract(gate1_unlock, post_unlock)
    observed = observe_formal_slurm_environment(registered["environment"])
    context = _load_validated_stage_c_seed(
        registration_path=registration_path,
        registration=registered,
        registration_commit=registration_commit,
        gate1_unlock_path=gate1_unlock_path,
        gates23_replay_path=pre_stage_c_gates23_replay_path,
        gates23_report_path=pre_stage_c_gates23_report_path,
        phase_marker_paths=phase_marker_paths,
        seed=seed,
        entrypoint_relative=GATE4_RUNNER_ENTRYPOINT,
    )
    pipeline = None
    arms = None
    sampler = None
    try:
        if context.binding != post_unlock["stage_c_bindings"][str(seed)]:
            raise ValueError("formal Gate4 Stage-C binding differs from post-Gate3 unlock")
        arms = _configure_arms(
            context,
            registration=registered,
            scheduler=scheduler,
            seed=seed,
        )
        pipeline = _OfficialPopulationPipeline(context.components.cfg, population)
        timing_rows = []
        audits = []
        raw_predictions = {arm: {} for arm in ARMS}
        blocks = population["timing_blocks"]
        for block in blocks:
            arm_payloads = {}
            for arm_name in block["arm_order"]:
                payload, audit, output = _invoke_timed_arm(
                    arms[arm_name],
                    pipeline,
                    invocation_id=block["invocation_id"],
                )
                arm_payloads[arm_name] = payload
                audits.append(
                    {
                        "seed": seed,
                        "invocation_id": block["invocation_id"],
                        "repetition_id": block["repetition_id"],
                        "arm": arm_name,
                        **audit,
                    }
                )
                video = block["official_video_id"]
                prior = raw_predictions[arm_name].get(block["invocation_id"])
                current = output[video]
                if prior is None:
                    raw_predictions[arm_name][block["invocation_id"]] = current
                elif prior != current:
                    raise RuntimeError("formal Gate4 repeated invocation is nondeterministic")
            timing_rows.append(
                {
                    "seed": seed,
                    "official_video_id": block["official_video_id"],
                    "invocation_id": block["invocation_id"],
                    "repetition_id": block["repetition_id"],
                    "invocation_order_index": block["invocation_order_index"],
                    "arm_order": list(block["arm_order"]),
                    "arms": arm_payloads,
                }
            )
        metric_predictions = {}
        # Full-video NMS is part of total latency but necessarily runs after all
        # window invocations.  Use the seed-balanced Latin order here as well so
        # CPU cache/warmup position cannot systematically favor one arm.
        for arm_name in GATE4_ENERGY_ARM_ORDER_BY_SEED[seed]:
            by_video = {video: [] for video in population["official_video_ids"]}
            for invocation in population["unique_invocations"]:
                by_video[invocation["official_video_id"]].extend(
                    raw_predictions[arm_name][invocation["invocation_id"]]
                )
            nms_start = perf_counter()
            finalized = _official_sliding_nms(by_video, pipeline.post_cfg)
            nms_ms = (perf_counter() - nms_start) * 1000.0
            per_block_share = nms_ms / len(timing_rows)
            for row in timing_rows:
                row["arms"][arm_name]["total_ms"] += per_block_share
                row["arms"][arm_name]["stage_ms"]["postprocess_ms"] += per_block_share
            metric_predictions[arm_name] = [
                {
                    "official_video_id": video,
                    "label": str(item["label"]),
                    "segment": [float(value) for value in item["segment"]],
                    "score": float(item["score"]),
                }
                for video in population["official_video_ids"]
                for item in finalized[video]
            ]

        regret_rows = []
        for invocation in population["unique_invocations"]:
            losses = {
                arm_name: _regret_loss(
                    arms[arm_name], pipeline, invocation_id=invocation["invocation_id"]
                )
                for arm_name in ARMS
            }
            regret_rows.append(
                {
                    "seed": seed,
                    "official_video_id": invocation["official_video_id"],
                    "invocation_id": invocation["invocation_id"],
                    "dense_detector_loss": losses["dense"],
                    "chronotransport_detector_loss": losses["chronotransport"],
                    "static_detector_loss": losses["static"],
                }
            )
        sampler = _NvmlPowerSampler(str(observed["gpu_uuid"]))
        sampler.start()
        energy_arm_order = list(GATE4_ENERGY_ARM_ORDER_BY_SEED[seed])
        energy_blocks = [
            _invoke_energy_block(
                arms[arm_name],
                pipeline,
                invocations=population["unique_invocations"],
                sampler=sampler,
            )
            for arm_name in energy_arm_order
        ]
        power_samples = sampler.stop()
        sampler = None
        energy_by_arm = {}
        for block in energy_blocks:
            block["energy_j"] = integrate_power_trace_j(
                power_samples,
                block["start_ms"],
                block["end_ms"],
            )
            energy_by_arm[block["arm"]] = float(block["energy_j"])
        for timing_row in timing_rows:
            for arm_name in ARMS:
                timing_row["arms"][arm_name]["nvml_energy_j"] = energy_by_arm[
                    arm_name
                ]
        return build_gate4_seed_shard(
            seed=seed,
            registration_sha256=registered["registration_sha256"],
            registration_commit=registration_commit,
            population_artifact_sha256=population["artifact_sha256"],
            post_stage_c_gate3_unlock_sha256=post_unlock["artifact_sha256"],
            stage_c_binding=context.binding,
            scheduler_contract=scheduler,
            observed_environment=observed,
            power_sampling_hz=_POWER_SAMPLING_HZ,
            power_samples=power_samples,
            energy_arm_order=energy_arm_order,
            energy_blocks=energy_blocks,
            timing_rows=timing_rows,
            execution_audit=audits,
            predictions=metric_predictions,
            regret_rows=regret_rows,
        )
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
        if pipeline is not None:
            pipeline.close()
        if context is not None:
            context.close()
        del arms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def precheck_formal_gate4_seed(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    registration_relpath: str,
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path,
    pre_stage_c_gates23_replay_path: Path,
    pre_stage_c_gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
    post_stage_c_replay: Mapping[str, Any],
    post_stage_c_report: Mapping[str, Any],
    post_stage_c_unlock: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Exercise one invocation through every arm without minting evidence."""

    _reject_config_overrides()
    registered = validate_formal_gate1_context(
        registration,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    population = validate_gate4_population_artifact(
        registered["gate4_population"]["artifact"]
    )
    unlock = validate_post_stage_c_gate3_unlock(
        post_stage_c_unlock,
        report=post_stage_c_report,
        replay=post_stage_c_replay,
    )
    scheduler = _scheduler_contract(gate1_unlock, unlock)
    observed = observe_formal_slurm_environment(registered["environment"])
    context = _load_validated_stage_c_seed(
        registration_path=registration_path,
        registration=registered,
        registration_commit=registration_commit,
        gate1_unlock_path=gate1_unlock_path,
        gates23_replay_path=pre_stage_c_gates23_replay_path,
        gates23_report_path=pre_stage_c_gates23_report_path,
        phase_marker_paths=phase_marker_paths,
        seed=seed,
        entrypoint_relative=GATE4_RUNNER_ENTRYPOINT,
    )
    pipeline = None
    sampler = None
    arms = None
    try:
        if context.binding != unlock["stage_c_bindings"][str(seed)]:
            raise ValueError("formal Gate4 precheck Stage-C binding mismatch")
        arms = _configure_arms(
            context,
            registration=registered,
            scheduler=scheduler,
            seed=seed,
        )
        pipeline = _OfficialPopulationPipeline(context.components.cfg, population)
        sampler = _NvmlPowerSampler(str(observed["gpu_uuid"]))
        sampler.start()
        invocation = population["unique_invocations"][0]
        for arm_name in ARMS:
            _invoke_timed_arm(
                arms[arm_name],
                pipeline,
                invocation_id=invocation["invocation_id"],
            )
            _regret_loss(
                arms[arm_name], pipeline, invocation_id=invocation["invocation_id"]
            )
        samples = sampler.stop()
        sampler = None
        if len(samples) < 2:
            raise RuntimeError("formal Gate4 precheck did not observe an NVML trace")
        return {
            "status": "PRECHECK_OK",
            "seed": seed,
            "registration_sha256": registered["registration_sha256"],
            "population_artifact_sha256": population["artifact_sha256"],
            "post_stage_c_gate3_unlock_sha256": unlock["artifact_sha256"],
            "stage_c_binding_sha256": canonical_sha256(context.binding),
            "observed_environment_sha256": observed["observed_environment_sha256"],
            "persisted_scientific_evidence": False,
        }
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
        if pipeline is not None:
            pipeline.close()
        context.close()
        del arms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_formal_gate4_evidence(
    *,
    registration: Mapping[str, Any],
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
    post_stage_c_unlock: Mapping[str, Any],
    post_stage_c_report: Mapping[str, Any],
    post_stage_c_replay: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    seed_shards: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Combine the exact three immutable seed shards into formal evidence files."""

    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    population = validate_gate4_population_artifact(
        registered["gate4_population"]["artifact"]
    )
    unlock = validate_post_stage_c_gate3_unlock(
        post_stage_c_unlock,
        report=post_stage_c_report,
        replay=post_stage_c_replay,
    )
    if not isinstance(seed_shards, Mapping) or set(seed_shards) != {
        "3407",
        "3408",
        "3409",
    }:
        raise ValueError("formal Gate4 evidence requires exactly three seed shards")
    shards = {
        str(seed): validate_gate4_seed_shard(seed_shards[str(seed)])
        for seed in (3407, 3408, 3409)
    }
    scheduler = _scheduler_contract(gate1_unlock, unlock)
    for seed, shard in shards.items():
        if (
            shard["registration_sha256"] != registered["registration_sha256"]
            or shard["registration_commit"] != unlock["registration_commit"]
            or shard["population_artifact_sha256"] != population["artifact_sha256"]
            or shard["post_stage_c_gate3_unlock_sha256"] != unlock["artifact_sha256"]
            or shard["stage_c_binding"] != unlock["stage_c_bindings"][seed]
            or shard["scheduler_contract"] != scheduler
        ):
            raise ValueError(f"formal Gate4 seed shard chain mismatch: {seed}")
    identity = {
        "registration_sha256": registered["registration_sha256"],
        "registration_commit": unlock["registration_commit"],
        "population_artifact_sha256": population["artifact_sha256"],
        "post_stage_c_gate3_unlock_sha256": unlock["artifact_sha256"],
        "stage_c_bindings": unlock["stage_c_bindings"],
        "scheduler_contract": scheduler,
        "seed_shard_artifact_sha256_by_seed": {
            seed: shard["artifact_sha256"] for seed, shard in shards.items()
        },
        "observed_environment_by_seed": {
            seed: shard["observed_environment"] for seed, shard in shards.items()
        },
    }
    timing_rows = [
        row for seed in ("3407", "3408", "3409") for row in shards[seed]["timing_rows"]
    ]
    execution_audit = [
        row
        for seed in ("3407", "3408", "3409")
        for row in shards[seed]["execution_audit"]
    ]
    regret_rows = [
        row for seed in ("3407", "3408", "3409") for row in shards[seed]["regret_rows"]
    ]
    metric = {
        "schema": "chronotransport-r2-gate4-metric-evidence-v1",
        "official_video_ids": list(population["official_video_ids"]),
        "fit_duration_quartile_thresholds": list(
            population["fit_duration_quartile_thresholds"]
        ),
        "ground_truth": deepcopy(population["ground_truth"]),
        "predictions": {
            seed: deepcopy(shards[seed]["predictions"])
            for seed in ("3407", "3408", "3409")
        },
    }
    return {
        "timing": build_gate4_timing_evidence(
            timing_rows,
            execution_audit=execution_audit,
            **identity,
        ),
        "metric": build_gate4_metric_evidence(metric, **identity),
        "regret": build_gate4_regret_evidence(regret_rows, **identity),
    }


__all__ = [
    "GATE4_RUNNER_ENTRYPOINT",
    "build_formal_gate4_evidence",
    "build_formal_gate4_seed_shard",
    "precheck_formal_gate4_seed",
]
