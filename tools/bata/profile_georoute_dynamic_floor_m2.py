#!/usr/bin/env python3
"""Counterbalanced same-GPU full-stack cost replay for dynamic floor M2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_ARM_ORDER,
    DYNAMIC_FLOOR_M2_COST_ORDER,
    DYNAMIC_FLOOR_M2_COST_SCHEMA,
    DYNAMIC_FLOOR_M2_SEED,
    DYNAMIC_FLOOR_M2_STUDY_ID,
    DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
    build_dynamic_floor_m2_cost_config,
    require_clean_dynamic_floor_m2_checkout,
    validate_dynamic_floor_m2_checkpoint_sidecar,
    validate_dynamic_floor_m2_config,
    validate_dynamic_floor_m2_cost_profile,
    validate_dynamic_floor_m2_stage_result,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402
from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
    CudaMethodEvent,
    CudaModuleEvents,
    _hardware_identity,
    _measure_wall_ms,
    _move_to_device,
    _sample_identity,
    _software_identity,
    _strip_ddp_prefix,
    integrate_energy,
)
from tools.bata.spatial_zoom_s1_power import (  # noqa: E402
    NvmlSidecarPowerSampler,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_slurm_memory_limit_mb,
    require_slurm_single_gpu_allocation,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
WARMUP_SAMPLES = 50
POWER_INTERVAL_MS = 20


def _build_cost_cuda_events(
    torch_module: Any,
    *,
    model: Any,
    wrapper: Any,
    heavy: Any,
) -> tuple[CudaModuleEvents, dict[str, CudaMethodEvent]]:
    module_targets = {
        "backbone_wrapper_ms": wrapper,
        "patch_embed_ms": getattr(heavy, "patch_embed", None),
        "projection_ms": getattr(model, "projection", None),
        "neck_ms": getattr(model, "neck", None),
    }
    method_targets = {
        "scout_ms": (getattr(wrapper, "scout", None), "forward_dynamic"),
        "heavy_backbone_ms": (heavy, "forward_native_ragged"),
        "sparse_adapter_ms": (
            getattr(wrapper, "sparse_adapter", None),
            "forward_ragged",
        ),
        "head_ms": (getattr(model, "rpn_head", None), "forward_test"),
        "model_forward_ms": (model, "forward_test"),
        "postprocess_ms": (model, "post_processing"),
    }
    missing_bindings = sorted(
        [name for name, target in module_targets.items() if target is None]
        + [
            name
            for name, (target, method_name) in method_targets.items()
            if target is None or not callable(getattr(target, method_name, None))
        ]
    )
    if missing_bindings:
        raise RuntimeError(
            "dynamic floor M2 cost instrumentation has missing bindings: "
            + ", ".join(missing_bindings)
        )

    module_events = CudaModuleEvents(torch_module)
    method_events: dict[str, CudaMethodEvent] = {}
    try:
        for name, target in module_targets.items():
            module_events.register(name, target)
        for name, (target, method_name) in method_targets.items():
            method_events[name] = CudaMethodEvent(
                torch_module,
                target,
                method_name,
            )
    except Exception:
        for event in reversed(tuple(method_events.values())):
            event.close()
        module_events.close()
        raise
    return module_events, method_events


def _read_cost_cuda_timings(
    module_events: CudaModuleEvents,
    method_events: Mapping[str, CudaMethodEvent],
) -> dict[str, float]:
    return {
        "model_forward_ms": method_events["model_forward_ms"].elapsed(),
        "postprocess_ms": method_events["postprocess_ms"].elapsed(),
        "backbone_wrapper_ms": module_events.elapsed("backbone_wrapper_ms"),
        "scout_ms": method_events["scout_ms"].elapsed(),
        "patch_embed_ms": module_events.elapsed("patch_embed_ms"),
        "heavy_backbone_ms": method_events["heavy_backbone_ms"].elapsed(),
        "sparse_adapter_ms": method_events["sparse_adapter_ms"].elapsed(),
        "projection_ms": module_events.elapsed("projection_ms"),
        "neck_ms": module_events.elapsed("neck_ms"),
        "head_ms": method_events["head_ms"].elapsed(),
    }


def _invalid_cost_cuda_stages(timings: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in timings.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        )
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _cpu_ids(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(int(item) for item in value.split(",") if item))
    except ValueError as error:
        raise ValueError(f"invalid CPU list {value!r}") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid CPU list {value!r}")
    return values


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty cost distribution")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    checked = [float(value) for value in values]
    if not checked or any(value < 0.0 for value in checked):
        raise ValueError("cost latency samples must be non-negative")
    return {
        "mean": sum(checked) / len(checked),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _population_descriptor(
    dataset: Any,
) -> tuple[list[dict[str, Any]], str, str]:
    descriptors: list[dict[str, Any]] = []
    telemetry_descriptors: list[dict[str, Any]] = []
    for dataset_index, row in enumerate(dataset.data_list):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            raise ValueError("dynamic floor M2 cost dataset row is malformed")
        centers = row[3]
        if not len(centers):
            raise ValueError("dynamic floor M2 cost dataset contains an empty window")
        telemetry_descriptor = {
            "dataset_index": dataset_index,
            "rank": 0,
            "local_batch_index": dataset_index,
            "video_id": str(row[0]),
            "window_center_count": int(len(centers)),
            "window_center_first": float(centers[0]) if len(centers) else None,
            "window_center_last": float(centers[-1]) if len(centers) else None,
        }
        telemetry_descriptor["window_descriptor_sha256"] = canonical_sha256(
            telemetry_descriptor
        )
        telemetry_descriptors.append(telemetry_descriptor)
        descriptor = {
            **telemetry_descriptor,
            "window_centers": [float(center) for center in centers],
        }
        descriptor["physical_descriptor_sha256"] = canonical_sha256(descriptor)
        descriptors.append(descriptor)
    if not descriptors:
        raise ValueError("dynamic floor M2 cost population is empty")
    return (
        descriptors,
        canonical_sha256(descriptors),
        canonical_sha256(telemetry_descriptors),
    )


def _validate_cost_audit(audit: Any, *, floor_cells: int) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise RuntimeError("dynamic floor M2 cost forward emitted no route audit")
    if (
        audit.get("route_mode") != "dynamic_scnr"
        or audit.get("policy_estimator") != "straight_through"
        or int(audit.get("window_token_budget", -1))
        != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or audit.get("window_budget_is_global") is not True
        or audit.get("fixed_context_quota") is not False
        or audit.get("fixed_per_tubelet_k") is not False
        or audit.get("k_t_allows_zero") is not True
        or audit.get("zero_carrier_mode") != "masked_zero"
        or int(audit.get("requested_physical_tokens_per_window", -1))
        != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or int(audit.get("unique_physical_tokens_per_window", -1))
        != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or int(audit.get("executed_patch_tokens_per_window", -1))
        != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or int(audit.get("padded_heavy_tokens_per_window", -1)) != 0
        or int(audit.get("heavy_backbone_forward_count", -1)) != 1
        or audit.get("geometry_extent_floor_mode") != "native_cells"
        or int(audit.get("geometry_extent_floor_cells", -1)) != int(floor_cells)
        or audit.get("diagnostic_telemetry_enabled") is not False
        or audit.get("uses_gt_for_route") is not False
        or audit.get("uses_teacher") is not False
        or audit.get("uses_oracle") is not False
        or audit.get("uses_test_evidence") is not False
    ):
        raise RuntimeError("dynamic floor M2 timed forward violated route isolation")
    k_rows = audit.get("k_per_tubelet")
    if (
        not isinstance(k_rows, list)
        or len(k_rows) != 1
        or len(k_rows[0]) != 384
        or sum(map(int, k_rows[0])) != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
    ):
        raise RuntimeError("dynamic floor M2 timed forward lost exact K_t attribution")
    packed = audit.get("packed")
    if not isinstance(packed, Mapping):
        raise RuntimeError("dynamic floor M2 timed forward emitted no ragged ledger")
    clip_rows = packed.get("clip_token_counts")
    pair_rows = packed.get("attention_pairs_per_window")
    if (
        packed.get("schema_version") != "videomae_native_ragged_v1"
        or packed.get("execution_mode") != "true_clip_ragged_no_padding"
        or int(packed.get("batch_size", -1)) != 1
        or not isinstance(clip_rows, list)
        or len(clip_rows) != 1
        or not isinstance(clip_rows[0], list)
        or not isinstance(pair_rows, list)
        or len(pair_rows) != 1
    ):
        raise RuntimeError("dynamic floor M2 timed forward ragged ledger is invalid")
    try:
        clip_counts = [int(value) for value in clip_rows[0]]
        attention_pairs = int(pair_rows[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "dynamic floor M2 timed forward ragged ledger is invalid"
        ) from error
    if (
        any(value < 0 for value in clip_counts)
        or sum(clip_counts) != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or sum(value**2 for value in clip_counts) != attention_pairs
    ):
        raise RuntimeError("dynamic floor M2 timed forward ragged ledger is invalid")
    return {
        "physical_indices_sha256": str(audit["physical_indices_sha256"]),
        "k_t_min": int(audit["k_t_min"]),
        "k_t_max": int(audit["k_t_max"]),
        "k_t_zero_count": int(audit["k_t_zero_count"]),
        "role_counts": dict(audit["role_counts"]),
        "attention_pairs": attention_pairs,
        "exact_window_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "padded_heavy_tokens": 0,
    }


def _profile_one_pass(
    *,
    torch: Any,
    arm: str,
    pass_index: int,
    stage: Mapping[str, Any],
    expected_population_sha256: str,
    expected_accuracy_population_sha256: str,
    device: Any,
    power_sampler: NvmlSidecarPowerSampler,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from mmengine.config import Config
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores.test_engine import gather_ddp_results
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    cfg = build_dynamic_floor_m2_cost_config(stage, arm=arm)
    binding = dict(cfg.georoute_dynamic_floor_m2_binding)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    runtime_ids = {str(row[0]) for row in dataset.data_list}
    if runtime_ids != set(binding["evaluation_video_ids"]):
        raise ValueError("dynamic floor M2 cost population escaped development Gate")
    (
        descriptors,
        population_sha256,
        accuracy_population_sha256,
    ) = _population_descriptor(dataset)
    if (
        population_sha256 != expected_population_sha256
        or accuracy_population_sha256 != expected_accuracy_population_sha256
    ):
        raise ValueError("dynamic floor M2 cost pass changed the frozen population")
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=0,
    )
    if len(loader) != len(descriptors):
        raise ValueError("dynamic floor M2 cost loader changed population cardinality")

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint_path = Path(stage["checkpoint_receipt"]["path"])
    train_cfg = Config.fromfile(str(stage["config_receipts"]["train"]["path"]))
    if dict(
        validate_dynamic_floor_m2_config(train_cfg, arm=arm, phase="train")
    ) != binding:
        raise ValueError("dynamic floor M2 cost checkpoint/config binding changed")
    checkpoint_sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint_path,
        binding=binding,
        cfg=train_cfg,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint.get("experiment_metadata")
        != checkpoint_sidecar["experiment_metadata"]
        or int(checkpoint.get("epoch", -1)) != 59
        or "state_dict_ema" not in checkpoint
    ):
        raise ValueError("dynamic floor M2 cost checkpoint payload is invalid")
    model.load_state_dict(
        _strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True
    )
    del checkpoint
    model = model.to(device).eval()
    ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0)
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)
    use_amp = bool(cfg.solver.amp)
    if not use_amp:
        raise ValueError("dynamic floor M2 cost replay must preserve AMP inference")

    wrapper = getattr(model, "backbone", None)
    wrapped = getattr(wrapper, "model", None)
    heavy = getattr(wrapped, "backbone", None)
    events, method_events = _build_cost_cuda_events(
        torch,
        model=model,
        wrapper=wrapper,
        heavy=heavy,
    )

    def forward_once(batch):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            return ddp_model(
                **batch,
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=external_cls,
            )

    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    set_seed(DYNAMIC_FLOOR_M2_SEED, deterministic_warn_only=True)
    for _ in range(WARMUP_SAMPLES):
        forward_once(_move_to_device(next_batch(), device))
    synchronize()
    iterator = iter(loader)

    samples: list[dict[str, Any]] = []
    energy_windows: list[tuple[float, float]] = []
    video_rows: dict[str, list[dict[str, Any]]] = {}
    final_energy_window: tuple[float, float] | None = None
    floor_cells = int(binding["arm_spec"]["roi_extent_floor_cells"])
    try:
        for ordinal, descriptor in enumerate(descriptors):
            synchronize()
            continuous_started = time.perf_counter()
            energy_started = time.monotonic_ns() / 1_000_000_000.0
            cpu_batch, input_ms = _measure_wall_ms(
                next_batch, synchronize=synchronize
            )
            identity = _sample_identity(cpu_batch, ordinal)
            expected_physical = (
                f"{descriptor['video_id']}:"
                f"{int(descriptor['window_center_first'])}"
            )
            if identity["physical_window_id"] != expected_physical:
                raise ValueError("dynamic floor M2 cost loader order changed")
            torch.cuda.reset_peak_memory_stats(device)
            gpu_batch, h2d_ms = _measure_wall_ms(
                lambda: _move_to_device(cpu_batch, device), synchronize=synchronize
            )
            events.reset()
            for event in method_events.values():
                event.reset()
            post_result, _ = _measure_wall_ms(
                lambda: forward_once(gpu_batch), synchronize=synchronize
            )
            if not isinstance(post_result, Mapping):
                raise ValueError("dynamic floor M2 detector returned no result mapping")
            for video_id, rows in post_result.items():
                video_rows.setdefault(str(video_id), []).extend(rows)
            audit = _validate_cost_audit(
                getattr(wrapper, "latest_georoute_audit", None),
                floor_cells=floor_cells,
            )
            synchronize()
            continuous_ended = time.perf_counter()
            energy_ended = time.monotonic_ns() / 1_000_000_000.0
            continuous_ms = (continuous_ended - continuous_started) * 1000.0
            component_timings = _read_cost_cuda_timings(events, method_events)
            invalid_stages = _invalid_cost_cuda_stages(component_timings)
            if invalid_stages:
                raise RuntimeError(
                    "dynamic floor M2 cost instrumentation missed CUDA stages: "
                    + ", ".join(invalid_stages)
                )
            samples.append(
                {
                    "schema_version": "scnr_dynamic_floor_m2_cost_sample_v1",
                    "pass_index": int(pass_index),
                    "arm": arm,
                    "sample_ordinal": ordinal,
                    "population_sha256": population_sha256,
                    "input_pipeline_serial_ms": input_ms,
                    "h2d_ms": h2d_ms,
                    **component_timings,
                    "decode_to_window_output_wall_ms": continuous_ms,
                    "final_video_nms_ms": 0.0,
                    "end_to_end_serial_ms": continuous_ms,
                    "peak_gpu_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                    ),
                    "peak_gpu_reserved_mb": (
                        torch.cuda.max_memory_reserved(device) / (1024**2)
                    ),
                    "gpu_energy_j": None,
                    "route_audit": audit,
                    **identity,
                }
            )
            energy_windows.append((energy_started, energy_ended))
            del cpu_batch, gpu_batch, post_result
        synchronize()
        final_started = time.monotonic_ns() / 1_000_000_000.0
        finalized = gather_ddp_results(1, video_rows, cfg.post_processing)
        synchronize()
        final_ended = time.monotonic_ns() / 1_000_000_000.0
        final_energy_window = (final_started, final_ended)
        if not isinstance(finalized, Mapping) or not set(map(str, finalized)) <= runtime_ids:
            raise ValueError("dynamic floor M2 NMS returned non-Gate identities")
        amortized_nms_ms = (final_ended - final_started) * 1000.0 / len(samples)
        for sample in samples:
            sample["final_video_nms_ms"] = amortized_nms_ms
            sample["end_to_end_serial_ms"] += amortized_nms_ms
    finally:
        events.close()
        for event in reversed(tuple(method_events.values())):
            event.close()

    if final_energy_window is None:
        raise RuntimeError("dynamic floor M2 cost replay did not execute NMS")
    for sample, energy_window in zip(samples, energy_windows):
        sample["energy_window_monotonic_s"] = list(energy_window)
        sample["nms_energy_window_monotonic_s"] = list(final_energy_window)

    pass_receipt = {
        "pass_index": int(pass_index),
        "arm": arm,
        "sample_count": len(samples),
        "population_sha256": population_sha256,
        "accuracy_population_sha256": expected_accuracy_population_sha256,
        "sample_manifest_sha256": canonical_sha256(
            [sample["window_id"] for sample in samples]
        ),
        "checkpoint_sha256": stage["checkpoint_receipt"]["sha256"],
        "bound_accuracy_config_sha256": stage["config_receipts"]["accuracy"][
            "sha256"
        ],
        "cost_config_sha256": canonical_sha256(cfg.to_dict()),
        "diagnostic_telemetry_inside_timed_forward": False,
    }
    pass_receipt["pass_sha256"] = canonical_sha256(pass_receipt)
    del ddp_model, model, loader, dataset
    torch.cuda.empty_cache()
    return samples, pass_receipt


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from opentad.datasets import build_dataset

    expected_commit = str(args.expected_commit).lower()
    expected_execution_commit = str(args.expected_execution_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=expected_execution_commit, root=ROOT
    )
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("dynamic floor M2 cost run root leaves write boundary")
    if (
        int(os.environ.get("WORLD_SIZE", -1)) != 1
        or int(os.environ.get("RANK", -1)) != 0
        or int(os.environ.get("LOCAL_RANK", -1)) != 0
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("dynamic floor M2 cost requires torchrun world1 cuda:0")
    if dist.is_initialized():
        raise RuntimeError("dynamic floor M2 cost requires a fresh process group")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    allocated = _cpu_ids(args.allocated_cpus)
    detector = _cpu_ids(args.detector_cpus)
    sidecar_cpu = int(args.sidecar_cpu)
    if (
        len(allocated) != 5
        or len(detector) != 4
        or set(detector) | {sidecar_cpu} != set(allocated)
        or sidecar_cpu in detector
        or tuple(sorted(os.sched_getaffinity(0))) != detector
        or int(os.environ.get("SLURM_CPUS_PER_TASK", -1)) != 5
    ):
        raise RuntimeError("dynamic floor M2 detector/sidecar CPU partition changed")
    physical_gpu = require_slurm_single_gpu_allocation()
    memory_limit_mb = require_slurm_memory_limit_mb(minimum_mb=1)
    software_identity = _software_identity(torch)
    hardware_identity = _hardware_identity(
        torch,
        device,
        physical_gpu_id=physical_gpu,
        allocated_cpu_ids=allocated,
        detector_cpu_ids=detector,
        sidecar_cpu_id=sidecar_cpu,
        memory_limit_mb=memory_limit_mb,
    )
    software_fingerprint = canonical_sha256(software_identity)
    hardware_fingerprint = canonical_sha256(hardware_identity)

    stage_paths = {
        DYNAMIC_FLOOR_M2_ARM_ORDER[0]: args.stage_result_g1.resolve(),
        DYNAMIC_FLOOR_M2_ARM_ORDER[1]: args.stage_result_g2.resolve(),
    }
    stages = {
        arm: validate_dynamic_floor_m2_stage_result(
            _read_json(path), expected_arm=arm, expected_commit=expected_commit
        )
        for arm, path in stage_paths.items()
    }
    stage_result_receipts = {
        arm: {
            "path": str(stage_paths[arm]),
            "sha256": sha256_file(stage_paths[arm]),
            "stage_result_sha256": stages[arm]["stage_result_sha256"],
        }
        for arm in DYNAMIC_FLOOR_M2_ARM_ORDER
    }
    population_hashes = {
        stage["population_sha256"] for stage in stages.values()
    }
    if len(population_hashes) != 1:
        raise ValueError("dynamic floor M2 accuracy arms used different populations")
    expected_accuracy_population_sha256 = population_hashes.pop()
    cost_population_hashes = set()
    for arm, stage in stages.items():
        preflight_dataset = build_dataset(
            copy.deepcopy(
                build_dynamic_floor_m2_cost_config(stage, arm=arm).dataset.test
            )
        )
        _, cost_population, telemetry_population = _population_descriptor(
            preflight_dataset
        )
        if telemetry_population != expected_accuracy_population_sha256:
            raise ValueError(
                "dynamic floor M2 cost population differs from accuracy replay"
            )
        cost_population_hashes.add(cost_population)
    if len(cost_population_hashes) != 1:
        raise ValueError("dynamic floor M2 cost arms changed physical population")
    expected_population_sha256 = cost_population_hashes.pop()

    cost_root = run_root / "cost"
    cost_root.mkdir(parents=True, exist_ok=False)
    samples_path = cost_root / "paired_cost_samples.jsonl"
    power_path = cost_root / "paired_cost_power.jsonl"
    profile_path = cost_root / "paired_cost_profile.json"
    sidecar_prefix = cost_root / "paired_cost_sidecar"
    scratch_root = args.power_scratch_root.resolve()
    if not (
        str(scratch_root).startswith("/tmp/")
        or str(scratch_root).startswith("/var/tmp/")
    ):
        raise ValueError("dynamic floor M2 power scratch must be node-local")
    scratch = scratch_root / (
        f"job{os.environ['SLURM_JOB_ID']}_dynamic_floor_m2"
    )
    sampler = NvmlSidecarPowerSampler(
        expected_uuid=hardware_identity["nvidia_smi"]["uuid"],
        interval_ms=POWER_INTERVAL_MS,
        scratch_dir=scratch,
        attempt_prefix=sidecar_prefix,
        sidecar_cpu_id=sidecar_cpu,
        detector_cpu_ids=detector,
        allocated_cpu_ids=allocated,
    )
    all_samples: list[dict[str, Any]] = []
    pass_receipts: list[dict[str, Any]] = []
    sampler.start()
    time.sleep(sampler.interval_s * 1.5)
    try:
        for pass_index, arm in enumerate(DYNAMIC_FLOOR_M2_COST_ORDER):
            pass_samples, pass_receipt = _profile_one_pass(
                torch=torch,
                arm=arm,
                pass_index=pass_index,
                stage=stages[arm],
                expected_population_sha256=expected_population_sha256,
                expected_accuracy_population_sha256=(
                    expected_accuracy_population_sha256
                ),
                device=device,
                power_sampler=sampler,
            )
            all_samples.extend(pass_samples)
            pass_receipts.append(pass_receipt)
    finally:
        time.sleep(sampler.interval_s * 1.5)
        sampler.stop()

    pass_counts = {
        pass_index: sum(
            int(sample["pass_index"]) == pass_index for sample in all_samples
        )
        for pass_index in range(len(DYNAMIC_FLOOR_M2_COST_ORDER))
    }
    for sample in all_samples:
        start, end = map(float, sample["energy_window_monotonic_s"])
        nms_start, nms_end = map(float, sample["nms_energy_window_monotonic_s"])
        sample_energy = integrate_energy(sampler.samples, start=start, end=end)
        nms_energy = integrate_energy(
            sampler.samples, start=nms_start, end=nms_end
        )
        if sample_energy is None or nms_energy is None:
            raise RuntimeError("dynamic floor M2 power trace has incomplete coverage")
        sample["gpu_energy_j"] = sample_energy + nms_energy / pass_counts[
            int(sample["pass_index"])
        ]
        sample["sample_sha256"] = canonical_sha256(sample)

    power_origin = sampler.samples[0][0]
    power_rows = [
        {
            "sequence": index,
            "monotonic_s": timestamp,
            "timestamp_ms": (timestamp - power_origin) * 1000.0,
            "power_w": power,
        }
        for index, (timestamp, power) in enumerate(sampler.samples)
    ]
    _write_jsonl(samples_path, all_samples)
    _write_jsonl(power_path, power_rows)

    latency_keys = (
        "input_pipeline_serial_ms",
        "h2d_ms",
        "model_forward_ms",
        "postprocess_ms",
        "decode_to_window_output_wall_ms",
        "final_video_nms_ms",
        "end_to_end_serial_ms",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        grouped[str(sample["arm"])].append(sample)
    arm_summaries = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        rows = grouped[arm]
        arm_summaries[arm] = {
            "pass_count": sum(
                ordered_arm == arm for ordered_arm in DYNAMIC_FLOOR_M2_COST_ORDER
            ),
            "sample_count": len(rows),
            "population_sha256": expected_population_sha256,
            "latency_ms": {
                key: _latency_summary([row[key] for row in rows])
                for key in latency_keys
            },
            "resources": {
                "peak_gpu_allocated_mb": max(
                    row["peak_gpu_allocated_mb"] for row in rows
                ),
                "peak_gpu_reserved_mb": max(
                    row["peak_gpu_reserved_mb"] for row in rows
                ),
                "gross_gpu_energy_j": sum(row["gpu_energy_j"] for row in rows),
            },
        }

    profile_payload: dict[str, Any] = {
        "schema_version": DYNAMIC_FLOOR_M2_COST_SCHEMA,
        "status": "PASS_DYNAMIC_FLOOR_M2_FULL_STACK_COST",
        "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "runtime_commit": expected_commit,
        "execution_commit": expected_execution_commit,
        "run_root": str(run_root),
        "profile_order": list(DYNAMIC_FLOOR_M2_COST_ORDER),
        "warmup_samples_per_pass": WARMUP_SAMPLES,
        "batch_size": 1,
        "loader_workers": 0,
        "world_size": 1,
        "power_interval_ms": POWER_INTERVAL_MS,
        "population_sha256": expected_population_sha256,
        "accuracy_population_sha256": expected_accuracy_population_sha256,
        "raw_sample_count": len(all_samples),
        "scope": {
            "decode": True,
            "preprocess": True,
            "host_to_device": True,
            "scout": True,
            "route": True,
            "patch_embed": True,
            "backbone": True,
            "adapter": True,
            "detector": True,
            "nms": True,
            "diagnostic_telemetry_inside_timed_forward": False,
            "same_gpu_counterbalanced": True,
            "development_only": True,
        },
        "measurement_note": (
            "input_pipeline_serial_ms encloses serial decode, preprocess, and "
            "collate with num_workers=0"
        ),
        "arm_summaries": arm_summaries,
        "pass_receipts": pass_receipts,
        "stage_result_receipts": stage_result_receipts,
        "hardware_identity": hardware_identity,
        "hardware_fingerprint": hardware_fingerprint,
        "software_identity": software_identity,
        "software_fingerprint": software_fingerprint,
        "artifact_receipts": {
            "raw_samples": {
                "path": str(samples_path.resolve()),
                "sha256": sha256_file(samples_path),
            },
            "power_trace": {
                "path": str(power_path.resolve()),
                "sha256": sha256_file(power_path),
            },
            "sidecar_attempt_report": {
                "path": str(sampler.attempt_report_path.resolve()),
                "sha256": sha256_file(sampler.attempt_report_path),
            },
            "sidecar_attempt_trace": {
                "path": str(sampler.attempt_trace_path.resolve()),
                "sha256": sha256_file(sampler.attempt_trace_path),
            },
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"),
            "allocated_cpu_ids": list(allocated),
            "detector_cpu_ids": list(detector),
            "sidecar_cpu_id": sidecar_cpu,
            "logical_device": "cuda:0",
        },
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    profile_payload["profile_sha256"] = canonical_sha256(profile_payload)
    validate_dynamic_floor_m2_cost_profile(
        profile_payload, expected_commit=expected_commit
    )
    _atomic_write_json(profile_path, profile_payload)
    dist.destroy_process_group()
    return profile_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage-result-g1", type=Path, required=True)
    parser.add_argument("--stage-result-g2", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-execution-commit", required=True)
    parser.add_argument("--allocated-cpus", required=True)
    parser.add_argument("--detector-cpus", required=True)
    parser.add_argument("--sidecar-cpu", required=True, type=int)
    parser.add_argument("--power-scratch-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = profile(_parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
