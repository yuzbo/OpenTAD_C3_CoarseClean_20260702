#!/usr/bin/env python3
"""One-job same-GPU ABBA+BAAB full-stack cost replay for residual centering."""

from __future__ import annotations

import argparse
import copy
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
    DYNAMIC_FLOOR_M2_SEED,
    require_clean_dynamic_floor_m2_checkout,
    validate_dynamic_floor_m2_checkpoint_sidecar,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_residual_centering_cost_contract import (  # noqa: E402
    RESIDUAL_CENTERING_COST_ORDER,
    RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS,
    RESIDUAL_CENTERING_COST_PROFILE_SCHEMA,
    RESIDUAL_CENTERING_COST_SAMPLE_SCHEMA,
    RESIDUAL_CENTERING_COST_STUDY_ID,
    RESIDUAL_CENTERING_COST_WARMUP_SAMPLES,
    build_residual_centering_cost_config,
    validate_residual_centering_cost_config,
    validate_residual_centering_cost_deployment,
    validate_residual_centering_cost_profile,
    validate_residual_centering_cost_source,
)
from tools.bata.georoute_residual_centering_training_contract import (  # noqa: E402
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
    residual_centering_training_variant_spec,
    validate_residual_centering_training_config,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402
from tools.bata.profile_georoute_dynamic_floor_m2 import (  # noqa: E402
    _build_cost_cuda_events,
    _cpu_ids,
    _inside,
    _invalid_cost_cuda_stages,
    _latency_summary,
    _population_descriptor,
    _read_cost_cuda_timings,
    _read_json,
    _validate_cost_audit,
    _write_jsonl,
)
from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
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


def _summary(values: Sequence[float]) -> dict[str, float]:
    return _latency_summary([float(value) for value in values])


def _validate_branch_audit(audit: Mapping[str, Any], *, variant: str) -> dict[str, Any]:
    summary = _validate_cost_audit(audit, floor_cells=1)
    branch = audit.get("branch_calibration")
    spec = residual_centering_training_variant_spec(variant)
    expected_mode = spec["branch_calibration_mode"]
    expected_scope = spec["branch_calibration_scope"]
    if (
        not isinstance(branch, Mapping)
        or branch.get("mode") != expected_mode
        or branch.get("scope") != expected_scope
        or branch.get("changes_q_base") is not False
        or branch.get("changes_delta_roi") is not False
        or branch.get("changes_context_zero_modifier") is not False
        or branch.get("changes_budget_or_role_quota") is not False
        or branch.get("mean_detached") is not False
    ):
        raise RuntimeError("residual-centering timed forward changed branch semantics")
    summary["branch_calibration_mode"] = expected_mode
    summary["branch_calibration_scope"] = expected_scope
    return summary


def _profile_one_pass(
    *,
    torch: Any,
    variant: str,
    pass_index: int,
    stage: Mapping[str, Any],
    expected_population_sha256: str,
    expected_accuracy_population_sha256: str,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from mmengine.config import Config
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores.test_engine import gather_ddp_results
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    cfg = build_residual_centering_cost_config(stage, variant=variant)
    binding = validate_residual_centering_cost_config(
        cfg, stage=stage, variant=variant
    )
    training_binding = dict(cfg.georoute_residual_centering_training_binding)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    runtime_ids = {str(row[0]) for row in dataset.data_list}
    if runtime_ids != set(training_binding["shared_protocol"]["evaluation_video_ids"]):
        raise ValueError("residual-centering cost population escaped development Gate")
    descriptors, population_sha256, accuracy_population_sha256 = (
        _population_descriptor(dataset)
    )
    if (
        population_sha256 != expected_population_sha256
        or accuracy_population_sha256 != expected_accuracy_population_sha256
    ):
        raise ValueError("residual-centering cost pass changed frozen population")
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
        raise ValueError("residual-centering cost loader changed population cardinality")

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint_path = Path(stage["checkpoint_receipt"]["path"])
    train_cfg = Config.fromfile(str(stage["config_receipts"]["train"]["path"]))
    observed_training_binding = validate_residual_centering_training_config(
        train_cfg, variant=variant, phase="train"
    )
    if dict(observed_training_binding) != training_binding:
        raise ValueError("residual-centering cost checkpoint/config binding changed")
    checkpoint_sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint_path,
        binding=train_cfg.georoute_dynamic_floor_m2_binding,
        cfg=train_cfg,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (
        checkpoint.get("experiment_metadata")
        != checkpoint_sidecar["experiment_metadata"]
        or int(checkpoint.get("epoch", -1)) != 59
        or "state_dict_ema" not in checkpoint
    ):
        raise ValueError("residual-centering cost checkpoint payload is invalid")
    model.load_state_dict(_strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True)
    del checkpoint
    model = model.to(device).eval()
    ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0)
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)
    if not bool(cfg.solver.amp):
        raise ValueError("residual-centering cost replay must preserve AMP inference")

    wrapper = getattr(model, "backbone", None)
    wrapped = getattr(wrapper, "model", None)
    heavy = getattr(wrapped, "backbone", None)
    events, method_events = _build_cost_cuda_events(
        torch, model=model, wrapper=wrapper, heavy=heavy
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
    for _ in range(RESIDUAL_CENTERING_COST_WARMUP_SAMPLES):
        forward_once(_move_to_device(next_batch(), device))
    synchronize()
    iterator = iter(loader)

    samples: list[dict[str, Any]] = []
    energy_windows: list[tuple[float, float]] = []
    video_rows: dict[str, list[dict[str, Any]]] = {}
    final_energy_window: tuple[float, float] | None = None
    try:
        for ordinal, descriptor in enumerate(descriptors):
            synchronize()
            continuous_started = time.perf_counter()
            energy_started = time.monotonic_ns() / 1_000_000_000.0
            cpu_batch, input_ms = _measure_wall_ms(next_batch, synchronize=synchronize)
            identity = _sample_identity(cpu_batch, ordinal)
            expected_physical = (
                f"{descriptor['video_id']}:{int(descriptor['window_center_first'])}"
            )
            if identity["physical_window_id"] != expected_physical:
                raise ValueError("residual-centering cost loader order changed")
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
                raise ValueError("residual-centering detector returned no result mapping")
            for video_id, rows in post_result.items():
                video_rows.setdefault(str(video_id), []).extend(rows)
            route_audit = _validate_branch_audit(
                getattr(wrapper, "latest_georoute_audit", None), variant=variant
            )
            synchronize()
            continuous_ended = time.perf_counter()
            energy_ended = time.monotonic_ns() / 1_000_000_000.0
            continuous_ms = (continuous_ended - continuous_started) * 1000.0
            component_timings = _read_cost_cuda_timings(events, method_events)
            invalid = _invalid_cost_cuda_stages(component_timings)
            if invalid:
                raise RuntimeError(
                    "residual-centering cost instrumentation missed CUDA stages: "
                    + ", ".join(invalid)
                )
            samples.append(
                {
                    "schema_version": RESIDUAL_CENTERING_COST_SAMPLE_SCHEMA,
                    "pass_index": int(pass_index),
                    "arm": variant,
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
                    "route_audit": route_audit,
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
            raise ValueError("residual-centering NMS returned non-Gate identities")
        amortized_nms_ms = (final_ended - final_started) * 1000.0 / len(samples)
        for sample in samples:
            sample["final_video_nms_ms"] = amortized_nms_ms
            sample["end_to_end_serial_ms"] += amortized_nms_ms
    finally:
        events.close()
        for event in reversed(tuple(method_events.values())):
            event.close()

    if final_energy_window is None:
        raise RuntimeError("residual-centering cost replay did not execute NMS")
    for sample, energy_window in zip(samples, energy_windows):
        sample["energy_window_monotonic_s"] = list(energy_window)
        sample["nms_energy_window_monotonic_s"] = list(final_energy_window)

    pass_receipt = {
        "pass_index": int(pass_index),
        "variant": variant,
        "branch_calibration_mode": binding["branch_calibration_mode"],
        "sample_count": len(samples),
        "population_sha256": population_sha256,
        "accuracy_population_sha256": expected_accuracy_population_sha256,
        "sample_manifest_sha256": canonical_sha256(
            [sample["window_id"] for sample in samples]
        ),
        "checkpoint_sha256": stage["checkpoint_receipt"]["sha256"],
        "bound_accuracy_config_sha256": stage["config_receipts"]["accuracy_a"][
            "sha256"
        ],
        "cost_config_sha256": canonical_sha256(cfg.to_dict()),
        "diagnostic_telemetry_inside_timed_forward": False,
        "training_or_resume_executed": False,
    }
    pass_receipt["pass_sha256"] = canonical_sha256(pass_receipt)
    del ddp_model, model, loader, dataset
    torch.cuda.empty_cache()
    return samples, pass_receipt


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from opentad.datasets import build_dataset

    model_runtime_commit = str(args.model_runtime_commit).lower()
    execution_commit = str(args.execution_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=execution_commit, root=ROOT
    )
    run_root = args.run_root.resolve()
    training_root = args.training_run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or not run_root.is_dir():
        raise ValueError("residual-centering cost run root leaves write boundary")
    source = validate_residual_centering_cost_source(
        training_root, expected_model_runtime_commit=model_runtime_commit
    )
    deployment_path = args.deployment.resolve()
    if deployment_path != (run_root / "control" / "deployment.json").resolve():
        raise ValueError("residual-centering cost deployment path changed")
    deployment = validate_residual_centering_cost_deployment(
        _read_json(deployment_path),
        run_root=run_root,
        training_run_root=training_root,
        expected_model_runtime_commit=model_runtime_commit,
        expected_execution_commit=execution_commit,
        expected_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    if (
        int(os.environ.get("WORLD_SIZE", -1)) != 1
        or int(os.environ.get("RANK", -1)) != 0
        or int(os.environ.get("LOCAL_RANK", -1)) != 0
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("residual-centering cost requires torchrun world1 cuda:0")
    if dist.is_initialized():
        raise RuntimeError("residual-centering cost requires a fresh process group")
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
        raise RuntimeError("residual-centering detector/sidecar CPU partition changed")
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

    expected_accuracy_population_sha256 = source["accuracy_population_sha256"]
    cost_population_hashes = set()
    for variant, stage in source["stages"].items():
        preflight_dataset = build_dataset(
            copy.deepcopy(
                build_residual_centering_cost_config(
                    stage, variant=variant
                ).dataset.test
            )
        )
        _, physical_population, telemetry_population = _population_descriptor(
            preflight_dataset
        )
        if telemetry_population != expected_accuracy_population_sha256:
            raise ValueError("residual-centering cost population differs from accuracy")
        cost_population_hashes.add(physical_population)
    if len(cost_population_hashes) != 1:
        raise ValueError("residual-centering cost arms changed physical population")
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
        raise ValueError("residual-centering power scratch must be node-local")
    scratch = scratch_root / f"job{os.environ['SLURM_JOB_ID']}_residual_centering"
    sampler = NvmlSidecarPowerSampler(
        expected_uuid=hardware_identity["nvidia_smi"]["uuid"],
        interval_ms=RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS,
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
        for pass_index, variant in enumerate(RESIDUAL_CENTERING_COST_ORDER):
            pass_samples, pass_receipt = _profile_one_pass(
                torch=torch,
                variant=variant,
                pass_index=pass_index,
                stage=source["stages"][variant],
                expected_population_sha256=expected_population_sha256,
                expected_accuracy_population_sha256=(
                    expected_accuracy_population_sha256
                ),
                device=device,
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
        for pass_index in range(len(RESIDUAL_CENTERING_COST_ORDER))
    }
    for sample in all_samples:
        start, end = map(float, sample["energy_window_monotonic_s"])
        nms_start, nms_end = map(float, sample["nms_energy_window_monotonic_s"])
        sample_energy = integrate_energy(sampler.samples, start=start, end=end)
        nms_energy = integrate_energy(
            sampler.samples, start=nms_start, end=nms_end
        )
        if sample_energy is None or nms_energy is None:
            raise RuntimeError("residual-centering power trace has incomplete coverage")
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
    by_pass: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        grouped[str(sample["arm"])].append(sample)
        by_pass[int(sample["pass_index"])].append(sample)
    arm_summaries = {}
    for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
        rows = grouped[variant]
        gross_energy = sum(float(row["gpu_energy_j"]) for row in rows)
        arm_summaries[variant] = {
            "pass_count": sum(
                ordered == variant for ordered in RESIDUAL_CENTERING_COST_ORDER
            ),
            "sample_count": len(rows),
            "population_sha256": expected_population_sha256,
            "latency_ms": {
                key: _summary([row[key] for row in rows]) for key in latency_keys
            },
            "resources": {
                "peak_gpu_allocated_mb": max(
                    row["peak_gpu_allocated_mb"] for row in rows
                ),
                "peak_gpu_reserved_mb": max(
                    row["peak_gpu_reserved_mb"] for row in rows
                ),
                "gross_gpu_energy_j": gross_energy,
                "gpu_energy_j_per_sample": gross_energy / len(rows),
            },
        }
    pass_summaries = []
    for pass_index, variant in enumerate(RESIDUAL_CENTERING_COST_ORDER):
        rows = by_pass[pass_index]
        pass_summaries.append(
            {
                "pass_index": pass_index,
                "variant": variant,
                "sample_count": len(rows),
                "metrics": {
                    key: _summary([row[key] for row in rows])
                    for key in (*latency_keys, "gpu_energy_j")
                },
            }
        )

    profile_payload: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_COST_PROFILE_SCHEMA,
        "status": "PASS_RESIDUAL_CENTERING_PAIRED_FULL_STACK_COST",
        "study_id": RESIDUAL_CENTERING_COST_STUDY_ID,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "model_runtime_commit": model_runtime_commit,
        "execution_commit": execution_commit,
        "run_root": str(run_root),
        "training_run_root": str(training_root),
        "profile_order": list(RESIDUAL_CENTERING_COST_ORDER),
        "warmup_samples_per_pass": RESIDUAL_CENTERING_COST_WARMUP_SAMPLES,
        "batch_size": 1,
        "loader_workers": 0,
        "world_size": 1,
        "power_interval_ms": RESIDUAL_CENTERING_COST_POWER_INTERVAL_MS,
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
            "continuous_power_sidecar": True,
            "development_only": True,
        },
        "measurement_note": (
            "input_pipeline_serial_ms encloses serial decode, preprocess, and "
            "collate with num_workers=0; video NMS is amortized per window"
        ),
        "arm_summaries": arm_summaries,
        "pass_summaries": pass_summaries,
        "pass_receipts": pass_receipts,
        "stage_result_receipts": source["stage_result_receipts"],
        "training_finalization_receipt": source[
            "training_finalization_receipt"
        ],
        "deployment_receipt": {
            "path": str(deployment_path),
            "sha256": sha256_file(deployment_path),
            "deployment_sha256": deployment["deployment_sha256"],
        },
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
        "training_or_resume_executed": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    profile_payload["profile_sha256"] = canonical_sha256(profile_payload)
    validate_residual_centering_cost_profile(
        profile_payload,
        expected_model_runtime_commit=model_runtime_commit,
        expected_execution_commit=execution_commit,
    )
    _atomic_write_json(profile_path, profile_payload)
    dist.destroy_process_group()
    return profile_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--model-runtime-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
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
