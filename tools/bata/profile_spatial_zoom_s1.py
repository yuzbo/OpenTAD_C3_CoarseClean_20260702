from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_PROFILE_ORDER_SEED,
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_cost import (  # noqa: E402
    S1_PROFILE_PROTOCOL,
    build_profile_summary,
    compare_resolution_profiles,
    validate_profile_summary,
    write_profile_summary,
)
from tools.bata.validate_spatial_zoom_s1 import validate_config_matrix  # noqa: E402
from tools.bata.spatial_zoom_s1_test_open import (  # noqa: E402
    validate_test_open_certificate,
)
from tools.bata.spatial_zoom_s1_evidence import (  # noqa: E402
    validate_s1_checkpoint_metadata_for_binding,
    validate_s1_test_evidence,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_clean_git_checkout,
    require_gpu1_allocation,
    validate_bound_s1_training_config,
    validate_s1_checkpoint_sidecar,
)

S1_PROFILE_ATTEMPT_SCHEMA = "spatial_zoom_s1_profile_attempt_v3"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_profile_attempt_marker(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path).resolve()
    marker = {
        "schema_version": S1_PROFILE_ATTEMPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return marker


def validate_profile_attempt_marker(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("S1 profile-attempt marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    if marker.get("schema_version") != S1_PROFILE_ATTEMPT_SCHEMA:
        raise ValueError("unsupported S1 profile-attempt marker schema")
    return marker


def validate_profile_order_ready(
    *,
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    resolution: int,
    seed: int,
    hardware_fingerprint: str | None,
    software_fingerprint: str | None,
) -> tuple[dict[str, int], str]:
    order = build_s1_profile_order()
    if manifest.get("profile_matrix_order") != order:
        raise ValueError("S1 manifest profile order differs from the frozen schedule")
    matches = [
        row
        for row in order
        if int(row["resolution"]) == int(resolution) and int(row["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError("S1 profile cell has no unique frozen schedule ordinal")
    current = matches[0]
    canonical_root = Path(binding["canonical_experiment_root"])
    for row in order:
        prefix = (
            canonical_root
            / f"dense{int(row['resolution'])}"
            / f"seed{int(row['seed'])}"
            / "profile"
            / f"dense{int(row['resolution'])}_seed{int(row['seed'])}"
        )
        marker_path = prefix.with_suffix(".started.json")
        summary_path = prefix.with_suffix(".summary.json")
        samples_path = prefix.with_suffix(".samples.jsonl")
        power_path = prefix.with_suffix(".power.jsonl")
        ordinal = int(row["ordinal"])
        if ordinal < int(current["ordinal"]):
            if not all(
                path.is_file()
                for path in (marker_path, summary_path, samples_path, power_path)
            ):
                raise RuntimeError(
                    f"S1 profile order requires completed cell ordinal {ordinal}"
                )
            prior = validate_profile_summary(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
            expected = {
                "resolution": int(row["resolution"]),
                "seed": int(row["seed"]),
                "experiment_namespace": binding["experiment_namespace"],
                "canonical_experiment_root": binding["canonical_experiment_root"],
                "manifest_sha256": binding["manifest_sha256"],
                "precheck_file_sha256": binding["precheck_file_sha256"],
                "precheck_sha256": binding["precheck_sha256"],
                "profile_order_ordinal": ordinal,
            }
            if hardware_fingerprint is not None:
                expected["hardware_fingerprint"] = hardware_fingerprint
            if software_fingerprint is not None:
                expected["software_fingerprint"] = software_fingerprint
            for key, value in expected.items():
                if prior.get(key) != value:
                    raise ValueError(
                        f"completed S1 profile ordinal {ordinal} changed {key}"
                    )
            if sha256_file(power_path) != prior["power_trace_file_sha256"]:
                raise ValueError(
                    f"completed S1 profile ordinal {ordinal} power trace mismatch"
                )
        elif ordinal == int(current["ordinal"]) and (
            marker_path.exists()
            or summary_path.exists()
            or samples_path.exists()
            or power_path.exists()
        ):
            raise RuntimeError(f"S1 profile cell ordinal {ordinal} was already started")
        elif ordinal > int(current["ordinal"]) and (
            marker_path.exists()
            or summary_path.exists()
            or samples_path.exists()
            or power_path.exists()
        ):
            raise RuntimeError(
                f"S1 profile cell ordinal {ordinal} started before its turn"
            )
    return current, canonical_sha256(order)


def _strip_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = list(state)
    if keys and all(str(key).startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state.items()}
    return dict(state)


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _measure_wall_ms(fn, *, synchronize) -> tuple[Any, float]:
    synchronize()
    started = time.perf_counter()
    result = fn()
    synchronize()
    return result, (time.perf_counter() - started) * 1000.0


class CudaModuleEvents:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.handles = []
        self.starts: dict[str, list[Any]] = {}
        self.pairs: dict[str, list[tuple[Any, Any]]] = {}

    def register(self, name: str, module: Any) -> None:
        if module is None:
            return

        def before(_module, _inputs):
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts.setdefault(name, []).append(event)

        def after(_module, _inputs, _output):
            if not self.starts.get(name):
                raise RuntimeError(f"S1 profiler hook {name} has no start event")
            start = self.starts[name].pop()
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.pairs.setdefault(name, []).append((start, end))

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def reset(self) -> None:
        self.starts.clear()
        self.pairs.clear()

    def elapsed(self, name: str) -> float:
        return float(
            sum(start.elapsed_time(end) for start, end in self.pairs.get(name, ()))
        )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.reset()


class CudaMethodEvent:
    def __init__(self, torch_module: Any, target: Any, method_name: str) -> None:
        self.torch = torch_module
        self.target = target
        self.method_name = method_name
        self.original = getattr(target, method_name)
        self.pairs: list[tuple[Any, Any]] = []

        def wrapped(*args, **kwargs):
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            result = self.original(*args, **kwargs)
            end.record()
            self.pairs.append((start, end))
            return result

        setattr(target, method_name, wrapped)

    def reset(self) -> None:
        self.pairs.clear()

    def elapsed(self) -> float:
        return float(sum(start.elapsed_time(end) for start, end in self.pairs))

    def close(self) -> None:
        setattr(self.target, self.method_name, self.original)
        self.reset()


class PowerSampler:
    def __init__(self, *, gpu_id: str, interval_ms: int) -> None:
        self.gpu_id = str(gpu_id)
        self.interval_s = max(0.005, int(interval_ms) / 1000.0)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._process: subprocess.Popen[str] | None = None
        self._error: str | None = None

    def _loop(self) -> None:
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    self.gpu_id,
                    f"--loop-ms={int(round(self.interval_s * 1000.0))}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if self._stop.is_set():
                    break
                match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", line)
                if match:
                    self.samples.append((time.perf_counter(), float(match.group(0))))
                    self._ready.set()
            if not self._stop.is_set():
                stderr = ""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read().strip()
                self._error = (
                    f"persistent nvidia-smi power sampler exited early: {stderr}"
                )
        except Exception as exc:  # pragma: no cover - exercised on formal CUDA hosts
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("nvidia-smi power sampler did not produce a sample")
        if self._error or not self.samples:
            error = (
                self._error or "nvidia-smi power sampler produced no numeric samples"
            )
            self.stop()
            raise RuntimeError(error)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5.0)
        if self._error:
            raise RuntimeError(self._error)


def _interpolate_power(samples: list[tuple[float, float]], timestamp: float) -> float:
    if timestamp <= samples[0][0]:
        return samples[0][1]
    if timestamp >= samples[-1][0]:
        return samples[-1][1]
    for left, right in zip(samples[:-1], samples[1:]):
        if left[0] <= timestamp <= right[0]:
            width = right[0] - left[0]
            if width <= 0.0:
                return right[1]
            weight = (timestamp - left[0]) / width
            return left[1] * (1.0 - weight) + right[1] * weight
    return samples[-1][1]


def integrate_energy(
    samples: list[tuple[float, float]], *, start: float, end: float
) -> float | None:
    checked = sorted(samples)
    if (
        len(checked) < 2
        or checked[0][0] > start
        or checked[-1][0] < end
        or end <= start
    ):
        return None
    clipped = [
        (start, _interpolate_power(checked, start)),
        *(
            (timestamp, power)
            for timestamp, power in checked
            if start < timestamp < end
        ),
        (end, _interpolate_power(checked, end)),
    ]
    energy = 0.0
    for left, right in zip(clipped[:-1], clipped[1:]):
        energy += 0.5 * (left[1] + right[1]) * (right[0] - left[0])
    return float(energy)


def _software_identity(torch_module: Any) -> dict[str, Any]:
    packages = {}
    for name in ("mmengine", "mmcv", "mmaction2", "numpy", "decord"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    try:
        ffmpeg = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=False
        )
        ffmpeg_version = (
            ffmpeg.stdout.splitlines()[0].strip()
            if ffmpeg.returncode == 0 and ffmpeg.stdout
            else "unavailable"
        )
    except FileNotFoundError:
        ffmpeg_version = "unavailable"
    try:
        nccl_version = torch_module.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        nccl_version = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "cuda_runtime": torch_module.version.cuda,
        "cudnn": torch_module.backends.cudnn.version(),
        "nccl": (
            None
            if nccl_version is None
            else list(nccl_version)
            if isinstance(nccl_version, (tuple, list))
            else str(nccl_version)
        ),
        "packages": packages,
        "ffmpeg": ffmpeg_version,
    }


def _hardware_identity(
    torch_module: Any, device: Any, *, physical_gpu_id: str
) -> dict[str, Any]:
    properties = torch_module.cuda.get_device_properties(device)
    query_fields = (
        "uuid",
        "pci.bus_id",
        "driver_version",
        "persistence_mode",
        "compute_mode",
        "power.limit",
        "clocks.max.sm",
        "clocks.max.memory",
    )
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
            "-i",
            str(physical_gpu_id),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values = (
        [value.strip() for value in query.stdout.strip().split(",")]
        if query.returncode == 0
        else []
    )
    if len(values) != len(query_fields) or any(not value for value in values):
        raise RuntimeError(
            "formal S1 profiler could not freeze GPU hardware state: "
            f"{query.stderr.strip()}"
        )
    cpu_model = "unavailable"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    system_memory_bytes = None
    if hasattr(os, "sysconf"):
        try:
            system_memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_PHYS_PAGES")
            )
        except (OSError, ValueError):
            system_memory_bytes = None
    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "system_memory_bytes": system_memory_bytes,
        "gpu_name": properties.name,
        "gpu_total_memory": int(properties.total_memory),
        "gpu_compute_capability": [int(properties.major), int(properties.minor)],
        "gpu_multi_processor_count": int(properties.multi_processor_count),
        "physical_gpu_id": str(physical_gpu_id),
        "nvidia_smi": dict(zip(query_fields, values)),
    }


def _dataset_video_ids(dataset: Any) -> set[str]:
    return {str(row[0]) for row in dataset.data_list}


def _sample_identity(cpu_batch: Mapping[str, Any], ordinal: int) -> tuple[str, str]:
    metas = cpu_batch.get("metas")
    if not isinstance(metas, list) or len(metas) != 1:
        raise ValueError("formal S1 profile requires batch-one metadata")
    meta = metas[0]
    video_id = str(meta["video_name"])
    start = meta.get("window_start_frame", meta.get("window_start"))
    if start is None:
        raise ValueError(
            f"formal S1 profile window {ordinal} has no physical start-frame identity"
        )
    window_id = f"{video_id}:{int(start)}"
    return video_id, window_id


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from mmengine.config import Config
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores.test_engine import gather_ddp_results
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("formal S1 full-stack profiling requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if (local_rank, rank, world_size) != (0, 0, 1):
        raise RuntimeError(
            "formal S1 profiler must run under torchrun with rank 0/world_size 1"
        )
    if dist.is_initialized():
        raise RuntimeError("formal S1 profiler requires a fresh process group")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if int(args.warmup_samples) != 50 or int(args.power_interval_ms) != 20:
        raise ValueError(
            "formal S1 profile freezes 50 warmup windows and a 20 ms power interval"
        )
    if int(args.batch_size) != 1 or int(args.loader_workers) != 0:
        raise ValueError("S1 profiler requires batch_size=1 and loader_workers=0")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    manifest_path = Path(args.manifest)
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=args.annotation,
    )
    matrix = validate_config_matrix()
    cfg = Config.fromfile(str(args.config))
    require_gpu1_allocation()
    binding = validate_bound_s1_training_config(cfg, seed=int(args.seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError(
            "formal S1 profile requires the bound full precheck certificate"
        )
    require_clean_git_checkout(expected_commit=binding["code_commit"])
    resolution = int(cfg.spatial_zoom_s1_contract.runtime_resolution)
    canonical_prefix = (
        Path(binding["work_dir"])
        / "profile"
        / f"dense{resolution}_seed{int(args.seed)}"
    ).resolve()
    if Path(args.output_prefix).resolve() != canonical_prefix:
        raise ValueError(f"formal S1 profile output prefix must be {canonical_prefix}")
    device = torch.device(args.device)
    if str(device) != "cuda:0":
        raise ValueError(
            "formal S1 profiler requires cuda:0 inside CUDA_VISIBLE_DEVICES=1"
        )
    if str(args.power_gpu_id) != "1":
        raise ValueError("formal S1 power sampling must target physical GPU1")
    torch.cuda.set_device(device)
    set_seed(int(args.seed))
    hardware_identity = _hardware_identity(
        torch, device, physical_gpu_id=str(args.power_gpu_id)
    )
    software_identity = _software_identity(torch)
    hardware_fingerprint = canonical_sha256(hardware_identity)
    software_fingerprint = canonical_sha256(software_identity)

    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.test_mode = True
    if args.split != "test" or not args.test_open_certificate:
        raise ValueError(
            "formal S1 profiler only accepts the certificate-bound test split"
        )
    certificate_path = Path(args.test_open_certificate)
    certificate = validate_test_open_certificate(
        json.loads(certificate_path.read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(args.seed),
        checkpoint_path=checkpoint_path,
    )
    test_evidence_path = Path(args.test_evidence).resolve()
    canonical_test_evidence = (
        Path(binding["work_dir"]) / "gpu1_id0" / "test_evidence" / "test.evidence.json"
    ).resolve()
    if test_evidence_path != canonical_test_evidence:
        raise ValueError(
            "formal S1 profile requires the canonical sealed-test evidence"
        )
    test_evidence = validate_s1_test_evidence(
        json.loads(test_evidence_path.read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(args.seed),
    )
    if (
        Path(test_evidence["checkpoint_path"]).resolve() != checkpoint_path.resolve()
        or test_evidence["test_open_certificate_sha256"]
        != certificate["certificate_sha256"]
    ):
        raise ValueError("formal S1 profile does not match sealed-test evidence")
    profile_order_cell, profile_order_sha256 = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=resolution,
        seed=int(args.seed),
        hardware_fingerprint=hardware_fingerprint,
        software_fingerprint=software_fingerprint,
    )
    marker_path = canonical_prefix.with_suffix(".started.json").resolve()
    marker = create_profile_attempt_marker(
        marker_path,
        {
            "resolution": resolution,
            "seed": int(args.seed),
            "bound_config_sha256": canonical_sha256(cfg.to_dict()),
            "code_commit": binding["code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "protocol_fingerprint": matrix["protocol_fingerprint"],
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "test_open_certificate_sha256": certificate["certificate_sha256"],
            "test_evidence_sha256": test_evidence["evidence_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
            "hardware_fingerprint": canonical_sha256(hardware_identity),
            "software_fingerprint": canonical_sha256(software_identity),
            "profile_order_seed": S1_PROFILE_ORDER_SEED,
            "profile_order_sha256": profile_order_sha256,
            "profile_order_ordinal": int(profile_order_cell["ordinal"]),
            "canonical_output_prefix": str(canonical_prefix),
        },
    )
    dataset_cfg.subset_name = manifest["annotation_subsets"]["sealed_test"]
    dataset_cfg.block_list = None
    expected_ids = set(manifest["splits"]["test"])
    dataset = build_dataset(dataset_cfg)
    if _dataset_video_ids(dataset) != expected_ids:
        raise ValueError("profile dataset does not match the frozen S1 manifest split")
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=0,
    )
    measured_windows = len(loader)
    if measured_windows < 200:
        raise ValueError(
            "formal S1 profile requires at least 200 complete test windows"
        )
    if int(args.samples) not in (0, measured_windows):
        raise ValueError(
            "formal S1 profile measures the complete test loader; use --samples 0"
        )

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_key = "state_dict_ema" if args.use_ema else "state_dict"
    expected_state_key = (
        "state_dict_ema" if bool(cfg.solver.get("ema", False)) else "state_dict"
    )
    if state_key != expected_state_key:
        raise ValueError(
            f"formal S1 profiler must use checkpoint state {expected_state_key}"
        )
    if state_key not in checkpoint:
        raise ValueError(f"trained S1 checkpoint does not contain {state_key}")
    model.load_state_dict(_strip_ddp_prefix(checkpoint[state_key]), strict=True)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch < 0:
        raise ValueError("trained S1 checkpoint must record its epoch")
    sidecar = validate_s1_checkpoint_sidecar(checkpoint_path)
    if checkpoint.get("experiment_metadata") != sidecar["experiment_metadata"]:
        raise ValueError("S1 checkpoint payload metadata does not match sidecar")
    checkpoint_metadata = sidecar["experiment_metadata"]
    validate_s1_checkpoint_metadata_for_binding(
        checkpoint_metadata,
        binding=binding,
        epoch=checkpoint_epoch,
        cfg=cfg,
    )
    model = model.to(device).eval()
    ddp_model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    cfg.post_processing.sliding_window = True
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)

    events = CudaModuleEvents(torch)
    events.register("backbone_wrapper_ms", getattr(model, "backbone", None))
    wrapped = getattr(getattr(model, "backbone", None), "model", None)
    events.register("heavy_backbone_ms", getattr(wrapped, "backbone", None))
    events.register("projection_ms", getattr(model, "projection", None))
    events.register("neck_ms", getattr(model, "neck", None))
    head_event = CudaMethodEvent(torch, model.rpn_head, "forward_test")
    forward_test_event = CudaMethodEvent(torch, model, "forward_test")
    postprocess_event = CudaMethodEvent(torch, model, "post_processing")

    use_amp = bool(args.amp)

    def forward_once(batch):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
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

    for _ in range(int(args.warmup_samples)):
        cpu_batch = next_batch()
        gpu_batch = _move_to_device(cpu_batch, device)
        forward_once(gpu_batch)
    synchronize()
    iterator = iter(loader)

    power_sampler = None
    if not args.sample_power or not args.amp:
        raise ValueError("formal S1 profiler requires --sample-power and --amp")
    if args.sample_power:
        power_sampler = PowerSampler(
            gpu_id=args.power_gpu_id, interval_ms=args.power_interval_ms
        )
        power_sampler.start()
        time.sleep(power_sampler.interval_s * 1.5)

    samples = []
    energy_windows = []
    video_rows: dict[str, list[dict[str, Any]]] = {}
    final_result_energy_window: tuple[float, float] | None = None
    try:
        for ordinal in range(measured_windows):
            synchronize()
            continuous_started = time.perf_counter()
            start_energy = continuous_started
            cpu_batch, input_ms = _measure_wall_ms(next_batch, synchronize=synchronize)
            video_id, window_id = _sample_identity(cpu_batch, ordinal)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_batch, h2d_ms = _measure_wall_ms(
                lambda: _move_to_device(cpu_batch, device), synchronize=synchronize
            )
            events.reset()
            head_event.reset()
            forward_test_event.reset()
            postprocess_event.reset()
            post_result, _detection_wall_ms = _measure_wall_ms(
                lambda: forward_once(gpu_batch), synchronize=synchronize
            )
            if not isinstance(post_result, Mapping):
                raise ValueError("formal S1 detector did not return a result mapping")
            for result_video, rows in post_result.items():
                video_rows.setdefault(str(result_video), []).extend(rows)
            synchronize()
            end_energy = time.perf_counter()
            continuous_ms = (end_energy - continuous_started) * 1000.0
            sample = {
                "input_pipeline_serial_ms": input_ms,
                "h2d_ms": h2d_ms,
                "model_forward_ms": forward_test_event.elapsed(),
                "postprocess_ms": postprocess_event.elapsed(),
                "backbone_wrapper_ms": events.elapsed("backbone_wrapper_ms"),
                "heavy_backbone_ms": events.elapsed("heavy_backbone_ms"),
                "projection_ms": events.elapsed("projection_ms"),
                "neck_ms": events.elapsed("neck_ms"),
                "head_ms": head_event.elapsed(),
                "decode_to_window_output_wall_ms": continuous_ms,
                "final_video_nms_ms": 0.0,
                "end_to_end_serial_ms": continuous_ms,
                "video_id": video_id,
                "window_id": window_id,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                "gpu_energy_j": None,
            }
            samples.append(sample)
            energy_windows.append((start_energy, end_energy))
            del cpu_batch, gpu_batch, post_result
        synchronize()
        final_started = time.perf_counter()
        finalized_results = gather_ddp_results(
            world_size, video_rows, cfg.post_processing
        )
        synchronize()
        final_ended = time.perf_counter()
        final_result_energy_window = (final_started, final_ended)
        if not isinstance(finalized_results, Mapping) or not set(
            finalized_results
        ).issubset(expected_ids):
            raise ValueError(
                "formal S1 official result finalizer returned invalid video identities"
            )
        amortized_ms = (final_ended - final_started) * 1000.0 / len(samples)
        for sample in samples:
            sample["final_video_nms_ms"] = amortized_ms
            sample["end_to_end_serial_ms"] += amortized_ms
    finally:
        if power_sampler is not None:
            time.sleep(power_sampler.interval_s * 1.5)
            power_sampler.stop()
        events.close()
        head_event.close()
        forward_test_event.close()
        postprocess_event.close()

    if power_sampler is not None:
        for sample, (start, end) in zip(samples, energy_windows):
            sample["gpu_energy_j"] = integrate_energy(
                power_sampler.samples, start=start, end=end
            )
        if final_result_energy_window is None:
            raise RuntimeError(
                "formal S1 profiler did not execute the official result finalizer"
            )
        final_energy = integrate_energy(
            power_sampler.samples,
            start=final_result_energy_window[0],
            end=final_result_energy_window[1],
        )
        if final_energy is not None:
            for sample in samples:
                base = sample["gpu_energy_j"]
                sample["gpu_energy_j"] = (
                    None if base is None else base + final_energy / len(samples)
                )
        power_origin = power_sampler.samples[0][0]
        power_trace = [
            {"timestamp_ms": (timestamp - power_origin) * 1000.0, "power_w": power}
            for timestamp, power in power_sampler.samples
        ]
    else:
        power_trace = []
    sample_manifest = [sample["window_id"] for sample in samples]
    metadata = {
        "method": f"dense{resolution}",
        "resolution": resolution,
        "protocol": S1_PROFILE_PROTOCOL,
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "manifest_sha256": manifest["manifest_sha256"],
        "hardware_identity": hardware_identity,
        "hardware_fingerprint": hardware_fingerprint,
        "software_identity": software_identity,
        "software_fingerprint": software_fingerprint,
        "config_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "checkpoint_epoch": checkpoint_epoch,
        "trained_checkpoint": True,
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": int(args.warmup_samples),
        "amp": use_amp,
        "power_sampling_enabled": bool(args.sample_power),
        "split": args.split,
        "seed": int(args.seed),
        "video_count": len(expected_ids),
        "power_gpu_id": str(args.power_gpu_id) if args.sample_power else None,
        "power_interval_ms": int(args.power_interval_ms) if args.sample_power else None,
        "checkpoint_state_key": state_key,
        "formal_profile": True,
        "sample_manifest_sha256": canonical_sha256(sample_manifest),
        "test_open_certificate_sha256": certificate["certificate_sha256"],
        "test_evidence_sha256": test_evidence["evidence_sha256"],
        "test_open_marker_sha256": test_evidence["test_open_marker_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "world_size": world_size,
        "execution_wrapper": "torchrun_ddp_world1",
        "result_finalizer": "opentad.cores.test_engine.gather_ddp_results",
        "profile_attempt_marker_path": str(marker_path),
        "profile_attempt_marker_file_sha256": sha256_file(marker_path),
        "profile_attempt_marker_sha256": marker["marker_sha256"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_cell["ordinal"]),
    }
    return build_profile_summary(samples, metadata=metadata, power_trace=power_trace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile trained S1 dense-resolution models end to end"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--split", choices=("test",), required=True)
    parser.add_argument("--test-open-certificate", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--warmup-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--sample-power", action="store_true")
    parser.add_argument("--power-gpu-id", default="1")
    parser.add_argument("--power-interval-ms", type=int, default=20)
    parser.add_argument("--compare-baseline", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary_path = args.output_prefix.with_suffix(".summary.json")
        samples_path = args.output_prefix.with_suffix(".samples.jsonl")
        power_path = args.output_prefix.with_suffix(".power.jsonl")
        comparison_path = args.output_prefix.with_suffix(".comparison.json")
        output_paths = [summary_path, samples_path, power_path]
        if args.compare_baseline:
            output_paths.append(comparison_path)
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite formal S1 profile artifacts: {existing}"
            )
        report = profile(args)
        write_profile_summary(report, summary_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("x", encoding="utf-8") as handle:
            handle.write(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in report["raw_samples"]
                )
            )
        with power_path.open("x", encoding="utf-8") as handle:
            handle.write(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in report["raw_power_samples"]
                )
            )
        if sha256_file(power_path) != report["power_trace_file_sha256"]:
            raise RuntimeError(
                "S1 written power trace does not match the profile summary"
            )
        outputs = {
            "summary": str(summary_path),
            "samples": str(samples_path),
            "power": str(power_path),
        }
        if args.compare_baseline:
            baseline = json.loads(args.compare_baseline.read_text(encoding="utf-8"))
            comparison = compare_resolution_profiles(baseline, report)
            with comparison_path.open("x", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2, sort_keys=True)
                handle.write("\n")
            outputs["comparison"] = str(comparison_path)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    print(json.dumps({"status": "PASS", "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
