"""GPU power samplers and a no-test-data cadence diagnostic for Spatial Zoom S1."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
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
from statistics import median
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import atomic_publish_json, canonical_sha256


S1_POWER_DIAGNOSTIC_SCHEMA = "spatial_zoom_s1_power_sampler_diagnostic_v1"


def summarize_power_cadence(
    samples: Sequence[tuple[float, float]], *, target_interval_ms: int
) -> dict[str, Any]:
    """Summarize observed arrival cadence without changing the formal threshold."""

    checked = [(float(timestamp), float(power)) for timestamp, power in samples]
    finite = all(
        math.isfinite(timestamp) and math.isfinite(power) and power >= 0.0
        for timestamp, power in checked
    )
    strictly_increasing = all(
        right[0] > left[0] for left, right in zip(checked[:-1], checked[1:])
    )
    gaps_ms = [
        (right[0] - left[0]) * 1000.0
        for left, right in zip(checked[:-1], checked[1:])
    ]
    sorted_gaps = sorted(gaps_ms)

    def percentile(probability: float) -> float | None:
        if not sorted_gaps:
            return None
        rank = probability * (len(sorted_gaps) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(sorted_gaps) - 1)
        weight = rank - lower
        return float(
            sorted_gaps[lower] * (1.0 - weight) + sorted_gaps[upper] * weight
        )

    max_gap_limit_ms = max(100.0, float(target_interval_ms) * 5.0)
    max_gap_ms = max(gaps_ms) if gaps_ms else None
    return {
        "sample_count": len(checked),
        "duration_ms": (
            (checked[-1][0] - checked[0][0]) * 1000.0 if len(checked) >= 2 else 0.0
        ),
        "finite_nonnegative": finite,
        "strictly_increasing": strictly_increasing,
        "min_gap_ms": min(gaps_ms) if gaps_ms else None,
        "median_gap_ms": median(gaps_ms) if gaps_ms else None,
        "p95_gap_ms": percentile(0.95),
        "max_gap_ms": max_gap_ms,
        "max_gap_limit_ms": max_gap_limit_ms,
        "formal_cadence_pass": bool(
            len(checked) >= 2
            and finite
            and strictly_increasing
            and max_gap_ms is not None
            and max_gap_ms <= max_gap_limit_ms
        ),
    }


class NvidiaSmiPowerSampler:
    """The original persistent ``nvidia-smi --loop-ms`` sampler."""

    backend = "nvidia-smi-persistent-loop-ms"

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
        except Exception as exc:  # pragma: no cover - requires a formal GPU host
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("nvidia-smi power sampler did not produce a sample")
        if self._error or not self.samples:
            error = self._error or "nvidia-smi power sampler produced no numeric samples"
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


class _Nvml:
    NVML_SUCCESS = 0
    UUID_BUFFER_SIZE = 96

    def __init__(self) -> None:
        candidates = [
            ctypes.util.find_library("nvidia-ml"),
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
        ]
        last_error: Exception | None = None
        self.lib = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                last_error = exc
        if self.lib is None:
            raise RuntimeError(f"could not load NVML: {last_error}")

        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetUUID.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        self.lib.nvmlDeviceGetUUID.restype = ctypes.c_int
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self.lib.nvmlErrorString.argtypes = [ctypes.c_int]
        self.lib.nvmlErrorString.restype = ctypes.c_char_p

    def _check(self, result: int, operation: str) -> None:
        if int(result) == self.NVML_SUCCESS:
            return
        message = self.lib.nvmlErrorString(int(result))
        detail = message.decode("utf-8", errors="replace") if message else "unknown"
        raise RuntimeError(f"{operation} failed: NVML {result} ({detail})")

    def initialize(self) -> None:
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def shutdown(self) -> None:
        self._check(self.lib.nvmlShutdown(), "nvmlShutdown")

    def handle_by_index(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(index), ctypes.byref(handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        return handle

    def uuid(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(self.UUID_BUFFER_SIZE)
        self._check(
            self.lib.nvmlDeviceGetUUID(
                handle, buffer, ctypes.c_uint(self.UUID_BUFFER_SIZE)
            ),
            "nvmlDeviceGetUUID",
        )
        return buffer.value.decode("utf-8", errors="strict")

    def power_w(self, handle: ctypes.c_void_p) -> float:
        milliwatts = ctypes.c_uint()
        self._check(
            self.lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts)),
            "nvmlDeviceGetPowerUsage",
        )
        return float(milliwatts.value) / 1000.0


class NvmlPowerSampler:
    """Persistent in-process NVML polling with observed monotonic timestamps."""

    backend = "nvml-persistent-poll-v1"

    def __init__(
        self, *, local_gpu_index: int, expected_uuid: str, interval_ms: int
    ) -> None:
        self.local_gpu_index = int(local_gpu_index)
        self.expected_uuid = str(expected_uuid)
        self.interval_s = max(0.005, int(interval_ms) / 1000.0)
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._error: str | None = None

    def _loop(self) -> None:
        nvml: _Nvml | None = None
        try:
            nvml = _Nvml()
            nvml.initialize()
            handle = nvml.handle_by_index(self.local_gpu_index)
            actual_uuid = nvml.uuid(handle)
            if actual_uuid != self.expected_uuid:
                raise RuntimeError(
                    "NVML local device UUID differs from the Slurm-allocated GPU: "
                    f"expected {self.expected_uuid}, got {actual_uuid}"
                )
            deadline = time.perf_counter()
            while not self._stop.is_set():
                power = nvml.power_w(handle)
                observed = time.perf_counter()
                self.samples.append((observed, power))
                self._ready.set()
                deadline += self.interval_s
                if deadline <= observed:
                    missed = math.floor((observed - deadline) / self.interval_s) + 1
                    deadline += missed * self.interval_s
                self._stop.wait(max(0.0, deadline - time.perf_counter()))
        except Exception as exc:  # pragma: no cover - requires a formal GPU host
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            if nvml is not None:
                try:
                    nvml.shutdown()
                except Exception as exc:  # pragma: no cover - formal GPU host only
                    if self._error is None:
                        self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self.stop()
            raise RuntimeError("NVML power sampler did not produce a sample")
        if self._error or not self.samples:
            error = self._error or "NVML power sampler produced no numeric samples"
            self.stop()
            raise RuntimeError(error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("NVML power sampler thread did not stop")
        if self._error:
            raise RuntimeError(self._error)


def _query_uuid(gpu_id: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_id),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    uuid = result.stdout.strip()
    if result.returncode != 0 or not uuid:
        raise RuntimeError(f"could not query allocated GPU UUID: {result.stderr.strip()}")
    return uuid


def _run_cuda_load(duration_s: float) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("power-sampler diagnostic requires a Slurm CUDA allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("power-sampler diagnostic requires exactly one visible GPU")
    device = torch.device("cuda:0")
    left = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    right = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < duration_s:
        left = torch.matmul(left, right)
        left = torch.tanh(left)
        iterations += 1
    torch.cuda.synchronize(device)
    return {"iterations": iterations, "elapsed_s": time.perf_counter() - started}


def _diagnose_backend(
    sampler: NvidiaSmiPowerSampler | NvmlPowerSampler,
    *,
    duration_s: float,
    interval_ms: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        sampler.start()
        time.sleep(max(0.1, sampler.interval_s * 5.0))
        load = _run_cuda_load(duration_s)
        time.sleep(max(0.1, sampler.interval_s * 5.0))
        sampler.stop()
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            sampler.stop()
        except Exception as stop_exc:
            error = f"{error}; stop={type(stop_exc).__name__}: {stop_exc}"
        load = None
    cadence = summarize_power_cadence(
        sampler.samples, target_interval_ms=interval_ms
    )
    origin = sampler.samples[0][0] if sampler.samples else 0.0
    return {
        "backend": sampler.backend,
        "status": "PASS" if error is None and cadence["formal_cadence_pass"] else "FAIL",
        "error": error,
        "wall_s": time.perf_counter() - started,
        "cuda_load": load,
        "cadence": cadence,
        "raw_samples": [
            {
                "timestamp_ms": (timestamp - origin) * 1000.0,
                "power_w": power,
            }
            for timestamp, power in sampler.samples
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose S1 GPU power-sampler cadence without reading test data"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-gpu-id", default="0")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("power-sampler diagnostic must run in a Slurm allocation")
    if int(args.interval_ms) != 20 or float(args.duration_seconds) < 5.0:
        raise ValueError("formal diagnostic requires 20 ms sampling for at least 5 seconds")
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite S1 power diagnostic: {args.output}"
        )
    commit = str(args.code_commit).lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("power-sampler diagnostic requires a concrete Git commit")
    uuid = _query_uuid(args.logical_gpu_id)
    backends = [
        NvidiaSmiPowerSampler(
            gpu_id=args.logical_gpu_id, interval_ms=int(args.interval_ms)
        ),
        NvmlPowerSampler(
            local_gpu_index=int(args.logical_gpu_id),
            expected_uuid=uuid,
            interval_ms=int(args.interval_ms),
        ),
    ]
    report = {
        "schema_version": S1_POWER_DIAGNOSTIC_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_claim_allowed": False,
        "reads_test_data": False,
        "code_commit": commit,
        "node": platform.node(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "logical_gpu_id": str(args.logical_gpu_id),
        "gpu_uuid": uuid,
        "target_interval_ms": int(args.interval_ms),
        "duration_seconds_per_backend": float(args.duration_seconds),
        "backends": [
            _diagnose_backend(
                sampler,
                duration_s=float(args.duration_seconds),
                interval_ms=int(args.interval_ms),
            )
            for sampler in backends
        ],
    }
    report["diagnostic_sha256"] = canonical_sha256(report)
    atomic_publish_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if any(row["status"] == "PASS" for row in report["backends"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
