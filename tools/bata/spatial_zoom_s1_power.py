"""GPU power samplers and a no-test-data cadence diagnostic for Spatial Zoom S1."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
)


S1_POWER_DIAGNOSTIC_SCHEMA = "spatial_zoom_s1_power_sampler_diagnostic_v1"
S1_POWER_SIDECAR_BACKEND = "nvml-sidecar-process-v1"
S1_POWER_SIDECAR_ATTEMPT_SCHEMA = "spatial_zoom_s1_power_sidecar_attempt_v1"
S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA = (
    "spatial_zoom_s1_power_sidecar_attempt_v2"
)
S1_POWER_SIDECAR_RESULT_SCHEMA = "spatial_zoom_s1_power_sidecar_result_v1"
S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA = (
    "spatial_zoom_s1_power_sidecar_result_v2"
)
S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE = "post_sampling_atomic_jsonl_v1"
S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX = "formal S1 sidecar cadence failed:"
S1_POWER_PARENT_FAILURE_SCHEMA = "spatial_zoom_s1_profile_parent_failure_v1"


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
        self.lib.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
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

    def handle_by_uuid(self, uuid: str) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByUUID(
                str(uuid).encode("utf-8"), ctypes.byref(handle)
            ),
            "nvmlDeviceGetHandleByUUID",
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
    """Diagnostic-only in-process NVML polling.

    Formal profiles use :class:`NvmlSidecarPowerSampler` so detector GIL and
    memory pressure cannot directly stall the sampler.
    """

    backend = "nvml-persistent-poll-v1"

    def __init__(self, *, expected_uuid: str, interval_ms: int) -> None:
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
            # NVML indices are node-physical, while cuda:0 is Slurm-local. The
            # frozen UUID is the stable identity shared by both namespaces.
            handle = nvml.handle_by_uuid(self.expected_uuid)
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


def _atomic_publish_bytes(path: str | Path, payload: bytes) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_cpu_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
        try:
            parsed = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"invalid CPU list: {value!r}") from exc
    else:
        parsed = tuple(int(cpu) for cpu in value)
    if not parsed or len(set(parsed)) != len(parsed) or any(cpu < 0 for cpu in parsed):
        raise ValueError(f"invalid CPU list: {value!r}")
    return tuple(sorted(parsed))


def _load_sidecar_trace(payload: bytes) -> list[tuple[float, float]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("S1 power sidecar trace is empty or truncated")
    samples: list[tuple[float, float]] = []
    previous_ns = -1
    for expected_sequence, raw_line in enumerate(payload.splitlines()):
        try:
            row = json.loads(raw_line.decode("utf-8", errors="strict"))
            sequence = int(row["sequence"])
            monotonic_ns = int(row["monotonic_ns"])
            power_w = float(row["power_w"])
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"S1 power sidecar trace row {expected_sequence} is invalid"
            ) from exc
        if (
            sequence != expected_sequence
            or monotonic_ns <= previous_ns
            or monotonic_ns < 0
            or not math.isfinite(power_w)
            or power_w < 0.0
        ):
            raise ValueError(
                f"S1 power sidecar trace row {expected_sequence} violates ordering"
            )
        previous_ns = monotonic_ns
        samples.append((monotonic_ns / 1_000_000_000.0, power_w))
    return samples


def _clock_identity() -> dict[str, Any]:
    info = time.get_clock_info("monotonic")
    return {
        "clock": "time.monotonic_ns",
        "implementation": info.implementation,
        "monotonic": bool(info.monotonic),
        "adjustable": bool(info.adjustable),
        "resolution_seconds": float(info.resolution),
    }


def validate_nvml_sidecar_attempt(
    report_path: str | Path,
    trace_path: str | Path,
    *,
    expected_uuid: str | None = None,
    require_pass: bool = True,
    require_process_integrity: bool = False,
) -> dict[str, Any]:
    """Validate the self-hashed report and its exact immutable raw trace."""

    report_path = Path(report_path).resolve()
    trace_path = Path(trace_path).resolve()
    if not report_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("S1 sidecar attempt report/trace pair is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = report.pop("attempt_sha256", None)
    attempt_schema = report.get("schema_version")
    buffered_attempt = attempt_schema == S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA
    if (
        not report_hash
        or canonical_sha256(report) != report_hash
        or attempt_schema
        not in {
            S1_POWER_SIDECAR_ATTEMPT_SCHEMA,
            S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA,
        }
        or report.get("backend") != S1_POWER_SIDECAR_BACKEND
    ):
        raise ValueError("S1 sidecar attempt report identity mismatch")
    report["attempt_sha256"] = report_hash
    trace_payload = trace_path.read_bytes()
    samples = _load_sidecar_trace(trace_payload)
    trace_lines = trace_payload.splitlines()
    trace_first_monotonic_ns = int(
        json.loads(trace_lines[0].decode("utf-8"))["monotonic_ns"]
    )
    trace_last_monotonic_ns = int(
        json.loads(trace_lines[-1].decode("utf-8"))["monotonic_ns"]
    )
    cadence = summarize_power_cadence(
        samples, target_interval_ms=int(report.get("interval_ms", -1))
    )
    allocated = _parse_cpu_ids(report.get("allocated_cpu_ids", ()))
    detector = _parse_cpu_ids(report.get("detector_cpu_ids", ()))
    sidecar_cpu = int(report.get("sidecar_cpu_id", -1))
    clock = report.get("clock_identity", {})
    if (
        report.get("trace_path") != str(trace_path)
        or report.get("trace_file_sha256") != _sha256_bytes(trace_payload)
        or canonical_sha256(report.get("cadence", {})) != canonical_sha256(cadence)
        or int(report.get("interval_ms", -1)) != 20
        or len(allocated) != 5
        or len(detector) != 4
        or sidecar_cpu in detector
        or set(detector) | {sidecar_cpu} != set(allocated)
        or clock.get("clock") != "time.monotonic_ns"
        or clock.get("monotonic") is not True
        or clock.get("adjustable") is not False
        or (
            buffered_attempt
            and (
                report.get("trace_publication_mode")
                != S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE
                or report.get("trace_io_inside_sampling_loop") is not False
            )
        )
    ):
        raise ValueError("S1 sidecar attempt trace, cadence, clock, or CPU mismatch")
    attempt_uuid = str(report.get("expected_uuid", ""))
    if (
        not attempt_uuid.startswith("GPU-")
        or (expected_uuid is not None and attempt_uuid != str(expected_uuid))
    ):
        raise ValueError("S1 sidecar attempt GPU UUID mismatch")
    if require_pass or require_process_integrity:
        pid_record = report.get("pid_record") or {}
        ready_record = report.get("ready_record") or {}
        result_record = report.get("result_record") or {}

        def embedded_hash_valid(record: dict[str, Any], key: str) -> bool:
            checked = dict(record)
            value = checked.pop(key, None)
            return bool(value and canonical_sha256(checked) == value)

        process_pid = int(report.get("process_pid", -1))
        process_parent_pid = int(pid_record.get("parent_pid", -1))
        expected_result_schema = (
            S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA
            if buffered_attempt
            else S1_POWER_SIDECAR_RESULT_SCHEMA
        )
        if (
            int(report.get("process_exit_code", -1)) != 0
            or process_pid <= 1
            or process_parent_pid <= 1
            or not embedded_hash_valid(pid_record, "pid_sha256")
            or not embedded_hash_valid(ready_record, "ready_sha256")
            or not embedded_hash_valid(result_record, "result_sha256")
            or pid_record.get("schema_version")
            != "spatial_zoom_s1_power_sidecar_pid_v1"
            or int(pid_record.get("pid", -1)) != process_pid
            or pid_record.get("expected_uuid") != attempt_uuid
            or int(pid_record.get("sidecar_cpu_id", -1)) != sidecar_cpu
            or tuple(pid_record.get("actual_cpu_affinity", ())) != (sidecar_cpu,)
            or tuple(sorted(map(int, pid_record.get("allocated_cpu_ids", ()))))
            != allocated
            or ready_record.get("schema_version")
            != "spatial_zoom_s1_power_sidecar_ready_v1"
            or int(ready_record.get("pid", -1)) != process_pid
            or int(ready_record.get("parent_pid", -1)) != process_parent_pid
            or ready_record.get("expected_uuid") != attempt_uuid
            or ready_record.get("actual_uuid") != attempt_uuid
            or int(ready_record.get("sidecar_cpu_id", -1)) != sidecar_cpu
            or tuple(ready_record.get("actual_cpu_affinity", ()))
            != (sidecar_cpu,)
            or tuple(sorted(map(int, ready_record.get("allocated_cpu_ids", ()))))
            != allocated
            or int(ready_record.get("interval_ms", -1)) != 20
            or result_record.get("schema_version") != expected_result_schema
            or int(result_record.get("pid", -1)) != process_pid
            or int(result_record.get("parent_pid", -1)) != process_parent_pid
            or result_record.get("status") != "PASS"
            or result_record.get("error") is not None
            or result_record.get("expected_uuid") != attempt_uuid
            or result_record.get("actual_uuid") != attempt_uuid
            or int(result_record.get("sidecar_cpu_id", -1)) != sidecar_cpu
            or tuple(result_record.get("actual_cpu_affinity", ()))
            != (sidecar_cpu,)
            or tuple(sorted(map(int, result_record.get("allocated_cpu_ids", ()))))
            != allocated
            or int(result_record.get("interval_ms", -1)) != 20
            or int(result_record.get("sample_count", -1)) != len(samples)
            or result_record.get("trace_sha256") != _sha256_bytes(trace_payload)
            or int(result_record.get("started_monotonic_ns", -1)) <= 0
            or int(result_record.get("started_monotonic_ns", -1))
            > trace_first_monotonic_ns
            or int(ready_record.get("first_sample_monotonic_ns", -1))
            != trace_first_monotonic_ns
            or trace_last_monotonic_ns
            > int(result_record.get("finished_monotonic_ns", -1))
            or (
                buffered_attempt
                and (
                    result_record.get("trace_publication_mode")
                    != S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE
                    or result_record.get("trace_io_inside_sampling_loop") is not False
                )
            )
        ):
            raise ValueError("S1 sidecar attempt process identity mismatch")
    if require_pass and (
        report.get("status") != "PASS"
        or report.get("error") is not None
        or cadence.get("formal_cadence_pass") is not True
        or float(cadence.get("max_gap_ms", math.inf)) > 100.0
    ):
        raise ValueError("S1 sidecar PASS attempt cadence or status mismatch")
    return report


def validate_nvml_sidecar_cadence_failure(
    report_path: str | Path,
    trace_path: str | Path,
    *,
    expected_uuid: str,
) -> dict[str, Any]:
    """Prove that a healthy sidecar failed for cadence, and cadence alone."""

    report = validate_nvml_sidecar_attempt(
        report_path,
        trace_path,
        expected_uuid=expected_uuid,
        require_pass=False,
        require_process_integrity=True,
    )
    cadence = report["cadence"]
    expected_error = (
        f"{S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX} "
        f"max_gap_ms={cadence['max_gap_ms']}"
    )
    if (
        report.get("status") != "FAIL"
        or report.get("error") != expected_error
        or cadence.get("formal_cadence_pass") is not False
        or float(cadence.get("max_gap_limit_ms", -1.0)) != 100.0
        or float(cadence.get("max_gap_ms", -1.0)) <= 100.0
    ):
        raise ValueError(
            "S1 sidecar attempt does not prove an isolated cadence failure"
        )
    return report


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"sidecar JSON artifact is not an object: {path}")
    return value


def _attempt_artifact_paths(
    attempt_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(attempt_prefix).resolve()
    return (
        Path(f"{prefix}.power_attempt.jsonl"),
        Path(f"{prefix}.power_attempt.json"),
    )


def _parent_failure_path(attempt_prefix: str | Path) -> Path:
    return Path(f"{Path(attempt_prefix).resolve()}.power_parent_failure.json")


def _validated_sidecar_pid_for_salvage(
    pid_record: dict[str, Any] | None,
    *,
    expected_uuid: str,
    sidecar_cpu_id: int,
    allocated_cpu_ids: Sequence[int],
    proc_root: str | Path = "/proc",
) -> tuple[int | None, str | None]:
    """Return a live, command-verified sidecar PID without trusting scratch."""

    if pid_record is None:
        return None, "NVML sidecar PID record is missing during salvage"
    checked = dict(pid_record)
    pid_hash = checked.pop("pid_sha256", None)
    if not pid_hash or canonical_sha256(checked) != pid_hash:
        return None, "NVML sidecar PID record self-hash mismatch during salvage"
    pid = int(checked.get("pid", -1))
    if (
        pid <= 1
        or checked.get("expected_uuid") != str(expected_uuid)
        or int(checked.get("sidecar_cpu_id", -1)) != int(sidecar_cpu_id)
        or tuple(checked.get("actual_cpu_affinity", ()))
        != (int(sidecar_cpu_id),)
        or tuple(sorted(map(int, checked.get("allocated_cpu_ids", ()))))
        != _parse_cpu_ids(allocated_cpu_ids)
    ):
        return None, "NVML sidecar PID identity mismatch during salvage"
    cmdline_path = Path(proc_root).resolve() / str(pid) / "cmdline"
    if not cmdline_path.is_file():
        return None, None
    try:
        command = cmdline_path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        return None, f"could not verify NVML sidecar command: {type(exc).__name__}"
    tokens = [token for token in command.split("\0") if token]
    if (
        "sidecar" not in tokens
        or str(expected_uuid) not in tokens
        or not any(Path(token).name == Path(__file__).name for token in tokens)
    ):
        return None, "live PID does not match the NVML sidecar command"
    return pid, None


def run_nvml_sidecar(
    *,
    expected_uuid: str,
    interval_ms: int,
    trace_path: str | Path,
    ready_path: str | Path,
    result_path: str | Path,
    sidecar_cpu_id: int,
    allocated_cpu_ids: Sequence[int],
    stop_after_samples: int = 0,
) -> int:
    """Run the minimal formal sampler process.

    ``stop_after_samples`` exists only for focused process tests. Formal runs
    pass zero and stop the process with SIGTERM after detector finalization.
    """

    trace_path = Path(trace_path).resolve()
    ready_path = Path(ready_path).resolve()
    result_path = Path(result_path).resolve()
    pid_path = trace_path.parent / "pid.json"
    if not (
        trace_path.parent == ready_path.parent == result_path.parent
    ):
        raise ValueError("S1 sidecar scratch artifacts must share one directory")
    allocated = _parse_cpu_ids(allocated_cpu_ids)
    sidecar_cpu_id = int(sidecar_cpu_id)
    if sidecar_cpu_id not in allocated:
        raise ValueError("sidecar CPU is outside the Slurm allocation")
    if interval_ms != 20:
        raise ValueError("formal S1 sidecar freezes a 20 ms interval")
    if any(path.exists() for path in (trace_path, ready_path, result_path, pid_path)):
        raise FileExistsError("refusing to overwrite S1 sidecar scratch artifacts")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("formal S1 sidecar requires Linux CPU affinity")
    os.sched_setaffinity(0, {sidecar_cpu_id})
    actual_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if actual_affinity != (sidecar_cpu_id,):
        raise RuntimeError("S1 sidecar could not reserve its dedicated CPU")
    pid_record = {
        "schema_version": "spatial_zoom_s1_power_sidecar_pid_v1",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "expected_uuid": expected_uuid,
        "sidecar_cpu_id": sidecar_cpu_id,
        "actual_cpu_affinity": list(actual_affinity),
        "allocated_cpu_ids": list(allocated),
        "clock_identity": _clock_identity(),
    }
    pid_record["pid_sha256"] = canonical_sha256(pid_record)
    atomic_publish_json(pid_path, pid_record)

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    nvml: _Nvml | None = None
    sample_count = 0
    started_ns = time.monotonic_ns()
    error: str | None = None
    actual_uuid: str | None = None
    try:
        nvml = _Nvml()
        nvml.initialize()
        handle = nvml.handle_by_uuid(expected_uuid)
        actual_uuid = nvml.uuid(handle)
        if actual_uuid != expected_uuid:
            raise RuntimeError(
                "NVML sidecar UUID differs from the Slurm-allocated GPU: "
                f"expected {expected_uuid}, got {actual_uuid}"
            )
        interval_ns = int(interval_ms) * 1_000_000
        deadline_ns = time.monotonic_ns()
        previous_observed_ns = -1
        trace_buffer = io.BytesIO()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            while not stop_requested:
                power = nvml.power_w(handle)
                observed_ns = time.monotonic_ns()
                while observed_ns <= previous_observed_ns:
                    observed_ns = time.monotonic_ns()
                previous_observed_ns = observed_ns
                trace_buffer.write(
                    json.dumps(
                        {
                            "sequence": sample_count,
                            "monotonic_ns": observed_ns,
                            "power_w": power,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                sample_count += 1
                if sample_count == 1:
                    ready = {
                        "schema_version": "spatial_zoom_s1_power_sidecar_ready_v1",
                        "pid": os.getpid(),
                        "parent_pid": os.getppid(),
                        "expected_uuid": expected_uuid,
                        "actual_uuid": actual_uuid,
                        "sidecar_cpu_id": sidecar_cpu_id,
                        "actual_cpu_affinity": list(actual_affinity),
                        "allocated_cpu_ids": list(allocated),
                        "interval_ms": interval_ms,
                        "clock_identity": _clock_identity(),
                        "first_sample_monotonic_ns": observed_ns,
                    }
                    ready["ready_sha256"] = canonical_sha256(ready)
                    atomic_publish_json(ready_path, ready)
                if stop_after_samples > 0 and sample_count >= stop_after_samples:
                    break
                deadline_ns += interval_ns
                if deadline_ns <= observed_ns:
                    missed = (observed_ns - deadline_ns) // interval_ns + 1
                    deadline_ns += missed * interval_ns
                time.sleep(max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9))
        finally:
            if gc_was_enabled:
                gc.enable()
        _atomic_publish_bytes(trace_path, trace_buffer.getvalue())
    except Exception as exc:  # pragma: no cover - exercised on a formal GPU host
        error = f"{type(exc).__name__}: {exc}"
        if "trace_buffer" in locals() and trace_buffer.tell() > 0:
            try:
                _atomic_publish_bytes(trace_path, trace_buffer.getvalue())
            except FileExistsError:
                pass
            except Exception as publish_exc:
                error = (
                    f"{error}; trace_publish={type(publish_exc).__name__}: "
                    f"{publish_exc}"
                )
    finally:
        if nvml is not None:
            try:
                nvml.shutdown()
            except Exception as exc:  # pragma: no cover - formal GPU host only
                if error is None:
                    error = f"{type(exc).__name__}: {exc}"

    result = {
        "schema_version": S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA,
        "status": "PASS" if error is None else "FAIL",
        "error": error,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "expected_uuid": expected_uuid,
        "actual_uuid": actual_uuid,
        "sidecar_cpu_id": sidecar_cpu_id,
        "actual_cpu_affinity": list(actual_affinity),
        "allocated_cpu_ids": list(allocated),
        "interval_ms": interval_ms,
        "clock_identity": _clock_identity(),
        "sample_count": sample_count,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
        "trace_publication_mode": S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_publish_json(result_path, result)
    return 0 if error is None else 1


class NvmlSidecarPowerSampler:
    """Formal UUID-bound NVML sampler isolated from the detector process."""

    backend = S1_POWER_SIDECAR_BACKEND

    def __init__(
        self,
        *,
        expected_uuid: str,
        interval_ms: int,
        scratch_dir: str | Path,
        attempt_prefix: str | Path,
        sidecar_cpu_id: int,
        detector_cpu_ids: str | Sequence[int],
        allocated_cpu_ids: str | Sequence[int],
        source_path: str | Path | None = None,
        startup_timeout_s: float = 20.0,
        stop_timeout_s: float = 20.0,
    ) -> None:
        self.expected_uuid = str(expected_uuid)
        self.interval_s = max(0.005, int(interval_ms) / 1000.0)
        self.interval_ms = int(interval_ms)
        self.scratch_dir = Path(scratch_dir).resolve()
        self.attempt_prefix = Path(attempt_prefix).resolve()
        self.sidecar_cpu_id = int(sidecar_cpu_id)
        self.detector_cpu_ids = _parse_cpu_ids(detector_cpu_ids)
        self.allocated_cpu_ids = _parse_cpu_ids(allocated_cpu_ids)
        self.source_path = (
            Path(source_path).resolve()
            if source_path is not None
            else Path(__file__).resolve()
        )
        self.startup_timeout_s = float(startup_timeout_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self.samples: list[tuple[float, float]] = []
        self.attempt_report: dict[str, Any] | None = None
        self._process: subprocess.Popen | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self._finalized = False
        self._trace_path = self.scratch_dir / "power.jsonl"
        self._pid_path = self.scratch_dir / "pid.json"
        self._ready_path = self.scratch_dir / "ready.json"
        self._result_path = self.scratch_dir / "result.json"
        self._stdout_path = self.scratch_dir / "stdout.log"
        self._stderr_path = self.scratch_dir / "stderr.log"
        self.attempt_trace_path, self.attempt_report_path = _attempt_artifact_paths(
            self.attempt_prefix
        )
        self._validate_contract()

    def _validate_contract(self) -> None:
        if not self.expected_uuid.startswith("GPU-"):
            raise ValueError("formal S1 sidecar requires an NVML GPU UUID")
        if self.interval_ms != 20:
            raise ValueError("formal S1 sidecar freezes a 20 ms interval")
        if len(self.allocated_cpu_ids) != 5 or len(self.detector_cpu_ids) != 4:
            raise ValueError("formal S1 profile requires four detector CPUs plus one")
        if self.sidecar_cpu_id in self.detector_cpu_ids or set(
            self.detector_cpu_ids
        ) | {self.sidecar_cpu_id} != set(self.allocated_cpu_ids):
            raise ValueError("formal S1 detector and sidecar CPU partitions are invalid")
        if any(
            path.exists()
            for path in (
                self.scratch_dir,
                self.attempt_trace_path,
                self.attempt_report_path,
            )
        ):
            raise FileExistsError("refusing to overwrite S1 sidecar attempt artifacts")
        if not self.source_path.is_file():
            raise FileNotFoundError(self.source_path)
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("formal S1 sidecar requires Linux CPU affinity")
        actual_detector_affinity = tuple(sorted(os.sched_getaffinity(0)))
        if actual_detector_affinity != self.detector_cpu_ids:
            raise RuntimeError(
                "detector process CPU affinity differs from the reserved four CPUs"
            )

    def _command(self) -> list[str]:
        return [
            sys.executable,
            str(self.source_path),
            "sidecar",
            "--expected-uuid",
            self.expected_uuid,
            "--interval-ms",
            str(self.interval_ms),
            "--trace",
            str(self._trace_path),
            "--ready",
            str(self._ready_path),
            "--result",
            str(self._result_path),
            "--sidecar-cpu-id",
            str(self.sidecar_cpu_id),
            "--allocated-cpus",
            ",".join(map(str, self.allocated_cpu_ids)),
        ]

    def start(self) -> None:
        self.scratch_dir.parent.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(exist_ok=False)
        self._stdout_handle = self._stdout_path.open("x", encoding="utf-8")
        self._stderr_handle = self._stderr_path.open("x", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.DEVNULL,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout_s
            while not self._ready_path.is_file():
                if self._process.poll() is not None:
                    self._finalize_attempt(
                        forced_error="NVML sidecar exited before its ready record"
                    )
                if time.monotonic() >= deadline:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
                    self._finalize_attempt(
                        forced_error="NVML sidecar did not become ready before timeout"
                    )
                time.sleep(0.02)
            ready = _read_json_if_present(self._ready_path)
            if ready is None:
                self._finalize_attempt(forced_error="NVML sidecar ready record vanished")
            ready_hash = ready.pop("ready_sha256", None)
            if not ready_hash or canonical_sha256(ready) != ready_hash:
                self._finalize_attempt(
                    forced_error="NVML sidecar ready record self-hash mismatch"
                )
            ready["ready_sha256"] = ready_hash
            if (
                ready.get("expected_uuid") != self.expected_uuid
                or ready.get("actual_uuid") != self.expected_uuid
                or tuple(ready.get("actual_cpu_affinity", ()))
                != (self.sidecar_cpu_id,)
            ):
                self._finalize_attempt(
                    forced_error="NVML sidecar ready identity mismatch"
                )
        except Exception:
            if not self._finalized:
                self._finalize_attempt(forced_error="NVML sidecar startup failed")
            raise

    def stop(self) -> None:
        if self._finalized:
            if self.attempt_report and self.attempt_report.get("status") != "PASS":
                raise RuntimeError(str(self.attempt_report.get("error")))
            return
        forced_error = None
        if self._process is None:
            forced_error = "NVML sidecar process was never started"
        elif self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
                forced_error = "NVML sidecar did not stop before timeout"
        self._finalize_attempt(forced_error=forced_error)

    def _close_logs(self) -> None:
        for handle in (self._stdout_handle, self._stderr_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    def _terminate_live_process(self) -> str | None:
        if self._process is None or self._process.poll() is not None:
            return None
        self._process.terminate()
        try:
            self._process.wait(timeout=self.stop_timeout_s)
            return None
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5.0)
            return "NVML sidecar required SIGKILL during finalization"

    def _finalize_attempt(self, *, forced_error: str | None) -> None:
        if self._finalized:
            return
        termination_error = self._terminate_live_process()
        if termination_error:
            forced_error = (
                f"{forced_error}; {termination_error}"
                if forced_error
                else termination_error
            )
        self._close_logs()
        trace_payload = self._trace_path.read_bytes() if self._trace_path.is_file() else b""
        _atomic_publish_bytes(self.attempt_trace_path, trace_payload)
        errors = [forced_error] if forced_error else []
        ready = None
        pid_record = None
        result = None
        try:
            pid_record = _read_json_if_present(self._pid_path)
            ready = _read_json_if_present(self._ready_path)
            result = _read_json_if_present(self._result_path)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if pid_record is None:
            errors.append("NVML sidecar PID record is missing")
        else:
            pid_hash = pid_record.pop("pid_sha256", None)
            if not pid_hash or canonical_sha256(pid_record) != pid_hash:
                errors.append("NVML sidecar PID record self-hash mismatch")
            pid_record["pid_sha256"] = pid_hash
            if (
                pid_record.get("expected_uuid") != self.expected_uuid
                or tuple(pid_record.get("actual_cpu_affinity", ()))
                != (self.sidecar_cpu_id,)
                or (
                    self._process is not None
                    and int(pid_record.get("pid", -1)) != self._process.pid
                )
            ):
                errors.append("NVML sidecar PID record identity mismatch")
        try:
            self.samples = _load_sidecar_trace(trace_payload)
        except Exception as exc:
            self.samples = []
            errors.append(f"{type(exc).__name__}: {exc}")
        cadence = summarize_power_cadence(
            self.samples, target_interval_ms=self.interval_ms
        )
        returncode = self._process.poll() if self._process is not None else None
        if returncode != 0:
            errors.append(f"sidecar exit code is {returncode}")
        if result is None:
            errors.append("NVML sidecar result record is missing")
        else:
            result_hash = result.pop("result_sha256", None)
            if not result_hash or canonical_sha256(result) != result_hash:
                errors.append("NVML sidecar result self-hash mismatch")
            result["result_sha256"] = result_hash
            if (
                result.get("status") != "PASS"
                or result.get("expected_uuid") != self.expected_uuid
                or result.get("actual_uuid") != self.expected_uuid
                or int(result.get("sample_count", -1)) != len(self.samples)
                or result.get("trace_sha256") != _sha256_bytes(trace_payload)
            ):
                errors.append("NVML sidecar result does not match the raw trace")
        if not cadence["formal_cadence_pass"]:
            errors.append(
                f"{S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX} "
                f"max_gap_ms={cadence['max_gap_ms']}"
            )
        stdout_payload = (
            self._stdout_path.read_bytes() if self._stdout_path.is_file() else b""
        )
        stderr_payload = (
            self._stderr_path.read_bytes() if self._stderr_path.is_file() else b""
        )
        report = {
            "schema_version": S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA,
            "backend": self.backend,
            "status": "FAIL" if errors else "PASS",
            "error": "; ".join(filter(None, errors)) or None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "expected_uuid": self.expected_uuid,
            "interval_ms": self.interval_ms,
            "allocated_cpu_ids": list(self.allocated_cpu_ids),
            "detector_cpu_ids": list(self.detector_cpu_ids),
            "sidecar_cpu_id": self.sidecar_cpu_id,
            "process_pid": self._process.pid if self._process is not None else None,
            "process_exit_code": returncode,
            "clock_identity": _clock_identity(),
            "pid_record": pid_record,
            "ready_record": ready,
            "result_record": result,
            "cadence": cadence,
            "trace_path": str(self.attempt_trace_path),
            "trace_file_sha256": _sha256_bytes(trace_payload),
            "stdout_sha256": _sha256_bytes(stdout_payload),
            "stderr_sha256": _sha256_bytes(stderr_payload),
            "trace_publication_mode": S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
            "trace_io_inside_sampling_loop": False,
        }
        report["attempt_sha256"] = canonical_sha256(report)
        atomic_publish_json(self.attempt_report_path, report)
        self.attempt_report = report
        self._finalized = True
        if errors:
            raise RuntimeError(report["error"])


def salvage_nvml_sidecar_attempt(
    *,
    scratch_dir: str | Path,
    attempt_prefix: str | Path,
    expected_uuid: str,
    interval_ms: int,
    sidecar_cpu_id: int,
    detector_cpu_ids: str | Sequence[int],
    allocated_cpu_ids: str | Sequence[int],
) -> dict[str, Any]:
    """Preserve a failed attempt when the detector process cannot finalize it."""

    scratch_dir = Path(scratch_dir).resolve()
    trace_path = scratch_dir / "power.jsonl"
    ready_path = scratch_dir / "ready.json"
    pid_path = scratch_dir / "pid.json"
    result_path = scratch_dir / "result.json"
    stdout_path = scratch_dir / "stdout.log"
    stderr_path = scratch_dir / "stderr.log"
    attempt_trace_path, attempt_report_path = _attempt_artifact_paths(
        attempt_prefix
    )
    parent_failure_path = _parent_failure_path(attempt_prefix)
    if parent_failure_path.is_file():
        existing_failure = json.loads(
            parent_failure_path.read_text(encoding="utf-8")
        )
        failure_hash = existing_failure.pop("parent_failure_sha256", None)
        if (
            not failure_hash
            or canonical_sha256(existing_failure) != failure_hash
            or existing_failure.get("schema_version")
            != S1_POWER_PARENT_FAILURE_SCHEMA
            or existing_failure.get("status") != "FAIL"
        ):
            raise ValueError("existing S1 parent-failure report is corrupt")
        existing_failure["parent_failure_sha256"] = failure_hash
        return existing_failure

    existing_attempt = None
    if attempt_report_path.is_file():
        existing_attempt = json.loads(
            attempt_report_path.read_text(encoding="utf-8")
        )
        existing_hash = existing_attempt.pop("attempt_sha256", None)
        if (
            not existing_hash
            or canonical_sha256(existing_attempt) != existing_hash
            or existing_attempt.get("schema_version")
            not in {
                S1_POWER_SIDECAR_ATTEMPT_SCHEMA,
                S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA,
            }
        ):
            raise ValueError("existing S1 sidecar attempt report is corrupt")
        existing_attempt["attempt_sha256"] = existing_hash
        if (
            attempt_trace_path.is_file()
            and existing_attempt.get("trace_file_sha256")
            != sha256_file(attempt_trace_path)
        ):
            raise ValueError("existing S1 sidecar attempt trace is corrupt")

    ready = _read_json_if_present(ready_path)
    pid_record = _read_json_if_present(pid_path)
    if pid_record is None and not trace_path.exists():
        pid_deadline = time.monotonic() + 2.0
        while time.monotonic() < pid_deadline and pid_record is None:
            time.sleep(0.05)
            pid_record = _read_json_if_present(pid_path)
    errors = [
        "detector/profile process exited before formal sidecar finalization"
    ]
    pid, pid_error = _validated_sidecar_pid_for_salvage(
        pid_record,
        expected_uuid=str(expected_uuid),
        sidecar_cpu_id=int(sidecar_cpu_id),
        allocated_cpu_ids=_parse_cpu_ids(allocated_cpu_ids),
    )
    if pid_error:
        errors.append(pid_error)
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not result_path.is_file():
            time.sleep(0.05)
        if not result_path.is_file():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    trace_available = attempt_trace_path.is_file()
    if trace_available:
        trace_payload = attempt_trace_path.read_bytes()
    else:
        trace_payload = trace_path.read_bytes() if trace_path.is_file() else b""
    if existing_attempt is not None and not trace_available:
        if existing_attempt.get("trace_file_sha256") == _sha256_bytes(trace_payload):
            _atomic_publish_bytes(attempt_trace_path, trace_payload)
            trace_available = True
        else:
            errors.append(
                "shared sidecar report exists without its hash-matching raw trace"
            )
    elif existing_attempt is None and not trace_available:
        _atomic_publish_bytes(attempt_trace_path, trace_payload)
        trace_available = True
    try:
        samples = _load_sidecar_trace(trace_payload)
    except Exception as exc:
        samples = []
        errors.append(f"{type(exc).__name__}: {exc}")
    cadence = summarize_power_cadence(samples, target_interval_ms=int(interval_ms))
    result = None
    try:
        result = _read_json_if_present(result_path)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    stdout_payload = stdout_path.read_bytes() if stdout_path.is_file() else b""
    stderr_payload = stderr_path.read_bytes() if stderr_path.is_file() else b""
    if existing_attempt is None:
        existing_attempt = {
            "schema_version": S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA,
            "backend": S1_POWER_SIDECAR_BACKEND,
            "status": "FAIL",
            "error": "; ".join(errors),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "salvaged_after_parent_failure": True,
            "expected_uuid": str(expected_uuid),
            "interval_ms": int(interval_ms),
            "allocated_cpu_ids": list(_parse_cpu_ids(allocated_cpu_ids)),
            "detector_cpu_ids": list(_parse_cpu_ids(detector_cpu_ids)),
            "sidecar_cpu_id": int(sidecar_cpu_id),
            "clock_identity": _clock_identity(),
            "pid_record": pid_record,
            "ready_record": ready,
            "result_record": result,
            "cadence": cadence,
            "trace_path": str(attempt_trace_path),
            "trace_file_sha256": _sha256_bytes(trace_payload),
            "stdout_sha256": _sha256_bytes(stdout_payload),
            "stderr_sha256": _sha256_bytes(stderr_payload),
            "trace_publication_mode": S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
            "trace_io_inside_sampling_loop": False,
        }
        existing_attempt["attempt_sha256"] = canonical_sha256(existing_attempt)
        atomic_publish_json(attempt_report_path, existing_attempt)

    attempt_trace_hash = (
        sha256_file(attempt_trace_path) if attempt_trace_path.is_file() else None
    )
    parent_failure = {
        "schema_version": S1_POWER_PARENT_FAILURE_SCHEMA,
        "status": "FAIL",
        "paper_claim_allowed": False,
        "salvaged_after_parent_failure": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "error": "detector/profile process returned a non-zero exit status",
        "expected_uuid": str(expected_uuid),
        "interval_ms": int(interval_ms),
        "allocated_cpu_ids": list(_parse_cpu_ids(allocated_cpu_ids)),
        "detector_cpu_ids": list(_parse_cpu_ids(detector_cpu_ids)),
        "sidecar_cpu_id": int(sidecar_cpu_id),
        "power_attempt_status": existing_attempt["status"],
        "power_attempt_report_path": str(attempt_report_path),
        "power_attempt_report_file_sha256": sha256_file(attempt_report_path),
        "power_attempt_sha256": existing_attempt["attempt_sha256"],
        "power_attempt_trace_path": str(attempt_trace_path),
        "power_attempt_trace_file_sha256": attempt_trace_hash,
        "power_attempt_artifacts_complete": bool(
            attempt_report_path.is_file() and attempt_trace_path.is_file()
        ),
        "salvage_errors": errors,
    }
    parent_failure["parent_failure_sha256"] = canonical_sha256(parent_failure)
    atomic_publish_json(parent_failure_path, parent_failure)
    return parent_failure


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
    parser.add_argument("--physical-gpu-id", required=True)
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--code-commit", required=True)
    return parser


def build_sidecar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated UUID-bound Spatial Zoom S1 NVML sidecar"
    )
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--interval-ms", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--sidecar-cpu-id", type=int, required=True)
    parser.add_argument("--allocated-cpus", required=True)
    return parser


def build_salvage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal a failed Spatial Zoom S1 sidecar attempt"
    )
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--attempt-prefix", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--interval-ms", type=int, required=True)
    parser.add_argument("--sidecar-cpu-id", type=int, required=True)
    parser.add_argument("--detector-cpus", required=True)
    parser.add_argument("--allocated-cpus", required=True)
    return parser


def sidecar_main(argv: list[str] | None = None) -> int:
    args = build_sidecar_parser().parse_args(argv)
    return run_nvml_sidecar(
        expected_uuid=str(args.expected_uuid),
        interval_ms=int(args.interval_ms),
        trace_path=args.trace,
        ready_path=args.ready,
        result_path=args.result,
        sidecar_cpu_id=int(args.sidecar_cpu_id),
        allocated_cpu_ids=_parse_cpu_ids(str(args.allocated_cpus)),
    )


def salvage_main(argv: list[str] | None = None) -> int:
    args = build_salvage_parser().parse_args(argv)
    report = salvage_nvml_sidecar_attempt(
        scratch_dir=args.scratch_dir,
        attempt_prefix=args.attempt_prefix,
        expected_uuid=str(args.expected_uuid),
        interval_ms=int(args.interval_ms),
        sidecar_cpu_id=int(args.sidecar_cpu_id),
        detector_cpu_ids=str(args.detector_cpus),
        allocated_cpu_ids=str(args.allocated_cpus),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


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
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    allocated = [
        value.strip()
        for value in os.environ.get("SLURM_JOB_GPUS", "").split(",")
        if value.strip()
    ]
    if len(visible) != 1 or allocated != [str(args.physical_gpu_id)]:
        raise RuntimeError(
            "power-sampler diagnostic requires one matching Slurm GPU identity"
        )
    uuid = _query_uuid(args.physical_gpu_id)
    backends = [
        NvidiaSmiPowerSampler(
            gpu_id=args.physical_gpu_id, interval_ms=int(args.interval_ms)
        ),
        NvmlPowerSampler(
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
        "physical_gpu_id": str(args.physical_gpu_id),
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


def cli(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "sidecar":
        return sidecar_main(arguments[1:])
    if arguments and arguments[0] == "salvage":
        return salvage_main(arguments[1:])
    return main(arguments)


__all__ = [
    "S1_POWER_DIAGNOSTIC_SCHEMA",
    "S1_POWER_SIDECAR_BACKEND",
    "S1_POWER_SIDECAR_ATTEMPT_SCHEMA",
    "S1_POWER_BUFFERED_SIDECAR_ATTEMPT_SCHEMA",
    "S1_POWER_SIDECAR_RESULT_SCHEMA",
    "S1_POWER_BUFFERED_SIDECAR_RESULT_SCHEMA",
    "S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE",
    "S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX",
    "S1_POWER_PARENT_FAILURE_SCHEMA",
    "NvmlPowerSampler",
    "NvmlSidecarPowerSampler",
    "NvidiaSmiPowerSampler",
    "run_nvml_sidecar",
    "salvage_nvml_sidecar_attempt",
    "summarize_power_cadence",
    "validate_nvml_sidecar_attempt",
    "validate_nvml_sidecar_cadence_failure",
]


if __name__ == "__main__":
    raise SystemExit(cli())
