#!/usr/bin/env python3
"""Prove two same-node GeoRoute torchrun parents have independent stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata.georoute_stage_runner import (
    _job_scoped_loopback,
    build_torchrun_prefix,
)


ROOT = Path(__file__).resolve().parents[2]
GEOROUTE_RENDEZVOUS_GATE_SCHEMA = "georoute_rendezvous_isolation_gate_v4"
GEOROUTE_RENDEZVOUS_FAILURE_SCHEMA = "georoute_rendezvous_isolation_failure_v1"
READINESS_TIMEOUT_SECONDS = 120.0
_DIAGNOSTIC_OUTPUT_LIMIT = 32 * 1024
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_rendezvous_gate_receipt(
    payload: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    expected_node_name: str | None = None,
) -> None:
    digest_payload = dict(payload)
    digest = digest_payload.pop("gate_sha256", None)
    if digest != canonical_sha256(digest_payload):
        raise ValueError("GeoRoute rendezvous gate self-hash mismatch")
    if payload.get("schema_version") != GEOROUTE_RENDEZVOUS_GATE_SCHEMA:
        raise ValueError("unexpected GeoRoute rendezvous gate schema")
    if payload.get("status") != "PASS_CONCURRENT_RENDEZVOUS_ISOLATION":
        raise ValueError("GeoRoute rendezvous isolation gate did not pass")
    if expected_commit is not None and payload.get("runtime_commit") != expected_commit:
        raise ValueError("GeoRoute rendezvous gate commit mismatch")
    if payload.get("same_node_concurrent") is not True:
        raise ValueError("GeoRoute rendezvous gate did not run concurrently on one node")
    if payload.get("long_probe_alive_after_short_exit") is not True:
        raise ValueError("GeoRoute rendezvous lifetime isolation was not demonstrated")
    if float(payload.get("readiness_timeout_seconds", -1.0)) != READINESS_TIMEOUT_SECONDS:
        raise ValueError("GeoRoute rendezvous readiness timeout is not the audited value")
    short_exit = float(payload.get("release_to_short_exit_seconds", -1.0))
    long_exit = float(payload.get("release_to_long_exit_seconds", -1.0))
    if short_exit <= 0 or long_exit <= short_exit:
        raise ValueError("GeoRoute rendezvous gate lacks ordered lifetime evidence")
    slurm_job_id = payload.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id:
        raise ValueError("GeoRoute rendezvous gate lacks its Slurm job identity")
    node_name = payload.get("node_name")
    if not isinstance(node_name, str) or not node_name:
        raise ValueError("GeoRoute rendezvous gate lacks its node identity")
    if expected_node_name is not None and node_name != expected_node_name:
        raise ValueError("GeoRoute rendezvous gate node differs from the current leaf")
    probes = payload.get("probes")
    if not isinstance(probes, Mapping) or set(probes) != {"short", "long"}:
        raise ValueError("GeoRoute rendezvous gate lacks both probes")
    identities: set[str] = set()
    ports: set[int] = set()
    for label in ("short", "long"):
        probe = probes[label]
        if not isinstance(probe, Mapping):
            raise ValueError(f"GeoRoute {label} rendezvous probe is malformed")
        rendezvous = probe.get("rendezvous")
        if not isinstance(rendezvous, Mapping):
            raise ValueError(f"GeoRoute {label} probe lacks a rendezvous receipt")
        endpoint_host = _job_scoped_loopback(slurm_job_id)
        expected_slot = 0 if label == "long" else 1
        if (
            rendezvous.get("phase") != "train"
            or rendezvous.get("backend") != "c10d"
            or rendezvous.get("endpoint") != f"{endpoint_host}:0"
            or rendezvous.get("endpoint_host") != endpoint_host
            or rendezvous.get("endpoint_policy")
            != "job_scoped_loopback_and_kernel_assigned_port"
            or int(rendezvous.get("rendezvous_slot", -1)) != expected_slot
            or rendezvous.get("slurm_job_id") != slurm_job_id
            or rendezvous.get("stage") != "p0"
            or rendezvous.get("variant") != f"rendezvous_probe_{label}"
            or int(rendezvous.get("seed", -1)) != 3407
            or int(rendezvous.get("nnodes", -1)) != 1
            or int(rendezvous.get("nproc_per_node", -1)) != 1
            or int(probe.get("exit_code", -1)) != 0
            or probe.get("ready_marker_seen") is not True
            or probe.get("done_marker_seen") is not True
            or (
                label == "long"
                and probe.get("peer_exit_marker_seen") is not True
            )
            or _SHA256.fullmatch(str(probe.get("output_sha256", ""))) is None
        ):
            raise ValueError(f"GeoRoute {label} rendezvous probe did not pass")
        identity = rendezvous.get("rendezvous_id")
        expected_identity = (
            f"georoute-{slurm_job_id}-p0-rendezvous_probe_{label}"
            "-s3407-train"
        )
        if identity != expected_identity:
            raise ValueError(f"GeoRoute {label} rendezvous ID is missing")
        runtime = probe.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            raise ValueError(f"GeoRoute {label} runtime rendezvous identity is missing")
        try:
            master_port = int(runtime.get("master_port", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"GeoRoute {label} runtime rendezvous port is invalid"
            ) from exc
        if (
            runtime.get("event") != "GEOROUTE_RDZV_READY"
            or runtime.get("label") != label
            or int(runtime.get("rank", -1)) != 0
            or int(runtime.get("world_size", -1)) != 1
            or runtime.get("torchelastic_run_id") != identity
            or runtime.get("master_addr") != node_name
            or runtime.get("node_name") != node_name
            or runtime.get("slurm_job_id") != slurm_job_id
            or not 1 <= master_port <= 65535
        ):
            raise ValueError(f"GeoRoute {label} runtime identity did not match torchrun")
        identities.add(identity)
        ports.add(master_port)
    if len(identities) != 2:
        raise ValueError("GeoRoute concurrent probes reused a rendezvous ID")
    if len(ports) != 2:
        raise ValueError("GeoRoute concurrent probes reused an actual TCPStore port")


def _wait_until_ready(
    *,
    processes: Mapping[str, subprocess.Popen[str]],
    ready_files: Mapping[str, Path],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if all(path.is_file() for path in ready_files.values()):
            return
        failed = {
            label: process.returncode
            for label, process in processes.items()
            if process.poll() is not None
        }
        if failed:
            raise RuntimeError(
                f"GeoRoute rendezvous probe exited before both were ready: {failed}"
            )
        if time.monotonic() >= deadline:
            ready = {
                label: path.is_file()
                for label, path in ready_files.items()
            }
            states = {
                label: process.poll()
                for label, process in processes.items()
            }
            raise TimeoutError(
                "GeoRoute concurrent rendezvous readiness timed out: "
                f"ready={ready}, returncodes={states}, timeout_seconds={timeout_seconds}"
            )
        time.sleep(0.05)


def _diagnostic_output(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    truncated = len(encoded) > _DIAGNOSTIC_OUTPUT_LIMIT
    if truncated:
        tail = encoded[-_DIAGNOSTIC_OUTPUT_LIMIT:].decode(
            "utf-8",
            errors="replace",
        )
    else:
        tail = text
    return {
        "output_sha256": _sha256_text(text),
        "output_bytes": len(encoded),
        "output_truncated_to_tail": truncated,
        "output_tail": tail,
    }


def _signal_process_group(
    process: subprocess.Popen[str],
    requested_signal: signal.Signals,
) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, requested_signal)
        elif process.poll() is None:
            if requested_signal == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
    except ProcessLookupError:
        return


def _stop_process_groups(
    processes: Mapping[str, subprocess.Popen[str]],
) -> None:
    for process in processes.values():
        _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes.values()):
            break
        time.sleep(0.05)
    for process in processes.values():
        _signal_process_group(process, _KILL_SIGNAL)
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                continue


def run_gate(
    *,
    output: Path,
    expected_commit: str,
    write_boundary: Path | None = None,
) -> dict[str, Any]:
    failure_output = output.with_suffix(".failure.json")
    if output.exists() or failure_output.exists():
        raise FileExistsError(output if output.exists() else failure_output)
    boundary = (
        Path("/data/run01/sczc063/yuzibo")
        if write_boundary is None
        else write_boundary
    ).resolve()
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(boundary)
    except ValueError as error:
        raise ValueError(
            "GeoRoute rendezvous gate output leaves the remote write boundary"
        ) from error
    if resolved_output == boundary:
        raise ValueError("GeoRoute rendezvous gate output cannot be the boundary root")
    output = resolved_output
    failure_output = output.with_suffix(".failure.json")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("GeoRoute rendezvous gate must run inside Slurm")
    runtime_commit = _git_output("rev-parse", "HEAD").lower()
    if runtime_commit != expected_commit.lower() or len(runtime_commit) != 40:
        raise RuntimeError("GeoRoute rendezvous gate source commit mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("GeoRoute rendezvous gate requires a clean source snapshot")

    output.parent.mkdir(parents=True, exist_ok=True)
    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    inherited["OMP_NUM_THREADS"] = "1"
    started = time.monotonic()
    probe_records: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix="georoute_rdzv_",
        dir=output.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        release_file = temporary_root / "release"
        short_exited_file = temporary_root / "short.exited"
        ready_files = {
            "short": temporary_root / "short.ready.json",
            "long": temporary_root / "long.ready.json",
        }
        durations = {"short": 0.1, "long": 0.1}
        processes: dict[str, subprocess.Popen[str]] = {}
        rendezvous_receipts: dict[str, dict[str, Any]] = {}
        commands: dict[str, list[str]] = {}
        outputs = {"short": "", "long": ""}
        collected: set[str] = set()
        runtime_identities: dict[str, dict[str, Any]] = {}
        failure: Exception | None = None
        release_to_short_exit = -1.0
        release_to_long_exit = -1.0
        long_alive_after_short = False
        for label in ("long", "short"):
            prefix, rendezvous = build_torchrun_prefix(
                phase="train",
                slurm_job_id=slurm_job_id,
                stage="p0",
                variant=f"rendezvous_probe_{label}",
                seed=3407,
                rendezvous_slot=0 if label == "long" else 1,
            )
            command = [
                *prefix,
                "-m",
                "tools.bata.georoute_rendezvous_probe",
                "--label",
                label,
                "--ready-file",
                str(ready_files[label]),
                "--release-file",
                str(release_file),
                "--post-release-seconds",
                str(durations[label]),
            ]
            if label == "long":
                command.extend(
                    [
                        "--peer-exit-file",
                        str(short_exited_file),
                    ]
                )
            processes[label] = subprocess.Popen(
                command,
                cwd=ROOT,
                env=inherited,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            commands[label] = command
            rendezvous_receipts[label] = rendezvous
        try:
            _wait_until_ready(
                processes=processes,
                ready_files=ready_files,
                timeout_seconds=READINESS_TIMEOUT_SECONDS,
            )
            runtime_identities = {
                label: json.loads(path.read_text(encoding="utf-8"))
                for label, path in ready_files.items()
            }
            release_file.write_text("release\n", encoding="utf-8")
            released = time.monotonic()
            outputs["short"], _ = processes["short"].communicate(timeout=30.0)
            collected.add("short")
            release_to_short_exit = time.monotonic() - released
            long_alive_after_short = processes["long"].poll() is None
            short_exited_file.write_text("short exited\n", encoding="utf-8")
            outputs["long"], _ = processes["long"].communicate(timeout=30.0)
            collected.add("long")
            release_to_long_exit = time.monotonic() - released
        except Exception as error:
            failure = error
        finally:
            if failure is not None or any(
                process.poll() is None for process in processes.values()
            ):
                _stop_process_groups(processes)
            for label, process in processes.items():
                if label not in collected:
                    try:
                        captured, _ = process.communicate(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _signal_process_group(process, _KILL_SIGNAL)
                        try:
                            captured, _ = process.communicate(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            captured = (
                                "[GeoRoute diagnostic drain timed out after "
                                "process-group SIGKILL]\n"
                            )
                    outputs[label] = captured or ""
                    collected.add(label)
        if failure is not None:
            ready_payloads: dict[str, Any] = {}
            for label, path in ready_files.items():
                if path.is_file():
                    try:
                        ready_payloads[label] = json.loads(
                            path.read_text(encoding="utf-8")
                        )
                    except Exception as parse_error:
                        ready_payloads[label] = {
                            "parse_error": type(parse_error).__name__,
                            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
            failure_core: dict[str, Any] = {
                "schema_version": GEOROUTE_RENDEZVOUS_FAILURE_SCHEMA,
                "status": "FAIL_CONCURRENT_RENDEZVOUS_ISOLATION",
                "runtime_commit": runtime_commit,
                "slurm_job_id": slurm_job_id,
                "node_name": socket.gethostname(),
                "readiness_timeout_seconds": READINESS_TIMEOUT_SECONDS,
                "elapsed_seconds": time.monotonic() - started,
                "exception_type": type(failure).__name__,
                "exception_message": str(failure),
                "probes": {
                    label: {
                        "command": commands[label],
                        "rendezvous": rendezvous_receipts[label],
                        "return_code": processes[label].returncode,
                        "ready_marker_seen": ready_files[label].is_file(),
                        "runtime_identity": ready_payloads.get(label),
                        **_diagnostic_output(outputs[label]),
                    }
                    for label in ("short", "long")
                },
                "selected_environment": {
                    key: inherited.get(key)
                    for key in (
                        "SLURM_JOB_ID",
                        "SLURM_STEP_ID",
                        "SLURM_NODEID",
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                    )
                },
                "official_test_opened": False,
                "model_forward_executed": False,
                "paper_claim_allowed": False,
            }
            failure_payload = {
                **failure_core,
                "failure_sha256": canonical_sha256(failure_core),
            }
            _atomic_write_json(failure_output, failure_payload)
            raise RuntimeError(
                f"{failure}; rendezvous failure receipt: {failure_output}"
            ) from failure
        for label in ("short", "long"):
            output_text = outputs[label]
            probe_records[label] = {
                "rendezvous": rendezvous_receipts[label],
                "runtime_identity": runtime_identities[label],
                "exit_code": int(processes[label].returncode),
                "ready_marker_seen": ready_files[label].is_file(),
                "done_marker_seen": (
                    '"event": "GEOROUTE_RDZV_DONE"' in output_text
                    and f'"label": "{label}"' in output_text
                ),
                "peer_exit_marker_seen": (
                    label == "short"
                    or '"peer_exit_observed": true' in output_text
                ),
                "output_sha256": _sha256_text(output_text),
                "requested_post_release_seconds": durations[label],
            }
    core: dict[str, Any] = {
        "schema_version": GEOROUTE_RENDEZVOUS_GATE_SCHEMA,
        "status": "PASS_CONCURRENT_RENDEZVOUS_ISOLATION",
        "runtime_commit": runtime_commit,
        "slurm_job_id": slurm_job_id,
        "node_name": socket.gethostname(),
        "same_node_concurrent": True,
        "long_probe_alive_after_short_exit": long_alive_after_short,
        "release_to_short_exit_seconds": release_to_short_exit,
        "release_to_long_exit_seconds": release_to_long_exit,
        "readiness_timeout_seconds": READINESS_TIMEOUT_SECONDS,
        "elapsed_seconds": time.monotonic() - started,
        "probes": probe_records,
        "official_test_opened": False,
        "model_forward_executed": False,
        "paper_claim_allowed": False,
    }
    payload = {**core, "gate_sha256": canonical_sha256(core)}
    validate_rendezvous_gate_receipt(
        payload,
        expected_commit=runtime_commit,
        expected_node_name=socket.gethostname(),
    )
    _atomic_write_json(output, payload)
    return payload


def _write_gate_failsafe_failure(
    *,
    output: Path,
    expected_commit: str,
    error: Exception,
    write_boundary: Path | None = None,
) -> None:
    boundary = (
        Path("/data/run01/sczc063/yuzibo")
        if write_boundary is None
        else write_boundary
    ).resolve()
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(boundary)
    except ValueError:
        return
    failure_output = resolved_output.with_suffix(".failure.json")
    if (
        resolved_output == boundary
        or resolved_output.exists()
        or failure_output.exists()
    ):
        return
    try:
        runtime_commit = _git_output("rev-parse", "HEAD").lower()
    except Exception:
        runtime_commit = None
    failure_core: dict[str, Any] = {
        "schema_version": GEOROUTE_RENDEZVOUS_FAILURE_SCHEMA,
        "status": "FAIL_CONCURRENT_RENDEZVOUS_ISOLATION",
        "runtime_commit": runtime_commit,
        "expected_runtime_commit": expected_commit.lower(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node_name": socket.gethostname(),
        "failure_phase": "gate_prevalidation_or_namespace_setup",
        "readiness_timeout_seconds": READINESS_TIMEOUT_SECONDS,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "probes": {},
        "official_test_opened": False,
        "model_forward_executed": False,
        "paper_claim_allowed": False,
    }
    failure = {
        **failure_core,
        "failure_sha256": canonical_sha256(failure_core),
    }
    _atomic_write_json(failure_output, failure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        payload = run_gate(
            output=args.output.resolve(),
            expected_commit=args.expected_commit.lower(),
        )
    except Exception as exc:
        try:
            _write_gate_failsafe_failure(
                output=args.output,
                expected_commit=args.expected_commit,
                error=exc,
            )
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
