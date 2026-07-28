#!/usr/bin/env python3
"""Prove two same-node GeoRoute torchrun parents have independent stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata.georoute_stage_runner import build_torchrun_prefix


ROOT = Path(__file__).resolve().parents[2]
GEOROUTE_RENDEZVOUS_GATE_SCHEMA = "georoute_rendezvous_isolation_gate_v2"
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
    short_exit = float(payload.get("release_to_short_exit_seconds", -1.0))
    long_exit = float(payload.get("release_to_long_exit_seconds", -1.0))
    if short_exit <= 0 or long_exit <= short_exit:
        raise ValueError("GeoRoute rendezvous gate lacks ordered lifetime evidence")
    slurm_job_id = payload.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id:
        raise ValueError("GeoRoute rendezvous gate lacks its Slurm job identity")
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
        if (
            rendezvous.get("phase") != "train"
            or rendezvous.get("backend") != "c10d"
            or rendezvous.get("endpoint") != "127.0.0.1:0"
            or rendezvous.get("endpoint_policy")
            != "kernel_assigned_loopback_port"
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
            or runtime.get("master_addr") != "127.0.0.1"
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
            raise TimeoutError("GeoRoute concurrent rendezvous readiness timed out")
        time.sleep(0.05)


def run_gate(*, output: Path, expected_commit: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
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
        short_output = ""
        long_output = ""
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
            )
            rendezvous_receipts[label] = rendezvous
        try:
            _wait_until_ready(
                processes=processes,
                ready_files=ready_files,
                timeout_seconds=30.0,
            )
            runtime_identities = {
                label: json.loads(path.read_text(encoding="utf-8"))
                for label, path in ready_files.items()
            }
            release_file.write_text("release\n", encoding="utf-8")
            released = time.monotonic()
            short_output, _ = processes["short"].communicate(timeout=30.0)
            release_to_short_exit = time.monotonic() - released
            long_alive_after_short = processes["long"].poll() is None
            short_exited_file.write_text("short exited\n", encoding="utf-8")
            long_output, _ = processes["long"].communicate(timeout=30.0)
            release_to_long_exit = time.monotonic() - released
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                if process.poll() is None:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
        outputs = {"short": short_output, "long": long_output}
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
        "same_node_concurrent": True,
        "long_probe_alive_after_short_exit": long_alive_after_short,
        "release_to_short_exit_seconds": release_to_short_exit,
        "release_to_long_exit_seconds": release_to_long_exit,
        "elapsed_seconds": time.monotonic() - started,
        "probes": probe_records,
        "official_test_opened": False,
        "model_forward_executed": False,
        "paper_claim_allowed": False,
    }
    payload = {**core, "gate_sha256": canonical_sha256(core)}
    validate_rendezvous_gate_receipt(payload, expected_commit=runtime_commit)
    _atomic_write_json(output, payload)
    return payload


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
