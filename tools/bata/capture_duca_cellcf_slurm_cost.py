from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from tools.bata.duca_cellcf_training import canonical_sha256


SCHEMA = "duca_cellcf_slurm_training_cost_v1"
_TERMINAL_STATES = {"COMPLETED"}
_MEMORY_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)([KMGTP]?)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_memory_bytes(value: str) -> int | None:
    normalized = value.strip().upper()
    if not normalized or normalized in {"N/A", "UNKNOWN"}:
        return None
    match = _MEMORY_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(f"unsupported Slurm memory value: {value!r}")
    number = float(match.group(1))
    exponent = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2)]
    return int(number * (1024**exponent))


def _parse_allocated_gpus(alloc_tres: str) -> int:
    matches = re.findall(
        r"(?:^|,)(?:gres/)?gpu(?::[^=,]+)?=([0-9]+)(?:,|$)", alloc_tres
    )
    _require(len(matches) == 1, f"cannot resolve one GPU count from AllocTRES={alloc_tres!r}")
    return int(matches[0])


def _parse_energy_joules(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "UNKNOWN"}:
        return None
    energy = float(normalized)
    if energy <= 0:
        return None
    return energy


def _run_sacct(command: Sequence[str]) -> str:
    return subprocess.check_output(
        list(command), text=True, encoding="utf-8", stderr=subprocess.STDOUT
    )


def parse_sacct_output(
    output: str,
    *,
    job_id: int,
    expected_job_name: str,
    expected_cluster: str,
    command: Sequence[str],
) -> dict[str, Any]:
    rows = [
        line.split("|")
        for line in output.splitlines()
        if line.strip()
    ]
    _require(len(rows) == 1, f"expected one top-level sacct row, got {len(rows)}")
    row = rows[0]
    _require(len(row) == 11, f"unexpected sacct field count: {len(row)}")
    (
        observed_id,
        job_name,
        cluster,
        state,
        exit_code,
        elapsed_raw,
        alloc_tres,
        max_rss,
        energy_raw,
        start,
        end,
    ) = row
    _require(observed_id == str(job_id), "sacct returned another job id")
    _require(job_name == expected_job_name, "Slurm job name mismatch")
    _require(cluster == expected_cluster, "Slurm cluster mismatch")
    _require(state in _TERMINAL_STATES, f"Slurm job is not successfully complete: {state}")
    _require(exit_code == "0:0", f"Slurm job exit code is not zero: {exit_code}")
    _require(elapsed_raw.isdigit() and int(elapsed_raw) > 0, "invalid Slurm elapsed time")
    allocated_gpus = _parse_allocated_gpus(alloc_tres)
    _require(allocated_gpus == 1, "formal CellCF training must use exactly one GPU")
    elapsed_seconds = int(elapsed_raw)
    raw_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
    return {
        "measurement_scope": (
            "entire_slurm_allocation_including_training_terminal_evaluation_"
            "and_finalization"
        ),
        "job_id": job_id,
        "job_name": job_name,
        "cluster": cluster,
        "state": state,
        "exit_code": exit_code,
        "allocation_elapsed_seconds": elapsed_seconds,
        "allocated_gpus": allocated_gpus,
        "allocation_gpu_hours": elapsed_seconds * allocated_gpus / 3600.0,
        "allocation_peak_cpu_rss_bytes": _parse_memory_bytes(max_rss),
        "allocation_consumed_energy_joules": _parse_energy_joules(energy_raw),
        "allocation_energy_scope": (
            "slurm_consumed_energy_raw_not_assumed_to_be_gpu_only"
        ),
        "gpu_peak_memory_bytes": None,
        "gpu_peak_memory_source": "not_available_from_top_level_sacct",
        "start": start,
        "end": end,
        "alloc_tres": alloc_tres,
        "sacct_command": list(command),
        "sacct_raw_sha256": raw_sha256,
    }


def _exclusive_write_text(path: str | Path, text: str) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def capture_slurm_cost(
    *,
    job_id: int,
    expected_job_name: str,
    expected_cluster: str,
    raw_output_path: str | Path | None = None,
    run_command: Callable[[Sequence[str]], str] = _run_sacct,
) -> dict[str, Any]:
    _require(job_id > 0, "job id must be positive")
    _require(bool(expected_job_name), "expected job name is required")
    _require(bool(expected_cluster), "expected cluster is required")
    fields = (
        "JobIDRaw,JobName%128,Cluster%64,State,ExitCode,ElapsedRaw,AllocTRES%512,"
        "MaxRSS,ConsumedEnergyRaw,Start,End"
    )
    command = [
        "sacct",
        "--clusters",
        expected_cluster,
        "--jobs",
        str(job_id),
        "--parsable2",
        "--noheader",
        "--allocations",
        "--format",
        fields,
    ]
    output = run_command(command)
    parsed = parse_sacct_output(
        output,
        job_id=job_id,
        expected_job_name=expected_job_name,
        expected_cluster=expected_cluster,
        command=command,
    )
    raw_path = (
        None
        if raw_output_path is None
        else _exclusive_write_text(raw_output_path, output)
    )
    payload = {
        "schema": SCHEMA,
        "ok": True,
        **parsed,
        "sacct_raw_artifact_path": (
            None if raw_path is None else str(raw_path)
        ),
        "sacct_raw_artifact_sha256": (
            None
            if raw_path is None
            else hashlib.sha256(raw_path.read_bytes()).hexdigest()
        ),
        "sacct_raw_replayable": raw_path is not None,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--expected-job-name", required=True)
    parser.add_argument("--expected-cluster", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output_json).expanduser().resolve()
    raw_output_path = Path(args.raw_output).expanduser().resolve()
    if output_path.exists() or raw_output_path.exists():
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": "FileExistsError",
            "error": "refusing to overwrite Slurm cost evidence",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    try:
        payload = capture_slurm_cost(
            job_id=args.job_id,
            expected_job_name=args.expected_job_name,
            expected_cluster=args.expected_cluster,
            raw_output_path=raw_output_path,
        )
        _exclusive_write_json(output_path, payload)
        code = 0
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not output_path.exists():
            _exclusive_write_json(output_path, payload)
        code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
