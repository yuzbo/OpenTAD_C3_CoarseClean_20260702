#!/usr/bin/env python3
"""Capture and validate Slurm identity for a decode-cross experiment DAG."""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-tsv", required=True)
    parser.add_argument("--dag-token", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("submission", "terminal"),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jobs(path):
    path = Path(path).resolve()
    require(path.is_file(), f"jobs TSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    require(len(records) == 6, "decode-cross jobs TSV must contain six jobs")
    require(
        len({record["job_id"] for record in records}) == len(records),
        "decode-cross jobs TSV contains duplicate job IDs",
    )
    return path, records


def parse_scontrol_line(line):
    fields = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def capture_submission(records, dag_token):
    reports = {}
    for record in records:
        output = subprocess.check_output(
            ["scontrol", "show", "job", "-o", record["job_id"]],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        fields = parse_scontrol_line(output)
        require(
            fields.get("JobId") == record["job_id"],
            f"{record['variant']} scontrol JobId mismatch",
        )
        require(
            fields.get("JobName") == record["job_name"],
            f"{record['variant']} scontrol JobName mismatch",
        )
        require(
            fields.get("Comment") == record["comment"]
            == f"{dag_token}:{record['variant']}",
            f"{record['variant']} Slurm comment mismatch",
        )
        expected_dependency = record["dependency"]
        actual_dependency = fields.get("Dependency", "")
        normalized_dependency = re.sub(
            r"\([^)]*\)",
            "",
            actual_dependency,
        )
        if expected_dependency == "none":
            require(
                actual_dependency in {"", "(null)", "none"},
                "gate unexpectedly has a Slurm dependency",
            )
        else:
            require(
                normalized_dependency == expected_dependency,
                f"{record['variant']} Slurm dependency mismatch: "
                f"{actual_dependency} != {expected_dependency}",
            )
        require(
            str(Path(fields.get("StdOut", "")).resolve())
            == str(Path(record["stdout"]).resolve())
            and str(Path(fields.get("StdErr", "")).resolve())
            == str(Path(record["stderr"]).resolve()),
            f"{record['variant']} Slurm log path mismatch",
        )
        reports[record["variant"]] = {
            "job_id": record["job_id"],
            "job_name": fields["JobName"],
            "comment": fields["Comment"],
            "dependency": actual_dependency,
            "stdout": str(Path(fields["StdOut"]).resolve()),
            "stderr": str(Path(fields["StdErr"]).resolve()),
            "raw_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    return reports


def parse_sacct(records):
    job_ids = ",".join(record["job_id"] for record in records)
    output = subprocess.check_output(
        [
            "sacct",
            "-nX",
            "-j",
            job_ids,
            "--format=JobIDRaw,JobName%128,State,ExitCode,Comment%256",
            "-P",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    reports = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        reports[parts[0]] = {
            "job_id": parts[0],
            "job_name": parts[1].strip(),
            "state": parts[2].strip().split()[0],
            "exit_code": parts[3].strip(),
            "comment": parts[4].strip(),
        }
    return reports


def capture_terminal(records, dag_token):
    reports = {}
    sacct_records = {}
    for _ in range(12):
        sacct_records = parse_sacct(records)
        if all(record["job_id"] in sacct_records for record in records):
            break
        time.sleep(5)
    require(
        all(record["job_id"] in sacct_records for record in records),
        "sacct did not expose all decode-cross jobs",
    )
    for record in records:
        current = sacct_records[record["job_id"]]
        require(
            current["job_name"] == record["job_name"],
            f"{record['variant']} sacct JobName mismatch",
        )
        require(
            current["comment"] == record["comment"]
            == f"{dag_token}:{record['variant']}",
            f"{record['variant']} sacct comment mismatch",
        )
        if record["variant"] == "decode_cross_suite":
            require(
                current["state"] in {"RUNNING", "COMPLETING"},
                "suite job is not the active terminal validator",
            )
        else:
            require(
                current["state"] == "COMPLETED"
                and current["exit_code"] == "0:0",
                f"{record['variant']} did not complete cleanly in sacct",
            )
        reports[record["variant"]] = current
    return reports


def main():
    args = parse_args()
    jobs_path, records = read_jobs(args.jobs_tsv)
    require(
        all(record["dag_token"] == args.dag_token for record in records),
        "jobs TSV DAG token mismatch",
    )
    if args.mode == "submission":
        reports = capture_submission(records, args.dag_token)
    else:
        reports = capture_terminal(records, args.dag_token)
    payload = {
        "schema_version": "phystime_decode_cross_scheduler_snapshot_v1",
        "validation_pass": True,
        "mode": args.mode,
        "dag_token": args.dag_token,
        "jobs_tsv": str(jobs_path),
        "jobs_tsv_sha256": hashlib.sha256(
            jobs_path.read_bytes()
        ).hexdigest(),
        "jobs": reports,
        "captured_at_unix": time.time(),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
