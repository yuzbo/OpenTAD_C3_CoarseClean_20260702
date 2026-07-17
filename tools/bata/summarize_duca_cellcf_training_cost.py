from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.bata.capture_duca_cellcf_slurm_cost import (
    SCHEMA as SLURM_COST_SCHEMA,
    parse_sacct_output,
)
from tools.bata.duca_cellcf_protocol import (
    protocol_for_name,
    protocol_from_environment,
)
from tools.bata.duca_cellcf_suite_binding import load_suite_aggregate_binding
from tools.bata.duca_cellcf_training import VARIANTS, canonical_sha256, sha256_file


SCHEMA = "duca_cellcf_training_cost_summary_v1"
POST_RUN_SCHEMA = "duca_cellcf_post_run_evidence_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_hash(payload: Mapping[str, Any], key: str, label: str) -> None:
    observed = payload.get(key)
    unsigned = dict(payload)
    unsigned.pop(key, None)
    _require(
        isinstance(observed, str) and observed == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _parse_bindings(values: Sequence[str], label: str) -> dict[str, str]:
    output = {}
    for value in values:
        variant, separator, path = value.partition("=")
        _require(separator == "=" and variant in VARIANTS and path, f"invalid {label}: {value}")
        _require(variant not in output, f"duplicate {label}: {variant}")
        output[variant] = path
    _require(set(output) == set(VARIANTS), f"{label} must cover exactly three variants")
    return output


def _replay_slurm_cost(
    cost: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    raw_path_text = cost.get("sacct_raw_artifact_path")
    _require(
        isinstance(raw_path_text, str) and bool(raw_path_text),
        f"{label} has no raw sacct artifact path",
    )
    _require(
        cost.get("sacct_raw_replayable") is True,
        f"{label} is not marked replayable",
    )
    raw_path = Path(raw_path_text).expanduser().resolve()
    _require(raw_path.is_file(), f"{label} raw sacct artifact is missing: {raw_path}")
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    _require(
        cost.get("sacct_raw_artifact_sha256") == raw_sha256,
        f"{label} raw sacct artifact hash mismatch",
    )
    command = cost.get("sacct_command")
    _require(
        isinstance(command, list)
        and command
        and all(isinstance(item, str) for item in command),
        f"{label} sacct command is invalid",
    )
    job_id = cost.get("job_id")
    _require(
        isinstance(job_id, int) and not isinstance(job_id, bool) and job_id > 0,
        f"{label} job id is invalid",
    )
    job_name = cost.get("job_name")
    cluster = cost.get("cluster")
    _require(
        isinstance(job_name, str) and bool(job_name),
        f"{label} job name is invalid",
    )
    _require(
        isinstance(cluster, str) and bool(cluster),
        f"{label} cluster is invalid",
    )
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} raw sacct artifact is not UTF-8") from exc
    replayed = parse_sacct_output(
        raw_text,
        job_id=job_id,
        expected_job_name=job_name,
        expected_cluster=cluster,
        command=command,
    )
    for key, value in replayed.items():
        _require(
            cost.get(key) == value,
            f"{label} differs from replayed raw sacct data on {key}",
        )
    _require(
        replayed["sacct_raw_sha256"] == raw_sha256,
        f"{label} replayed raw sacct hash mismatch",
    )
    return raw_path, replayed


def summarize_training_cost(
    *,
    expected_commit: str,
    suite_aggregate_path: str | Path,
    suite_aggregate_sha256: str,
    post_run_paths: Mapping[str, str | Path],
    slurm_cost_paths: Mapping[str, str | Path],
    suite_loader: Callable[..., Mapping[str, Any]] = load_suite_aggregate_binding,
) -> dict[str, Any]:
    _require(set(post_run_paths) == set(VARIANTS), "post-run paths are incomplete")
    _require(set(slurm_cost_paths) == set(VARIANTS), "Slurm cost paths are incomplete")
    expected_profile = protocol_from_environment().name
    suite_binding = suite_loader(
        suite_aggregate_path,
        suite_aggregate_sha256,
        expected_commit=expected_commit,
        expected_profile=expected_profile,
        post_run_paths=post_run_paths,
    )
    rows = []
    profile_name = str(suite_binding["training_profile"])
    for variant in VARIANTS:
        post_record = suite_binding["post_runs"][variant]
        post_path = Path(post_record["path"])
        post = post_record["payload"]
        cost_path, cost = _load_json(slurm_cost_paths[variant], f"{variant} Slurm cost")
        _require(post.get("schema") == POST_RUN_SCHEMA and post.get("ok") is True, f"{variant} post-run is invalid")
        _require(cost.get("schema") == SLURM_COST_SCHEMA and cost.get("ok") is True, f"{variant} Slurm cost is invalid")
        _validate_hash(cost, "record_sha256", f"{variant} Slurm cost")
        raw_cost_path, _ = _replay_slurm_cost(
            cost,
            label=f"{variant} Slurm cost",
        )
        _require(post.get("variant") == variant, f"{variant} post-run variant mismatch")
        _require(post.get("git_commit") == expected_commit, f"{variant} post-run commit mismatch")
        protocol = protocol_for_name(profile_name)
        _require(
            int(post.get("successful_optimizer_updates", -1))
            == protocol.expected_successful_optimizer_updates,
            f"{variant} post-run update count mismatch",
        )
        audit_record = post_record["training_audit"]
        job_id = str(cost.get("job_id"))
        _require(
            int(audit_record["slurm_job_id"]) == int(job_id),
            f"{variant} Slurm job id differs from training audit",
        )
        _require(
            cost.get("measurement_scope")
            == (
                "entire_slurm_allocation_including_training_terminal_"
                "evaluation_and_finalization"
            ),
            f"{variant} Slurm cost scope is invalid",
        )
        elapsed = float(cost.get("allocation_elapsed_seconds", -1))
        gpu_hours = float(cost.get("allocation_gpu_hours", -1))
        _require(elapsed > 0 and math.isfinite(elapsed), f"{variant} elapsed time is invalid")
        _require(gpu_hours > 0 and math.isfinite(gpu_hours), f"{variant} allocation GPU hours are invalid")
        expected_gpu_hours = elapsed * int(cost.get("allocated_gpus", 0)) / 3600.0
        _require(math.isclose(gpu_hours, expected_gpu_hours, rel_tol=0.0, abs_tol=1e-12), f"{variant} GPU-hour accounting mismatch")
        rows.append(
            {
                "variant": variant,
                "training_profile": profile_name,
                "job_id": int(cost["job_id"]),
                "allocation_elapsed_seconds": elapsed,
                "allocation_gpu_hours": gpu_hours,
                "successful_optimizer_updates": protocol.expected_successful_optimizer_updates,
                "allocation_seconds_per_successful_update": elapsed
                / protocol.expected_successful_optimizer_updates,
                "allocation_peak_cpu_rss_bytes": cost.get(
                    "allocation_peak_cpu_rss_bytes"
                ),
                "gpu_peak_memory_bytes": cost.get("gpu_peak_memory_bytes"),
                "allocation_consumed_energy_joules": cost.get(
                    "allocation_consumed_energy_joules"
                ),
                "training_only_gpu_hours": None,
                "post_run_path": str(post_path),
                "post_run_sha256": sha256_file(post_path),
                "training_audit_path": audit_record["path"],
                "training_audit_sha256": audit_record["sha256"],
                "slurm_cost_path": str(cost_path),
                "slurm_cost_sha256": sha256_file(cost_path),
                "sacct_raw_artifact_path": str(raw_cost_path),
                "sacct_raw_artifact_sha256": sha256_file(raw_cost_path),
            }
        )

    by_variant = {row["variant"]: row for row in rows}
    uniform = by_variant["uniform"]
    transition = by_variant["transition_beta0"]
    cellcf = by_variant["cellcf"]
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "training_profile": profile_name,
        "training_protocol": protocol_for_name(profile_name).to_dict(),
        "suite_aggregate_binding": {
            key: suite_binding[key]
            for key in (
                "path",
                "sha256",
                "seed",
                "variant_order",
                "shared_protocol_sha256",
                "ordered_exposure_sha256",
                "real_loader_gate",
                "ddp_pilot",
            )
        },
        "rows": rows,
        "total_three_arm_allocation_gpu_hours": sum(
            row["allocation_gpu_hours"] for row in rows
        ),
        "training_only_gpu_hours": {
            "available": False,
            "reason": (
                "slurm_accounting_covers_the_full_allocation_and_does_not_"
                "separate_training_from_terminal_evaluation_and_finalization"
            ),
        },
        "relative_training_cost": {
            "transition_minus_uniform_allocation_gpu_hours": (
                transition["allocation_gpu_hours"]
                - uniform["allocation_gpu_hours"]
            ),
            "cellcf_minus_transition_allocation_gpu_hours": (
                cellcf["allocation_gpu_hours"]
                - transition["allocation_gpu_hours"]
            ),
            "cellcf_over_uniform_allocation_gpu_hour_ratio": (
                cellcf["allocation_gpu_hours"]
                / uniform["allocation_gpu_hours"]
            ),
        },
        "availability": {
            "all_gpu_peak_memory_measured": all(
                row["gpu_peak_memory_bytes"] is not None for row in rows
            ),
            "all_energy_measured": all(
                row["allocation_consumed_energy_joules"] is not None
                for row in rows
            ),
        },
        "break_even": {
            "available": False,
            "reason": "requires_separate_dense_vs_cellcf_full_stack_inference_saving",
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _exclusive_write_tsv(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "training_profile",
        "job_id",
        "allocation_elapsed_seconds",
        "allocation_gpu_hours",
        "successful_optimizer_updates",
        "allocation_seconds_per_successful_update",
        "allocation_peak_cpu_rss_bytes",
        "gpu_peak_memory_bytes",
        "allocation_consumed_energy_joules",
        "training_only_gpu_hours",
    ]
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--suite-aggregate", required=True)
    parser.add_argument("--suite-aggregate-sha256", required=True)
    parser.add_argument("--post-run", action="append", required=True)
    parser.add_argument("--slurm-cost", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    args = parser.parse_args(argv)
    output_json = Path(args.output_json).expanduser().resolve()
    output_tsv = Path(args.output_tsv).expanduser().resolve()
    if output_json.exists() or output_tsv.exists():
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": "FileExistsError",
            "error": "refusing to overwrite CellCF training-cost evidence",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    try:
        payload = summarize_training_cost(
            expected_commit=args.expected_commit,
            suite_aggregate_path=args.suite_aggregate,
            suite_aggregate_sha256=args.suite_aggregate_sha256,
            post_run_paths=_parse_bindings(args.post_run, "post-run binding"),
            slurm_cost_paths=_parse_bindings(args.slurm_cost, "Slurm cost binding"),
        )
        _exclusive_write_tsv(output_tsv, payload["rows"])
        _exclusive_write_json(output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not output_json.exists():
            _exclusive_write_json(output_json, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
