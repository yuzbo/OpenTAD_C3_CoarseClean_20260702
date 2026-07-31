#!/usr/bin/env python3
"""Run one development-only GeoRoute-AdaTAD training/evaluation cell.

This is intentionally a narrow execution adapter around the existing OpenTAD
``tools/train.py`` and ``tools/test.py`` entry points.  It never opens an
official-test split, never chooses a checkpoint by a metric, and never emits a
paper-grade cost record.  The only checkpoint it evaluates is the final EMA
checkpoint of the frozen stage budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    GEOROUTE_EXPERIMENT_SCHEMA,
    DEVELOPMENT_SEEDS,
    bind_development_config,
    canonical_sha256,
    paper_variant_name,
    sha256_file,
    stage_cell_relative_path,
    stage_epochs,
    variant_spec,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


GEOROUTE_STAGE_RESULT_SCHEMA = "georoute_adatad_stage_result_v3"
_AVERAGE_MAP = re.compile(r"Average-mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)")
_TIOU_MAP = re.compile(
    r"mAP at tIoU\s+([0-9]+(?:\.[0-9]+)?)\s+is\s+([0-9]+(?:\.[0-9]+)?)%"
)
_RENDEZVOUS_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_JOB_SCOPED_LOOPBACK_RADIX = 254
_MAX_JOB_SCOPED_LOOPBACK_ID = _JOB_SCOPED_LOOPBACK_RADIX**3
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
_LOG_DRAIN_TIMEOUT_SECONDS = 5.0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _inside(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
    except ValueError:
        return False
    return path != boundary


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


def _stop_logged_process_group(process: subprocess.Popen[str]) -> None:
    """Bound cleanup of a torchrun parent and every process in its session."""

    _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + _LOG_DRAIN_TIMEOUT_SECONDS
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.05)
    _signal_process_group(process, _KILL_SIGNAL)
    if process.poll() is None:
        try:
            process.wait(timeout=_LOG_DRAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return


def _run_logged(command: list[str], *, log_path: Path, env: Mapping[str, str]) -> None:
    """Stream one command to an immutable log and fail closed on process leaks."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        reader_errors: list[BaseException] = []

        def _drain_output() -> None:
            try:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
            except BaseException as error:
                reader_errors.append(error)

        reader = threading.Thread(
            target=_drain_output,
            name="georoute-log-drain",
            daemon=True,
        )
        reader.start()
        try:
            return_code = process.wait()
            if return_code != 0:
                _stop_logged_process_group(process)
            reader.join(timeout=_LOG_DRAIN_TIMEOUT_SECONDS)
            if reader.is_alive():
                _stop_logged_process_group(process)
                reader.join(timeout=_LOG_DRAIN_TIMEOUT_SECONDS)
            if reader.is_alive():
                raise RuntimeError(
                    "GeoRoute command output pipe remained open after bounded "
                    "process-group cleanup"
                )
            if reader_errors:
                raise RuntimeError("GeoRoute command log drain failed") from reader_errors[0]
        except BaseException:
            _stop_logged_process_group(process)
            reader.join(timeout=_LOG_DRAIN_TIMEOUT_SECONDS)
            raise
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")


def parse_official_style_map(log_text: str) -> dict[str, float]:
    """Parse OpenTAD's evaluator log without accepting partial metrics."""

    average_matches = _AVERAGE_MAP.findall(log_text)
    if not average_matches:
        raise ValueError("test log has no Average-mAP line")
    values: dict[str, float] = {"average_mAP": float(average_matches[-1])}
    for threshold, metric in _TIOU_MAP.findall(log_text):
        normalized = f"{float(threshold):.1f}"
        values[f"mAP@{normalized}"] = float(metric)
    required = {f"mAP@{threshold}" for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7")}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"test log lacks required tIoU metrics: {missing}")
    return values


def _development_profile(profile_path: Path) -> dict[str, Any]:
    profile = _read_json(profile_path)
    scope = profile.get("scope")
    if not isinstance(scope, dict) or scope.get("development_only") is not True:
        raise ValueError("GeoRoute profile is not development-only")
    if scope.get("paper_grade_end_to_end_claim_allowed") is not False:
        raise ValueError("GeoRoute development profiler was incorrectly promoted to paper-grade cost")
    for key in ("window_wall_p50_ms", "window_wall_p95_ms", "peak_allocated_mb"):
        value = profile.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"GeoRoute development profile lacks finite {key}")
    return {
        "development_window_wall_p50_ms": float(profile["window_wall_p50_ms"]),
        "development_window_wall_p95_ms": float(profile["window_wall_p95_ms"]),
        "development_peak_allocated_mb": float(profile["peak_allocated_mb"]),
        "paper_grade_end_to_end_claim_allowed": False,
        "profile_file_sha256": sha256_file(profile_path),
        "raw_scope": scope,
    }


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40:
        raise RuntimeError("GeoRoute runner could not resolve a full runtime commit")
    return commit


def _job_scoped_loopback(slurm_job_id: str) -> str:
    """Encode one decimal Slurm job ID into a distinct Linux loopback address."""

    value = str(slurm_job_id)
    if not value.isdigit():
        raise ValueError(f"unsafe GeoRoute rendezvous slurm_job_id: {value!r}")
    numeric = int(value)
    if not 1 <= numeric <= _MAX_JOB_SCOPED_LOOPBACK_ID:
        raise ValueError(
            "GeoRoute Slurm job ID exceeds the audited job-scoped loopback range"
        )
    remainder = numeric - 1
    octets = []
    for _ in range(3):
        remainder, digit = divmod(remainder, _JOB_SCOPED_LOOPBACK_RADIX)
        octets.append(digit + 1)
    if remainder:
        raise AssertionError("job-scoped loopback encoding overflow")
    return f"127.{octets[2]}.{octets[1]}.{octets[0]}"


def build_torchrun_prefix(
    *,
    phase: str,
    slurm_job_id: str,
    stage: str,
    variant: str,
    seed: int,
    rendezvous_slot: int | None = None,
    nproc_per_node: int = 1,
) -> tuple[list[str], dict[str, Any]]:
    """Build one collision-isolated single-node torchrun prefix and receipt."""

    components = {
        "phase": phase,
        "slurm_job_id": slurm_job_id,
        "stage": stage,
        "variant": variant,
        "seed": str(int(seed)),
    }
    if phase not in {"train", "test"}:
        raise ValueError("GeoRoute rendezvous phase must be train or test")
    if rendezvous_slot is None:
        rendezvous_slot = 0 if phase == "train" else 1
    if not isinstance(rendezvous_slot, int) or rendezvous_slot not in {0, 1}:
        raise ValueError("GeoRoute rendezvous slot must be 0 or 1")
    if not isinstance(nproc_per_node, int) or nproc_per_node <= 0:
        raise ValueError("GeoRoute rendezvous process count must be positive")
    for name, value in components.items():
        if not value or _RENDEZVOUS_COMPONENT.fullmatch(value) is None:
            raise ValueError(f"unsafe GeoRoute rendezvous {name}: {value!r}")
    endpoint_host = _job_scoped_loopback(slurm_job_id)
    endpoint = f"{endpoint_host}:0"
    rendezvous_id = (
        f"georoute-{slurm_job_id}-{stage}-{variant}-s{int(seed)}-{phase}"
    )
    receipt: dict[str, Any] = {
        "phase": phase,
        "backend": "c10d",
        "endpoint": endpoint,
        "endpoint_host": endpoint_host,
        "endpoint_policy": "job_scoped_loopback_and_kernel_assigned_port",
        "rendezvous_slot": rendezvous_slot,
        "rendezvous_id": rendezvous_id,
        "slurm_job_id": slurm_job_id,
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "nnodes": 1,
        "nproc_per_node": int(nproc_per_node),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={int(nproc_per_node)}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={endpoint}",
        f"--rdzv_id={rendezvous_id}",
    ]
    return command, receipt


def _validate_rendezvous_receipt(
    rendezvous: Mapping[str, Any],
    *,
    stage: str,
    variant: str,
    seed: int,
    nproc_per_node: int = 1,
) -> dict[str, Any]:
    """Fail closed if a stage result is not bound to two isolated launches."""

    validated: dict[str, Any] = {}
    for phase in ("train", "test"):
        record = rendezvous.get(phase)
        if not isinstance(record, Mapping):
            raise ValueError(f"GeoRoute stage result lacks {phase} rendezvous receipt")
        if (
            record.get("phase") != phase
            or record.get("backend") != "c10d"
            or record.get("stage") != stage
            or record.get("variant") != variant
            or int(record.get("seed", -1)) != int(seed)
            or int(record.get("nnodes", -1)) != 1
            or int(record.get("nproc_per_node", -1))
            != int(nproc_per_node)
        ):
            raise ValueError(f"invalid GeoRoute {phase} rendezvous receipt")
        rendezvous_id = record.get("rendezvous_id")
        slurm_job_id = record.get("slurm_job_id")
        if not isinstance(rendezvous_id, str) or not isinstance(slurm_job_id, str):
            raise ValueError(f"GeoRoute {phase} rendezvous identity is missing")
        expected_endpoint_host = _job_scoped_loopback(slurm_job_id)
        expected_slot = 0 if phase == "train" else 1
        if (
            record.get("endpoint") != f"{expected_endpoint_host}:0"
            or record.get("endpoint_host") != expected_endpoint_host
            or record.get("endpoint_policy")
            != "job_scoped_loopback_and_kernel_assigned_port"
            or int(record.get("rendezvous_slot", -1)) != expected_slot
        ):
            raise ValueError(
                f"GeoRoute {phase} rendezvous endpoint is not job-scoped"
            )
        expected_id = (
            f"georoute-{slurm_job_id}-{stage}-{variant}-s{int(seed)}-{phase}"
        )
        if rendezvous_id != expected_id:
            raise ValueError(f"GeoRoute {phase} rendezvous ID is not cell-bound")
        validated[phase] = dict(record)
    if validated["train"]["rendezvous_id"] == validated["test"]["rendezvous_id"]:
        raise ValueError("GeoRoute train and test rendezvous IDs must differ")
    if validated["train"]["slurm_job_id"] != validated["test"]["slurm_job_id"]:
        raise ValueError("GeoRoute train and test must share the bound Slurm leaf")
    return {
        "isolation_policy": (
            "job_scoped_loopback_kernel_assigned_endpoint_and_unique_cell_phase_id"
        ),
        **validated,
    }


def build_stage_result(
    *,
    stage: str,
    variant: str,
    seed: int,
    token_budget: int | None,
    binding: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    storage_receipt_path: Path,
    prediction_path: Path,
    profile_path: Path,
    test_log_path: Path,
    runtime_commit: str,
    rendezvous: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one result-blind, development-only cell receipt."""

    metrics = parse_official_style_map(test_log_path.read_text(encoding="utf-8", errors="replace"))
    profile = _development_profile(profile_path)
    raw_profile = _read_json(profile_path)
    audit = raw_profile.get("last_georoute_audit")
    if not isinstance(audit, dict):
        raise ValueError("GeoRoute development profile lacks the last routing audit")
    if int(audit.get("heavy_backbone_forward_count", -1)) != 1:
        raise ValueError("development cell did not preserve exactly one heavy backbone forward")
    if audit.get("uses_grid_sample") is not False or audit.get("uses_resized_local_crop") is not False:
        raise ValueError("development cell violated native-token routing constraints")
    selected_k = int(audit.get("target_k", -1))
    spec = variant_spec(variant, token_budget=token_budget)
    expected_k = selected_k if spec["tokens_per_tubelet"] is None else int(spec["tokens_per_tubelet"])
    if selected_k != expected_k:
        raise ValueError("routing audit target K does not match the bound variant")
    result = {
        "schema_version": GEOROUTE_STAGE_RESULT_SCHEMA,
        "status": "PASS_DEVELOPMENT_ONLY",
        "experiment_schema_version": GEOROUTE_EXPERIMENT_SCHEMA,
        "stage": stage,
        "variant": variant,
        "paper_variant": paper_variant_name(variant),
        "seed": int(seed),
        "token_budget": int(expected_k),
        "metrics": metrics,
        "profile": profile,
        "routing_audit": audit,
        "binding_sha256": str(binding["binding_sha256"]),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_receipt": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
            "policy": "final_only_atomic",
        },
        "storage_receipt": _read_json(storage_receipt_path),
        "prediction_sha256": sha256_file(prediction_path),
        "test_log_sha256": sha256_file(test_log_path),
        "runtime_commit": runtime_commit,
        "rendezvous": _validate_rendezvous_receipt(
            rendezvous,
            stage=stage,
            variant=variant,
            seed=seed,
        ),
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p1", "p2", "p3"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seed not in DEVELOPMENT_SEEDS:
        raise ValueError("stage runner seed is outside the frozen development seed set")
    if not args.expected_commit or _current_commit() != args.expected_commit.lower():
        raise RuntimeError("GeoRoute source snapshot does not match the bound commit")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("GeoRoute development cell must run inside a Slurm job")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(","):
        raise RuntimeError("GeoRoute development cell must see exactly one Slurm GPU")

    run_root = args.run_root.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, write_boundary):
        raise ValueError("GeoRoute run root must remain inside the remote write boundary")
    cell_path = stage_cell_relative_path(
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
        token_budget=args.token_budget,
    )
    budget_label = cell_path.parts[-2]
    work_root = run_root / cell_path
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{args.stage}_{args.variant}_{budget_label}_seed{args.seed}.py"
    )
    if work_root.exists() or bound_config.exists():
        raise FileExistsError("GeoRoute cell namespace already exists; refusing overwrite or resume")
    storage_profile_path = (
        run_root / "control" / "georoute_storage_profile.json"
    )
    if not storage_profile_path.is_file():
        raise FileNotFoundError(
            "GeoRoute cell requires the same-commit P0 storage profile"
        )
    storage_receipt = storage_capacity_receipt(
        run_root,
        cell_count=1,
        storage_profile=_read_json(storage_profile_path),
        expected_commit=args.expected_commit.lower(),
    )
    work_root.mkdir(parents=True, exist_ok=False)
    storage_receipt_path = work_root / "storage_preflight.json"
    _atomic_write_json(storage_receipt_path, storage_receipt)
    bound_config.parent.mkdir(parents=True, exist_ok=True)

    cfg = bind_development_config(
        source_config_path=args.source_config,
        variant=args.variant,
        stage=args.stage,
        seed=args.seed,
        work_dir=work_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        token_budget=args.token_budget,
    )
    cfg.post_processing.save_dict = True
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    train_log = work_root / "train.out"
    test_log = work_root / "test.out"
    train_torchrun, train_rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=slurm_job_id,
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
    )
    _run_logged(
        [*train_torchrun, "tools/train.py", str(bound_config), "--seed", str(args.seed), "--id", "0"],
        log_path=train_log,
        env=inherited,
    )
    effective_work_dir = work_root / "gpu1_id0"
    checkpoint_path = effective_work_dir / "checkpoint" / f"epoch_{stage_epochs(args.stage) - 1}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GeoRoute final EMA checkpoint is missing: {checkpoint_path}")
    checkpoint_dir = checkpoint_path.parent
    checkpoint_payloads = sorted(checkpoint_dir.glob("*.pth"))
    checkpoint_temporaries = sorted(checkpoint_dir.glob("*.tmp*"))
    if checkpoint_payloads != [checkpoint_path] or checkpoint_temporaries:
        raise RuntimeError(
            "GeoRoute final-only policy requires exactly one complete checkpoint: "
            f"payloads={checkpoint_payloads}, temporaries={checkpoint_temporaries}"
        )
    test_torchrun, test_rendezvous = build_torchrun_prefix(
        phase="test",
        slurm_job_id=slurm_job_id,
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
    )
    _run_logged(
        [
            *test_torchrun,
            "tools/test.py",
            str(bound_config),
            "--checkpoint",
            str(checkpoint_path),
            "--seed",
            str(args.seed),
            "--id",
            "0",
        ],
        log_path=test_log,
        env=inherited,
    )
    prediction_path = effective_work_dir / "result_detection.json"
    profile_path = effective_work_dir / "georoute_development_profile.json"
    for artifact in (prediction_path, profile_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"GeoRoute evaluation artifact is missing: {artifact}")
    result = build_stage_result(
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
        token_budget=args.token_budget,
        binding=cfg.georoute_runtime_binding,
        config_path=bound_config,
        checkpoint_path=checkpoint_path,
        storage_receipt_path=storage_receipt_path,
        prediction_path=prediction_path,
        profile_path=profile_path,
        test_log_path=test_log,
        runtime_commit=args.expected_commit.lower(),
        rendezvous={
            "train": train_rendezvous,
            "test": test_rendezvous,
        },
    )
    _atomic_write_json(work_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
