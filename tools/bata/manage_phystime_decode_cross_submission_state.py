import argparse
import json
import os
import time
from pathlib import Path


SCHEMA_VERSION = "phystime_decode_cross_submission_attempt_v1"


def _atomic_write_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _read_attempt(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"submission attempt schema mismatch: {path}")
    return payload


def record_attempt(
    *,
    output_path,
    run_root,
    dag_token,
    variant,
    comment,
    runtime_commit,
    runtime_tree,
    phase,
    expected_job_id="",
    sbatch_output="",
):
    output_path = Path(output_path).resolve()
    existing = _read_attempt(output_path) if output_path.is_file() else None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(Path(run_root).resolve()),
        "dag_token": dag_token,
        "variant": variant,
        "comment": comment,
        "runtime_commit": runtime_commit,
        "runtime_tree": runtime_tree,
    }
    if existing is not None:
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"submission attempt mismatch for {key}: "
                    f"{existing.get(key)!r} != {value!r}"
                )
    payload = {
        **expected,
        "state": "ambiguous",
        "phase": phase,
        "expected_job_id": str(expected_job_id),
        "sbatch_output": str(sbatch_output)[:4096],
        "created_at_unix": (
            existing.get("created_at_unix", time.time())
            if existing is not None
            else time.time()
        ),
        "updated_at_unix": time.time(),
    }
    _atomic_write_json(output_path, payload)
    return payload


def resolve_attempt(*, ambiguous_path, resolved_path, job_id):
    ambiguous_path = Path(ambiguous_path).resolve()
    resolved_path = Path(resolved_path).resolve()
    payload = _read_attempt(ambiguous_path)
    expected_job_id = payload.get("expected_job_id", "")
    if expected_job_id and expected_job_id != str(job_id):
        raise RuntimeError(
            "visible job differs from the accepted sbatch job: "
            f"{job_id} != {expected_job_id}"
        )
    payload.update(
        {
            "state": "resolved",
            "resolved_job_id": str(job_id),
            "resolved_at_unix": time.time(),
        }
    )
    _atomic_write_json(resolved_path, payload)
    ambiguous_path.unlink()
    if os.name != "nt":
        directory_fd = os.open(ambiguous_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return payload


def abort_attempt(*, ambiguous_path, fatal_path, reason):
    ambiguous_path = Path(ambiguous_path).resolve()
    fatal_path = Path(fatal_path).resolve()
    payload = _read_attempt(ambiguous_path)
    payload.update(
        {
            "state": "fatal",
            "fatal_reason": str(reason),
            "fatal_at_unix": time.time(),
        }
    )
    _atomic_write_json(fatal_path, payload)
    ambiguous_path.unlink()
    if os.name != "nt":
        directory_fd = os.open(ambiguous_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return payload


def inspect_resolved_attempt(
    *,
    resolved_path,
    run_root,
    dag_token,
    variant,
    comment,
    runtime_commit,
    runtime_tree,
):
    payload = _read_attempt(resolved_path)
    expected = {
        "run_root": str(Path(run_root).resolve()),
        "dag_token": dag_token,
        "variant": variant,
        "comment": comment,
        "runtime_commit": runtime_commit,
        "runtime_tree": runtime_tree,
        "state": "resolved",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"resolved submission mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    job_id = str(payload.get("resolved_job_id", ""))
    if not job_id.isdigit():
        raise RuntimeError("resolved submission has no numeric job ID")
    return job_id


def fatalize_attempt(*, source_path, fatal_path, reason):
    source_path = Path(source_path).resolve()
    fatal_path = Path(fatal_path).resolve()
    payload = _read_attempt(source_path)
    payload.update(
        {
            "state": "fatal",
            "fatal_reason": str(reason),
            "fatal_at_unix": time.time(),
        }
    )
    _atomic_write_json(fatal_path, payload)
    source_path.unlink()
    if os.name != "nt":
        directory_fd = os.open(source_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--output", required=True)
    record.add_argument("--run-root", required=True)
    record.add_argument("--dag-token", required=True)
    record.add_argument("--variant", required=True)
    record.add_argument("--comment", required=True)
    record.add_argument("--runtime-commit", required=True)
    record.add_argument("--runtime-tree", required=True)
    record.add_argument("--phase", required=True)
    record.add_argument("--expected-job-id", default="")
    record.add_argument("--sbatch-output", default="")

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--ambiguous", required=True)
    resolve.add_argument("--resolved", required=True)
    resolve.add_argument("--job-id", required=True)

    abort = subparsers.add_parser("abort")
    abort.add_argument("--ambiguous", required=True)
    abort.add_argument("--fatal", required=True)
    abort.add_argument("--reason", required=True)

    inspect = subparsers.add_parser("inspect-resolved")
    inspect.add_argument("--resolved", required=True)
    inspect.add_argument("--run-root", required=True)
    inspect.add_argument("--dag-token", required=True)
    inspect.add_argument("--variant", required=True)
    inspect.add_argument("--comment", required=True)
    inspect.add_argument("--runtime-commit", required=True)
    inspect.add_argument("--runtime-tree", required=True)

    fatalize = subparsers.add_parser("fatalize")
    fatalize.add_argument("--source", required=True)
    fatalize.add_argument("--fatal", required=True)
    fatalize.add_argument("--reason", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "record":
        record_attempt(
            output_path=args.output,
            run_root=args.run_root,
            dag_token=args.dag_token,
            variant=args.variant,
            comment=args.comment,
            runtime_commit=args.runtime_commit,
            runtime_tree=args.runtime_tree,
            phase=args.phase,
            expected_job_id=args.expected_job_id,
            sbatch_output=args.sbatch_output,
        )
    elif args.command == "resolve":
        resolve_attempt(
            ambiguous_path=args.ambiguous,
            resolved_path=args.resolved,
            job_id=args.job_id,
        )
    elif args.command == "abort":
        abort_attempt(
            ambiguous_path=args.ambiguous,
            fatal_path=args.fatal,
            reason=args.reason,
        )
    elif args.command == "inspect-resolved":
        print(
            inspect_resolved_attempt(
                resolved_path=args.resolved,
                run_root=args.run_root,
                dag_token=args.dag_token,
                variant=args.variant,
                comment=args.comment,
                runtime_commit=args.runtime_commit,
                runtime_tree=args.runtime_tree,
            )
        )
    else:
        fatalize_attempt(
            source_path=args.source,
            fatal_path=args.fatal,
            reason=args.reason,
        )


if __name__ == "__main__":
    main()
