import argparse
import json
import os
import re
import time
from pathlib import Path


SCHEMA_VERSION = "phystime_decode_cross_submission_owner_v1"
RUN_UUID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


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


def _read_owner(path, expected):
    path = Path(path).resolve()
    if not path.is_file():
        return None
    owner = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if owner.get(key) != value:
            raise RuntimeError(
                f"submission owner mismatch at {path} for {key}: "
                f"{owner.get(key)!r} != {value!r}"
            )
    run_uuid = owner.get("run_uuid")
    if not isinstance(run_uuid, str) or RUN_UUID_PATTERN.fullmatch(run_uuid) is None:
        raise RuntimeError(f"submission owner has invalid run_uuid: {path}")
    return owner


def claim_submission_ownership(
    *,
    global_owner_path,
    local_owner_path,
    run_root,
    dag_token,
    runtime_commit,
    runtime_tree,
    run_uuid,
):
    if RUN_UUID_PATTERN.fullmatch(run_uuid) is None:
        raise ValueError("run_uuid must contain 32 lowercase hexadecimal characters")
    global_owner_path = Path(global_owner_path).resolve()
    local_owner_path = Path(local_owner_path).resolve()
    run_root = Path(run_root).resolve()
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "dag_token": dag_token,
        "runtime_commit": runtime_commit,
        "runtime_tree": runtime_tree,
    }
    global_owner = _read_owner(global_owner_path, expected)
    local_owner = _read_owner(local_owner_path, expected)
    existing = [owner for owner in (global_owner, local_owner) if owner is not None]
    existing_run_uuids = {owner["run_uuid"] for owner in existing}
    if len(existing_run_uuids) > 1:
        raise RuntimeError("global and local submission owners disagree on run_uuid")

    canonical_uuid = next(iter(existing_run_uuids), run_uuid)
    created_at_unix = (
        min(float(owner.get("created_at_unix", time.time())) for owner in existing)
        if existing
        else time.time()
    )
    payload = {
        **expected,
        "run_uuid": canonical_uuid,
        "created_at_unix": created_at_unix,
        "global_owner_manifest": str(global_owner_path),
        "local_owner_manifest": str(local_owner_path),
    }
    if global_owner is None:
        _atomic_write_json(global_owner_path, payload)
    if local_owner is None:
        _atomic_write_json(local_owner_path, payload)

    return {
        "recovery_mode": bool(existing),
        "run_uuid": canonical_uuid,
        "global_owner_manifest": str(global_owner_path),
        "local_owner_manifest": str(local_owner_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-owner", required=True)
    parser.add_argument("--local-owner", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dag-token", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--runtime-tree", required=True)
    parser.add_argument("--run-uuid", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = claim_submission_ownership(
        global_owner_path=args.global_owner,
        local_owner_path=args.local_owner,
        run_root=args.run_root,
        dag_token=args.dag_token,
        runtime_commit=args.runtime_commit,
        runtime_tree=args.runtime_tree,
        run_uuid=args.run_uuid,
    )
    print("1" if result["recovery_mode"] else "0")


if __name__ == "__main__":
    main()
