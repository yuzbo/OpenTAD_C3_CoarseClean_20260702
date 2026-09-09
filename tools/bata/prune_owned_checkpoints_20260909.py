"""One-time user-authorized cleanup of the stopped six-route checkpoint roots."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import re
import subprocess


EPOCH_FILE = re.compile(r"epoch_(\d+)\.pth$")
SKIP_DIRS = {".git", "data", "datasets", "raw", "pretrained", "pretrain", "__pycache__"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_record(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev, allocated_bytes=stat.st_blocks * 512)


def checked_path(raw, roots):
    path = Path(raw)
    if path.resolve() != path or not any(root in path.parents for root in roots):
        raise RuntimeError(f"path outside exact owned roots or through a link: {path}")
    if not (path.parent.name == "checkpoint" and EPOCH_FILE.fullmatch(path.name)
            or path.parent.name == "test_guided" and path.name == "best_test.pth"):
        raise RuntimeError(f"not an approved checkpoint filename: {path}")
    return path


def validate_weights(path, expected_epoch):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise RuntimeError("checkpoint epoch does not match the saved snapshot")
    states = {}
    for key in ("state_dict", "state_dict_ema"):
        state = checkpoint.get(key)
        if state is None:
            continue
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"empty {key}")
        tensors = 0
        elements = 0
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            tensors += 1
            elements += tensor.numel()
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"non-finite {key}.{name}")
        if not tensors:
            raise RuntimeError(f"{key} has no tensors")
        states[key] = dict(tensors=tensors, elements=elements)
    if not states:
        raise RuntimeError("no model or EMA parameters")
    result = dict(epoch=expected_epoch, states=states,
                  optimizer_present="optimizer" in checkpoint,
                  sidecar_present=Path(str(path) + ".metadata.json").is_file(),
                  validation="CPU torch.load and all model/EMA tensors finite; no GPU or performance evaluation")
    del checkpoint
    gc.collect()
    return result


def make_plan(scope, roots):
    groups = {}
    for root in roots:
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not Path(folder, name).is_symlink()]
            parent = Path(folder)
            for name in names:
                match = EPOCH_FILE.fullmatch(name)
                if parent.name == "checkpoint" and match:
                    epoch = int(match[1])
                elif parent.name == "test_guided" and name == "best_test.pth":
                    epoch = int(json.loads((parent / "best_test_metrics.json").read_text())["epoch"])
                else:
                    continue
                path = checked_path(str(parent / name), roots)
                groups.setdefault(str(parent.parent), []).append(dict(file_record(path), epoch=epoch))
    runs = []
    for run, files in sorted(groups.items()):
        candidates = sorted(files, key=lambda x: (x["epoch"], Path(x["path"]).parent.name == "checkpoint"), reverse=True)
        invalid = []
        keep = None
        for candidate in candidates:
            try:
                validation = validate_weights(Path(candidate["path"]), candidate["epoch"])
            except Exception as exc:
                invalid.append(dict(path=candidate["path"], error=f"{type(exc).__name__}: {exc}"))
                continue
            keep = dict(candidate, validation=validation)
            break
        removed = [x for x in files if keep and x["path"] != keep["path"]]
        record = dict(run=run, kept=keep, invalid_candidates=invalid, remove=removed,
                      status="VALIDATED" if keep else "BLOCKED_NO_USABLE_CHECKPOINT",
                      original_file_count=len(files))
        runs.append(record)
        print(json.dumps(dict(event="validated", run=run, kept=keep and keep["path"],
                              delete_count=len(removed), invalid=invalid)), flush=True)
    return dict(schema_version="DUCA-CHECKPOINT-PRUNE-v001", planned_at_utc=now(),
                scope=scope, status="VALIDATED_PLAN", runs=runs,
                metadata_policy="Keep all JSON sidecars, logs, configs, metrics and receipts unchanged")


def assert_not_running(roots):
    ids = subprocess.check_output(["squeue", "-r", "-h", "-u", "sczc063", "-o", "%i"], text=True).split()
    for job in ids:
        metadata = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
        if any(str(root) + "/" in metadata for root in roots):
            raise RuntimeError(f"owned checkpoint namespace still used by job {job}")
    return ids


def apply_plan(plan, roots):
    if plan["status"] != "VALIDATED_PLAN":
        raise RuntimeError("expected an unapplied validated plan")
    others = assert_not_running(roots)
    keep_paths = {r["kept"]["path"] for r in plan["runs"] if r["kept"]}
    for run in plan["runs"]:
        for item in ([run["kept"]] if run["kept"] else []) + run["remove"]:
            path = checked_path(item["path"], roots)
            current = file_record(path)
            if any(current[key] != item[key] for key in ("bytes", "mtime_ns", "inode", "device")):
                raise RuntimeError(f"checkpoint changed since validation: {path}")
        if any(item["path"] in keep_paths for item in run["remove"]):
            raise RuntimeError("plan attempts to delete a retained checkpoint")
    deleted = []
    for run in plan["runs"]:
        for item in run["remove"]:
            Path(item["path"]).unlink()
            deleted.append(item)
            print(json.dumps(dict(event="deleted", path=item["path"], bytes=item["bytes"])), flush=True)
    for run in plan["runs"]:
        if run["kept"]:
            assert Path(run["kept"]["path"]).is_file()
        assert all(not Path(item["path"]).exists() for item in run["remove"])
    plan.update(status="APPLIED", applied_at_utc=now(),
                deleted_files=len(deleted), deleted_bytes=sum(x["bytes"] for x in deleted),
                deleted_allocated_bytes=sum(x["allocated_bytes"] for x in deleted),
                kept_files=len(keep_paths), other_job_ids_not_touched=others)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply-plan")
    args = parser.parse_args()
    scope = json.loads(Path(args.scope).read_text())
    base = Path(scope["remote_base"])
    if str(base) != "/data/run01/sczc063/yuzibo" or not base.is_dir():
        raise RuntimeError("this maintenance task is restricted to N16R4 yuzibo storage")
    roots = [Path(raw) for raw in scope["roots"]]
    if any(not root.is_dir() or root.resolve() != root or base not in root.parents for root in roots):
        raise RuntimeError("scope contains an unavailable or unowned root")
    if args.apply_plan:
        plan = json.loads(Path(args.apply_plan).read_text())
        if plan["scope"] != scope:
            raise RuntimeError("scope changed after validation")
        payload = apply_plan(plan, roots)
    else:
        import torch
        torch.set_num_threads(2)
        payload = make_plan(scope, roots)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(event="finished", status=payload["status"], output=args.output,
                          runs=len(payload["runs"]), deleted=payload.get("deleted_files", 0))), flush=True)


if __name__ == "__main__":
    main()
