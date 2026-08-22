import hashlib
import json
import os
import re
import tempfile
import torch


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path):
    with open(path, "r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value, destination):
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".tmp.",
        dir=directory,
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        _fsync_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _prune_recovery_checkpoints(save_dir, keep_latest):
    """Retain only the newest bounded ordinary recovery checkpoints."""

    keep_latest = int(keep_latest)
    if keep_latest < 1:
        raise ValueError("recovery checkpoint retention must be positive")
    candidates = []
    pattern = re.compile(r"^recovery_epoch_(\d+)\.pth$")
    for name in os.listdir(save_dir):
        match = pattern.fullmatch(name)
        if match is not None:
            candidates.append((int(match.group(1)), os.path.join(save_dir, name)))
    for _epoch, checkpoint_path in sorted(candidates)[:-keep_latest]:
        sidecar_path = checkpoint_path + ".metadata.json"
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    _fsync_directory(save_dir)


def save_checkpoint(
    model,
    model_ema,
    optimizer,
    scheduler,
    epoch,
    work_dir=None,
    experiment_metadata=None,
    experiment_sidecar_schema=None,
    scaler=None,
    training_state=None,
    checkpoint_role=None,
    recovery_keep_latest=None,
):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {
        "epoch": epoch,
        "checkpoint_role": checkpoint_role or "final",
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})
    if scaler is not None:
        save_states.update({"scaler": scaler.state_dict()})
    if training_state is not None:
        if not isinstance(training_state, dict):
            raise ValueError("training_state must be a mapping")
        save_states.update({"training_state": dict(training_state)})
    if experiment_metadata is not None:
        save_states.update({"experiment_metadata": dict(experiment_metadata)})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if checkpoint_role is None or checkpoint_role == "final":
        checkpoint_name = f"epoch_{epoch}.pth"
    elif checkpoint_role == "recovery":
        checkpoint_name = f"recovery_epoch_{epoch}.pth"
    elif checkpoint_role == "milestone":
        checkpoint_name = f"milestone_epoch_{epoch}.pth"
    else:
        raise ValueError(f"unsupported checkpoint role {checkpoint_role!r}")
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    r1_final = bool(
        checkpoint_role == "final"
        and isinstance(training_state, dict)
        and training_state.get("arm_surface") == "R1"
    )
    final_ema_path = os.path.join(save_dir, "final_ema.pth")
    final_raw_path = os.path.join(save_dir, "final_raw.pth")
    if r1_final:
        if model_ema is None or "state_dict_ema" not in save_states:
            raise ValueError("ZoomToken R1 final requires the frozen EMA state")
        existing = [
            path
            for path in (checkpoint_path, final_ema_path, final_raw_path)
            if os.path.exists(path)
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite immutable R1 final artifacts: {existing}"
            )
    if experiment_metadata is not None:
        if (
            not isinstance(experiment_sidecar_schema, str)
            or not experiment_sidecar_schema
        ):
            raise ValueError(
                "experiment_sidecar_schema is required when experiment_metadata is saved"
            )
        sidecar_path = checkpoint_path + ".metadata.json"
        checkpoint_tmp = checkpoint_path + ".tmp"
        sidecar_tmp = sidecar_path + ".tmp"
        existing = [
            path
            for path in (checkpoint_path, sidecar_path, checkpoint_tmp, sidecar_tmp)
            if os.path.exists(path)
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite frozen S1 checkpoint artifacts: {existing}"
            )
        sidecar_published = False
        try:
            torch.save(save_states, checkpoint_tmp)
            _fsync_file(checkpoint_tmp)
            sidecar = {
                "schema_version": experiment_sidecar_schema,
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_tmp),
                "experiment_metadata": dict(experiment_metadata),
            }
            sidecar["sidecar_sha256"] = _canonical_sha256(sidecar)
            with open(sidecar_tmp, "x", encoding="utf-8") as handle:
                json.dump(sidecar, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(sidecar_tmp, sidecar_path)
            sidecar_published = True
            # The checkpoint path is the commit marker. Consumers can never
            # observe a final checkpoint before its verified sidecar exists.
            os.replace(checkpoint_tmp, checkpoint_path)
            _fsync_directory(save_dir)
        except Exception:
            if sidecar_published and not os.path.exists(checkpoint_path):
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
            raise
        finally:
            for path in (checkpoint_tmp, sidecar_tmp):
                if os.path.exists(path):
                    os.remove(path)
    else:
        _atomic_torch_save(save_states, checkpoint_path)
    if r1_final:
        lineage = dict(training_state)
        _atomic_torch_save(
            {
                "epoch": epoch,
                "checkpoint_role": "final_ema",
                "state_dict_ema": save_states["state_dict_ema"],
                "training_state": lineage,
            },
            final_ema_path,
        )
        _atomic_torch_save(
            {
                "epoch": epoch,
                "checkpoint_role": "final_raw",
                "state_dict": save_states["state_dict"],
                "training_state": lineage,
            },
            final_raw_path,
        )
    if recovery_keep_latest is not None:
        _prune_recovery_checkpoints(save_dir, recovery_keep_latest)
    return checkpoint_path


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"best.pth")
    _atomic_torch_save(save_states, checkpoint_path)
