import hashlib
import json
import os

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


def save_checkpoint(
    model,
    model_ema,
    optimizer,
    scheduler,
    epoch,
    work_dir=None,
    scaler=None,
    rng_state=None,
    data_loader_state=None,
    update_audit_state=None,
    successful_optimizer_updates=None,
    experiment_metadata=None,
    experiment_sidecar_schema=None,
):
    save_dir = os.path.join(work_dir, "checkpoint")
    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if model_ema is not None:
        save_states["state_dict_ema"] = model_ema.module.state_dict()
    if scaler is not None:
        save_states["grad_scaler"] = scaler.state_dict()
    if rng_state is not None:
        save_states["rng_state"] = dict(rng_state)
    if data_loader_state is not None:
        save_states["data_loader_state"] = dict(data_loader_state)
    if update_audit_state is not None:
        save_states["update_audit_state"] = dict(update_audit_state)
    if successful_optimizer_updates is not None:
        save_states["successful_optimizer_updates"] = int(successful_optimizer_updates)
    if experiment_metadata is not None:
        save_states["experiment_metadata"] = dict(experiment_metadata)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    if experiment_metadata is None:
        torch.save(save_states, checkpoint_path)
        return checkpoint_path

    if not isinstance(experiment_sidecar_schema, str) or not experiment_sidecar_schema:
        raise ValueError(
            "experiment_sidecar_schema is required when experiment metadata is saved"
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
            f"refusing to overwrite frozen checkpoint artifacts: {existing}"
        )
    torch.save(save_states, checkpoint_tmp)
    sidecar = {
        "schema_version": experiment_sidecar_schema,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_tmp),
        "experiment_metadata": dict(experiment_metadata),
    }
    sidecar["sidecar_sha256"] = _canonical_sha256(sidecar)
    try:
        with open(sidecar_tmp, "x", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(checkpoint_tmp, checkpoint_path)
        os.replace(sidecar_tmp, sidecar_path)
    finally:
        for path in (checkpoint_tmp, sidecar_tmp):
            if os.path.exists(path):
                os.remove(path)
    return checkpoint_path


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")
    save_states = {"epoch": epoch, "state_dict": model.state_dict()}
    if model_ema is not None:
        save_states["state_dict_ema"] = model_ema.module.state_dict()
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best.pth")
    torch.save(save_states, checkpoint_path)
    return checkpoint_path
