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

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})
    if experiment_metadata is not None:
        save_states.update({"experiment_metadata": dict(experiment_metadata)})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
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
        checkpoint_published = False
        try:
            torch.save(save_states, checkpoint_tmp)
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
            os.replace(checkpoint_tmp, checkpoint_path)
            checkpoint_published = True
            os.replace(sidecar_tmp, sidecar_path)
        except Exception:
            if checkpoint_published and not os.path.exists(sidecar_path):
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
            raise
        finally:
            for path in (checkpoint_tmp, sidecar_tmp):
                if os.path.exists(path):
                    os.remove(path)
    else:
        torch.save(save_states, checkpoint_path)


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"best.pth")
    torch.save(save_states, checkpoint_path)
