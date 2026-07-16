import torch
from pathlib import Path


def _atomic_torch_save(save_states, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(save_states, tmp_path)
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"checkpoint write produced an empty file: {tmp_path}")
    tmp_path.replace(checkpoint_path)


def save_checkpoint(
    model,
    model_ema,
    optimizer,
    scheduler,
    epoch,
    work_dir=None,
    include_optimizer=True,
    include_scheduler=True,
    include_ema=True,
):
    save_dir = Path(work_dir) / "checkpoint"

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }

    if include_optimizer:
        save_states["optimizer"] = optimizer.state_dict()
    if include_scheduler:
        save_states["scheduler"] = scheduler.state_dict()
    if include_ema and model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"epoch_{epoch}.pth"
    _atomic_torch_save(save_states, checkpoint_path)


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = Path(work_dir) / "checkpoint"

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "best.pth"
    _atomic_torch_save(save_states, checkpoint_path)
