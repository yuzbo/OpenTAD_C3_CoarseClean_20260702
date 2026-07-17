import os
from types import SimpleNamespace

import pytest


def _torch_or_skip():
    if os.name == "nt":
        pytest.skip("PyTorch checkpoint tests run in the remote Linux environment")
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyTorch runtime is unavailable: {exc}")
    return torch


def test_matched_medium_checkpoint_requires_replayable_ema(tmp_path):
    torch = _torch_or_skip()
    from opentad.utils.checkpoint import save_checkpoint
    from tools.bata.validate_phystime_g1_matched_medium_artifacts import (
        _validate_lightweight_checkpoint,
    )

    model = torch.nn.Linear(3, 2)
    model_ema = SimpleNamespace(module=torch.nn.Linear(3, 2))
    model_ema.module.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    save_checkpoint(
        model,
        model_ema,
        optimizer,
        scheduler,
        epoch=19,
        work_dir=tmp_path,
        include_optimizer=False,
        include_scheduler=False,
        include_ema=True,
    )
    checkpoint_path = tmp_path / "checkpoint" / "epoch_19.pth"
    contract = _validate_lightweight_checkpoint(checkpoint_path, final_epoch=19)
    assert contract["evaluated_weights_replayable"] is True
    assert contract["state_dict_entries"] == contract["state_dict_ema_entries"]

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("state_dict_ema")
    missing_ema_path = tmp_path / "missing_ema.pth"
    torch.save(checkpoint, missing_ema_path)
    with pytest.raises(RuntimeError, match="state_dict_ema is empty"):
        _validate_lightweight_checkpoint(missing_ema_path, final_epoch=19)


def test_matched_medium_checkpoint_rejects_non_finite_ema(tmp_path):
    torch = _torch_or_skip()
    from tools.bata.validate_phystime_g1_matched_medium_artifacts import (
        _validate_lightweight_checkpoint,
    )

    checkpoint_path = tmp_path / "non_finite_ema.pth"
    torch.save(
        {
            "epoch": 19,
            "state_dict": {"weight": torch.ones(1)},
            "state_dict_ema": {"weight": torch.tensor([float("nan")])},
        },
        checkpoint_path,
    )
    with pytest.raises(RuntimeError, match="non-finite tensor"):
        _validate_lightweight_checkpoint(checkpoint_path, final_epoch=19)
