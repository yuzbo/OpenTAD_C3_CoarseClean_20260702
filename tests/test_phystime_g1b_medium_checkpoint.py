from pathlib import Path

import torch

from opentad.utils.checkpoint import save_checkpoint


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_lightweight_checkpoint_is_atomic_and_excludes_training_state(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    save_checkpoint(
        model,
        None,
        optimizer,
        scheduler,
        19,
        work_dir=tmp_path,
        include_optimizer=False,
        include_scheduler=False,
        include_ema=False,
    )

    checkpoint_path = tmp_path / "checkpoint" / "epoch_19.pth"
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0
    assert not Path(str(checkpoint_path) + ".tmp").exists()
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["epoch"] == 19
    assert "state_dict" in payload
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert "state_dict_ema" not in payload


def test_g1b_sdpq_script_supports_medium_run_and_dynamic_final_epoch():
    script = Path("scripts/run_phystime_g1b_sdpq_pilot_slurm.sh").read_text(encoding="utf-8")
    submit = Path("scripts/submit_phystime_g1b_sdpq_pilot.sh").read_text(encoding="utf-8")

    assert "PHYSTIME_G1B_PILOT_EPOCHS:-20" in script
    assert "PHYSTIME_G1B_PILOT_EPOCHS:-20" in submit
    assert "fixed to six epochs" not in script
    assert "workflow.checkpoint_save_mode=lightweight" in script
    assert "workflow.checkpoint_include_ema=False" in script
    assert "final_epoch = pilot_epochs - 1" in script
    assert 'epoch_5.pth' not in script
