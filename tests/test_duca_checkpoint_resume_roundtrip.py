import random

import numpy as np
import torch

from opentad.utils.checkpoint import save_checkpoint
from tools.bata.duca_p0_training import capture_global_rng_state, restore_global_rng_state


def test_checkpoint_round_trip_preserves_optimizer_update_count_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    snapshot = capture_global_rng_state()
    path = save_checkpoint(
        model, None, optimizer, scheduler, 4, work_dir=str(tmp_path),
        rng_state=snapshot, successful_optimizer_updates=7,
    )
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["successful_optimizer_updates"] == 7
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2)
    restored_model.load_state_dict(checkpoint["state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_scheduler.load_state_dict(checkpoint["scheduler"])
    restore_global_rng_state(checkpoint["rng_state"])
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert restored_scheduler.last_epoch == scheduler.last_epoch

