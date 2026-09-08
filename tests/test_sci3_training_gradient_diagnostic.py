"""Checkpoint selection and an ordinary backward on real small source modules."""
import copy
import random

import numpy as np
import pytest
import torch

from geosparse_research.training_gradients import measure_training_gradients
from geosparse_research.training_gradient_diagnostic import restore_probe_state, ordinary_gradient_norm, normalizer_restore_record
from test_geosparse_detector import batch, detector


def saved_state(model):
    state = copy.deepcopy(model.state_dict())
    ema = {name: value + 1 for name, value in state.items()}
    return dict(format="geosparse_full_state_v2", epoch=19, completed_epochs=20,
                state_dict=state, state_dict_ema=ema, minibatch=2000,
                grad_scaler={"scale": 128.},
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(), cuda=[]))


def test_restores_ordinary_weights_normalizer_rng_and_next_epoch():
    model = detector("B", query_length=16)
    model.rpn_head.loss_normalizer.fill_(137)
    saved = saved_state(model)
    expected = random.random(), np.random.rand(), torch.rand(1)
    with torch.no_grad():
        next(model.parameters()).add_(9)
    model.rpn_head.loss_normalizer.fill_(3)
    context = restore_probe_state(model, saved, 20, amp=True)
    assert context["loss_scale"] == 128. and context["weights"] == "ordinary_state_dict"
    assert model.epoch == 20 and model.minibatch == 2000
    assert float(model.rpn_head.loss_normalizer) == 137.
    assert context["normalizer_restore"]["exact"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["state_dict"][name], rtol=0, atol=0)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)


def test_receipt_exposes_fractional_buffer_conversion_without_changing_it():
    saved = torch.tensor(58.29771423339844)
    restored = torch.tensor(100)
    restored.copy_(saved)
    record = normalizer_restore_record(saved, restored)
    assert record == dict(checkpoint_value=float(saved), checkpoint_dtype="torch.float32",
                          restored_value=58., restored_dtype="torch.int64", exact=False)
    assert float(restored) == 58. and restored.dtype == torch.int64


@pytest.mark.parametrize("change", ["wrong_epoch", "terminal", "missing_scale", "wrong_completed"])
def test_wrong_checkpoint_or_amp_scale_cannot_be_used(change):
    model = detector("A", query_length=16)
    saved = saved_state(model)
    completed = 20
    if change == "wrong_epoch":
        saved["epoch"] = 18
    elif change == "terminal":
        saved.update(epoch=59, completed_epochs=60)
        completed = 60
    elif change == "missing_scale":
        saved["grad_scaler"] = {}
    else:
        saved["completed_epochs"] = 19
    with pytest.raises(ValueError):
        restore_probe_state(model, saved, completed, amp=True)


def test_ordinary_reference_matches_decomposition_and_preserves_state():
    torch.manual_seed(5)
    model = detector("B", query_length=16)
    model.epoch = 6
    inputs = batch(empty=True)
    before = copy.deepcopy(model.state_dict())
    measured = measure_training_gradients(model, inputs, loss_scale=128.)
    direct = ordinary_gradient_norm(model, inputs, 1, False, 128.)
    assert direct == pytest.approx(measured["gradients"]["total"]["norm"], rel=1e-5)
    assert model.minibatch == 0 and all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
