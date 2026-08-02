from __future__ import annotations

import pytest
import torch

from opentad.models.backbones.georoute_routing import (
    select_fixed_quota_structured_exact_k,
)


def _draw(device: torch.device, *, update: int):
    logits = torch.arange(60, device=device, dtype=torch.float32).reshape(1, 2, 30)
    return select_fixed_quota_structured_exact_k(
        roi_logits=logits.sin(),
        residual_logits=logits.cos(),
        mode="structured_hybrid",
        context_tokens=4,
        roi_tokens=6,
        residual_tokens=6,
        training=True,
        estimator="score_function",
        temperature=0.7,
        valid_mask=torch.ones_like(logits, dtype=torch.bool),
        study_seed=5227,
        successful_update_index=update,
        distributed_rank=0,
    )


def test_route_private_cpu_generator_does_not_advance_global_rng():
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    first = _draw(torch.device("cpu"), update=17)
    after = torch.get_rng_state().clone()
    second = _draw(torch.device("cpu"), update=17)
    changed = _draw(torch.device("cpu"), update=18)

    assert torch.equal(before, after)
    assert torch.equal(first["ordered_indices"], second["ordered_indices"])
    assert first["route_rng"] == second["route_rng"]
    assert first["route_rng"]["role_seeds"]["roi"] != first["route_rng"]["role_seeds"]["residual"]
    assert not torch.equal(first["ordered_indices"], changed["ordered_indices"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA route RNG test")
def test_route_private_cuda_generator_does_not_advance_global_cuda_rng():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(303)
    before = torch.cuda.get_rng_state(device).clone()
    first = _draw(device, update=22)
    after = torch.cuda.get_rng_state(device).clone()
    replay = _draw(device, update=22)

    assert torch.equal(before, after)
    assert torch.equal(first["ordered_indices"], replay["ordered_indices"])
    assert torch.equal(first["ordered_log_prob"], replay["ordered_log_prob"])
