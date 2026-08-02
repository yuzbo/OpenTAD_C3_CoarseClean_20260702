from __future__ import annotations

import torch

from opentad.models.backbones.georoute_wrapper import GeoRouteBackboneWrapper


def test_geometry_shift127_is_a_pure_temporal_permutation_with_gradient():
    geometry = torch.arange(2 * 384 * 4, dtype=torch.float32).reshape(2, 384, 4)
    geometry.requires_grad_()
    shifted = GeoRouteBackboneWrapper._geometry_shift_control(
        geometry,
        shift_tubelets=127,
    )

    timeline = (torch.arange(384) + 127).remainder(384)
    assert torch.equal(shifted, geometry.index_select(1, timeline))
    assert torch.equal(
        torch.sort(shifted.detach().reshape(-1)).values,
        torch.sort(geometry.detach().reshape(-1)).values,
    )
    weights = torch.arange(384, dtype=torch.float32).view(1, 384, 1)
    (shifted * weights).sum().backward()
    assert geometry.grad is not None
    assert torch.isfinite(geometry.grad).all()
    assert torch.count_nonzero(geometry.grad) > 0


def test_geometry_shift_rejects_identity_or_out_of_range_controls():
    geometry = torch.zeros(1, 384, 4)
    for shift in (0, 384, -1):
        try:
            GeoRouteBackboneWrapper._geometry_shift_control(
                geometry,
                shift_tubelets=shift,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"shift={shift} should fail closed")
