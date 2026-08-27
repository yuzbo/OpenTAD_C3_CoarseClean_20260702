"""F/N/Q/D arm-identifiability probe for the dynamic SCNR route.

This is a CPU synthetic routing probe, not a performance or efficacy test.  It
answers one narrow mechanism question for the accepted F/N/Q causal matrix:

    does the continuous ROI modifier *change which tokens are selected* and
    *concentrate selection toward the decoded ROI*, at a matched global exact-B
    budget, beyond both the no-ROI arm (N) and the content-only free-token arm
    (Q)?

If the ROI modifier were degenerate (for example always dominated by the
residual modifier, the historical role-collapse failure mode), then F would
select the same tokens as N and the paper's primary causal contrast F-N would
be vacuous.  A passing run therefore rejects that degeneracy for a synthetic,
mechanically representative field; it does NOT measure mAP, cost, energy, or
any paper-level claim.

No dataset, checkpoint, ground truth, teacher, oracle, GPU, or official-test
path is touched.  The module under test is the existing routing primitive
``select_dynamic_global_exact_budget`` (and the ROI/residual field helpers),
plus the semantics of the two new wrapper switches that zero the ROI/residual
modifier for the N and Q arms.
"""

from __future__ import annotations

import torch

from opentad.models.backbones.georoute_routing import (
    calibrate_dynamic_residual_modifier,
    roi_modifier_from_geometry,
    select_dynamic_global_exact_budget,
)


def _build_synthetic_fields(
    *,
    tubelets: int,
    grid_h: int,
    grid_w: int,
    roi_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (q_base, delta_roi, delta_residual_raw, valid_mask) for one window.

    The ROI is a fixed central ellipse with full extent (w=0.4, h=0.4) whose
    modifier magnitude is comparable to (not swamped by) the residual field, so
    the probe exercises the post-centering regime where ROI is identifiable.
    """

    batch, item_count = 1, grid_h * grid_w
    valid_mask = torch.ones((batch, tubelets, item_count), dtype=torch.bool)

    # Content-only base utility: a deterministic, content-shaped field with a
    # weak spatial gradient and temporal modulation.  It must be non-trivial so
    # the Q arm is a real free-token selection, not a uniform scaffold.
    t = torch.arange(tubelets, dtype=torch.float32).view(1, tubelets, 1)
    s = torch.arange(item_count, dtype=torch.float32).view(1, 1, item_count)
    q_base = 0.5 * torch.sin(0.1 * t) + 0.05 * torch.sin(0.3 * s)

    # Continuous ROI geometry: central ellipse, full extents w=0.4, h=0.4.
    geometry = torch.zeros((batch, tubelets, 4), dtype=torch.float32)
    geometry[..., 0] = 0.5  # cx
    geometry[..., 1] = 0.5  # cy
    geometry[..., 2] = 0.4  # w
    geometry[..., 3] = 0.4  # h
    delta_roi = roi_modifier_from_geometry(
        geometry,
        grid_height=grid_h,
        grid_width=grid_w,
        temperature=roi_temperature,
    )

    # Residual saliency: a deterministic field with scale deliberately below the
    # ROI modifier scale so residual does not trivially dominate the argmax.
    n = torch.arange(item_count, dtype=torch.float32).view(1, 1, item_count)
    delta_residual_raw = 0.15 * torch.sin(0.5 * t + 0.7 * n)

    return q_base, delta_roi, delta_residual_raw, valid_mask


def _route_arm(
    q_base: torch.Tensor,
    delta_roi: torch.Tensor,
    delta_residual_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    budget: int,
    roi_enabled: bool,
    residual_enabled: bool,
) -> dict:
    """Run one arm with the same field assembly as ``_forward_dynamic_scnr``."""

    effective_roi = delta_roi if roi_enabled else torch.zeros_like(delta_roi)
    raw_residual = delta_residual_raw if residual_enabled else torch.zeros_like(
        delta_residual_raw
    )
    delta_residual, _mean = calibrate_dynamic_residual_modifier(
        raw_residual,
        valid_mask=valid_mask,
        mode="residual_window_center",
    )
    return select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=effective_roi,
        delta_residual=delta_residual,
        window_budget=budget,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=valid_mask,
    )


def _inside_roi_fraction(route: dict, delta_roi: torch.Tensor) -> float:
    """Fraction of selected physical tokens whose patch centre is inside the ROI.

    ``roi_modifier_from_geometry`` is positive exactly inside the decoded
    ellipse, so ``delta_roi > 0`` is the membership test.
    """

    selected = route["selected_mask"].bool()
    inside = (delta_roi > 0.0) & selected
    return float(inside.sum().item()) / float(selected.sum().item())


def test_fnmqd_arm_identifiability() -> None:
    torch.manual_seed(0)
    tubelets, grid_h, grid_w = 64, 11, 20
    budget = 4096  # 0.29 of the 14,080 valid physical tokens, mirroring B=24576/84480

    q_base, delta_roi, delta_residual_raw, valid_mask = _build_synthetic_fields(
        tubelets=tubelets,
        grid_h=grid_h,
        grid_w=grid_w,
        roi_temperature=0.25,
    )

    f_route = _route_arm(
        q_base,
        delta_roi,
        delta_residual_raw,
        valid_mask,
        budget=budget,
        roi_enabled=True,
        residual_enabled=True,
    )
    n_route = _route_arm(
        q_base,
        delta_roi,
        delta_residual_raw,
        valid_mask,
        budget=budget,
        roi_enabled=False,
        residual_enabled=True,
    )
    q_route = _route_arm(
        q_base,
        delta_roi,
        delta_residual_raw,
        valid_mask,
        budget=budget,
        roi_enabled=False,
        residual_enabled=False,
    )

    # 1. All three sparse arms keep the global exact-B contract.
    for name, route in (("F", f_route), ("N", n_route), ("Q", q_route)):
        selected_count = int(route["selected_mask"].sum().item())
        assert selected_count == budget, (
            f"{name} selected {selected_count} tokens, expected exact B={budget}"
        )
        assert route["padded_token_count"] == 0, f"{name} reported padding"

    f_indices = f_route["physical_indices"]
    n_indices = n_route["physical_indices"]
    q_indices = q_route["physical_indices"]

    # 2. ROI modifier must actually change the selected set vs the no-ROI arm.
    assert not torch.equal(f_indices, n_indices), (
        "F and N selected identical tokens: the ROI modifier is degenerate and "
        "the F-N causal contrast would be vacuous."
    )

    # 3. Content-only Q must differ from both modifier-bearing arms.
    assert not torch.equal(q_indices, n_indices), (
        "Q and N selected identical tokens: the residual modifier has no effect."
    )

    # 4. ROI concentration: F should put a larger fraction of its budget inside
    #    the decoded ellipse than either N or Q, otherwise the ROI modifier is
    #    merely re-ranking tokens without spatially concentrating compute.
    f_inside = _inside_roi_fraction(f_route, delta_roi)
    n_inside = _inside_roi_fraction(n_route, delta_roi)
    q_inside = _inside_roi_fraction(q_route, delta_roi)
    assert f_inside > n_inside, (
        f"F inside-ROI fraction {f_inside:.4f} did not exceed N {n_inside:.4f}"
    )
    assert f_inside > q_inside, (
        f"F inside-ROI fraction {f_inside:.4f} did not exceed Q {q_inside:.4f}"
    )

    # 5. Role accounting stays internally consistent for F.
    assert f_route["role_counts"]["roi"] > 0, (
        "F selected no ROI-role tokens: ROI modifier is not operational"
    )
    assert (
        f_route["role_counts"]["context"]
        + f_route["role_counts"]["roi"]
        + f_route["role_counts"]["residual"]
        == budget
    )
