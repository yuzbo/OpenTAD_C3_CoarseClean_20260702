from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ._fixed_budget_autograd import FixedBudgetRateGradient


@dataclass(frozen=True)
class StructuredSelectionOutput:
    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    log_partition: torch.Tensor
    k: int
    max_unselected_hole: int | None
    temperature: float
    selection_scope: str = "full_window_non_streaming"


@dataclass(frozen=True)
class PhysicalExactKSelectionOutput:
    hard_occupancy: torch.Tensor
    hard_slot_assignment: torch.Tensor
    hard_positions: torch.Tensor
    hard_slot_mask: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    log_partition: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor
    temperature: float
    selection_scope: str = "full_window_offline_physical_exact_k"


@dataclass(frozen=True)
class PhysicalExactKGraph:
    predecessor_index: torch.Tensor
    predecessor_valid: torch.Tensor
    successor_index: torch.Tensor
    successor_valid: torch.Tensor
    source_valid: torch.Tensor
    sink_valid: torch.Tensor
    max_gap_seconds: torch.Tensor
    edge_count: int


@dataclass(frozen=True)
class PhysicalExactKHardOutput:
    hard_occupancy: torch.Tensor
    hard_slot_assignment: torch.Tensor
    hard_positions: torch.Tensor
    hard_slot_mask: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor


@dataclass(frozen=True)
class PhysicalExactKSoftOutput:
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    log_partition: torch.Tensor
    edge_count: torch.Tensor
    effective_k: torch.Tensor
    max_gap_seconds: torch.Tensor
    temperature: float


@dataclass(frozen=True)
class LocalCellSelectionOutput:
    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    log_partition: torch.Tensor
    anchor_positions: torch.Tensor
    cell_starts: torch.Tensor
    cell_ends: torch.Tensor
    k: int
    max_unselected_hole: int
    temperature: float
    force_exact_uniform: bool
    selection_scope: str = "full_window_non_streaming_local_cells"


@dataclass(frozen=True)
class ContinuousDensitySelectionOutput:
    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    continuous_positions: torch.Tensor
    projection_abs_displacement: torch.Tensor
    slot_mask: torch.Tensor
    density: torch.Tensor
    component_densities: torch.Tensor
    mixture_weights: torch.Tensor
    cdf: torch.Tensor
    effective_k: torch.Tensor
    observed_max_unselected_hole: torch.Tensor
    k: int
    max_unselected_hole: int | None
    temperature: float
    coverage_floor: float
    smoothing_kernel: int
    policy_alpha: float
    selection_scope: str = "full_window_offline_continuous_density_transport"


@dataclass(frozen=True)
class SamplingRateSelectionOutput:
    """Exact-K observations generated from calibrated per-frame retention rates.

    ``sampling_rates[t]`` is a probability-like retention frequency in
    ``[0, 1]`` rather than an unbounded ranking score.  Its valid-time sum is
    exactly the requested observation budget.  The deterministic cumulative
    sampler turns that rate field into K distinct integer observations: a
    rate near one keeps every nearby frame, while a rate near 0.75 keeps about
    three out of every four candidates.  The relaxed slots are hard-anchored
    in the forward pass, so their detector-gradient path cannot silently
    evaluate a different fractional observation from the real detector input.
    """

    hard_occupancy: torch.Tensor
    soft_occupancy: torch.Tensor
    soft_slot_assignment: torch.Tensor
    selection_st: torch.Tensor
    selected_positions: torch.Tensor
    continuous_positions: torch.Tensor
    slot_mask: torch.Tensor
    sampling_rates: torch.Tensor
    sampling_density: torch.Tensor
    cumulative_rates: torch.Tensor
    effective_k: torch.Tensor
    observed_max_unselected_hole: torch.Tensor
    calibration_residual: torch.Tensor
    k: int
    temperature: float
    coverage_floor: float
    smoothing_kernel: int
    policy_alpha: float
    selection_scope: str = "full_window_offline_budget_calibrated_sampling_rate"


def exact_uniform_positions(temporal_len: int, k: int, *, device=None) -> torch.Tensor:
    """Return rounded-endpoint anchors with explicit round-half-to-even semantics."""

    temporal_len = int(temporal_len)
    k = int(k)
    if temporal_len < 0 or k < 0 or k > temporal_len:
        raise ValueError("exact-uniform requires 0 <= k <= temporal_len")
    if k == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if k == 1:
        return torch.zeros((1,), device=device, dtype=torch.long)
    denominator = k - 1
    anchors = []
    for index in range(k):
        numerator = index * (temporal_len - 1)
        quotient, remainder = divmod(numerator, denominator)
        if 2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1):
            quotient += 1
        anchors.append(quotient)
    return torch.tensor(anchors, device=device, dtype=torch.long)


def _project_density_quantiles_row(
    continuous_positions: torch.Tensor,
    *,
    temporal_len: int,
    max_unselected_hole: int | None,
) -> torch.Tensor:
    """Project ordered quantiles to the nearest feasible integer sequence."""

    temporal_len = int(temporal_len)
    max_hole = None if max_unselected_hole is None else int(max_unselected_hole)
    k = int(continuous_positions.numel())
    if k == 0:
        return torch.empty((0,), device=continuous_positions.device, dtype=torch.long)
    if max_hole is not None and temporal_len - k > (k + 1) * max_hole:
        raise ValueError(
            "infeasible density-transport exact-K/max-hole contract: "
            f"T={temporal_len}, K={k}, G={max_hole}"
        )
    if max_hole is None:
        # Strictly increasing integer positions y_i are equivalent to a
        # non-decreasing sequence z_i=y_i-i. Isotonic projection centres a
        # collapsed group around its continuous target instead of pushing all
        # duplicate quantiles to one side.
        shifted = (
            continuous_positions.detach().float().cpu()
            - torch.arange(k, dtype=torch.float32)
        ).tolist()
        block_means: list[float] = []
        block_sizes: list[int] = []
        for value in shifted:
            block_means.append(float(value))
            block_sizes.append(1)
            while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
                total = block_sizes[-2] + block_sizes[-1]
                merged = (
                    block_means[-2] * block_sizes[-2]
                    + block_means[-1] * block_sizes[-1]
                ) / float(total)
                block_means[-2:] = [merged]
                block_sizes[-2:] = [total]
        expanded = []
        for value, size in zip(block_means, block_sizes):
            expanded.extend([value] * size)
        max_shift = temporal_len - k
        shifted_integer = torch.tensor(
            [min(max(int(round(value)), 0), max_shift) for value in expanded],
            device=continuous_positions.device,
            dtype=torch.long,
        )
        shifted_integer = torch.cummax(shifted_integer, dim=0).values.clamp_max(max_shift)
        return shifted_integer + torch.arange(k, device=continuous_positions.device)

    projected = []
    previous = -1
    detached = continuous_positions.detach().float()
    for slot_index in range(k):
        remaining = k - slot_index - 1
        lower = max(
            previous + 1,
            temporal_len - 1 - max_hole - remaining * (max_hole + 1),
        )
        upper = min(
            temporal_len - 1 - remaining,
            previous + max_hole + 1,
        )
        if lower > upper:
            raise RuntimeError("density quantile projection reached an infeasible slot")
        target = int(torch.floor(detached[slot_index] + 0.5).item())
        value = min(max(target, lower), upper)
        projected.append(value)
        previous = value
    positions = torch.tensor(projected, device=continuous_positions.device, dtype=torch.long)
    if _max_unselected_hole(positions, temporal_len) > max_hole:
        raise RuntimeError("density quantile projection violated max_unselected_hole")
    return positions


def continuous_density_transport(
    policy_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_unselected_hole: int | None = None,
    component_logits: torch.Tensor | None = None,
    component_mixture_logits: torch.Tensor | None = None,
    temperature: float = 0.7,
    coverage_floor: float = 0.05,
    smoothing_kernel: int = 5,
    policy_alpha: float = 1.0,
    training: bool = False,
    force_exact_uniform: bool = False,
) -> ContinuousDensitySelectionOutput:
    """Turn a smooth temporal density into ordered inverse-CDF observations.

    The hard path is a deterministic exact-K integer projection. The relaxed
    path linearly splats each continuous quantile onto its two neighbouring
    candidates, so detector and boundary losses can differentiate through the
    inverse CDF without changing the hard detector input.
    """

    if not torch.is_tensor(policy_logits) or policy_logits.ndim != 2:
        raise ValueError("policy_logits must be a [B,T] tensor")
    if not policy_logits.is_floating_point() or not bool(torch.isfinite(policy_logits).all().item()):
        raise ValueError("policy_logits must contain finite floating-point values")
    valid = valid_mask.to(device=policy_logits.device, dtype=torch.bool)
    if valid.shape != policy_logits.shape:
        raise ValueError("valid_mask must align with policy_logits")
    if component_logits is None:
        components = policy_logits[:, None, :]
    else:
        if (
            not torch.is_tensor(component_logits)
            or component_logits.ndim != 3
            or component_logits.shape[0] != policy_logits.shape[0]
            or component_logits.shape[2] != policy_logits.shape[1]
        ):
            raise ValueError("component_logits must be [B,C,T] and align with policy_logits")
        if not component_logits.is_floating_point() or not bool(
            torch.isfinite(component_logits).all().item()
        ):
            raise ValueError("component_logits must contain finite floating-point values")
        components = component_logits.to(device=policy_logits.device, dtype=policy_logits.dtype)
    component_count = int(components.shape[1])
    if component_mixture_logits is None:
        if component_count != 1:
            raise ValueError("multiple density components require component_mixture_logits")
        mixture_logits = policy_logits.new_zeros((policy_logits.shape[0], 1))
    else:
        if (
            not torch.is_tensor(component_mixture_logits)
            or component_mixture_logits.shape != (policy_logits.shape[0], component_count)
        ):
            raise ValueError("component_mixture_logits must be [B,C]")
        if not component_mixture_logits.is_floating_point() or not bool(
            torch.isfinite(component_mixture_logits).all().item()
        ):
            raise ValueError("component_mixture_logits must contain finite floating-point values")
        mixture_logits = component_mixture_logits.to(
            device=policy_logits.device,
            dtype=policy_logits.dtype,
        )
    k = int(k)
    max_hole = None if max_unselected_hole is None else int(max_unselected_hole)
    temperature = float(temperature)
    coverage_floor = float(coverage_floor)
    smoothing_kernel = int(smoothing_kernel)
    policy_alpha = float(policy_alpha)
    if k < 0 or k > int(policy_logits.shape[1]):
        raise ValueError("k must lie in [0,T]")
    if max_hole is not None and max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(coverage_floor) or not 0.0 <= coverage_floor < 1.0:
        raise ValueError("coverage_floor must lie in [0,1)")
    if smoothing_kernel <= 0 or smoothing_kernel % 2 == 0:
        raise ValueError("smoothing_kernel must be a positive odd integer")
    if not math.isfinite(policy_alpha) or not 0.0 <= policy_alpha <= 1.0:
        raise ValueError("policy_alpha must lie in [0,1]")

    batch, temporal_len = policy_logits.shape
    hard_rows = []
    soft_rows = []
    slot_rows = []
    position_rows = []
    continuous_rows = []
    slot_mask_rows = []
    density_rows = []
    component_density_rows = []
    mixture_weight_rows = []
    cdf_rows = []
    effective_rows = []
    observed_max_hole_rows = []
    projection_displacement_rows = []
    for batch_index in range(batch):
        valid_positions = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
        valid_len = int(valid_positions.numel())
        expected = torch.arange(valid_len, device=valid.device, dtype=torch.long)
        if not torch.equal(valid_positions, expected):
            raise ValueError("continuous density transport requires a contiguous valid prefix")
        effective_k = min(k, valid_len)
        if effective_k <= 0:
            raise ValueError("continuous density transport requires one valid candidate")
        if max_hole is not None and valid_len - effective_k > (effective_k + 1) * max_hole:
            raise ValueError(
                "infeasible density-transport exact-K/max-hole contract: "
                f"T={valid_len}, K={effective_k}, G={max_hole}"
            )

        row_components = components[batch_index, :, :valid_len].float()
        if smoothing_kernel > 1:
            radius = smoothing_kernel // 2
            padded = F.pad(row_components[:, None, :], (radius, radius), mode="replicate")
            row_components = F.avg_pool1d(
                padded,
                kernel_size=smoothing_kernel,
                stride=1,
            ).squeeze(1)
        learned_components = torch.softmax(row_components / temperature, dim=-1)
        mixture_weights = torch.softmax(mixture_logits[batch_index].float(), dim=0)
        learned_density = torch.sum(
            mixture_weights[:, None] * learned_components,
            dim=0,
            keepdim=True,
        )
        uniform_density = torch.full_like(learned_density, 1.0 / float(valid_len))
        target_density = (
            (1.0 - coverage_floor) * learned_density
            + coverage_floor * uniform_density
        )
        alpha = 0.0 if force_exact_uniform else policy_alpha
        density = (1.0 - alpha) * uniform_density + alpha * target_density

        if valid_len == 1:
            cdf = density.new_zeros((1,))
            continuous = density.new_zeros((effective_k,))
        else:
            interval_mass = 0.5 * (density[0, :-1] + density[0, 1:])
            interval_mass = interval_mass / interval_mass.sum().clamp_min(
                torch.finfo(interval_mass.dtype).eps
            )
            cdf = torch.cat((interval_mass.new_zeros((1,)), interval_mass.cumsum(dim=0)))
            cdf = cdf / cdf[-1].clamp_min(torch.finfo(cdf.dtype).eps)
            quantiles = (
                density.new_tensor([0.5])
                if effective_k == 1
                else torch.linspace(0.0, 1.0, effective_k, device=density.device, dtype=density.dtype)
            )
            right = torch.searchsorted(cdf.detach().contiguous(), quantiles).clamp(1, valid_len - 1)
            left = right - 1
            cdf_left = cdf[left]
            cdf_right = cdf[right]
            fraction = (quantiles - cdf_left) / (cdf_right - cdf_left).clamp_min(
                torch.finfo(cdf.dtype).eps
            )
            continuous = left.to(dtype=density.dtype) + fraction.clamp(0.0, 1.0)

        if force_exact_uniform:
            hard_positions = exact_uniform_positions(valid_len, effective_k, device=policy_logits.device)
        else:
            hard_positions = _project_density_quantiles_row(
                continuous,
                temporal_len=valid_len,
                max_unselected_hole=max_hole,
            )
        hard = policy_logits.new_zeros((temporal_len,))
        hard[hard_positions] = 1.0

        slots = policy_logits.new_zeros((k, temporal_len))
        left_index = continuous.detach().floor().long().clamp(0, valid_len - 1)
        right_index = (left_index + 1).clamp_max(valid_len - 1)
        fraction = (continuous - left_index.to(dtype=continuous.dtype)).clamp(0.0, 1.0)
        active_slots = torch.arange(effective_k, device=policy_logits.device)
        slots[active_slots, left_index] += (1.0 - fraction).to(dtype=slots.dtype)
        slots[active_slots, right_index] += fraction.to(dtype=slots.dtype)
        soft = slots.sum(dim=0)
        selection_st = hard + soft - soft.detach() if training else hard

        padded_positions = torch.full((k,), -1, device=policy_logits.device, dtype=torch.long)
        padded_positions[:effective_k] = hard_positions
        padded_continuous = policy_logits.new_zeros((k,))
        padded_continuous[:effective_k] = continuous.to(dtype=policy_logits.dtype)
        padded_projection_displacement = policy_logits.new_zeros((k,))
        padded_projection_displacement[:effective_k] = (
            hard_positions.to(dtype=continuous.dtype) - continuous
        ).abs().to(dtype=policy_logits.dtype)
        slot_mask = torch.zeros((k,), device=policy_logits.device, dtype=torch.bool)
        slot_mask[:effective_k] = True
        padded_density = policy_logits.new_zeros((temporal_len,))
        padded_density[:valid_len] = density[0].to(dtype=policy_logits.dtype)
        padded_component_density = policy_logits.new_zeros(
            (component_count, temporal_len)
        )
        padded_component_density[:, :valid_len] = learned_components.to(
            dtype=policy_logits.dtype
        )
        padded_cdf = policy_logits.new_zeros((temporal_len,))
        padded_cdf[:valid_len] = cdf.to(dtype=policy_logits.dtype)

        hard_rows.append(hard)
        soft_rows.append(soft)
        slot_rows.append(slots)
        position_rows.append(padded_positions)
        continuous_rows.append(padded_continuous)
        projection_displacement_rows.append(padded_projection_displacement)
        slot_mask_rows.append(slot_mask)
        density_rows.append(padded_density)
        component_density_rows.append(padded_component_density)
        mixture_weight_rows.append(mixture_weights.to(dtype=policy_logits.dtype))
        cdf_rows.append(padded_cdf)
        effective_rows.append(effective_k)
        observed_max_hole_rows.append(
            _max_unselected_hole(hard_positions, valid_len)
        )

    hard_occupancy = torch.stack(hard_rows, dim=0)
    soft_occupancy = torch.stack(soft_rows, dim=0)
    return ContinuousDensitySelectionOutput(
        hard_occupancy=hard_occupancy,
        soft_occupancy=soft_occupancy,
        soft_slot_assignment=torch.stack(slot_rows, dim=0),
        selection_st=(
            hard_occupancy + soft_occupancy - soft_occupancy.detach()
            if training
            else hard_occupancy
        ),
        selected_positions=torch.stack(position_rows, dim=0),
        continuous_positions=torch.stack(continuous_rows, dim=0),
        projection_abs_displacement=torch.stack(
            projection_displacement_rows, dim=0
        ),
        slot_mask=torch.stack(slot_mask_rows, dim=0),
        density=torch.stack(density_rows, dim=0),
        component_densities=torch.stack(component_density_rows, dim=0),
        mixture_weights=torch.stack(mixture_weight_rows, dim=0),
        cdf=torch.stack(cdf_rows, dim=0),
        effective_k=torch.tensor(effective_rows, device=policy_logits.device, dtype=torch.long),
        observed_max_unselected_hole=torch.tensor(
            observed_max_hole_rows,
            device=policy_logits.device,
            dtype=torch.long,
        ),
        k=k,
        max_unselected_hole=max_hole,
        temperature=temperature,
        coverage_floor=coverage_floor,
        smoothing_kernel=smoothing_kernel,
        policy_alpha=policy_alpha,
    )


def _calibrated_retention_rates(
    logits: torch.Tensor,
    *,
    k: int,
    temperature: float,
    iterations: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map logits to capped rates whose sum is exactly the requested budget.

    The bisection threshold is detached for the forward solve, then the
    backward pass is restored on the fixed-budget tangent space. In
    particular, adding a common constant to every logit cannot receive a
    gradient. This is the capped counterpart of a softmax density: no time
    point can claim more than one real video frame.
    """

    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("sampling-rate calibration expects one floating-point row")
    count = int(logits.numel())
    if count <= 0 or k <= 0 or k > count:
        raise ValueError("sampling-rate calibration requires 0 < k <= row length")
    if k == count:
        rates = torch.ones_like(logits)
        return rates, rates.sum() - float(k)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("sampling-rate temperature must be finite and positive")

    work = logits.float()
    scale = float(temperature)
    lower = (work.min().detach() - 32.0 * scale).clone()
    upper = (work.max().detach() + 32.0 * scale).clone()
    target = work.new_tensor(float(k))
    for _ in range(int(iterations)):
        midpoint = 0.5 * (lower + upper)
        mass = torch.sigmoid((work - midpoint) / scale).sum()
        lower = torch.where(mass > target, midpoint, lower)
        upper = torch.where(mass > target, upper, midpoint)
    threshold = (0.5 * (lower + upper)).detach()
    rates = torch.sigmoid((work - threshold) / scale)
    rates = FixedBudgetRateGradient.apply(work, rates, scale)
    residual = rates.sum() - target
    return rates.to(dtype=logits.dtype), residual.to(dtype=logits.dtype)


def budget_calibrated_sampling_rate(
    policy_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    temperature: float = 0.7,
    coverage_floor: float = 0.05,
    smoothing_kernel: int = 5,
    policy_alpha: float = 1.0,
    training: bool = False,
    force_exact_uniform: bool = False,
) -> SamplingRateSelectionOutput:
    """Select K real frames from a calibrated, bounded sampling-rate field.

    This is deliberately not an inverse-CDF density followed by a global
    integer projection.  Each frame owns at most one unit of capacity, the
    calibrated rates sum to K, and cumulative systematic sampling therefore
    produces exactly K strictly increasing original-time indices without any
    duplicate-collision repair.  The relaxed route is anchored at those same
    integer positions in the forward pass and only contributes a local
    temporal derivative in backward.
    """

    if not torch.is_tensor(policy_logits) or policy_logits.ndim != 2:
        raise ValueError("policy_logits must be a [B,T] tensor")
    if not policy_logits.is_floating_point() or not bool(torch.isfinite(policy_logits).all().item()):
        raise ValueError("policy_logits must contain finite floating-point values")
    valid = valid_mask.to(device=policy_logits.device, dtype=torch.bool)
    if valid.shape != policy_logits.shape:
        raise ValueError("valid_mask must align with policy_logits")
    k = int(k)
    temperature = float(temperature)
    coverage_floor = float(coverage_floor)
    smoothing_kernel = int(smoothing_kernel)
    policy_alpha = float(policy_alpha)
    if k <= 0 or k > int(policy_logits.shape[1]):
        raise ValueError("k must lie in (0,T]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(coverage_floor) or not 0.0 <= coverage_floor < 1.0:
        raise ValueError("coverage_floor must lie in [0,1)")
    if smoothing_kernel <= 0 or smoothing_kernel % 2 == 0:
        raise ValueError("smoothing_kernel must be a positive odd integer")
    if not math.isfinite(policy_alpha) or not 0.0 <= policy_alpha <= 1.0:
        raise ValueError("policy_alpha must lie in [0,1]")

    batch, temporal_len = policy_logits.shape
    hard_rows = []
    soft_rows = []
    slot_rows = []
    position_rows = []
    continuous_rows = []
    slot_mask_rows = []
    rate_rows = []
    density_rows = []
    cdf_rows = []
    effective_rows = []
    max_hole_rows = []
    residual_rows = []
    for batch_index in range(batch):
        valid_positions = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
        valid_len = int(valid_positions.numel())
        expected = torch.arange(valid_len, device=valid.device, dtype=torch.long)
        if not torch.equal(valid_positions, expected):
            raise ValueError("sampling-rate selection requires a contiguous valid prefix")
        effective_k = min(k, valid_len)
        if effective_k <= 0:
            raise ValueError("sampling-rate selection requires one valid candidate")

        row_logits = policy_logits[batch_index, :valid_len].float()
        if smoothing_kernel > 1:
            radius = smoothing_kernel // 2
            padded = F.pad(row_logits[None, None, :], (radius, radius), mode="replicate")
            row_logits = F.avg_pool1d(
                padded,
                kernel_size=smoothing_kernel,
                stride=1,
            ).reshape(-1)
        learned_rates, residual = _calibrated_retention_rates(
            row_logits,
            k=effective_k,
            temperature=temperature,
        )
        uniform_rates = torch.full_like(
            learned_rates,
            float(effective_k) / float(valid_len),
        )
        calibrated_rates = (
            (1.0 - coverage_floor) * learned_rates
            + coverage_floor * uniform_rates
        )
        alpha = 0.0 if force_exact_uniform else policy_alpha
        rates = (1.0 - alpha) * uniform_rates + alpha * calibrated_rates
        rates = rates.clamp(min=0.0, max=1.0)
        rate_residual = rates.sum() - float(effective_k)
        if float(rate_residual.detach().abs().item()) > 2.0e-4:
            raise RuntimeError("sampling-rate calibration did not preserve the exact budget")

        if force_exact_uniform:
            hard_positions = exact_uniform_positions(
                valid_len,
                effective_k,
                device=policy_logits.device,
            )
            local_delta = rates.new_zeros((effective_k,))
        else:
            cumulative = rates.cumsum(dim=0)
            thresholds = torch.arange(
                effective_k,
                device=policy_logits.device,
                dtype=rates.dtype,
            ) + 0.5
            hard_positions = torch.searchsorted(
                cumulative.detach().contiguous(),
                thresholds,
                right=False,
            ).clamp(0, valid_len - 1)
            if effective_k > 1 and bool(torch.any(hard_positions[1:] <= hard_positions[:-1]).item()):
                raise RuntimeError("bounded sampling rates must yield strictly increasing observations")
            cumulative_before = F.pad(cumulative[:-1], (1, 0), value=0.0)
            local_rate = rates[hard_positions].clamp_min(torch.finfo(rates.dtype).eps)
            crossing_fraction = (
                (thresholds - cumulative_before[hard_positions]) / local_rate
            ).clamp(0.0, 1.0)
            # Exact hard values in forward, local cumulative-rate derivative in backward.
            local_delta = crossing_fraction - crossing_fraction.detach()

        active_slots = torch.arange(effective_k, device=policy_logits.device)
        direction = torch.where(
            hard_positions < valid_len - 1,
            torch.ones_like(hard_positions),
            -torch.ones_like(hard_positions),
        )
        neighbours = (hard_positions + direction).clamp(0, valid_len - 1)
        hard_slots = F.one_hot(hard_positions, num_classes=valid_len).to(dtype=policy_logits.dtype)
        neighbour_slots = F.one_hot(neighbours, num_classes=valid_len).to(dtype=policy_logits.dtype)
        slots_valid = hard_slots + local_delta[:, None].to(dtype=hard_slots.dtype) * (
            neighbour_slots - hard_slots
        )
        soft = slots_valid.sum(dim=0)
        hard = policy_logits.new_zeros((temporal_len,))
        hard[hard_positions] = 1.0
        slots = policy_logits.new_zeros((k, temporal_len))
        slots[active_slots, :valid_len] = slots_valid

        padded_positions = torch.full((k,), -1, device=policy_logits.device, dtype=torch.long)
        padded_positions[:effective_k] = hard_positions
        padded_continuous = policy_logits.new_zeros((k,))
        padded_continuous[:effective_k] = (
            hard_positions.to(dtype=policy_logits.dtype)
            + direction.to(dtype=policy_logits.dtype) * local_delta.to(dtype=policy_logits.dtype)
        )
        slot_mask = torch.zeros((k,), device=policy_logits.device, dtype=torch.bool)
        slot_mask[:effective_k] = True
        padded_rates = policy_logits.new_zeros((temporal_len,))
        padded_rates[:valid_len] = rates.to(dtype=policy_logits.dtype)
        padded_density = policy_logits.new_zeros((temporal_len,))
        padded_density[:valid_len] = rates.to(dtype=policy_logits.dtype) / float(effective_k)
        padded_cdf = policy_logits.new_zeros((temporal_len,))
        padded_cdf[:valid_len] = rates.cumsum(dim=0).to(dtype=policy_logits.dtype)
        padded_soft = policy_logits.new_zeros((temporal_len,))
        padded_soft[:valid_len] = soft

        hard_rows.append(hard)
        soft_rows.append(padded_soft)
        slot_rows.append(slots)
        position_rows.append(padded_positions)
        continuous_rows.append(padded_continuous)
        slot_mask_rows.append(slot_mask)
        rate_rows.append(padded_rates)
        density_rows.append(padded_density)
        cdf_rows.append(padded_cdf)
        effective_rows.append(effective_k)
        max_hole_rows.append(_max_unselected_hole(hard_positions, valid_len))
        residual_rows.append(rate_residual.to(dtype=policy_logits.dtype))

    hard_occupancy = torch.stack(hard_rows, dim=0)
    soft_occupancy = torch.stack(soft_rows, dim=0)
    return SamplingRateSelectionOutput(
        hard_occupancy=hard_occupancy,
        soft_occupancy=soft_occupancy,
        soft_slot_assignment=torch.stack(slot_rows, dim=0),
        selection_st=(
            hard_occupancy + soft_occupancy - soft_occupancy.detach()
            if training
            else hard_occupancy
        ),
        selected_positions=torch.stack(position_rows, dim=0),
        continuous_positions=torch.stack(continuous_rows, dim=0),
        slot_mask=torch.stack(slot_mask_rows, dim=0),
        sampling_rates=torch.stack(rate_rows, dim=0),
        sampling_density=torch.stack(density_rows, dim=0),
        cumulative_rates=torch.stack(cdf_rows, dim=0),
        effective_k=torch.tensor(effective_rows, device=policy_logits.device, dtype=torch.long),
        observed_max_unselected_hole=torch.tensor(
            max_hole_rows,
            device=policy_logits.device,
            dtype=torch.long,
        ),
        calibration_residual=torch.stack(residual_rows, dim=0),
        k=k,
        temperature=temperature,
        coverage_floor=coverage_floor,
        smoothing_kernel=smoothing_kernel,
        policy_alpha=policy_alpha,
    )


def _validate_physical_axis_row(
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
) -> int:
    if physical_seconds.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("physical time and validity rows must be one-dimensional")
    if physical_seconds.shape != valid_mask.shape:
        raise ValueError("physical time and validity rows must align")
    valid = valid_mask.to(dtype=torch.bool)
    valid_len = int(valid.sum().item())
    expected = torch.arange(
        valid_len,
        device=valid.device,
        dtype=torch.long,
    )
    observed = torch.nonzero(valid, as_tuple=False).flatten()
    if not torch.equal(observed, expected):
        raise ValueError("physical exact-K selection requires a contiguous valid prefix")
    if valid_len == 0:
        return 0
    active = physical_seconds[:valid_len]
    if not active.is_floating_point() or not bool(torch.isfinite(active).all().item()):
        raise ValueError("valid physical seconds must be finite floating-point values")
    if valid_len > 1 and not bool(torch.all(active[1:] > active[:-1]).item()):
        raise ValueError("valid physical seconds must be strictly increasing")
    return valid_len


def physical_exact_uniform_gap_cap(
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Return each sample's exact-uniform-reference maximum interval in seconds."""

    if physical_seconds.ndim != 2 or not physical_seconds.is_floating_point():
        raise ValueError("physical_seconds must be a floating-point [B,T] tensor")
    valid = valid_mask.to(device=physical_seconds.device, dtype=torch.bool)
    if valid.shape != physical_seconds.shape:
        raise ValueError("valid_mask must align with physical_seconds")
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    caps = []
    for batch_idx in range(int(physical_seconds.shape[0])):
        valid_len = _validate_physical_axis_row(
            physical_seconds[batch_idx],
            valid[batch_idx],
        )
        effective_k = min(k, valid_len)
        if valid_len == 0 or effective_k == 0:
            caps.append(physical_seconds.new_zeros(()))
            continue
        row = physical_seconds[batch_idx, :valid_len]
        anchors = exact_uniform_positions(
            valid_len,
            effective_k,
            device=physical_seconds.device,
        )
        selected = row[anchors]
        intervals = [selected[0] - row[0], row[-1] - selected[-1]]
        if effective_k > 1:
            intervals.append(selected[1:] - selected[:-1])
        caps.append(torch.cat([item.reshape(-1) for item in intervals]).max())
    return torch.stack(caps)


def _build_physical_exact_k_graph(
    physical_seconds: torch.Tensor,
    max_gap_seconds: torch.Tensor,
) -> PhysicalExactKGraph:
    """Build the one graph object consumed by both Viterbi and Gibbs paths."""

    temporal_len = int(physical_seconds.numel())
    if temporal_len == 0:
        raise ValueError("physical exact-K graph requires at least one valid candidate")
    cap = max_gap_seconds.to(
        device=physical_seconds.device,
        dtype=physical_seconds.dtype,
    )
    if cap.ndim != 0 or not bool(torch.isfinite(cap).item()) or float(cap.item()) < 0.0:
        raise ValueError("max_gap_seconds must be a finite non-negative scalar")
    tolerance = max(1.0e-9, 8.0 * torch.finfo(physical_seconds.dtype).eps)
    lower = physical_seconds - cap - tolerance
    left = torch.searchsorted(physical_seconds, lower, right=False)
    positions = torch.arange(
        temporal_len,
        device=physical_seconds.device,
        dtype=torch.long,
    )
    predecessor_span = positions - left
    max_span = int(predecessor_span.max().item())
    if max_span == 0:
        predecessor_index = positions.new_zeros((temporal_len, 0))
        predecessor_valid = torch.zeros(
            (temporal_len, 0),
            device=physical_seconds.device,
            dtype=torch.bool,
        )
    else:
        columns = torch.arange(
            max_span,
            device=physical_seconds.device,
            dtype=torch.long,
        )
        predecessor_index = left[:, None] + columns[None, :]
        predecessor_valid = columns[None, :] < predecessor_span[:, None]
        predecessor_index = predecessor_index.clamp(max=temporal_len - 1)

    upper = physical_seconds + cap + tolerance
    right = torch.searchsorted(physical_seconds, upper, right=True)
    successor_span = right - positions - 1
    max_successors = int(successor_span.max().item())
    if max_successors == 0:
        successor_index = positions.new_zeros((temporal_len, 0))
        successor_valid = torch.zeros(
            (temporal_len, 0),
            device=physical_seconds.device,
            dtype=torch.bool,
        )
    else:
        columns = torch.arange(
            max_successors,
            device=physical_seconds.device,
            dtype=torch.long,
        )
        successor_index = positions[:, None] + 1 + columns[None, :]
        successor_valid = columns[None, :] < successor_span[:, None]
        successor_index = successor_index.clamp(max=temporal_len - 1)

    source_valid = physical_seconds - physical_seconds[0] <= cap + tolerance
    sink_valid = physical_seconds[-1] - physical_seconds <= cap + tolerance
    predecessor_edges = int(predecessor_valid.sum().item())
    successor_edges = int(successor_valid.sum().item())
    if predecessor_edges != successor_edges:
        raise RuntimeError("physical exact-K predecessor/successor edge tables disagree")
    return PhysicalExactKGraph(
        predecessor_index=predecessor_index,
        predecessor_valid=predecessor_valid,
        successor_index=successor_index,
        successor_valid=successor_valid,
        source_valid=source_valid,
        sink_valid=sink_valid,
        max_gap_seconds=cap,
        edge_count=(
            predecessor_edges
            + int(source_valid.sum().item())
            + int(sink_valid.sum().item())
        ),
    )


def _safe_masked_logsumexp(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    valid = valid.to(device=values.device, dtype=torch.bool)
    if valid.shape != values.shape:
        valid = valid.expand_as(values)
    reachable = valid & torch.isfinite(values)
    has_reachable = reachable.any(dim=dim)
    finite_floor = values.new_full((), -1.0e30)
    reduced = torch.logsumexp(
        torch.where(reachable, values, finite_floor),
        dim=dim,
    )
    return torch.where(
        has_reachable,
        reduced,
        values.new_full(reduced.shape, float("-inf")),
    )


def _physical_row_viterbi(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: PhysicalExactKGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal_len = int(node_log_probs.numel())
    if k == 0:
        return (
            node_log_probs.new_zeros((temporal_len,)),
            torch.empty((0,), device=node_log_probs.device, dtype=torch.long),
            node_log_probs.new_zeros((0, temporal_len)),
        )
    if graph.source_valid.numel() != temporal_len:
        raise ValueError("physical exact-K graph length does not match node scores")
    scores = node_log_probs.float()
    neg_inf = scores.new_full((), float("-inf"))
    dp = scores.new_full((k, temporal_len), float("-inf"))
    back = torch.full(
        (k, temporal_len),
        -1,
        device=scores.device,
        dtype=torch.long,
    )
    ranks = torch.full(
        (k, temporal_len),
        -1,
        device=scores.device,
        dtype=torch.long,
    )
    dp[0] = torch.where(graph.source_valid, scores, neg_inf)
    initial_positions = torch.nonzero(graph.source_valid, as_tuple=False).flatten()
    ranks[0, initial_positions] = torch.arange(
        initial_positions.numel(),
        device=scores.device,
        dtype=torch.long,
    )
    for slot_idx in range(1, k):
        if graph.predecessor_index.shape[1] == 0:
            continue
        candidates = dp[slot_idx - 1][graph.predecessor_index]
        predecessor_ranks = ranks[slot_idx - 1][graph.predecessor_index]
        candidate_valid = (
            graph.predecessor_valid
            & torch.isfinite(candidates)
            & (predecessor_ranks >= 0)
        )
        candidates = candidates.masked_fill(~candidate_valid, float("-inf"))
        best_value = candidates.max(dim=1).values
        best_score_mask = candidate_valid & (candidates == best_value[:, None])
        rank_sentinel = torch.iinfo(torch.long).max
        tied_ranks = predecessor_ranks.masked_fill(~best_score_mask, rank_sentinel)
        best_rank, best_offset = tied_ranks.min(dim=1)
        reachable = torch.isfinite(best_value) & (best_rank != rank_sentinel)
        best_predecessor = graph.predecessor_index.gather(
            1,
            best_offset[:, None],
        ).squeeze(1)
        dp[slot_idx] = torch.where(
            reachable,
            scores + best_value,
            neg_inf,
        )
        back[slot_idx] = torch.where(
            reachable,
            best_predecessor,
            back[slot_idx],
        )
        reachable_positions = torch.nonzero(reachable, as_tuple=False).flatten()
        if reachable_positions.numel():
            parent_rank = ranks[slot_idx - 1, best_predecessor[reachable_positions]]
            lex_keys = parent_rank * (temporal_len + 1) + reachable_positions
            order = torch.argsort(lex_keys, stable=True)
            ranks[slot_idx, reachable_positions[order]] = torch.arange(
                reachable_positions.numel(),
                device=scores.device,
                dtype=torch.long,
            )
    terminal_valid = graph.sink_valid & torch.isfinite(dp[k - 1]) & (ranks[k - 1] >= 0)
    terminal_value = dp[k - 1].masked_fill(~terminal_valid, float("-inf")).max()
    if not bool(torch.isfinite(terminal_value).item()):
        raise RuntimeError(
            "physical exact-K Viterbi could not reach a legal source-to-sink path"
        )
    tied_terminals = terminal_valid & (dp[k - 1] == terminal_value)
    terminal_ranks = ranks[k - 1].masked_fill(
        ~tied_terminals,
        torch.iinfo(torch.long).max,
    )
    terminal_index = terminal_ranks.argmin()
    positions = torch.empty(
        (k,),
        device=node_log_probs.device,
        dtype=torch.long,
    )
    positions[-1] = terminal_index
    for slot_idx in range(k - 1, 0, -1):
        predecessor = back[slot_idx, positions[slot_idx]]
        if int(predecessor.item()) < 0:
            raise RuntimeError("physical exact-K Viterbi backtracking failed")
        positions[slot_idx - 1] = predecessor
    if k > 1 and not bool(torch.all(positions[1:] > positions[:-1]).item()):
        raise RuntimeError("physical exact-K Viterbi produced unordered positions")
    hard = node_log_probs.new_zeros((temporal_len,))
    hard.scatter_(0, positions, 1.0)
    hard_slots = node_log_probs.new_zeros((k, temporal_len))
    hard_slots.scatter_(1, positions[:, None], 1.0)
    return hard, positions, hard_slots


def _physical_row_forward_backward(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: PhysicalExactKGraph,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal_len = int(node_log_probs.numel())
    if k == 0:
        return (
            node_log_probs.new_zeros((temporal_len,)),
            node_log_probs.new_zeros((0, temporal_len)),
            node_log_probs.float().new_zeros(()),
        )
    if graph.source_valid.numel() != temporal_len:
        raise ValueError("physical exact-K graph length does not match node scores")
    scores = node_log_probs.float() / float(temperature)
    alpha_rows = []
    alpha = torch.where(
        graph.source_valid,
        scores,
        scores.new_full(scores.shape, float("-inf")),
    )
    alpha_rows.append(alpha)
    for _slot_idx in range(1, k):
        if graph.predecessor_index.shape[1] == 0:
            alpha = scores.new_full(scores.shape, float("-inf"))
        else:
            candidates = alpha[graph.predecessor_index]
            predecessor_mass = _safe_masked_logsumexp(
                candidates,
                graph.predecessor_valid,
                dim=1,
            )
            alpha = scores + predecessor_mass
        alpha_rows.append(alpha)
    alpha_table = torch.stack(alpha_rows, dim=0)
    log_partition = _safe_masked_logsumexp(
        alpha_table[-1],
        graph.sink_valid,
        dim=0,
    )
    if not bool(torch.isfinite(log_partition).item()):
        raise RuntimeError(
            "physical exact-K forward-backward could not reach a legal source-to-sink path"
        )

    beta_rows = [scores.new_empty((temporal_len,)) for _ in range(k)]
    beta = torch.where(
        graph.sink_valid,
        scores.new_zeros(scores.shape),
        scores.new_full(scores.shape, float("-inf")),
    )
    beta_rows[k - 1] = beta
    for slot_idx in range(k - 2, -1, -1):
        if graph.successor_index.shape[1] == 0:
            beta = scores.new_full(scores.shape, float("-inf"))
        else:
            candidates = (
                scores[graph.successor_index]
                + beta[graph.successor_index]
            )
            beta = _safe_masked_logsumexp(
                candidates,
                graph.successor_valid,
                dim=1,
            )
        beta_rows[slot_idx] = beta
    beta_table = torch.stack(beta_rows, dim=0)
    log_marginal = alpha_table + beta_table - log_partition
    finite = torch.isfinite(log_marginal)
    slots = torch.where(
        finite,
        torch.exp(log_marginal),
        log_marginal.new_zeros(()),
    )
    row_mass = slots.sum(dim=1)
    if not torch.allclose(
        row_mass,
        torch.ones_like(row_mass),
        atol=2.0e-4,
        rtol=2.0e-4,
    ):
        raise RuntimeError("physical exact-K slot marginals do not sum to one")
    occupancy = slots.sum(dim=0)
    if bool(torch.any(occupancy > 1.0 + 5.0e-4).item()):
        raise RuntimeError("physical exact-K slot marginals exceed unit column occupancy")
    expectations = (
        slots
        * torch.arange(
            temporal_len,
            device=slots.device,
            dtype=slots.dtype,
        )[None, :]
    ).sum(dim=1)
    if k > 1 and not bool(torch.all(expectations[1:] > expectations[:-1]).item()):
        raise RuntimeError("physical exact-K slot expectations are not strictly ordered")
    return occupancy, slots, log_partition


def _prepare_physical_exact_k_batch(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if (
        node_log_probs.ndim != 2
        or not node_log_probs.is_floating_point()
        or physical_seconds.shape != node_log_probs.shape
    ):
        raise ValueError(
            "node_log_probs and physical_seconds must be aligned floating-point [B,T] tensors"
        )
    valid = valid_mask.to(device=node_log_probs.device, dtype=torch.bool)
    physical_seconds = physical_seconds.to(
        device=node_log_probs.device,
        dtype=torch.float64,
    )
    if valid.shape != node_log_probs.shape:
        raise ValueError("valid_mask must align with node_log_probs")
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = _validate_physical_axis_row(
            physical_seconds[batch_idx],
            valid[batch_idx],
        )
        if valid_len == 0:
            raise ValueError("physical exact-K selection requires at least one valid candidate")
        if not bool(
            torch.isfinite(node_log_probs[batch_idx, :valid_len]).all().item()
        ):
            raise ValueError("valid node_log_probs must be finite")
    if max_gap_seconds is None:
        caps = physical_exact_uniform_gap_cap(
            physical_seconds,
            valid,
            k=k,
        )
    else:
        caps = torch.as_tensor(
            max_gap_seconds,
            device=node_log_probs.device,
            dtype=torch.float64,
        ).reshape(-1)
        if caps.numel() == 1 and node_log_probs.shape[0] > 1:
            caps = caps.expand(node_log_probs.shape[0])
        if caps.shape != (node_log_probs.shape[0],):
            raise ValueError("max_gap_seconds must be scalar or [B]")
        if not bool(torch.isfinite(caps).all().item()) or bool(torch.any(caps < 0).item()):
            raise ValueError("max_gap_seconds must contain finite non-negative values")
    return physical_seconds, valid, k, caps


def _pad_physical_hard_row(
    hard_valid: torch.Tensor,
    active_positions: torch.Tensor,
    active_slots: torch.Tensor,
    *,
    temporal_len: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    effective_k = int(active_positions.numel())
    valid_len = int(hard_valid.numel())
    hard = F.pad(hard_valid, (0, temporal_len - valid_len))
    hard_slots = F.pad(
        active_slots,
        (0, temporal_len - valid_len, 0, k - effective_k),
    )
    positions = torch.full(
        (k,),
        -1,
        device=active_positions.device,
        dtype=torch.long,
    )
    positions[:effective_k] = active_positions
    slot_mask = torch.zeros(
        (k,),
        device=active_positions.device,
        dtype=torch.bool,
    )
    slot_mask[:effective_k] = True
    return hard, hard_slots, positions, slot_mask


def _pad_physical_soft_row(
    soft_valid: torch.Tensor,
    active_slots: torch.Tensor,
    *,
    temporal_len: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_k, valid_len = active_slots.shape
    return (
        F.pad(soft_valid, (0, temporal_len - valid_len)),
        F.pad(
            active_slots,
            (0, temporal_len - valid_len, 0, k - effective_k),
        ),
    )


def physical_exact_k_select(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> PhysicalExactKSelectionOutput:
    """Return hard Viterbi and soft Gibbs assignments from the same graph."""

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )

    hard_rows = []
    hard_slot_rows = []
    position_rows = []
    slot_mask_rows = []
    soft_rows = []
    slot_rows = []
    partition_rows = []
    edge_counts = []
    effective_rows = []
    temporal_len = int(node_log_probs.shape[1])
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        effective_rows.append(effective_k)
        row_scores = node_log_probs[batch_idx, :valid_len]
        row_seconds = physical_seconds[batch_idx, :valid_len]
        graph = _build_physical_exact_k_graph(row_seconds, caps[batch_idx])
        hard_valid, active_positions, active_hard_slots = _physical_row_viterbi(
            row_scores.detach(),
            k=effective_k,
            graph=graph,
        )
        soft_valid, active_soft_slots, partition = _physical_row_forward_backward(
            row_scores,
            k=effective_k,
            graph=graph,
            temperature=temperature,
        )
        hard, hard_slots, positions, slot_mask = _pad_physical_hard_row(
            hard_valid,
            active_positions,
            active_hard_slots,
            temporal_len=temporal_len,
            k=k,
        )
        soft, slots = _pad_physical_soft_row(
            soft_valid,
            active_soft_slots,
            temporal_len=temporal_len,
            k=k,
        )
        hard_rows.append(hard)
        hard_slot_rows.append(hard_slots)
        position_rows.append(positions)
        slot_mask_rows.append(slot_mask)
        soft_rows.append(soft)
        slot_rows.append(slots)
        partition_rows.append(partition)
        edge_counts.append(graph.edge_count)

    hard_occupancy = torch.stack(hard_rows, dim=0)
    hard_slot_assignment = torch.stack(hard_slot_rows, dim=0)
    soft_occupancy = torch.stack(soft_rows, dim=0)
    soft_slot_assignment = torch.stack(slot_rows, dim=0)
    selection_st = (
        hard_slot_assignment
        + (soft_slot_assignment - soft_slot_assignment.detach())
    )
    return PhysicalExactKSelectionOutput(
        hard_occupancy=hard_occupancy,
        hard_slot_assignment=hard_slot_assignment,
        hard_positions=torch.stack(position_rows, dim=0),
        hard_slot_mask=torch.stack(slot_mask_rows, dim=0),
        soft_occupancy=soft_occupancy,
        soft_slot_assignment=soft_slot_assignment,
        selection_st=selection_st,
        log_partition=torch.stack(partition_rows, dim=0),
        edge_count=torch.tensor(
            edge_counts,
            device=node_log_probs.device,
            dtype=torch.long,
        ),
        effective_k=torch.tensor(
            effective_rows,
            device=node_log_probs.device,
            dtype=torch.long,
        ),
        # Physical feasibility is a control-plane contract. Keep the exact
        # float64 cap used to build the graph instead of narrowing it to AMP
        # policy-score precision.
        max_gap_seconds=caps.to(device=node_log_probs.device),
        temperature=temperature,
    )


def physical_exact_k_viterbi(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
) -> PhysicalExactKHardOutput:
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )
    temporal_len = int(node_log_probs.shape[1])
    hard_rows = []
    hard_slot_rows = []
    position_rows = []
    slot_mask_rows = []
    edge_counts = []
    effective_rows = []
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        graph = _build_physical_exact_k_graph(
            physical_seconds[batch_idx, :valid_len],
            caps[batch_idx],
        )
        hard_valid, active_positions, active_slots = _physical_row_viterbi(
            node_log_probs[batch_idx, :valid_len].detach(),
            k=effective_k,
            graph=graph,
        )
        hard, hard_slots, positions, slot_mask = _pad_physical_hard_row(
            hard_valid,
            active_positions,
            active_slots,
            temporal_len=temporal_len,
            k=k,
        )
        hard_rows.append(hard)
        hard_slot_rows.append(hard_slots)
        position_rows.append(positions)
        slot_mask_rows.append(slot_mask)
        edge_counts.append(graph.edge_count)
        effective_rows.append(effective_k)
    return PhysicalExactKHardOutput(
        hard_occupancy=torch.stack(hard_rows, dim=0),
        hard_slot_assignment=torch.stack(hard_slot_rows, dim=0),
        hard_positions=torch.stack(position_rows, dim=0),
        hard_slot_mask=torch.stack(slot_mask_rows, dim=0),
        edge_count=torch.tensor(edge_counts, device=node_log_probs.device, dtype=torch.long),
        effective_k=torch.tensor(effective_rows, device=node_log_probs.device, dtype=torch.long),
        max_gap_seconds=caps.to(device=node_log_probs.device),
    )


def physical_exact_k_forward_backward(
    node_log_probs: torch.Tensor,
    physical_seconds: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    max_gap_seconds: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> PhysicalExactKSoftOutput:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    physical_seconds, valid, k, caps = _prepare_physical_exact_k_batch(
        node_log_probs,
        physical_seconds,
        valid_mask,
        k=k,
        max_gap_seconds=max_gap_seconds,
    )
    temporal_len = int(node_log_probs.shape[1])
    soft_rows = []
    slot_rows = []
    partition_rows = []
    edge_counts = []
    effective_rows = []
    for batch_idx in range(int(node_log_probs.shape[0])):
        valid_len = int(valid[batch_idx].sum().item())
        effective_k = min(k, valid_len)
        graph = _build_physical_exact_k_graph(
            physical_seconds[batch_idx, :valid_len],
            caps[batch_idx],
        )
        soft_valid, active_slots, partition = _physical_row_forward_backward(
            node_log_probs[batch_idx, :valid_len],
            k=effective_k,
            graph=graph,
            temperature=temperature,
        )
        soft, slots = _pad_physical_soft_row(
            soft_valid,
            active_slots,
            temporal_len=temporal_len,
            k=k,
        )
        soft_rows.append(soft)
        slot_rows.append(slots)
        partition_rows.append(partition)
        edge_counts.append(graph.edge_count)
        effective_rows.append(effective_k)
    return PhysicalExactKSoftOutput(
        soft_occupancy=torch.stack(soft_rows, dim=0),
        soft_slot_assignment=torch.stack(slot_rows, dim=0),
        log_partition=torch.stack(partition_rows, dim=0),
        edge_count=torch.tensor(edge_counts, device=node_log_probs.device, dtype=torch.long),
        effective_k=torch.tensor(effective_rows, device=node_log_probs.device, dtype=torch.long),
        max_gap_seconds=caps.to(device=node_log_probs.device),
        temperature=temperature,
    )


def exact_uniform_reference_scores(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Score valid positions by distance to the canonical exact-uniform anchors."""

    if not torch.is_tensor(scores) or scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating-point [B,T] tensor")
    valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    if valid.shape != scores.shape:
        raise ValueError("valid_mask must align with scores")
    reference = scores.new_zeros(scores.shape)
    for batch_idx in range(int(scores.shape[0])):
        valid_positions = torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
        effective_k = min(max(int(k), 0), int(valid_positions.numel()))
        if effective_k == 0:
            continue
        anchors = exact_uniform_positions(
            int(valid_positions.numel()),
            effective_k,
            device=scores.device,
        )
        ranks = torch.arange(valid_positions.numel(), device=scores.device)
        distance = (ranks[:, None] - anchors[None, :]).abs().min(dim=1).values
        reference[batch_idx, valid_positions] = -distance.to(dtype=scores.dtype)
    return reference


def physical_exact_k_homotopy_log_potential(
    learned_log_potential: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
    alpha: float,
) -> torch.Tensor:
    """Mix exact-uniform and learned node potentials before exact-K decoding."""

    if (
        not torch.is_tensor(learned_log_potential)
        or learned_log_potential.ndim != 2
        or not learned_log_potential.is_floating_point()
    ):
        raise ValueError(
            "learned_log_potential must be a floating-point [B,T] tensor"
        )
    valid = valid_mask.to(device=learned_log_potential.device, dtype=torch.bool)
    if valid.shape != learned_log_potential.shape:
        raise ValueError("valid_mask must align with learned_log_potential")
    if bool(torch.any(valid.sum(dim=1) == 0).item()):
        raise ValueError("physical exact-K homotopy requires one valid point per row")
    if not bool(torch.isfinite(learned_log_potential[valid]).all().item()):
        raise ValueError("valid learned log-potentials must be finite")
    k = int(k)
    if k < 1 or k > int(learned_log_potential.shape[1]):
        raise ValueError("k must lie in [1,T]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("homotopy alpha must lie in [0,1]")

    reference = exact_uniform_reference_scores(
        learned_log_potential,
        valid,
        k,
    )
    learned_valid = learned_log_potential.masked_fill(~valid, 0.0)
    mixed = (1.0 - alpha) * reference + alpha * learned_valid
    return mixed.masked_fill(~valid, float("-inf"))


def exact_uniform_cell_bounds(
    temporal_len: int,
    k: int,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition a window into nearest-anchor cells with deterministic midpoint ties.

    Every exact-uniform anchor owns one non-empty contiguous cell. A temporal
    midpoint equidistant from adjacent anchors is assigned to the earlier cell.
    """

    temporal_len = int(temporal_len)
    k = int(k)
    anchors = exact_uniform_positions(temporal_len, k, device=device)
    if k == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return anchors, empty, empty
    starts = torch.zeros((k,), device=device, dtype=torch.long)
    ends = torch.full((k,), temporal_len, device=device, dtype=torch.long)
    if k > 1:
        starts[1:] = torch.div(anchors[:-1] + anchors[1:], 2, rounding_mode="floor") + 1
        ends[:-1] = starts[1:]
    if torch.any(starts >= ends):
        raise RuntimeError("exact-uniform cells must be non-empty")
    if torch.any(anchors < starts) or torch.any(anchors >= ends):
        raise RuntimeError("each exact-uniform anchor must lie inside its cell")
    return anchors, starts, ends


def _max_unselected_hole(positions: torch.Tensor, temporal_len: int) -> int:
    if positions.numel() == 0:
        return int(temporal_len)
    sentinels = torch.cat(
        (
            positions.new_tensor([-1]),
            positions,
            positions.new_tensor([int(temporal_len)]),
        )
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _local_cell_max_hole(starts: torch.Tensor, ends: torch.Tensor, temporal_len: int) -> int:
    if starts.numel() == 0:
        return int(temporal_len)
    candidates = [int(ends[0].item()) - 1, int(temporal_len) - int(starts[-1].item()) - 1]
    for cell_index in range(int(starts.numel()) - 1):
        candidates.append(int(ends[cell_index + 1].item()) - int(starts[cell_index].item()) - 2)
    return max(candidates)


def local_cell_deformation(
    policy_logits: torch.Tensor,
    *,
    k: int,
    temperature: float = 1.0,
    training: bool = False,
    force_exact_uniform: bool = False,
) -> LocalCellSelectionOutput:
    """Select exactly one frame per exact-uniform cell.

    The hard and relaxed paths share the same product-of-categorical policy.
    Hard ties explicitly choose the cell's exact-uniform anchor, making a
    zero-initialized policy identical to exact uniform sampling.
    """

    if not torch.is_tensor(policy_logits) or policy_logits.ndim != 2:
        raise ValueError("policy_logits must be a [B,T] tensor")
    if not policy_logits.is_floating_point() or not bool(torch.isfinite(policy_logits).all().item()):
        raise ValueError("policy_logits must contain finite floating-point values")
    temporal_len = int(policy_logits.shape[1])
    k = int(k)
    temperature = float(temperature)
    if k < 0 or k > temporal_len:
        raise ValueError("k must lie in [0,T]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, k, device=policy_logits.device)
    batch = int(policy_logits.shape[0])
    hard = policy_logits.new_zeros((batch, temporal_len))
    slots = policy_logits.new_zeros((batch, k, temporal_len))
    selected = torch.empty((batch, k), device=policy_logits.device, dtype=torch.long)
    log_partition = policy_logits.new_zeros((batch,), dtype=torch.float32)

    for cell_index in range(k):
        start = int(starts[cell_index].item())
        end = int(ends[cell_index].item())
        anchor = int(anchors[cell_index].item())
        cell_scores = policy_logits[:, start:end]
        if force_exact_uniform:
            hard_position = torch.full((batch,), anchor, device=policy_logits.device, dtype=torch.long)
        else:
            detached = cell_scores.detach()
            max_value = detached.max(dim=1).values
            tied = detached == max_value[:, None]
            absolute_positions = torch.arange(start, end, device=policy_logits.device)
            anchor_distance = (absolute_positions - anchor).abs()[None, :].expand(batch, -1)
            invalid_distance = torch.full_like(anchor_distance, temporal_len + 1)
            closest_tied = torch.where(tied, anchor_distance, invalid_distance).argmin(dim=1)
            hard_position = closest_tied + start
        selected[:, cell_index] = hard_position
        hard.scatter_(1, hard_position[:, None], 1.0)

        if training and not force_exact_uniform:
            cell_logits = cell_scores.float() / temperature
            cell_weights = torch.softmax(cell_logits, dim=1).to(dtype=policy_logits.dtype)
            slots[:, cell_index, start:end] = cell_weights
            log_partition = log_partition + torch.logsumexp(cell_logits, dim=1)
        else:
            slots[:, cell_index].scatter_(1, hard_position[:, None], 1.0)

    soft = slots.sum(dim=1)
    selection_st = hard + soft - soft.detach() if training and not force_exact_uniform else hard
    theoretical_max_hole = _local_cell_max_hole(starts, ends, temporal_len)
    if k > 0:
        observed = max(_max_unselected_hole(selected[row], temporal_len) for row in range(batch))
        if observed > theoretical_max_hole:
            raise RuntimeError("local-cell selection violated its maximum-hole contract")
    return LocalCellSelectionOutput(
        hard_occupancy=hard,
        soft_occupancy=soft,
        soft_slot_assignment=slots,
        selection_st=selection_st,
        selected_positions=selected,
        log_partition=log_partition,
        anchor_positions=anchors,
        cell_starts=starts,
        cell_ends=ends,
        k=k,
        max_unselected_hole=theoretical_max_hole,
        temperature=temperature,
        force_exact_uniform=bool(force_exact_uniform),
    )


def _validate_contract(logits: torch.Tensor, k: int, max_hole: int, temperature: float) -> tuple[int, int, float]:
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("policy logits must be a [B,T] tensor")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("policy logits must be finite floating-point values")
    k = int(k)
    max_hole = int(max_hole)
    temperature = float(temperature)
    if k < 0 or k > int(logits.shape[1]):
        raise ValueError("k must lie in [0,T]")
    if max_hole < 0:
        raise ValueError("max_unselected_hole must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    temporal_len = int(logits.shape[1])
    if temporal_len - k > (k + 1) * max_hole:
        raise ValueError(
            "infeasible exact-K/max-unselected-hole contract: "
            f"T={temporal_len}, K={k}, G={max_hole}"
        )
    return k, max_hole, temperature


def _required_selection_mask(
    logits: torch.Tensor,
    required_mask: torch.Tensor | None,
    *,
    k: int,
) -> torch.Tensor:
    if required_mask is None:
        return torch.zeros_like(logits, dtype=torch.bool)
    required = required_mask.to(device=logits.device, dtype=torch.bool)
    if required.shape != logits.shape:
        raise ValueError("required_mask must align with policy logits")
    if bool(torch.any(required.sum(dim=1) > int(k)).item()):
        raise ValueError("required structured selections exceed exact K")
    return required


def _hard_viterbi(
    logits: torch.Tensor,
    *,
    k: int,
    max_hole: int,
    required_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, temporal_len = logits.shape
    work = logits.float()
    required = _required_selection_mask(logits, required_mask, k=k)
    neg_inf = torch.tensor(float("-inf"), device=work.device, dtype=work.dtype)
    dp = torch.full((batch, k + 1, max_hole + 1), neg_inf, device=work.device, dtype=work.dtype)
    dp[:, 0, 0] = 0.0
    select_prev_gaps: list[torch.Tensor] = []

    for time_idx in range(temporal_len):
        if k > 0:
            best_select, best_gap = dp[:, :k, :].max(dim=2)
            selected_scores = best_select + work[:, time_idx, None]
            select_gap_zero = torch.cat(
                (torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype), selected_scores),
                dim=1,
            )
            back_gap = torch.cat(
                (torch.zeros((batch, 1), device=work.device, dtype=torch.long), best_gap),
                dim=1,
            )
        else:
            select_gap_zero = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
            back_gap = torch.zeros((batch, 1), device=work.device, dtype=torch.long)
        skipped = dp[:, :, :max_hole] if max_hole > 0 else dp[:, :, :0]
        if bool(required[:, time_idx].any().item()):
            skipped = skipped.masked_fill(
                required[:, time_idx, None, None],
                neg_inf,
            )
        dp = torch.cat((select_gap_zero[:, :, None], skipped), dim=2)
        select_prev_gaps.append(back_gap)

    terminal_scores, terminal_gap = dp[:, k, :].max(dim=1)
    if not bool(torch.isfinite(terminal_scores).all().item()):
        raise RuntimeError("structured Viterbi failed to reach an exact-budget terminal state")

    hard = torch.zeros((batch, temporal_len), device=logits.device, dtype=logits.dtype)
    for batch_idx in range(batch):
        count = int(k)
        gap = int(terminal_gap[batch_idx].item())
        for time_idx in range(temporal_len - 1, -1, -1):
            if gap == 0:
                hard[batch_idx, time_idx] = 1.0
                gap = int(select_prev_gaps[time_idx][batch_idx, count].item())
                count -= 1
            else:
                gap -= 1
        if count != 0:
            raise RuntimeError("structured Viterbi backtracking did not recover exactly K selections")
    positions = torch.arange(temporal_len, device=logits.device, dtype=torch.long)[None, :]
    positions = positions.expand(batch, -1)[hard.bool()].reshape(batch, k)
    if not bool(torch.all(hard.bool() | ~required).item()):
        raise RuntimeError("structured Viterbi omitted a required selection")
    return hard, positions


def _soft_forward_backward(
    logits: torch.Tensor,
    *,
    k: int,
    max_hole: int,
    temperature: float,
    required_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, temporal_len = logits.shape
    required = _required_selection_mask(logits, required_mask, k=k)
    work = logits.float() / float(temperature)
    work = work - work.detach().amax(dim=1, keepdim=True)
    work = work.clamp(min=-80.0, max=0.0)
    # A finite sentinel avoids undefined gradients from logaddexp(-inf, -inf)
    # while remaining far outside any attainable path score.
    neg_inf = torch.tensor(-1.0e9, device=work.device, dtype=work.dtype)
    alpha = torch.full((batch, k + 1, max_hole + 1), neg_inf, device=work.device, dtype=work.dtype)
    alpha[:, 0, 0] = 0.0
    alphas = [alpha]
    for time_idx in range(temporal_len):
        if k > 0:
            select_score = torch.logsumexp(alpha[:, :k, :], dim=2) + work[:, time_idx, None]
            select_zero = torch.cat(
                (torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype), select_score),
                dim=1,
            )
        else:
            select_zero = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
        skipped = alpha[:, :, :max_hole] if max_hole > 0 else alpha[:, :, :0]
        if bool(required[:, time_idx].any().item()):
            skipped = skipped.masked_fill(
                required[:, time_idx, None, None],
                neg_inf,
            )
        alpha = torch.cat((select_zero[:, :, None], skipped), dim=2)
        alphas.append(alpha)

    beta = torch.full_like(alpha, neg_inf)
    beta[:, k, :] = 0.0
    betas: list[torch.Tensor] = [beta]
    for time_idx in range(temporal_len - 1, -1, -1):
        if k > 0:
            select_by_count = torch.cat(
                (
                    beta[:, 1:, 0] + work[:, time_idx, None],
                    torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype),
                ),
                dim=1,
            )
        else:
            select_by_count = torch.full((batch, 1), neg_inf, device=work.device, dtype=work.dtype)
        select_candidate = select_by_count[:, :, None].expand(-1, -1, max_hole + 1)
        if max_hole > 0:
            skip_candidate = torch.cat(
                (
                    beta[:, :, 1:],
                    torch.full((batch, k + 1, 1), neg_inf, device=work.device, dtype=work.dtype),
                ),
                dim=2,
            )
        else:
            skip_candidate = torch.full_like(beta, neg_inf)
        if bool(required[:, time_idx].any().item()):
            skip_candidate = skip_candidate.masked_fill(
                required[:, time_idx, None, None],
                neg_inf,
            )
        beta = torch.logaddexp(select_candidate, skip_candidate)
        betas.append(beta)
    betas.reverse()

    log_partition = betas[0][:, 0, 0]
    if not bool(torch.isfinite(log_partition).all().item()):
        raise RuntimeError("structured soft DP failed to reach an exact-budget terminal state")
    if k == 0:
        slots = work.new_zeros((batch, 0, temporal_len))
    else:
        slot_steps = []
        for time_idx in range(temporal_len):
            transition_log_prob = (
                alphas[time_idx][:, :k, :]
                + work[:, time_idx, None, None]
                + betas[time_idx + 1][:, 1:, 0, None]
                - log_partition[:, None, None]
            )
            slot_steps.append(torch.exp(torch.logsumexp(transition_log_prob, dim=2)))
        slots = torch.stack(slot_steps, dim=2)
        slot_mass = slots.sum(dim=2, keepdim=True).clamp_min(torch.finfo(slots.dtype).tiny)
        slots = slots / slot_mass
    occupancy = slots.sum(dim=1)
    return occupancy.to(dtype=logits.dtype), slots.to(dtype=logits.dtype), log_partition


def _structured_log_partition(
    logits: torch.Tensor,
    *,
    k: int,
    max_hole: int,
    temperature: float,
    selection_allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return log Z for exact-K/max-hole paths, optionally forbidding selections."""

    batch, temporal_len = logits.shape
    work = logits.float() / float(temperature)
    if selection_allowed is None:
        allowed = torch.ones_like(logits, dtype=torch.bool)
    else:
        allowed = selection_allowed.to(device=logits.device, dtype=torch.bool)
        if allowed.shape != logits.shape:
            raise ValueError("selection_allowed must align with logits")
    neg_inf = work.new_tensor(-1.0e9)

    def safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
        reachable = values > -5.0e8
        has_reachable = reachable.any(dim=dim, keepdim=True)
        masked_values = torch.where(reachable, values, values.new_full((), -1.0e30))
        reduced = torch.logsumexp(masked_values, dim=dim, keepdim=True)
        return torch.where(has_reachable, reduced, neg_inf).squeeze(dim)

    alpha = work.new_full((batch, k + 1, max_hole + 1), neg_inf.item())
    alpha[:, 0, 0] = 0.0
    for time_idx in range(temporal_len):
        if k > 0:
            selected = safe_logsumexp(alpha[:, :k, :], dim=2) + work[:, time_idx, None]
            selected = selected.masked_fill(~allowed[:, time_idx, None], neg_inf.item())
            select_zero = torch.cat((work.new_full((batch, 1), neg_inf.item()), selected), dim=1)
        else:
            select_zero = work.new_full((batch, 1), neg_inf.item())
        skipped = alpha[:, :, :max_hole] if max_hole > 0 else alpha[:, :, :0]
        alpha = torch.cat((select_zero[:, :, None], skipped), dim=2)
    return safe_logsumexp(alpha[:, k, :], dim=1)


def structured_local_coverage_probability(
    policy_logits: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    k: int,
    max_unselected_hole: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact probability that a structured path selects inside each event mask.

    ``event_mask`` is ``[B,N,T]``. The result is ``[B,N]`` and is computed as
    ``1 - Z(no selection in event) / Z`` under the same exact-K/max-hole path
    distribution used by :func:`global_structured_topk`.
    """

    k, max_hole, temperature = _validate_contract(
        policy_logits, k, max_unselected_hole, temperature
    )
    events = event_mask.to(device=policy_logits.device, dtype=torch.bool)
    if events.ndim != 3 or events.shape[0] != policy_logits.shape[0] or events.shape[2] != policy_logits.shape[1]:
        raise ValueError("event_mask must be [B,N,T] and align with policy_logits")
    log_z = _structured_log_partition(
        policy_logits, k=k, max_hole=max_hole, temperature=temperature
    )
    probabilities = []
    for event_idx in range(events.shape[1]):
        log_z_miss = _structured_log_partition(
            policy_logits,
            k=k,
            max_hole=max_hole,
            temperature=temperature,
            selection_allowed=~events[:, event_idx],
        )
        impossible_miss = log_z_miss <= -5.0e8
        safe_log_z_miss = torch.where(impossible_miss, log_z.detach(), log_z_miss)
        log_miss = (safe_log_z_miss - log_z).clamp(max=0.0)
        probability = -torch.expm1(log_miss)
        probabilities.append(torch.where(impossible_miss, torch.ones_like(probability), probability))
    if not probabilities:
        return policy_logits.new_zeros((policy_logits.shape[0], 0))
    return torch.stack(probabilities, dim=1).to(dtype=policy_logits.dtype).clamp(0.0, 1.0)


def global_structured_topk(
    policy_logits: torch.Tensor,
    *,
    k: int,
    max_unselected_hole: int,
    required_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    training: bool = False,
) -> StructuredSelectionOutput:
    """Full-window exact-budget selection with a shared hard/soft structured policy."""

    k, max_hole, temperature = _validate_contract(
        policy_logits,
        k,
        max_unselected_hole,
        temperature,
    )
    required = _required_selection_mask(policy_logits, required_mask, k=k)
    hard, positions = _hard_viterbi(
        policy_logits.detach(),
        k=k,
        max_hole=max_hole,
        required_mask=required,
    )
    if training:
        soft, slots, log_partition = _soft_forward_backward(
            policy_logits,
            k=k,
            max_hole=max_hole,
            temperature=temperature,
            required_mask=required,
        )
        selection_st = hard + (soft - soft.detach())
    else:
        soft = hard
        slots = policy_logits.new_zeros((policy_logits.shape[0], k, policy_logits.shape[1]))
        if k > 0:
            slots.scatter_(2, positions[:, :, None], 1.0)
        selection_st = hard
        log_partition = policy_logits.new_full((policy_logits.shape[0],), float("nan"))
    return StructuredSelectionOutput(
        hard_occupancy=hard,
        soft_occupancy=soft,
        soft_slot_assignment=slots,
        selection_st=selection_st,
        selected_positions=positions,
        log_partition=log_partition,
        k=k,
        max_unselected_hole=max_hole,
        temperature=temperature,
    )


__all__ = [
    "LocalCellSelectionOutput",
    "PhysicalExactKGraph",
    "PhysicalExactKHardOutput",
    "PhysicalExactKSelectionOutput",
    "PhysicalExactKSoftOutput",
    "StructuredSelectionOutput",
    "exact_uniform_cell_bounds",
    "exact_uniform_positions",
    "exact_uniform_reference_scores",
    "global_structured_topk",
    "local_cell_deformation",
    "physical_exact_k_forward_backward",
    "physical_exact_k_homotopy_log_potential",
    "physical_exact_k_select",
    "physical_exact_k_viterbi",
    "physical_exact_uniform_gap_cap",
    "structured_local_coverage_probability",
]
