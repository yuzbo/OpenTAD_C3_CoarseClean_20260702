"""Pure routing primitives for GeoRoute-AdaTAD.

The module deliberately separates three objects which are often conflated in
vision routing implementations:

* the hard, exact-K native-patch route used by the forward pass;
* the straight-through (ST) content gate, which is a biased surrogate; and
* the Plackett-Luce score-function log-probability for stochastic hard routes.

It contains no dataset, detector, ground-truth, teacher, or evaluation code.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F


GEOROUTE_ROUTING_SCHEMA = "georoute_native_routing_v2"
GEOROUTE_STRUCTURED_ROUTING_SCHEMA = "georoute_fixed_quota_structured_routing_v1"
GEOROUTE_DYNAMIC_ROUTING_SCHEMA = "georoute_dynamic_global_routing_v2"
LEGACY_ROUTE_MODES = frozenset(
    {"dense", "uniform", "random", "free", "roi", "hybrid"}
)
STRUCTURED_ROUTE_MODES = frozenset(
    {
        "structured_context_residual",
        "structured_context_roi",
        "structured_hybrid",
        "structured_hybrid_geometry_shift",
    }
)
DYNAMIC_ROUTE_MODES = frozenset({"dynamic_scnr"})
ROUTE_MODES = LEGACY_ROUTE_MODES | STRUCTURED_ROUTE_MODES | DYNAMIC_ROUTE_MODES
POLICY_ESTIMATORS = frozenset({"none", "straight_through", "score_function"})
SCORE_FUNCTION_TEMPORAL_REDUCTIONS = frozenset({"sum", "mean"})
_ROUTE_PRIVATE_RNG_SCHEMA = "georoute_route_private_rng_v1"


def _extent_wh(
    value: float | tuple[float, float],
    *,
    name: str,
) -> tuple[float, float]:
    """Normalize one scalar or explicit ``(width, height)`` extent pair."""

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or a (width,height) pair")
        width, height = map(float, value)
    else:
        width = height = float(value)
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError(f"{name} must be finite")
    return width, height


def native_cell_extent_floor(
    grid_height: int,
    grid_width: int,
    *,
    cells_per_axis: int,
) -> tuple[float, float]:
    """Return a runtime native-cell ``(width, height)`` ROI floor.

    A one-cell floor is ``(1 / W_grid, 1 / H_grid)``.  Keeping the two axes
    separate is essential on the production 11x20 lattice: replacing them by a
    shared maximum would silently turn the approved rectangular parameterization
    into a larger square-footprint constraint.
    """

    if int(grid_height) <= 0 or int(grid_width) <= 0:
        raise ValueError("native patch grid dimensions must be positive")
    if isinstance(cells_per_axis, bool) or int(cells_per_axis) != cells_per_axis:
        raise ValueError("native ROI floor cells must be one positive integer")
    cells = int(cells_per_axis)
    if cells <= 0:
        raise ValueError("native ROI floor cells must be one positive integer")
    if cells > int(grid_height) or cells > int(grid_width):
        raise ValueError("native ROI floor cells exceed the runtime patch grid")
    return cells / float(grid_width), cells / float(grid_height)


def decode_continuous_geometry(
    logits: torch.Tensor,
    *,
    min_extent: float | tuple[float, float],
    max_extent: float | tuple[float, float],
) -> torch.Tensor:
    """Decode unconstrained logits to in-bounds ``(cx, cy, w, h)`` boxes.

    Centres are parameterized *after* width/height are decoded, so every box
    remains in the normalized source frame without a non-differentiable clamp.
    """

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("geometry logits must be [B,T,4]")
    min_width, min_height = _extent_wh(min_extent, name="min_extent")
    max_width, max_height = _extent_wh(max_extent, name="max_extent")
    if not all(
        0.0 < minimum <= maximum <= 1.0
        for minimum, maximum in (
            (min_width, max_width),
            (min_height, max_height),
        )
    ):
        raise ValueError(
            "geometry width/height extents must independently satisfy "
            "0 < min <= max <= 1"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("geometry logits must be finite")

    minimum = logits.new_tensor((min_width, min_height))
    maximum = logits.new_tensor((max_width, max_height))
    extent = minimum + (maximum - minimum) * torch.sigmoid(logits[..., 2:])
    center_unit = torch.sigmoid(logits[..., :2])
    center = 0.5 * extent + (1.0 - extent) * center_unit
    geometry = torch.cat((center, extent), dim=-1)
    if not bool(torch.isfinite(geometry).all().item()):
        raise FloatingPointError("decoded geometry is non-finite")
    return geometry


def native_patch_centers(
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized native-patch centres in row-major patch order."""

    if int(grid_height) <= 0 or int(grid_width) <= 0:
        raise ValueError("native patch grid dimensions must be positive")
    yy, xx = torch.meshgrid(
        (torch.arange(grid_height, device=device, dtype=dtype) + 0.5) / float(grid_height),
        (torch.arange(grid_width, device=device, dtype=dtype) + 0.5) / float(grid_width),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def roi_logits_from_geometry(
    geometry: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    temperature: float,
) -> torch.Tensor:
    """Score every native patch by smooth elliptical membership in an ROI."""

    if geometry.ndim != 3 or geometry.shape[-1] != 4:
        raise ValueError("geometry must be [B,T,4]")
    if float(temperature) <= 0.0:
        raise ValueError("ROI temperature must be positive")
    centers = native_patch_centers(
        grid_height,
        grid_width,
        device=geometry.device,
        dtype=geometry.dtype,
    ).view(1, 1, -1, 2)
    normalized = (centers - geometry[..., None, :2]) / geometry[..., None, 2:].clamp_min(1e-6)
    return -0.5 * normalized.square().sum(dim=-1) / float(temperature)


def roi_modifier_from_geometry(
    geometry: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    temperature: float,
) -> torch.Tensor:
    """Return a signed ROI-membership modifier on native patch centres.

    The zero contour is the decoded ellipse boundary.  Positive values denote
    patch centres inside the ROI and negative values denote centres outside it.
    Unlike ``exp(roi_logits)``, the signed margin leaves context (the zero
    modifier) identifiable outside the ROI.
    """

    if geometry.ndim != 3 or geometry.shape[-1] != 4:
        raise ValueError("geometry must be [B,T,4]")
    if float(temperature) <= 0.0:
        raise ValueError("ROI temperature must be positive")
    centers = native_patch_centers(
        grid_height,
        grid_width,
        device=geometry.device,
        dtype=geometry.dtype,
    ).view(1, 1, -1, 2)
    # ``decode_continuous_geometry`` defines w/h as full box extents (the
    # centre is constrained by extent/2).  The ellipse semi-axes are therefore
    # w/2 and h/2.  Dividing by the full extent would place the claimed zero
    # contour at twice the decoded ROI width/height and systematically suppress
    # the context/residual roles.
    semi_extent = 0.5 * geometry[..., None, 2:]
    normalized = (
        centers - geometry[..., None, :2]
    ) / semi_extent.clamp_min(1e-6)
    return (1.0 - normalized.square().sum(dim=-1)) / (
        2.0 * float(temperature)
    )


def interpolate_temporal_knots(values: torch.Tensor, *, stride: int) -> torch.Tensor:
    """Linearly interpolate an explicit temporal knot trajectory.

    ``stride`` is measured in native two-frame tubelets.  A stride of one is
    the finest route and returns the original tensor exactly; larger strides
    retain the final tubelet as a knot so interpolation remains aligned to the
    complete detector window.  All operations are tensor operations, so the
    interpolated geometry keeps gradients to the selected knots.
    """

    if values.ndim < 3:
        raise ValueError("temporal knot values must be [B,T,...]")
    if int(stride) <= 0:
        raise ValueError("temporal knot stride must be positive")
    tubelets = int(values.shape[1])
    if tubelets <= 0:
        raise ValueError("temporal knot values require a non-empty time axis")
    if int(stride) == 1 or tubelets == 1:
        return values

    device = values.device
    knot_positions = torch.arange(0, tubelets, int(stride), device=device, dtype=torch.long)
    if int(knot_positions[-1].item()) != tubelets - 1:
        knot_positions = torch.cat((knot_positions, knot_positions.new_tensor([tubelets - 1])))
    knots = values.index_select(1, knot_positions)
    timeline = torch.arange(tubelets, device=device, dtype=torch.long)
    right = torch.bucketize(timeline, knot_positions, right=False).clamp(max=knot_positions.numel() - 1)
    left = (right - 1).clamp(min=0)
    left_positions = knot_positions.index_select(0, left)
    right_positions = knot_positions.index_select(0, right)
    span = (right_positions - left_positions).clamp_min(1)
    alpha = (timeline - left_positions).to(dtype=values.dtype) / span.to(dtype=values.dtype)
    alpha = alpha.reshape(1, tubelets, *([1] * (values.ndim - 2)))
    left_values = knots.index_select(1, left)
    right_values = knots.index_select(1, right)
    return left_values + alpha * (right_values - left_values)


def _stable_argsort_descending(scores: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(scores, dim=-1, descending=True, stable=True)
    except TypeError:  # pragma: no cover - compatibility with older torch builds
        # A tiny deterministic index offset resolves exact ties without affecting
        # ordinary score comparisons at float32 precision.
        count = scores.shape[-1]
        offset = torch.arange(count, device=scores.device, dtype=scores.dtype) * -1e-12
        return torch.argsort(scores + offset, dim=-1, descending=True)


def _deterministic_uniform_indices(
    *,
    item_count: int,
    select_count: int,
    device: torch.device,
) -> torch.Tensor:
    if not (0 < int(select_count) <= int(item_count)):
        raise ValueError("uniform selection count must lie in [1,item_count]")
    if select_count == 1:
        return torch.tensor([item_count // 2], device=device, dtype=torch.long)
    numerators = torch.arange(select_count, device=device, dtype=torch.long) * (item_count - 1)
    return torch.div(numerators, select_count - 1, rounding_mode="floor")


def _deterministic_uniform_valid_indices(
    valid_mask: torch.Tensor,
    *,
    select_count: int,
) -> torch.Tensor:
    """Choose a row-major uniform subset from each valid native lattice."""

    if valid_mask.ndim != 3 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,T,N]")
    valid_counts = valid_mask.sum(dim=-1)
    if bool((valid_counts < int(select_count)).any().item()):
        raise ValueError("valid native patch count is smaller than exact-K")
    item_count = int(valid_mask.shape[-1])
    row_major = torch.arange(
        item_count,
        device=valid_mask.device,
        dtype=torch.long,
    ).view(1, 1, item_count)
    compact = torch.sort(
        row_major.expand_as(valid_mask).masked_fill(~valid_mask, item_count),
        dim=-1,
    ).values
    if int(select_count) == 1:
        compact_positions = torch.div(valid_counts, 2, rounding_mode="floor").unsqueeze(-1)
    else:
        numerators = torch.arange(
            int(select_count),
            device=valid_mask.device,
            dtype=torch.long,
        ).view(1, 1, -1) * (valid_counts - 1).unsqueeze(-1)
        compact_positions = torch.div(
            numerators,
            int(select_count) - 1,
            rounding_mode="floor",
        )
    return compact.gather(-1, compact_positions)


def _mask_from_indices(indices: torch.Tensor, item_count: int) -> torch.Tensor:
    if indices.ndim != 3:
        raise ValueError("indices must be [B,T,K]")
    mask = torch.zeros(
        (*indices.shape[:2], int(item_count)),
        device=indices.device,
        dtype=torch.bool,
    )
    return mask.scatter(-1, indices, True)


def _topk_excluding(scores: torch.Tensor, *, count: int, excluded: torch.Tensor) -> torch.Tensor:
    if count == 0:
        return torch.empty((*scores.shape[:2], 0), device=scores.device, dtype=torch.long)
    available = int((~excluded).sum(dim=-1).min().item())
    if count > available:
        raise ValueError("not enough unselected tokens for exact-K route")
    masked = scores.masked_fill(excluded, float("-inf"))
    return _stable_argsort_descending(masked)[..., :count]


def _sample_plackett_luce_order(
    logits: torch.Tensor,
    *,
    count: int,
    temperature: float,
) -> torch.Tensor:
    if count <= 0:
        return torch.empty((*logits.shape[:2], 0), device=logits.device, dtype=torch.long)
    if float(temperature) <= 0:
        raise ValueError("score-function temperature must be positive")
    uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    return _stable_argsort_descending(logits / float(temperature) + gumbel)[..., :count]


def _route_private_seed(
    *,
    study_seed: int,
    successful_update_index: int,
    distributed_rank: int,
    role_id: str,
) -> int:
    """Derive one immutable route-only seed without touching a global RNG."""

    if int(successful_update_index) < 0:
        raise ValueError("successful update index must be non-negative")
    if int(distributed_rank) < 0:
        raise ValueError("distributed rank must be non-negative")
    if not isinstance(role_id, str) or not role_id:
        raise ValueError("route RNG role_id must be a non-empty string")
    payload = (
        f"{_ROUTE_PRIVATE_RNG_SCHEMA}:{int(study_seed)}:"
        f"{int(successful_update_index)}:{int(distributed_rank)}:{role_id}"
    ).encode("utf-8")
    # torch.Generator.manual_seed accepts signed-64-bit-compatible seeds.  The
    # modulus also keeps the value portable across the CUDA/PyTorch versions
    # frozen by the experiment contract.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _route_private_generator(
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def sample_ordered_pl_with_exclusion(
    logits: torch.Tensor,
    *,
    count: int,
    temperature: float,
    valid_mask: torch.Tensor,
    excluded_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample and score an ordered PL action on an explicitly reduced support.

    The returned likelihood uses exactly the same valid-minus-excluded support
    as the hard Gumbel-top-k draw.  Constructing ``generator`` from a route-only
    key makes the draw replayable across AMP retries without advancing either
    the process-global CPU RNG or CUDA RNG.
    """

    if logits.ndim != 3:
        raise ValueError("conditional Plackett-Luce logits must be [B,T,N]")
    if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("conditional Plackett-Luce valid_mask must match logits")
    if excluded_mask.shape != logits.shape or excluded_mask.dtype != torch.bool:
        raise ValueError("conditional Plackett-Luce excluded_mask must match logits")
    if not isinstance(generator, torch.Generator):
        raise TypeError("conditional Plackett-Luce requires a private Generator")
    if int(count) < 0:
        raise ValueError("conditional Plackett-Luce count must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("conditional Plackett-Luce temperature must be positive")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("conditional Plackett-Luce logits must be finite")

    available = valid_mask & ~excluded_mask
    if bool((available.sum(dim=-1) < int(count)).any().item()):
        raise ValueError("conditional Plackett-Luce support is smaller than its quota")
    if int(count) == 0:
        return (
            torch.empty((*logits.shape[:2], 0), device=logits.device, dtype=torch.long),
            torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.float32),
        )

    uniform = torch.rand(
        logits.shape,
        device=logits.device,
        dtype=logits.dtype,
        generator=generator,
    ).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    sampled_scores = (logits / float(temperature) + gumbel).masked_fill(
        ~available,
        float("-inf"),
    )
    ordered = _stable_argsort_descending(sampled_scores)[..., : int(count)]
    log_probability = ordered_plackett_luce_log_prob(
        logits,
        ordered,
        temperature=temperature,
        valid_mask=available,
    )
    return ordered, log_probability


def ordered_plackett_luce_log_prob(
    logits: torch.Tensor,
    ordered_indices: torch.Tensor,
    *,
    temperature: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-probability of an ordered sample without replacement.

    Gumbel-top-k sampling with logits divided by ``temperature`` is equivalent
    to this Plackett-Luce distribution.  This is the score-function estimator's
    likelihood term, not a differentiable relaxation of membership.
    """

    if logits.ndim != 3 or ordered_indices.ndim != 3:
        raise ValueError("logits and ordered indices must both be [B,T,*]")
    if logits.shape[:2] != ordered_indices.shape[:2]:
        raise ValueError("logits and ordered indices batch/time axes must match")
    if float(temperature) <= 0:
        raise ValueError("score-function temperature must be positive")
    if valid_mask is None:
        available = torch.ones_like(logits, dtype=torch.bool)
    else:
        if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool and match Plackett-Luce logits")
        available = valid_mask.clone()
        if bool((available.sum(dim=-1) < ordered_indices.shape[-1]).any().item()):
            raise ValueError("valid native patch count is smaller than ordered sample")
    # Route logits are produced under AMP in the real training path.  Summing
    # K ordered log-probabilities in fp16 can lose precision before the later
    # temporal reduction, so compute the likelihood in fp32 while preserving
    # float64 known-answer tests.
    likelihood_dtype = (
        torch.float32
        if logits.dtype in {torch.float16, torch.bfloat16}
        else logits.dtype
    )
    result = torch.zeros(
        logits.shape[:2],
        device=logits.device,
        dtype=likelihood_dtype,
    )
    scaled = logits.to(dtype=likelihood_dtype) / float(temperature)
    for slot in range(ordered_indices.shape[-1]):
        choice = ordered_indices[..., slot : slot + 1]
        candidate_logits = scaled.masked_fill(~available, float("-inf"))
        result = result + F.log_softmax(candidate_logits, dim=-1).gather(-1, choice).squeeze(-1)
        if bool((~available.gather(-1, choice)).any().item()):
            raise ValueError("ordered Plackett-Luce indices contain a duplicate")
        available = available.scatter(-1, choice, False)
    return result


def _soft_topk_gate(scores: torch.Tensor, *, count: int, temperature: float) -> torch.Tensor:
    if count <= 0:
        return torch.zeros_like(scores)
    if count >= scores.shape[-1]:
        return torch.ones_like(scores)
    threshold = torch.topk(scores, k=count, dim=-1).values[..., -1:].detach()
    return torch.sigmoid((scores - threshold) / float(temperature))


def _stateless_random_scores(
    logits: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Return a reproducible, data-independent random-ranking control.

    The control must not accidentally depend on a worker's global RNG state
    or on the content/label of a window.  The integer mixing function assigns
    every temporal-token lattice position one deterministic pseudo-random
    rank.  It is deliberately only a control, never a learned score.
    """

    if logits.ndim != 3:
        raise ValueError("stateless random scores require [B,T,N] logits")
    batch, tubelets, item_count = map(int, logits.shape)
    batch_index = torch.arange(batch, device=logits.device, dtype=torch.int64).view(batch, 1, 1)
    time_index = torch.arange(tubelets, device=logits.device, dtype=torch.int64).view(1, tubelets, 1)
    token_index = torch.arange(item_count, device=logits.device, dtype=torch.int64).view(1, 1, item_count)
    # A 64-bit integer hash with no mutable generator state.  Modulo 2**31-1
    # keeps the conversion to float32 exact enough for a total ordering.
    mixed = (
        (batch_index + 1) * 1_103_515_245
        + (time_index + 1) * 12_345
        + (token_index + 1) * 2_654_435_761
        + int(seed) * 97_531
    )
    mixed = (mixed ^ (mixed >> 16)) * 2_246_822_519
    mixed = mixed ^ (mixed >> 13)
    return (mixed.remainder(2_147_483_647)).to(dtype=logits.dtype)


class _ImplicitSigmoidBudgetProjection(torch.autograd.Function):
    """Exact-sum sigmoid projection with its implicit threshold gradient."""

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        window_budget: int,
        temperature: float,
    ) -> torch.Tensor:
        flat_scores = scores.reshape(scores.shape[0], -1)
        flat_valid = valid_mask.reshape(valid_mask.shape[0], -1)
        work_dtype = (
            torch.float64
            if scores.dtype == torch.float64
            else torch.float32
        )
        work = flat_scores.to(dtype=work_dtype)
        valid_count = flat_valid.sum(dim=-1)
        masked_min = work.masked_fill(~flat_valid, float("inf")).amin(dim=-1)
        masked_max = work.masked_fill(~flat_valid, float("-inf")).amax(dim=-1)
        tau = float(temperature)
        lower = masked_min - 32.0 * tau
        upper = masked_max + 32.0 * tau
        epsilon = torch.finfo(work_dtype).eps

        # The threshold is monotone: increasing lambda decreases total mass.
        # Eighty iterations are inexpensive relative to the heavy path and make
        # the residual negligible in both float32 production and float64 KATs.
        with torch.no_grad():
            for _ in range(80):
                midpoint = 0.5 * (lower + upper)
                probability = torch.sigmoid(
                    (work - midpoint[:, None]) / tau
                ).clamp(min=epsilon, max=1.0 - epsilon)
                mass = probability.masked_fill(~flat_valid, 0.0).sum(dim=-1)
                lower = torch.where(
                    mass > float(window_budget),
                    midpoint,
                    lower,
                )
                upper = torch.where(
                    mass > float(window_budget),
                    upper,
                    midpoint,
                )
            threshold = 0.5 * (lower + upper)
            probability = torch.sigmoid(
                (work - threshold[:, None]) / tau
            ).clamp(min=epsilon, max=1.0 - epsilon)
            probability = probability.masked_fill(~flat_valid, 0.0)

        probability = probability.to(dtype=scores.dtype).reshape_as(scores)
        ctx.save_for_backward(probability, valid_mask)
        ctx.temperature = tau
        return probability

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        probability, valid_mask = ctx.saved_tensors
        valid = valid_mask.to(dtype=probability.dtype)
        weight = probability * (1.0 - probability) * valid
        flat_weight = weight.reshape(weight.shape[0], -1)
        flat_gradient = grad_output.reshape(grad_output.shape[0], -1)
        denominator = flat_weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probability.dtype).tiny
        )
        implicit_threshold_gradient = (
            flat_gradient * flat_weight
        ).sum(dim=-1, keepdim=True) / denominator
        grad_scores = (
            flat_weight
            * (flat_gradient - implicit_threshold_gradient)
            / float(ctx.temperature)
        ).reshape_as(probability)
        return grad_scores, None, None, None


def global_sigmoid_budget_projection(
    scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    window_budget: int,
    temperature: float,
) -> torch.Tensor:
    """Project valid global candidate scores to strict probabilities summing B.

    Invalid candidates receive exactly zero.  For every window, valid
    probabilities are strictly between zero and one and their sum matches the
    configured budget within the floating-point tolerance implied by the output
    dtype.  The custom backward differentiates the implicitly solved threshold,
    so a global shift of all candidate utilities has zero effect.
    """

    if scores.ndim != 3:
        raise ValueError("global budget scores must be [B,T,N]")
    if not scores.is_floating_point():
        raise TypeError("global budget scores must be floating point")
    if valid_mask.shape != scores.shape or valid_mask.dtype != torch.bool:
        raise ValueError("global budget valid_mask must be bool [B,T,N]")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("global budget scores must be finite")
    if isinstance(window_budget, bool) or int(window_budget) != window_budget:
        raise ValueError("window token budget must be one positive integer")
    budget = int(window_budget)
    if budget <= 0:
        raise ValueError("window token budget must be one positive integer")
    if float(temperature) <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("global soft-budget temperature must be positive and finite")
    valid_counts = valid_mask.reshape(valid_mask.shape[0], -1).sum(dim=-1)
    if bool((valid_counts <= budget).any().item()):
        raise ValueError(
            "strict global soft-budget projection requires B smaller than valid capacity"
        )

    # AMP callers may supply FP16/BF16.  Solving in at least FP32 is required
    # not only for mass accuracy: casting ``1 - float32_eps`` to a low-precision
    # dtype can round back to exactly one and violate the strict-probability
    # contract.  Autograd carries gradients through this promotion to the
    # original score dtype.
    projection_scores = (
        scores.float()
        if scores.dtype in {torch.float16, torch.bfloat16}
        else scores
    )
    probability = _ImplicitSigmoidBudgetProjection.apply(
        projection_scores,
        valid_mask,
        budget,
        float(temperature),
    )
    valid_probability = probability.masked_select(valid_mask)
    if not bool(((valid_probability > 0.0) & (valid_probability < 1.0)).all().item()):
        raise FloatingPointError("global soft-budget probabilities must be strict")
    observed = probability.reshape(probability.shape[0], -1).sum(dim=-1)
    tolerance = max(
        1e-5,
        float(probability.numel() // probability.shape[0])
        * torch.finfo(probability.dtype).eps
        * 4.0,
    )
    if not bool(
        torch.allclose(
            observed,
            observed.new_full(observed.shape, float(budget)),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise FloatingPointError(
            "global soft-budget projection did not preserve its exact-sum contract"
        )
    return probability


def select_dynamic_global_exact_budget(
    *,
    q_base: torch.Tensor,
    delta_roi: torch.Tensor,
    delta_residual: torch.Tensor,
    window_budget: int,
    training: bool,
    estimator: str,
    temperature: float,
    valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Select one unique physical-token union under a global window budget.

    ``K_t`` and context/ROI/residual role counts are induced by the hard union.
    The forward route is a stable physical top-B.  During training, the
    selected-token gate is exactly one in value and uses the approved LSE plus
    implicit global soft-budget projection in backward.
    """

    if estimator not in {"none", "straight_through"}:
        raise ValueError(
            "dynamic SCNR supports none or straight_through; PL is a separate ablation"
        )
    if training and estimator != "straight_through":
        raise ValueError("dynamic SCNR training requires straight_through")
    if q_base.ndim != 3:
        raise ValueError("dynamic SCNR utilities must be [B,T,N]")
    if delta_roi.shape != q_base.shape or delta_residual.shape != q_base.shape:
        raise ValueError("dynamic SCNR utility fields must share [B,T,N]")
    if valid_mask.shape != q_base.shape or valid_mask.dtype != torch.bool:
        raise ValueError("dynamic SCNR valid_mask must be bool [B,T,N]")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (q_base, delta_roi, delta_residual)
    ):
        raise ValueError("dynamic SCNR utilities must be finite")
    if float(temperature) <= 0.0 or not math.isfinite(float(temperature)):
        raise ValueError("dynamic SCNR temperature must be positive and finite")
    if isinstance(window_budget, bool) or int(window_budget) != window_budget:
        raise ValueError("dynamic SCNR window budget must be one positive integer")
    budget = int(window_budget)
    batch_size, tubelets, item_count = map(int, q_base.shape)
    valid_counts = valid_mask.reshape(batch_size, -1).sum(dim=-1)
    if budget <= 0 or bool((valid_counts <= budget).any().item()):
        raise ValueError(
            "dynamic SCNR requires 0 < window budget < valid physical capacity"
        )

    zero_modifier = torch.zeros_like(q_base)
    modifiers = torch.stack(
        (zero_modifier, delta_roi, delta_residual),
        dim=-1,
    )
    hard_modifier, role_lattice = modifiers.max(dim=-1)
    hard_utility = q_base + hard_modifier
    soft_utility = q_base + float(temperature) * torch.logsumexp(
        modifiers / float(temperature),
        dim=-1,
    )
    flat_hard = hard_utility.masked_fill(~valid_mask, float("-inf")).reshape(
        batch_size,
        -1,
    )
    ordered_physical_indices = _stable_argsort_descending(flat_hard)[..., :budget]
    physical_indices = torch.sort(ordered_physical_indices, dim=-1).values
    selected_mask_flat = torch.zeros_like(flat_hard, dtype=torch.bool).scatter(
        -1,
        physical_indices,
        True,
    )
    selected_mask = selected_mask_flat.reshape_as(valid_mask)
    if not bool((selected_mask_flat.sum(dim=-1) == budget).all().item()):
        raise RuntimeError("dynamic SCNR did not produce exact global B")
    if not bool((selected_mask <= valid_mask).all().item()):
        raise RuntimeError("dynamic SCNR selected invalid physical support")

    tubelet_indices = torch.div(
        physical_indices,
        item_count,
        rounding_mode="floor",
    )
    spatial_indices = physical_indices.remainder(item_count)
    selected_role_ids = role_lattice.reshape(batch_size, -1).gather(
        -1,
        physical_indices,
    )
    k_per_tubelet = selected_mask.sum(dim=-1)
    role_counts_per_window = torch.stack(
        tuple((selected_role_ids == role_id).sum(dim=-1) for role_id in range(3)),
        dim=-1,
    )
    if not bool(
        (role_counts_per_window.sum(dim=-1) == budget).all().item()
    ):
        raise RuntimeError("dynamic SCNR selected token lacks one operational role")

    soft_probability = None
    selected_surrogate = torch.ones_like(hard_utility.reshape(batch_size, -1).gather(-1, physical_indices))
    if training:
        soft_probability = global_sigmoid_budget_projection(
            soft_utility,
            valid_mask=valid_mask,
            window_budget=budget,
            temperature=temperature,
        )
        selected_surrogate = soft_probability.reshape(batch_size, -1).gather(
            -1,
            physical_indices,
        )
        st_gate = torch.ones_like(selected_surrogate) + (
            selected_surrogate - selected_surrogate.detach()
        )
    else:
        st_gate = torch.ones_like(selected_surrogate)

    selected_utility = hard_utility.reshape(batch_size, -1).gather(
        -1,
        physical_indices,
    )
    aggregate_role_counts = role_counts_per_window.sum(dim=0)
    return {
        "schema_version": GEOROUTE_DYNAMIC_ROUTING_SCHEMA,
        "mode": "dynamic_scnr",
        "physical_indices": physical_indices,
        "ordered_physical_indices": ordered_physical_indices,
        "tubelet_indices": tubelet_indices,
        "spatial_indices": spatial_indices,
        "selected_mask": selected_mask,
        "selected_role_ids": selected_role_ids,
        "role_id_values": {"context": 0, "roi": 1, "residual": 2},
        "role_counts_per_window": role_counts_per_window,
        "role_counts": {
            "context": int(aggregate_role_counts[0].item()),
            "roi": int(aggregate_role_counts[1].item()),
            "residual": int(aggregate_role_counts[2].item()),
        },
        "k_per_tubelet": k_per_tubelet,
        "hard_utility": hard_utility,
        "soft_utility": soft_utility,
        "soft_probability": soft_probability,
        "selected_surrogate": selected_surrogate,
        "selected_aggregation_logits": selected_utility,
        "st_gate": st_gate,
        "window_budget": budget,
        "target_k": None,
        "item_count": item_count,
        "tubelet_count": tubelets,
        "valid_physical_count_min": int(valid_counts.min().item()),
        "valid_physical_count_max": int(valid_counts.max().item()),
        "padded_token_count": 0,
    }


def select_fixed_quota_structured_exact_k(
    *,
    roi_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    mode: str,
    context_tokens: int,
    roi_tokens: int,
    residual_tokens: int,
    training: bool,
    estimator: str,
    temperature: float,
    valid_mask: torch.Tensor,
    study_seed: int,
    successful_update_index: int | None,
    distributed_rank: int,
) -> dict[str, Any]:
    """Select a deterministic-context/ROI/residual exact-K structured route.

    The two learned roles form the explicit joint policy
    ``p(roi | context) * p(residual | context, roi)``.  Residual logits do not
    consume sampled ROI features; conditioning is solely the hard exclusion of
    deterministic context and the complete sampled ROI set.  This function is
    intentionally separate from :func:`select_exact_k`, whose legacy behavior
    and likelihood boundary remain unchanged.
    """

    if mode not in STRUCTURED_ROUTE_MODES:
        raise ValueError(f"unsupported structured GeoRoute mode {mode!r}")
    if estimator not in POLICY_ESTIMATORS:
        raise ValueError(f"unsupported policy estimator {estimator!r}")
    if roi_logits.shape != residual_logits.shape or roi_logits.ndim != 3:
        raise ValueError("structured ROI and residual logits must match [B,T,N]")
    if valid_mask.shape != roi_logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("structured valid_mask must be bool and match [B,T,N]")
    if not bool(torch.isfinite(roi_logits).all().item()) or not bool(
        torch.isfinite(residual_logits).all().item()
    ):
        raise ValueError("structured routing logits must be finite")
    counts = {
        "context": int(context_tokens),
        "roi": int(roi_tokens),
        "residual": int(residual_tokens),
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("structured role quotas must be non-negative")
    target_k = sum(counts.values())
    if target_k <= 0:
        raise ValueError("structured exact-K route requires a positive total quota")
    expected_zero_role = {
        "structured_context_residual": "roi",
        "structured_context_roi": "residual",
    }.get(mode)
    if expected_zero_role is not None and counts[expected_zero_role] != 0:
        raise ValueError(f"{mode} requires zero {expected_zero_role} quota")
    if mode in {"structured_hybrid", "structured_hybrid_geometry_shift"} and (
        counts["roi"] <= 0 or counts["residual"] <= 0
    ):
        raise ValueError("structured hybrid requires positive ROI and residual quotas")
    learned_count = counts["roi"] + counts["residual"]
    if learned_count <= 0:
        raise ValueError("structured route requires at least one learned-role token")
    if float(temperature) <= 0.0:
        raise ValueError("structured learned routing requires positive temperature")
    if training and estimator == "none":
        raise ValueError("structured learned routing requires an estimator in training")
    if training and estimator == "score_function" and successful_update_index is None:
        raise ValueError("structured PL requires the successful-update index")
    if int(distributed_rank) < 0:
        raise ValueError("structured route rank must be non-negative")

    batch, tubelets, item_count = map(int, roi_logits.shape)
    valid_counts = valid_mask.sum(dim=-1)
    if bool((valid_counts < target_k).any().item()):
        raise ValueError("valid native support is smaller than structured exact-K")

    excluded = ~valid_mask
    role_indices: dict[str, torch.Tensor] = {}
    parts: list[torch.Tensor] = []
    soft_membership = torch.zeros_like(roi_logits)
    aggregation_logits = torch.zeros_like(roi_logits)
    branch_log_probabilities: dict[str, torch.Tensor] = {}
    route_rng_seeds: dict[str, int] = {}

    if counts["context"]:
        context = _deterministic_uniform_valid_indices(
            valid_mask,
            select_count=counts["context"],
        )
        role_indices["context"] = context
        parts.append(context)
        excluded = excluded | _mask_from_indices(context, item_count)
    else:
        role_indices["context"] = torch.empty(
            (batch, tubelets, 0),
            device=roi_logits.device,
            dtype=torch.long,
        )

    for role, logits in (("roi", roi_logits), ("residual", residual_logits)):
        count = counts[role]
        if count == 0:
            indices = torch.empty(
                (batch, tubelets, 0),
                device=logits.device,
                dtype=torch.long,
            )
            log_probability = torch.zeros(
                (batch, tubelets),
                device=logits.device,
                dtype=torch.float32,
            )
        elif training and estimator == "score_function":
            assert successful_update_index is not None
            private_seed = _route_private_seed(
                study_seed=study_seed,
                successful_update_index=successful_update_index,
                distributed_rank=distributed_rank,
                role_id=role,
            )
            route_rng_seeds[role] = private_seed
            indices, log_probability = sample_ordered_pl_with_exclusion(
                logits,
                count=count,
                temperature=temperature,
                valid_mask=valid_mask,
                excluded_mask=excluded,
                generator=_route_private_generator(
                    device=logits.device,
                    seed=private_seed,
                ),
            )
        else:
            indices = _topk_excluding(
                logits,
                count=count,
                excluded=excluded,
            )
            log_probability = torch.zeros(
                (batch, tubelets),
                device=logits.device,
                dtype=torch.float32,
            )

        role_indices[role] = indices
        branch_log_probabilities[role] = log_probability
        if count:
            parts.append(indices)
            available_scores = logits.masked_fill(excluded, float("-inf"))
            soft_membership = soft_membership + _soft_topk_gate(
                available_scores,
                count=count,
                temperature=temperature,
            ).masked_fill(excluded, 0.0)
            role_mask = _mask_from_indices(indices, item_count)
            aggregation_logits = torch.where(role_mask, logits, aggregation_logits)
            # The residual policy is conditional on the complete sampled ROI
            # set, not on the ROI sampling order.  Updating exclusion only after
            # the full role draw makes that factorization explicit.
            excluded = excluded | role_mask

    ordered = torch.cat(parts, dim=-1)
    joint_log_probability = None
    if training and estimator == "score_function":
        joint_log_probability = (
            branch_log_probabilities["roi"]
            + branch_log_probabilities["residual"]
        )

    if ordered.shape != (batch, tubelets, target_k):
        raise RuntimeError("structured exact-K route produced an invalid shape")
    selected_mask = _mask_from_indices(ordered, item_count)
    if not bool((selected_mask.sum(dim=-1) == target_k).all().item()):
        raise RuntimeError("structured exact-K route contains duplicate indices")
    if not bool((selected_mask <= valid_mask).all().item()):
        raise RuntimeError("structured exact-K route selected invalid support")

    indices = torch.sort(ordered, dim=-1).values
    role_id_values = {"context": 0, "roi": 1, "residual": 2}
    role_id_lattice = torch.full(
        (batch, tubelets, item_count),
        -1,
        device=indices.device,
        dtype=torch.long,
    )
    for role, role_id in role_id_values.items():
        role_id_lattice = role_id_lattice.scatter(
            -1,
            role_indices[role],
            int(role_id),
        )
    role_ids = role_id_lattice.gather(-1, indices)
    if bool((role_ids < 0).any().item()):
        raise RuntimeError("structured exact-K route lacks a role id")
    selected_surrogate = soft_membership.gather(-1, indices)
    selected_aggregation_logits = aggregation_logits.gather(-1, indices)
    if training and estimator == "straight_through":
        st_gate = torch.ones_like(selected_surrogate) + (
            selected_surrogate - selected_surrogate.detach()
        )
    else:
        st_gate = torch.ones_like(selected_surrogate)

    return {
        "schema_version": GEOROUTE_STRUCTURED_ROUTING_SCHEMA,
        "mode": mode,
        "indices": indices,
        "ordered_indices": ordered,
        "role_indices": role_indices,
        "role_ids": role_ids,
        "role_id_values": role_id_values,
        "selected_mask": selected_mask,
        "selected_surrogate": selected_surrogate,
        "soft_membership": soft_membership,
        "selected_aggregation_logits": selected_aggregation_logits,
        "st_gate": st_gate,
        "ordered_log_prob": joint_log_probability,
        "branch_log_probabilities": branch_log_probabilities,
        "target_k": target_k,
        "item_count": item_count,
        "valid_patch_count_min": int(valid_counts.min().item()),
        "valid_patch_count_max": int(valid_counts.max().item()),
        "role_counts": {
            "context": counts["context"],
            "roi": counts["roi"],
            "residual": counts["residual"],
            "free": 0,
            "dense": 0,
            "uniform": 0,
            "random": 0,
        },
        "route_rng": {
            "schema_version": _ROUTE_PRIVATE_RNG_SCHEMA,
            "enabled": bool(training and estimator == "score_function"),
            "study_seed": int(study_seed),
            "successful_update_index": (
                None
                if successful_update_index is None
                else int(successful_update_index)
            ),
            "distributed_rank": int(distributed_rank),
            "role_seeds": route_rng_seeds,
            "global_rng_consumed": False,
        },
    }


def select_exact_k(
    *,
    roi_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    mode: str,
    tokens_per_tubelet: int,
    context_tokens: int,
    roi_fraction: float,
    training: bool,
    estimator: str,
    temperature: float,
    valid_mask: torch.Tensor,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Produce a no-duplicate exact-K route for every temporal tubelet.

    ``hybrid`` reserves deterministic context tokens, then chooses ROI tokens
    and residual tokens from the remaining native-patch lattice.  A
    score-function estimator is intentionally allowed only for single-family
    free/ROI routes; treating the staged hybrid route as a single categorical
    draw would make its likelihood claim false.
    """

    if mode not in LEGACY_ROUTE_MODES:
        raise ValueError(f"unsupported GeoRoute mode {mode!r}")
    if estimator not in POLICY_ESTIMATORS:
        raise ValueError(f"unsupported policy estimator {estimator!r}")
    if roi_logits.shape != residual_logits.shape or roi_logits.ndim != 3:
        raise ValueError("ROI and residual logits must match [B,T,N]")
    if valid_mask.shape != roi_logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool and match routing logits [B,T,N]")
    if not bool(torch.isfinite(roi_logits).all().item()) or not bool(torch.isfinite(residual_logits).all().item()):
        raise ValueError("routing logits must be finite")
    batch, tubelets, item_count = map(int, roi_logits.shape)
    if item_count <= 0:
        raise ValueError("routing requires at least one native patch per tubelet")
    valid_counts = valid_mask.sum(dim=-1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("routing requires at least one valid native patch per tubelet")
    if mode == "dense":
        if not torch.equal(valid_counts, valid_counts.reshape(-1)[:1].expand_as(valid_counts)):
            raise ValueError("dense routing requires equal valid patch counts across batch/time")
        target_k = int(valid_counts.reshape(-1)[0].item())
    else:
        target_k = int(tokens_per_tubelet)
    if not (0 < target_k <= item_count) or bool((valid_counts < target_k).any().item()):
        raise ValueError("valid native patch count is smaller than exact-K")
    if not (0 <= int(context_tokens) < target_k or (mode == "dense" and context_tokens == 0)):
        raise ValueError("context_tokens must lie in [0,K)")
    if not (0.0 <= float(roi_fraction) <= 1.0):
        raise ValueError("roi_fraction must lie in [0,1]")
    if mode in {"roi", "free", "hybrid"} and float(temperature) <= 0.0:
        raise ValueError("learned GeoRoute modes require a positive temperature")
    if estimator == "score_function" and mode not in {"roi", "free"}:
        raise ValueError("score_function is only valid for single-family roi/free routes")
    if estimator == "none" and mode in {"roi", "free", "hybrid"} and training:
        raise ValueError("learned GeoRoute modes require an explicit gradient estimator during training")

    device = roi_logits.device
    masked_roi_logits = roi_logits.masked_fill(~valid_mask, float("-inf"))
    masked_residual_logits = residual_logits.masked_fill(~valid_mask, float("-inf"))
    role_counts = {
        "context": 0,
        "roi": 0,
        "residual": 0,
        "free": 0,
        "dense": 0,
        "uniform": 0,
        "random": 0,
    }
    ordered_log_prob = None

    if mode == "dense":
        ordered = _deterministic_uniform_valid_indices(
            valid_mask,
            select_count=target_k,
        )
        surrogate = torch.ones_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
        role_counts["dense"] = target_k
    elif mode == "uniform":
        ordered = _deterministic_uniform_valid_indices(
            valid_mask,
            select_count=target_k,
        )
        surrogate = torch.zeros_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
        role_counts["uniform"] = target_k
    elif mode == "random":
        ordered = _stable_argsort_descending(
            _stateless_random_scores(roi_logits, seed=random_seed).masked_fill(
                ~valid_mask,
                float("-inf"),
            )
        )[..., :target_k]
        surrogate = torch.zeros_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
        role_counts["random"] = target_k
    elif mode in {"roi", "free"}:
        logits = masked_roi_logits if mode == "roi" else masked_residual_logits
        if training and estimator == "score_function":
            ordered = _sample_plackett_luce_order(logits, count=target_k, temperature=temperature)
            ordered_log_prob = ordered_plackett_luce_log_prob(
                logits,
                ordered,
                temperature=temperature,
                valid_mask=valid_mask,
            )
        else:
            ordered = _stable_argsort_descending(logits)[..., :target_k]
        surrogate = _soft_topk_gate(
            logits,
            count=target_k,
            temperature=temperature,
        ).masked_fill(~valid_mask, 0.0)
        aggregation_logits = logits
        role_counts[mode] = target_k
    else:  # hybrid
        context_count = min(int(context_tokens), target_k)
        remainder = target_k - context_count
        roi_count = int(round(float(roi_fraction) * remainder))
        roi_count = min(max(roi_count, 0), remainder)
        residual_count = remainder - roi_count
        parts = []
        excluded = ~valid_mask
        surrogate = torch.zeros_like(roi_logits)
        if context_count:
            context = _deterministic_uniform_valid_indices(
                valid_mask,
                select_count=context_count,
            )
            parts.append(context)
            excluded = excluded | _mask_from_indices(context, item_count)
        roi_scores = masked_roi_logits.masked_fill(excluded, float("-inf"))
        roi_indices = _topk_excluding(
            masked_roi_logits,
            count=roi_count,
            excluded=excluded,
        )
        if roi_count:
            parts.append(roi_indices)
            surrogate = surrogate + _soft_topk_gate(
                roi_scores,
                count=roi_count,
                temperature=temperature,
            ).masked_fill(excluded, 0.0)
            excluded = excluded | _mask_from_indices(roi_indices, item_count)
        residual_scores = masked_residual_logits.masked_fill(excluded, float("-inf"))
        residual_indices = _topk_excluding(
            masked_residual_logits,
            count=residual_count,
            excluded=excluded,
        )
        if residual_count:
            parts.append(residual_indices)
            surrogate = surrogate + _soft_topk_gate(
                residual_scores,
                count=residual_count,
                temperature=temperature,
            ).masked_fill(excluded, 0.0)
        ordered = torch.cat(parts, dim=-1)
        aggregation_logits = masked_roi_logits + masked_residual_logits
        role_counts.update(context=context_count, roi=roi_count, residual=residual_count)

    if ordered.shape != (batch, tubelets, target_k):
        raise RuntimeError("exact-K route produced an invalid selection shape")
    selected_mask = _mask_from_indices(ordered, item_count)
    if not bool((selected_mask.sum(dim=-1) == target_k).all().item()):
        raise RuntimeError("exact-K route contains duplicate native patch indices")
    if not bool((selected_mask <= valid_mask).all().item()):
        raise RuntimeError("exact-K route selected an invalid native patch")

    # Packed execution does not care about policy ranking order. Sorting makes
    # gather/scatter deterministic and preserves row-major native-token layout.
    indices = torch.sort(ordered, dim=-1).values
    selected_surrogate = surrogate.gather(-1, indices)
    selected_aggregation_logits = aggregation_logits.gather(-1, indices)
    if training and estimator == "straight_through" and mode in {"roi", "free", "hybrid"}:
        # Subtract first so hard forward values are bitwise one; the previous
        # left-associated form could retain a floating-point roundoff residue.
        st_gate = torch.ones_like(selected_surrogate) + (
            selected_surrogate - selected_surrogate.detach()
        )
    else:
        st_gate = torch.ones_like(selected_surrogate)

    return {
        "schema_version": GEOROUTE_ROUTING_SCHEMA,
        "mode": mode,
        "indices": indices,
        "ordered_indices": ordered,
        "selected_mask": selected_mask,
        "selected_surrogate": selected_surrogate,
        "soft_membership": surrogate,
        "selected_aggregation_logits": selected_aggregation_logits,
        "st_gate": st_gate,
        "ordered_log_prob": ordered_log_prob,
        "target_k": target_k,
        "item_count": item_count,
        "valid_patch_count_min": int(valid_counts.min().item()),
        "valid_patch_count_max": int(valid_counts.max().item()),
        "role_counts": role_counts,
    }


def score_function_policy_loss(
    *,
    detector_cost: torch.Tensor,
    ordered_log_prob: torch.Tensor,
    baseline: torch.Tensor,
    weight: float,
    temporal_reduction: str = "sum",
) -> torch.Tensor:
    """Minimization objective whose gradient is the REINFORCE risk gradient.

    For the detector risk ``J = E[L_det]``, the log-derivative identity is
    ``grad J = E[(L_det - b) grad log p]``.  Optimizers minimize the returned
    scalar, so the loss must carry the **positive** advantage times log
    probability.  A leading minus sign would reward routes with higher
    detector loss and silently reverse the policy update.

    ``temporal_reduction="sum"`` retains the historical joint-window
    objective.  ``"mean"`` is the explicit per-tubelet normalization used by
    the repaired estimator: it preserves every ordered exact-K
    Plackett-Luce likelihood and the policy-gradient direction while removing
    the otherwise linear dependence of gradient scale on window length.
    """

    if detector_cost.ndim != 0 or not bool(torch.isfinite(detector_cost).item()):
        raise ValueError("detector_cost must be one finite scalar")
    if ordered_log_prob.ndim != 2 or not bool(torch.isfinite(ordered_log_prob).all().item()):
        raise ValueError("ordered_log_prob must be finite [B,T]")
    if baseline.ndim != 0 or not bool(torch.isfinite(baseline).item()):
        raise ValueError("baseline must be one finite scalar")
    if float(weight) < 0.0:
        raise ValueError("score-function weight must be non-negative")
    if temporal_reduction not in SCORE_FUNCTION_TEMPORAL_REDUCTIONS:
        raise ValueError(
            "score-function temporal reduction must be 'sum' or 'mean'"
        )
    # The production route has hundreds of temporal tubelets.  Even when each
    # per-tubelet log-probability is finite, the fp16 temporal sum can exceed
    # 65504 and become -inf before GradScaler can reduce its scale.  Upcasting
    # changes only numerical evaluation, not the registered sum-then-batch-mean
    # objective or its gradient direction.
    advantage = detector_cost.detach().to(torch.float32) - baseline.detach().to(
        torch.float32
    )
    ordered_log_prob = ordered_log_prob.to(torch.float32)
    if temporal_reduction == "sum":
        joint_log_probability = ordered_log_prob.sum(dim=1)
    else:
        joint_log_probability = ordered_log_prob.mean(dim=1)
    return float(weight) * advantage * joint_log_probability.mean()
