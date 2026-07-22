"""Pure routing primitives for GeoRoute-AdaTAD.

The module deliberately separates three objects which are often conflated in
vision routing implementations:

* the hard, exact-K native-patch route used by the forward pass;
* the straight-through (ST) content gate, which is a biased surrogate; and
* the Plackett-Luce score-function log-probability for stochastic hard routes.

It contains no dataset, detector, ground-truth, teacher, or evaluation code.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


GEOROUTE_ROUTING_SCHEMA = "georoute_native_routing_v1"
ROUTE_MODES = frozenset({"dense", "uniform", "random", "free", "roi", "hybrid"})
POLICY_ESTIMATORS = frozenset({"none", "straight_through", "score_function"})


def decode_continuous_geometry(
    logits: torch.Tensor,
    *,
    min_extent: float,
    max_extent: float,
) -> torch.Tensor:
    """Decode unconstrained logits to in-bounds ``(cx, cy, w, h)`` boxes.

    Centres are parameterized *after* width/height are decoded, so every box
    remains in the normalized source frame without a non-differentiable clamp.
    """

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("geometry logits must be [B,T,4]")
    if not (0.0 < float(min_extent) <= float(max_extent) <= 1.0):
        raise ValueError("geometry extents must satisfy 0 < min <= max <= 1")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("geometry logits must be finite")

    extent = float(min_extent) + (float(max_extent) - float(min_extent)) * torch.sigmoid(logits[..., 2:])
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


def ordered_plackett_luce_log_prob(
    logits: torch.Tensor,
    ordered_indices: torch.Tensor,
    *,
    temperature: float,
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
    available = torch.ones_like(logits, dtype=torch.bool)
    result = torch.zeros(logits.shape[:2], device=logits.device, dtype=logits.dtype)
    scaled = logits / float(temperature)
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
    random_seed: int = 0,
) -> dict[str, Any]:
    """Produce a no-duplicate exact-K route for every temporal tubelet.

    ``hybrid`` reserves deterministic context tokens, then chooses ROI tokens
    and residual tokens from the remaining native-patch lattice.  A
    score-function estimator is intentionally allowed only for single-family
    free/ROI routes; treating the staged hybrid route as a single categorical
    draw would make its likelihood claim false.
    """

    if mode not in ROUTE_MODES:
        raise ValueError(f"unsupported GeoRoute mode {mode!r}")
    if estimator not in POLICY_ESTIMATORS:
        raise ValueError(f"unsupported policy estimator {estimator!r}")
    if roi_logits.shape != residual_logits.shape or roi_logits.ndim != 3:
        raise ValueError("ROI and residual logits must match [B,T,N]")
    if not bool(torch.isfinite(roi_logits).all().item()) or not bool(torch.isfinite(residual_logits).all().item()):
        raise ValueError("routing logits must be finite")
    batch, tubelets, item_count = map(int, roi_logits.shape)
    if item_count <= 0:
        raise ValueError("routing requires at least one native patch per tubelet")
    if mode == "dense":
        target_k = item_count
    else:
        target_k = int(tokens_per_tubelet)
    if not (0 < target_k <= item_count):
        raise ValueError("tokens_per_tubelet must lie in [1, native_patch_count]")
    if not (0 <= int(context_tokens) < target_k or (mode == "dense" and context_tokens == 0)):
        raise ValueError("context_tokens must lie in [0,K)")
    if not (0.0 <= float(roi_fraction) <= 1.0):
        raise ValueError("roi_fraction must lie in [0,1]")
    if estimator == "score_function" and mode not in {"roi", "free"}:
        raise ValueError("score_function is only valid for single-family roi/free routes")
    if estimator == "none" and mode in {"roi", "free", "hybrid"} and training:
        raise ValueError("learned GeoRoute modes require an explicit gradient estimator during training")

    device = roi_logits.device
    role_counts = {"context": 0, "roi": 0, "residual": 0, "free": 0, "dense": 0}
    ordered_log_prob = None

    if mode == "dense":
        ordered = torch.arange(item_count, device=device, dtype=torch.long).view(1, 1, -1).expand(batch, tubelets, -1)
        surrogate = torch.ones_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
        role_counts["dense"] = item_count
    elif mode == "uniform":
        base = _deterministic_uniform_indices(item_count=item_count, select_count=target_k, device=device)
        ordered = base.view(1, 1, -1).expand(batch, tubelets, -1)
        surrogate = torch.zeros_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
    elif mode == "random":
        ordered = _stable_argsort_descending(
            _stateless_random_scores(roi_logits, seed=random_seed)
        )[..., :target_k]
        surrogate = torch.zeros_like(roi_logits)
        aggregation_logits = torch.zeros_like(roi_logits)
    elif mode in {"roi", "free"}:
        logits = roi_logits if mode == "roi" else residual_logits
        if training and estimator == "score_function":
            ordered = _sample_plackett_luce_order(logits, count=target_k, temperature=temperature)
            ordered_log_prob = ordered_plackett_luce_log_prob(logits, ordered, temperature=temperature)
        else:
            ordered = _stable_argsort_descending(logits)[..., :target_k]
        surrogate = _soft_topk_gate(logits, count=target_k, temperature=temperature)
        aggregation_logits = logits
        role_counts[mode] = target_k
    else:  # hybrid
        context_count = min(int(context_tokens), target_k)
        remainder = target_k - context_count
        roi_count = int(round(float(roi_fraction) * remainder))
        roi_count = min(max(roi_count, 0), remainder)
        residual_count = remainder - roi_count
        parts = []
        excluded = torch.zeros_like(roi_logits, dtype=torch.bool)
        if context_count:
            context = _deterministic_uniform_indices(
                item_count=item_count,
                select_count=context_count,
                device=device,
            ).view(1, 1, -1).expand(batch, tubelets, -1)
            parts.append(context)
            excluded = excluded | _mask_from_indices(context, item_count)
        roi_indices = _topk_excluding(roi_logits, count=roi_count, excluded=excluded)
        if roi_count:
            parts.append(roi_indices)
            excluded = excluded | _mask_from_indices(roi_indices, item_count)
        residual_indices = _topk_excluding(residual_logits, count=residual_count, excluded=excluded)
        if residual_count:
            parts.append(residual_indices)
        ordered = torch.cat(parts, dim=-1)
        surrogate = _soft_topk_gate(
            roi_logits + residual_logits,
            count=target_k,
            temperature=temperature,
        )
        aggregation_logits = roi_logits + residual_logits
        role_counts.update(context=context_count, roi=roi_count, residual=residual_count)

    if ordered.shape != (batch, tubelets, target_k):
        raise RuntimeError("exact-K route produced an invalid selection shape")
    selected_mask = _mask_from_indices(ordered, item_count)
    if not bool((selected_mask.sum(dim=-1) == target_k).all().item()):
        raise RuntimeError("exact-K route contains duplicate native patch indices")

    # Packed execution does not care about policy ranking order. Sorting makes
    # gather/scatter deterministic and preserves row-major native-token layout.
    indices = torch.sort(ordered, dim=-1).values
    selected_surrogate = surrogate.gather(-1, indices)
    selected_aggregation_logits = aggregation_logits.gather(-1, indices)
    if training and estimator == "straight_through" and mode in {"roi", "free", "hybrid"}:
        st_gate = torch.ones_like(selected_surrogate) + selected_surrogate - selected_surrogate.detach()
    else:
        st_gate = torch.ones_like(selected_surrogate)

    return {
        "schema_version": GEOROUTE_ROUTING_SCHEMA,
        "mode": mode,
        "indices": indices,
        "ordered_indices": ordered,
        "selected_mask": selected_mask,
        "selected_surrogate": selected_surrogate,
        "selected_aggregation_logits": selected_aggregation_logits,
        "st_gate": st_gate,
        "ordered_log_prob": ordered_log_prob,
        "target_k": target_k,
        "item_count": item_count,
        "role_counts": role_counts,
    }


def score_function_policy_loss(
    *,
    detector_cost: torch.Tensor,
    ordered_log_prob: torch.Tensor,
    baseline: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Minimization objective whose gradient is the REINFORCE risk gradient.

    For the detector risk ``J = E[L_det]``, the log-derivative identity is
    ``grad J = E[(L_det - b) grad log p]``.  Optimizers minimize the returned
    scalar, so the loss must carry the **positive** advantage times log
    probability.  A leading minus sign would reward routes with higher
    detector loss and silently reverse the policy update.
    """

    if detector_cost.ndim != 0 or not bool(torch.isfinite(detector_cost).item()):
        raise ValueError("detector_cost must be one finite scalar")
    if ordered_log_prob.ndim != 2 or not bool(torch.isfinite(ordered_log_prob).all().item()):
        raise ValueError("ordered_log_prob must be finite [B,T]")
    if baseline.ndim != 0 or not bool(torch.isfinite(baseline).item()):
        raise ValueError("baseline must be one finite scalar")
    if float(weight) < 0.0:
        raise ValueError("score-function weight must be non-negative")
    advantage = detector_cost.detach() - baseline.detach()
    return float(weight) * advantage * ordered_log_prob.mean()
