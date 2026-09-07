from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


PHASE_ORDER = ("scaffold", "onset", "offset", "core")
DEFAULT_FIXED_QUOTA = {"scaffold": 128, "onset": 64, "offset": 64, "core": 128}
DEFAULT_ADAPTIVE_MINIMA = {"scaffold": 96, "onset": 32, "offset": 32, "core": 64}
DEFAULT_ADAPTIVE_CAPS = {"scaffold": 160, "onset": 96, "offset": 96, "core": 192}


@dataclass(frozen=True)
class PhaseFieldResult:
    core: torch.Tensor
    onset: torch.Tensor
    offset: torch.Tensor
    curvature: torch.Tensor
    smoothed_logit: torch.Tensor
    valid_mask: torch.Tensor
    diagnostics: Dict[str, Any]


def _as_valid(reference: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if reference.ndim != 2:
        raise ValueError(f"phase field input must be [B,T], got {tuple(reference.shape)}")
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if valid_mask.shape != reference.shape:
        raise ValueError("valid_mask must match phase field input")
    valid = valid_mask.to(device=reference.device, dtype=torch.bool)
    if torch.any(valid.long().sum(dim=1) <= 0):
        raise ValueError("each sample must have at least one valid phase token")
    return valid


def gaussian_derivative_kernels(sigma: float, *, device=None, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma).pow(2))
    g = g / g.sum().clamp_min(torch.finfo(dtype).eps)

    d1 = -(x / (sigma * sigma)) * g
    d1 = 0.5 * (d1 - d1.flip(0))
    d1 = d1 - d1.mean()
    d1[radius] = 0.0

    d2 = ((x.pow(2) - sigma * sigma) / (sigma ** 4)) * g
    d2 = 0.5 * (d2 + d2.flip(0))
    d2 = d2 - d2.mean()
    return {"gaussian": g, "first": d1, "second": d2}


def _pad_for_kernel(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return values
    mode = "reflect" if int(values.shape[-1]) > radius else "replicate"
    return F.pad(values, (radius, radius), mode=mode)


def _conv1d_same(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must be [B,T]")
    if kernel.ndim != 1 or int(kernel.numel()) % 2 != 1:
        raise ValueError("kernel must be a 1-D odd-length tensor")
    radius = int(kernel.numel() // 2)
    x = _pad_for_kernel(values[:, None, :], radius)
    weight = kernel.to(device=values.device, dtype=values.dtype).view(1, 1, -1)
    return F.conv1d(x, weight).squeeze(1)


def masked_gaussian_smooth(values: torch.Tensor, valid_mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(device=values.device, dtype=values.dtype)
    weighted = _conv1d_same(values.float() * valid, kernel.float())
    denom = _conv1d_same(valid, kernel.float()).clamp_min(1.0e-6)
    return (weighted / denom).masked_fill(~valid_mask, 0.0)


def robust_normalize(field: torch.Tensor, valid_mask: torch.Tensor, *, quantile: float = 0.95) -> torch.Tensor:
    if field.shape != valid_mask.shape:
        raise ValueError("field and valid_mask must have the same shape")
    quantile = float(quantile)
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must lie in (0, 1]")
    rows = []
    for row, valid in zip(field.float(), valid_mask.bool()):
        active = row[valid].abs()
        if active.numel() == 0:
            rows.append(torch.zeros_like(row))
            continue
        scale = torch.quantile(active.detach(), quantile)
        if float(scale.item()) <= 1.0e-8:
            median = torch.median(active.detach())
            mad = torch.median((active.detach() - median).abs())
            scale = mad * 1.4826
        if float(scale.item()) <= 1.0e-8:
            scale = active.detach().amax()
        if float(scale.item()) <= 1.0e-8:
            rows.append(torch.zeros_like(row))
        else:
            rows.append((row / scale.clamp_min(1.0e-8)).masked_fill(~valid, 0.0))
    return torch.stack(rows, dim=0)


def compute_phase_fields(
    *,
    logits: Optional[torch.Tensor] = None,
    p_action: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    sigmas: Sequence[float] = (1.5, 3.0),
    aggregate: str = "median",
    normalization_quantile: float = 0.95,
) -> PhaseFieldResult:
    if logits is None:
        if p_action is None:
            raise ValueError("logits or p_action must be provided")
        source = torch.logit(p_action.float().clamp(1.0e-6, 1.0 - 1.0e-6))
    else:
        source = logits.float()
    valid = _as_valid(source, valid_mask)
    if not sigmas:
        raise ValueError("at least one Gaussian scale is required")
    if aggregate not in {"median", "capped_max"}:
        raise ValueError("aggregate must be median or capped_max")

    per_scale = {"core": [], "onset": [], "offset": [], "curvature": [], "smoothed_logit": []}
    kernel_summaries = []
    for sigma in sigmas:
        kernels = gaussian_derivative_kernels(float(sigma), device=source.device, dtype=torch.float32)
        smooth = masked_gaussian_smooth(source, valid, kernels["gaussian"]).to(source.dtype)
        d1 = -_conv1d_same(smooth, kernels["first"]).masked_fill(~valid, 0.0)
        d2 = _conv1d_same(smooth, kernels["second"]).masked_fill(~valid, 0.0)
        per_scale["smoothed_logit"].append(smooth)
        per_scale["core"].append(robust_normalize(torch.sigmoid(smooth).masked_fill(~valid, 0.0), valid, quantile=normalization_quantile))
        per_scale["onset"].append(robust_normalize(F.relu(float(sigma) * d1), valid, quantile=normalization_quantile))
        per_scale["offset"].append(robust_normalize(F.relu(-float(sigma) * d1), valid, quantile=normalization_quantile))
        per_scale["curvature"].append(
            robust_normalize((float(sigma) ** 2) * d2.abs(), valid, quantile=normalization_quantile)
        )
        kernel_summaries.append(
            {
                "sigma": float(sigma),
                "kernel_size": int(kernels["gaussian"].numel()),
                "first_derivative_sum_abs": float(kernels["first"].sum().abs().detach().cpu().item()),
                "first_derivative_antisymmetric": bool(
                    torch.allclose(kernels["first"], -kernels["first"].flip(0), atol=1.0e-6)
                ),
                "second_derivative_sum_abs": float(kernels["second"].sum().abs().detach().cpu().item()),
            }
        )

    def combine(name: str) -> torch.Tensor:
        stack = torch.stack(per_scale[name], dim=0)
        if aggregate == "median":
            return stack.median(dim=0).values.masked_fill(~valid, 0.0)
        capped = stack.clamp_max(2.0).amax(dim=0)
        return capped.masked_fill(~valid, 0.0)

    return PhaseFieldResult(
        core=combine("core"),
        onset=combine("onset"),
        offset=combine("offset"),
        curvature=combine("curvature"),
        smoothed_logit=torch.stack(per_scale["smoothed_logit"], dim=0).mean(dim=0).masked_fill(~valid, 0.0),
        valid_mask=valid,
        diagnostics={
            "schema_version": "duca_phase_field_diagnostics_v1",
            "sigmas": [float(sigma) for sigma in sigmas],
            "aggregate": aggregate,
            "normalization": f"valid_region_q{int(float(normalization_quantile) * 100)}_with_mad_fallback",
            "statistics_detached": True,
            "kernels": kernel_summaries,
        },
    )


def _scaled_quota(base: Mapping[str, int], total: int) -> Dict[str, int]:
    total = int(total)
    if total <= 0:
        raise ValueError("total quota must be positive")
    base_sum = sum(int(base[name]) for name in PHASE_ORDER)
    if base_sum <= 0:
        raise ValueError("base quota sum must be positive")
    ideal = {name: float(total) * float(base[name]) / float(base_sum) for name in PHASE_ORDER}
    quota = {name: int(math.floor(ideal[name])) for name in PHASE_ORDER}
    remaining = total - sum(quota.values())
    for name in sorted(PHASE_ORDER, key=lambda item: (-(ideal[item] - quota[item]), PHASE_ORDER.index(item))):
        if remaining <= 0:
            break
        quota[name] += 1
        remaining -= 1
    return quota


def _largest_remainder(
    masses: Mapping[str, float],
    remaining: int,
    caps_remaining: Mapping[str, int],
) -> Dict[str, int]:
    remaining = int(remaining)
    out = {name: 0 for name in PHASE_ORDER}
    if remaining <= 0:
        return out
    positive = {name: max(0.0, float(masses.get(name, 0.0))) for name in PHASE_ORDER}
    mass_sum = sum(positive.values())
    if mass_sum <= 1.0e-12:
        positive = {name: 1.0 for name in PHASE_ORDER}
        mass_sum = float(len(PHASE_ORDER))
    ideal = {name: float(remaining) * positive[name] / mass_sum for name in PHASE_ORDER}
    for name in PHASE_ORDER:
        out[name] = min(int(math.floor(ideal[name])), int(caps_remaining.get(name, remaining)))
    left = remaining - sum(out.values())
    while left > 0:
        candidates = [name for name in PHASE_ORDER if out[name] < int(caps_remaining.get(name, remaining))]
        if not candidates:
            break
        name = sorted(candidates, key=lambda item: (-(ideal[item] - math.floor(ideal[item])), PHASE_ORDER.index(item)))[0]
        out[name] += 1
        left -= 1
    return out


def adaptive_phase_quotas(
    fields: PhaseFieldResult,
    *,
    total_budget: int,
    minima: Optional[Mapping[str, int]] = None,
    caps: Optional[Mapping[str, int]] = None,
) -> list[Dict[str, int]]:
    total_budget = int(total_budget)
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    minima = dict(minima or DEFAULT_ADAPTIVE_MINIMA)
    caps = dict(caps or DEFAULT_ADAPTIVE_CAPS)
    if sum(int(minima[name]) for name in PHASE_ORDER) >= total_budget:
        base = _scaled_quota(minima, total_budget)
        return [dict(base) for _ in range(int(fields.valid_mask.shape[0]))]

    quotas = []
    for row in range(int(fields.valid_mask.shape[0])):
        valid = fields.valid_mask[row]
        target = min(total_budget, int(valid.long().sum().item()))
        if sum(int(minima[name]) for name in PHASE_ORDER) >= target:
            quotas.append(_scaled_quota(minima, target))
            continue
        base = {name: int(minima[name]) for name in PHASE_ORDER}
        scaled_caps = _scaled_quota(caps, total_budget)
        scaled_caps = {name: max(base[name], int(scaled_caps[name])) for name in PHASE_ORDER}
        masses = {
            "scaffold": float(valid.float().sum().detach().cpu().item()),
            "onset": float(fields.onset[row][valid].detach().sum().cpu().item()),
            "offset": float(fields.offset[row][valid].detach().sum().cpu().item()),
            "core": float(fields.core[row][valid].detach().sum().cpu().item()),
        }
        remaining = target - sum(base.values())
        extra = _largest_remainder(
            masses,
            remaining,
            {name: max(0, scaled_caps[name] - base[name]) for name in PHASE_ORDER},
        )
        quotas.append({name: base[name] + extra[name] for name in PHASE_ORDER})
    return quotas


def _positions_tensor(values: Iterable[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), device=device, dtype=torch.long)


def _uniform_positions(valid_positions: torch.Tensor, count: int) -> list[int]:
    count = min(int(count), int(valid_positions.numel()))
    if count <= 0:
        return []
    if count >= int(valid_positions.numel()):
        return [int(item) for item in valid_positions.detach().cpu().tolist()]
    local = torch.linspace(0, int(valid_positions.numel()) - 1, steps=count, device=valid_positions.device).round().long()
    selected = []
    for item in local.detach().cpu().tolist():
        pos = int(valid_positions[int(item)].item())
        if pos not in selected:
            selected.append(pos)
    for item in valid_positions.detach().cpu().tolist():
        if len(selected) >= count:
            break
        pos = int(item)
        if pos not in selected:
            selected.append(pos)
    return sorted(selected[:count])


def _ranked_positions(
    score: torch.Tensor,
    valid_positions: torch.Tensor,
    count: int,
    *,
    excluded: set[int],
    nms_radius: int = 0,
) -> list[int]:
    count = min(int(count), int(valid_positions.numel()) - len(excluded))
    if count <= 0:
        return []
    candidates = []
    for pos in valid_positions.detach().cpu().tolist():
        pos = int(pos)
        if pos in excluded:
            continue
        candidates.append((float(score[pos].detach().cpu().item()), pos))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[int] = []
    for _value, pos in candidates:
        if nms_radius > 0 and any(abs(pos - prev) <= int(nms_radius) for prev in chosen):
            continue
        chosen.append(pos)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for _value, pos in candidates:
            if pos not in chosen:
                chosen.append(pos)
                if len(chosen) >= count:
                    break
    return sorted(chosen)


def select_exact_uniform_positions(valid_mask: torch.Tensor, *, total_budget: int) -> Dict[str, Any]:
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [B,T]")
    valid = valid_mask.to(dtype=torch.bool)
    total_budget = int(total_budget)
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    batch, temporal_len = valid.shape
    rows = []
    masks = []
    for row in range(batch):
        valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
        target = min(total_budget, int(valid_positions.numel()))
        if target <= 0:
            raise ValueError("exact-uniform selector cannot select from an empty valid region")
        selected = _uniform_positions(valid_positions, target)
        row_tensor = torch.full((total_budget,), -1, device=valid.device, dtype=torch.long)
        row_tensor[: len(selected)] = _positions_tensor(selected, device=valid.device)
        dense_mask = torch.zeros((temporal_len,), device=valid.device, dtype=torch.bool)
        dense_mask[_positions_tensor(selected, device=valid.device)] = True
        rows.append(row_tensor)
        masks.append(dense_mask)
    selected_mask = torch.stack(masks, dim=0)
    selected_positions = torch.stack(rows, dim=0)
    return {
        "selected_positions": selected_positions,
        "positions": selected_positions,
        "selected_mask": selected_mask,
        "detector_input_length": selected_mask.long().sum(dim=1),
        "effective_budget": selected_mask.long().sum(dim=1),
        "fill_strategy": ["exact_uniform"] * batch,
    }


def select_phase_positions(
    fields: PhaseFieldResult,
    *,
    total_budget: int = 384,
    quota_mode: str = "adaptive",
    fixed_quota: Optional[Mapping[str, int]] = None,
    adaptive_minima: Optional[Mapping[str, int]] = None,
    adaptive_caps: Optional[Mapping[str, int]] = None,
    temporal_nms_radius: int = 1,
    use_curvature: bool = False,
    curvature_weight: float = 0.05,
) -> Dict[str, Any]:
    total_budget = int(total_budget)
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if quota_mode not in {"fixed", "adaptive"}:
        raise ValueError("quota_mode must be fixed or adaptive")
    valid = fields.valid_mask
    batch, temporal_len = valid.shape
    if quota_mode == "fixed":
        base = _scaled_quota(dict(fixed_quota or DEFAULT_FIXED_QUOTA), total_budget)
        quotas = [dict(base) for _ in range(batch)]
    else:
        quotas = adaptive_phase_quotas(
            fields,
            total_budget=total_budget,
            minima=adaptive_minima,
            caps=adaptive_caps,
        )

    rows = []
    masks = []
    actual_phase_counts = []
    row_diagnostics = []
    global_score = fields.core + 0.5 * fields.onset + 0.5 * fields.offset
    if use_curvature:
        global_score = global_score + float(curvature_weight) * fields.curvature

    for row in range(batch):
        valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
        target = min(total_budget, int(valid_positions.numel()))
        if target <= 0:
            raise ValueError("phase selector cannot select from an empty valid region")
        if int(valid_positions.numel()) <= target:
            selected = [int(item) for item in valid_positions.detach().cpu().tolist()]
            counts = {"scaffold": len(selected), "onset": 0, "offset": 0, "core": 0, "global_refill": 0}
        else:
            quota = dict(quotas[row])
            selected_set: set[int] = set()
            phase_lists: Dict[str, list[int]] = {}
            scaffold = _uniform_positions(valid_positions, quota["scaffold"])
            phase_lists["scaffold"] = scaffold
            selected_set.update(scaffold)

            onset_score = fields.onset[row] + (float(curvature_weight) * fields.curvature[row] if use_curvature else 0.0)
            onset = _ranked_positions(
                onset_score,
                valid_positions,
                quota["onset"],
                excluded=selected_set,
                nms_radius=int(temporal_nms_radius),
            )
            phase_lists["onset"] = onset
            selected_set.update(onset)

            offset_score = fields.offset[row] + (float(curvature_weight) * fields.curvature[row] if use_curvature else 0.0)
            offset = _ranked_positions(
                offset_score,
                valid_positions,
                quota["offset"],
                excluded=selected_set,
                nms_radius=int(temporal_nms_radius),
            )
            phase_lists["offset"] = offset
            selected_set.update(offset)

            core = _ranked_positions(
                fields.core[row],
                valid_positions,
                quota["core"],
                excluded=selected_set,
                nms_radius=0,
            )
            phase_lists["core"] = core
            selected_set.update(core)

            refill_needed = target - len(selected_set)
            refill = _ranked_positions(
                global_score[row],
                valid_positions,
                refill_needed,
                excluded=selected_set,
                nms_radius=0,
            )
            phase_lists["global_refill"] = refill
            selected_set.update(refill)
            selected = sorted(selected_set)
            if len(selected) != target:
                raise RuntimeError("phase selector failed to produce an exact-K selection")
            counts = {name: len(values) for name, values in phase_lists.items()}

        row_tensor = torch.full((total_budget,), -1, device=valid.device, dtype=torch.long)
        row_tensor[: len(selected)] = _positions_tensor(selected, device=valid.device)
        dense_mask = torch.zeros((temporal_len,), device=valid.device, dtype=torch.bool)
        dense_mask[_positions_tensor(selected, device=valid.device)] = True
        rows.append(row_tensor)
        masks.append(dense_mask)
        actual_phase_counts.append(counts)
        if len(selected) > 1:
            gaps = [int(curr) - int(prev) for prev, curr in zip(selected[:-1], selected[1:])]
            max_gap = max(gaps)
        else:
            max_gap = 0
        row_diagnostics.append(
            {
                "valid_count": int(valid_positions.numel()),
                "requested_budget": int(total_budget),
                "effective_budget": int(target),
                "selected_count": int(len(selected)),
                "exact_k": int(len(selected)) == int(target),
                "max_gap": int(max_gap),
            }
        )

    selected_positions = torch.stack(rows, dim=0)
    selected_mask = torch.stack(masks, dim=0)
    detector_input_length = selected_mask.long().sum(dim=1)
    expected = torch.minimum(
        torch.full_like(detector_input_length, int(total_budget)),
        valid.long().sum(dim=1),
    )
    return {
        "selected_positions": selected_positions,
        "positions": selected_positions,
        "selected_mask": selected_mask,
        "detector_input_length": detector_input_length,
        "effective_budget": detector_input_length,
        "phase_requested_quota": quotas,
        "phase_actual_counts": actual_phase_counts,
        "phase_quota_mode": quota_mode,
        "phase_use_curvature": bool(use_curvature),
        "phase_curvature_weight": float(curvature_weight),
        "phase_temporal_nms_radius": int(temporal_nms_radius),
        "diagnostics": {
            "schema_version": "duca_phase_selection_diagnostics_v1",
            "exact_k": bool(torch.equal(detector_input_length, expected)),
            "max_gap": max(item["max_gap"] for item in row_diagnostics),
            "rows": row_diagnostics,
        },
        "global_score": global_score.masked_fill(~valid, 0.0),
    }
