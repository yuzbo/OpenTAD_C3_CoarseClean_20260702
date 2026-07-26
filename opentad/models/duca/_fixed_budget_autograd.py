"""Autograd helpers for exact-budget temporal sampling."""

from __future__ import annotations

import torch


class FixedBudgetRateGradient(torch.autograd.Function):
    """Keep calibrated retention rates on their constant-sum tangent space.

    The forward value is produced by a detached bisection threshold. Its
    implicit derivative subtracts the one direction that only changes the
    global calibration threshold, so a common logit shift has zero gradient.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        rates: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        del logits
        ctx.save_for_backward(rates)
        ctx.temperature = float(temperature)
        return rates

    @staticmethod
    def backward(ctx, grad_rates: torch.Tensor):
        (rates,) = ctx.saved_tensors
        slope = rates * (1.0 - rates) / ctx.temperature
        slope_sum = slope.sum().clamp_min(torch.finfo(slope.dtype).eps)
        baseline = (grad_rates * slope).sum() / slope_sum
        grad_logits = slope * (grad_rates - baseline)
        return grad_logits, None, None
