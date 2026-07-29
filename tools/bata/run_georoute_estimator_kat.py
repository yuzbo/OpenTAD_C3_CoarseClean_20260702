#!/usr/bin/env python3
"""Numerical KATs for the GeoRoute estimator and representation split."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.backbones.georoute_routing import (  # noqa: E402
    ordered_plackett_luce_log_prob,
    score_function_policy_loss,
    select_exact_k,
)
from opentad.models.backbones.georoute_wrapper import (  # noqa: E402
    GeoRouteSparseTemporalAdapter,
)


KAT_SCHEMA = "georoute_estimator_representation_kat_v2"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout.strip()


def _require_source(expected_commit: str) -> str:
    actual = _git_output("rev-parse", "HEAD").lower()
    if actual != expected_commit.lower():
        raise RuntimeError("KAT source commit does not match --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("KAT requires a clean source snapshot")
    return actual


def _legacy_adapter_forward(
    adapter: GeoRouteSparseTemporalAdapter,
    selected_features: torch.Tensor,
    selected_scores: torch.Tensor,
    geometry: torch.Tensor,
    selected_coordinates: torch.Tensor,
) -> torch.Tensor:
    relative = (
        selected_coordinates - geometry[:, :, None, :2]
    ) / geometry[:, :, None, 2:].clamp_min(1e-6)
    coordinate_features = torch.cat(
        (selected_coordinates, relative), dim=-1
    )
    selected_features = selected_features + adapter.coordinate_projection(
        coordinate_features
    )
    weights = torch.full_like(
        selected_scores,
        1.0 / float(selected_scores.shape[-1]),
    ).unsqueeze(-1)
    pooled = (weights * selected_features).sum(dim=2)
    pooled = adapter.norm(pooled + adapter.geometry_projection(geometry))
    temporal = adapter.output(
        adapter.temporal(pooled.transpose(1, 2))
    ).transpose(1, 2)
    return (pooled + temporal).transpose(1, 2)


def _pl_probability_kats() -> dict[str, Any]:
    temperature = 0.7
    logits = torch.tensor(
        [[[0.2, -0.3, 1.1, 0.7]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    order = torch.tensor([[[2, 0]]], dtype=torch.long)
    observed = ordered_plackett_luce_log_prob(
        logits,
        order,
        temperature=temperature,
    )
    scaled = logits / temperature
    manual = torch.log_softmax(scaled, dim=-1)[..., 2]
    remaining = torch.tensor([True, True, False, True]).view(1, 1, 4)
    manual = manual + torch.log_softmax(
        scaled.masked_fill(~remaining, float("-inf")), dim=-1
    )[..., 0]
    manual_error = float((observed - manual).abs().max().item())

    ordered_probabilities = []
    for permutation in itertools.permutations(range(4), 2):
        candidate = torch.tensor([[list(permutation)]], dtype=torch.long)
        ordered_probabilities.append(
            ordered_plackett_luce_log_prob(
                logits,
                candidate,
                temperature=temperature,
            ).exp()
        )
    probability_sum = torch.stack(ordered_probabilities).sum()
    normalization_error = float((probability_sum - 1.0).abs().item())

    observed.sum().backward()
    gradient = logits.grad.detach()
    selected_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
        -1, order, True
    )
    selected_gradient_min_abs = float(
        gradient.masked_select(selected_mask).abs().min().item()
    )
    unselected_gradient_min_abs = float(
        gradient.masked_select(~selected_mask).abs().min().item()
    )
    return {
        "manual_log_probability_error": manual_error,
        "ordered_probability_sum": float(probability_sum.item()),
        "normalization_error": normalization_error,
        "selected_gradient_min_abs": selected_gradient_min_abs,
        "unselected_gradient_min_abs": unselected_gradient_min_abs,
        "passed": bool(
            manual_error <= 1e-7
            and normalization_error <= 1e-6
            and selected_gradient_min_abs > 1e-8
            and unselected_gradient_min_abs > 1e-8
            and torch.isfinite(gradient).all().item()
        ),
    }


def _risk_sign_kat() -> dict[str, Any]:
    logits = torch.tensor(
        [[[0.2, -0.3, 0.7]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    costs = torch.tensor([0.4, 1.1, -0.2], dtype=torch.float64)
    probabilities = torch.softmax(logits, dim=-1)
    risk = (probabilities * costs.view(1, 1, -1)).sum()
    expected_gradient = torch.autograd.grad(
        risk, logits, retain_graph=True
    )[0]
    estimator_expectation = logits.new_zeros(())
    for choice in range(costs.numel()):
        ordered = torch.tensor([[[choice]]], dtype=torch.long)
        log_probability = ordered_plackett_luce_log_prob(
            logits,
            ordered,
            temperature=1.0,
        )
        estimator_expectation = (
            estimator_expectation
            + probabilities[..., choice].detach()
            * score_function_policy_loss(
                detector_cost=costs[choice],
                ordered_log_prob=log_probability,
                baseline=torch.zeros((), dtype=torch.float64),
                weight=1.0,
            )
        )
    observed_gradient = torch.autograd.grad(
        estimator_expectation, logits
    )[0]
    max_abs_error = float(
        (expected_gradient - observed_gradient).abs().max().item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            expected_gradient.reshape(1, -1),
            observed_gradient.reshape(1, -1),
        ).item()
    )
    return {
        "max_abs_gradient_error": max_abs_error,
        "gradient_cosine": cosine,
        "passed": bool(max_abs_error <= 1e-6 and cosine > 0.999999),
    }


def _st_vs_pl_reachability_kat() -> dict[str, Any]:
    roi = torch.zeros(1, 1, 8, dtype=torch.float64)
    base = torch.tensor(
        [[[1.7, -0.4, 0.8, 2.1, -1.2, 0.3, 1.1, -0.7]]],
        dtype=torch.float64,
    )
    st_logits = base.clone().requires_grad_(True)
    st_route = select_exact_k(
        roi_logits=roi,
        residual_logits=st_logits,
        mode="free",
        tokens_per_tubelet=3,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="straight_through",
        temperature=0.7,
        valid_mask=torch.ones_like(st_logits, dtype=torch.bool),
    )
    st_route["st_gate"].sum().backward()
    st_selected = st_logits.grad.masked_select(st_route["selected_mask"])
    st_unselected = st_logits.grad.masked_select(
        ~st_route["selected_mask"]
    )

    pl_logits = base.clone().requires_grad_(True)
    torch.manual_seed(37)
    pl_route = select_exact_k(
        roi_logits=roi,
        residual_logits=pl_logits,
        mode="free",
        tokens_per_tubelet=3,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="score_function",
        temperature=0.7,
        valid_mask=torch.ones_like(pl_logits, dtype=torch.bool),
    )
    pl_route["ordered_log_prob"].sum().backward()
    pl_selected = pl_logits.grad.masked_select(pl_route["selected_mask"])
    pl_unselected = pl_logits.grad.masked_select(
        ~pl_route["selected_mask"]
    )
    values = {
        "st_selected_gradient_max_abs": float(
            st_selected.abs().max().item()
        ),
        "st_unselected_nonzero_count": int(
            torch.count_nonzero(st_unselected).item()
        ),
        "pl_selected_gradient_min_abs": float(
            pl_selected.abs().min().item()
        ),
        "pl_unselected_gradient_min_abs": float(
            pl_unselected.abs().min().item()
        ),
    }
    values["passed"] = bool(
        values["st_selected_gradient_max_abs"] > 1e-8
        and values["st_unselected_nonzero_count"] == 0
        and values["pl_selected_gradient_min_abs"] > 1e-8
        and values["pl_unselected_gradient_min_abs"] > 1e-8
        and torch.isfinite(st_logits.grad).all().item()
        and torch.isfinite(pl_logits.grad).all().item()
    )
    return values


def _amp_horizon_kat(
    *,
    device: torch.device | None = None,
    tubelets: int = 384,
    patch_capacity: int = 220,
    target_k: int = 64,
    loss_scale: float = 256.0,
) -> dict[str, Any]:
    """Reproduce the production PL horizon from an AMP-shaped fp16 source."""

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not 0 < int(target_k) <= int(patch_capacity):
        raise ValueError("AMP horizon KAT requires 0 < K <= patch capacity")
    logits = torch.zeros(
        1,
        int(tubelets),
        int(patch_capacity),
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    ordered = torch.arange(
        int(target_k),
        dtype=torch.long,
        device=device,
    ).view(1, 1, -1).expand(1, int(tubelets), -1)
    log_probability = ordered_plackett_luce_log_prob(
        logits,
        ordered,
        temperature=0.7,
    )
    policy_loss = score_function_policy_loss(
        detector_cost=torch.tensor(2.0, device=device),
        ordered_log_prob=log_probability,
        baseline=torch.tensor(1.0, device=device),
        weight=1.0,
    )
    gradient = torch.autograd.grad(
        policy_loss * float(loss_scale),
        logits,
    )[0]
    loss_magnitude = float(policy_loss.detach().abs().item())
    gradient_max_abs = float(gradient.detach().abs().max().item())
    passed = bool(
        log_probability.dtype == torch.float32
        and policy_loss.dtype == torch.float32
        and torch.isfinite(log_probability).all().item()
        and torch.isfinite(policy_loss).item()
        and torch.isfinite(gradient).all().item()
        and torch.count_nonzero(gradient).item() > 0
        and loss_magnitude > torch.finfo(torch.float16).max
    )
    return {
        "device_type": device.type,
        "source_dtype": str(logits.dtype),
        "likelihood_dtype": str(log_probability.dtype),
        "policy_loss_dtype": str(policy_loss.dtype),
        "tubelets": int(tubelets),
        "patch_capacity": int(patch_capacity),
        "target_k": int(target_k),
        "loss_scale": float(loss_scale),
        "policy_loss_abs": loss_magnitude,
        "fp16_max": float(torch.finfo(torch.float16).max),
        "gradient_max_abs": gradient_max_abs,
        "all_likelihoods_finite": bool(
            torch.isfinite(log_probability).all().item()
        ),
        "policy_loss_finite": bool(torch.isfinite(policy_loss).item()),
        "all_scaled_gradients_finite": bool(
            torch.isfinite(gradient).all().item()
        ),
        "passed": passed,
    }


def _representation_kats() -> dict[str, Any]:
    torch.manual_seed(41)
    adapter = GeoRouteSparseTemporalAdapter(channels=8).double()
    selected = torch.randn(1, 3, 4, 8, dtype=torch.float64)
    scores = torch.randn(1, 3, 4, dtype=torch.float64)
    geometry = (
        torch.tensor(
            [[[0.4, 0.6, 0.7, 0.8]]],
            dtype=torch.float64,
        )
        .expand(1, 3, 4)
        .clone()
        .requires_grad_(True)
    )
    coordinates = torch.rand(
        1, 3, 4, 2, dtype=torch.float64, requires_grad=True
    )
    disabled = adapter(
        selected,
        scores,
        geometry,
        coordinates,
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    changed = adapter(
        selected,
        scores,
        geometry.detach() + 0.05,
        (coordinates.detach() + 0.1).clamp_max(1.0),
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    disabled_invariant = bool(torch.equal(disabled, changed))
    geometry_grad, coordinate_grad = torch.autograd.grad(
        disabled.sum(),
        (geometry, coordinates),
        allow_unused=True,
    )
    disabled_gradients_absent = (
        geometry_grad is None and coordinate_grad is None
    )

    legacy = _legacy_adapter_forward(
        adapter,
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
    )
    split_all = adapter(
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
        use_absolute_coordinates=True,
        use_roi_relative_coordinates=True,
        use_geometry_projection=True,
        pooling_mode="uniform_selected",
    )
    legacy_max_abs_error = float((legacy - split_all).abs().max().item())

    channel_deltas = {}
    for name, absolute, relative, projection in (
        ("absolute", True, False, False),
        ("roi_relative", False, True, False),
        ("geometry_projection", False, False, True),
    ):
        output = adapter(
            selected,
            scores,
            geometry.detach(),
            coordinates.detach(),
            use_absolute_coordinates=absolute,
            use_roi_relative_coordinates=relative,
            use_geometry_projection=projection,
            pooling_mode="uniform_selected",
        )
        channel_deltas[name] = float((output - disabled).abs().max().item())
    passed = bool(
        disabled_invariant
        and disabled_gradients_absent
        and legacy_max_abs_error <= 1e-7
        and all(value > 1e-8 for value in channel_deltas.values())
    )
    return {
        "disabled_output_bitwise_invariant": disabled_invariant,
        "disabled_geometry_coordinate_gradients_absent": (
            disabled_gradients_absent
        ),
        "legacy_all_enabled_max_abs_error": legacy_max_abs_error,
        "individual_channel_max_abs_delta": channel_deltas,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GeoRoute estimator KAT must run inside Slurm")
    runtime_commit = _require_source(args.expected_commit)
    output_path = args.output.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(output_path, write_boundary) or output_path == write_boundary:
        raise ValueError("KAT output leaves the remote write boundary")
    checks = {
        "pl_probability": _pl_probability_kats(),
        "risk_gradient_sign": _risk_sign_kat(),
        "st_vs_pl_reachability": _st_vs_pl_reachability_kat(),
        "amp_production_horizon": _amp_horizon_kat(),
        "representation_isolation": _representation_kats(),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    receipt: dict[str, Any] = {
        "schema_version": KAT_SCHEMA,
        "status": "PASS_MECHANICAL_ONLY" if passed else "FAIL_KAT",
        "runtime_commit": runtime_commit,
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "torch_version": torch.__version__,
        "checks": checks,
        "development_metric_emitted": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _atomic_write_json(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
