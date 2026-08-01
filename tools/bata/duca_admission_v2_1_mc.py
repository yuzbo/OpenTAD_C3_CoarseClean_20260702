from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tools.bata.duca_admission_v2_1_metrics import METRIC_IDS
from tools.bata.duca_admission_v2_1_statistics import finalize_max_t


INITIAL_REPLICATES = 100_000
MAXIMUM_REPLICATES = 200_000
JACKKNIFE_BATCH_SIZE = 1_000
Z_99 = 2.5758293035489004


def _jackknife_half_width(values: Sequence[float]) -> float:
    count = len(values)
    if count < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("jackknife values must contain at least two finite values")
    mean = math.fsum(values) / count
    variance = ((count - 1) / count) * math.fsum(
        (value - mean) ** 2 for value in values
    )
    half_width = Z_99 * math.sqrt(variance)
    if not math.isfinite(half_width):
        raise ValueError("jackknife half-width is nonfinite")
    return half_width


def batch_delete_jackknife(
    *,
    delta_hat: Mapping[str, float],
    replicates: Sequence[Mapping[str, float]],
    exact_zero: Mapping[str, bool],
    batch_size: int = JACKKNIFE_BATCH_SIZE,
    alpha: float = 0.05,
) -> dict[str, Any]:
    total = len(replicates)
    if type(batch_size) is not int:
        raise TypeError("jackknife batch size must be an integer")
    if batch_size <= 0 or total % batch_size:
        raise ValueError(
            "replicate count must be divisible by the jackknife batch size"
        )
    batch_count = total // batch_size
    if batch_count < 2:
        raise ValueError("delete-one-batch jackknife requires at least two batches")
    deletion_results = []
    for batch_index in range(batch_count):
        lower = batch_index * batch_size
        upper = lower + batch_size
        remaining = [*replicates[:lower], *replicates[upper:]]
        if len(remaining) != total - batch_size:
            raise AssertionError("jackknife deletion did not remove exactly one batch")
        deletion_results.append(
            finalize_max_t(
                delta_hat=delta_hat,
                replicates=remaining,
                exact_zero=exact_zero,
                alpha=alpha,
            )
        )

    active = [metric_id for metric_id in METRIC_IDS if not exact_zero[metric_id]]
    half_widths = {
        "q_plus": _jackknife_half_width([row["q_plus"] for row in deletion_results]),
        "q_minus": _jackknife_half_width([row["q_minus"] for row in deletion_results]),
        "lower": {
            metric_id: _jackknife_half_width(
                [row["lower"][metric_id] for row in deletion_results]
            )
            for metric_id in active
        },
        "upper": {
            metric_id: _jackknife_half_width(
                [row["upper"][metric_id] for row in deletion_results]
            )
            for metric_id in active
        },
    }
    return {
        "batch_size": batch_size,
        "batch_count": batch_count,
        "deleted_per_replicate": batch_size,
        "remaining_per_replicate": total - batch_size,
        "half_widths_99": half_widths,
        "deletion_results": deletion_results,
    }


def certify_mc_error(
    *,
    full_result: Mapping[str, Any],
    delta_hat: Mapping[str, float],
    jackknife: Mapping[str, Any],
) -> dict[str, Any]:
    if not full_result.get("mc_required"):
        if (
            full_result.get("numeric_tail_status") != "PASSED_EXACT_ZERO"
            or full_result.get("active_metrics") != []
        ):
            raise ValueError("non-MC result is not the registered exact-zero branch")
        return {"status": "PASSED_EXACT_ZERO", "passed": True, "checks": []}
    half_widths = jackknife["half_widths_99"]
    checks: list[dict[str, Any]] = []
    for name in ("q_plus", "q_minus"):
        value = float(full_result[name])
        half_width = float(half_widths[name])
        if (
            not math.isfinite(value)
            or not math.isfinite(half_width)
            or half_width < 0.0
        ):
            raise ValueError("MC quantile or jackknife half-width is nonfinite")
        normalizer = max(1.0, abs(value))
        ratio = half_width / normalizer
        checks.append(
            {
                "parameter": name,
                "criterion": "relative_to_max_1_abs",
                "value": ratio,
                "limit": 0.005,
                "passed": ratio <= 0.005,
            }
        )
    for metric_id in full_result["active_metrics"]:
        scale = float(full_result["scales"][metric_id])
        lower = float(full_result["lower"][metric_id])
        upper = float(full_result["upper"][metric_id])
        observed = float(delta_hat[metric_id])
        if (
            not all(math.isfinite(value) for value in (scale, lower, upper, observed))
            or scale <= 0.0
        ):
            raise ValueError("MC interval inputs are zero-scale or nonfinite")
        normalizer = max(abs(observed), scale, abs(lower), abs(upper))
        if not math.isfinite(normalizer) or normalizer == 0.0:
            raise ValueError(
                "MC normalizer is zero or nonfinite outside exact-zero branch"
            )
        for side in ("lower", "upper"):
            half_width = float(half_widths[side][metric_id])
            if not math.isfinite(half_width) or half_width < 0.0:
                raise ValueError("MC interval half-width is negative or nonfinite")
            relative = half_width / normalizer
            scale_relative = half_width / scale
            checks.extend(
                (
                    {
                        "parameter": f"{side}:{metric_id}",
                        "criterion": "relative_to_mc_normalizer",
                        "value": relative,
                        "limit": 0.01,
                        "passed": relative <= 0.01,
                    },
                    {
                        "parameter": f"{side}:{metric_id}",
                        "criterion": "relative_to_scale",
                        "value": scale_relative,
                        "limit": 0.025,
                        "passed": scale_relative <= 0.025,
                    },
                )
            )
    passed = all(bool(row["passed"]) for row in checks)
    return {
        "status": "PASSED" if passed else "FAILED_CLOSED",
        "failure_code": None if passed else "MC_UNSTABLE",
        "passed": passed,
        "checks": checks,
    }


def run_prefix_extensible_mc(
    *,
    generator: Callable[[int, int], list[dict[str, float]]],
    diagnostic_generator: Callable[[int, int], list[dict[str, float]]] | None = None,
    delta_hat: Mapping[str, float],
    exact_zero: Mapping[str, bool],
    initial_replicates: int = INITIAL_REPLICATES,
    maximum_replicates: int = MAXIMUM_REPLICATES,
    batch_size: int = JACKKNIFE_BATCH_SIZE,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if type(initial_replicates) is not int or type(maximum_replicates) is not int:
        raise TypeError("MC replicate counts must be integers")
    if maximum_replicates != 2 * initial_replicates:
        raise ValueError(
            "maximum replicate count must be exactly twice the initial count"
        )
    replicates = generator(0, initial_replicates)
    if len(replicates) != initial_replicates:
        raise ValueError("generator did not return the registered initial prefix")
    full_result = finalize_max_t(
        delta_hat=delta_hat, replicates=replicates, exact_zero=exact_zero, alpha=alpha
    )
    jackknife = batch_delete_jackknife(
        delta_hat=delta_hat,
        replicates=replicates,
        exact_zero=exact_zero,
        batch_size=batch_size,
        alpha=alpha,
    )
    certificate = certify_mc_error(
        full_result=full_result, delta_hat=delta_hat, jackknife=jackknife
    )
    extended = False
    prefix_fingerprint = tuple(
        tuple(row[metric_id] for metric_id in METRIC_IDS) for row in replicates
    )
    if not certificate["passed"]:
        suffix = generator(initial_replicates, maximum_replicates)
        if len(suffix) != maximum_replicates - initial_replicates:
            raise ValueError("generator did not return the registered extension suffix")
        replicates.extend(suffix)
        if (
            tuple(
                tuple(row[metric_id] for metric_id in METRIC_IDS)
                for row in replicates[:initial_replicates]
            )
            != prefix_fingerprint
        ):
            raise AssertionError("100k prefix changed during deterministic extension")
        extended = True
        full_result = finalize_max_t(
            delta_hat=delta_hat,
            replicates=replicates,
            exact_zero=exact_zero,
            alpha=alpha,
        )
        jackknife = batch_delete_jackknife(
            delta_hat=delta_hat,
            replicates=replicates,
            exact_zero=exact_zero,
            batch_size=batch_size,
            alpha=alpha,
        )
        certificate = certify_mc_error(
            full_result=full_result, delta_hat=delta_hat, jackknife=jackknife
        )
        if not certificate["passed"]:
            certificate = {
                **certificate,
                "status": "FAILED_CLOSED",
                "failure_code": "MC_UNSTABLE",
            }
    diagnostic = None
    if diagnostic_generator is not None:
        diagnostic_replicates = diagnostic_generator(0, initial_replicates)
        if len(diagnostic_replicates) != initial_replicates:
            raise ValueError(
                "diagnostic generator did not return the registered 100k prefix"
            )
        diagnostic_result = finalize_max_t(
            delta_hat=delta_hat,
            replicates=diagnostic_replicates,
            exact_zero=exact_zero,
            alpha=alpha,
        )
        diagnostic_jackknife = batch_delete_jackknife(
            delta_hat=delta_hat,
            replicates=diagnostic_replicates,
            exact_zero=exact_zero,
            batch_size=batch_size,
            alpha=alpha,
        )
        diagnostic_certificate = certify_mc_error(
            full_result=diagnostic_result,
            delta_hat=delta_hat,
            jackknife=diagnostic_jackknife,
        )
        diagnostic = {
            "replicate_count": initial_replicates,
            "passed": bool(diagnostic_certificate["passed"]),
            "binary_pass_vector": [
                bool(row["passed"]) for row in diagnostic_certificate["checks"]
            ],
        }
    return {
        "replicate_count": len(replicates),
        "extended": extended,
        "result": full_result,
        "certificate": certificate,
        "secondary_stream_diagnostic": diagnostic,
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
