from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opentad.models.duca import structured_selection as structured


SCHEMA = "duca_paper_legacy_numeric_regression_v1"
FIXTURE_T = 96
FIXTURE_K = 48
FIXTURE_SHIFT = 2000.0
CURRENT_SHIFT = 37.0
TEMPERATURE = 1.0
THRESHOLDS = {
    "slot_atol": 5.0e-5,
    "slot_rtol": 5.0e-5,
    "gradient_atol": 2.0e-5,
    "gradient_rtol": 2.0e-3,
    "slot_row_mass_max_abs": 8.0e-6,
    "dual_logz_max_abs": 5.0e-5,
    "column_occupancy_max": 1.0 + 5.0e-4,
}


class RegressionFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionFailure(
            f"fail-closed DUCA legacy numeric regression: {message}"
        )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _legacy_raw_slot_mass_drift(
    node_log_probs: torch.Tensor,
    *,
    k: int,
    graph: structured.PhysicalExactKGraph,
    temperature: float,
) -> dict[str, float | bool | str | None]:
    """Evaluate the retired, gauge-dependent raw-message guard.

    This helper exists only for a fixed historical regression and for
    non-blocking production diagnostics.  It must never decide production
    capture, admission, search horizon, seed, or PASS/FAIL.
    """

    scores = node_log_probs.float() / float(temperature)
    temporal_len = int(scores.numel())
    alpha_rows = [
        torch.where(
            graph.source_valid,
            scores,
            scores.new_full(scores.shape, float("-inf")),
        )
    ]
    for _slot in range(1, int(k)):
        previous = alpha_rows[-1]
        candidates = previous[graph.predecessor_index]
        mass = structured._safe_masked_logsumexp(
            candidates, graph.predecessor_valid, dim=1
        )
        alpha_rows.append(scores + mass)
    alpha = torch.stack(alpha_rows, dim=0)
    logz = structured._safe_masked_logsumexp(
        alpha[-1], graph.sink_valid, dim=0
    )
    beta_rows = [scores.new_empty(scores.shape) for _ in range(int(k))]
    beta_rows[-1] = torch.where(
        graph.sink_valid,
        scores.new_zeros(scores.shape),
        scores.new_full(scores.shape, float("-inf")),
    )
    for slot in range(int(k) - 2, -1, -1):
        following = beta_rows[slot + 1]
        candidates = scores[graph.successor_index] + following[
            graph.successor_index
        ]
        beta_rows[slot] = structured._safe_masked_logsumexp(
            candidates, graph.successor_valid, dim=1
        )
    beta = torch.stack(beta_rows, dim=0)
    raw = alpha + beta - logz
    finite = torch.isfinite(raw)
    log_row_mass = structured._safe_masked_logsumexp(raw, finite, dim=1)
    all_finite = bool(torch.isfinite(log_row_mass).all().item())
    max_abs = float(log_row_mass.abs().max().item()) if all_finite else None
    envelope = max(
        5.0e-4,
        32.0 * torch.finfo(torch.float32).eps * float(max(temporal_len, int(k))),
    )
    triggered = not all_finite or bool(float(max_abs) > envelope)
    return {
        "legacy_status": "finite" if all_finite else "nonfinite",
        "old_raw_log_row_mass_max_abs": max_abs,
        "old_fp32_normalization_envelope": float(envelope),
        "old_guard_triggered": triggered,
    }


def _current_structural_result(
    scores: torch.Tensor,
    *,
    graph: structured.PhysicalExactKGraph,
) -> dict[str, Any]:
    x = scores.detach().clone().requires_grad_(True)
    occupancy, slots, logz = structured._physical_row_forward_backward(
        x,
        k=FIXTURE_K,
        graph=graph,
        temperature=TEMPERATURE,
    )
    weights = torch.linspace(
        -1.0,
        1.0,
        int(slots.numel()),
        dtype=slots.dtype,
        device=slots.device,
    ).reshape_as(slots)
    gradient = torch.autograd.grad((slots * weights).sum() + 0.01 * logz, x)[0]
    hard_positions = structured._physical_row_viterbi(
        scores.detach(), k=FIXTURE_K, graph=graph
    )[1]
    expectations = (
        slots
        * torch.arange(
            FIXTURE_T,
            dtype=slots.dtype,
            device=slots.device,
        )[None, :]
    ).sum(dim=1)
    return {
        "occupancy": occupancy,
        "slots": slots,
        "logz": logz,
        "gradient": gradient,
        "hard_positions": hard_positions,
        "slot_row_mass_max_abs": float(
            (slots.sum(dim=1) - 1.0).abs().max().item()
        ),
        "column_occupancy_max": float(occupancy.max().item()),
        "ordered_slot_expectation_min_gap": float(
            (expectations[1:] - expectations[:-1]).min().item()
        ),
    }


def run_regression() -> dict[str, Any]:
    seconds = torch.arange(FIXTURE_T, dtype=torch.float64)[None, :]
    valid = torch.ones((1, FIXTURE_T), dtype=torch.bool)
    cap = structured.physical_exact_uniform_gap_cap(
        seconds, valid, k=FIXTURE_K
    )
    graph = structured._build_physical_exact_k_graph(
        seconds[0], cap[0], torch.zeros(FIXTURE_T, dtype=torch.bool)
    )
    base32 = torch.linspace(-3.0, 3.0, FIXTURE_T, dtype=torch.float32)
    stress32 = base32 + FIXTURE_SHIFT
    legacy = _legacy_raw_slot_mass_drift(
        stress32,
        k=FIXTURE_K,
        graph=graph,
        temperature=TEMPERATURE,
    )
    _require(
        legacy["old_guard_triggered"] is True,
        "the frozen historical fixture did not trigger the retired raw guard",
    )

    current_stress = _current_structural_result(stress32, graph=graph)
    _require(
        bool(torch.isfinite(current_stress["slots"]).all().item())
        and bool(torch.isfinite(current_stress["gradient"]).all().item())
        and bool(torch.isfinite(current_stress["logz"]).item()),
        "current normalized solver is non-finite on the historical fixture",
    )
    _require(
        current_stress["slot_row_mass_max_abs"]
        <= THRESHOLDS["slot_row_mass_max_abs"],
        "current normalized solver violated slot-row mass",
    )
    _require(
        current_stress["column_occupancy_max"]
        <= THRESHOLDS["column_occupancy_max"],
        "current normalized solver violated column occupancy",
    )
    _require(
        current_stress["ordered_slot_expectation_min_gap"] > 0.0,
        "current normalized solver violated ordered slot expectation",
    )

    base = _current_structural_result(base32, graph=graph)
    shifted = _current_structural_result(base32 + CURRENT_SHIFT, graph=graph)
    slots_allclose = bool(
        torch.allclose(
            base["slots"],
            shifted["slots"],
            atol=THRESHOLDS["slot_atol"],
            rtol=THRESHOLDS["slot_rtol"],
        )
    )
    gradients_allclose = bool(
        torch.allclose(
            base["gradient"],
            shifted["gradient"],
            atol=THRESHOLDS["gradient_atol"],
            rtol=THRESHOLDS["gradient_rtol"],
        )
    )
    hard_path_identical = bool(
        torch.equal(base["hard_positions"], shifted["hard_positions"])
    )
    base64 = _current_structural_result(base32.double(), graph=graph)
    shifted64 = _current_structural_result(
        base32.double() + CURRENT_SHIFT, graph=graph
    )
    expected_logz_shift = float(FIXTURE_K) * CURRENT_SHIFT / TEMPERATURE
    logz_shift_residual = abs(
        float((shifted64["logz"] - base64["logz"]).item())
        - expected_logz_shift
    )
    _require(slots_allclose, "current slots are not additive-shift invariant")
    _require(
        gradients_allclose,
        "current gradients are not additive-shift invariant",
    )
    _require(hard_path_identical, "current hard path changed under additive shift")
    _require(
        logz_shift_residual <= THRESHOLDS["dual_logz_max_abs"],
        "current FP64 log-partition shift identity failed",
    )

    return {
        "fixture": {
            "t": FIXTURE_T,
            "k": FIXTURE_K,
            "temperature": TEMPERATURE,
            "seconds": "arange_0_95_float64",
            "base_scores": "linspace_minus3_plus3_float32",
            "historical_shift": FIXTURE_SHIFT,
            "current_shift_oracle": CURRENT_SHIFT,
            "max_gap_seconds": float(cap.item()),
        },
        "historical_negative_control": legacy,
        "current_solver": {
            "same_tensor_passed": True,
            "slot_row_mass_max_abs": current_stress["slot_row_mass_max_abs"],
            "column_occupancy_max": current_stress["column_occupancy_max"],
            "ordered_slot_expectation_min_gap": current_stress[
                "ordered_slot_expectation_min_gap"
            ],
            "additive_shift_slots_allclose": slots_allclose,
            "additive_shift_gradients_allclose": gradients_allclose,
            "additive_shift_hard_path_exact_identical": hard_path_identical,
            "fp64_logz_shift_residual": logz_shift_residual,
            "thresholds": dict(THRESHOLDS),
        },
    }


def build_artifact(*, expected_commit: str, slurm_job_id: str) -> dict[str, Any]:
    expected_commit = str(expected_commit).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "an exact expected commit is required",
    )
    _require(_git("rev-parse", "HEAD") == expected_commit, "Git commit drift")
    _require(
        not _git("status", "--porcelain", "--untracked-files=normal"),
        "regression requires a clean checkout",
    )
    _require(str(slurm_job_id).isdigit(), "a Slurm job id is required")
    result = run_regression()
    payload = {
        "schema_version": SCHEMA,
        "status": "passed",
        "fail_closed": True,
        "git_commit": expected_commit,
        "slurm_job_id": str(slurm_job_id),
        **result,
        "role": "deterministic_code_regression_negative_control",
        "production_admission_effect": "none",
        "validation_or_test_data_used": False,
        "checkpoint_created": False,
        "prediction_generated": False,
        "metric_accessed": False,
        "paper_metric_claim_allowed": False,
        "paper_method_performance_evidence": False,
        "claim_scope": "engineering_historical_solver_regression_only",
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_artifact(
        expected_commit=args.expected_commit,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    output = Path(args.output)
    _write_new_json(output, payload)
    print(
        json.dumps(
            {
                "path": str(output.expanduser().resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
