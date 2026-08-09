from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def validate_legacy_numeric_regression_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    expected_commit = str(expected_commit).strip().lower()
    expected_sha256 = str(expected_sha256).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("legacy regression requires an exact commit")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("legacy regression requires an exact SHA-256")
    if not source.is_file():
        raise FileNotFoundError(f"legacy regression is missing: {source}")
    observed_sha = _sha256(source)
    if observed_sha != expected_sha256:
        raise RuntimeError("legacy regression SHA-256 drift")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("legacy regression artifact is not a mapping")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    fixture = payload.get("fixture", {})
    legacy = payload.get("historical_negative_control", {})
    current = payload.get("current_solver", {})
    legacy_max_abs = legacy.get("old_raw_log_row_mass_max_abs")
    legacy_finite_or_explicit_nonfinite = (
        legacy.get("legacy_status") == "finite"
        and legacy_max_abs is not None
        and math.isfinite(float(legacy_max_abs))
    ) or (
        legacy.get("legacy_status") == "nonfinite" and legacy_max_abs is None
    )
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "passed"
        or payload.get("fail_closed") is not True
        or payload.get("git_commit") != expected_commit
        or not str(payload.get("slurm_job_id", "")).isdigit()
        or content_sha != _canonical_sha256(unsigned)
        or payload.get("role")
        != "deterministic_code_regression_negative_control"
        or payload.get("production_admission_effect") != "none"
        or int(fixture.get("t", -1)) != FIXTURE_T
        or int(fixture.get("k", -1)) != FIXTURE_K
        or float(fixture.get("temperature", math.nan)) != TEMPERATURE
        or float(fixture.get("historical_shift", math.nan)) != FIXTURE_SHIFT
        or float(fixture.get("current_shift_oracle", math.nan)) != CURRENT_SHIFT
        or not math.isfinite(float(fixture.get("max_gap_seconds", math.nan)))
        or legacy.get("old_guard_triggered") is not True
        or not legacy_finite_or_explicit_nonfinite
        or not math.isfinite(
            float(legacy.get("old_fp32_normalization_envelope", math.nan))
        )
        or current.get("same_tensor_passed") is not True
        or current.get("additive_shift_slots_allclose") is not True
        or current.get("additive_shift_gradients_allclose") is not True
        or current.get("additive_shift_hard_path_exact_identical") is not True
        or current.get("thresholds") != THRESHOLDS
        or float(current.get("slot_row_mass_max_abs", math.inf))
        > THRESHOLDS["slot_row_mass_max_abs"]
        or float(current.get("column_occupancy_max", math.inf))
        > THRESHOLDS["column_occupancy_max"]
        or float(current.get("ordered_slot_expectation_min_gap", -math.inf)) <= 0.0
        or float(current.get("fp64_logz_shift_residual", math.inf))
        > THRESHOLDS["dual_logz_max_abs"]
        or payload.get("validation_or_test_data_used") is not False
        or payload.get("checkpoint_created") is not False
        or payload.get("prediction_generated") is not False
        or payload.get("metric_accessed") is not False
        or payload.get("paper_metric_claim_allowed") is not False
        or payload.get("paper_method_performance_evidence") is not False
    ):
        raise RuntimeError("legacy numeric regression contract drift")
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "git_commit": expected_commit,
        "path": str(source),
        "sha256": observed_sha,
        "claim_scope": "engineering_historical_solver_regression_only",
        "performance_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_legacy_numeric_regression_artifact(
                args.receipt,
                expected_commit=args.expected_commit,
                expected_sha256=args.expected_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
