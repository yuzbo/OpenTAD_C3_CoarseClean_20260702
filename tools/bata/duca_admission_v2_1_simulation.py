from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.bata.duca_admission_v2_1_hashing import (
    PROTOCOL_ID,
    canonical_text,
    domain_hash,
    sha256_bytes,
    u64be,
)
from tools.bata.duca_admission_v2_1_incidence import validate_incidence
from tools.bata.duca_admission_v2_1_mc import run_prefix_extensible_mc
from tools.bata.duca_admission_v2_1_metrics import METRIC_IDS, type1_quantile
from tools.bata.duca_admission_v2_1_roles import ROLE_ORDER
from tools.bata.duca_admission_v2_1_statistics import (
    estimate_role_contrast,
    finalize_max_t,
    is_binary64_positive_zero,
    multiplier_replicates,
)
from tools.bata.duca_evidence_io import (
    canonical_sha256,
    verify_content_sha256,
    with_content_sha256,
)
from tools.bata.duca_safe_publication import read_file_without_symlinks


SIMULATION_REGISTRY_SCHEMA = "duca_admission_v2_1_simulation_registry_v1"
DEFAULT_REGISTRY = Path(
    "configs/protocols/duca_admission_v2_1_simulation_registry_v1.json"
)
SHIFT_PROFILES = (
    "NULL",
    "SAFE_ALL_M6",
    "HARM_ONE_P4",
    "HARM_ONE_P6",
    "HARM_ALL_P3",
)
EXPECTED_EFFECT_MIXES = {
    "ROW": {"sigma_v": 1, "sigma_p": 0, "sigma_e": 0},
    "PROCESS": {"sigma_v": 0, "sigma_p": 1, "sigma_e": 0},
    "CELL": {"sigma_v": 0, "sigma_p": 0, "sigma_e": 1},
    "EQUAL": {
        "sigma_v": "1/sqrt(3)",
        "sigma_p": "1/sqrt(3)",
        "sigma_e": "1/sqrt(3)",
    },
    "ZERO": {"sigma_v": 0, "sigma_p": 0, "sigma_e": 0},
}
EXPECTED_DISTRIBUTIONS = {
    "GAUSS": {"definition": "N(0,R)"},
    "T5": {
        "degrees_of_freedom": 5,
        "scale": "sqrt(3/5)",
        "shared_chi_square_across_12_metrics": True,
    },
    "LOGN075": {"lambda": 0.75, "centered": True, "unit_marginal_variance": True},
    "EXACT_ZERO": {"binary64_positive_zero": True},
}
EXPECTED_COUNT_LAWS = {
    "N16": {"n": 16, "h": 1},
    "NLINK": {
        "natural_short": "1+((3*i+5*p+slot) mod 4)",
        "natural_full": "8+((7*i+3*p+slot) mod 57)",
        "h": "min(4,max(0.5,sqrt(16/n)))",
    },
}
EXPECTED_SHIFT_PROFILES = {
    "NULL": {"multiplier_true_se": 0, "metrics": "all"},
    "SAFE_ALL_M6": {"multiplier_true_se": -6, "metrics": "all"},
    "HARM_ONE_P4": {"multiplier_true_se": 4, "metrics": ["M00"]},
    "HARM_ONE_P6": {"multiplier_true_se": 6, "metrics": ["M00"]},
    "HARM_ALL_P3": {"multiplier_true_se": 3, "metrics": "all"},
}
EXPECTED_EXECUTION = {
    "outer_datasets_per_scenario": 500,
    "initial_inner_replicates": 100_000,
    "maximum_inner_replicates": 200_000,
    "alpha_tail": 0.05,
    "jackknife_batch_size": 1_000,
    "outer_seed_domain": "DUCA-V21-SIM-OUTER-V1",
    "inner_factor_domain": "product-multiplier-factor",
    "recompute_per_outer": [
        "q_plus",
        "q_minus",
        "scales",
        "lower",
        "upper",
        "mc_certificate",
    ],
}
EXPECTED_GATES = {
    "nonzero_scenario": {
        "simultaneous_upper_coverage_min_count": 477,
        "simultaneous_lower_coverage_min_count": 477,
        "null_false_alarm_max_count": 23,
        "mc_unstable_max_count": 5,
        "denominator": 500,
    },
    "ordinary_S000_S047": {
        "median_vir_max": 3.25,
        "type1_p95_vir_max": 5,
        "median_nwr_max": 2.25,
        "type1_p95_nwr_max": 3.25,
    },
    "boundary_S048_S049_S051": {
        "median_vir_max": 4,
        "type1_p95_vir_max": 8,
        "median_nwr_max": 2.75,
        "type1_p95_nwr_max": 4.5,
    },
    "exact_zero_S050": {"exact_zero_branch_min_count": 500, "denominator": 500},
    "power": {
        "SAFE_ALL_M6_pass_min_count": 398,
        "HARM_ONE_P4_alarm_min_count": 250,
        "HARM_ONE_P6_alarm_min_count": 398,
        "HARM_ALL_P3_alarm_min_count": 398,
        "denominator": 500,
    },
    "mc_half_width": {
        "scenario_parameter_coverage_min_count": 197,
        "scenario_parameter_denominator": 200,
        "pooled_parameter_coverage_min_count": 4727,
        "pooled_parameter_denominator": 4800,
        "reference_replicates": 4_000_000,
        "reference_half_replicates": 2_000_000,
        "half_difference_over_median_operational_h_max": 0.25,
        "median_h200k_over_h100k_max": 0.8,
        "type1_p95_h200k_over_h100k_max": 1.05,
    },
}
BONFERRONI_REFERENCE_CRITICAL = 2.638257273476751
SIMULATION_SCENARIO_RECEIPT_SCHEMA = "duca_admission_v2_1_simulation_scenario_gate_v1"
MC_CALIBRATION_SCENARIO_SCHEMA = "duca_admission_v2_1_mc_calibration_scenario_v1"
MC_CALIBRATION_REGISTRY_SCHEMA = "duca_admission_v2_1_mc_calibration_registry_v1"
SIMULATION_REGISTRY_RECEIPT_SCHEMA = "duca_admission_v2_1_simulation_gate_v1"
MC_PARAMETER_IDS = (
    "q_plus",
    "q_minus",
    *(f"lower:{metric_id}" for metric_id in METRIC_IDS),
    *(f"upper:{metric_id}" for metric_id in METRIC_IDS),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_simulation_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.name == "posix":
        raw, _metadata = read_file_without_symlinks(registry_path)
    else:
        raw = registry_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("simulation registry is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("simulation registry must be a JSON object")
    validate_simulation_registry(payload)
    return {
        **payload,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": canonical_sha256(payload),
    }


def validate_simulation_registry(payload: Mapping[str, Any]) -> None:
    base_keys = {
        "schema",
        "protocol_id",
        "claim_scope",
        "reference_environment",
        "metric_ids",
        "effect_mixes",
        "distributions",
        "count_laws",
        "shift_profiles",
        "execution",
        "gates",
        "mc_calibration_scenarios",
        "scenarios",
        "authorization",
    }
    observed_keys = set(payload)
    if observed_keys not in (
        base_keys,
        base_keys | {"artifact_sha256", "semantic_sha256"},
    ):
        raise ValueError("simulation registry is not a closed-world object")
    if observed_keys != base_keys:
        core = {key: payload[key] for key in base_keys}
        sha256_bytes(
            payload["artifact_sha256"], field_name="simulation registry artifact sha256"
        )
        sha256_bytes(
            payload["semantic_sha256"], field_name="simulation registry semantic sha256"
        )
        if payload["semantic_sha256"] != canonical_sha256(core):
            raise ValueError("simulation registry semantic SHA-256 drifted")
        payload = core
    if payload.get("schema") != SIMULATION_REGISTRY_SCHEMA:
        raise ValueError("unsupported simulation registry schema")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("simulation registry protocol drift")
    if payload.get("claim_scope") != (
        "scenario-conditional empirical calibration on the registered DUCA 32x8 incidence grid"
    ):
        raise ValueError("simulation claim scope drift")
    if tuple(payload.get("metric_ids", ())) != METRIC_IDS:
        raise ValueError("simulation metric registry drift")
    reference = payload.get("reference_environment")
    if not isinstance(reference, Mapping) or set(reference) != {
        "python_version",
        "numpy_version",
        "bit_generator",
        "golden_seed",
        "golden_raw_byte_count",
        "golden_raw_sha256",
    }:
        raise ValueError("simulation reference environment is not closed")
    if dict(reference) != {
        "python_version": "3.11.7",
        "numpy_version": "1.23.5",
        "bit_generator": "numpy.random.Philox",
        "golden_seed": 0,
        "golden_raw_byte_count": 4096,
        "golden_raw_sha256": "d391e4f5c20f9e3df8971ba584af858ff37611d5a8005c5267725ddd3b98743d",
    }:
        raise ValueError("simulation reference environment drift")
    sha256_bytes(
        str(reference.get("golden_raw_sha256", "")), field_name="Philox golden sha256"
    )
    if payload.get("effect_mixes") != EXPECTED_EFFECT_MIXES:
        raise ValueError("simulation effect-mix registry drift")
    if payload.get("distributions") != EXPECTED_DISTRIBUTIONS:
        raise ValueError("simulation distribution registry drift")
    if payload.get("count_laws") != EXPECTED_COUNT_LAWS:
        raise ValueError("simulation count-law registry drift")
    if payload.get("shift_profiles") != EXPECTED_SHIFT_PROFILES:
        raise ValueError("simulation shift-profile registry drift")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 52:
        raise ValueError("simulation registry must contain exactly 52 scenarios")
    identifiers = [str(row.get("scenario_id", "")) for row in scenarios]
    if len(set(identifiers)) != 52:
        raise ValueError("simulation scenario IDs are not unique")
    for index, identifier in enumerate(identifiers):
        if not identifier.startswith(f"S{index:03d}_"):
            raise ValueError("simulation scenarios are not in frozen S000-S051 order")
    if identifiers[48:] != [
        "S048_TINY_SCALE_EQUAL_GAUSS_R085_N16",
        "S049_SINGLE_CELL_LEVERAGE_EQUAL_T5_R085_NLINK",
        "S050_ALL_ZERO",
        "S051_NEAR_COLLINEAR_EQUAL_LOGN075_R0999_NLINK",
    ]:
        raise ValueError("simulation boundary scenarios drifted")
    expected: list[tuple[str, str, str, str, float, str, str | None]] = []
    scenario_index = 0
    for mix in ("ROW", "PROCESS", "CELL", "EQUAL"):
        for distribution in ("GAUSS", "T5", "LOGN075"):
            for correlation_id, rho in (("R035", 0.35), ("R085", 0.85)):
                for count_law in ("N16", "NLINK"):
                    identifier = f"S{scenario_index:03d}_{mix}_{distribution}_{correlation_id}_{count_law}"
                    expected.append(
                        (
                            identifier,
                            mix,
                            distribution,
                            correlation_id,
                            rho,
                            count_law,
                            None,
                        )
                    )
                    scenario_index += 1
    expected.extend(
        (
            (
                "S048_TINY_SCALE_EQUAL_GAUSS_R085_N16",
                "EQUAL",
                "GAUSS",
                "R085",
                0.85,
                "N16",
                "all_sigmas_times_2_pow_minus_20",
            ),
            (
                "S049_SINGLE_CELL_LEVERAGE_EQUAL_T5_R085_NLINK",
                "EQUAL",
                "T5",
                "R085",
                0.85,
                "NLINK",
                "rank0_slot0_interaction_h_equals_8",
            ),
            (
                "S050_ALL_ZERO",
                "ZERO",
                "EXACT_ZERO",
                "NONE",
                0.0,
                "N16",
                "binary64_positive_zero",
            ),
            (
                "S051_NEAR_COLLINEAR_EQUAL_LOGN075_R0999_NLINK",
                "EQUAL",
                "LOGN075",
                "R0999",
                0.999,
                "NLINK",
                "near_collinear",
            ),
        )
    )
    for row, frozen in zip(scenarios, expected):
        if set(row) != {
            "scenario_id",
            "effect_mix",
            "effect_distribution",
            "correlation_id",
            "rho",
            "count_law",
            "special",
        }:
            raise ValueError("simulation scenario row is not closed-world")
        observed = (
            row["scenario_id"],
            row["effect_mix"],
            row["effect_distribution"],
            row["correlation_id"],
            float(row["rho"]),
            row["count_law"],
            row["special"],
        )
        if observed != frozen:
            raise ValueError(f"simulation scenario drift: {row['scenario_id']}")
    execution = payload.get("execution")
    if execution != EXPECTED_EXECUTION:
        raise ValueError("simulation execution counts drifted")
    if payload.get("gates") != EXPECTED_GATES:
        raise ValueError("simulation gate registry drifted")
    calibration = payload.get("mc_calibration_scenarios")
    if not isinstance(calibration, list) or len(calibration) != 24:
        raise ValueError("MC calibration registry must contain 24 scenarios")
    expected_calibration = [
        f"S{index:03d}"
        for index in (
            2,
            3,
            6,
            7,
            10,
            11,
            14,
            15,
            18,
            19,
            22,
            23,
            26,
            27,
            30,
            31,
            34,
            35,
            38,
            39,
            42,
            43,
            46,
            47,
        )
    ]
    if calibration != expected_calibration:
        raise ValueError("MC calibration scenario IDs drifted")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("simulation authorization boundary is missing")
    if authorization != {
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }:
        raise ValueError("simulation registry contains forbidden authorization")


def verify_reference_environment(registry: Mapping[str, Any]) -> dict[str, Any]:
    reference = registry["reference_environment"]
    raw = (
        np.random.Generator(np.random.Philox(int(reference["golden_seed"])))
        .bit_generator.random_raw(int(reference["golden_raw_byte_count"]) // 8)
        .astype("<u8", copy=False)
        .tobytes()
    )
    observed = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "golden_raw_byte_count": len(raw),
        "golden_raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    expected = {
        key: reference[key]
        for key in (
            "python_version",
            "numpy_version",
            "golden_raw_byte_count",
            "golden_raw_sha256",
        )
    }
    if observed != expected:
        raise RuntimeError(
            "simulation reference environment mismatch; use the exact frozen Python/NumPy/Philox runtime"
        )
    return observed


def _outer_seed(registry_sha: str, scenario_id: str, outer_index: int) -> int:
    digest = domain_hash(
        "DUCA-V21-SIM-OUTER-V1",
        sha256_bytes(registry_sha, field_name="simulation registry sha256"),
        canonical_text(scenario_id, field_name="scenario_id"),
        u64be(outer_index),
    )
    return int.from_bytes(digest[:16], "big")


def _correlation(rho: float) -> np.ndarray:
    indices = np.arange(len(METRIC_IDS), dtype=np.int64)
    return np.power(float(rho), np.abs(indices[:, None] - indices[None, :]))


def _draw_effect_vectors(
    rng: np.random.Generator,
    *,
    count: int,
    distribution: str,
    rho: float,
) -> np.ndarray:
    if distribution == "EXACT_ZERO":
        return np.zeros((count, len(METRIC_IDS)), dtype=np.float64)
    covariance = _correlation(rho)
    gaussian = rng.multivariate_normal(
        mean=np.zeros(len(METRIC_IDS), dtype=np.float64),
        cov=covariance,
        size=count,
        method="cholesky",
    ).astype(np.float64, copy=False)
    if distribution == "GAUSS":
        return gaussian
    if distribution == "T5":
        chi_square = rng.chisquare(df=5.0, size=(count, 1)).astype(np.float64)
        return math.sqrt(3.0 / 5.0) * gaussian / np.sqrt(chi_square / 5.0)
    if distribution == "LOGN075":
        lam = 0.75
        return (np.exp(lam * gaussian - lam * lam / 2.0) - 1.0) / math.sqrt(
            math.exp(lam * lam) - 1.0
        )
    raise ValueError(f"unsupported effect distribution {distribution}")


def _sigmas(
    registry: Mapping[str, Any], scenario: Mapping[str, Any]
) -> tuple[float, float, float]:
    mix = registry["effect_mixes"][scenario["effect_mix"]]
    values = []
    for key in ("sigma_v", "sigma_p", "sigma_e"):
        raw = mix[key]
        values.append(1.0 / math.sqrt(3.0) if raw == "1/sqrt(3)" else float(raw))
    if scenario.get("special") == "all_sigmas_times_2_pow_minus_20":
        values = [value * 2.0**-20 for value in values]
    return tuple(values)  # type: ignore[return-value]


def _cell_leverage(cell: Mapping[str, Any], scenario: Mapping[str, Any]) -> float:
    if scenario["count_law"] == "N16":
        value = 1.0
    else:
        rank = int(cell["canonical_video_rank"])
        process = int(cell["logical_process_index"])
        slot = int(cell["slot"])
        if cell["length_stratum"] == "natural_short":
            count = 1 + ((3 * rank + 5 * process + slot) % 4)
        else:
            count = 8 + ((7 * rank + 3 * process + slot) % 57)
        value = min(4.0, max(0.5, math.sqrt(16.0 / count)))
    if (
        scenario.get("special") == "rank0_slot0_interaction_h_equals_8"
        and int(cell["canonical_video_rank"]) == 0
        and int(cell["slot"]) == 0
    ):
        value = 8.0
    return value


def generate_outer_cells(
    *,
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    incidence: Mapping[str, Any],
    outer_index: int,
) -> list[dict[str, Any]]:
    validate_simulation_registry(registry)
    validate_incidence(incidence)
    if scenario not in registry["scenarios"]:
        raise ValueError("simulation scenario is not the frozen registry row")
    if type(outer_index) is not int or not 0 <= outer_index < 500:
        raise ValueError("simulation outer_index must be an integer in [0,499]")
    registry_sha = str(registry.get("semantic_sha256") or canonical_sha256(registry))
    rng = np.random.Generator(
        np.random.Philox(
            _outer_seed(registry_sha, str(scenario["scenario_id"]), outer_index)
        )
    )
    sigma_v, sigma_p, sigma_e = _sigmas(registry, scenario)
    output: list[dict[str, Any]] = []
    for role_id in ROLE_ORDER:
        role_cells = [
            cell for cell in incidence["cells"] if cell.get("role_id") == role_id
        ]
        if len(role_cells) != 64:
            raise ValueError(f"simulation incidence role {role_id} is not 64 cells")
        distribution = str(scenario["effect_distribution"])
        rho = float(scenario["rho"])
        video_effect = _draw_effect_vectors(
            rng, count=32, distribution=distribution, rho=rho
        )
        process_effect = _draw_effect_vectors(
            rng, count=8, distribution=distribution, rho=rho
        )
        cell_effect = _draw_effect_vectors(
            rng, count=64, distribution=distribution, rho=rho
        )
        for cell_index, cell in enumerate(role_cells):
            leverage = _cell_leverage(cell, scenario)
            value = (
                sigma_v * video_effect[int(cell["canonical_video_rank"])]
                + sigma_p * process_effect[int(cell["logical_process_index"])]
                + sigma_e * leverage * cell_effect[cell_index]
            )
            row = dict(cell)
            for metric_index, metric_id in enumerate(METRIC_IDS):
                row[metric_id] = float(value[metric_index])
            output.append(row)
    return output


def build_outer_reference_summary(
    *,
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    incidence: Mapping[str, Any],
    outer_index: int,
) -> dict[str, Any]:
    cells = generate_outer_cells(
        registry=registry,
        scenario=scenario,
        incidence=incidence,
        outer_index=outer_index,
    )
    contrast = estimate_role_contrast(
        cells=cells,
        scale_normalizers={metric_id: 1.0 for metric_id in METRIC_IDS},
        allow_signed_simulation_values=True,
    )
    payload = {
        "schema": "duca_admission_v2_1_simulation_outer_reference_v1",
        "status": "PASSED",
        "protocol_id": PROTOCOL_ID,
        "scenario_id": scenario["scenario_id"],
        "outer_index": int(outer_index),
        "cell_count": len(cells),
        "delta_hat": contrast["delta_hat"],
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def run_simulation_outer(
    *,
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    incidence: Mapping[str, Any],
    role_manifest_sha256: str,
    metric_registry_sha256: str,
    outer_index: int,
    stream_id: int = 0,
) -> dict[str, Any]:
    """Execute one registered outer dataset without consuming candidate outputs."""

    validate_simulation_registry(registry)
    validate_incidence(incidence)
    sha256_bytes(role_manifest_sha256, field_name="role manifest sha256")
    sha256_bytes(metric_registry_sha256, field_name="metric registry sha256")
    if role_manifest_sha256 != incidence["role_manifest_sha256"]:
        raise ValueError("simulation role-manifest/incidence binding drift")
    cells = generate_outer_cells(
        registry=registry,
        scenario=scenario,
        incidence=incidence,
        outer_index=outer_index,
    )
    contrast = estimate_role_contrast(
        cells=cells,
        scale_normalizers={metric_id: 1.0 for metric_id in METRIC_IDS},
        allow_signed_simulation_values=True,
    )
    if scenario["scenario_id"] == "S050_ALL_ZERO":
        result = finalize_max_t(
            delta_hat=contrast["delta_hat"],
            replicates=[],
            exact_zero=contrast["exact_zero"],
        )
        endpoint = build_endpoint_summary(
            max_t_result=result,
            mc_certificate={"status": "PASSED_EXACT_ZERO"},
            secondary_diagnostic=None,
        )
        return {"outer_index": outer_index, "endpoints": {"NULL": endpoint}}

    simulation_registry_sha = str(
        registry.get("semantic_sha256") or canonical_sha256(registry)
    )
    registry_hashes = (
        simulation_registry_sha,
        role_manifest_sha256,
        incidence["content_sha256"],
        metric_registry_sha256,
    )

    def generator_for(inner_stream_id: int):
        def generate(start: int, stop: int) -> list[dict[str, float]]:
            return multiplier_replicates(
                role_cells=contrast["role_cells"],
                residuals=contrast["residuals"],
                registry_hashes=registry_hashes,
                stream_id=inner_stream_id,
                replicate_start=start,
                replicate_stop=stop,
            )

        return generate

    generator = generator_for(stream_id)
    diagnostic_generator = generator_for(stream_id + 1)

    variance = true_contrast_variance(
        registry=registry, scenario=scenario, incidence=incidence
    )
    endpoints = {}
    for profile_id in SHIFT_PROFILES:
        truth = shift_truth(
            profile_id=profile_id, true_variance=variance, registry=registry
        )
        shifted_delta = {
            metric_id: float(contrast["delta_hat"][metric_id] + truth[metric_id])
            for metric_id in METRIC_IDS
        }
        mc = run_prefix_extensible_mc(
            generator=generator,
            diagnostic_generator=diagnostic_generator,
            delta_hat=shifted_delta,
            exact_zero=contrast["exact_zero"],
        )
        endpoints[profile_id] = build_endpoint_summary(
            max_t_result=mc["result"],
            mc_certificate=mc["certificate"],
            secondary_diagnostic=mc["secondary_stream_diagnostic"],
        )
    return {"outer_index": outer_index, "endpoints": endpoints}


def _finite_metric_map(
    value: Any, *, label: str, nonnegative: bool = False
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(METRIC_IDS):
        raise ValueError(f"{label} does not match the 12-metric registry")
    output: dict[str, float] = {}
    for metric_id in METRIC_IDS:
        item = value[metric_id]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"{label}:{metric_id} must be finite binary64")
        if nonnegative and item < 0.0:
            raise ValueError(f"{label}:{metric_id} must be nonnegative")
        output[metric_id] = item
    return output


def build_endpoint_summary(
    *,
    max_t_result: Mapping[str, Any],
    mc_certificate: Mapping[str, Any],
    secondary_diagnostic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reduce one recomputed endpoint to the closed simulation-gate surface."""

    summary = {
        "numeric_tail_passed": max_t_result.get("numeric_tail_passed"),
        "q_plus": max_t_result.get("q_plus"),
        "q_minus": max_t_result.get("q_minus"),
        "scales": max_t_result.get("scales"),
        "lower": max_t_result.get("lower"),
        "upper": max_t_result.get("upper"),
        "mc_status": mc_certificate.get("status"),
        "replicate_count": max_t_result.get("replicate_count", 0),
        "secondary_diagnostic": (
            None if secondary_diagnostic is None else dict(secondary_diagnostic)
        ),
    }
    validate_endpoint_summary(summary)
    return summary


def validate_endpoint_summary(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "numeric_tail_passed",
        "q_plus",
        "q_minus",
        "scales",
        "lower",
        "upper",
        "mc_status",
        "replicate_count",
        "secondary_diagnostic",
    }:
        raise ValueError("simulation endpoint summary is not closed-world")
    if type(payload.get("numeric_tail_passed")) is not bool:
        raise ValueError("numeric-tail decision must be boolean")
    for field in ("q_plus", "q_minus"):
        if type(payload.get(field)) is not float or not math.isfinite(payload[field]):
            raise ValueError(
                "simulation endpoint critical values must be finite binary64"
            )
    _finite_metric_map(payload.get("scales"), label="endpoint scale", nonnegative=True)
    _finite_metric_map(payload.get("lower"), label="endpoint lower")
    _finite_metric_map(payload.get("upper"), label="endpoint upper")
    if payload.get("mc_status") not in {"PASSED", "FAILED_CLOSED", "PASSED_EXACT_ZERO"}:
        raise ValueError("simulation endpoint MC status is invalid")
    if type(payload.get("replicate_count")) is not int or payload[
        "replicate_count"
    ] not in {
        0,
        100_000,
        200_000,
    }:
        raise ValueError("simulation endpoint replicate count is not registered")
    diagnostic = payload.get("secondary_diagnostic")
    if diagnostic is not None:
        if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
            "replicate_count",
            "passed",
            "binary_pass_vector",
        }:
            raise ValueError("secondary stream diagnostic is not closed-world")
        if (
            diagnostic.get("replicate_count") != 100_000
            or type(diagnostic.get("passed")) is not bool
        ):
            raise ValueError("secondary stream diagnostic identity is invalid")
        vector = diagnostic.get("binary_pass_vector")
        if not isinstance(vector, list) or any(
            type(value) is not bool for value in vector
        ):
            raise ValueError("secondary stream diagnostic pass vector is invalid")


def true_contrast_variance(
    *,
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    incidence: Mapping[str, Any],
) -> dict[str, float]:
    """Exact marginal contrast variance on the frozen calibration/holdout grids."""

    validate_simulation_registry(registry)
    validate_incidence(incidence)
    if scenario not in registry["scenarios"]:
        raise ValueError("true-variance scenario is not registered")
    sigma_v, sigma_p, sigma_e = _sigmas(registry, scenario)
    role_variances = []
    for role_id in ("calibration", "admission_holdout"):
        cells = [cell for cell in incidence["cells"] if cell["role_id"] == role_id]
        video_degrees: dict[str, int] = {}
        process_degrees: dict[int, int] = {}
        for cell in cells:
            video_degrees[cell["video_id"]] = video_degrees.get(cell["video_id"], 0) + 1
            process = cell["logical_process_index"]
            process_degrees[process] = process_degrees.get(process, 0) + 1
        covariance_sum = (
            sigma_v
            * sigma_v
            * math.fsum(value * value for value in video_degrees.values())
            + sigma_p
            * sigma_p
            * math.fsum(value * value for value in process_degrees.values())
            + sigma_e
            * sigma_e
            * math.fsum(_cell_leverage(cell, scenario) ** 2 for cell in cells)
        )
        role_variances.append(covariance_sum / (64.0 * 64.0))
    contrast_variance = math.fsum(role_variances)
    if not math.isfinite(contrast_variance) or contrast_variance < 0.0:
        raise ValueError("registered true contrast variance is invalid")
    return {metric_id: float(contrast_variance) for metric_id in METRIC_IDS}


def shift_truth(
    *,
    profile_id: str,
    true_variance: Mapping[str, float],
    registry: Mapping[str, Any],
) -> dict[str, float]:
    if profile_id not in SHIFT_PROFILES:
        raise ValueError("unregistered shift profile")
    _finite_metric_map(true_variance, label="true variance", nonnegative=True)
    profile = registry["shift_profiles"][profile_id]
    affected = (
        set(METRIC_IDS) if profile["metrics"] == "all" else set(profile["metrics"])
    )
    multiplier = float(profile["multiplier_true_se"])
    return {
        metric_id: float(multiplier * math.sqrt(true_variance[metric_id]))
        if metric_id in affected
        else 0.0
        for metric_id in METRIC_IDS
    }


def _beta_cdf_integer(x: float, a: int, b: int) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    n = a + b - 1
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * math.log(x)
        + (n - index) * math.log1p(-x)
        for index in range(a, n + 1)
    ]
    maximum = max(logs)
    return min(
        1.0, math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs)
    )


def _beta_quantile_integer(probability: float, a: int, b: int) -> float:
    if not 0.0 < probability < 1.0 or a <= 0 or b <= 0:
        raise ValueError("invalid integer beta quantile arguments")
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if _beta_cdf_integer(middle, a, b) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def clopper_pearson_onesided_99(successes: int, denominator: int) -> dict[str, float]:
    if type(successes) is not int or type(denominator) is not int:
        raise TypeError("Clopper-Pearson counts must be integers")
    if denominator <= 0 or not 0 <= successes <= denominator:
        raise ValueError("Clopper-Pearson counts are out of range")
    lower = (
        0.0
        if successes == 0
        else _beta_quantile_integer(0.01, successes, denominator - successes + 1)
    )
    upper = (
        1.0
        if successes == denominator
        else _beta_quantile_integer(0.99, successes + 1, denominator - successes)
    )
    return {"lower_99": lower, "upper_99": upper}


def _check(
    name: str, value: Any, comparator: str, limit: Any, passed: bool
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "comparator": comparator,
        "limit": limit,
        "passed": bool(passed),
    }


def _validate_protocol_only_receipt_fields(
    payload: Mapping[str, Any], *, expected_failure_code: str
) -> None:
    if (
        payload.get("authorization_scope") != "NONE"
        or payload.get("phase1_v2_authorized") is not False
        or payload.get("holdout_open_authorized") is not False
        or payload.get("paper_claim_allowed") is not False
        or payload.get("official_final_sealed") is not True
    ):
        raise ValueError("simulation receipt contains forbidden authorization")
    status = payload.get("status")
    failure_codes = payload.get("failure_codes")
    if status not in {"PASSED", "FAILED_CLOSED"} or not isinstance(failure_codes, list):
        raise ValueError("simulation receipt status/failure codes are invalid")
    if status == "PASSED" and failure_codes != []:
        raise ValueError("PASSED simulation receipt contains failure codes")
    if status == "FAILED_CLOSED" and failure_codes != [expected_failure_code]:
        raise ValueError("FAILED_CLOSED simulation receipt has the wrong failure code")


def _validate_check_rows(checks: Any) -> None:
    if not isinstance(checks, list) or not checks:
        raise ValueError("simulation receipt checks are missing")
    for row in checks:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "value",
            "comparator",
            "limit",
            "passed",
        }:
            raise ValueError("simulation receipt check row is not closed-world")
        if not isinstance(row["name"], str) or row["comparator"] not in {">=", "<="}:
            raise ValueError("simulation receipt check identity is invalid")
        if type(row["passed"]) is not bool:
            raise ValueError("simulation receipt check decision must be boolean")


def validate_simulation_scenario_receipt(
    payload: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> None:
    verify_content_sha256(payload)
    common = {
        "schema",
        "status",
        "protocol_id",
        "scenario_id",
        "outer_count",
        "checks",
        "count_intervals_onesided_99",
        "failure_codes",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "official_final_sealed",
        "content_sha256",
    }
    scenario_id = payload.get("scenario_id")
    expected_keys = (
        common
        if scenario_id == "S050_ALL_ZERO"
        else common
        | {
            "true_contrast_variance",
            "width_summaries",
        }
    )
    if set(payload) != expected_keys:
        raise ValueError("simulation scenario receipt is not closed-world")
    if payload.get("schema") != SIMULATION_SCENARIO_RECEIPT_SCHEMA:
        raise ValueError("simulation scenario receipt schema drifted")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("outer_count") != 500:
        raise ValueError("simulation scenario receipt identity drifted")
    if scenario_id not in {row["scenario_id"] for row in registry["scenarios"]}:
        raise ValueError("simulation scenario receipt ID is not registered")
    _validate_protocol_only_receipt_fields(
        payload, expected_failure_code="SIMULATION_GATE_FAILED"
    )
    _validate_check_rows(payload.get("checks"))
    checks = payload["checks"]
    if scenario_id == "S050_ALL_ZERO":
        expected_specs = [
            (
                "exact_zero_branch_count",
                ">=",
                registry["gates"]["exact_zero_S050"]["exact_zero_branch_min_count"],
                "count",
            )
        ]
    else:
        nonzero_gate = registry["gates"]["nonzero_scenario"]
        power_gate = registry["gates"]["power"]
        width_gate = (
            registry["gates"]["ordinary_S000_S047"]
            if int(str(scenario_id)[1:4]) <= 47
            else registry["gates"]["boundary_S048_S049_S051"]
        )
        expected_specs = [
            (
                "simultaneous_upper_coverage_count",
                ">=",
                nonzero_gate["simultaneous_upper_coverage_min_count"],
                "count",
            ),
            (
                "simultaneous_lower_coverage_count",
                ">=",
                nonzero_gate["simultaneous_lower_coverage_min_count"],
                "count",
            ),
            (
                "null_false_alarm_count",
                "<=",
                nonzero_gate["null_false_alarm_max_count"],
                "count",
            ),
            (
                "mc_unstable_count",
                "<=",
                nonzero_gate["mc_unstable_max_count"],
                "count",
            ),
            (
                "SAFE_ALL_M6_count",
                ">=",
                power_gate["SAFE_ALL_M6_pass_min_count"],
                "count",
            ),
            (
                "HARM_ONE_P4_count",
                ">=",
                power_gate["HARM_ONE_P4_alarm_min_count"],
                "count",
            ),
            (
                "HARM_ONE_P6_count",
                ">=",
                power_gate["HARM_ONE_P6_alarm_min_count"],
                "count",
            ),
            (
                "HARM_ALL_P3_count",
                ">=",
                power_gate["HARM_ALL_P3_alarm_min_count"],
                "count",
            ),
            *[
                (name, "<=", width_gate[f"{name}_max"], "float")
                for name in (
                    "median_vir",
                    "type1_p95_vir",
                    "median_nwr",
                    "type1_p95_nwr",
                )
            ],
        ]
    if len(checks) != len(expected_specs):
        raise ValueError("simulation scenario check family drifted")
    for row, (name, comparator, limit, value_type) in zip(checks, expected_specs):
        value = row["value"]
        if value_type == "count":
            valid_value = type(value) is int and 0 <= value <= 500
        else:
            valid_value = type(value) is float and math.isfinite(value)
        expected_pass = value >= limit if comparator == ">=" else value <= limit
        if (
            row["name"] != name
            or row["comparator"] != comparator
            or row["limit"] != limit
            or not valid_value
            or row["passed"] is not expected_pass
        ):
            raise ValueError("simulation scenario check drifted")
    passed = all(row["passed"] for row in payload["checks"])
    if (payload["status"] == "PASSED") is not passed:
        raise ValueError("simulation scenario receipt status disagrees with checks")
    intervals = payload.get("count_intervals_onesided_99")
    if not isinstance(intervals, Mapping) or not intervals:
        raise ValueError("simulation scenario receipt count intervals are missing")
    for interval in intervals.values():
        if (
            not isinstance(interval, Mapping)
            or set(interval) != {"lower_99", "upper_99"}
            or any(
                type(interval[field]) is not float
                or not math.isfinite(interval[field])
                or not 0.0 <= interval[field] <= 1.0
                for field in ("lower_99", "upper_99")
            )
        ):
            raise ValueError("simulation scenario count interval is invalid")
    count_checks = {
        row["name"]: row["value"]
        for row, spec in zip(checks, expected_specs)
        if spec[3] == "count"
    }
    expected_interval_counts = (
        {"exact_zero_branch": count_checks["exact_zero_branch_count"]}
        if scenario_id == "S050_ALL_ZERO"
        else {
            "simultaneous_upper_coverage": count_checks[
                "simultaneous_upper_coverage_count"
            ],
            "simultaneous_lower_coverage": count_checks[
                "simultaneous_lower_coverage_count"
            ],
            "null_false_alarm": count_checks["null_false_alarm_count"],
            "mc_unstable": count_checks["mc_unstable_count"],
            "SAFE_ALL_M6": count_checks["SAFE_ALL_M6_count"],
            "HARM_ONE_P4": count_checks["HARM_ONE_P4_count"],
            "HARM_ONE_P6": count_checks["HARM_ONE_P6_count"],
            "HARM_ALL_P3": count_checks["HARM_ALL_P3_count"],
        }
    )
    if set(intervals) != set(expected_interval_counts):
        raise ValueError("simulation scenario interval family drifted")
    for name, count in expected_interval_counts.items():
        if intervals[name] != clopper_pearson_onesided_99(count, 500):
            raise ValueError("simulation scenario interval does not match its count")
    if scenario_id != "S050_ALL_ZERO":
        variance = _finite_metric_map(
            payload.get("true_contrast_variance"),
            label="scenario receipt true variance",
            nonnegative=True,
        )
        if any(value <= 0.0 for value in variance.values()):
            raise ValueError("nonzero scenario receipt has zero true variance")
        widths = payload.get("width_summaries")
        if not isinstance(widths, Mapping) or set(widths) != {
            "median_vir",
            "type1_p95_vir",
            "median_nwr",
            "type1_p95_nwr",
        }:
            raise ValueError("simulation scenario width summaries are invalid")
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in widths.values()
        ):
            raise ValueError("simulation scenario width summary is nonfinite")
        width_check_map = {row["name"]: row["value"] for row in checks}
        if any(widths[name] != width_check_map[name] for name in widths):
            raise ValueError("simulation scenario width summary/check drift")


def validate_mc_calibration_scenario_receipt(
    payload: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> None:
    verify_content_sha256(payload)
    if set(payload) != {
        "schema",
        "status",
        "protocol_id",
        "scenario_id",
        "operational_stream_count",
        "coverage_counts",
        "checks",
        "failure_codes",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "official_final_sealed",
        "content_sha256",
    }:
        raise ValueError("MC calibration scenario receipt is not closed-world")
    if (
        payload.get("schema") != MC_CALIBRATION_SCENARIO_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("operational_stream_count") != 200
        or payload.get("scenario_id") not in registry["mc_calibration_scenarios"]
    ):
        raise ValueError("MC calibration scenario receipt identity drifted")
    _validate_protocol_only_receipt_fields(
        payload, expected_failure_code="MC_HALF_WIDTH_CALIBRATION_FAILED"
    )
    _validate_check_rows(payload.get("checks"))
    counts = payload.get("coverage_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(MC_PARAMETER_IDS):
        raise ValueError("MC calibration coverage counts drifted")
    if any(
        type(value) is not int or not 0 <= value <= 200 for value in counts.values()
    ):
        raise ValueError("MC calibration coverage count is invalid")
    gate = registry["gates"]["mc_half_width"]
    expected_specs = []
    for parameter_id in MC_PARAMETER_IDS:
        expected_specs.extend(
            (
                (
                    f"coverage:{parameter_id}",
                    ">=",
                    gate["scenario_parameter_coverage_min_count"],
                    "count",
                ),
                (
                    f"reference_half_difference:{parameter_id}",
                    "<=",
                    gate["half_difference_over_median_operational_h_max"],
                    "float",
                ),
                (
                    f"median_h200_over_h100:{parameter_id}",
                    "<=",
                    gate["median_h200k_over_h100k_max"],
                    "float",
                ),
                (
                    f"type1_p95_h200_over_h100:{parameter_id}",
                    "<=",
                    gate["type1_p95_h200k_over_h100k_max"],
                    "float",
                ),
            )
        )
    checks = payload["checks"]
    if len(checks) != len(expected_specs):
        raise ValueError("MC calibration scenario check family drifted")
    for row, (name, comparator, limit, value_type) in zip(checks, expected_specs):
        value = row["value"]
        valid_value = (
            type(value) is int and 0 <= value <= 200
            if value_type == "count"
            else type(value) is float and math.isfinite(value)
        )
        expected_pass = value >= limit if comparator == ">=" else value <= limit
        if (
            row["name"] != name
            or row["comparator"] != comparator
            or row["limit"] != limit
            or not valid_value
            or row["passed"] is not expected_pass
        ):
            raise ValueError("MC calibration scenario check drifted")
    for parameter_id in MC_PARAMETER_IDS:
        coverage_row = checks[4 * MC_PARAMETER_IDS.index(parameter_id)]
        if coverage_row["value"] != counts[parameter_id]:
            raise ValueError("MC calibration coverage count/check drift")
    passed = all(row["passed"] for row in payload["checks"])
    if (payload["status"] == "PASSED") is not passed:
        raise ValueError("MC calibration scenario receipt status disagrees with checks")


def _roundoff_close(left: float, right: float, *scales: float) -> bool:
    tolerance = 64.0 * max(
        math.ulp(value) for value in (left, right, *scales) if math.isfinite(value)
    )
    return abs(left - right) <= tolerance


def _endpoint_observed(payload: Mapping[str, Any]) -> dict[str, float]:
    observed = {}
    for metric_id in METRIC_IDS:
        scale = payload["scales"][metric_id]
        from_lower = payload["lower"][metric_id] + payload["q_plus"] * scale
        from_upper = payload["upper"][metric_id] - payload["q_minus"] * scale
        if not _roundoff_close(from_lower, from_upper, scale):
            raise ValueError(
                "simulation endpoint bounds do not reconstruct one contrast"
            )
        observed[metric_id] = (from_lower + from_upper) / 2.0
    active = [
        metric_id for metric_id in METRIC_IDS if payload["scales"][metric_id] > 0.0
    ]
    if active:
        statistic = max(
            observed[metric_id] / payload["scales"][metric_id] for metric_id in active
        )
        critical = payload["q_plus"]
        if not _roundoff_close(statistic, critical, critical):
            if payload["numeric_tail_passed"] is not (statistic <= critical):
                raise ValueError(
                    "simulation endpoint numeric-tail decision is inconsistent"
                )
    return observed


def evaluate_simulation_scenario(
    *,
    registry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    incidence: Mapping[str, Any],
    outer_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply all frozen integer/width/power gates to 500 sealed outer summaries."""

    validate_simulation_registry(registry)
    validate_incidence(incidence)
    if scenario not in registry["scenarios"]:
        raise ValueError("simulation scenario gate received an unregistered scenario")
    if not isinstance(outer_records, Sequence) or isinstance(
        outer_records, (str, bytes)
    ):
        raise ValueError("outer simulation records must be a sequence")
    if len(outer_records) != 500:
        raise ValueError("simulation scenario gate requires exactly 500 outer records")
    expected_profiles = (
        {"NULL"} if scenario["scenario_id"] == "S050_ALL_ZERO" else set(SHIFT_PROFILES)
    )
    records: list[Mapping[str, Any]] = []
    for expected_index, row in enumerate(outer_records):
        if not isinstance(row, Mapping) or set(row) != {"outer_index", "endpoints"}:
            raise ValueError("outer simulation record is not closed-world")
        if row.get("outer_index") != expected_index:
            raise ValueError("outer simulation records are not in frozen 0..499 order")
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != expected_profiles:
            raise ValueError("outer simulation endpoint family drifted")
        for endpoint in endpoints.values():
            if not isinstance(endpoint, Mapping):
                raise ValueError("outer simulation endpoint must be an object")
            validate_endpoint_summary(endpoint)
            if (
                scenario["scenario_id"] != "S050_ALL_ZERO"
                and endpoint["secondary_diagnostic"] is None
            ):
                raise ValueError(
                    "nonzero simulation endpoint lacks the stream-1 diagnostic"
                )
        records.append(row)

    scenario_id = scenario["scenario_id"]
    checks: list[dict[str, Any]] = []
    if scenario_id == "S050_ALL_ZERO":
        exact_count = 0
        for row in records:
            endpoint = row["endpoints"]["NULL"]
            exact = (
                endpoint["mc_status"] == "PASSED_EXACT_ZERO"
                and endpoint["replicate_count"] == 0
                and endpoint["numeric_tail_passed"] is True
                and is_binary64_positive_zero(endpoint["q_plus"])
                and is_binary64_positive_zero(endpoint["q_minus"])
                and all(
                    is_binary64_positive_zero(endpoint[field][metric_id])
                    for field in ("scales", "lower", "upper")
                    for metric_id in METRIC_IDS
                )
            )
            exact_count += int(exact)
        limit = registry["gates"]["exact_zero_S050"]["exact_zero_branch_min_count"]
        checks.append(
            _check(
                "exact_zero_branch_count",
                exact_count,
                ">=",
                limit,
                exact_count >= limit,
            )
        )
        passed = all(row["passed"] for row in checks)
        return with_content_sha256(
            {
                "schema": SIMULATION_SCENARIO_RECEIPT_SCHEMA,
                "status": "PASSED" if passed else "FAILED_CLOSED",
                "protocol_id": PROTOCOL_ID,
                "scenario_id": scenario_id,
                "outer_count": 500,
                "checks": checks,
                "count_intervals_onesided_99": {
                    "exact_zero_branch": clopper_pearson_onesided_99(exact_count, 500)
                },
                "failure_codes": [] if passed else ["SIMULATION_GATE_FAILED"],
                "authorization_scope": "NONE",
                "phase1_v2_authorized": False,
                "holdout_open_authorized": False,
                "paper_claim_allowed": False,
                "official_final_sealed": True,
            }
        )

    true_variance = true_contrast_variance(
        registry=registry, scenario=scenario, incidence=incidence
    )
    true_se = {
        metric_id: math.sqrt(value) for metric_id, value in true_variance.items()
    }
    truths = {
        profile_id: shift_truth(
            profile_id=profile_id, true_variance=true_variance, registry=registry
        )
        for profile_id in SHIFT_PROFILES
    }
    null_truth = truths["NULL"]
    upper_coverage = 0
    lower_coverage = 0
    null_alarm = 0
    mc_unstable = 0
    power_counts = {
        profile_id: 0 for profile_id in SHIFT_PROFILES if profile_id != "NULL"
    }
    vir_values: list[float] = []
    nwr_values: list[float] = []
    for row in records:
        endpoints = row["endpoints"]
        null = endpoints["NULL"]
        observed_by_profile = {
            profile_id: _endpoint_observed(endpoint)
            for profile_id, endpoint in endpoints.items()
        }
        null_observed = observed_by_profile["NULL"]
        for profile_id in SHIFT_PROFILES:
            for metric_id in METRIC_IDS:
                observed_shift = (
                    observed_by_profile[profile_id][metric_id]
                    - null_observed[metric_id]
                )
                expected_shift = truths[profile_id][metric_id]
                if not _roundoff_close(
                    observed_shift,
                    expected_shift,
                    observed_by_profile[profile_id][metric_id],
                    null_observed[metric_id],
                ):
                    raise ValueError(
                        "simulation endpoint does not implement its frozen shift"
                    )
        stable = all(
            endpoint["mc_status"] != "FAILED_CLOSED" for endpoint in endpoints.values()
        )
        mc_unstable += int(not stable)
        upper_coverage += int(
            null["mc_status"] != "FAILED_CLOSED"
            and all(
                null_truth[metric_id] <= null["upper"][metric_id]
                for metric_id in METRIC_IDS
            )
        )
        lower_coverage += int(
            null["mc_status"] != "FAILED_CLOSED"
            and all(
                null["lower"][metric_id] <= null_truth[metric_id]
                for metric_id in METRIC_IDS
            )
        )
        null_alarm += int(not null["numeric_tail_passed"])
        for profile_id in power_counts:
            endpoint = endpoints[profile_id]
            if profile_id == "SAFE_ALL_M6":
                power_counts[profile_id] += int(
                    endpoint["mc_status"] != "FAILED_CLOSED"
                    and endpoint["numeric_tail_passed"]
                )
            else:
                power_counts[profile_id] += int(
                    endpoint["mc_status"] != "FAILED_CLOSED"
                    and not endpoint["numeric_tail_passed"]
                )
        critical = max(null["q_plus"], null["q_minus"])
        for metric_id in METRIC_IDS:
            variance = true_variance[metric_id]
            scale = null["scales"][metric_id]
            if variance <= 0.0 or scale <= 0.0:
                raise ValueError(
                    "nonzero scenario produced nonpositive variance or scale"
                )
            vir_values.append(scale * scale / variance)
            nwr_values.append(
                critical * scale / (BONFERRONI_REFERENCE_CRITICAL * true_se[metric_id])
            )
    gate = registry["gates"]["nonzero_scenario"]
    checks.extend(
        (
            _check(
                "simultaneous_upper_coverage_count",
                upper_coverage,
                ">=",
                gate["simultaneous_upper_coverage_min_count"],
                upper_coverage >= gate["simultaneous_upper_coverage_min_count"],
            ),
            _check(
                "simultaneous_lower_coverage_count",
                lower_coverage,
                ">=",
                gate["simultaneous_lower_coverage_min_count"],
                lower_coverage >= gate["simultaneous_lower_coverage_min_count"],
            ),
            _check(
                "null_false_alarm_count",
                null_alarm,
                "<=",
                gate["null_false_alarm_max_count"],
                null_alarm <= gate["null_false_alarm_max_count"],
            ),
            _check(
                "mc_unstable_count",
                mc_unstable,
                "<=",
                gate["mc_unstable_max_count"],
                mc_unstable <= gate["mc_unstable_max_count"],
            ),
        )
    )
    power_gate = registry["gates"]["power"]
    for profile_id, count in power_counts.items():
        key = (
            f"{profile_id}_pass_min_count"
            if profile_id == "SAFE_ALL_M6"
            else f"{profile_id}_alarm_min_count"
        )
        checks.append(
            _check(
                f"{profile_id}_count",
                count,
                ">=",
                power_gate[key],
                count >= power_gate[key],
            )
        )
    width_gate = (
        registry["gates"]["ordinary_S000_S047"]
        if int(scenario_id[1:4]) <= 47
        else registry["gates"]["boundary_S048_S049_S051"]
    )
    width_values = {
        "median_vir": type1_quantile(vir_values, 0.5),
        "type1_p95_vir": type1_quantile(vir_values, 0.95),
        "median_nwr": type1_quantile(nwr_values, 0.5),
        "type1_p95_nwr": type1_quantile(nwr_values, 0.95),
    }
    for name, value in width_values.items():
        limit = width_gate[f"{name}_max"]
        checks.append(_check(name, value, "<=", limit, value <= limit))
    passed = all(row["passed"] for row in checks)
    return with_content_sha256(
        {
            "schema": SIMULATION_SCENARIO_RECEIPT_SCHEMA,
            "status": "PASSED" if passed else "FAILED_CLOSED",
            "protocol_id": PROTOCOL_ID,
            "scenario_id": scenario_id,
            "outer_count": 500,
            "true_contrast_variance": true_variance,
            "checks": checks,
            "width_summaries": width_values,
            "count_intervals_onesided_99": {
                "simultaneous_upper_coverage": clopper_pearson_onesided_99(
                    upper_coverage, 500
                ),
                "simultaneous_lower_coverage": clopper_pearson_onesided_99(
                    lower_coverage, 500
                ),
                "null_false_alarm": clopper_pearson_onesided_99(null_alarm, 500),
                "mc_unstable": clopper_pearson_onesided_99(mc_unstable, 500),
                **{
                    profile_id: clopper_pearson_onesided_99(count, 500)
                    for profile_id, count in power_counts.items()
                },
            },
            "failure_codes": [] if passed else ["SIMULATION_GATE_FAILED"],
            "authorization_scope": "NONE",
            "phase1_v2_authorized": False,
            "holdout_open_authorized": False,
            "paper_claim_allowed": False,
            "official_final_sealed": True,
        }
    )


def _finite_parameter_map(
    value: Any, *, label: str, nonnegative: bool = False
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(MC_PARAMETER_IDS):
        raise ValueError(f"{label} does not match the 26 MC parameters")
    output = {}
    for parameter_id in MC_PARAMETER_IDS:
        item = value[parameter_id]
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"{label}:{parameter_id} must be finite binary64")
        if nonnegative and item < 0.0:
            raise ValueError(f"{label}:{parameter_id} must be nonnegative")
        output[parameter_id] = item
    return output


def evaluate_mc_calibration_scenario(
    *,
    registry: Mapping[str, Any],
    scenario_id: str,
    operational_streams: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    validate_simulation_registry(registry)
    if scenario_id not in registry["mc_calibration_scenarios"]:
        raise ValueError(
            "MC calibration scenario is not in the frozen 24-scenario registry"
        )
    if len(operational_streams) != 200:
        raise ValueError("MC calibration requires exactly 200 operational streams")
    if set(reference) != {"combined_4m", "half0_2m", "half1_2m"}:
        raise ValueError("MC calibration reference is not closed-world")
    references = {
        key: _finite_parameter_map(value, label=f"reference:{key}")
        for key, value in reference.items()
    }
    streams = []
    for index, row in enumerate(operational_streams):
        if not isinstance(row, Mapping) or set(row) != {
            "stream_index",
            "estimate_100k",
            "half_width_100k",
            "estimate_200k",
            "half_width_200k",
        }:
            raise ValueError("MC operational stream is not closed-world")
        if row.get("stream_index") != index:
            raise ValueError("MC operational streams are not in frozen 0..199 order")
        streams.append(
            {
                "estimate_100k": _finite_parameter_map(
                    row["estimate_100k"], label="estimate_100k"
                ),
                "half_width_100k": _finite_parameter_map(
                    row["half_width_100k"], label="half_width_100k", nonnegative=True
                ),
                "estimate_200k": _finite_parameter_map(
                    row["estimate_200k"], label="estimate_200k"
                ),
                "half_width_200k": _finite_parameter_map(
                    row["half_width_200k"], label="half_width_200k", nonnegative=True
                ),
            }
        )
    gate = registry["gates"]["mc_half_width"]
    coverage_counts = {}
    parameter_checks: list[dict[str, Any]] = []
    for parameter_id in MC_PARAMETER_IDS:
        combined = references["combined_4m"][parameter_id]
        coverage = sum(
            abs(row["estimate_200k"][parameter_id] - combined)
            <= row["half_width_200k"][parameter_id]
            for row in streams
        )
        coverage_counts[parameter_id] = int(coverage)
        h100 = [row["half_width_100k"][parameter_id] for row in streams]
        h200 = [row["half_width_200k"][parameter_id] for row in streams]
        if any(value <= 0.0 for value in h100):
            raise ValueError("MC calibration h100 must be strictly positive")
        ratios = [right / left for left, right in zip(h100, h200)]
        median_h200 = type1_quantile(h200, 0.5)
        if median_h200 <= 0.0:
            raise ValueError("MC calibration median h200 must be strictly positive")
        half_difference_ratio = (
            abs(
                references["half0_2m"][parameter_id]
                - references["half1_2m"][parameter_id]
            )
            / median_h200
        )
        parameter_checks.extend(
            (
                _check(
                    f"coverage:{parameter_id}",
                    coverage,
                    ">=",
                    gate["scenario_parameter_coverage_min_count"],
                    coverage >= gate["scenario_parameter_coverage_min_count"],
                ),
                _check(
                    f"reference_half_difference:{parameter_id}",
                    half_difference_ratio,
                    "<=",
                    gate["half_difference_over_median_operational_h_max"],
                    half_difference_ratio
                    <= gate["half_difference_over_median_operational_h_max"],
                ),
                _check(
                    f"median_h200_over_h100:{parameter_id}",
                    type1_quantile(ratios, 0.5),
                    "<=",
                    gate["median_h200k_over_h100k_max"],
                    type1_quantile(ratios, 0.5) <= gate["median_h200k_over_h100k_max"],
                ),
                _check(
                    f"type1_p95_h200_over_h100:{parameter_id}",
                    type1_quantile(ratios, 0.95),
                    "<=",
                    gate["type1_p95_h200k_over_h100k_max"],
                    type1_quantile(ratios, 0.95)
                    <= gate["type1_p95_h200k_over_h100k_max"],
                ),
            )
        )
    passed = all(row["passed"] for row in parameter_checks)
    return with_content_sha256(
        {
            "schema": MC_CALIBRATION_SCENARIO_SCHEMA,
            "status": "PASSED" if passed else "FAILED_CLOSED",
            "protocol_id": PROTOCOL_ID,
            "scenario_id": scenario_id,
            "operational_stream_count": 200,
            "coverage_counts": coverage_counts,
            "checks": parameter_checks,
            "failure_codes": [] if passed else ["MC_HALF_WIDTH_CALIBRATION_FAILED"],
            "authorization_scope": "NONE",
            "phase1_v2_authorized": False,
            "holdout_open_authorized": False,
            "paper_claim_allowed": False,
            "official_final_sealed": True,
        }
    )


def evaluate_mc_calibration_registry(
    *,
    registry: Mapping[str, Any],
    scenario_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_simulation_registry(registry)
    if len(scenario_receipts) != 24:
        raise ValueError(
            "MC calibration registry requires exactly 24 scenario receipts"
        )
    expected_ids = list(registry["mc_calibration_scenarios"])
    observed_ids = []
    for receipt in scenario_receipts:
        validate_mc_calibration_scenario_receipt(receipt, registry=registry)
        observed_ids.append(receipt.get("scenario_id"))
    if observed_ids != expected_ids:
        raise ValueError("MC calibration scenario receipts are not in frozen order")
    gate = registry["gates"]["mc_half_width"]
    pooled_checks = []
    for parameter_id in MC_PARAMETER_IDS:
        count = sum(
            receipt["coverage_counts"][parameter_id] for receipt in scenario_receipts
        )
        pooled_checks.append(
            _check(
                f"pooled_coverage:{parameter_id}",
                count,
                ">=",
                gate["pooled_parameter_coverage_min_count"],
                count >= gate["pooled_parameter_coverage_min_count"],
            )
        )
    passed = all(
        receipt.get("status") == "PASSED" for receipt in scenario_receipts
    ) and all(row["passed"] for row in pooled_checks)
    return with_content_sha256(
        {
            "schema": MC_CALIBRATION_REGISTRY_SCHEMA,
            "status": "PASSED" if passed else "FAILED_CLOSED",
            "protocol_id": PROTOCOL_ID,
            "scenario_count": 24,
            "checks": pooled_checks,
            "failure_codes": [] if passed else ["MC_HALF_WIDTH_CALIBRATION_FAILED"],
            "authorization_scope": "NONE",
            "phase1_v2_authorized": False,
            "holdout_open_authorized": False,
            "paper_claim_allowed": False,
            "official_final_sealed": True,
        }
    )


def validate_mc_calibration_registry_receipt(
    payload: Mapping[str, Any], *, registry: Mapping[str, Any]
) -> None:
    validate_simulation_registry(registry)
    verify_content_sha256(payload)
    if set(payload) != {
        "schema",
        "status",
        "protocol_id",
        "scenario_count",
        "checks",
        "failure_codes",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "official_final_sealed",
        "content_sha256",
    }:
        raise ValueError("MC calibration registry receipt is not closed-world")
    if (
        payload.get("schema") != MC_CALIBRATION_REGISTRY_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("scenario_count") != 24
    ):
        raise ValueError("MC calibration registry receipt identity drifted")
    _validate_protocol_only_receipt_fields(
        payload, expected_failure_code="MC_HALF_WIDTH_CALIBRATION_FAILED"
    )
    _validate_check_rows(payload.get("checks"))
    expected_limit = registry["gates"]["mc_half_width"][
        "pooled_parameter_coverage_min_count"
    ]
    checks = payload["checks"]
    if len(checks) != len(MC_PARAMETER_IDS):
        raise ValueError("MC calibration registry check family drifted")
    for row, parameter_id in zip(checks, MC_PARAMETER_IDS):
        if (
            row["name"] != f"pooled_coverage:{parameter_id}"
            or row["comparator"] != ">="
            or row["limit"] != expected_limit
            or type(row["value"]) is not int
            or not 0 <= row["value"] <= 4_800
            or row["passed"] is not (row["value"] >= expected_limit)
        ):
            raise ValueError("MC calibration registry check drifted")
    passed = all(row["passed"] for row in checks)
    if (payload["status"] == "PASSED") is not passed:
        raise ValueError("MC calibration registry status disagrees with checks")


def evaluate_simulation_registry(
    *,
    registry: Mapping[str, Any],
    scenario_receipts: Sequence[Mapping[str, Any]],
    mc_calibration_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_simulation_registry(registry)
    if len(scenario_receipts) != 52:
        raise ValueError("simulation gate requires exactly 52 scenario receipts")
    expected_ids = [row["scenario_id"] for row in registry["scenarios"]]
    observed_ids = []
    for receipt in scenario_receipts:
        validate_simulation_scenario_receipt(receipt, registry=registry)
        observed_ids.append(receipt.get("scenario_id"))
    if observed_ids != expected_ids:
        raise ValueError("simulation scenario receipts are not in frozen order")
    validate_mc_calibration_registry_receipt(mc_calibration_receipt, registry=registry)
    passed = all(
        receipt.get("status") == "PASSED" for receipt in scenario_receipts
    ) and (mc_calibration_receipt.get("status") == "PASSED")
    return with_content_sha256(
        {
            "schema": SIMULATION_REGISTRY_RECEIPT_SCHEMA,
            "status": "PASSED" if passed else "FAILED_CLOSED",
            "protocol_id": PROTOCOL_ID,
            "scenario_count": 52,
            "mc_calibration_scenario_count": 24,
            "claim_scope": registry["claim_scope"],
            "failure_codes": [] if passed else ["SIMULATION_GATE_FAILED"],
            "authorization_scope": "NONE",
            "phase1_v2_authorized": False,
            "holdout_open_authorized": False,
            "paper_claim_allowed": False,
            "official_final_sealed": True,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the candidate-free DUCA v2.1 simulation registry"
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--verify-reference-environment", action="store_true")
    args = parser.parse_args(argv)
    registry = load_simulation_registry(args.registry)
    environment = None
    if args.verify_reference_environment:
        environment = verify_reference_environment(registry)
    print(
        json.dumps(
            {
                "schema": "duca_admission_v2_1_simulation_registry_validation_v1",
                "status": "PASSED",
                "scenario_count": len(registry["scenarios"]),
                "artifact_sha256": registry["artifact_sha256"],
                "semantic_sha256": registry["semantic_sha256"],
                "reference_environment": environment,
                "authorization_scope": "NONE",
                "phase1_v2_authorized": False,
                "holdout_open_authorized": False,
                "paper_claim_allowed": False,
                "official_final_sealed": True,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
