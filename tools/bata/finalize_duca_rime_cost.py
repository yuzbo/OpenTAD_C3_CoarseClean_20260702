from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    validate_and_rebuild_profile_summary,
)


SCHEMA = "duca_rime_paired_full_stack_cost_v2"
PAIR_KEYS = (
    "hardware_fingerprint",
    "host_fingerprint",
    "software_fingerprint",
    "protocol",
    "source_dataset_fingerprint",
    "batch_size",
    "loader_workers",
    "warmup_samples",
    "amp",
    "power_sampling_enabled",
    "power_interval_ms",
    "power_gpu_id",
    "sample_count",
    "profile_session_id",
)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_profile(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cost profile must be a JSON object: {resolved}")
    validate_and_rebuild_profile_summary(payload)
    if (
        payload.get("schema_version") != "duca-full-stack-cost-v1"
        or payload.get("protocol") != OFFLINE_FULL_WINDOW_PROTOCOL
        or payload.get("tracked_tree_clean") is not True
        or payload.get("random_init") is not False
        or payload.get("uses_ema") is not True
        or int(payload.get("loader_workers", -1)) != 0
        or int(payload.get("sample_count", 0)) < 30
        or payload.get("claims", {}).get("full_stack_latency_measured") is not True
        or payload.get("claims", {}).get("decoder_and_preprocess_included") is not True
        or payload.get("claims", {}).get("estimated_flops_used_as_latency") is not False
    ):
        raise ValueError(f"profile is not formal full-stack evidence: {resolved}")
    return resolved, payload


def _summary(profile: Mapping[str, Any], *path: str, statistic: str = "p50") -> float:
    value: Any = profile
    try:
        for key in path:
            value = value[key]
        number = float(value[statistic])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"profile is missing {'.'.join(path)}.{statistic}"
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"profile {'.'.join(path)}.{statistic} is invalid")
    return number


def _optional_summary(
    profile: Mapping[str, Any],
    *path: str,
    statistic: str = "p50",
) -> float | None:
    value: Any = profile
    try:
        for key in path:
            value = value[key]
        if value is None:
            return None
        number = float(value[statistic])
    except (KeyError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _require_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in PAIR_KEYS:
        if left.get(key) != right.get(key):
            raise ValueError(f"{label} profiles differ on {key}")
    if not str(left.get("profile_pair_id", "")).strip():
        raise ValueError(f"{label} profiles have no pair identity")
    if left.get("profile_pair_id") != right.get("profile_pair_id"):
        raise ValueError(f"{label} profiles were not measured as one pair")
    if {
        int(left.get("profile_order_position", 0)),
        int(right.get("profile_order_position", 0)),
    } != {1, 2}:
        raise ValueError(f"{label} pair must use opposite registered order positions")


def _rime_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    identity = profile.get("rime_cost_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("DUCA-RIME profile lacks its sealed cost identity")
    unsigned = dict(identity)
    embedded = unsigned.pop("identity_sha256", None)
    if (
        identity.get("schema_version") != "duca_rime_cost_identity_v1"
        or embedded != _canonical_sha256(unsigned)
        or profile.get("rime_cost_identity_sha256")
        != _canonical_sha256(identity)
        or identity.get("official_final_subset_consumed_during_training") is not False
    ):
        raise ValueError("DUCA-RIME profile cost identity is invalid")
    return dict(identity)


def finalize_cost(
    *,
    candidate_fixed_profile: str | Path,
    fixed_profile: str | Path,
    candidate_dense_profile: str | Path,
    dense_profile: str | Path,
    output: str | Path,
    expected_phase: int,
    expected_arm: str,
    expected_seed: int,
    expected_backend: str,
    expected_target_mean_cost: float,
    matched_k_tolerance: float,
) -> dict[str, Any]:
    if int(expected_phase) not in {3, 4}:
        raise ValueError("RIME cost evidence phase must be 3 or 4")
    if float(matched_k_tolerance) < 0.0:
        raise ValueError("matched-K tolerance must be non-negative")
    loaded = [
        _load_profile(path)
        for path in (
            candidate_fixed_profile,
            fixed_profile,
            candidate_dense_profile,
            dense_profile,
        )
    ]
    paths = [item[0] for item in loaded]
    candidate_fixed, fixed, candidate_dense, dense = [
        item[1] for item in loaded
    ]
    _require_pair(candidate_fixed, fixed, label="candidate/fixed")
    _require_pair(candidate_dense, dense, label="candidate/dense")
    if candidate_fixed.get("profile_session_id") != candidate_dense.get(
        "profile_session_id"
    ):
        raise ValueError("candidate/fixed and candidate/dense pairs use different sessions")

    identity_fixed_pair = _rime_identity(candidate_fixed)
    identity_dense_pair = _rime_identity(candidate_dense)
    if identity_fixed_pair != identity_dense_pair:
        raise ValueError("repeated candidate profiles do not share one RIME identity")
    identity = identity_fixed_pair
    fixed_identity = _rime_identity(fixed)
    if (
        int(identity.get("research_phase", -1)) != int(expected_phase)
        or identity.get("evaluation_arm") != str(expected_arm)
        or int(identity.get("seed", -1)) != int(expected_seed)
        or identity.get("detector_backend") != str(expected_backend)
        or fixed_identity.get("evaluation_arm") not in {
            "U-same-K",
            "U-same-K-TriDet",
        }
        or int(fixed_identity.get("research_phase", -1)) != int(expected_phase)
        or int(fixed_identity.get("seed", -1)) != int(expected_seed)
        or fixed_identity.get("detector_backend") != str(expected_backend)
    ):
        raise ValueError("RIME candidate/matched-control cost identity disagrees with the cell")
    for key in (
        "source_training_arm",
        "training_receipt_sha256",
        "checkpoint_sha256",
        "training_exposure_sha256",
        "initialization_sha256",
    ):
        if identity.get(key) != fixed_identity.get(key):
            raise ValueError(
                f"RIME candidate/U-same-K matched control differs on {key}"
            )
    target = float(expected_target_mean_cost)
    if int(expected_phase) == 4:
        if (
            float(identity.get("target_mean_cost", math.nan)) != target
            or float(fixed_identity.get("target_mean_cost", math.nan)) != target
        ):
            raise ValueError("Phase-4 cost profiles disagree with their budget panel")
    elif target != 384.0:
        raise ValueError("Phase-3 cost target is frozen to 384")

    candidate_k = _summary(candidate_fixed, "selected_count", statistic="mean")
    fixed_k = _summary(fixed, "selected_count", statistic="mean")
    matched = (
        abs(candidate_k - fixed_k) <= float(matched_k_tolerance)
        and candidate_k <= target + float(matched_k_tolerance)
        and fixed_k <= target + float(matched_k_tolerance)
    )
    candidate_p50 = _summary(
        candidate_fixed, "stages", "end_to_end_serial_ms"
    )
    candidate_p95 = _summary(
        candidate_fixed,
        "stages",
        "end_to_end_serial_ms",
        statistic="p95",
    )
    fixed_p50 = _summary(fixed, "stages", "end_to_end_serial_ms")
    dense_p50 = _summary(dense, "stages", "end_to_end_serial_ms")
    dense_p95 = _summary(
        dense, "stages", "end_to_end_serial_ms", statistic="p95"
    )
    memory = _summary(candidate_fixed, "resources", "peak_gpu_memory_mb")
    energy = _optional_summary(candidate_fixed, "energy", "gpu_energy_j")
    if int(expected_phase) == 4 and (
        candidate_fixed.get("power_sampling_enabled") is not True
        or energy is None
        or energy <= 0.0
    ):
        raise ValueError("Phase-4 cost evidence requires measured GPU energy")
    if candidate_p50 <= 0.0 or candidate_p95 <= 0.0 or memory <= 0.0:
        raise ValueError("RIME latency/memory measurements must be positive")

    payload = {
        "schema_version": SCHEMA,
        "research_phase": int(expected_phase),
        "arm": str(expected_arm),
        "seed": int(expected_seed),
        "detector_backend": str(expected_backend),
        "target_mean_cost": target,
        "rime_cost_identity": identity,
        "fixed_cost_identity": fixed_identity,
        "profile_session_id": candidate_fixed["profile_session_id"],
        "candidate_fixed_pair_id": candidate_fixed["profile_pair_id"],
        "candidate_dense_pair_id": candidate_dense["profile_pair_id"],
        "real_full_stack_measurement": True,
        "includes_probe_decoder_solver": True,
        "matched_realized_cost": matched,
        "target_budget_respected": (
            candidate_k <= target + float(matched_k_tolerance)
        ),
        "matched_k_tolerance": float(matched_k_tolerance),
        "candidate_effective_mean_k": candidate_k,
        "matched_control_arm": fixed_identity["evaluation_arm"],
        "matched_control_effective_mean_k": fixed_k,
        "latency_p50_ms": candidate_p50,
        "latency_p95_ms": candidate_p95,
        "throughput_videos_per_second": 1000.0 / candidate_p50,
        "energy_joules_per_video": energy,
        "peak_gpu_memory_mb": memory,
        "matched_control_latency_p50_ms": fixed_p50,
        "dense_latency_p50_ms": dense_p50,
        "dense_latency_p95_ms": dense_p95,
        "candidate_below_dense": candidate_p50 < dense_p50,
        "profile_artifacts": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in paths
        ],
        "official_final_labels_used_for_cost_decision": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    target_path = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target_path.exists() and target_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite different RIME cost evidence: {target_path}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    return {"path": str(target_path), "sha256": _sha256_file(target_path), "payload": payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal paired, measured DUCA-RIME full-stack cost evidence."
    )
    parser.add_argument("--candidate-fixed-profile", required=True)
    parser.add_argument("--fixed-profile", required=True)
    parser.add_argument("--candidate-dense-profile", required=True)
    parser.add_argument("--dense-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-phase", type=int, choices=(3, 4), required=True)
    parser.add_argument("--expected-arm", required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-backend", choices=("ActionFormer", "TriDet"), required=True)
    parser.add_argument("--expected-target-mean-cost", type=float, required=True)
    parser.add_argument("--matched-k-tolerance", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = finalize_cost(
        candidate_fixed_profile=args.candidate_fixed_profile,
        fixed_profile=args.fixed_profile,
        candidate_dense_profile=args.candidate_dense_profile,
        dense_profile=args.dense_profile,
        output=args.output,
        expected_phase=args.expected_phase,
        expected_arm=args.expected_arm,
        expected_seed=args.expected_seed,
        expected_backend=args.expected_backend,
        expected_target_mean_cost=args.expected_target_mean_cost,
        matched_k_tolerance=args.matched_k_tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
