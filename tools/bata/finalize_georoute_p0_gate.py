"""Seal the three complementary CUDA one-step checks for GeoRoute-AdaTAD.

The three reports intentionally answer different mechanical questions across
three isolated Slurm sub-gates: all-native-token execution, the representative
hard hybrid path, and the stochastic score-function path.  This tool makes no
accuracy or paper claim and never reads a dataset or an evaluator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.bata.run_georoute_p0_gate import validate_p0_gate_report


SCHEMA_VERSION = "georoute_adatad_p0_suite_v1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_p0_gate_report(payload)
    return payload


def finalize(*, dense: Path, hybrid: Path, score_function: Path) -> dict[str, Any]:
    reports = {
        "dense_native_parity": _load_report(dense),
        "hybrid_straight_through": _load_report(hybrid),
        "roi_score_function": _load_report(score_function),
    }
    dense_report = reports["dense_native_parity"]
    hybrid_report = reports["hybrid_straight_through"]
    score_report = reports["roi_score_function"]
    if dense_report["estimator"]["name"] != "none":
        raise ValueError("dense parity report must use no policy gradient")
    if dense_report["exact_k"]["target_k"] != dense_report["source_grid"]["patch_capacity"]:
        raise ValueError("dense parity report did not select every native source token")
    dense_reference = dense_report.get("dense_native_reference")
    if (
        not isinstance(dense_reference, Mapping)
        or dense_reference.get("passed") is not True
        or int(dense_reference.get("reference_heavy_backbone_forward_count", -1)) != 1
        or int(dense_reference.get("real_route_heavy_backbone_forward_count", -1)) != 1
    ):
        raise ValueError("dense parity report lacks a passed one-reference/one-runtime numerical comparison")
    if (hybrid_report["estimator"]["name"], hybrid_report["estimator"]["claim"]) != (
        "straight_through",
        "biased_straight_through",
    ):
        raise ValueError("hybrid report must label straight-through as a biased surrogate")
    if not {"scout_geometry", "scout_residual"} <= set(hybrid_report["gradient"]["nonzero_components"]):
        raise ValueError("hybrid detector loss did not reach both scout routing heads")
    if (score_report["estimator"]["name"], score_report["estimator"]["claim"]) != (
        "score_function",
        "score_function_candidate",
    ):
        raise ValueError("score-function report has the wrong estimator identity")
    if "scout_geometry" not in score_report["gradient"]["nonzero_components"]:
        raise ValueError("score-function detector loss did not reach the geometry scout head")
    score_binding = score_report.get("score_function_detector_binding")
    if not isinstance(score_binding, Mapping) or not {"cls_loss", "reg_loss"} <= set(
        score_binding.get("detector_loss_keys", [])
    ):
        raise ValueError("score-function report is not bound to real classification/regression detector losses")
    core = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_MECHANICAL_ONLY",
        "reports": {
            name: {
                "path": str(path.resolve()),
                "file_sha256": _sha256_file(path),
                "report_sha256": reports[name]["report_sha256"],
            }
            for name, path in {
                "dense_native_parity": dense,
                "hybrid_straight_through": hybrid,
                "roi_score_function": score_function,
            }.items()
        },
        "verified_properties": {
            "one_heavy_videomae_forward_per_subgate": True,
            "dense_native_selects_all_source_tokens": True,
            "dense_native_packed_numerical_reference_passed": True,
            "representative_hybrid_detector_gradient_reaches_scout": True,
            "score_function_detector_gradient_reaches_scout": True,
            "score_function_known_answer_required_before_paper_claim": True,
            "official_test_opened": False,
            "full_training_completed": False,
            "accuracy_claim_allowed": False,
        },
    }
    return {**core, "suite_sha256": _canonical_sha256(core)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--score-function", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite a sealed GeoRoute P0 suite")
        summary = finalize(
            dense=args.dense,
            hybrid=args.hybrid,
            score_function=args.score_function,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
