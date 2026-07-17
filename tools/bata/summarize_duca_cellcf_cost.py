from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata.duca_full_stack_cost import (
    validate_and_rebuild_profile_summary,
)
from tools.bata.profile_duca_full_stack_cost import (
    _canonical_sha256,
    _sha256_file,
    load_cellcf_cost_binding,
)


SCHEMA = "duca_cellcf_cost_pair_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"cost profile is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"cost profile is not an object: {resolved}")
    raw_sample_fingerprints = validate_and_rebuild_profile_summary(payload)
    payload["_path"] = str(resolved)
    payload["_sha256"] = _sha256_file(resolved)
    payload["_raw_samples_sha256"] = raw_sample_fingerprints[
        "ordered_sha256"
    ]
    payload["_raw_samples_multiset_sha256"] = raw_sample_fingerprints[
        "multiset_sha256"
    ]
    return payload


def _stage(report: Mapping[str, Any], name: str, statistic: str = "p50") -> float:
    try:
        return float(report["stages"][name][statistic])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"profile is missing stages.{name}.{statistic}") from exc


def _validate_profile_binding(
    report: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    method: str,
) -> None:
    path = report.get("_path")
    _require(report.get("cellcf_cost_binding") == binding, f"{path} differs from the frozen CellCF cost binding")
    _require(
        report.get("cellcf_cost_binding_sha256") == _canonical_sha256(binding),
        f"{path} has an invalid CellCF cost-binding hash",
    )
    expected = {
        "config_commit": binding["git_commit"],
        "checkpoint_path": binding["checkpoint_path"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_epoch": binding["checkpoint_epoch"],
        "checkpoint_state_key": binding["checkpoint_state_key"],
        "weight_source": "cellcf_trained_terminal_state_dict_ema",
        "dense_full_stack_savings_claimed": False,
    }
    for key, value in expected.items():
        _require(report.get(key) == value, f"{path} differs from the CellCF cost binding on {key}")
    _require(report.get("uses_ema") is True, f"{path} did not load state_dict_ema")
    _require(report.get("random_init") is False, f"{path} used random initialization")

    profile_config_sha = str(report.get("profile_config_sha256") or "")
    profile_resolved_sha = str(report.get("profile_resolved_config_sha256") or "")
    _require(re.fullmatch(r"[0-9a-f]{64}", profile_config_sha) is not None, f"{path} has no profile config SHA256")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", profile_resolved_sha) is not None,
        f"{path} has no resolved profile config SHA256",
    )
    if method == "cellcf-fixed384":
        _require(report.get("frontend_variant") == "cellcf", f"{path} is not the CellCF frontend")
        _require(profile_config_sha == binding["config_sha256"], f"{path} source config differs from CellCF training")
        _require(
            profile_resolved_sha == binding["resolved_config_sha256"],
            f"{path} resolved config differs from CellCF training",
        )
        _require(report.get("checkpoint_dropped_prefixes") == [], f"{path} dropped trained CellCF weights")
        _require(int(report.get("checkpoint_dropped_key_count", -1)) == 0, f"{path} dropped checkpoint keys")
    else:
        _require(
            report.get("frontend_variant") == "bare_exact_uniform_lower_bound",
            f"{path} is not the bare exact-uniform frontend lower-bound",
        )
        _require(
            report.get("checkpoint_dropped_prefixes") == ["frame_selector."],
            f"{path} did not remove exactly the CellCF frontend weights",
        )
        _require(
            int(report.get("checkpoint_dropped_key_count", 0)) > 0,
            f"{path} did not remove CellCF frontend checkpoint keys",
        )


def _validate_group(
    reports: list[dict[str, Any]],
    method: str,
    binding: Mapping[str, Any],
) -> None:
    _require(len(reports) >= 3, f"{method} requires at least three fresh-process repeats")
    _require(
        len(
            {
                report["_raw_samples_multiset_sha256"]
                for report in reports
            }
        )
        == len(reports),
        f"{method} repeats reuse the same raw sample multiset",
    )
    reference = reports[0]
    for report in reports:
        _require(report.get("method") == method, f"unexpected method in {report['_path']}")
        _require(
            isinstance(report.get("profile_repeat_index"), int)
            and not isinstance(report.get("profile_repeat_index"), bool)
            and int(report["profile_repeat_index"]) > 0,
            f"{method} profile repeat index is invalid",
        )
        _require(
            report.get("profile_order_position") in (1, 2),
            f"{method} profile order position is invalid",
        )
        _require(
            bool(str(report.get("profile_session_id", "")).strip())
            and bool(str(report.get("profile_pair_id", "")).strip()),
            f"{method} profile lacks session/pair identity",
        )
        _require(int(report.get("sample_count", 0)) >= 500, f"{method} requires at least 500 real windows per repeat")
        _require(report.get("tracked_tree_clean") is True, f"{method} profile used a dirty tree")
        _require(report.get("random_init") is False and report.get("uses_ema") is True, f"{method} must use terminal EMA weights")
        _validate_profile_binding(report, binding, method=method)
        for key in (
            "protocol",
            "hardware_fingerprint",
            "host_fingerprint",
            "software_fingerprint",
            "config_commit",
            "source_dataset_fingerprint",
            "inference_fingerprint",
            "detector_stack_fingerprint",
            "batch_size",
            "loader_workers",
            "amp",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_epoch",
            "checkpoint_state_key",
            "cellcf_cost_binding_sha256",
        ):
            _require(report.get(key) == reference.get(key), f"{method} repeat drifted on {key}")


def summarize(
    cellcf_paths: list[str],
    bare_paths: list[str],
    *,
    post_run_evidence_path: str | Path,
    post_run_evidence_sha256: str,
) -> dict[str, Any]:
    binding = load_cellcf_cost_binding(
        post_run_evidence_path,
        post_run_evidence_sha256,
    )
    cellcf = [_load(path) for path in cellcf_paths]
    bare = [_load(path) for path in bare_paths]
    _validate_group(cellcf, "cellcf-fixed384", binding)
    _validate_group(bare, "bare-uniform384", binding)
    _require(
        len(cellcf) == len(bare),
        "CellCF and bare-uniform repeat counts differ",
    )
    cellcf = sorted(
        cellcf, key=lambda report: int(report["profile_repeat_index"])
    )
    bare = sorted(
        bare, key=lambda report: int(report["profile_repeat_index"])
    )
    expected_indices = list(range(1, len(cellcf) + 1))
    _require(
        [int(report.get("profile_repeat_index", -1)) for report in cellcf]
        == expected_indices
        and [
            int(report.get("profile_repeat_index", -1))
            for report in bare
        ]
        == expected_indices,
        "cost repeats must be contiguous and start at one",
    )
    sessions = {
        str(report.get("profile_session_id", ""))
        for report in [*cellcf, *bare]
    }
    _require(
        len(sessions) == 1 and bool(next(iter(sessions)).strip()),
        "cost repeats span multiple or empty profiling sessions",
    )
    order_receipt = []
    for repeat, (cellcf_report, bare_report) in enumerate(
        zip(cellcf, bare), start=1
    ):
        expected_pair = f"repeat-{repeat}"
        expected_cellcf_position = 1 if repeat % 2 == 1 else 2
        expected_bare_position = 2 if repeat % 2 == 1 else 1
        _require(
            cellcf_report.get("profile_pair_id")
            == bare_report.get("profile_pair_id")
            == expected_pair,
            f"repeat {repeat} pair identity mismatch",
        )
        _require(
            cellcf_report.get("profile_order_position")
            == expected_cellcf_position
            and bare_report.get("profile_order_position")
            == expected_bare_position,
            f"repeat {repeat} did not follow the alternating order",
        )
        order_receipt.append(
            {
                "repeat": repeat,
                "pair_id": expected_pair,
                "first": (
                    "cellcf-fixed384"
                    if expected_cellcf_position == 1
                    else "bare-uniform384"
                ),
                "second": (
                    "bare-uniform384"
                    if expected_bare_position == 2
                    else "cellcf-fixed384"
                ),
                "cellcf_profile_sha256": cellcf_report["_sha256"],
                "bare_profile_sha256": bare_report["_sha256"],
                "cellcf_raw_samples_sha256": cellcf_report[
                    "_raw_samples_sha256"
                ],
                "bare_raw_samples_sha256": bare_report[
                    "_raw_samples_sha256"
                ],
                "cellcf_raw_samples_multiset_sha256": cellcf_report[
                    "_raw_samples_multiset_sha256"
                ],
                "bare_raw_samples_multiset_sha256": bare_report[
                    "_raw_samples_multiset_sha256"
                ],
            }
        )
    for key in (
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "config_commit",
        "source_dataset_fingerprint",
        "inference_fingerprint",
        "detector_stack_fingerprint",
        "batch_size",
        "loader_workers",
        "amp",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
        "cellcf_cost_binding_sha256",
    ):
        _require(cellcf[0].get(key) == bare[0].get(key), f"paired profiles differ on {key}")
    _require(all(_stage(report, "frame_selector_total_ms") > 0.0 for report in cellcf), "CellCF selector was not measured")
    _require(all(_stage(report, "coarse_probe_ms") > 0.0 for report in cellcf), "CellCF coarse probe was not measured")
    _require(all(_stage(report, "frame_selector_total_ms") == 0.0 for report in bare), "bare uniform unexpectedly built a selector")
    _require(all(_stage(report, "coarse_probe_ms") == 0.0 for report in bare), "bare uniform unexpectedly built a probe")

    def run_medians(reports: list[dict[str, Any]], stage: str) -> list[float]:
        return [_stage(report, stage) for report in reports]

    cellcf_e2e = run_medians(cellcf, "end_to_end_serial_ms")
    bare_e2e = run_medians(bare, "end_to_end_serial_ms")
    cellcf_median = statistics.median(cellcf_e2e)
    bare_median = statistics.median(bare_e2e)
    _require(bare_median > 0.0, "bare exact-uniform end-to-end latency must be positive")
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "status": "complete",
        "pass": True,
        "task": "offline_temporal_action_detection",
        "git_commit": binding["git_commit"],
        "config_commit": binding["git_commit"],
        "seed": binding["seed"],
        "variant": binding["variant"],
        "training_profile": binding["training_profile"],
        "training_protocol": binding["training_protocol"],
        "config_sha256": binding["config_sha256"],
        "resolved_config_sha256": binding["resolved_config_sha256"],
        "runtime_config_sha256": binding["runtime_config_sha256"],
        "evaluation_runtime_config_sha256": binding["evaluation_runtime_config_sha256"],
        "hardware_fingerprint": cellcf[0]["hardware_fingerprint"],
        "profile_session_id": next(iter(sessions)),
        "paired_repeat_order_required": True,
        "paired_repeat_order_receipt": order_receipt,
        "checkpoint_path": binding["checkpoint_path"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_epoch": binding["checkpoint_epoch"],
        "checkpoint_state_key": binding["checkpoint_state_key"],
        "cellcf_cost_binding": binding,
        "cellcf_cost_binding_sha256": _canonical_sha256(binding),
        "repeats_per_method": min(len(cellcf), len(bare)),
        "samples_per_repeat": min(
            min(int(report["sample_count"]) for report in cellcf),
            min(int(report["sample_count"]) for report in bare),
        ),
        "cellcf": {
            "end_to_end_p50_ms_by_repeat": cellcf_e2e,
            "end_to_end_p50_ms_median": cellcf_median,
            "selector_p50_ms_median": statistics.median(run_medians(cellcf, "frame_selector_total_ms")),
            "coarse_probe_p50_ms_median": statistics.median(run_medians(cellcf, "coarse_probe_ms")),
        },
        "bare_uniform": {
            "end_to_end_p50_ms_by_repeat": bare_e2e,
            "end_to_end_p50_ms_median": bare_median,
        },
        "frontend_overhead": {
            "end_to_end_ms": cellcf_median - bare_median,
            "fraction_of_bare_uniform": (cellcf_median - bare_median) / bare_median,
            "cellcf_to_bare_ratio": cellcf_median / bare_median,
        },
        "comparison_contract": {
            "weights": "same_cellcf_trained_terminal_state_dict_ema",
            "candidate_frontend": "cellcf",
            "reference_frontend": "bare_exact_uniform_lower_bound",
            "downstream_detector_weights_matched": True,
            "dense_full_stack_baseline_included": False,
            "dense_full_stack_savings_claimed": False,
        },
        "claim_scope": "cellcf_frontend_vs_bare_exact_uniform_lower_bound_only",
        "dense_full_stack_savings_claimed": False,
        "dense_baseline_still_required": True,
        "cellcf_profile_paths": [report["_path"] for report in cellcf],
        "bare_uniform_profile_paths": [report["_path"] for report in bare],
        "profile_artifacts": {
            "cellcf": [
                {
                    "path": report["_path"],
                    "sha256": report["_sha256"],
                    "raw_samples_sha256": report[
                        "_raw_samples_sha256"
                    ],
                    "raw_samples_multiset_sha256": report[
                        "_raw_samples_multiset_sha256"
                    ],
                }
                for report in cellcf
            ],
            "bare_uniform": [
                {
                    "path": report["_path"],
                    "sha256": report["_sha256"],
                    "raw_samples_sha256": report[
                        "_raw_samples_sha256"
                    ],
                    "raw_samples_multiset_sha256": report[
                        "_raw_samples_multiset_sha256"
                    ],
                }
                for report in bare
            ],
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellcf", action="append", required=True)
    parser.add_argument("--bare-uniform", action="append", required=True)
    parser.add_argument("--post-run-evidence", required=True)
    parser.add_argument("--post-run-evidence-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "status": "incomplete",
            "pass": False,
            "error_type": "FileExistsError",
            "error": "refusing to overwrite CellCF frontend cost evidence",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    try:
        payload = summarize(
            args.cellcf,
            args.bare_uniform,
            post_run_evidence_path=args.post_run_evidence,
            post_run_evidence_sha256=args.post_run_evidence_sha256,
        )
        code = 0
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "ok": False,
            "status": "incomplete",
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(output + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
