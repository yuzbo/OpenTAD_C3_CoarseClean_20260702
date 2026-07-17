from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_cellcf_training import canonical_sha256, sha256_file
from tools.bata.duca_full_stack_cost import (
    compare_profile_summaries,
    validate_and_rebuild_profile_summary,
)
from tools.bata.duca_trained_checkpoint_binding import (
    load_trained_checkpoint_binding,
)
from tools.bata.profile_duca_full_stack_cost import load_cellcf_cost_binding


SCHEMA = "duca_dense_vs_cellcf_full_stack_cost_v1"
MIN_REPEATS = 3
MIN_SAMPLES = 500
MIN_WARMUP_SAMPLES = 20
PAIR_COMPLETION_SCHEMA = "duca_dense_full_stack_cost_pair_completion_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_profile_evidence(
    profile: Mapping[str, Any], *, expected_method: str
) -> None:
    _require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            str(profile.get("evidence_git_commit") or ""),
        )
        is not None,
        f"{expected_method} profile lacks an exact evidence commit",
    )
    checkpoint_keys = (
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
    )
    if expected_method == "dense-adatad":
        binding = profile.get("trained_checkpoint_binding")
        _require(
            isinstance(binding, Mapping),
            "dense profile is missing its trained-checkpoint binding",
        )
        _require(
            profile.get("trained_checkpoint_binding_sha256")
            == canonical_sha256(binding),
            "dense profile trained-checkpoint binding hash mismatch",
        )
        _require(
            profile.get("config_path") == binding.get("config_path")
            and profile.get("profile_config_sha256")
            == binding.get("config_sha256")
            and profile.get("profile_resolved_config_sha256")
            == binding.get("resolved_config_sha256"),
            "dense profile config differs from its trained-checkpoint binding",
        )
        reopened = load_trained_checkpoint_binding(
            binding.get("path", ""),
            binding.get("sha256", ""),
            expected_role="dense_adatad_baseline",
            expected_commit=str(profile.get("trained_commit", "")),
            expected_config_path=str(profile.get("config_path", "")),
            expected_config_sha256=str(profile.get("profile_config_sha256", "")),
            expected_resolved_config_sha256=str(
                profile.get("profile_resolved_config_sha256", "")
            ),
            expected_checkpoint_path=str(profile.get("checkpoint_path", "")),
        )
    else:
        binding = profile.get("cellcf_cost_binding")
        _require(
            isinstance(binding, Mapping),
            "CellCF profile is missing its post-run checkpoint binding",
        )
        _require(
            profile.get("cellcf_cost_binding_sha256")
            == canonical_sha256(binding),
            "CellCF profile post-run binding hash mismatch",
        )
        reopened = load_cellcf_cost_binding(
            binding.get("post_run_evidence_path", ""),
            binding.get("post_run_evidence_sha256", ""),
            expected_checkpoint_path=str(profile.get("checkpoint_path", "")),
            expected_commit=str(profile.get("trained_commit", "")),
        )
        _require(
            profile.get("profile_config_sha256") == reopened.get("config_sha256")
            and profile.get("profile_resolved_config_sha256")
            == reopened.get("resolved_config_sha256"),
            "CellCF profile config differs from its post-run binding",
        )
    _require(
        canonical_sha256(reopened) == canonical_sha256(binding),
        f"{expected_method} profile binding differs after independent reopening",
    )
    for key in checkpoint_keys:
        _require(
            profile.get(key) == reopened.get(key),
            f"{expected_method} profile checkpoint binding mismatch: {key}",
        )


def _load_profiles(
    paths: Sequence[str | Path], *, expected_method: str
) -> list[dict[str, Any]]:
    _require(len(paths) >= MIN_REPEATS, f"{expected_method} requires at least three repeats")
    resolved_paths = [Path(path).expanduser().resolve() for path in paths]
    _require(
        len(set(resolved_paths)) == len(resolved_paths),
        f"{expected_method} repeat paths contain duplicates",
    )
    profiles = []
    seen_hashes: set[str] = set()
    seen_raw_sample_multiset_hashes: set[str] = set()
    for resolved in resolved_paths:
        _require(resolved.is_file(), f"cost profile is missing: {resolved}")
        profile_sha = sha256_file(resolved)
        _require(
            profile_sha not in seen_hashes,
            f"{expected_method} repeat files are byte-identical",
        )
        seen_hashes.add(profile_sha)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        _require(isinstance(payload, dict), f"cost profile is not a JSON object: {resolved}")
        _require(payload.get("method") == expected_method, f"unexpected cost method: {resolved}")
        _require(
            int(payload.get("sample_count", 0)) >= MIN_SAMPLES,
            f"{expected_method} profile requires at least {MIN_SAMPLES} measured samples",
        )
        _require(
            int(payload.get("warmup_samples", 0)) >= MIN_WARMUP_SAMPLES,
            (
                f"{expected_method} profile requires at least "
                f"{MIN_WARMUP_SAMPLES} warmup samples"
            ),
        )
        raw_sample_fingerprints = validate_and_rebuild_profile_summary(payload)
        _require(
            raw_sample_fingerprints["multiset_sha256"]
            not in seen_raw_sample_multiset_hashes,
            f"{expected_method} repeats reuse the same raw sample multiset",
        )
        seen_raw_sample_multiset_hashes.add(
            raw_sample_fingerprints["multiset_sha256"]
        )
        _require(
            isinstance(payload.get("profile_repeat_index"), int)
            and not isinstance(payload.get("profile_repeat_index"), bool)
            and int(payload["profile_repeat_index"]) > 0,
            f"{expected_method} profile repeat index is invalid",
        )
        _require(
            payload.get("profile_order_position") in (1, 2),
            f"{expected_method} profile order position is invalid",
        )
        _require(
            bool(str(payload.get("profile_session_id", "")).strip())
            and bool(str(payload.get("profile_pair_id", "")).strip()),
            f"{expected_method} profile lacks session/pair identity",
        )
        _validate_profile_evidence(payload, expected_method=expected_method)
        payload = dict(payload)
        payload["_path"] = str(resolved)
        payload["_sha256"] = profile_sha
        payload["_raw_samples_sha256"] = raw_sample_fingerprints[
            "ordered_sha256"
        ]
        payload["_raw_samples_multiset_sha256"] = (
            raw_sample_fingerprints["multiset_sha256"]
        )
        profiles.append(payload)
    return sorted(profiles, key=lambda item: int(item["profile_repeat_index"]))


def _require_same_bound_model(
    profiles: Sequence[Mapping[str, Any]], *, method: str
) -> None:
    keys = (
        "trained_commit",
        "evidence_git_commit",
        "profile_config_sha256",
        "profile_resolved_config_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
    )
    reference = profiles[0]
    for profile in profiles[1:]:
        for key in keys:
            _require(
                profile.get(key) == reference.get(key),
                f"{method} repeats use different bound models: {key}",
            )


def summarize_dense_full_stack_cost(
    dense_paths: Sequence[str | Path],
    cellcf_paths: Sequence[str | Path],
) -> dict[str, Any]:
    _require(len(dense_paths) == len(cellcf_paths), "dense and CellCF repeat counts differ")
    dense = _load_profiles(dense_paths, expected_method="dense-adatad")
    cellcf = _load_profiles(cellcf_paths, expected_method="cellcf-fixed384")
    repeat_count = len(dense)
    expected_indices = list(range(1, repeat_count + 1))
    _require(
        [int(item["profile_repeat_index"]) for item in dense] == expected_indices
        and [int(item["profile_repeat_index"]) for item in cellcf]
        == expected_indices,
        "cost repeats must be contiguous and start at one",
    )
    sessions = {
        str(item["profile_session_id"]) for item in [*dense, *cellcf]
    }
    _require(len(sessions) == 1, "cost repeats span multiple profiling sessions")
    _require_same_bound_model(dense, method="dense-adatad")
    _require_same_bound_model(cellcf, method="cellcf-fixed384")
    evidence_commits = {
        str(item["evidence_git_commit"]) for item in [*dense, *cellcf]
    }
    _require(
        len(evidence_commits) == 1,
        "dense and CellCF profiles use different evidence code commits",
    )
    comparisons = []
    order_receipt = []
    for repeat, (baseline, candidate) in enumerate(zip(dense, cellcf), start=1):
        expected_pair_id = f"repeat-{repeat}"
        _require(
            baseline.get("profile_pair_id")
            == candidate.get("profile_pair_id")
            == expected_pair_id,
            f"repeat {repeat} pair identity mismatch",
        )
        expected_dense_position = 1 if repeat % 2 == 1 else 2
        expected_cellcf_position = 2 if repeat % 2 == 1 else 1
        _require(
            baseline.get("profile_order_position") == expected_dense_position
            and candidate.get("profile_order_position")
            == expected_cellcf_position,
            f"repeat {repeat} did not follow the preregistered alternating order",
        )
        comparison = compare_profile_summaries(baseline, candidate)
        comparison["repeat"] = repeat
        comparisons.append(comparison)
        order_receipt.append(
            {
                "repeat": repeat,
                "pair_id": expected_pair_id,
                "first": (
                    "dense-adatad"
                    if expected_dense_position == 1
                    else "cellcf-fixed384"
                ),
                "second": (
                    "cellcf-fixed384"
                    if expected_cellcf_position == 2
                    else "dense-adatad"
                ),
                "dense_profile_sha256": baseline["_sha256"],
                "cellcf_profile_sha256": candidate["_sha256"],
                "dense_raw_samples_sha256": baseline[
                    "_raw_samples_sha256"
                ],
                "cellcf_raw_samples_sha256": candidate[
                    "_raw_samples_sha256"
                ],
                "dense_raw_samples_multiset_sha256": baseline[
                    "_raw_samples_multiset_sha256"
                ],
                "cellcf_raw_samples_multiset_sha256": candidate[
                    "_raw_samples_multiset_sha256"
                ],
            }
        )

    saving_fractions = [
        item["end_to_end_serial"]["latency_saving_fraction"]
        for item in comparisons
    ]
    p95_saving_fractions = [
        item["end_to_end_serial"]["p95_latency_saving_fraction"]
        for item in comparisons
    ]
    speedups = [item["end_to_end_serial"]["speedup"] for item in comparisons]
    all_repeat_gates = all(item["gates"]["all_cost_gates_pass"] for item in comparisons)
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "evidence_git_commit": next(iter(evidence_commits)),
        "claim_scope": "dense_adatad_vs_cellcf_full_stack_serial_inference",
        "dense_full_stack_baseline_included": True,
        "repeat_count": len(comparisons),
        "paired_repeat_order_required": True,
        "profile_session_id": next(iter(sessions)),
        "paired_repeat_order_receipt": order_receipt,
        "comparisons": comparisons,
        "aggregate": {
            "median_latency_saving_fraction": statistics.median(saving_fractions),
            "min_latency_saving_fraction": min(saving_fractions),
            "median_p95_latency_saving_fraction": statistics.median(
                p95_saving_fractions
            ),
            "min_p95_latency_saving_fraction": min(p95_saving_fractions),
            "median_speedup": statistics.median(speedups),
            "all_repeat_cost_gates_pass": all_repeat_gates,
        },
        "inference_cost_measurement_claim_allowed": False,
        "inference_cost_measurement_claim_blockers": [
            "formal_scheduler_receipt_not_bound",
        ],
        "paper_cost_claim_allowed": False,
        "paper_cost_claim_blockers": [
            "formal_scheduler_receipt_not_bound",
            "dense_training_and_evaluation_semantic_validation_not_integrated",
            "training_inference_break_even_unavailable",
        ],
        "training_inference_break_even": {
            "available": False,
            "reason": "dense_training_cost_not_bound_in_this_inference_artifact",
        },
        "dense_profile_paths": [
            {
                "path": item["_path"],
                "sha256": item["_sha256"],
                "raw_samples_sha256": item["_raw_samples_sha256"],
                "raw_samples_multiset_sha256": item[
                    "_raw_samples_multiset_sha256"
                ],
            }
            for item in dense
        ],
        "cellcf_profile_paths": [
            {
                "path": item["_path"],
                "sha256": item["_sha256"],
                "raw_samples_sha256": item["_raw_samples_sha256"],
                "raw_samples_multiset_sha256": item[
                    "_raw_samples_multiset_sha256"
                ],
            }
            for item in cellcf
        ],
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _exclusive_write_tsv(
    path: str | Path, comparisons: Sequence[Mapping[str, Any]]
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "repeat",
        "baseline_ms",
        "candidate_ms",
        "latency_saving_fraction",
        "speedup",
        "baseline_p95_ms",
        "candidate_p95_ms",
        "p95_latency_saving_fraction",
        "all_cost_gates_pass",
    ]
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for item in comparisons:
            end_to_end = item["end_to_end_serial"]
            writer.writerow(
                {
                    "repeat": item["repeat"],
                    "baseline_ms": end_to_end["baseline_ms"],
                    "candidate_ms": end_to_end["candidate_ms"],
                    "latency_saving_fraction": end_to_end[
                        "latency_saving_fraction"
                    ],
                    "speedup": end_to_end["speedup"],
                    "baseline_p95_ms": end_to_end["baseline_p95_ms"],
                    "candidate_p95_ms": end_to_end["candidate_p95_ms"],
                    "p95_latency_saving_fraction": end_to_end[
                        "p95_latency_saving_fraction"
                    ],
                    "all_cost_gates_pass": item["gates"][
                        "all_cost_gates_pass"
                    ],
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pair_paths(
    output_json: Path,
    output_tsv: Path,
) -> tuple[Path, Path]:
    _require(
        output_json.parent == output_tsv.parent,
        "dense cost JSON and TSV must share one output directory",
    )
    token = canonical_sha256(
        {
            "output_json": output_json.name,
            "output_tsv": output_tsv.name,
        }
    )[:16]
    return (
        output_json.parent / f".dense-cost-pair-{token}.staging",
        output_json.parent / f".dense-cost-pair-{token}.complete.json",
    )


def _recover_incomplete_pair(
    output_json: Path,
    output_tsv: Path,
    staging_root: Path,
    completion_marker: Path,
) -> None:
    if completion_marker.exists():
        raise FileExistsError("refusing to overwrite completed dense cost evidence")
    if not staging_root.exists():
        _require(
            not output_json.exists() and not output_tsv.exists(),
            "unowned partial dense cost evidence blocks recovery",
        )
        return
    manifest_path = staging_root / "transaction.json"
    if not manifest_path.is_file():
        _require(
            not output_json.exists() and not output_tsv.exists(),
            "incomplete dense cost transaction lacks a recovery manifest",
        )
        shutil.rmtree(staging_root)
        _fsync_directory(output_json.parent)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        isinstance(manifest, dict)
        and manifest.get("output_json") == str(output_json)
        and manifest.get("output_tsv") == str(output_tsv),
        "incomplete dense cost transaction targets differ",
    )
    for target, key in (
        (output_json, "output_json_sha256"),
        (output_tsv, "output_tsv_sha256"),
    ):
        if target.exists():
            _require(
                sha256_file(target) == manifest.get(key),
                "partial dense cost output differs from its recovery manifest",
            )
            target.unlink()
    shutil.rmtree(staging_root)
    _fsync_directory(output_json.parent)


def publish_output_pair(
    output_json: Path,
    output_tsv: Path,
    payload: Mapping[str, Any],
) -> Path:
    output_json = output_json.expanduser().resolve()
    output_tsv = output_tsv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    staging_root, completion_marker = _pair_paths(output_json, output_tsv)
    _recover_incomplete_pair(
        output_json,
        output_tsv,
        staging_root,
        completion_marker,
    )
    staging_root.mkdir(mode=0o700)
    staged_json = staging_root / "evidence.json"
    staged_tsv = staging_root / "evidence.tsv"
    staged_marker = staging_root / "complete.json"
    published: list[Path] = []
    try:
        _exclusive_write_json(staged_json, payload)
        _exclusive_write_tsv(staged_tsv, payload["comparisons"])
        transaction = {
            "output_json": str(output_json),
            "output_tsv": str(output_tsv),
            "output_json_sha256": sha256_file(staged_json),
            "output_tsv_sha256": sha256_file(staged_tsv),
        }
        _exclusive_write_json(staging_root / "transaction.json", transaction)
        marker = {
            "schema": PAIR_COMPLETION_SCHEMA,
            "ok": True,
            "output_json": str(output_json),
            "output_tsv": str(output_tsv),
            "output_json_sha256": transaction["output_json_sha256"],
            "output_tsv_sha256": transaction["output_tsv_sha256"],
            "evidence_artifact_sha256": payload["artifact_sha256"],
        }
        marker["artifact_sha256"] = canonical_sha256(marker)
        _exclusive_write_json(staged_marker, marker)
        _fsync_directory(staging_root)
        for source, target in (
            (staged_json, output_json),
            (staged_tsv, output_tsv),
            (staged_marker, completion_marker),
        ):
            os.link(source, target)
            published.append(target)
        _fsync_directory(output_json.parent)
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        shutil.rmtree(staging_root, ignore_errors=True)
        _fsync_directory(output_json.parent)
        raise
    shutil.rmtree(staging_root)
    _fsync_directory(output_json.parent)
    return completion_marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="append", required=True)
    parser.add_argument("--cellcf", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    args = parser.parse_args(argv)
    output_json = Path(args.output_json).expanduser().resolve()
    output_tsv = Path(args.output_tsv).expanduser().resolve()
    try:
        payload = summarize_dense_full_stack_cost(args.dense, args.cellcf)
        completion_marker = publish_output_pair(
            output_json,
            output_tsv,
            payload,
        )
        payload = dict(payload)
        payload["output_pair_completion_marker"] = str(completion_marker)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
