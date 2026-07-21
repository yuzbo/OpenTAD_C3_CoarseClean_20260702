from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from tools.bata.analyze_duca_selection_quality import exact_uniform_positions


SCHEMA = "duca_frontend_checkpoint_selection_v1"
MANIFEST_SCHEMA = "duca_frontend_candidate_manifest_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _read_bound(path_value: str, sha256_value: str, label: str) -> tuple[Path, Any]:
    path = Path(path_value).expanduser().resolve()
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(sha256_file(path) == str(sha256_value).lower(), f"{label} SHA256 mismatch")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _short_action_both_endpoint_r1(records_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    durations = [
        float(end) - float(start)
        for row in rows
        for start, end in row.get("gt_segments", [])
        if float(end) >= float(start)
    ]
    _require(bool(durations), "candidate records contain no GT action instances")
    duration_threshold = float(median(durations))
    learned_hits = 0
    uniform_hits = 0
    total = 0
    for row in rows:
        valid_len = int(row["valid_len"])
        learned = {
            int(value)
            for value in row.get("selected_positions", [])
            if 0 <= int(value) < valid_len
        }
        uniform = set(exact_uniform_positions(valid_len, int(row["budget"])))
        for start, end in row.get("gt_segments", []):
            start = max(0.0, min(float(valid_len - 1), float(start)))
            end = max(0.0, min(float(valid_len - 1), float(end)))
            if end - start > duration_threshold:
                continue
            total += 1
            learned_hits += int(
                any(abs(position - start) <= 1.0 for position in learned)
                and any(abs(position - end) <= 1.0 for position in learned)
            )
            uniform_hits += int(
                any(abs(position - start) <= 1.0 for position in uniform)
                and any(abs(position - end) <= 1.0 for position in uniform)
            )
    _require(total > 0, "candidate records contain no short action instances")
    return {
        "definition": "duration_at_or_below_train_holdout_instance_median",
        "duration_threshold_dense_candidates": duration_threshold,
        "instance_count": total,
        "learned_both_endpoint_r1": learned_hits / float(total),
        "uniform_both_endpoint_r1": uniform_hits / float(total),
    }


def _candidate_metrics(candidate: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(candidate["checkpoint_path"])).expanduser().resolve()
    _require(checkpoint.is_file(), f"candidate checkpoint is missing: {checkpoint}")
    _require(
        sha256_file(checkpoint) == str(candidate["checkpoint_sha256"]).lower(),
        "candidate checkpoint SHA256 mismatch",
    )
    summary_path, summary = _read_bound(
        str(candidate["summary_path"]),
        str(candidate["summary_sha256"]),
        "candidate selection-quality summary",
    )
    records_path = Path(str(candidate["records_path"])).expanduser().resolve()
    _require(records_path.is_file(), f"candidate records are missing: {records_path}")
    _require(
        sha256_file(records_path) == str(candidate["records_sha256"]).lower(),
        "candidate records SHA256 mismatch",
    )
    _require(
        summary.get("schema_version") == "duca_selection_quality_summary_v2",
        "unexpected selection-quality summary schema",
    )
    _require(int(summary.get("sample_count", 0)) > 0, "empty holdout summary")
    coarse_auroc = _finite(summary["coarse"]["pooled"]["auroc"], "coarse AUROC")
    policy_auroc = _finite(summary["transition"]["r1"]["policy"]["auroc"], "policy transition AUROC")
    delta_auroc = _finite(
        summary["transition"]["r1"]["pure_abs_delta_p_action"]["auroc"],
        "pure-delta transition AUROC",
    )
    learned_r1 = _finite(
        summary["selection"]["learned"]["boundary_recall"]["r1"]["mean"],
        "learned boundary recall r1",
    )
    uniform_r1 = _finite(
        summary["selection"]["uniform"]["boundary_recall"]["r1"]["mean"],
        "uniform boundary recall r1",
    )
    learned_distance = _finite(
        summary["selection"]["learned"]["mean_endpoint_distance"]["mean"],
        "learned endpoint distance",
    )
    uniform_distance = _finite(
        summary["selection"]["uniform"]["mean_endpoint_distance"]["mean"],
        "uniform endpoint distance",
    )
    learned_max_hole = _finite(
        summary["selection"]["learned"]["max_unselected_hole"]["mean"],
        "learned max hole",
    )
    short = _short_action_both_endpoint_r1(records_path)
    gates = {
        "coarse_action_auroc_at_least_0_55": coarse_auroc >= 0.55,
        "policy_transition_auroc_not_below_pure_delta": policy_auroc >= delta_auroc,
        "learned_boundary_recall_r1_not_below_uniform": learned_r1 >= uniform_r1,
        "learned_endpoint_distance_not_above_uniform": learned_distance <= uniform_distance,
        "short_action_both_endpoint_r1_not_below_uniform": (
            short["learned_both_endpoint_r1"] >= short["uniform_both_endpoint_r1"]
        ),
        "mean_max_unselected_hole_at_most_2": learned_max_hole <= 2.0,
    }
    epoch_one_based = int(candidate["epoch_one_based"])
    _require(epoch_one_based in {5, 10, 15, 20}, "candidate epoch is outside the preregistered set")
    return {
        "variant": str(candidate["variant"]),
        "epoch_one_based": epoch_one_based,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "loss_weights": dict(candidate["loss_weights"]),
        "metrics": {
            "coarse_action_auroc": coarse_auroc,
            "policy_transition_auroc_r1": policy_auroc,
            "pure_delta_transition_auroc_r1": delta_auroc,
            "learned_boundary_recall_r1": learned_r1,
            "uniform_boundary_recall_r1": uniform_r1,
            "learned_minus_uniform_boundary_recall_r1": learned_r1 - uniform_r1,
            "learned_mean_endpoint_distance": learned_distance,
            "uniform_mean_endpoint_distance": uniform_distance,
            "uniform_minus_learned_endpoint_distance": uniform_distance - learned_distance,
            "learned_mean_max_unselected_hole": learned_max_hole,
            "short_action": short,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def select_checkpoint(
    candidate_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(candidate_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    _require(manifest_path.is_file(), f"candidate manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "candidate manifest schema mismatch")
    _require(manifest.get("source_subset") == "training", "P0 must use the training subset only")
    _require(manifest.get("test_subset_consumed") is False, "P0 cannot consume the test subset")
    split_path, split = _read_bound(
        str(manifest["split_manifest_path"]),
        str(manifest["split_manifest_sha256"]),
        "frontend train/holdout split manifest",
    )
    _require(split.get("test_subset_consumed") is False, "split manifest consumed the test subset")
    candidates = [_candidate_metrics(item) for item in manifest.get("candidates", [])]
    _require(len(candidates) == 12, "P0 requires exactly 3 weights x 4 checkpoints")
    identities = {(item["variant"], item["epoch_one_based"]) for item in candidates}
    _require(len(identities) == 12, "P0 candidate identities must be unique")
    eligible = [item for item in candidates if item["all_gates_pass"]]
    eligible.sort(
        key=lambda item: (
            -item["metrics"]["learned_minus_uniform_boundary_recall_r1"],
            -(
                item["metrics"]["short_action"]["learned_both_endpoint_r1"]
                - item["metrics"]["short_action"]["uniform_both_endpoint_r1"]
            ),
            -item["metrics"]["uniform_minus_learned_endpoint_distance"],
            -item["metrics"]["policy_transition_auroc_r1"],
            -item["metrics"]["coarse_action_auroc"],
            item["epoch_one_based"],
            item["variant"],
        )
    )
    winner = eligible[0] if eligible else None
    payload = {
        "schema": SCHEMA,
        "ok": winner is not None,
        "status": "GO_TO_MATCHED_OFFICIAL60" if winner is not None else "HOLD_FRONTEND_MECHANISM_FAILED",
        "decision_metric_scope": "train_subset_internal_holdout_only",
        "test_subset_consumed": False,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "selection_rule": [
            "all six mechanism gates must pass",
            "maximize learned-minus-uniform boundary recall at radius 1",
            "then maximize short-action both-endpoint gain at radius 1",
            "then maximize endpoint-distance gain, policy AUROC, and action AUROC",
            "then prefer the earlier checkpoint",
        ],
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "winner": winner,
        "candidates": candidates,
        "paper_metric_claim_allowed": False,
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P0 decision: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a DUCA frontend checkpoint using only a train-side holdout."
    )
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = select_checkpoint(args.candidate_manifest, args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
