"""Freeze the label-free H65 multi-budget exposure calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


BUDGETS = (256, 384, 512)
BASELINE_BUDGET = 384
TOTAL_UPDATES = 6000
TRAINING_ID_SHA256 = "5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0"
HELD_OUT_ID_SHA256 = "5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e"
MANIFEST_NONCE = "duca-h65-system-multibudget-exposure-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_manifest_sha256(video_ids: Iterable[str]) -> str:
    payload = "".join(f"{item}\n" for item in sorted(set(video_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def valid_observations(video_info: dict[str, Any], *, dense_length: int = 768) -> int:
    frame_count = int(video_info["frame"])
    if frame_count <= 0:
        raise ValueError("video frame count must be positive")
    return min(int(dense_length), int(math.ceil(frame_count / 4.0)))


def calibrated_probabilities(valid_counts: Iterable[int]) -> dict[int, float]:
    counts = [int(value) for value in valid_counts]
    if not counts:
        raise ValueError("training calibration requires at least one video")
    means = {
        budget: sum(min(budget, value) for value in counts) / len(counts)
        for budget in BUDGETS
    }
    if not means[256] < means[384] < means[512]:
        raise ValueError(f"actual observations are not strictly monotone: {means}")
    p256 = 0.5 * (means[512] - means[384]) / (means[512] - means[256])
    probabilities = {256: p256, 384: 0.5, 512: 0.5 - p256}
    if any(value < 0.0 or value > 0.5 for value in probabilities.values()):
        raise ValueError(f"calibrated probabilities are outside [0,0.5]: {probabilities}")
    return probabilities


def quantized_update_counts(probabilities: dict[int, float]) -> dict[int, int]:
    count_384 = TOTAL_UPDATES // 2
    count_256 = int(round(probabilities[256] * TOTAL_UPDATES))
    counts = {
        256: count_256,
        384: count_384,
        512: TOTAL_UPDATES - count_256 - count_384,
    }
    if sum(counts.values()) != TOTAL_UPDATES or any(value <= 0 for value in counts.values()):
        raise ValueError(f"invalid 6000-update budget multiset: {counts}")
    return counts


def budget_for_update(step: int, seed: int, counts: dict[int, int]) -> int:
    rank = (int(step) * 2593 + int(seed)) % TOTAL_UPDATES
    if rank < counts[256]:
        return 256
    if rank < counts[256] + counts[384]:
        return 384
    return 512


def training_cost_replay(
    valid_counts: list[int],
    *,
    seed: int,
    counts: dict[int, int],
) -> dict[str, int | float]:
    if len(valid_counts) != 200:
        raise ValueError("formal H65 training cost replay requires 200 videos")
    control_total = 0
    candidate_total = 0
    update = 0
    for epoch in range(60):
        generator = torch.Generator()
        generator.manual_seed(epoch)
        order = torch.randperm(len(valid_counts), generator=generator).tolist()
        for offset in range(0, len(order), 2):
            batch = [valid_counts[index] for index in order[offset : offset + 2]]
            budget = budget_for_update(update, seed, counts)
            control_total += sum(min(384, value) for value in batch)
            candidate_total += sum(min(budget, value) for value in batch)
            update += 1
    if update != TOTAL_UPDATES:
        raise ValueError(f"training replay produced {update} updates instead of 6000")
    relative_delta = (candidate_total - control_total) / control_total
    if abs(relative_delta) > 0.005:
        raise ValueError(
            "calibrated candidate training cost differs from K384 by more than 0.5%: "
            f"{relative_delta:.8f}"
        )
    return {
        "control_actual_observations": control_total,
        "candidate_actual_observations": candidate_total,
        "relative_delta": relative_delta,
    }


def held_out_windows(database: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror SlidingWindowDataset.split_video_to_windows without reading GT.

    The official test configuration uses a 768-snippet window with 0.5 overlap,
    hence a 384-snippet stride.  Its implementation intentionally emits a final
    right-aligned window even when the preceding window ends exactly at the
    video boundary; this function preserves that established evaluation input
    identity rather than substituting a different ceiling formula.
    """

    rows_by_key: dict[str, dict[str, Any]] = {}
    for video_name, video_info in database.items():
        if video_info.get("subset") != "validation":
            continue
        snippet_count = int(math.ceil(int(video_info["frame"]) / 4.0))
        last_window = False
        for index in range(max(1, snippet_count // 384)):
            start = index * 384
            end = start + 768
            if end > snippet_count:
                end = snippet_count
                start = max(0, end - 768)
                last_window = True
            key = f"{video_name}|{start * 4}"
            row = {
                "key": key,
                "video_name": video_name,
                "window_start_frame": start * 4,
                "valid_observations": end - start,
                "multiplicity": 1,
            }
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = row
            else:
                comparable = {
                    name: value for name, value in row.items() if name != "multiplicity"
                }
                prior = {
                    name: value for name, value in existing.items() if name != "multiplicity"
                }
                if comparable != prior:
                    raise ValueError("a repeated held-out window key has inconsistent geometry")
                existing["multiplicity"] = int(existing["multiplicity"]) + 1
            if last_window:
                break
    return list(rows_by_key.values())


def held_out_inference_annotation(
    database: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep only the video geometry required by label-free held-out inference."""

    inference_database = {}
    for video_name, video_info in database.items():
        if video_info.get("subset") != "validation":
            continue
        inference_database[video_name] = {
            "subset": "validation",
            "frame": int(video_info["frame"]),
            "duration": float(video_info["duration"]),
        }
    return {"database": inference_database}


def fixed_mixed_manifest(
    rows: list[dict[str, Any]], probabilities: dict[int, float]
) -> tuple[dict[str, int], dict[str, Any]]:
    if not rows:
        raise ValueError("fixed mixed manifest requires held-out windows")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{MANIFEST_NONCE}|{row['key']}".encode("utf-8")
        ).digest(),
    )
    total = len(ordered)
    count_256 = int(round(probabilities[256] * total))
    count_384 = int(round(0.5 * total))
    budget_multiset = (
        [256] * count_256
        + [384] * count_384
        + [512] * (total - count_256 - count_384)
    )
    budget_counts = {
        256: count_256,
        384: count_384,
        512: total - count_256 - count_384,
    }
    if any(value < 0 for value in budget_counts.values()) or sum(budget_counts.values()) != total:
        raise ValueError(f"invalid held-out budget multiset: {budget_counts}")
    if len(budget_multiset) != total:
        raise ValueError("held-out budget multiset must contain exactly one budget per window")
    control_cost = sum(
        int(row.get("multiplicity", 1)) * min(384, int(row["valid_observations"]))
        for row in ordered
    )
    candidates = []
    for rotation in range(total):
        rotated = [budget_multiset[(index + rotation) % total] for index in range(total)]
        cost = sum(
            int(row.get("multiplicity", 1)) * min(budget, int(row["valid_observations"]))
            for budget, row in zip(rotated, ordered)
        )
        if cost <= control_cost:
            candidates.append((control_cost - cost, rotation, cost, rotated))
    if not candidates:
        raise ValueError("no label-free metadata rotation satisfies the K384 cost ceiling")
    deficit, rotation, mixed_cost, selected = min(candidates, key=lambda item: (item[0], item[1]))
    manifest = {row["key"]: budget for row, budget in zip(ordered, selected)}
    summary = {
        "nonce": MANIFEST_NONCE,
        "assignment": "sha256_identity_ordered_budget_multiset_with_length_only_cost_rotation",
        "unique_manifest_key_count": total,
        "executed_window_count": sum(int(row.get("multiplicity", 1)) for row in ordered),
        "rotation": rotation,
        "unique_key_budget_counts": {
            str(budget): selected.count(budget) for budget in BUDGETS
        },
        "executed_window_budget_counts": {
            str(budget): sum(
                int(row.get("multiplicity", 1))
                for row, assigned in zip(ordered, selected)
                if assigned == budget
            )
            for budget in BUDGETS
        },
        "control_actual_observations": control_cost,
        "mixed_actual_observations": mixed_cost,
        "mixed_minus_control": -deficit,
    }
    return manifest, summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare(annotation: Path, output_dir: Path) -> dict[str, Any]:
    database = json.loads(annotation.read_text(encoding="utf-8"))["database"]
    training = [
        (video_name, video_info)
        for video_name, video_info in database.items()
        if video_info.get("subset") == "training"
    ]
    held_out_ids = [
        video_name
        for video_name, video_info in database.items()
        if video_info.get("subset") == "validation"
    ]
    if len(training) != 200 or len(held_out_ids) != 211:
        raise ValueError("formal data identity must be 200 training and 211 held-out videos")
    if id_manifest_sha256(name for name, _ in training) != TRAINING_ID_SHA256:
        raise ValueError("training video identity differs from the admitted 200-video manifest")
    if id_manifest_sha256(held_out_ids) != HELD_OUT_ID_SHA256:
        raise ValueError("held-out video identity differs from the admitted 211-video manifest")

    training_valid = [valid_observations(info) for _, info in training]
    means = {
        budget: sum(min(budget, value) for value in training_valid) / len(training_valid)
        for budget in BUDGETS
    }
    probabilities = calibrated_probabilities(training_valid)
    update_counts = quantized_update_counts(probabilities)
    seed_costs = {
        str(seed): training_cost_replay(training_valid, seed=seed, counts=update_counts)
        for seed in (3407, 3408, 3409)
    }
    windows = held_out_windows(database)
    manifest, manifest_summary = fixed_mixed_manifest(windows, probabilities)

    output_dir.mkdir(parents=True, exist_ok=False)
    inference_annotation_path = output_dir / "held_out_inference_annotation.json"
    _write_json(inference_annotation_path, held_out_inference_annotation(database))
    manifest_path = output_dir / "held_out_fixed_mixed_budget_manifest.json"
    _write_json(manifest_path, manifest)
    bootstrap_path = output_dir / "paired_bootstrap_indices.npy"
    rng = np.random.default_rng(3407)
    np.save(
        bootstrap_path,
        rng.integers(0, len(held_out_ids), size=(10_000, len(held_out_ids)), dtype=np.uint16),
        allow_pickle=False,
    )
    (output_dir / "held_out_video_ids.txt").write_text(
        "".join(f"{item}\n" for item in sorted(held_out_ids)), encoding="utf-8"
    )
    report = {
        "schema_version": "duca_h65_system_multibudget_exposure_pre_run_v1",
        "annotation_path": str(annotation.resolve()),
        "annotation_sha256": sha256_file(annotation),
        "training_video_count": len(training),
        "held_out_video_count": len(held_out_ids),
        "training_id_sha256": TRAINING_ID_SHA256,
        "held_out_id_sha256": HELD_OUT_ID_SHA256,
        "actual_observation_means": {str(key): value for key, value in means.items()},
        "probabilities": {str(key): value for key, value in probabilities.items()},
        "update_counts": {str(key): value for key, value in update_counts.items()},
        "seed_training_cost_replay": seed_costs,
        "held_out_manifest": {
            **manifest_summary,
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "held_out_inference_annotation": {
            "path": str(inference_annotation_path.resolve()),
            "sha256": sha256_file(inference_annotation_path),
            "video_count": len(held_out_ids),
            "fields": ["subset", "frame", "duration"],
            "contains_action_labels_or_segments": False,
        },
        "paired_bootstrap_indices": {
            "path": str(bootstrap_path.resolve()),
            "shape": [10_000, len(held_out_ids)],
            "dtype": "uint16",
            "seed": 3407,
            "sha256": sha256_file(bootstrap_path),
        },
        "held_out_fields_used_before_prediction_sealing": [
            "video_id",
            "subset",
            "frame_count",
        ],
        "held_out_action_labels_or_segments_used_before_prediction_sealing": False,
    }
    _write_json(output_dir / "pre_run_calibration.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.annotation.resolve(), args.output_dir.resolve())
    probabilities = report["probabilities"]
    print(
        "PASS H65 multi-budget calibration: "
        f"p256={probabilities['256']:.12f} p384={probabilities['384']:.12f} "
        f"p512={probabilities['512']:.12f}"
    )


if __name__ == "__main__":
    main()
