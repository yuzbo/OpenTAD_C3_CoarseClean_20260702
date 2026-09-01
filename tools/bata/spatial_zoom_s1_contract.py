from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

S1_RESOLUTIONS = (160, 224, 256)
S1_TRAINING_SEEDS = (3407, 3408, 3409)
S1_SPLIT_SEED = 3407
S1_GATE_RATIO = 0.2
S1_BOOTSTRAP_SEED = 3407001
S1_PROFILE_ORDER_SEED = 3407002
S1_BOOTSTRAP_REPLICATES = 10_000
S1_CHECKPOINT_RULE = "max_gate_high_tiou_headroom_earliest_epoch_tie"
S1_MANIFEST_SCHEMA = "spatial_zoom_s1_manifest_v3"
S1_PRETRAINED_CHECKPOINT_FILENAME = (
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
)
S1_PRETRAINED_CHECKPOINT_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_id_hash(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(map(str, values))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_s1_profile_order() -> list[dict[str, int]]:
    cells = [
        (resolution, seed)
        for resolution in S1_RESOLUTIONS
        for seed in S1_TRAINING_SEEDS
    ]
    rng = np.random.default_rng(S1_PROFILE_ORDER_SEED)
    permutation = rng.permutation(len(cells)).tolist()
    return [
        {
            "ordinal": ordinal,
            "resolution": int(cells[index][0]),
            "seed": int(cells[index][1]),
        }
        for ordinal, index in enumerate(permutation)
    ]


def _valid_annotations(video: Mapping[str, Any]) -> list[dict[str, Any]]:
    valid = []
    for annotation in video.get("annotations", []):
        label = str(annotation.get("label", ""))
        segment = annotation.get("segment", ())
        if label == "Ambiguous" or len(segment) != 2:
            continue
        start, end = float(segment[0]), float(segment[1])
        if math.isfinite(start) and math.isfinite(end) and end > start:
            valid.append({"label": label, "segment": (start, end)})
    return valid


def _duration_bin(value: float, quartiles: tuple[float, float, float]) -> int:
    q1, q2, q3 = quartiles
    if value <= q1:
        return 0
    if value <= q2:
        return 1
    if value <= q3:
        return 2
    return 3


def _stable_order_key(seed: int, video_id: str) -> str:
    return hashlib.sha256(f"{int(seed)}\0{video_id}".encode("utf-8")).hexdigest()


def _stratified_gate_split(
    videos: Mapping[str, Mapping[str, Any]], *, gate_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    ids = sorted(videos)
    if len(ids) < 4:
        raise ValueError("S1 fit/gate split requires at least four development videos")
    if not 0.0 < float(gate_ratio) < 0.5:
        raise ValueError("gate_ratio must lie in (0, 0.5)")
    target = max(1, min(len(ids) - 1, int(round(len(ids) * float(gate_ratio)))))

    all_durations = [
        end - start
        for video in videos.values()
        for start, end in (item["segment"] for item in _valid_annotations(video))
    ]
    if not all_durations:
        raise ValueError("development videos contain no valid action annotations")
    quartiles = tuple(
        float(value) for value in np.quantile(all_durations, [0.25, 0.5, 0.75])
    )
    labels_by_video = {
        video_id: {item["label"] for item in _valid_annotations(video)}
        for video_id, video in videos.items()
    }
    duration_groups_by_video = {
        video_id: {
            _duration_bin(end - start, quartiles)
            for start, end in (item["segment"] for item in _valid_annotations(video))
        }
        for video_id, video in videos.items()
    }
    label_support = Counter(
        label for labels in labels_by_video.values() for label in labels
    )
    duration_support = Counter(
        group for groups in duration_groups_by_video.values() for group in groups
    )
    desired_labels = {
        label: max(1, min(count - 1, int(round(count * float(gate_ratio)))))
        for label, count in label_support.items()
        if count >= 2
    }
    desired_duration = {
        group: max(1, min(count - 1, int(round(count * float(gate_ratio)))))
        for group, count in duration_support.items()
        if count >= 2
    }

    selected: list[str] = []
    selected_labels: Counter[str] = Counter()
    selected_duration: Counter[int] = Counter()
    remaining = set(ids)
    while len(selected) < target:
        candidates = []
        for video_id in remaining:
            labels = labels_by_video[video_id]
            if any(
                label_support[label] - selected_labels[label] <= 1 for label in labels
            ):
                continue
            groups = duration_groups_by_video[video_id]
            if any(
                duration_support[group] - selected_duration[group] <= 1
                for group in groups
            ):
                continue
            label_deficit = sum(
                max(0, desired_labels.get(label, 0) - selected_labels[label])
                for label in labels
            )
            duration_deficit = sum(
                max(0, desired_duration.get(group, 0) - selected_duration[group])
                for group in groups
            )
            rarity = sum(1.0 / label_support[label] for label in labels)
            candidates.append(
                (
                    label_deficit + duration_deficit,
                    rarity,
                    _stable_order_key(seed, video_id),
                    video_id,
                )
            )
        if not candidates:
            candidates = [
                (0.0, 0.0, _stable_order_key(seed, video_id), video_id)
                for video_id in remaining
            ]
        _, _, _, chosen = max(candidates)
        selected.append(chosen)
        remaining.remove(chosen)
        selected_labels.update(labels_by_video[chosen])
        selected_duration.update(duration_groups_by_video[chosen])

    fit = sorted(remaining)
    gate = sorted(selected)
    for label, count in label_support.items():
        if count < 2:
            continue
        if not any(label in labels_by_video[video_id] for video_id in fit):
            raise ValueError(f"fit split lost class support for {label}")
        if not any(label in labels_by_video[video_id] for video_id in gate):
            raise ValueError(
                f"gate split lost class support for {label}; adjust gate_ratio or split seed"
            )
    for group, count in duration_support.items():
        if count < 2:
            continue
        if not any(group in duration_groups_by_video[video_id] for video_id in fit):
            raise ValueError(f"fit split lost action-duration group {group}")
        if not any(group in duration_groups_by_video[video_id] for video_id in gate):
            raise ValueError(f"gate split lost action-duration group {group}")
    return fit, gate


def _class_support(
    database: Mapping[str, Mapping[str, Any]], video_ids: Iterable[str]
) -> dict[str, int]:
    support: Counter[str] = Counter()
    for video_id in video_ids:
        labels = {item["label"] for item in _valid_annotations(database[str(video_id)])}
        support.update(labels)
    return dict(sorted(support.items()))


def _duration_support(
    database: Mapping[str, Mapping[str, Any]],
    video_ids: Iterable[str],
    quartiles: tuple[float, float, float],
) -> dict[str, int]:
    support: Counter[int] = Counter()
    for video_id in video_ids:
        groups = {
            _duration_bin(end - start, quartiles)
            for start, end in (
                item["segment"] for item in _valid_annotations(database[str(video_id)])
            )
        }
        support.update(groups)
    return {str(key): value for key, value in sorted(support.items())}


def build_s1_manifest(
    annotation_path: str | Path,
    *,
    gate_ratio: float = S1_GATE_RATIO,
    split_seed: int = S1_SPLIT_SEED,
) -> dict[str, Any]:
    if float(gate_ratio) != S1_GATE_RATIO or int(split_seed) != S1_SPLIT_SEED:
        raise ValueError("formal S1 manifest gate ratio and split seed are frozen")
    annotation_path = Path(annotation_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("THUMOS annotation must contain a database mapping")
    development = {
        str(video_id): video
        for video_id, video in database.items()
        if str(video.get("subset")) == "training" and _valid_annotations(video)
    }
    test = sorted(
        str(video_id)
        for video_id, video in database.items()
        if str(video.get("subset")) == "validation" and _valid_annotations(video)
    )
    if not test:
        raise ValueError(
            "S1 manifest requires a non-empty sealed validation/test split"
        )
    fit, gate = _stratified_gate_split(
        development, gate_ratio=gate_ratio, seed=split_seed
    )
    fit_durations = [
        end - start
        for video_id in fit
        for start, end in (
            item["segment"] for item in _valid_annotations(database[video_id])
        )
    ]
    q1, q2, q3 = (
        float(value) for value in np.quantile(fit_durations, [0.25, 0.5, 0.75])
    )
    fit_class_support = _class_support(database, fit)
    gate_class_support = _class_support(database, gate)
    if set(fit_class_support) != set(gate_class_support):
        raise ValueError(
            "formal S1 fit and gate must both cover every development class"
        )
    fit_duration_support = _duration_support(database, fit, (q1, q2, q3))
    gate_duration_support = _duration_support(database, gate, (q1, q2, q3))
    if set(fit_duration_support) != {"0", "1", "2", "3"} or set(
        gate_duration_support
    ) != {"0", "1", "2", "3"}:
        raise ValueError(
            "formal S1 fit and gate must both cover all duration quartiles"
        )
    splits = {"fit": fit, "gate": gate, "test": test}
    manifest: dict[str, Any] = {
        "schema_version": S1_MANIFEST_SCHEMA,
        "annotation_sha256": sha256_file(annotation_path),
        "annotation_subsets": {"development": "training", "sealed_test": "validation"},
        "split_seed": int(split_seed),
        "gate_ratio": float(gate_ratio),
        "splits": splits,
        "split_hashes": {
            name: stable_id_hash(values) for name, values in splits.items()
        },
        "class_support": {
            "fit": fit_class_support,
            "gate": gate_class_support,
        },
        "duration_video_support": {
            "fit": fit_duration_support,
            "gate": gate_duration_support,
        },
        "duration_quartiles_seconds": {"q1": q1, "q2": q2, "q3": q3},
        "seeds": {
            "training": list(S1_TRAINING_SEEDS),
            "split": int(split_seed),
            "bootstrap": S1_BOOTSTRAP_SEED,
            "profile_order": S1_PROFILE_ORDER_SEED,
        },
        "profile_matrix_order": build_s1_profile_order(),
        "bootstrap": {
            "unit": "video_cluster",
            "paired": True,
            "replicates": S1_BOOTSTRAP_REPLICATES,
            "recompute_full_class_ap": True,
            "require_nonempty_class_support": True,
            "simultaneous_correction": "max_t_for_224_and_256",
        },
        "checkpoint_selection_rule": S1_CHECKPOINT_RULE,
        "pretrained_checkpoint": {
            "filename": S1_PRETRAINED_CHECKPOINT_FILENAME,
            "sha256": S1_PRETRAINED_CHECKPOINT_SHA256,
            "source": "Kinetics-400 VideoMAE-S checkpoint used by the official-derived AdaTAD config",
        },
        "official_test_sealed_until_protocol_freeze": True,
        "runtime_overrides": {
            "development_train": {
                "dataset.train.block_list": "fit_block_list.txt",
                "dataset.val.subset_name": "training",
                "dataset.val.block_list": "gate_block_list.txt",
                "dataset.test.subset_name": "training",
                "dataset.test.block_list": "gate_block_list.txt",
                "evaluation.subset": "training",
            },
            "frozen_test": {
                "dataset.test.subset_name": "validation",
                "dataset.test.block_list": None,
                "evaluation.subset": "validation",
            },
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_s1_manifest(
    manifest: Mapping[str, Any], *, annotation_path: str | Path | None = None
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(manifest)))
    if checked.get("schema_version") != S1_MANIFEST_SCHEMA:
        raise ValueError("unsupported S1 manifest schema")
    expected_hash = checked.pop("manifest_sha256", None)
    if not expected_hash or canonical_sha256(checked) != expected_hash:
        raise ValueError("S1 manifest hash mismatch")
    checked["manifest_sha256"] = expected_hash
    splits = checked.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"fit", "gate", "test"}:
        raise ValueError("S1 manifest requires fit/gate/test splits")
    normalized = {
        name: sorted(set(map(str, splits[name]))) for name in ("fit", "gate", "test")
    }
    if any(not values for values in normalized.values()):
        raise ValueError("S1 fit/gate/test splits must be non-empty")
    if set(normalized["fit"]) & set(normalized["gate"]):
        raise ValueError("S1 fit and gate splits must be disjoint")
    if (set(normalized["fit"]) | set(normalized["gate"])) & set(normalized["test"]):
        raise ValueError("S1 development and sealed test splits must be disjoint")
    expected_split_hashes = {
        name: stable_id_hash(values) for name, values in normalized.items()
    }
    if checked.get("split_hashes") != expected_split_hashes:
        raise ValueError("S1 split hash mismatch")
    if checked.get("seeds", {}).get("training") != list(S1_TRAINING_SEEDS):
        raise ValueError("S1 training seeds must remain frozen")
    if (
        checked.get("seeds", {}).get("profile_order") != S1_PROFILE_ORDER_SEED
        or checked.get("profile_matrix_order") != build_s1_profile_order()
    ):
        raise ValueError("S1 profile matrix order must remain frozen")
    if (
        checked.get("split_seed") != S1_SPLIT_SEED
        or checked.get("gate_ratio") != S1_GATE_RATIO
    ):
        raise ValueError("S1 manifest split parameters changed")
    if checked.get("bootstrap", {}).get("replicates") != S1_BOOTSTRAP_REPLICATES:
        raise ValueError("S1 bootstrap replicate count must remain frozen")
    if checked.get("checkpoint_selection_rule") != S1_CHECKPOINT_RULE:
        raise ValueError("S1 checkpoint selection rule changed")
    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        if sha256_file(annotation_path) != checked.get("annotation_sha256"):
            raise ValueError("annotation hash does not match the frozen S1 manifest")
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        database = payload["database"]
        development = {
            str(video_id)
            for video_id, video in database.items()
            if str(video.get("subset")) == "training" and _valid_annotations(video)
        }
        test = {
            str(video_id)
            for video_id, video in database.items()
            if str(video.get("subset")) == "validation" and _valid_annotations(video)
        }
        if set(normalized["fit"]) | set(normalized["gate"]) != development:
            raise ValueError(
                "S1 fit/gate splits do not exactly cover development videos"
            )
        if set(normalized["test"]) != test:
            raise ValueError(
                "S1 test split does not exactly cover the sealed test videos"
            )
        expected_class_support = {
            name: _class_support(database, normalized[name]) for name in ("fit", "gate")
        }
        if checked.get("class_support") != expected_class_support:
            raise ValueError("S1 fit/gate class-support summary mismatch")
        if set(expected_class_support["fit"]) != set(expected_class_support["gate"]):
            raise ValueError("S1 fit/gate class support is not matched")
        quartiles = checked["duration_quartiles_seconds"]
        bounds = (
            float(quartiles["q1"]),
            float(quartiles["q2"]),
            float(quartiles["q3"]),
        )
        expected_duration_support = {
            name: _duration_support(database, normalized[name], bounds)
            for name in ("fit", "gate")
        }
        if checked.get("duration_video_support") != expected_duration_support:
            raise ValueError("S1 fit/gate action-duration support summary mismatch")
        if any(
            set(expected_duration_support[name]) != {"0", "1", "2", "3"}
            for name in ("fit", "gate")
        ):
            raise ValueError("S1 fit/gate duration-quartile support is incomplete")
        deterministic = build_s1_manifest(annotation_path)
        if checked != deterministic:
            differing = sorted(
                key
                for key in set(checked) | set(deterministic)
                if checked.get(key) != deterministic.get(key)
            )
            raise ValueError(
                "S1 manifest differs from the deterministic frozen protocol: "
                f"fields={differing}"
            )
    checked["splits"] = normalized
    return checked


def write_s1_manifest_bundle(
    manifest: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    checked = validate_s1_manifest(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "spatial_zoom_s1_manifest.json"
    fit_block_path = output_dir / "fit_block_list.txt"
    gate_block_path = output_dir / "gate_block_list.txt"
    existing = [
        str(path)
        for path in (manifest_path, fit_block_path, gate_block_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite frozen S1 manifest artifacts: {existing}"
        )
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(checked, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with fit_block_path.open("x", encoding="utf-8") as handle:
        handle.write("".join(f"{item}\n" for item in checked["splits"]["gate"]))
    with gate_block_path.open("x", encoding="utf-8") as handle:
        handle.write("".join(f"{item}\n" for item in checked["splits"]["fit"]))
    return {
        "manifest": manifest_path,
        "fit_block_list": fit_block_path,
        "gate_block_list": gate_block_path,
    }


__all__ = [
    "S1_BOOTSTRAP_REPLICATES",
    "S1_CHECKPOINT_RULE",
    "S1_MANIFEST_SCHEMA",
    "S1_PRETRAINED_CHECKPOINT_FILENAME",
    "S1_PRETRAINED_CHECKPOINT_SHA256",
    "S1_PROFILE_ORDER_SEED",
    "S1_RESOLUTIONS",
    "S1_TRAINING_SEEDS",
    "build_s1_manifest",
    "build_s1_profile_order",
    "canonical_sha256",
    "sha256_file",
    "stable_id_hash",
    "validate_s1_manifest",
    "write_s1_manifest_bundle",
]
