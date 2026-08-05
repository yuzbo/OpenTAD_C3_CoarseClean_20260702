from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from tools.bata import duca_paper_training
from tools.bata.validate_duca_paper_numeric_gate import validate_numeric_gate_artifact


SCHEMA = "duca_paper_exact211_physical_uid_gate_v1"
CONFIG_DEFAULT = "configs/adatad/thumos/duca_paper_duca_fixed_k384_full200.py"


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(f"fail-closed DUCA paper exact-211 UID gate: {message}")


def _path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return duca_paper_training.canonical_sha256(value)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _bind_clean_commit(expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected) is not None,
        "an exact 40-character commit is required",
    )
    head = _git("rev-parse", "--verify", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    _require(head == expected, "checked-out commit drift")
    _require(not status, "gate requires a clean checkout")
    return {
        "git_commit": head,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_tree_clean": True,
    }


def _require_hashed_file(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    source = _path(path)
    expected = str(expected_sha256).strip().lower()
    _require(source.is_file(), f"{label} is missing: {source}")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        f"invalid {label} SHA-256",
    )
    observed = _sha256(source)
    _require(observed == expected, f"{label} SHA-256 drift")
    return {"path": str(source), "sha256": observed}


def _official_validation_ids(annotation: Path) -> list[str]:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    database = payload.get("database", {})
    _require(isinstance(database, Mapping), "annotation lacks a database mapping")
    validation = sorted(
        str(video_id)
        for video_id, row in database.items()
        if str(row.get("subset", "")) == "validation"
    )
    _require(
        len(validation) == duca_paper_training.EVALUATION_VIDEO_COUNT,
        "annotation does not contain exactly 211 validation videos",
    )
    return validation


def enumerate_exact211_physical_identities(dataset) -> dict[str, Any]:
    _require(
        dataset.__class__.__name__ == "ThumosSlidingDataset",
        "formal evaluation dataset is not ThumosSlidingDataset",
    )
    _require(str(dataset.subset_name) == "validation", "dataset subset is not validation")
    rows = []
    video_counts: Counter[str] = Counter()
    physical_uids = set()
    window_keys = set()
    ordered_video_ids = []
    seen_videos = set()
    for source_index, row in enumerate(dataset.data_list):
        _require(len(row) == 4, "sliding-window metadata row is malformed")
        video_id = str(row[0])
        centers = row[3]
        center_values = [int(value) for value in centers.tolist()]
        _require(center_values, "sliding-window metadata row has no source frames")
        _require(
            center_values == sorted(set(center_values)),
            "dense source-frame sequence is not strictly unique and ordered",
        )
        window_start = int(center_values[0])
        dense_valid_len = len(center_values)
        source_sha = _canonical_sha256(center_values)
        physical_identity = {
            "video_id": video_id,
            "window_start_frame": window_start,
            "dense_valid_len": dense_valid_len,
            "dense_source_frame_sequence_sha256": source_sha,
        }
        physical_uid = _canonical_sha256(physical_identity)
        window_key = (video_id, window_start)
        _require(window_key not in window_keys, "duplicate (video,start) window identity")
        _require(physical_uid not in physical_uids, "duplicate physical-window UID")
        window_keys.add(window_key)
        physical_uids.add(physical_uid)
        video_counts[video_id] += 1
        if video_id not in seen_videos:
            seen_videos.add(video_id)
            ordered_video_ids.append(video_id)
        rows.append(
            {
                "source_index": int(source_index),
                **physical_identity,
                "physical_window_uid": physical_uid,
            }
        )
    _require(rows, "evaluation metadata enumeration is empty")
    target_count = sum(
        1
        for row in rows
        if row["video_id"] == "video_test_0001431"
        and int(row["window_start_frame"]) == 7680
    )
    _require(
        target_count == 1,
        "historical duplicate regression key is not enumerated exactly once",
    )
    return {
        "dataset_class": dataset.__class__.__name__,
        "subset_name": str(dataset.subset_name),
        "video_count": len(video_counts),
        "window_count": len(rows),
        "physical_window_uid_count": len(physical_uids),
        "duplicate_video_start_count": 0,
        "duplicate_physical_uid_count": 0,
        "ordered_video_ids_sha256": _canonical_sha256(ordered_video_ids),
        "sorted_video_ids_sha256": _canonical_sha256(sorted(video_counts)),
        "ordered_physical_windows_sha256": _canonical_sha256(rows),
        "per_video_window_counts_sha256": _canonical_sha256(
            {key: video_counts[key] for key in sorted(video_counts)}
        ),
        "historical_regression_key": {
            "video_id": "video_test_0001431",
            "window_start_frame": 7680,
            "count": target_count,
        },
    }


def run_gate(
    *,
    expected_commit: str,
    numeric_gate_path: str | Path,
    numeric_gate_sha256: str,
    annotation_path: str | Path,
    annotation_sha256: str,
    class_map_path: str | Path,
    class_map_sha256: str,
    test_data_path: str | Path,
    config_path: str | Path = CONFIG_DEFAULT,
) -> dict[str, Any]:
    git = _bind_clean_commit(expected_commit)
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    _require(job_id.isdigit(), "gate must run inside a numeric Slurm job")
    numeric_gate = validate_numeric_gate_artifact(
        numeric_gate_path,
        expected_commit=expected_commit,
        expected_sha256=numeric_gate_sha256,
    )
    annotation = _require_hashed_file(
        annotation_path, annotation_sha256, "THUMOS14 annotation"
    )
    class_map = _require_hashed_file(
        class_map_path, class_map_sha256, "THUMOS14 class map"
    )
    test_data = _path(test_data_path)
    _require(test_data.is_dir(), "THUMOS14 validation video directory is missing")
    config_file = _path(
        ROOT / config_path if not Path(config_path).is_absolute() else config_path
    )
    _require(
        config_file == _path(ROOT / CONFIG_DEFAULT),
        "gate is fixed to the formal learned-DUCA config",
    )
    cfg = Config.fromfile(str(config_file))
    static = duca_paper_training.validate_static_config(cfg)
    _require(static["variant"] == "duca_fixed_k384", "config arm drift")
    cfg.dataset.test.ann_file = annotation["path"]
    cfg.dataset.test.class_map = class_map["path"]
    cfg.dataset.test.data_path = str(test_data)
    # Metadata-only construction: Compose([]) builds the exact sliding-window
    # index but never opens, decodes, transforms, or evaluates a video.
    cfg.dataset.test.pipeline = []
    from opentad.datasets import build_dataset

    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    identity = enumerate_exact211_physical_identities(dataset)
    official_ids = _official_validation_ids(_path(annotation["path"]))
    _require(identity["video_count"] == 211, "metadata dataset is not exact-211")
    _require(
        identity["sorted_video_ids_sha256"] == _canonical_sha256(official_ids),
        "dataset IDs differ from the official validation annotation",
    )
    payload = {
        "schema_version": SCHEMA,
        "status": "passed",
        "fail_closed": True,
        "git_commit": git["git_commit"],
        "git_binding": git,
        "slurm_job_id": job_id,
        "prerequisite_numeric_gate": numeric_gate,
        "config": {
            "path": str(config_file),
            "sha256": _sha256(config_file),
            "resolved_before_runtime_binding_sha256": _canonical_sha256(
                Config.fromfile(str(config_file)).to_dict()
            ),
            "arm": static["variant"],
        },
        "assets": {
            "annotation": annotation,
            "class_map": class_map,
            "test_data_path": str(test_data),
        },
        "enumeration": identity,
        "metadata_only": True,
        "video_decode_executed": False,
        "model_or_backbone_executed": False,
        "prediction_generated": False,
        "metric_accessed": False,
        "validation_labels_used_for_selection": False,
        "paper_metric_claim_allowed": False,
        "paper_method_performance_evidence": False,
        "claim_scope": "engineering_exact211_physical_identity_only",
        "stage_a_release_prerequisite_satisfied": True,
        "stage_b_enabled": False,
        "official_final_consumed": False,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _path(path)
    try:
        target.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise GateFailure("gate receipt must stay outside the Git worktree")
    _require(not target.exists(), "refusing to overwrite exact-211 gate evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--numeric-gate", required=True)
    parser.add_argument("--numeric-gate-sha256", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--class-map-sha256", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = run_gate(
        expected_commit=args.expected_commit,
        numeric_gate_path=args.numeric_gate,
        numeric_gate_sha256=args.numeric_gate_sha256,
        annotation_path=args.annotation,
        annotation_sha256=args.annotation_sha256,
        class_map_path=args.class_map,
        class_map_sha256=args.class_map_sha256,
        test_data_path=args.test_data_path,
        config_path=args.config,
    )
    target = _write_new(args.output_json, payload)
    print(json.dumps({"path": str(target), "sha256": _sha256(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
