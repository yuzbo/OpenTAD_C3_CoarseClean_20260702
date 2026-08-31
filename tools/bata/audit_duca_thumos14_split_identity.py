"""Read-only THUMOS14 split identity audit for the DUCA full-data protocol.

The audit deliberately never decodes held-out annotation labels or temporal
segments.  It materializes literal video identities, resolved config identity,
physical-media identity, loader/evaluator membership, and the historical
211/ActionFormer source sets.  It does not load a checkpoint, run a model, or
compute a metric.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1 = ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py"
DEFAULT_STAGE2 = ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
FROZEN_H65_BASE = "04c35a3b76897e6c1569eeede41ed3aecaf7f854"
PADDING_DATASET_SOURCE = ROOT / "opentad/datasets/base/padding_dataset.py"
SLIDING_DATASET_SOURCE = ROOT / "opentad/datasets/base/sliding_dataset.py"
EVALUATOR_SOURCE = ROOT / "opentad/evaluations/mAP.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_id_manifest(path: Path, ids: Iterable[str]) -> str:
    path.write_text("".join(f"{item}\n" for item in sorted(set(ids))), encoding="utf-8")
    return sha256_file(path)


class IdentityJsonError(ValueError):
    pass


class _JsonCursor:
    """Small JSON scanner that can skip values without decoding their contents."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.decoded_strings: list[str] = []

    def _ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def peek(self) -> str:
        self._ws()
        if self.pos >= len(self.text):
            raise IdentityJsonError("unexpected end of JSON")
        return self.text[self.pos]

    def expect(self, token: str) -> None:
        self._ws()
        if not self.text.startswith(token, self.pos):
            raise IdentityJsonError(f"expected {token!r} at byte {self.pos}")
        self.pos += len(token)

    def _string_end(self) -> int:
        self._ws()
        if self.pos >= len(self.text) or self.text[self.pos] != '"':
            raise IdentityJsonError(f"expected string at byte {self.pos}")
        idx = self.pos + 1
        while idx < len(self.text):
            char = self.text[idx]
            if char == "\\":
                idx += 2
                continue
            if char == '"':
                return idx + 1
            idx += 1
        raise IdentityJsonError("unterminated JSON string")

    def parse_string(self) -> str:
        end = self._string_end()
        raw = self.text[self.pos : end]
        self.pos = end
        value = json.loads(raw)
        if not isinstance(value, str):
            raise IdentityJsonError("decoded JSON string is not a string")
        self.decoded_strings.append(value)
        return value

    def skip_string(self) -> None:
        self.pos = self._string_end()

    def parse_scalar(self) -> Any:
        self._ws()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",]} \t\r\n":
            self.pos += 1
        if self.pos == start:
            raise IdentityJsonError(f"expected scalar at byte {self.pos}")
        return json.loads(self.text[start : self.pos])

    def skip_value(self) -> tuple[int, int]:
        self._ws()
        start = self.pos
        token = self.peek()
        if token == '"':
            self.skip_string()
        elif token == "{":
            self.expect("{")
            if self.peek() == "}":
                self.expect("}")
            else:
                while True:
                    self.skip_string()
                    self.expect(":")
                    self.skip_value()
                    if self.peek() == "}":
                        self.expect("}")
                        break
                    self.expect(",")
        elif token == "[":
            self.expect("[")
            if self.peek() == "]":
                self.expect("]")
            else:
                while True:
                    self.skip_value()
                    if self.peek() == "]":
                        self.expect("]")
                        break
                    self.expect(",")
        else:
            self.parse_scalar()
        return start, self.pos


def _count_array_items_without_decoding(text: str) -> int:
    cursor = _JsonCursor(text)
    cursor.expect("[")
    if cursor.peek() == "]":
        cursor.expect("]")
        return 0
    count = 0
    while True:
        cursor.skip_value()
        count += 1
        if cursor.peek() == "]":
            cursor.expect("]")
            break
        cursor.expect(",")
    return count


def _count_non_ambiguous_training_annotations(text: str) -> int:
    """Read only training-side label membership; segment values stay skipped."""
    cursor = _JsonCursor(text)
    cursor.expect("[")
    if cursor.peek() == "]":
        cursor.expect("]")
        return 0
    count = 0
    while True:
        cursor.expect("{")
        label: str | None = None
        if cursor.peek() == "}":
            cursor.expect("}")
        else:
            while True:
                key = cursor.parse_string()
                cursor.expect(":")
                if key == "label":
                    label = cursor.parse_string()
                else:
                    cursor.skip_value()
                if cursor.peek() == "}":
                    cursor.expect("}")
                    break
                cursor.expect(",")
        if label is not None and label != "Ambiguous":
            count += 1
        if cursor.peek() == "]":
            cursor.expect("]")
            break
        cursor.expect(",")
    return count


@dataclass(frozen=True)
class VideoIdentity:
    video_id: str
    subset: str
    annotation_count: int
    valid_training_annotation_count: int | None
    frame: int | None
    duration: float | None
    fps: float | None


@dataclass(frozen=True)
class AnnotationIdentity:
    records: tuple[VideoIdentity, ...]
    decoded_strings: tuple[str, ...]
    held_out_annotation_values_decoded: bool


def _parse_video_record(cursor: _JsonCursor, training_subset: str | None) -> VideoIdentity:
    cursor.expect("{")
    subset: str | None = None
    annotations_span: tuple[int, int] | None = None
    frame: int | None = None
    duration: float | None = None
    fps: float | None = None
    values: dict[str, Any] = {}
    if cursor.peek() == "}":
        cursor.expect("}")
    else:
        while True:
            key = cursor.parse_string()
            cursor.expect(":")
            if key == "subset":
                subset = cursor.parse_string()
            elif key == "annotations":
                annotations_span = cursor.skip_value()
            elif key in {"frame", "duration", "fps"}:
                values[key] = cursor.parse_scalar()
            else:
                cursor.skip_value()
            if cursor.peek() == "}":
                cursor.expect("}")
                break
            cursor.expect(",")
    if subset is None:
        raise IdentityJsonError("video record lacks literal subset")
    annotation_text = "[]"
    if annotations_span is not None:
        annotation_text = cursor.text[annotations_span[0] : annotations_span[1]]
    annotation_count = _count_array_items_without_decoding(annotation_text)
    valid_training_count = None
    if training_subset is not None and subset == training_subset:
        valid_training_count = _count_non_ambiguous_training_annotations(annotation_text)
    if values.get("frame") is not None:
        frame = int(values["frame"])
    if values.get("duration") is not None:
        duration = float(values["duration"])
    if values.get("fps") is not None:
        fps = float(values["fps"])
    return VideoIdentity("", subset, annotation_count, valid_training_count, frame, duration, fps)


def parse_annotation_identity(path: Path, training_subset: str | None = "training") -> AnnotationIdentity:
    cursor = _JsonCursor(path.read_text(encoding="utf-8-sig"))
    cursor.expect("{")
    records: list[VideoIdentity] = []
    seen: set[str] = set()
    found_database = False
    if cursor.peek() == "}":
        cursor.expect("}")
    else:
        while True:
            key = cursor.parse_string()
            cursor.expect(":")
            if key == "database":
                if found_database:
                    raise IdentityJsonError("duplicate database field")
                found_database = True
                cursor.expect("{")
                if cursor.peek() == "}":
                    cursor.expect("}")
                else:
                    while True:
                        video_id = cursor.parse_string()
                        if video_id in seen:
                            raise IdentityJsonError(f"duplicate video ID: {video_id}")
                        seen.add(video_id)
                        cursor.expect(":")
                        record = _parse_video_record(cursor, training_subset)
                        records.append(
                            VideoIdentity(
                                video_id,
                                record.subset,
                                record.annotation_count,
                                record.valid_training_annotation_count,
                                record.frame,
                                record.duration,
                                record.fps,
                            )
                        )
                        if cursor.peek() == "}":
                            cursor.expect("}")
                            break
                        cursor.expect(",")
            else:
                cursor.skip_value()
            if cursor.peek() == "}":
                cursor.expect("}")
                break
            cursor.expect(",")
    if not found_database:
        raise IdentityJsonError("annotation JSON lacks database")
    return AnnotationIdentity(tuple(records), tuple(cursor.decoded_strings), False)


def extract_mapping_ids(path: Path, mapping_key: str = "results") -> set[str]:
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    cursor = _JsonCursor(path.read_text(encoding="utf-8-sig"))
    if cursor.peek() == "[":
        cursor.expect("[")
        ids: set[str] = set()
        if cursor.peek() == "]":
            cursor.expect("]")
            return ids
        while True:
            value = cursor.parse_string()
            if value in ids:
                raise IdentityJsonError(f"duplicate ID in {path}: {value}")
            ids.add(value)
            if cursor.peek() == "]":
                cursor.expect("]")
                return ids
            cursor.expect(",")
    cursor.expect("{")
    ids: set[str] | None = None
    while True:
        key = cursor.parse_string()
        cursor.expect(":")
        if key == mapping_key:
            ids = set()
            cursor.expect("{")
            if cursor.peek() == "}":
                cursor.expect("}")
            else:
                while True:
                    item = cursor.parse_string()
                    if item in ids:
                        raise IdentityJsonError(f"duplicate ID in {path}: {item}")
                    ids.add(item)
                    cursor.expect(":")
                    cursor.skip_value()
                    if cursor.peek() == "}":
                        cursor.expect("}")
                        break
                    cursor.expect(",")
        else:
            cursor.skip_value()
        if cursor.peek() == "}":
            cursor.expect("}")
            break
        cursor.expect(",")
    if ids is None:
        raise IdentityJsonError(f"{path} lacks {mapping_key!r} mapping")
    return ids


def _literal_bases(path: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
            value = ast.literal_eval(node.value)
            values = [value] if isinstance(value, str) else list(value)
            return [(path.parent / item).resolve() for item in values]
    return []


def config_inheritance_chain(path: Path) -> list[Path]:
    ordered: list[Path] = []
    active: set[Path] = set()
    visited: set[Path] = set()

    def visit(current: Path) -> None:
        current = current.resolve()
        if current in active:
            raise ValueError(f"config inheritance cycle at {current}")
        if current in visited:
            return
        if not current.is_file():
            raise FileNotFoundError(current)
        active.add(current)
        ordered.append(current)
        for base in _literal_bases(current):
            visit(base)
        active.remove(current)
        visited.add(current)

    visit(path)
    return ordered


@contextmanager
def _stage2_identity_environment() -> Iterable[None]:
    names = {
        "DUCA_STAGE1_CHECKPOINT": "IDENTITY_AUDIT_PLACEHOLDER_DO_NOT_LOAD",
        "DUCA_STAGE1_CHECKPOINT_SHA256": "0" * 64,
        "DUCA_STAGE1_CHECKPOINT_EPOCH": "29",
    }
    prior = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_resolved_config(path: Path) -> Mapping[str, Any]:
    from mmengine.config import Config

    with _stage2_identity_environment():
        return Config.fromfile(str(path)).to_dict()


def _plain_dataset_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = cfg["dataset"]
    fields = ("type", "ann_file", "subset_name", "block_list", "class_map", "data_path", "filter_gt", "test_mode")

    def select(section: Mapping[str, Any]) -> dict[str, Any]:
        return {field: section.get(field) for field in fields}

    return {
        "train": select(dataset["train"]),
        "val": select(dataset["val"]),
        "test": select(dataset["test"]),
        "evaluation": {
            key: cfg["evaluation"].get(key)
            for key in ("type", "subset", "ground_truth_filename", "tiou_thresholds")
        },
    }


def _block_list_ids(value: Any, repo_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    if value is None:
        return set(), []
    if isinstance(value, (list, tuple)):
        return set(map(str, value)), [{"kind": "config_literal", "ids": sorted(map(str, value))}]
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"block list does not exist: {path}")
    ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        item = line.strip()
        if item:
            ids.add(item)
            sources.append({"id": item, "path": str(path), "line": line_number, "text": line})
    return ids, sources


def discover_physical_files(root: Path, suffix: str) -> tuple[dict[str, Path], list[str]]:
    mapping: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name.lower().endswith(suffix.lower()):
            video_id = path.name[: -len(suffix)]
            if video_id in mapping:
                duplicates.append(video_id)
            else:
                mapping[video_id] = path
    return mapping, sorted(duplicates)


def probe_with_ffprobe(path: Path, executable: str, timeout: int = 30) -> dict[str, Any]:
    resolved = shutil.which(executable) if not Path(executable).is_file() else executable
    if not resolved:
        raise RuntimeError(f"ffprobe executable not found: {executable}")
    result = subprocess.run(
        [
            str(resolved),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1 or int(streams[0].get("width", 0)) <= 0 or int(streams[0].get("height", 0)) <= 0:
        raise RuntimeError("no decodable primary video stream")
    return streams[0]


def discover_feature_ids(root: Path, suffixes: Sequence[str]) -> set[str]:
    normalized = tuple(suffix.lower() for suffix in suffixes)
    ids: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized:
            ids.add(path.stem)
    return ids


def source_hits(ids: Iterable[str], sources: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    wanted = set(ids)
    hits = {item: [] for item in wanted}
    for source in sources:
        if not source.is_file():
            continue
        for line_number, line in enumerate(source.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            for item in wanted:
                if item in line:
                    hits[item].append(
                        {"path": str(source.resolve()), "line": line_number, "text": line.strip()}
                    )
    return hits


def _source_line_map(path: Path, patterns: Mapping[str, str]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    locations: dict[str, int | None] = {}
    for name, pattern in patterns.items():
        locations[name] = next((idx for idx, line in enumerate(lines, 1) if pattern in line), None)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "lines": locations}


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "head": git("rev-parse", "HEAD"),
        "parent": git("rev-parse", "HEAD^"),
        "branch": git("branch", "--show-current"),
        "clean": status == "",
        "status": status.splitlines(),
    }


def _difference(left_name: str, left: set[str], right_name: str, right: set[str]) -> dict[str, Any]:
    return {
        "left": left_name,
        "right": right_name,
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
        "equal": left == right,
    }


def run_audit(
    args: argparse.Namespace,
    probe: Callable[[Path, str, int], Mapping[str, Any]] = probe_with_ffprobe,
) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    stage1_path = Path(args.stage1_config).resolve()
    stage2_path = Path(args.stage2_config).resolve()
    annotation_path = Path(args.annotation).resolve()
    class_map_path = Path(args.class_map).resolve()
    media_root = Path(args.media_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (stage1_path, stage2_path, annotation_path, class_map_path, media_root):
        if not path.exists():
            raise FileNotFoundError(path)

    stage1_cfg = load_resolved_config(stage1_path)
    stage2_cfg = load_resolved_config(stage2_path)
    stage1_data = _plain_dataset_identity(stage1_cfg)
    stage2_data = _plain_dataset_identity(stage2_cfg)
    config_blockers: list[str] = []
    if stage1_data != stage2_data:
        config_blockers.append("Stage-1 and Stage-2 resolved dataset/evaluator identities differ")

    train_subset = str(stage1_data["train"]["subset_name"])
    heldout_subset = str(stage1_data["test"]["subset_name"])
    evaluation_subset = str(stage1_data["evaluation"]["subset"])
    annotation = parse_annotation_identity(annotation_path, training_subset=train_subset)
    records = {record.video_id: record for record in annotation.records}
    annotation_train = {record.video_id for record in annotation.records if record.subset == train_subset}
    annotation_heldout = {record.video_id for record in annotation.records if record.subset == heldout_subset}

    train_blocked, train_block_sources = _block_list_ids(stage1_data["train"]["block_list"], repo_root)
    test_blocked, test_block_sources = _block_list_ids(stage1_data["test"]["block_list"], repo_root)
    evaluator_blocked = set(test_blocked)
    if bool(stage1_data["train"].get("test_mode", False)):
        loader_train = annotation_train - train_blocked
    else:
        loader_train = {
            item
            for item in annotation_train - train_blocked
            if (records[item].valid_training_annotation_count or 0) > 0
        }
    if not bool(stage1_data["test"].get("test_mode", False)):
        config_blockers.append("held-out loader is not test_mode=True; label-free membership cannot be proven")
        loader_heldout: set[str] = set()
    else:
        loader_heldout = annotation_heldout - test_blocked
    evaluator_heldout = {
        item
        for item in annotation_heldout - evaluator_blocked
        if records[item].annotation_count > 0 and records[item].subset == evaluation_subset
    }

    physical_map, physical_duplicates = discover_physical_files(media_root, args.media_suffix)
    physical_ids = set(physical_map)
    physical_train = annotation_train & physical_ids
    physical_heldout = annotation_heldout & physical_ids
    decode_failures: dict[str, str] = {}
    decoded_ids: set[str] = set()
    if args.skip_media_decode:
        config_blockers.append("basic media decode was explicitly skipped")
    else:
        for video_id in sorted(annotation_train | annotation_heldout):
            path = physical_map.get(video_id)
            if path is None or not path.is_file():
                decode_failures[video_id] = "physical file missing or broken symlink"
                continue
            try:
                probe(path, args.ffprobe, args.ffprobe_timeout)
                decoded_ids.add(video_id)
            except Exception as exc:  # one bounded identity failure per video
                decode_failures[video_id] = str(exc)

    historical_status = "FOUND"
    historical_211: set[str] = set()
    if args.historical_211 is None:
        historical_status = "SOURCE_NOT_FOUND"
    else:
        historical_211 = extract_mapping_ids(Path(args.historical_211).resolve(), args.historical_mapping_key)

    actionformer_status = "FOUND"
    actionformer_annotation: set[str] = set()
    actionformer_feature: set[str] | None = None
    actionformer_loader: set[str] | None = None
    if args.actionformer_annotation is None:
        actionformer_status = "SOURCE_NOT_FOUND"
    else:
        af_identity = parse_annotation_identity(Path(args.actionformer_annotation).resolve(), training_subset=None)
        actionformer_annotation = {
            record.video_id for record in af_identity.records if record.subset == args.actionformer_subset
        }
        if args.actionformer_feature_root is not None:
            actionformer_feature = discover_feature_ids(
                Path(args.actionformer_feature_root).resolve(), args.actionformer_feature_suffix
            )
            actionformer_loader = actionformer_annotation & actionformer_feature

    differences = [
        _difference("T_annotation", annotation_train, "T_loader", loader_train),
        _difference("T_annotation", annotation_train, "T_physical", physical_train),
        _difference("H_annotation", annotation_heldout, "H_loader", loader_heldout),
        _difference("H_annotation", annotation_heldout, "H_physical", physical_heldout),
        _difference("H_annotation", annotation_heldout, "H_evaluator", evaluator_heldout),
    ]
    if historical_status == "FOUND":
        differences.append(_difference("H_annotation", annotation_heldout, "H_prediction_211", historical_211))
    if actionformer_status == "FOUND":
        differences.append(_difference("H_annotation", annotation_heldout, "AF_annotation", actionformer_annotation))
    if actionformer_loader is not None:
        differences.append(_difference("AF_annotation", actionformer_annotation, "AF_loader", actionformer_loader))

    af_difference_ids = annotation_heldout ^ actionformer_annotation if actionformer_status == "FOUND" else set()
    exclusion_sources = [Path(item).resolve() for item in args.exclusion_source]
    exclusion_hits = source_hits(af_difference_ids, exclusion_sources)
    unexplained_af_ids = sorted(item for item in af_difference_ids if not exclusion_hits.get(item))

    blockers = list(config_blockers)
    if len(annotation_train) != 200:
        blockers.append(f"training annotation count is {len(annotation_train)}, expected 200")
    if annotation_train & annotation_heldout:
        blockers.append("training and held-out annotation IDs intersect")
    for diff in differences[:5]:
        if not diff["equal"]:
            blockers.append(f"{diff['left']} != {diff['right']}")
    if physical_duplicates:
        blockers.append("duplicate literal physical IDs")
    if decode_failures:
        blockers.append("one or more expected physical videos failed basic decode")
    if historical_status != "FOUND":
        blockers.append("historical 211 ID source not found")
    elif historical_211 != annotation_heldout:
        blockers.append("historical 211 IDs differ from held-out annotation IDs")
    if actionformer_status != "FOUND":
        blockers.append("ActionFormer 212 source not found")
    elif unexplained_af_ids:
        blockers.append("ActionFormer/OpenTAD ID difference lacks source-backed explanation")
    if actionformer_loader is not None and actionformer_loader != actionformer_annotation:
        blockers.append("ActionFormer loader silently drops one or more annotation IDs")

    git_identity = _git_identity(repo_root)
    if git_identity["head"] != FROZEN_H65_BASE and git_identity["parent"] != FROZEN_H65_BASE:
        blockers.append("audit revision is not the frozen H65 base or its direct child")
    if not git_identity["clean"]:
        # The audit source and test are expected to be uncommitted during local development.
        allowed = {
            "tools/bata/audit_duca_thumos14_split_identity.py",
            "tests/test_audit_duca_thumos14_split_identity.py",
        }
        dirty_paths = {
            line[3:].replace("\\", "/") for line in git_identity["status"] if len(line) >= 4
        }
        if not dirty_paths <= allowed:
            blockers.append("audit worktree has changes outside the two authorized files")

    class_map_lines = [line for line in class_map_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    source_identity = {
        "padding_dataset": _source_line_map(
            PADDING_DATASET_SOURCE,
            {"get_dataset": "def get_dataset", "subset_filter": 'video_info["subset"] not in self.subset_name'},
        ),
        "sliding_dataset": _source_line_map(
            SLIDING_DATASET_SOURCE,
            {"get_dataset": "def get_dataset", "subset_filter": 'video_info["subset"] not in self.subset_name'},
        ),
        "evaluator": _source_line_map(
            EVALUATOR_SOURCE,
            {"ground_truth_import": "def _import_ground_truth", "subset_filter": 'self.subset != v["subset"]'},
        ),
    }

    manifests = {
        "T_annotation": write_id_manifest(output_dir / "training_annotation_ids.txt", annotation_train),
        "T_loader": write_id_manifest(output_dir / "training_loader_ids.txt", loader_train),
        "T_physical": write_id_manifest(output_dir / "training_physical_ids.txt", physical_train),
        "H_annotation": write_id_manifest(output_dir / "held_out_annotation_ids.txt", annotation_heldout),
        "H_loader": write_id_manifest(output_dir / "held_out_loader_ids.txt", loader_heldout),
        "H_physical": write_id_manifest(output_dir / "held_out_physical_ids.txt", physical_heldout),
        "H_evaluator": write_id_manifest(output_dir / "held_out_evaluator_ids.txt", evaluator_heldout),
        "H_prediction_211": write_id_manifest(output_dir / "historical_211_ids.txt", historical_211),
        "AF_annotation": write_id_manifest(output_dir / "actionformer_annotation_ids.txt", actionformer_annotation),
    }
    if actionformer_feature is not None:
        manifests["AF_feature"] = write_id_manifest(output_dir / "actionformer_feature_ids.txt", actionformer_feature)
    if actionformer_loader is not None:
        manifests["AF_loader"] = write_id_manifest(output_dir / "actionformer_loader_ids.txt", actionformer_loader)

    with (output_dir / "set_differences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right", "side", "video_id"])
        for diff in differences:
            for item in diff["left_only"]:
                writer.writerow([diff["left"], diff["right"], "left_only", item])
            for item in diff["right_only"]:
                writer.writerow([diff["left"], diff["right"], "right_only", item])

    report = {
        "schema": "duca_thumos14_split_identity_v1",
        "verdict": "BLOCK" if blockers else "PASS",
        "blockers": sorted(set(blockers)),
        "git": git_identity,
        "config": {
            "stage1": {
                "path": str(stage1_path),
                "inheritance_chain": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in config_inheritance_chain(stage1_path)
                ],
            },
            "stage2": {
                "path": str(stage2_path),
                "inheritance_chain": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in config_inheritance_chain(stage2_path)
                ],
                "resolution_placeholders_only": True,
                "checkpoint_loaded": False,
            },
            "resolved_identity": stage1_data,
            "resolved_identity_sha256": canonical_sha256(stage1_data),
            "training_subset": train_subset,
            "held_out_subset": heldout_subset,
            "evaluation_subset": evaluation_subset,
        },
        "sources": source_identity,
        "artifacts": {
            "annotation": {"path": str(annotation_path), "sha256": sha256_file(annotation_path)},
            "class_map": {
                "path": str(class_map_path),
                "sha256": sha256_file(class_map_path),
                "nonempty_line_count": len(class_map_lines),
            },
            "media_root": str(media_root),
            "historical_211_status": historical_status,
            "actionformer_source_status": actionformer_status,
        },
        "identity": {
            "T_annotation": sorted(annotation_train),
            "T_loader": sorted(loader_train),
            "T_physical": sorted(physical_train),
            "H_annotation": sorted(annotation_heldout),
            "H_loader": sorted(loader_heldout),
            "H_physical": sorted(physical_heldout),
            "H_evaluator": sorted(evaluator_heldout),
            "H_prediction_211": sorted(historical_211),
            "AF_annotation": sorted(actionformer_annotation),
            "AF_feature": None if actionformer_feature is None else sorted(actionformer_feature),
            "AF_loader": None if actionformer_loader is None else sorted(actionformer_loader),
            "all_physical_ids": sorted(physical_ids),
        },
        "counts": {
            "T_annotation": len(annotation_train),
            "T_loader": len(loader_train),
            "T_physical": len(physical_train),
            "H_annotation": len(annotation_heldout),
            "H_loader": len(loader_heldout),
            "H_physical": len(physical_heldout),
            "H_evaluator": len(evaluator_heldout),
            "H_prediction_211": len(historical_211),
            "AF_annotation": len(actionformer_annotation),
            "decoded_expected_videos": len(decoded_ids),
        },
        "differences": differences,
        "decode_failures": decode_failures,
        "physical_duplicate_ids": physical_duplicates,
        "physical_unassigned_ids": sorted(physical_ids - annotation_train - annotation_heldout),
        "train_held_out_intersection": sorted(annotation_train & annotation_heldout),
        "exclusions": {
            "train_block_list": train_block_sources,
            "held_out_block_list": test_block_sources,
            "actionformer_open_tad_difference": exclusion_hits,
            "unexplained_actionformer_open_tad_ids": unexplained_af_ids,
        },
        "isolation": {
            "held_out_annotation_values_decoded": annotation.held_out_annotation_values_decoded,
            "training_label_access": "membership-only: non-Ambiguous presence for formal loader replay",
            "held_out_label_or_segment_access": False,
            "checkpoint_loaded": False,
            "prediction_content_read": False,
            "metric_computed": False,
            "gpu_used": False,
        },
        "manifest_sha256": manifests,
    }
    write_json(output_dir / "split_identity_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--stage1-config", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-config", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-suffix", default=".mp4")
    parser.add_argument("--historical-211", type=Path)
    parser.add_argument("--historical-mapping-key", default="results")
    parser.add_argument("--actionformer-annotation", type=Path)
    parser.add_argument("--actionformer-subset", default="test")
    parser.add_argument("--actionformer-feature-root", type=Path)
    parser.add_argument("--actionformer-feature-suffix", nargs="+", default=[".npy"])
    parser.add_argument("--exclusion-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffprobe-timeout", type=int, default=30)
    parser.add_argument("--skip-media-decode", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run_audit(parse_args())
    print(json.dumps({"verdict": report["verdict"], "blockers": report["blockers"]}, ensure_ascii=False))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
