"""Auditable run records and a split shared by every seed and route."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def content_id(value):
    # These digests are checked at resume and dependency loading, not decorative.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_id(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def source_commit(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("formal execution requires a committed clean source snapshot")
    return git("rev-parse", "HEAD")


def prepare_split(annotation, output):
    """Keep every official training and test video; never create a holdout."""
    original = load_json(annotation)
    database = original["database"]
    names = sorted(name for name, row in database.items() if row["subset"] == "training")
    train = names
    split = dict(version="geosparse-official-full-data-20260908", seed=0, training=train,
                 internal_diagnostic=train, diagnostic_scope="training data; not a held-out subset",
                 validation=sorted(k for k, v in database.items() if v["subset"] == "validation"),
                 annotation_sha256=file_id(annotation))
    path = Path(output) / "split.json"
    if path.exists() and load_json(path) != split:
        raise ValueError("split or source annotations changed; create an explicit protocol amendment")
    save_json(path, split)
    save_json(Path(output) / "annotations.json", original)
    durations = sorted(float(a["segment"][1] - a["segment"][0]) for name in train
                       for a in database[name].get("annotations", []) if a["segment"][1] > a["segment"][0])
    index = (len(durations) - 1) * .25
    threshold = durations[math.floor(index)] * (1 - index % 1) + durations[math.ceil(index)] * (index % 1)
    save_json(Path(output) / "risk_thresholds.json", dict(short_action_seconds=threshold,
              definition="training duration 25th percentile; linear interpolation", train_videos=len(train),
              official_train_videos=len(names), small_object="NA: no object-size annotations"))
    return split
