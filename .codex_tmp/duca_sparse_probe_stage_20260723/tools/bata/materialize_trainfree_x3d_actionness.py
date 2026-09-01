from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_SCHEMA_VERSION = "trainfree_x3d_actionness_materialization_v1"
READY = "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED"
GRID_READY = "TRAINFREE_X3D_INTERVAL_GRID_SUMMARY_READY"


def _path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_row_count(path: str | Path) -> int:
    count = 0
    with _path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            count += 1
    return count


def _require_grid_summary(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("decision") != GRID_READY:
        raise ValueError(f"grid summary decision must be {GRID_READY}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("grid summary must contain non-empty rows")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"grid summary rows[{idx}] must be an object")
        out.append(dict(row))
    return out


def _select_preregistered(
    rows: Sequence[Mapping[str, Any]],
    *,
    provider: str | None,
    frame_interval: int | None,
    clip_frames: int | None,
) -> dict[str, Any]:
    if not provider or frame_interval is None:
        raise ValueError("pre_registered materialization requires provider and frame_interval")
    matches = [
        dict(row)
        for row in rows
        if str(row.get("provider")) == str(provider)
        and int(row.get("frame_interval", -1)) == int(frame_interval)
        and (clip_frames is None or int(row.get("clip_frames", -1)) == int(clip_frames))
    ]
    if len(matches) != 1:
        raise ValueError(
            "pre_registered materialization must match exactly one grid cell; "
            f"matched={len(matches)} provider={provider} frame_interval={frame_interval} clip_frames={clip_frames}"
        )
    return matches[0]


def _select_best_by_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str | None,
    allow_validation_selection: bool,
) -> dict[str, Any]:
    if not allow_validation_selection:
        raise ValueError("validation metric selection requires --allow-validation-selection and is not a main-claim path")
    if not metric:
        raise ValueError("best_by_metric materialization requires metric")
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            scored.append((float(value), dict(row)))
    if not scored:
        raise ValueError(f"no grid rows contain numeric metric {metric!r}")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _actionness_jsonl_for_cell(row: Mapping[str, Any], *, subset: str) -> Path:
    explicit = row.get("actionness_jsonl")
    if explicit:
        return _path(str(explicit))
    provider = str(row.get("provider"))
    out_root = row.get("out_root")
    if not provider or not out_root:
        raise ValueError("selected X3D grid row must contain provider and out_root")
    return _path(out_root) / f"{provider}_{subset}_actionness.jsonl"


def _validate_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    provenance = row.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected X3D row must contain source_provenance")
    out = dict(provenance)
    for key in ("thumos_trained", "uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
        if out.get(key) is not False:
            raise ValueError(f"selected X3D source_provenance must set {key}=False")
    return out


def _materialize_file(source: Path, output: Path, *, link_mode: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        return
    if output.exists() or output.is_symlink():
        output.unlink()
    if link_mode == "copy":
        shutil.copyfile(source, output)
    elif link_mode == "symlink":
        output.symlink_to(source)
    elif link_mode == "hardlink":
        os.link(source, output)
    else:
        raise ValueError("link_mode must be one of copy, symlink, hardlink")


def materialize_actionness(
    *,
    grid_summary_json: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    provider: str | None = None,
    frame_interval: int | None = None,
    clip_frames: int | None = None,
    selection_policy: str = "pre_registered",
    metric: str | None = None,
    allow_validation_selection: bool = False,
    link_mode: str = "copy",
) -> dict[str, Any]:
    grid_summary = _read_json(grid_summary_json)
    rows = _require_grid_summary(grid_summary)
    subset = str(grid_summary.get("subset", "validation"))
    if selection_policy == "pre_registered":
        selected = _select_preregistered(
            rows,
            provider=provider,
            frame_interval=frame_interval,
            clip_frames=clip_frames,
        )
        validation_metric_selection = False
    elif selection_policy == "best_by_metric":
        selected = _select_best_by_metric(
            rows,
            metric=metric,
            allow_validation_selection=bool(allow_validation_selection),
        )
        validation_metric_selection = True
    else:
        raise ValueError("selection_policy must be pre_registered or best_by_metric")

    source = _actionness_jsonl_for_cell(selected, subset=subset)
    if not source.is_file():
        raise ValueError(f"selected actionness JSONL does not exist: {source}")
    source_row_count = _jsonl_row_count(source)
    if source_row_count <= 0:
        raise ValueError(f"selected actionness JSONL has no rows: {source}")
    expected_rows = int(selected.get("row_count", source_row_count) or source_row_count)
    if expected_rows != source_row_count:
        raise ValueError(f"selected actionness row_count mismatch: summary={expected_rows} jsonl={source_row_count}")
    provenance = _validate_provenance(selected)
    output = _path(output_jsonl)
    _materialize_file(source, output, link_mode=link_mode)
    source_sha = _sha256(source)
    output_sha = _sha256(output)
    if source_sha != output_sha:
        raise RuntimeError("materialized X3D actionness hash does not match source")

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "grid_summary_json": str(_path(grid_summary_json)),
        "selection_policy": selection_policy,
        "validation_metric_selection": bool(validation_metric_selection),
        "metric": metric,
        "train_free_baseline": True,
        "not_main_method": True,
        "downstream_detector_ready": True,
        "subset": subset,
        "source_jsonl": str(source),
        "output_jsonl": str(output),
        "source_sha256": source_sha,
        "output_sha256": output_sha,
        "output_row_count": int(source_row_count),
        "selected_cell": {
            "provider": selected.get("provider"),
            "clip_frames": selected.get("clip_frames"),
            "frame_interval": selected.get("frame_interval"),
            "crop_size": selected.get("crop_size"),
            "batch_size": selected.get("batch_size"),
            "out_root": selected.get("out_root"),
            "row_count": selected.get("row_count"),
            "video_count": selected.get("video_count"),
            "uses_original_x3d_clip_window": selected.get("uses_original_x3d_clip_window"),
            "source_provenance": provenance,
        },
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a formal train-free X3D actionness JSONL for DUCA downstream runs.")
    parser.add_argument("--grid-summary-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--selection-policy", default="pre_registered", choices=("pre_registered", "best_by_metric"))
    parser.add_argument("--provider")
    parser.add_argument("--frame-interval", type=int)
    parser.add_argument("--clip-frames", type=int)
    parser.add_argument("--metric")
    parser.add_argument("--allow-validation-selection", action="store_true")
    parser.add_argument("--link-mode", default="copy", choices=("copy", "symlink", "hardlink"))
    args = parser.parse_args(argv)
    summary = materialize_actionness(
        grid_summary_json=args.grid_summary_json,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        provider=args.provider,
        frame_interval=args.frame_interval,
        clip_frames=args.clip_frames,
        selection_policy=args.selection_policy,
        metric=args.metric,
        allow_validation_selection=bool(args.allow_validation_selection),
        link_mode=args.link_mode,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
