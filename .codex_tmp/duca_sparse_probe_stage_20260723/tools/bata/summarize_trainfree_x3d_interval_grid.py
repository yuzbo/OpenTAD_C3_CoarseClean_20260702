from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_SCHEMA_VERSION = "trainfree_x3d_interval_grid_summary_v1"
READY = "TRAINFREE_X3D_INTERVAL_GRID_SUMMARY_READY"

ORIGINAL_X3D_CLIP_FRAMES = {
    "x3d_xs": 4,
    "efficient_x3d_xs": 4,
    "x3d_s": 13,
    "efficient_x3d_s": 13,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "provider",
        "clip_frames",
        "frame_interval",
        "crop_size",
        "batch_size",
        "uses_original_x3d_clip_window",
        "row_count",
        "video_count",
        "coarse_auroc",
        "coarse_auprc",
        "coarse_recall_at_budget",
        "selection_manual_mean_action_touched_recall",
        "selection_manual_mean_boundary_radius_recall",
        "selection_manual_mean_short_action_recall",
        "selection_manual_mean_p95_hole",
        "selection_manual_mean_uniform_similarity",
        "out_root",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _row_from_cell(cell: Mapping[str, str], *, subset: str) -> dict[str, Any]:
    provider = str(cell["provider"])
    out_root = Path(cell["out_root"]).expanduser()
    actionness_summary = _read_json(out_root / f"{provider}_{subset}_actionness.summary.json")
    coarse_summary = _read_json(out_root / f"{provider}_{subset}_coarse_eval.summary.json")
    selection_summary = _read_json(out_root / f"{provider}_{subset}_selection.summary.json")
    manual_selection = _nested(selection_summary, "baseline_summaries", "manual") or {}
    budget = int(_nested(selection_summary, "budget") or 384)
    recall_at_budget = _nested(coarse_summary, "metrics", "recall_at_k", str(budget))
    clip_frames = int(cell["clip_frames"])
    expected_clip_frames = ORIGINAL_X3D_CLIP_FRAMES.get(provider)
    return {
        "provider": provider,
        "clip_frames": clip_frames,
        "frame_interval": int(cell["frame_interval"]),
        "crop_size": int(cell["crop_size"]),
        "batch_size": int(cell["batch_size"]),
        "out_root": str(out_root),
        "status": str(cell.get("status", "")),
        "expected_original_x3d_clip_frames": expected_clip_frames,
        "uses_original_x3d_clip_window": expected_clip_frames is not None and clip_frames == expected_clip_frames,
        "row_count": int(actionness_summary.get("row_count", 0)),
        "video_count": int(actionness_summary.get("video_count", 0)),
        "source_provenance": actionness_summary.get("source_provenance", {}),
        "coarse_auroc": _float_or_none(_nested(coarse_summary, "metrics", "auroc")),
        "coarse_auprc": _float_or_none(_nested(coarse_summary, "metrics", "auprc")),
        "coarse_recall_at_budget": _float_or_none(recall_at_budget),
        "selection_manual_mean_action_touched_recall": _float_or_none(
            _nested(manual_selection, "mean_action_touched_recall")
        ),
        "selection_manual_mean_boundary_radius_recall": _float_or_none(
            _nested(manual_selection, "mean_boundary_radius_recall")
        ),
        "selection_manual_mean_short_action_recall": _float_or_none(
            _nested(manual_selection, "mean_short_action_recall")
        ),
        "selection_manual_mean_p95_hole": _float_or_none(_nested(manual_selection, "mean_p95_hole")),
        "selection_manual_mean_uniform_similarity": _float_or_none(
            _nested(manual_selection, "mean_uniform_similarity")
        ),
    }


def summarize_grid(
    *,
    manifest_tsv: str | Path,
    summary_json: str | Path | None = None,
    summary_tsv: str | Path | None = None,
    subset: str = "validation",
) -> dict[str, Any]:
    manifest_rows = _read_manifest(manifest_tsv)
    rows = [_row_from_cell(row, subset=subset) for row in manifest_rows if row.get("status") == "complete"]
    bad_clip = [
        row
        for row in rows
        if row.get("expected_original_x3d_clip_frames") is not None and not row.get("uses_original_x3d_clip_window")
    ]
    if bad_clip:
        provider = bad_clip[0]["provider"]
        raise ValueError(f"{provider}: clip_frames does not match original X3D window")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "manifest_tsv": str(manifest_tsv),
        "subset": subset,
        "row_count": len(rows),
        "providers": sorted({str(row["provider"]) for row in rows}),
        "frame_intervals": sorted({int(row["frame_interval"]) for row in rows}),
        "original_x3d_clip_frames": dict(ORIGINAL_X3D_CLIP_FRAMES),
        "rows": rows,
        "not_a_detector_mAP_result": True,
        "purpose": "train_free_x3d_coarse_actionness_and_indirect_selection_feasibility",
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    if summary_tsv is not None:
        _write_tsv(summary_tsv, rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize train-free X3D interval actionness/selection grid results.")
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-tsv")
    parser.add_argument("--subset", default="validation")
    args = parser.parse_args(argv)
    summary = summarize_grid(
        manifest_tsv=args.manifest_tsv,
        summary_json=args.summary_json,
        summary_tsv=args.summary_tsv,
        subset=args.subset,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
