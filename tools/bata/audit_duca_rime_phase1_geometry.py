from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import torch

from opentad.models.detectors.single_stage import SingleStageDetector
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.models.utils.truetime_geometry import (
    SELECTED_AXIS,
    TRUE_TIME_AXIS,
    TrueTimeMap,
)


SCHEMA = "duca_rime_phase1_geometry_audit_v1"
ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _source_binding(relative: str) -> dict[str, str]:
    path = (ROOT / relative).resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def audit_geometry(
    *,
    expected_commit: str,
    split_assignment_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("Phase-1 geometry audit requires an exact Git commit")
    if re.fullmatch(r"[0-9a-f]{64}", str(split_assignment_sha256)) is None:
        raise ValueError("Phase-1 geometry audit requires the split assignment hash")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase-1 geometry audit must run inside Slurm")
    if _git("rev-parse", "HEAD") != str(expected_commit):
        raise RuntimeError("Phase-1 geometry audit commit drift")
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Phase-1 geometry audit requires a clean tree")

    selected_positions = torch.tensor(
        [0.0, 3.0, 8.0, 15.0],
        dtype=torch.float32,
    )
    time_map = TrueTimeMap(
        selected_positions=selected_positions,
        dense_len=16,
        valid_len=16,
    )
    selected_segments = torch.tensor(
        [[0.5, 2.5], [1.0, 3.0]],
        dtype=torch.float32,
    )
    expected_true_segments = time_map.selected_to_true(selected_segments)
    roundtrip = time_map.true_to_selected(expected_true_segments)
    roundtrip_error = float(
        (roundtrip - selected_segments).abs().max().item()
    )

    metadata = {
        "video_name": "duca_rime_phase1_geometry_fixture",
        "detector_prediction_inverse_map_required": True,
        "detector_output_coordinate_space": SELECTED_AXIS,
        "selected_axis_to_true_time_dense_index": [
            int(value) for value in selected_positions.tolist()
        ],
        "irregular_selected_count": 4,
        "irregular_selected_valid_len": 4,
        "irregular_dense_valid_len": 16,
        "truetime_dense_len": 16,
        "window_size": 16,
    }
    captured: dict[str, Any] = {"nms_calls": 0}

    def fake_nms(segments, scores, labels, **_kwargs):
        captured["nms_calls"] += 1
        captured["nms_segments"] = segments.detach().clone()
        return segments, scores, labels

    def fake_convert_to_seconds(segments, meta):
        captured["post_nms_coordinate_space"] = meta[
            "detector_output_coordinate_space"
        ]
        return segments

    def external_classifier(_video_id, segments, scores):
        return segments, ["action"] * int(segments.shape[0]), scores

    detector = SingleStageDetector()
    post_cfg = SimpleNamespace(
        pre_nms_thresh=0.0,
        pre_nms_topk=100,
        sliding_window=False,
        nms={},
    )
    predictions = (
        [selected_segments],
        [torch.tensor([[0.9], [0.8]], dtype=torch.float32)],
    )
    with patch(
        "opentad.models.detectors.single_stage.batched_nms",
        side_effect=fake_nms,
    ), patch(
        "opentad.models.detectors.single_stage.convert_to_seconds",
        side_effect=fake_convert_to_seconds,
    ):
        detector.post_processing(
            predictions,
            [metadata],
            post_cfg,
            external_classifier,
        )

    nms_segments = captured.get("nms_segments")
    remap_error = (
        float("inf")
        if not torch.is_tensor(nms_segments)
        else float((nms_segments - expected_true_segments).abs().max().item())
    )
    remap_before_nms = (
        int(captured["nms_calls"]) == 1
        and remap_error <= 1.0e-6
        and captured.get("post_nms_coordinate_space") == TRUE_TIME_AXIS
    )

    already_physical_meta = dict(metadata)
    already_physical_meta["detector_output_coordinate_space"] = TRUE_TIME_AXIS
    physical_segments, physical_meta = (
        detector._remap_selector_segments_for_post_processing(
            expected_true_segments,
            already_physical_meta,
        )
    )
    physical_passthrough_error = float(
        (physical_segments - expected_true_segments).abs().max().item()
    )

    gap_rows = []
    max_gap_violation_count = 0
    for budget in (192, 384):
        positions = exact_uniform_positions(768, budget).to(torch.float32)
        observed = float((positions[1:] - positions[:-1]).max().item())
        reference_cap = float(math.ceil((768 - 1) / (budget - 1)))
        violated = observed > reference_cap + 1.0e-8
        endpoints_and_cardinality_valid = (
            int(positions.numel()) == budget
            and int(positions[0].item()) == 0
            and int(positions[-1].item()) == 767
            and int(torch.unique(positions).numel()) == budget
        )
        violated = violated or not endpoints_and_cardinality_valid
        max_gap_violation_count += int(violated)
        gap_rows.append(
            {
                "dense_len": 768,
                "budget": budget,
                "first": int(positions[0].item()),
                "last": int(positions[-1].item()),
                "observed_max_gap_dense_indices": observed,
                "exact_uniform_reference_cap_dense_indices": reference_cap,
                "endpoints_and_cardinality_valid": (
                    endpoints_and_cardinality_valid
                ),
                "violation": violated,
            }
        )

    if (
        not remap_before_nms
        or roundtrip_error > 1.0e-6
        or physical_passthrough_error > 1.0e-6
        or max_gap_violation_count != 0
    ):
        raise RuntimeError("DUCA-RIME Phase-1 geometry audit failed")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "passed",
        "gate_pass": True,
        "git_commit": str(expected_commit),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "split_assignment_sha256": str(split_assignment_sha256),
        "uses_official_final": False,
        "checks": {
            "remap_before_official_nms": True,
            "official_nms_call_count": int(captured["nms_calls"]),
            "pre_nms_remap_max_abs": remap_error,
            "coordinate_roundtrip_max_abs": roundtrip_error,
            "roundtrip_violation_count": 0,
            "physical_head_passthrough_max_abs": physical_passthrough_error,
            "physical_head_output_remapped_twice": False,
            "max_gap_violation_count": max_gap_violation_count,
        },
        "gap_audit": gap_rows,
        "source_artifacts": {
            "single_stage_detector": _source_binding(
                "opentad/models/detectors/single_stage.py"
            ),
            "true_time_geometry": _source_binding(
                "opentad/models/utils/truetime_geometry.py"
            ),
            "official_nms": _source_binding(
                "opentad/models/utils/post_processing/nms/nms.py"
            ),
        },
        "claim_scope": "coordinate_order_and_exact_uniform_gap_infrastructure_only",
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    target = Path(output).expanduser().resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite a different Phase-1 geometry audit: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "payload": payload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit DUCA-RIME q-to-true-time ordering before official NMS."
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-assignment-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_geometry(
        expected_commit=args.expected_commit,
        split_assignment_sha256=args.split_assignment_sha256,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
