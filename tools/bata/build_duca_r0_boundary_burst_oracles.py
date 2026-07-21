from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tools.bata.diagnose_duca_allocation_family_ceiling import (
    allocation_metrics,
    axis_from_record,
)
from tools.bata.duca_allocation_families import (
    physical_gap_report,
    resolve_physical_cap,
    select_family_a,
)
from tools.bata.duca_exact_physical_solver import solve_boundary_burst_oracle
from tools.bata.export_duca_allocation_ceiling_inputs import (
    canonical_sha256,
    deduplicate_sliding_windows,
    sha256,
    write_json_exclusive,
)


SCHEMA = "duca_allocation_family_ceiling_record_v1"
SUMMARY_SCHEMA = "duca_r0_boundary_burst_oracle_summary_v1"
FAMILY_SPECS = (
    ("R2Q3_privileged_boundary_burst", 2, 3),
    ("R4Q5_privileged_boundary_burst", 4, 5),
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            recorded = row.get("record_sha256")
            unhashed = dict(row)
            unhashed.pop("record_sha256", None)
            if not isinstance(recorded, str) or canonical_sha256(unhashed) != recorded:
                raise ValueError(f"{path}:{line_number}: input record hash mismatch")
            rows.append(row)
    if not rows:
        raise ValueError("R0 input JSONL is empty")
    return rows


def _validity_lookup(config: str | Path) -> dict[str, dict[str, Any]]:
    from mmengine.config import Config
    from opentad.datasets import build_dataset

    cfg = Config.fromfile(str(Path(config).resolve()))
    dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    deduplicate_sliding_windows(dataset)
    lookup: dict[str, dict[str, Any]] = {}
    for item in dataset.data_list:
        if not isinstance(item, Sequence) or len(item) < 4:
            raise ValueError("R0 validity audit requires sliding-window entries")
        video_id, _, annotation, window = item[:4]
        start_frame = float(window[0])
        sample_id = f"{video_id}|{int(round(start_frame))}"
        if sample_id in lookup:
            raise ValueError(f"duplicate R0 validity identity: {sample_id}")
        segments = annotation.get("gt_segments", [])
        validity = annotation.get("gt_boundary_validity")
        if validity is None:
            raise ValueError(f"{sample_id}: sliding window omitted endpoint validity")
        local_segments = (
            (segments - start_frame - float(dataset.offset_frames))
            / float(dataset.snippet_stride)
        ).tolist()
        validity_rows = [[bool(value) for value in row] for row in validity.tolist()]
        if len(local_segments) != len(validity_rows) or any(
            len(row) != 2 for row in validity_rows
        ):
            raise ValueError(f"{sample_id}: endpoint validity shape mismatch")
        lookup[sample_id] = {
            "gt_segments": local_segments,
            "gt_boundary_validity": validity_rows,
        }
    return lookup


def _segments_close(left: Sequence[Any], right: Sequence[Any]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        len(a) == 2
        and len(b) == 2
        and all(math.isclose(float(x), float(y), abs_tol=1.0e-6) for x, y in zip(a, b))
        for a, b in zip(left, right)
    )


def _uniform_family(row: Mapping[str, Any], cap: Any) -> dict[str, Any]:
    axis = axis_from_record(row)
    selected = select_family_a(
        axis,
        requested_budget=int(row["requested_budget"]),
        cap=cap,
    )
    payload = selected.to_dict()
    payload["family_key"] = "A_exact_uniform"
    payload["allocation_metrics"] = allocation_metrics(
        selected.positions,
        row.get("gt_segments", []),
        valid_len=axis.valid_len,
        radii=(0, 1, 2, 4),
        short_action_max_length=16.0,
    )
    payload["allocation_metrics"]["uniform_overlap"] = 1.0
    payload["r0_contract"] = {
        "exact_k": len(selected.positions)
        == min(axis.valid_len, int(row["requested_budget"])),
        "max_unselected_hole": selected.gap_report.dense_max_unselected_hole,
        "privileged": False,
        "diagnostic_only": True,
    }
    return payload


def _burst_family(
    row: Mapping[str, Any],
    validity: Sequence[Sequence[bool]],
    cap: Any,
    *,
    family_key: str,
    radius: int,
    quota: int,
    max_unselected_hole: int,
) -> dict[str, Any]:
    axis = axis_from_record(row)
    solved = solve_boundary_burst_oracle(
        axis,
        row.get("gt_segments", []),
        validity,
        requested_budget=int(row["requested_budget"]),
        cap=cap,
        radius=radius,
        quota=quota,
        max_unselected_hole=max_unselected_hole,
    )
    report = physical_gap_report(axis, solved.positions)
    uniform = set(_uniform_family(row, cap)["positions"])
    metrics = allocation_metrics(
        solved.positions,
        row.get("gt_segments", []),
        valid_len=axis.valid_len,
        radii=(0, 1, 2, 4),
        short_action_max_length=16.0,
    )
    metrics["uniform_overlap"] = len(set(solved.positions) & uniform) / len(uniform)
    return {
        "family": family_key,
        "family_key": family_key,
        "positions": solved.positions,
        "budget": len(solved.positions),
        "score_sum": None,
        "exact": True,
        "deployable": False,
        "privileged": True,
        "solver_status": "OPTIMAL",
        "physical_cap_compliant": True,
        "gap_report": report.to_dict(),
        "scaffold_positions": (),
        "residual_positions": (),
        "allocation_metrics": metrics,
        "gt_solver": solved.to_dict(),
        "r0_contract": {
            "exact_k": len(solved.positions) == min(axis.valid_len, int(row["requested_budget"])),
            "max_unselected_hole": report.dense_max_unselected_hole,
            "max_unselected_hole_limit": max_unselected_hole,
            "all_endpoint_quotas_pass": all(
                bool(item["quota_pass"]) for item in solved.endpoint_contracts
            ),
            "all_applicable_bilateral_pass": all(
                bool(item["bilateral_pass"]) for item in solved.endpoint_contracts
            ),
            "residual_fill_count": solved.residual_fill_count,
            "background_selected_count": solved.background_selected_count,
            "background_component_count": solved.background_component_count,
            "invalid_crop_endpoint_count": solved.invalid_endpoint_count,
            "diagnostic_only": True,
        },
    }


def build_oracles(
    *,
    input_jsonl: str | Path,
    config: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    max_unselected_hole: int = 2,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    validity_by_sample = _validity_lookup(config)
    output_path = Path(output_jsonl)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite R0 artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    family_counts = {"A_exact_uniform": 0, **{name: 0 for name, _, _ in FAMILY_SPECS}}
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                sample_id = str(row["sample_id"])
                bound = validity_by_sample.get(sample_id)
                if bound is None:
                    raise ValueError(f"{sample_id}: no sealed sliding-window validity row")
                if not _segments_close(row.get("gt_segments", []), bound["gt_segments"]):
                    raise ValueError(f"{sample_id}: exported GT differs from validity-bound GT")
                axis = axis_from_record(row)
                source_steps = [
                    right - left
                    for left, right in zip(axis.source_frames, axis.source_frames[1:])
                ]
                step = max(source_steps, default=0.0)
                cap = resolve_physical_cap(
                    axis,
                    requested_budget=int(row["requested_budget"]),
                    policy="explicit_frames",
                    value=(int(max_unselected_hole) + 1) * step,
                )
                families = [_uniform_family(row, cap)]
                for family_key, radius, quota in FAMILY_SPECS:
                    families.append(
                        _burst_family(
                            row,
                            bound["gt_boundary_validity"],
                            cap,
                            family_key=family_key,
                            radius=radius,
                            quota=quota,
                            max_unselected_hole=max_unselected_hole,
                        )
                    )
                for family in families:
                    contract = family["r0_contract"]
                    if contract["exact_k"] is not True:
                        raise ValueError(f"{sample_id}: {family['family_key']} violates exact-K")
                    if int(contract["max_unselected_hole"]) > int(max_unselected_hole):
                        raise ValueError(f"{sample_id}: {family['family_key']} violates G")
                    if family["privileged"] and (
                        contract["all_endpoint_quotas_pass"] is not True
                        or contract["all_applicable_bilateral_pass"] is not True
                    ):
                        raise ValueError(f"{sample_id}: burst endpoint contract failed")
                    family_counts[family["family_key"]] += 1
                output = {
                    "schema_version": SCHEMA,
                    "sample_id": sample_id,
                    "video_id": row["video_id"],
                    "split": row["split"],
                    "valid_len": row["valid_len"],
                    "requested_budget": row["requested_budget"],
                    "score_key": "privileged_boundary_burst_oracle",
                    "cap": cap.to_dict(),
                    "coarse_signal_metrics": {},
                    "families": families,
                    "input_record_sha256": row["record_sha256"],
                    "contract": {
                        "offline_full_window": True,
                        "runtime_gt_input_to_replay": False,
                        "gt_families_privileged_only": True,
                        "crop_cut_endpoints_excluded": True,
                        "per_sample_exact_k_g_bilateral_quota_validated": True,
                        "detector_mAP_absolute_diagnostic_only": True,
                    },
                }
                output["record_sha256"] = canonical_sha256(output)
                handle.write(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "ok": all(count == len(rows) for count in family_counts.values()),
        "sample_count": len(rows),
        "families": family_counts,
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "input_jsonl_sha256": sha256(input_jsonl),
        "output_jsonl": str(output_path.resolve()),
        "output_jsonl_sha256": sha256(output_path),
        "max_unselected_hole": int(max_unselected_hole),
        "crop_cut_endpoints_excluded": True,
        "diagnostic_only": True,
        "absolute_map_paper_claim_allowed": False,
    }
    write_json_exclusive(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed R0 boundary-burst Oracles")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-unselected-hole", type=int, default=2)
    args = parser.parse_args(argv)
    summary = build_oracles(
        input_jsonl=args.input_jsonl,
        config=args.config,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        max_unselected_hole=args.max_unselected_hole,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
