from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    ARMS,
    PROTOCOL_ID,
    atomic_publish_json,
    canonical_sha256,
)


LEDGER_SCHEMA = "s2_v3_full_operator_c_exec_v1"
AUTOMATIC_OPERATOR_ALLOWLIST = {
    "aten.addmm",
    "aten.bmm",
    "aten.convolution",
    "aten.linear",
    "aten.matmul",
    "aten.mm",
}
MANUAL_OPERATOR_ALLOWLIST = {
    "bilinear_grid_sample_upper_bound",
    "bilinear_resize_upper_bound",
    "detector_proposal_arithmetic_upper_bound",
    "elementwise_upper_bound",
    "layer_norm_upper_bound",
    "mean_reduction_upper_bound",
    "normalization_upper_bound",
    "softmax_upper_bound",
    "temporal_linear_interpolation_upper_bound",
}


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"{field} must be {'non-negative' if allow_zero else 'positive'}")
    return parsed


def linear_fma2(*, batch: int, positions: int, inputs: int, outputs: int) -> int:
    return 2 * math.prod(
        _positive_int(value, field=name)
        for name, value in (
            ("batch", batch),
            ("positions", positions),
            ("inputs", inputs),
            ("outputs", outputs),
        )
    )


def convolution_fma2(
    *,
    batch: int,
    output_positions: int,
    output_channels: int,
    input_channels_per_group: int,
    kernel_elements: int,
) -> int:
    return 2 * math.prod(
        _positive_int(value, field=name)
        for name, value in (
            ("batch", batch),
            ("output_positions", output_positions),
            ("output_channels", output_channels),
            ("input_channels_per_group", input_channels_per_group),
            ("kernel_elements", kernel_elements),
        )
    )


def batched_matmul_fma2(*, batches: int, m: int, n: int, k: int) -> int:
    return 2 * math.prod(
        _positive_int(value, field=name)
        for name, value in (("batches", batches), ("m", m), ("n", n), ("k", k))
    )


def conservative_elementwise_upper_bound(*, elements: int, ops_per_element: int) -> int:
    return _positive_int(elements, field="elements") * _positive_int(
        ops_per_element, field="ops_per_element"
    )


@dataclass(frozen=True)
class OperatorEntry:
    event_id: str
    operator: str
    integer_operations: int
    source: str
    formula: str
    conservative_upper_bound: bool


class FullOperatorLedger:
    """A create-once integer ledger; unsupported or duplicate work fails closed."""

    def __init__(self, *, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported arm: {arm}")
        self.arm = arm
        self._entries: dict[str, OperatorEntry] = {}
        self._unsupported: set[str] = set()

    def add_automatic(
        self, *, event_id: str, operator: str, integer_operations: int
    ) -> None:
        if operator not in AUTOMATIC_OPERATOR_ALLOWLIST:
            raise ValueError(f"automatic operator has no frozen support: {operator}")
        self._add(
            OperatorEntry(
                event_id=str(event_id),
                operator=operator,
                integer_operations=_positive_int(
                    integer_operations, field="integer_operations", allow_zero=True
                ),
                source="automatic_execution_trace",
                formula="runtime_flop_counter_fma_equals_2",
                conservative_upper_bound=False,
            )
        )

    def add_manual(
        self,
        *,
        event_id: str,
        operator: str,
        integer_operations: int,
        formula: str,
        conservative_upper_bound: bool,
    ) -> None:
        if operator not in MANUAL_OPERATOR_ALLOWLIST:
            raise ValueError(f"manual operator has no frozen rule: {operator}")
        if not formula:
            raise ValueError("manual operator requires an explicit formula")
        self._add(
            OperatorEntry(
                event_id=str(event_id),
                operator=operator,
                integer_operations=_positive_int(
                    integer_operations, field="integer_operations", allow_zero=True
                ),
                source="manual_runtime_shape_rule",
                formula=str(formula),
                conservative_upper_bound=bool(conservative_upper_bound),
            )
        )

    def mark_unsupported(self, operator: str) -> None:
        self._unsupported.add(str(operator))

    def _add(self, entry: OperatorEntry) -> None:
        if not entry.event_id:
            raise ValueError("operator event_id cannot be empty")
        if entry.event_id in self._entries:
            raise ValueError(f"operator event was counted twice: {entry.event_id}")
        self._entries[entry.event_id] = entry

    def receipt(
        self,
        *,
        execution_identity: Mapping[str, Any],
        boundary_trace: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self._unsupported:
            raise RuntimeError(
                "FAILED_C_EXEC_INCOMPLETE unsupported="
                + ",".join(sorted(self._unsupported))
            )
        if not self._entries:
            raise RuntimeError("FAILED_C_EXEC_INCOMPLETE empty operator ledger")
        required_boundaries = {
            "start": "first_arm_dependent_decoded_rgb_transform",
            "end": "pre_nms_raw_detections",
            "nms_called": False,
            "evaluator_called": False,
        }
        for key, value in required_boundaries.items():
            if boundary_trace.get(key) != value:
                raise ValueError(f"C_exec boundary trace changed: {key}")
        rows = [asdict(self._entries[key]) for key in sorted(self._entries)]
        payload: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "arm": self.arm,
            "fused_multiply_add_operations": 2,
            "boundary_trace": dict(boundary_trace),
            "execution_identity": dict(execution_identity),
            "operator_rows": rows,
            "operator_row_count": len(rows),
            "full_operator_c_exec": sum(
                int(row["integer_operations"]) for row in rows
            ),
            "unsupported_operators": [],
            "complete": True,
            "admission_axes": ["detection_performance", "full_operator_c_exec"],
            "diagnostic_only_not_gated": [
                "latency",
                "throughput",
                "peak_allocated_memory",
                "peak_reserved_memory",
                "energy",
                "power",
            ],
        }
        payload["ledger_sha256"] = canonical_sha256(payload)
        return payload


def validate_c_exec_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    checked = dict(payload)
    digest = checked.pop("ledger_sha256", None)
    if not digest or canonical_sha256(checked) != digest:
        raise ValueError("C_exec receipt self-hash mismatch")
    checked["ledger_sha256"] = digest
    if (
        checked.get("schema_version") != LEDGER_SCHEMA
        or checked.get("protocol_id") != PROTOCOL_ID
        or checked.get("arm") not in ARMS
        or checked.get("complete") is not True
        or checked.get("unsupported_operators") != []
    ):
        raise ValueError("C_exec receipt identity or completeness is invalid")
    rows = checked.get("operator_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("C_exec receipt has no operator rows")
    event_ids = [str(row.get("event_id")) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("C_exec receipt double-counts an event")
    total = sum(_positive_int(row["integer_operations"], field="row count", allow_zero=True) for row in rows)
    if total <= 0:
        raise ValueError("C_exec receipt total must be positive")
    if total != int(checked.get("full_operator_c_exec", -1)):
        raise ValueError("C_exec receipt total differs from its operator ledger")
    return checked


def compare_c_exec_receipts(
    receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(receipts) != set(ARMS):
        raise ValueError("compute comparison requires D160, G96 and U128-A0")
    checked = {arm: validate_c_exec_receipt(receipts[arm]) for arm in ARMS}
    identity_keys = (
        "candidate_commit",
        "protocol_sha256",
        "evaluation_manifest_sha256",
        "checkpoint_policy",
        "dtype",
        "batch_size",
        "ordered_window_count",
    )
    for key in identity_keys:
        if len({json.dumps(row["execution_identity"].get(key), sort_keys=True) for row in checked.values()}) != 1:
            raise ValueError(f"C_exec receipts are not matched on {key}")
    counts = {arm: int(checked[arm]["full_operator_c_exec"]) for arm in ARMS}
    result = {
        "schema_version": "s2_v3_c_exec_comparison_v1",
        "protocol_id": PROTOCOL_ID,
        "counts": counts,
        "primary_exact_10u_le_9d": 10 * counts["U128-A0"] <= 9 * counts["D160"],
        "g96_not_more_than_u128_a0": counts["G96"] <= counts["U128-A0"],
        "ratio_disclosure": {
            "u128_a0_over_d160": counts["U128-A0"] / counts["D160"],
            "g96_over_u128_a0": counts["G96"] / counts["U128-A0"],
        },
        "gate_uses_latency_or_memory": False,
    }
    result["comparison_sha256"] = canonical_sha256(result)
    return result


def _load_ledger_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ledger = FullOperatorLedger(arm=str(payload["arm"]))
    for row in payload.get("automatic_events", ()):
        ledger.add_automatic(**row)
    for row in payload.get("manual_events", ()):
        ledger.add_manual(**row)
    for operator in payload.get("unsupported_operators", ()):
        ledger.mark_unsupported(str(operator))
    return ledger.receipt(
        execution_identity=payload["execution_identity"],
        boundary_trace=payload["boundary_trace"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal a task-local full-operator C_exec ledger")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = _load_ledger_input(args.input)
    atomic_publish_json(args.output, receipt)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullOperatorLedger",
    "OperatorEntry",
    "batched_matmul_fma2",
    "compare_c_exec_receipts",
    "conservative_elementwise_upper_bound",
    "convolution_fma2",
    "linear_fma2",
    "validate_c_exec_receipt",
]
