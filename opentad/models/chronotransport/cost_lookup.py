from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CostLookupKey:
    hardware: str
    precision: str
    batch_size: int
    candidate_schedule: str
    selected_rows_per_group: tuple[int, ...]
    gpu_uuid: str = ""
    driver: str = ""
    cuda: str = ""
    pytorch: str = ""
    source_commit: str = ""
    spec_sha256: str = ""
    config_sha256: str = ""
    checkpoint_sha256: str = ""
    library_sha256: str = ""
    environment_sha256: str = ""
    registration_sha256: str = ""
    factory_config_sha256: str = ""
    requested_action_sha256: str = ""
    executed_action_sha256: str = ""

    def encode(self) -> str:
        rows = ",".join(str(value) for value in self.selected_rows_per_group)
        formal_fields = (
            self.gpu_uuid,
            self.driver,
            self.cuda,
            self.pytorch,
            self.source_commit,
            self.spec_sha256,
            self.config_sha256,
            self.checkpoint_sha256,
            self.library_sha256,
            self.environment_sha256,
            self.registration_sha256,
            self.factory_config_sha256,
            self.requested_action_sha256,
            self.executed_action_sha256,
        )
        if not any(formal_fields):
            return (
                f"{self.hardware}|{self.precision}|{self.batch_size}|"
                f"{self.candidate_schedule}|{rows}"
            )
        fields = (
            self.hardware,
            self.gpu_uuid,
            self.driver,
            self.cuda,
            self.pytorch,
            self.precision,
            str(self.batch_size),
            self.source_commit,
            self.spec_sha256,
            self.config_sha256,
            self.checkpoint_sha256,
            self.library_sha256,
            self.environment_sha256,
            self.registration_sha256,
            self.factory_config_sha256,
            self.candidate_schedule,
            self.requested_action_sha256,
            self.executed_action_sha256,
            rows,
        )
        return "|".join(fields)

    @classmethod
    def from_provenance(cls, provenance: Mapping[str, object]) -> "CostLookupKey":
        required = {
            "gpu_model",
            "gpu_uuid",
            "driver",
            "cuda",
            "pytorch",
            "precision",
            "batch_size",
            "source_commit",
            "spec_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "library_sha256",
            "environment_sha256",
            "registration_sha256",
            "factory_config_sha256",
            "candidate_name",
            "requested_action_sha256",
            "executed_action_sha256",
            "selected_rows_per_group",
        }
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"formal provenance is missing fields: {missing}")
        for field in (
            "gpu_model",
            "gpu_uuid",
            "driver",
            "cuda",
            "pytorch",
            "precision",
            "source_commit",
            "spec_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "library_sha256",
            "environment_sha256",
            "registration_sha256",
            "factory_config_sha256",
            "candidate_name",
            "requested_action_sha256",
            "executed_action_sha256",
        ):
            if not isinstance(provenance[field], str):
                raise TypeError(f"formal provenance {field} must be a string")
        if type(provenance["batch_size"]) is not int:
            raise TypeError("formal provenance batch_size must be an integer")
        rows = provenance["selected_rows_per_group"]
        if (
            not isinstance(rows, (list, tuple))
            or any(type(value) is not int for value in rows)
        ):
            raise TypeError("formal provenance selected_rows_per_group must contain integers")
        return cls(
            hardware=provenance["gpu_model"],
            gpu_uuid=provenance["gpu_uuid"],
            driver=provenance["driver"],
            cuda=provenance["cuda"],
            pytorch=provenance["pytorch"],
            precision=provenance["precision"],
            batch_size=provenance["batch_size"],
            source_commit=provenance["source_commit"],
            spec_sha256=provenance["spec_sha256"],
            config_sha256=provenance["config_sha256"],
            checkpoint_sha256=provenance["checkpoint_sha256"],
            library_sha256=provenance["library_sha256"],
            environment_sha256=provenance["environment_sha256"],
            registration_sha256=provenance["registration_sha256"],
            factory_config_sha256=provenance["factory_config_sha256"],
            candidate_schedule=provenance["candidate_name"],
            requested_action_sha256=provenance["requested_action_sha256"],
            executed_action_sha256=provenance["executed_action_sha256"],
            selected_rows_per_group=tuple(rows),
        )

    def validate_formal(self) -> None:
        nonempty = {
            "hardware": self.hardware,
            "gpu_uuid": self.gpu_uuid,
            "driver": self.driver,
            "cuda": self.cuda,
            "pytorch": self.pytorch,
            "precision": self.precision,
            "candidate_schedule": self.candidate_schedule,
        }
        for name, value in nonempty.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"formal cost key requires non-empty {name}")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("formal cost key requires batch_size=1")
        if self.precision != "amp_fp16":
            raise ValueError("formal cost key requires precision=amp_fp16")
        if len(self.selected_rows_per_group) != 3 or any(
            type(v) is not int or v < 0 or v > 48 for v in self.selected_rows_per_group
        ):
            raise ValueError("formal cost key requires three selected_rows_per_group in [0,48]")
        if not _GIT_COMMIT.fullmatch(self.source_commit):
            raise ValueError("formal cost key requires 40-hex source_commit")
        for name in (
            "spec_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "library_sha256",
            "environment_sha256",
            "registration_sha256",
            "factory_config_sha256",
            "requested_action_sha256",
            "executed_action_sha256",
        ):
            value = getattr(self, name)
            if not _SHA256.fullmatch(value):
                raise ValueError(f"formal cost key requires lowercase SHA-256 {name}")


class ScheduleCostLookup:
    # Keep the on-disk discriminator compatible with the Stage-A loader;
    # formal r2 completeness is enforced by CostLookupKey.validate_formal().
    schema_version = "chronotransport_schedule_cost_v1"

    def __init__(self, entries: Mapping[str, Mapping[str, float]]) -> None:
        self.entries = {str(key): dict(value) for key, value in entries.items()}
        for value in self.entries.values():
            for statistic in ("p50", "p95"):
                number = float(value[statistic])
                if not math.isfinite(number) or number < 0:
                    raise ValueError("p50/p95 costs must be finite and non-negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "ScheduleCostLookup":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported schedule cost schema")
        return cls(payload["entries"])

    def get(self, key: CostLookupKey, statistic: str = "p50") -> float:
        if statistic not in {"p50", "p95"}:
            raise ValueError("statistic must be p50 or p95")
        encoded = key.encode()
        if encoded not in self.entries:
            raise KeyError(f"missing measured schedule cost: {encoded}")
        return float(self.entries[encoded][statistic])

    @classmethod
    def payload(cls, entries: Sequence[tuple[CostLookupKey, float, float]]) -> dict:
        return {
            "schema_version": cls.schema_version,
            "entries": {
                key.encode(): {"p50": float(p50), "p95": float(p95)}
                for key, p50, p95 in entries
            },
        }


def _finite_cost(value: object, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a measured number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def build_execution_cost_ledger_entry(
    *,
    requested_schedule_name: str,
    requested_action_sha256: str,
    requested_cost_p50: float,
    executed_schedule_name: str,
    executed_action_sha256: str,
    repair_count: int,
    nan_fallback: bool,
    whole_window_dense_fallback: bool,
    executed_lookup_cost_p50: float | None,
    actual_total_ms: float,
    safety_override_budget_violation: bool = False,
) -> dict[str, object]:
    for field, value in (
        ("requested_schedule_name", requested_schedule_name),
        ("executed_schedule_name", executed_schedule_name),
        ("requested_action_sha256", requested_action_sha256),
        ("executed_action_sha256", executed_action_sha256),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
    if type(repair_count) is not int:
        raise TypeError("repair_count must be an integer")
    for field, value in (
        ("nan_fallback", nan_fallback),
        ("whole_window_dense_fallback", whole_window_dense_fallback),
        ("safety_override_budget_violation", safety_override_budget_violation),
    ):
        if type(value) is not bool:
            raise TypeError(f"{field} must be boolean")
    _finite_cost(requested_cost_p50, "requested_cost_p50")
    _finite_cost(executed_lookup_cost_p50, "executed_lookup_cost_p50", nullable=True)
    _finite_cost(actual_total_ms, "actual_total_ms")
    changed = requested_action_sha256 != executed_action_sha256
    schedule_changed = requested_schedule_name != executed_schedule_name
    evidence_valid = (
        not changed
        and not schedule_changed
        and repair_count == 0
        and not nan_fallback
        and not whole_window_dense_fallback
        and not safety_override_budget_violation
    )
    entry: dict[str, object] = {
        "schema": "chronotransport-r2-execution-cost-ledger-v1",
        "requested_schedule_name": requested_schedule_name,
        "requested_action_sha256": requested_action_sha256,
        "requested_cost_p50": requested_cost_p50,
        "executed_schedule_name": executed_schedule_name,
        "executed_action_sha256": executed_action_sha256,
        "repair_count": repair_count,
        "nan_fallback": nan_fallback,
        "whole_window_dense_fallback": whole_window_dense_fallback,
        "executed_lookup_cost_p50": executed_lookup_cost_p50,
        "actual_total_ms": actual_total_ms,
        "safety_override_budget_violation": safety_override_budget_violation,
        "evidence_valid": evidence_valid,
    }
    validate_execution_cost_ledger_entry(entry, formal=False)
    return entry


def validate_execution_cost_ledger_entry(
    entry: Mapping[str, object], *, formal: bool
) -> None:
    expected_fields = {
        "schema",
        "requested_schedule_name",
        "requested_action_sha256",
        "requested_cost_p50",
        "executed_schedule_name",
        "executed_action_sha256",
        "repair_count",
        "nan_fallback",
        "whole_window_dense_fallback",
        "executed_lookup_cost_p50",
        "actual_total_ms",
        "safety_override_budget_violation",
        "evidence_valid",
    }
    if set(entry) != expected_fields:
        raise ValueError("execution cost ledger fields mismatch")
    if entry.get("schema") != "chronotransport-r2-execution-cost-ledger-v1":
        raise ValueError("unsupported execution cost ledger schema")
    for field in ("requested_schedule_name", "executed_schedule_name"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise ValueError(f"cost ledger requires {field}")
    for field in ("requested_action_sha256", "executed_action_sha256"):
        if not isinstance(entry.get(field), str) or not _SHA256.fullmatch(str(entry[field])):
            raise ValueError(f"cost ledger requires lowercase SHA-256 {field}")
    _finite_cost(entry.get("requested_cost_p50"), "requested_cost_p50")
    _finite_cost(entry.get("executed_lookup_cost_p50"), "executed_lookup_cost_p50", nullable=True)
    _finite_cost(entry.get("actual_total_ms"), "actual_total_ms")
    repair_count = entry.get("repair_count")
    if isinstance(repair_count, bool) or not isinstance(repair_count, int) or repair_count < 0:
        raise ValueError("repair_count must be a non-negative integer")
    for field in (
        "nan_fallback",
        "whole_window_dense_fallback",
        "safety_override_budget_violation",
        "evidence_valid",
    ):
        if not isinstance(entry.get(field), bool):
            raise ValueError(f"cost ledger {field} must be boolean")
    action_changed = entry["requested_action_sha256"] != entry["executed_action_sha256"]
    schedule_changed = entry["requested_schedule_name"] != entry["executed_schedule_name"]
    derived_valid = (
        not action_changed
        and not schedule_changed
        and repair_count == 0
        and not entry["nan_fallback"]
        and not entry["whole_window_dense_fallback"]
        and not entry["safety_override_budget_violation"]
    )
    if formal:
        if entry["requested_schedule_name"] != entry["executed_schedule_name"]:
            raise ValueError("formal evidence forbids requested/executed schedule identity changes")
        if action_changed:
            raise ValueError("formal evidence forbids requested/executed action hash changes")
        if repair_count:
            raise ValueError("formal evidence forbids runtime repair")
        if entry["nan_fallback"]:
            raise ValueError("formal evidence forbids NaN fallback")
        if entry["whole_window_dense_fallback"]:
            raise ValueError("formal evidence forbids whole-window dense fallback")
        if entry["safety_override_budget_violation"]:
            raise ValueError("formal evidence forbids safety override budget violation")
        if entry["executed_lookup_cost_p50"] is None:
            raise ValueError("formal evidence requires exact executed lookup cost")
    if entry["evidence_valid"] != derived_valid:
        raise ValueError("cost ledger evidence_valid is inconsistent with execution evidence")
