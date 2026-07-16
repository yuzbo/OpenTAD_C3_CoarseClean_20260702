"""Closed repository-owned construction boundary for formal r2 profiling."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import torch


REGISTERED_BACKEND_IDENTITY = (
    "tools.bata.chronotransport_r2_opentad_profile_backend:"
    "OpenTADRegisteredProfileBackend"
)


class RegisteredOpenTADProfileSession:
    """Own and execute the one exact OpenTAD backend and registered dataset."""

    def __init__(self, registration: Mapping[str, object]) -> None:
        from tools.bata.chronotransport_r2_opentad_profile_backend import (
            OpenTADRegisteredProfileBackend,
        )

        self._registration = registration
        self._backend = OpenTADRegisteredProfileBackend(registration)
        if type(self._backend) is not OpenTADRegisteredProfileBackend:
            raise TypeError("formal profile requires the fixed repository-owned backend")

    def _profile_registered_candidate(
        self, candidate_plan: Mapping[str, object]
    ) -> dict[str, object]:
        from opentad.models.chronotransport.cost_lookup import (
            CostLookupKey,
            build_execution_cost_ledger_entry,
        )
        from opentad.models.chronotransport.full_stack_profiler import (
            _FORMAL_CANDIDATE_SCHEMA,
            _distribution,
            _percentile,
            _registered_candidate_provenance_from_validated,
            _validate_fixed_backend_result,
        )
        from opentad.models.chronotransport.profiler import REQUIRED_STAGE_FIELDS
        from opentad.models.chronotransport.protocol import canonical_sha256
        from opentad.models.chronotransport.registration import (
            REGISTERED_PROFILE_BACKEND_SOURCE,
        )

        if not isinstance(candidate_plan, Mapping):
            raise TypeError("registered candidate plan must be a mapping")
        if candidate_plan.get("factory_identity") != (
            "tools.bata.chronotransport_r2_profile_factory:"
            "build_registered_profile_session"
        ):
            raise ValueError("candidate plan does not bind the closed profile session")
        candidate_name = candidate_plan.get("candidate_name")
        invocation_ids = self._registration["profiler"]["invocation_ids"]
        action_hashes = candidate_plan["requested_action_sha256_by_invocation"]
        for invocation_id in invocation_ids[:50]:
            self._backend.invoke_registered_window(
                window_id=invocation_id,
                candidate_name=candidate_name,
            )

        pending = []
        for index, (invocation_id, action_sha) in enumerate(
            zip(invocation_ids, action_hashes)
        ):
            if not torch.cuda.is_available():
                raise RuntimeError("formal full-stack profiling requires CUDA")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            result = _validate_fixed_backend_result(
                self._backend.invoke_registered_window(
                    window_id=invocation_id,
                    candidate_name=candidate_name,
                ),
                candidate_name=candidate_name,
                expected_action_sha256=action_sha,
                expected_backend_source_sha256=self._registration["source_files"][
                    REGISTERED_PROFILE_BACKEND_SOURCE
                ],
            )
            torch.cuda.synchronize()
            total_ms = (perf_counter() - start) * 1000.0
            pending.append(
                {
                    "invocation_index": index,
                    "invocation_id": invocation_id,
                    "total_ms": total_ms,
                    "diagnostic_ms": dict(result["diagnostic_ms"]),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "action_sha256": action_sha,
                    "execution_provenance": dict(result["execution_provenance"]),
                }
            )

        measured_p50 = _percentile([row["total_ms"] for row in pending], 0.50)
        rows = []
        for pending_row in pending:
            row = dict(pending_row)
            action_sha = row.pop("action_sha256")
            row["cost_ledger"] = build_execution_cost_ledger_entry(
                requested_schedule_name=candidate_name,
                requested_action_sha256=action_sha,
                requested_cost_p50=measured_p50,
                executed_schedule_name=candidate_name,
                executed_action_sha256=action_sha,
                repair_count=0,
                nan_fallback=False,
                whole_window_dense_fallback=False,
                executed_lookup_cost_p50=measured_p50,
                actual_total_ms=row["total_ms"],
                safety_override_budget_violation=False,
            )
            rows.append(row)

        provenance = _registered_candidate_provenance_from_validated(
            self._registration,
            candidate_name,
            observed_environment=self._backend.observed_environment,
        )
        total_values = [float(row["total_ms"]) for row in rows]
        body: dict[str, object] = {
            "schema": _FORMAL_CANDIDATE_SCHEMA,
            "provenance": provenance,
            "cost_lookup_key": CostLookupKey.from_provenance(provenance).encode(),
            "warmup_count": 50,
            "sample_count": len(rows),
            "invocation_ids": list(invocation_ids),
            "invocation_order_sha256": canonical_sha256(invocation_ids),
            "requested_action_order_sha256": canonical_sha256(action_hashes),
            "executed_action_order_sha256": canonical_sha256(
                [row["cost_ledger"]["executed_action_sha256"] for row in rows]
            ),
            "execution_provenance_order_sha256": canonical_sha256(
                [row["execution_provenance"] for row in rows]
            ),
            "total_ms": _distribution(total_values),
            "diagnostic_ms": {
                name: _distribution(
                    [float(row["diagnostic_ms"][name]) for row in rows]
                )
                for name in REQUIRED_STAGE_FIELDS
            },
            "peak_gpu_memory_bytes": max(
                row["peak_gpu_memory_bytes"] for row in rows
            ),
            "invocations": rows,
        }
        body["candidate_profile_sha256"] = canonical_sha256(body)
        return body

    def run_fixed_profile(self) -> dict[str, object]:
        """Execute and serialize the exact 23-candidate formal profile."""

        from tools.bata.chronotransport_r2_opentad_profile_backend import (
            OpenTADRegisteredProfileBackend,
        )

        if type(self._backend) is not OpenTADRegisteredProfileBackend:
            raise TypeError("formal profile requires the fixed repository-owned backend")
        from opentad.models.chronotransport.full_stack_profiler import (
            PROFILE_ARTIFACT_SCHEMA,
            _formal_common_from_serialized_candidates,
        )
        from opentad.models.chronotransport.protocol import canonical_sha256

        candidates = [
            self._profile_registered_candidate(plan)
            for plan in self._registration["profiler"]["candidate_plan"]
        ]
        common = _formal_common_from_serialized_candidates(
            candidates, validated=self._registration
        )
        body = {"schema": PROFILE_ARTIFACT_SCHEMA, **common}
        body["profile_sha256"] = canonical_sha256(body)
        return body


def build_registered_profile_session(
    registration: Mapping[str, object],
) -> RegisteredOpenTADProfileSession:
    """Construct the sole production profile session."""

    if not isinstance(registration, Mapping):
        raise TypeError("registered profile session requires a registration mapping")
    return RegisteredOpenTADProfileSession(registration)
