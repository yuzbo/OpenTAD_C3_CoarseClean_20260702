"""Closed repository-owned construction boundary for formal Gate-1 replay."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class RegisteredGate1ReplaySession:
    """Own the fixed detector, checkpoint, manifest, data, and formal serializer."""

    def __init__(self, registration: Mapping[str, object]) -> None:
        from tools.bata.chronotransport_r2_opentad_profile_backend import (
            OpenTADRegisteredGate1ReplayBackend,
        )

        self._registration = registration
        self._backend = OpenTADRegisteredGate1ReplayBackend(registration)
        if type(self._backend) is not OpenTADRegisteredGate1ReplayBackend:
            raise TypeError("formal Gate 1 replay requires the fixed repository-owned backend")

    def _execute_registered_split(self, split: str) -> list[dict[str, object]]:
        from opentad.models.chronotransport.protocol import canonical_sha256
        from opentad.models.chronotransport.registration import (
            EXPECTED_PROFILE_CANDIDATE_ORDER,
            REGISTERED_PROFILE_BACKEND_IDENTITY,
            REGISTERED_PROFILE_BACKEND_SOURCE,
            validate_formal_random_control_lock,
        )
        from opentad.models.chronotransport.replay import (
            PairedReplaySnapshot,
            _LOWER_SHA256,
            _gate1_detector_loss,
            materialized_batch_sha256,
        )
        from tools.bata.chronotransport_r2_opentad_profile_backend import (
            resolve_registered_action_payload,
        )

        registered = self._registration
        validate_formal_random_control_lock(registered)
        if split not in ("calibration", "evaluation"):
            raise ValueError("formal Gate 1 replay split must be calibration or evaluation")
        detector = self._backend.model
        if detector.training or any(module.training for module in detector.modules()):
            raise RuntimeError("Gate 1 paired replay requires detector/head eval mode")
        window_ids = registered["window_manifest"]["artifact"]["splits"][split]
        global_index = {
            window_id: index
            for index, window_id in enumerate(registered["profiler"]["invocation_ids"])
        }
        plans = {
            plan["candidate_name"]: plan
            for plan in registered["profiler"]["candidate_plan"]
        }
        canonical_order = tuple(EXPECTED_PROFILE_CANDIDATE_ORDER)
        reverse_order = tuple(reversed(canonical_order))
        rows: list[dict[str, object]] = []
        for window_id in window_ids:
            batch_record = self._backend.materialize_registered_replay_window(window_id)
            if not isinstance(batch_record, Mapping) or set(batch_record) != {
                "forward_kwargs",
                "augmentation_sha256",
            }:
                raise ValueError("Gate 1 fixed backend materialized batch fields mismatch")
            forward_kwargs = batch_record["forward_kwargs"]
            augmentation_sha256 = batch_record["augmentation_sha256"]
            if not isinstance(forward_kwargs, Mapping):
                raise TypeError("Gate 1 fixed backend forward_kwargs must be a mapping")
            if not isinstance(augmentation_sha256, str) or not _LOWER_SHA256.fullmatch(
                augmentation_sha256
            ):
                raise ValueError("Gate 1 augmentation_sha256 must be lowercase SHA-256")
            batch_sha256 = materialized_batch_sha256(forward_kwargs)
            snapshot = PairedReplaySnapshot.capture()
            dense_payload = resolve_registered_action_payload(
                registered,
                window_id=window_id,
                candidate_name="dense",
            )
            captured_motion: list[torch.Tensor] = []
            _gate1_detector_loss(
                detector,
                forward_kwargs,
                candidate_name="dense",
                actions=torch.as_tensor(dense_payload, dtype=torch.long),
                snapshot=snapshot,
                expected_batch_sha256=batch_sha256,
                capture_motion=captured_motion,
            )
            if len(captured_motion) != 1:
                raise RuntimeError("Gate 1 did not derive one deploy-visible motion vector")
            motion = captured_motion[0]
            action_payloads = {
                name: resolve_registered_action_payload(
                    registered,
                    window_id=window_id,
                    candidate_name=name,
                    deploy_visible_motion=(
                        motion if name.startswith("motion_topk_p") else None
                    ),
                )
                for name in canonical_order
            }
            actions = {
                name: torch.as_tensor(payload, dtype=torch.long)
                for name, payload in action_payloads.items()
            }

            def execute(order: tuple[str, ...]) -> dict[str, float]:
                return {
                    name: _gate1_detector_loss(
                        detector,
                        forward_kwargs,
                        candidate_name=name,
                        actions=actions[name],
                        snapshot=snapshot,
                        expected_batch_sha256=batch_sha256,
                    )
                    for name in order
                }

            canonical_losses = execute(canonical_order)
            reverse_losses = execute(reverse_order)
            if any(
                reverse_losses[name] != canonical_losses[name]
                for name in canonical_order
            ):
                raise RuntimeError("Gate 1 paired runner candidate-order probe changed loss")
            dense_loss = canonical_losses["dense"]
            invocation_index = global_index[window_id]
            expected_action_hashes = [
                plans[name]["requested_action_sha256_by_invocation"][invocation_index]
                for name in canonical_order
            ]
            actual_action_hashes = [
                canonical_sha256(action_payloads[name]) for name in canonical_order
            ]
            if actual_action_hashes != expected_action_hashes:
                raise RuntimeError("Gate 1 paired runner action provenance mismatch")
            rows.append(
                {
                    "window_id": window_id,
                    "candidate_names": list(canonical_order),
                    "dense_detector_loss": dense_loss,
                    "candidate_detector_loss": [
                        canonical_losses[name] for name in canonical_order
                    ],
                    "order_probe_candidate_names": list(reverse_order),
                    "order_probe_candidate_detector_loss": [
                        reverse_losses[name] for name in reverse_order
                    ],
                    "materialized_window_sha256": batch_sha256,
                    "augmentation_sha256": augmentation_sha256,
                    "deploy_visible_motion_sha256": canonical_sha256(
                        motion.float().tolist()
                    ),
                    "dense_reference_sha256": canonical_sha256(
                        {
                            "window_id": window_id,
                            "materialized_window_sha256": batch_sha256,
                            "dense_detector_loss": dense_loss,
                        }
                    ),
                    "dense_checkpoint_sha256": registered["dense_checkpoint"]["sha256"],
                    "config_sha256": registered["profiler"]["model_config_sha256"],
                    "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
                    "backend_source_sha256": registered["source_files"][
                        REGISTERED_PROFILE_BACKEND_SOURCE
                    ],
                    "candidate_action_sha256": actual_action_hashes,
                }
            )
        return rows

    def run_split(self, split: str) -> dict[str, object]:
        from tools.bata.chronotransport_r2_opentad_profile_backend import (
            OpenTADRegisteredGate1ReplayBackend,
        )

        if type(self._backend) is not OpenTADRegisteredGate1ReplayBackend:
            raise TypeError("formal Gate 1 replay requires the fixed repository-owned backend")
        from opentad.models.chronotransport.adjudication import (
            GATE1_PAIRED_CANDIDATE_ORDER,
            GATE1_PAIRED_REPLAY_SCHEMA,
            _validate_serialized_formal_replay_row,
        )
        from opentad.models.chronotransport.protocol import canonical_sha256

        source_rows = self._execute_registered_split(split)
        expected_ids = self._registration["window_manifest"]["artifact"]["splits"][split]
        global_index = {
            window_id: index
            for index, window_id in enumerate(self._registration["profiler"]["invocation_ids"])
        }
        rows = []
        for row, window_id in zip(source_rows, expected_ids):
            serialized = dict(row)
            dense_loss = float(serialized["dense_detector_loss"])
            serialized["detector_regret"] = [
                max(float(loss) - dense_loss, 0.0)
                for loss in serialized["candidate_detector_loss"]
            ]
            rows.append(
                _validate_serialized_formal_replay_row(
                    serialized,
                    registration=self._registration,
                    window_id=window_id,
                    invocation_index=global_index[window_id],
                )
            )
        body: dict[str, object] = {
            "schema": GATE1_PAIRED_REPLAY_SCHEMA,
            "registration_sha256": self._registration["registration_sha256"],
            "observed_environment": dict(self._backend.observed_environment),
            "split": split,
            "window_ids": list(expected_ids),
            "window_order_sha256": canonical_sha256(expected_ids),
            "candidate_names": list(GATE1_PAIRED_CANDIDATE_ORDER),
            "candidate_order_sha256": canonical_sha256(GATE1_PAIRED_CANDIDATE_ORDER),
            "order_probe_candidate_names": list(
                reversed(GATE1_PAIRED_CANDIDATE_ORDER)
            ),
            "order_probe_candidate_order_sha256": canonical_sha256(
                tuple(reversed(GATE1_PAIRED_CANDIDATE_ORDER))
            ),
            "rows": rows,
        }
        body["artifact_sha256"] = canonical_sha256(body)
        return body


def build_registered_gate1_replay_session(
    registration: Mapping[str, object],
) -> RegisteredGate1ReplaySession:
    """Construct the sole production Gate-1 paired-replay session."""

    if not isinstance(registration, Mapping):
        raise TypeError("registered Gate 1 replay session requires a registration mapping")
    return RegisteredGate1ReplaySession(registration)
