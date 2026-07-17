from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_TRAINING_PROFILE = "exposure132"
LEGACY_EXPOSURE132_COMMITS = frozenset(
    {"1642f265e48391418a7c8a4a087e33e2b7bf6899"}
)


@dataclass(frozen=True)
class CellCFTrainingProtocol:
    name: str
    purpose: str
    end_epoch: int
    steps_per_epoch: int
    checkpoint_interval: int = 5

    @property
    def expected_successful_optimizer_updates(self) -> int:
        return self.end_epoch * self.steps_per_epoch

    @property
    def terminal_epoch(self) -> int:
        return self.end_epoch - 1

    @property
    def terminal_state_key(self) -> str:
        return "state_dict_ema"

    @property
    def checkpoint_criterion(self) -> str:
        return (
            f"terminal_epoch_{self.terminal_epoch}_{self.terminal_state_key}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            expected_successful_optimizer_updates=self.expected_successful_optimizer_updates,
            terminal_epoch=self.terminal_epoch,
            terminal_state_key=self.terminal_state_key,
            checkpoint_criterion=self.checkpoint_criterion,
        )
        return payload


TRAINING_PROTOCOLS = {
    "exposure132": CellCFTrainingProtocol(
        name="exposure132",
        purpose="sufficient_exposure_diagnostic_not_default_paper_recipe",
        end_epoch=132,
        steps_per_epoch=100,
    ),
    "official60": CellCFTrainingProtocol(
        name="official60",
        purpose="official_adatad_length_matched_paper_protocol",
        end_epoch=60,
        steps_per_epoch=100,
    ),
}


def protocol_for_name(name: str | None) -> CellCFTrainingProtocol:
    normalized = DEFAULT_TRAINING_PROFILE if name is None else str(name)
    try:
        return TRAINING_PROTOCOLS[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(TRAINING_PROTOCOLS))
        raise ValueError(
            f"unknown CellCF training profile {normalized!r}; expected one of {allowed}"
        ) from exc


def protocol_from_environment() -> CellCFTrainingProtocol:
    return protocol_for_name(os.environ.get("DUCA_CELLCF_TRAINING_PROFILE"))


def protocol_from_workflow(workflow: Mapping[str, Any]) -> CellCFTrainingProtocol:
    protocol = protocol_for_name(workflow.get("training_profile"))
    expected = {
        "end_epoch": protocol.end_epoch,
        "expected_train_batches_per_epoch": protocol.steps_per_epoch,
        "expected_successful_optimizer_updates": protocol.expected_successful_optimizer_updates,
        "checkpoint_interval": protocol.checkpoint_interval,
        "primary_checkpoint_epoch": protocol.terminal_epoch,
        "primary_checkpoint_state_key": protocol.terminal_state_key,
        "checkpoint_criterion": protocol.checkpoint_criterion,
    }
    for key, value in expected.items():
        observed = workflow.get(key)
        if observed != value:
            raise ValueError(
                f"CellCF {protocol.name} workflow {key} mismatch: "
                f"expected {value!r}, got {observed!r}"
            )
    return protocol


__all__ = [
    "CellCFTrainingProtocol",
    "DEFAULT_TRAINING_PROFILE",
    "LEGACY_EXPOSURE132_COMMITS",
    "TRAINING_PROTOCOLS",
    "protocol_for_name",
    "protocol_from_environment",
    "protocol_from_workflow",
]
