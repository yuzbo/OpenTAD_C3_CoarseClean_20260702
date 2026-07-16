#!/usr/bin/env python3
"""Repository-owned post-Stage-C replay producer for formal Gate 3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Mapping

import torch

from opentad.models.chronotransport.formal_stage_b import (
    logical_risk_predictor_state_sha256,
)
from opentad.models.chronotransport.formal_stage_c import (
    STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    build_stage_c_completion_marker,
    load_paired_stage_c_checkpoint,
    validate_paired_stage_c_checkpoint,
)
from opentad.models.chronotransport.filesystem import (
    load_bound_torch,
    read_bound_bytes,
)
from opentad.models.chronotransport.gates23 import (
    R2_SEEDS,
    load_exact_canonical_json,
)
from opentad.models.chronotransport.post_stage_c import (
    build_post_stage_c_replay_artifact,
)
from opentad.models.chronotransport.registration import FORMAL_OUTPUT_BASE
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from opentad.utils import set_seed
from tools.bata.chronotransport_r2_gates23_replay_factory import (
    ManifestFitBatchSequence,
    _DETERMINISTIC_EVAL_SEED,
    _EXECUTION,
    _NO_LEAK,
    _deterministic_split_dataset,
    _run_window_vector,
)
from tools.bata.chronotransport_r2_stage_b_factory import _runtime
from tools.bata.train_chronotransport_r2_stage_c import (
    _ledger_bytes,
    _prepare,
)


_POST_STAGE_C_ENTRYPOINT = "tools/bata/run_chronotransport_r2_post_stage_c_gate3.py"


@dataclass(frozen=True)
class ValidatedStageCSeed:
    seed: int
    registration: dict[str, Any]
    registration_commit: str
    components: Any
    state: Any
    binding: dict[str, str]

    def close(self) -> None:
        self.components.close()


def _stage_c_outputs(registration_commit: str, seed: int) -> dict[str, Path]:
    root = Path(FORMAL_OUTPUT_BASE) / registration_commit / str(seed) / "stage_c"
    return {
        "output": root / "stage_c_paired_complete.pth",
        "ledger": root / "stage_c_paired_ledger.jsonl",
        "terminal": root / "stage_c_paired_terminal.json",
    }


def _stage_c_args(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    gate1_unlock_path: Path,
    gates23_replay_path: Path,
    gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
    seed: int,
) -> argparse.Namespace:
    outputs = _stage_c_outputs(registration_commit, seed)
    manifest = registration["window_manifest"]
    return argparse.Namespace(
        registration=registration_path,
        gate1_unlock=gate1_unlock_path,
        gates23_replay=gates23_replay_path,
        gates23_report=gates23_report_path,
        phase_marker_3407=phase_marker_paths[3407],
        phase_marker_3408=phase_marker_paths[3408],
        phase_marker_3409=phase_marker_paths[3409],
        manifest=Path(manifest["source_path"]),
        media_registry=Path(manifest["registry_path"]),
        config_identity=Path(manifest["config_identity_path"]),
        seed=seed,
        output=outputs["output"],
        ledger=outputs["ledger"],
        terminal=outputs["terminal"],
        resume=None,
        precheck_only=False,
    )


def _load_validated_stage_c_seed(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    gate1_unlock_path: Path,
    gates23_replay_path: Path,
    gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
    seed: int,
    entrypoint_relative: str,
) -> ValidatedStageCSeed:
    args = _stage_c_args(
        registration_path=registration_path,
        registration=registration,
        registration_commit=registration_commit,
        gate1_unlock_path=gate1_unlock_path,
        gates23_replay_path=gates23_replay_path,
        gates23_report_path=gates23_report_path,
        phase_marker_paths=phase_marker_paths,
        seed=seed,
    )
    prepared = _prepare(args, entrypoint_relative=entrypoint_relative)
    (
        prepared_registration,
        prepared_commit,
        outputs,
        components,
        provenance,
        state,
    ) = prepared
    try:
        if (
            prepared_commit != registration_commit
            or prepared_registration != dict(registration)
        ):
            raise RuntimeError("post-Stage-C seed preparation changed registration identity")
        _, checkpoint, _, checkpoint_sha256 = load_bound_torch(
            outputs["output"], label=f"post-Stage-C checkpoint {seed}"
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("post-Stage-C checkpoint must be a mapping")
        checkpoint = validate_paired_stage_c_checkpoint(
            checkpoint,
            expected_seed=seed,
            expected_fit_window_ids=components.fit_window_ids,
            expected_provenance=provenance,
            formal=True,
            require_complete=True,
            expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
        )
        _, ledger_bytes, ledger_sha256 = read_bound_bytes(
            outputs["ledger"], label=f"post-Stage-C ledger {seed}"
        )
        if ledger_bytes != _ledger_bytes(checkpoint):
            raise ValueError("post-Stage-C ledger differs from completed checkpoint")
        terminal = load_exact_canonical_json(
            outputs["terminal"], label=f"post-Stage-C terminal {seed}"
        )
        expected_terminal = build_stage_c_completion_marker(
            checkpoint,
            checkpoint_path=str(outputs["output"]),
            checkpoint_file_sha256=checkpoint_sha256,
            ledger_path=str(outputs["ledger"]),
            ledger_file_sha256=ledger_sha256,
        )
        if not isinstance(terminal, Mapping) or dict(terminal) != expected_terminal:
            raise ValueError("post-Stage-C completion marker differs from recomputation")

        load_paired_stage_c_checkpoint(
            state,
            checkpoint,
            expected_seed=seed,
            expected_fit_window_ids=components.fit_window_ids,
            expected_provenance=provenance,
            formal=True,
            expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
        )
        state.ct_objects["ema"].copy_to(components.ct_model)
        components.ct_model.eval()
        runtime = _runtime(components.ct_model)
        runtime.capture_replay_signals = True
        runtime.set_checkpoint_loaded(True)
        predictor_sha256 = logical_risk_predictor_state_sha256(
            components.ct_model, components.ct_model.state_dict()
        )
        phase = load_exact_canonical_json(
            phase_marker_paths[seed], label=f"post-Stage-C phase marker {seed}"
        )
        binding = {
            "completion_artifact_sha256": expected_terminal["artifact_sha256"],
            "checkpoint_file_sha256": checkpoint_sha256,
            "checkpoint_provenance_sha256": checkpoint["provenance_sha256"],
            "predictor_canonical_sha256": predictor_sha256,
            "fit_baseline_payload_sha256": phase["fit_baseline"]["payload_sha256"],
        }
        return ValidatedStageCSeed(
            seed=seed,
            registration=dict(prepared_registration),
            registration_commit=prepared_commit,
            components=components,
            state=state,
            binding=binding,
        )
    except BaseException:
        components.close()
        raise


def validate_completed_stage_c_population(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    gate1_unlock_path: Path,
    gates23_replay_path: Path,
    gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
    entrypoint_relative: str = _POST_STAGE_C_ENTRYPOINT,
) -> dict[str, dict[str, str]]:
    """Revalidate all three completed Stage-C chains without producing rows."""

    bindings: dict[str, dict[str, str]] = {}
    for seed in R2_SEEDS:
        context = _load_validated_stage_c_seed(
            registration_path=registration_path,
            registration=registration,
            registration_commit=registration_commit,
            gate1_unlock_path=gate1_unlock_path,
            gates23_replay_path=gates23_replay_path,
            gates23_report_path=gates23_report_path,
            phase_marker_paths=phase_marker_paths,
            seed=seed,
            entrypoint_relative=entrypoint_relative,
        )
        try:
            bindings[str(seed)] = dict(context.binding)
        finally:
            context.close()
            del context
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return bindings


def build_registered_post_stage_c_replay_artifact(
    *,
    registration_path: Path,
    registration: Mapping[str, Any],
    registration_commit: str,
    gate1_unlock: Mapping[str, Any],
    gate1_unlock_path: Path,
    pre_stage_c_gates23_replay: Mapping[str, Any],
    gates23_replay_path: Path,
    pre_stage_c_gates23_report: Mapping[str, Any],
    gates23_report_path: Path,
    phase_marker_paths: Mapping[int, Path],
) -> dict[str, Any]:
    """Execute Stage-C EMA checkpoints on the frozen calibration/evaluation set."""

    if (
        pre_stage_c_gates23_report.get("status") != "PASS"
        or pre_stage_c_gates23_report.get("gates23_replay_artifact_sha256")
        != pre_stage_c_gates23_replay.get("artifact_sha256")
        or pre_stage_c_gates23_report.get("registration_sha256")
        != registration.get("registration_sha256")
        or pre_stage_c_gates23_replay.get("registration_sha256")
        != registration.get("registration_sha256")
        or gate1_unlock.get("artifact_sha256")
        != pre_stage_c_gates23_replay.get("gate1_unlock_artifact_sha256")
    ):
        raise ValueError("post-Stage-C producer input chain identity mismatch")

    manifest = registration["window_manifest"]["artifact"]
    windows_by_id = {str(row["window_id"]): row for row in manifest["windows"]}
    split_windows = {
        split: list(map(str, manifest["splits"][split]))
        for split in ("calibration", "evaluation")
    }
    videos = {
        window: str(windows_by_id[window]["video_id"])
        for split in ("calibration", "evaluation")
        for window in split_windows[split]
    }
    library = {
        str(row["name"]): row
        for row in registration["candidate_library"]["candidates"]
    }
    actions = {
        name: str(library[name]["action_sha256"])
        for name in R2_NON_DENSE_NAMES
    }
    vectors: dict[tuple[str, int, str], dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}

    for seed in R2_SEEDS:
        context = _load_validated_stage_c_seed(
            registration_path=registration_path,
            registration=registration,
            registration_commit=registration_commit,
            gate1_unlock_path=gate1_unlock_path,
            gates23_replay_path=gates23_replay_path,
            gates23_report_path=gates23_report_path,
            phase_marker_paths=phase_marker_paths,
            seed=seed,
            entrypoint_relative=_POST_STAGE_C_ENTRYPOINT,
        )
        try:
            bindings[str(seed)] = dict(context.binding)
            for split_index, split in enumerate(("calibration", "evaluation")):
                dataset, windows = _deterministic_split_dataset(
                    context.components.cfg,
                    context.components.manifest,
                    split,
                    context.registration,
                )
                try:
                    batches = ManifestFitBatchSequence(
                        dataset, windows, torch.device("cuda:0")
                    )
                    for window_index in range(len(batches)):
                        set_seed(
                            _DETERMINISTIC_EVAL_SEED
                            + split_index * 30
                            + window_index
                        )
                        window = windows[window_index]
                        vector = _run_window_vector(
                            context.components.ct_model,
                            batches[window_index],
                            registration=context.registration,
                            registered_actions=actions,
                        )
                        vectors[(split, seed, str(window["window_id"]))] = vector
                finally:
                    dataset.close()
        finally:
            context.close()
            del context
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for split in ("calibration", "evaluation"):
        for seed in R2_SEEDS:
            binding = bindings[str(seed)]
            for window in split_windows[split]:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "window_id": window,
                        "video_id": videos[window],
                        "trained_checkpoint_sha256": binding[
                            "checkpoint_file_sha256"
                        ],
                        "predictor_canonical_sha256": binding[
                            "predictor_canonical_sha256"
                        ],
                        "candidate_order": list(R2_NON_DENSE_NAMES),
                        "execution": dict(_EXECUTION),
                        "no_leak": dict(_NO_LEAK),
                        **vectors[(split, seed, window)],
                    }
                )
    return build_post_stage_c_replay_artifact(
        rows,
        registration_sha256=registration["registration_sha256"],
        registration_commit=registration_commit,
        gate1_unlock_artifact_sha256=gate1_unlock["artifact_sha256"],
        pre_stage_c_gates23_report_sha256=pre_stage_c_gates23_report[
            "artifact_sha256"
        ],
        manifest_sha256=manifest["manifest_sha256"],
        library_sha256=registration["candidate_library"]["library_sha256"],
        split_window_ids=split_windows,
        video_id_by_window=videos,
        candidate_action_sha256_by_name=actions,
        stage_c_bindings=bindings,
    )


__all__ = [
    "build_registered_post_stage_c_replay_artifact",
    "validate_completed_stage_c_population",
]
