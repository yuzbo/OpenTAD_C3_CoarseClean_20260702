from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.bata.duca_protected_physical_p3 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DURATION_STRATA,
    SWAPS_PER_WINDOW,
    WINDOWS_PER_STRATUM,
    aggregate_p3_rows,
    stratified_window_manifest,
)
from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    sha256_file,
)
ROOT = Path(__file__).resolve().parents[2]
CONTEXT_SCHEMA = "duca_boundary_burst_hard_swap_alignment_context_v1"
SHARD_SCHEMA = "duca_boundary_burst_hard_swap_alignment_shard_v1"
ARTIFACT_SCHEMA = "duca_boundary_burst_hard_swap_alignment_v1"
FULL_MODEL_GATE_SCHEMA = "duca_protected_e2e_exact_full_model_gradient_gate_v1"
POPULATION_CONFIG = (
    "configs/adatad/thumos/duca_protected_physical_p3_train_windows.py"
)
UNIFORM_OFFICIAL_VARIANT = "two_stage_exact_uniform"

FAMILY_FEEDBACK_ROUTES = {
    "R2Q3_privileged_boundary_burst": {
        "g1_variant": "boundary_burst_r2q3_g1",
        "g1_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_g1_protected_fixed384_official60.py"
        ),
        "g2_variant": "boundary_burst_r2q3_g2",
        "g2_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_g2_uni_companion_fixed384_official60.py"
        ),
    },
    "R4Q5_privileged_boundary_burst": {
        "g1_variant": "boundary_burst_r4q5_g1",
        "g1_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_r4q5_g1_protected_fixed384_official60.py"
        ),
        "g2_variant": "boundary_burst_r4q5_g2",
        "g2_config": (
            "configs/adatad/thumos/"
            "duca_boundary_burst_r4q5_g2_uni_companion_fixed384_official60.py"
        ),
    },
}


class BoundaryBurstAlignmentFailure(RuntimeError):
    pass


def _upstream_validators():
    from tools.bata.aggregate_duca_boundary_burst_results import (
        validate_suite_self_hash,
    )
    from tools.bata.select_duca_boundary_burst_candidates import (
        validate_frontend_decision,
        validate_full_model_gate,
    )

    return (
        validate_suite_self_hash,
        validate_frontend_decision,
        validate_full_model_gate,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryBurstAlignmentFailure(
            f"boundary-burst hard-swap alignment failed: {message}"
        )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    _require(
        len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        f"{label} is not SHA-256",
    )
    return digest


def _require_file(path: Any, digest: Any, label: str) -> Path:
    resolved = _resolve(str(path))
    expected = _require_sha256(digest, f"{label} hash")
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    _require(sha256_file(resolved) == expected, f"{label} content drift")
    return resolved


def _load_json(path: Any, digest: Any, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _require_file(path, digest, label)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload, resolved


def _validate_self_hash(
    payload: Mapping[str, Any], *, hash_key: str, label: str
) -> str:
    digest = _require_sha256(payload.get(hash_key), f"{label} self hash")
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    _require(canonical_sha256(unsigned) == digest, f"{label} self-hash mismatch")
    return digest


def _atomic_write_sealed(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    hash_key: str,
) -> Path:
    output = Path(path).expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    sealed = dict(payload)
    sealed[hash_key] = canonical_sha256(sealed)
    encoded = json.dumps(sealed, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = json.loads(temporary.read_text(encoding="utf-8"))
        _validate_self_hash(persisted, hash_key=hash_key, label=output.name)
        os.replace(temporary, output)
        if os.name != "nt":
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _source_identity(expected_commit: str) -> dict[str, str]:
    commit = str(expected_commit).lower()
    _require(len(commit) == 40, "an exact commit is required")
    _require(_git_output("rev-parse", "HEAD") == commit, "commit drift")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=normal"),
        "clean tree required",
    )
    return {"git_commit": commit, "git_tree": _git_output("rev-parse", "HEAD^{tree}")}


def _terminal_rows(suite: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = suite.get("results")
    _require(isinstance(rows, list), "terminal suite results are missing")
    by_variant = {
        str(row.get("variant")): row for row in rows if isinstance(row, Mapping)
    }
    _require(len(by_variant) == len(rows), "terminal suite variants are duplicated")
    return by_variant


def _average_map(row: Mapping[str, Any], label: str) -> float:
    value = row.get("average_mAP")
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} average_mAP is invalid",
    )
    return float(value)


def _load_terminal_suite(
    *, path: str | Path, digest: str, expected_commit: str
) -> tuple[dict[str, Any], Path]:
    suite, resolved = _load_json(path, digest, "terminal U/G0 suite")
    validate_suite_self_hash, _, _ = _upstream_validators()
    try:
        validate_suite_self_hash(suite)
    except RuntimeError as exc:
        raise BoundaryBurstAlignmentFailure(str(exc)) from exc
    _require(suite.get("git_commit") == expected_commit, "terminal suite commit drift")
    _require(
        suite.get("status") == "matched_u_selected_g0_terminal_ema_results_sealed",
        "terminal U/G0 suite is not complete",
    )
    return suite, resolved


def _validate_full_gate_record(
    *,
    record: Mapping[str, Any],
    variant: str,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    gate, gate_path = _load_json(record.get("path"), record.get("sha256"), f"{variant} gate")
    runtime = gate.get("runtime")
    selector_initialization = gate.get("selector_initialization")
    p0 = context["p0_initialization"]
    _require(
        gate.get("schema") == FULL_MODEL_GATE_SCHEMA
        and gate.get("ok") is True
        and isinstance(runtime, Mapping)
        and runtime.get("git_commit") == context["git_commit"]
        and runtime.get("git_tree") == context["git_tree"],
        f"{variant} full-model gate source identity drift",
    )
    _require(
        gate.get("config_sha256") == config["config_sha256"],
        f"{variant} full-model gate config drift",
    )
    _require(
        gate.get("adatad_pretrain") == context["adatad_pretrain"],
        f"{variant} full-model gate AdaTAD pretrain drift",
    )
    _require(
        isinstance(selector_initialization, Mapping)
        and selector_initialization.get("checkpoint_sha256") == p0["sha256"]
        and int(selector_initialization.get("checkpoint_epoch", -1)) == p0["epoch"]
        and selector_initialization.get("checkpoint_state_key") == "state_dict_ema"
        and selector_initialization.get("detector_state_loaded") is False
        and selector_initialization.get("optimizer_state_loaded") is False
        and selector_initialization.get("scheduler_state_loaded") is False,
        f"{variant} full-model gate P0 initialization drift",
    )
    return {"path": str(gate_path), "sha256": sha256_file(gate_path)}


def validate_alignment_context(
    payload: Mapping[str, Any], *, expected_commit: str | None = None
) -> dict[str, Any]:
    _require(
        payload.get("schema") == CONTEXT_SCHEMA
        and payload.get("ok") is True
        and payload.get("fail_closed") is True
        and payload.get("task") == "offline_temporal_action_detection",
        "alignment context schema/status mismatch",
    )
    _validate_self_hash(payload, hash_key="context_sha256", label="alignment context")
    if expected_commit is not None:
        _require(payload.get("git_commit") == expected_commit, "alignment context commit drift")
    suite_binding = payload.get("terminal_suite")
    _require(isinstance(suite_binding, Mapping), "terminal suite binding is missing")
    suite, _ = _load_terminal_suite(
        path=suite_binding.get("path"),
        digest=suite_binding.get("sha256"),
        expected_commit=str(payload.get("git_commit")),
    )
    _require(
        suite.get("suite_sha256") == suite_binding.get("self_sha256"),
        "terminal suite self-hash copy drift",
    )
    decision_binding = payload.get("frontend_decision")
    gate_binding = payload.get("gate_suite")
    _require(
        isinstance(decision_binding, Mapping) and isinstance(gate_binding, Mapping),
        "upstream decision/gate bindings are missing",
    )
    _, validate_frontend_decision, validate_full_model_gate = _upstream_validators()
    decision = validate_frontend_decision(
        decision_path=decision_binding.get("path"),
        decision_sha256=decision_binding.get("sha256"),
        expected_commit=str(payload.get("git_commit")),
    )
    validate_full_model_gate(
        gate_path=gate_binding.get("path"),
        gate_sha256=gate_binding.get("sha256"),
        decision_path=decision_binding.get("path"),
        decision_sha256=decision_binding.get("sha256"),
        expected_commit=str(payload.get("git_commit")),
    )
    _require(
        decision["family_routing"] == suite["family_routing"],
        "terminal suite family routing drift",
    )
    family = str(payload.get("selected_weakest_projected_family", ""))
    _require(
        family == decision["family_routing"]["selected_weakest_projected_family"]
        and family in FAMILY_FEEDBACK_ROUTES,
        "alignment context is not bound to the R0-selected family",
    )
    rows = _terminal_rows(suite)
    selected_variant = decision["family_routing"]["selected_official60_variant"]
    _require(
        set(rows) == {UNIFORM_OFFICIAL_VARIANT, selected_variant},
        "terminal suite must contain only matched U and selected G0",
    )
    uniform_map = _average_map(rows[UNIFORM_OFFICIAL_VARIANT], "uniform")
    selected_map = _average_map(rows[selected_variant], "selected G0")
    _require(selected_map > uniform_map, "G0 did not beat matched exact-uniform")
    _require(
        math.isclose(
            float(payload.get("g0_minus_uniform_average_mAP", float("nan"))),
            selected_map - uniform_map,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "G0-U prerequisite value drift",
    )
    selected_g0 = payload.get("selected_g0")
    _require(
        isinstance(selected_g0, Mapping)
        and selected_g0.get("variant") == selected_variant,
        "selected G0 binding drift",
    )
    _require_file(
        selected_g0.get("config", {}).get("path"),
        selected_g0.get("config", {}).get("sha256"),
        "selected G0 config",
    )
    _require_file(
        selected_g0.get("checkpoint", {}).get("path"),
        selected_g0.get("checkpoint", {}).get("sha256"),
        "selected G0 checkpoint",
    )
    _require(
        selected_g0.get("checkpoint", {}).get("epoch") == 59
        and selected_g0.get("checkpoint", {}).get("state_key") == "state_dict_ema",
        "selected G0 checkpoint is not terminal EMA",
    )
    _require_file(
        payload.get("adatad_pretrain", {}).get("path"),
        payload.get("adatad_pretrain", {}).get("sha256"),
        "AdaTAD pretrain",
    )
    _require_file(
        payload.get("p0_initialization", {}).get("path"),
        payload.get("p0_initialization", {}).get("sha256"),
        "selected P0 initialization",
    )
    population = payload.get("population")
    _require(isinstance(population, Mapping), "alignment population binding is missing")
    _require_file(population.get("config_path"), population.get("config_sha256"), "population config")
    windows = population.get("windows")
    _require(
        isinstance(windows, list)
        and len(windows) == len(DURATION_STRATA) * WINDOWS_PER_STRATUM
        and canonical_sha256(windows) == population.get("windows_sha256"),
        "alignment population window seal drift",
    )
    authorized = payload.get("authorized_variants")
    _require(isinstance(authorized, Mapping), "authorized G1/G2 variants are missing")
    expected_feedback = FAMILY_FEEDBACK_ROUTES[family]
    expected_variants = {expected_feedback["g1_variant"], expected_feedback["g2_variant"]}
    _require(set(authorized) == expected_variants, "authorized G1/G2 set drift")
    for variant, record in authorized.items():
        _require(isinstance(record, Mapping), f"{variant} config binding is invalid")
        config_path = _require_file(
            record.get("config_path"), record.get("config_sha256"), f"{variant} config"
        )
        expected_path = _resolve(
            expected_feedback["g1_config"]
            if variant == expected_feedback["g1_variant"]
            else expected_feedback["g2_config"]
        )
        _require(config_path == expected_path, f"{variant} config path drift")
    alignment_model = payload.get("alignment_model")
    _require(
        isinstance(alignment_model, Mapping)
        and alignment_model.get("variant") == expected_feedback["g1_variant"]
        and alignment_model.get("config_sha256")
        == authorized[expected_feedback["g1_variant"]]["config_sha256"],
        "alignment model is not selected-family G1",
    )
    return dict(payload)


def load_alignment_context(
    *, path: str | Path, digest: str, expected_commit: str | None = None
) -> tuple[dict[str, Any], Path]:
    payload, resolved = _load_json(path, digest, "alignment context")
    return validate_alignment_context(payload, expected_commit=expected_commit), resolved


def prepare_context(
    *,
    expected_commit: str,
    terminal_suite_path: str | Path,
    terminal_suite_sha256: str,
    population_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source = _source_identity(expected_commit)
    suite, suite_path = _load_terminal_suite(
        path=terminal_suite_path,
        digest=terminal_suite_sha256,
        expected_commit=expected_commit,
    )
    _, validate_frontend_decision, validate_full_model_gate = _upstream_validators()
    decision_path = _resolve(suite["frontend_decision_path"])
    decision_sha256 = str(suite["frontend_decision_sha256"])
    decision = validate_frontend_decision(
        decision_path=decision_path,
        decision_sha256=decision_sha256,
        expected_commit=expected_commit,
    )
    gate_path = _resolve(suite["gate_path"])
    gate_sha256 = str(suite["gate_sha256"])
    validate_full_model_gate(
        gate_path=gate_path,
        gate_sha256=gate_sha256,
        decision_path=decision_path,
        decision_sha256=decision_sha256,
        expected_commit=expected_commit,
    )
    routing = decision["family_routing"]
    family = routing["selected_weakest_projected_family"]
    _require(family in FAMILY_FEEDBACK_ROUTES, "unsupported R0-selected family")
    selected_variant = routing["selected_official60_variant"]
    rows = _terminal_rows(suite)
    _require(
        set(rows) == {UNIFORM_OFFICIAL_VARIANT, selected_variant},
        "terminal suite does not contain matched U/G0",
    )
    uniform_map = _average_map(rows[UNIFORM_OFFICIAL_VARIANT], "uniform")
    selected_map = _average_map(rows[selected_variant], "selected G0")
    _require(selected_map > uniform_map, "G0 must beat U before R4")

    selected_row = rows[selected_variant]
    completion, _ = _load_json(
        selected_row["completion_path"],
        selected_row["completion_sha256"],
        "selected G0 completion",
    )
    _require(
        completion.get("ok") is True
        and completion.get("variant") == selected_variant
        and completion.get("git_commit") == expected_commit,
        "selected G0 completion identity drift",
    )
    checkpoint = _require_file(
        completion.get("checkpoint_path"),
        completion.get("checkpoint_sha256"),
        "selected G0 terminal checkpoint",
    )
    evaluation, _ = _load_json(
        selected_row["evaluation_path"],
        selected_row["evaluation_sha256"],
        "selected G0 terminal evaluation",
    )
    _require(
        evaluation.get("checkpoint_sha256") == completion.get("checkpoint_sha256")
        and evaluation.get("checkpoint_epoch") == 59
        and evaluation.get("checkpoint_state_key") == "state_dict_ema",
        "selected G0 evaluation/checkpoint binding drift",
    )
    selected_config = _require_file(
        routing["selected_official60_config"],
        evaluation.get("config_sha256"),
        "selected G0 source config",
    )
    pretrain_binding = dict(suite["matched_arm_identity"]["adatad_pretrain"])
    pretrain_path = _require_file(
        pretrain_binding["path"], pretrain_binding["sha256"], "AdaTAD pretrain"
    )
    pretrain_binding["path"] = str(pretrain_path)
    p0_winner = decision["winners"][routing["selected_p0_variant"]]
    p0_path = _require_file(
        p0_winner["checkpoint_path"], p0_winner["checkpoint_sha256"], "selected P0 winner"
    )
    p0_initialization = {
        "path": str(p0_path),
        "sha256": p0_winner["checkpoint_sha256"],
        "epoch": int(p0_winner["epoch_one_based"]) - 1,
        "state_key": "state_dict_ema",
    }

    population_path = _require_file(
        population_config_path,
        sha256_file(_resolve(population_config_path)),
        "alignment population config",
    )
    from mmengine.config import Config
    from opentad.datasets import build_dataset

    population_cfg = Config.fromfile(str(population_path))
    _require(population_cfg.dataset.val is None, "population config exposes validation")
    dataset = build_dataset(population_cfg.dataset.train, default_args={"logger": None})
    windows = stratified_window_manifest(dataset)
    feedback = FAMILY_FEEDBACK_ROUTES[family]
    authorized: dict[str, dict[str, Any]] = {}
    for stage in ("g1", "g2"):
        variant = feedback[f"{stage}_variant"]
        config_path = _resolve(feedback[f"{stage}_config"])
        _require(config_path.is_file(), f"{variant} production config is missing")
        authorized[variant] = {
            "stage": stage.upper(),
            "selected_weakest_projected_family": family,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        }
    g1 = authorized[feedback["g1_variant"]]
    payload = {
        "schema": CONTEXT_SCHEMA,
        "ok": True,
        "fail_closed": True,
        "task": "offline_temporal_action_detection",
        **source,
        "terminal_suite": {
            "path": str(suite_path),
            "sha256": terminal_suite_sha256,
            "self_sha256": suite["suite_sha256"],
        },
        "frontend_decision": {
            "path": str(decision_path),
            "sha256": decision_sha256,
        },
        "gate_suite": {"path": str(gate_path), "sha256": gate_sha256},
        "selected_weakest_projected_family": family,
        "selected_g0": {
            "variant": selected_variant,
            "average_mAP": selected_map,
            "config": {"path": str(selected_config), "sha256": sha256_file(selected_config)},
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": completion["checkpoint_sha256"],
                "epoch": 59,
                "state_key": "state_dict_ema",
            },
        },
        "matched_uniform": {
            "variant": UNIFORM_OFFICIAL_VARIANT,
            "average_mAP": uniform_map,
        },
        "g0_minus_uniform_average_mAP": selected_map - uniform_map,
        "prerequisite_g0_beats_uniform": True,
        "alignment_model": {
            "variant": feedback["g1_variant"],
            "config_path": g1["config_path"],
            "config_sha256": g1["config_sha256"],
            "checkpoint_source": "selected_g0_terminal_epoch_59_state_dict_ema",
        },
        "authorized_variants": authorized,
        "p0_initialization": p0_initialization,
        "adatad_pretrain": pretrain_binding,
        "population": {
            "split": "train_only",
            "config_path": str(population_path),
            "config_sha256": sha256_file(population_path),
            "windows": windows,
            "windows_sha256": canonical_sha256(windows),
            "window_count": len(windows),
            "swaps_per_window": SWAPS_PER_WINDOW,
            "preregistered_swap_count": len(windows) * SWAPS_PER_WINDOW,
        },
        "statistics_contract": {
            "signed_quantity": "detector_utility=base_loss-candidate_loss",
            "alignment_predictor": "negative_protected_bridge_score_gradient_delta",
            "cluster_unit": "video_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "paper_claim_allowed": False,
    }
    output = _atomic_write_sealed(output_path, payload, hash_key="context_sha256")
    persisted = json.loads(output.read_text(encoding="utf-8"))
    validate_alignment_context(persisted, expected_commit=expected_commit)
    return persisted


def _validate_shard(
    *, payload: Mapping[str, Any], stratum: str, context: Mapping[str, Any], context_sha256: str
) -> tuple[list[dict[str, Any]], set[tuple[str, int]], set[tuple[str, int, int, int]]]:
    _require(
        payload.get("schema") == SHARD_SCHEMA
        and payload.get("ok") is True
        and payload.get("stratum") == stratum,
        f"{stratum} shard schema/status mismatch",
    )
    _require(
        payload.get("alignment_context_sha256") == context_sha256
        and payload.get("alignment_context_self_sha256") == context["context_sha256"]
        and payload.get("runtime", {}).get("git_commit") == context["git_commit"]
        and payload.get("runtime", {}).get("git_tree") == context["git_tree"],
        f"{stratum} shard context/source drift",
    )
    _require(
        payload.get("config_sha256") == context["alignment_model"]["config_sha256"]
        and payload.get("population_config_sha256")
        == context["population"]["config_sha256"]
        and payload.get("selected_g0_checkpoint")
        == context["selected_g0"]["checkpoint"],
        f"{stratum} shard model identity drift",
    )
    hard_swap_semantics = payload.get("hard_swap_semantics")
    _require(
        payload.get("optimizer_step") == 0
        and payload.get("loss_normalizer_frozen") is True
        and payload.get("train_split_only") is True
        and payload.get("test_loader_built") is False
        and payload.get("checkpoint_written") is False
        and isinstance(hard_swap_semantics, Mapping)
        and hard_swap_semantics.get("type")
        == "actual_hard_selected_position_one_swap"
        and hard_swap_semantics.get("exact_k_preserved") is True
        and hard_swap_semantics.get("physical_cap_preserved") is True,
        f"{stratum} shard is not a frozen-detector hard-swap experiment",
    )
    windows = payload.get("windows")
    rows = payload.get("rows")
    _require(
        isinstance(windows, list)
        and len(windows) == WINDOWS_PER_STRATUM
        and isinstance(rows, list)
        and len(rows) == WINDOWS_PER_STRATUM * SWAPS_PER_WINDOW,
        f"{stratum} shard population size drift",
    )
    _require(canonical_sha256(rows) == payload.get("row_sha256"), f"{stratum} row hash drift")
    registered_windows = {
        (str(item["video_id"]), int(item["window_start"]))
        for item in context["population"]["windows"]
        if item["duration_stratum"] == stratum
    }
    observed_windows = {(str(item["video_id"]), int(item["window_start"])) for item in windows}
    _require(observed_windows == registered_windows, f"{stratum} windows drifted")
    window_bases = {}
    for item in windows:
        key = (str(item["video_id"]), int(item["window_start"]))
        base = [int(value) for value in item.get("base_selected_positions", [])]
        _require(
            len(base) == 384
            and base == sorted(set(base))
            and canonical_sha256(base) == item.get("base_selected_positions_sha256"),
            f"{stratum} base selected-position identity drift",
        )
        window_bases[key] = base
    swaps: set[tuple[str, int, int, int]] = set()
    counts: Counter[tuple[str, int]] = Counter()
    quartiles: dict[tuple[str, int], Counter[int]] = {}
    typed_rows = []
    for raw in rows:
        _require(isinstance(raw, Mapping), f"{stratum} shard row is invalid")
        row = dict(raw)
        window = (str(row["video_id"]), int(row["window_start"]))
        key = (*window, int(row["removed"]), int(row["incoming"]))
        _require(window in observed_windows and key not in swaps, f"{stratum} duplicate/unregistered swap")
        _require(
            row.get("legal_one_swap") is True
            and row.get("hard_symmetric_difference_count") == 2
            and row.get("base_selected_count") == row.get("candidate_selected_count")
            and row.get("hard_forward_equal") is True,
            f"{stratum} row is not an actual legal hard RGB swap",
        )
        _require(
            isinstance(row.get("candidate_positions_sha256"), str)
            and row.get("base_positions_sha256")
            == canonical_sha256(window_bases[window])
            and int(row["removed"]) in window_bases[window]
            and int(row["incoming"]) not in window_bases[window],
            f"{stratum} base/candidate position seal is missing",
        )
        reconstructed = sorted(
            (set(window_bases[window]) - {int(row["removed"])})
            | {int(row["incoming"])}
        )
        _require(
            canonical_sha256(reconstructed) == row["candidate_positions_sha256"],
            f"{stratum} candidate selected-position identity drift",
        )
        _require(
            math.isclose(
                float(row["detector_utility"]),
                -float(row["actual_delta"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(row["predicted_utility"]),
                -float(row["predicted_delta"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"{stratum} signed utility alias drift",
        )
        swaps.add(key)
        counts[window] += 1
        quartiles.setdefault(window, Counter())[int(row["quartile"])] += 1
        typed_rows.append(row)
    _require(
        all(counts[key] == SWAPS_PER_WINDOW for key in observed_windows)
        and all(quartiles[key] == Counter({0: 3, 1: 3, 2: 3, 3: 3}) for key in observed_windows),
        f"{stratum} deterministic quartile sampling drift",
    )
    return typed_rows, observed_windows, swaps


def aggregate_alignment(
    *,
    expected_commit: str,
    context_path: str | Path,
    context_sha256: str,
    shard_paths: Mapping[str, str | Path],
    shard_sha256s: Mapping[str, str],
    full_model_gate_paths: Mapping[str, str | Path],
    full_model_gate_sha256s: Mapping[str, str],
    output_path: str | Path,
) -> dict[str, Any]:
    context, resolved_context = load_alignment_context(
        path=context_path, digest=context_sha256, expected_commit=expected_commit
    )
    _require(set(shard_paths) == set(DURATION_STRATA), "all three duration shards are required")
    all_rows: list[dict[str, Any]] = []
    all_windows: set[tuple[str, int]] = set()
    all_swaps: set[tuple[str, int, int, int]] = set()
    shard_bindings = {}
    for stratum in DURATION_STRATA:
        shard, path = _load_json(shard_paths[stratum], shard_sha256s[stratum], f"{stratum} shard")
        rows, windows, swaps = _validate_shard(
            payload=shard,
            stratum=stratum,
            context=context,
            context_sha256=context_sha256,
        )
        _require(all_windows.isdisjoint(windows) and all_swaps.isdisjoint(swaps), "shards overlap")
        all_rows.extend(rows)
        all_windows.update(windows)
        all_swaps.update(swaps)
        shard_bindings[stratum] = {"path": str(path), "sha256": sha256_file(path)}
    _require(
        len(all_windows) == 48 and len(all_swaps) == len(all_rows) == 576,
        "alignment population is not 48 windows/576 unique legal swaps",
    )
    summary = aggregate_p3_rows(
        all_rows,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    _require(summary.get("ok") is True, "preregistered signed alignment gate did not pass")
    authorized = context["authorized_variants"]
    _require(
        set(full_model_gate_paths) == set(authorized)
        and set(full_model_gate_sha256s) == set(authorized),
        "G1/G2 full-model gate set is incomplete",
    )
    gate_bindings = {}
    for variant, config in authorized.items():
        gate_bindings[variant] = _validate_full_gate_record(
            record={
                "path": full_model_gate_paths[variant],
                "sha256": full_model_gate_sha256s[variant],
            },
            variant=variant,
            config=config,
            context=context,
        )
    best_delta = summary["predicted_best_actual_delta_median"]
    best_delta_ci = list(best_delta["ci95"])
    signed_utility_alignment = {
        "predicted_utility": "negative_protected_bridge_score_gradient_delta",
        "detector_utility": "base_loss_minus_candidate_loss",
        "sign_agreement": dict(summary["sign_agreement"]),
        "window_spearman_median": dict(summary["window_spearman_median"]),
        "predicted_best_detector_utility_median": {
            "point": -float(best_delta["point"]),
            "ci95": [-float(best_delta_ci[1]), -float(best_delta_ci[0])],
        },
        "bootstrap_unit": "video_cluster",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "ok": True,
        "fail_closed": True,
        "production_feedback_unlocked": True,
        "status": "legal_hard_swap_alignment_and_g1_g2_full_model_gates_passed",
        "task": "offline_temporal_action_detection",
        "git_commit": context["git_commit"],
        "git_tree": context["git_tree"],
        "selected_weakest_projected_family": context[
            "selected_weakest_projected_family"
        ],
        "context": {
            "path": str(resolved_context),
            "sha256": context_sha256,
            "self_sha256": context["context_sha256"],
        },
        "terminal_suite": context["terminal_suite"],
        "selected_g0": context["selected_g0"],
        "g0_minus_uniform_average_mAP": context["g0_minus_uniform_average_mAP"],
        "authorized_variants": authorized,
        "full_model_gates": gate_bindings,
        "shards": shard_bindings,
        "hard_swap_count": len(all_swaps),
        "video_cluster_count": len({row["video_id"] for row in all_rows}),
        "alignment_summary": summary,
        "alignment_summary_sha256": canonical_sha256(summary),
        "signed_utility_alignment": signed_utility_alignment,
        "signed_utility_alignment_sha256": canonical_sha256(
            signed_utility_alignment
        ),
        "frozen_detector_contract": {
            "backend": "official_adatad_actionformer",
            "checkpoint": context["selected_g0"]["checkpoint"],
            "optimizer_steps": 0,
            "parameter_version_unchanged": True,
            "loss_normalizer_frozen": True,
        },
        "utility_semantics": {
            "predicted": "negative_protected_bridge_score_gradient_delta",
            "observed": "base_frozen_official_adatad_loss_minus_swapped_loss",
            "rgb_forward": "actual_remove_add_selected_rgb_materialization",
        },
        "paper_claim_allowed": False,
    }
    output = _atomic_write_sealed(output_path, payload, hash_key="alignment_sha256")
    persisted = json.loads(output.read_text(encoding="utf-8"))
    validate_alignment_artifact_payload(persisted, expected_commit=expected_commit)
    return persisted


def validate_alignment_artifact_payload(
    payload: Mapping[str, Any], *, expected_commit: str | None = None
) -> dict[str, Any]:
    _require(
        payload.get("schema") == ARTIFACT_SCHEMA
        and payload.get("ok") is True
        and payload.get("fail_closed") is True
        and payload.get("production_feedback_unlocked") is True,
        "alignment artifact schema/status mismatch",
    )
    _validate_self_hash(payload, hash_key="alignment_sha256", label="alignment artifact")
    if expected_commit is not None:
        _require(payload.get("git_commit") == expected_commit, "alignment artifact commit drift")
    context_binding = payload.get("context")
    _require(isinstance(context_binding, Mapping), "alignment context binding is missing")
    context, _ = load_alignment_context(
        path=context_binding.get("path"),
        digest=context_binding.get("sha256"),
        expected_commit=str(payload.get("git_commit")),
    )
    _require(
        context_binding.get("self_sha256") == context["context_sha256"]
        and payload.get("git_tree") == context["git_tree"]
        and payload.get("selected_weakest_projected_family")
        == context["selected_weakest_projected_family"]
        and payload.get("selected_g0") == context["selected_g0"]
        and payload.get("authorized_variants") == context["authorized_variants"],
        "alignment artifact/context identity drift",
    )
    summary = payload.get("alignment_summary")
    _require(
        isinstance(summary, Mapping)
        and summary.get("schema") == "duca_protected_physical_p3_aggregate_v1"
        and summary.get("ok") is True
        and canonical_sha256(summary) == payload.get("alignment_summary_sha256")
        and summary.get("bootstrap", {}).get("unit") == "video_cluster"
        and summary.get("bootstrap", {}).get("replicates") == BOOTSTRAP_REPLICATES
        and summary.get("bootstrap", {}).get("seed") == BOOTSTRAP_SEED,
        "alignment summary/bootstrap contract drift",
    )
    _require(payload.get("hard_swap_count") == 576, "hard-swap count drift")
    signed = payload.get("signed_utility_alignment")
    frozen = payload.get("frozen_detector_contract")
    _require(
        isinstance(signed, Mapping)
        and canonical_sha256(signed) == payload.get("signed_utility_alignment_sha256")
        and signed.get("sign_agreement") == summary["sign_agreement"]
        and signed.get("window_spearman_median")
        == summary["window_spearman_median"]
        and signed.get("bootstrap_unit") == "video_cluster"
        and signed.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
        and signed.get("bootstrap_seed") == BOOTSTRAP_SEED,
        "signed utility alignment contract drift",
    )
    _require(
        isinstance(frozen, Mapping)
        and frozen.get("backend") == "official_adatad_actionformer"
        and frozen.get("checkpoint") == context["selected_g0"]["checkpoint"]
        and frozen.get("optimizer_steps") == 0
        and frozen.get("parameter_version_unchanged") is True
        and frozen.get("loss_normalizer_frozen") is True,
        "frozen official AdaTAD contract drift",
    )
    for stratum, binding in payload.get("shards", {}).items():
        _require(stratum in DURATION_STRATA and isinstance(binding, Mapping), "shard binding drift")
        _require_file(binding.get("path"), binding.get("sha256"), f"{stratum} shard")
    gates = payload.get("full_model_gates")
    _require(isinstance(gates, Mapping) and set(gates) == set(context["authorized_variants"]), "G1/G2 gate binding drift")
    for variant, record in gates.items():
        _validate_full_gate_record(
            record=record,
            variant=variant,
            config=context["authorized_variants"][variant],
            context=context,
        )
    return dict(payload)


def validate_alignment_artifact(
    *,
    path: str | Path,
    digest: str,
    expected_commit: str,
    expected_variant: str | None = None,
    source_config_path: str | Path | None = None,
    source_config_sha256: str | None = None,
) -> dict[str, Any]:
    payload, resolved = _load_json(path, digest, "hard-swap alignment artifact")
    validate_alignment_artifact_payload(payload, expected_commit=expected_commit)
    if expected_variant is not None:
        authorized = payload["authorized_variants"]
        _require(expected_variant in authorized, f"artifact does not authorize {expected_variant}")
        record = authorized[expected_variant]
        if source_config_path is not None:
            _require(
                _resolve(source_config_path) == _resolve(record["config_path"]),
                "runtime G1/G2 config path drift",
            )
        if source_config_sha256 is not None:
            _require(
                source_config_sha256 == record["config_sha256"],
                "runtime G1/G2 config hash drift",
            )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "self_sha256": payload["alignment_sha256"],
        "context_sha256": payload["context"]["sha256"],
        "selected_weakest_projected_family": payload[
            "selected_weakest_projected_family"
        ],
        "selected_g0_checkpoint_sha256": payload["selected_g0"]["checkpoint"][
            "sha256"
        ],
        "terminal_suite_sha256": payload["terminal_suite"]["sha256"],
        "full_model_gate": (
            None
            if expected_variant is None
            else dict(payload["full_model_gates"][expected_variant])
        ),
    }


def run_shard_from_context(
    *,
    stratum: str,
    expected_commit: str,
    context_path: str | Path,
    context_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    context, resolved = load_alignment_context(
        path=context_path, digest=context_sha256, expected_commit=expected_commit
    )
    from tools.bata.run_duca_protected_physical_p3_shard import run_shard

    return run_shard(
        stratum=stratum,
        config_path=context["alignment_model"]["config_path"],
        expected_commit=expected_commit,
        protocol_manifest="",
        protocol_manifest_sha256="",
        adatad_pretrain=context["adatad_pretrain"]["path"],
        adatad_pretrain_sha256=context["adatad_pretrain"]["sha256"],
        output_json=str(output_path),
        alignment_context=context,
        alignment_context_path=str(resolved),
        alignment_context_sha256=context_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and seal DUCA boundary-burst legal hard-swap alignment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-context")
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--terminal-suite", required=True)
    prepare.add_argument("--terminal-suite-sha256", required=True)
    prepare.add_argument("--population-config", default=POPULATION_CONFIG)
    prepare.add_argument("--output-json", required=True)

    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--stratum", choices=DURATION_STRATA, required=True)
    shard.add_argument("--expected-commit", required=True)
    shard.add_argument("--context", required=True)
    shard.add_argument("--context-sha256", required=True)
    shard.add_argument("--output-json", required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--expected-commit", required=True)
    aggregate.add_argument("--context", required=True)
    aggregate.add_argument("--context-sha256", required=True)
    for stratum in DURATION_STRATA:
        aggregate.add_argument(f"--{stratum}-shard", required=True)
        aggregate.add_argument(f"--{stratum}-shard-sha256", required=True)
    aggregate.add_argument("--g1-variant", required=True)
    aggregate.add_argument("--g1-gate", required=True)
    aggregate.add_argument("--g1-gate-sha256", required=True)
    aggregate.add_argument("--g2-variant", required=True)
    aggregate.add_argument("--g2-gate", required=True)
    aggregate.add_argument("--g2-gate-sha256", required=True)
    aggregate.add_argument("--output-json", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--expected-commit", required=True)
    validate.add_argument("--artifact", required=True)
    validate.add_argument("--artifact-sha256", required=True)
    validate.add_argument("--variant")

    args = parser.parse_args(argv)
    if args.command == "prepare-context":
        prepare_context(
            expected_commit=args.expected_commit,
            terminal_suite_path=args.terminal_suite,
            terminal_suite_sha256=args.terminal_suite_sha256,
            population_config_path=args.population_config,
            output_path=args.output_json,
        )
    elif args.command == "run-shard":
        run_shard_from_context(
            stratum=args.stratum,
            expected_commit=args.expected_commit,
            context_path=args.context,
            context_sha256=args.context_sha256,
            output_path=args.output_json,
        )
    elif args.command == "aggregate":
        shard_paths = {
            stratum: getattr(args, f"{stratum}_shard") for stratum in DURATION_STRATA
        }
        shard_sha256s = {
            stratum: getattr(args, f"{stratum}_shard_sha256")
            for stratum in DURATION_STRATA
        }
        aggregate_alignment(
            expected_commit=args.expected_commit,
            context_path=args.context,
            context_sha256=args.context_sha256,
            shard_paths=shard_paths,
            shard_sha256s=shard_sha256s,
            full_model_gate_paths={args.g1_variant: args.g1_gate, args.g2_variant: args.g2_gate},
            full_model_gate_sha256s={
                args.g1_variant: args.g1_gate_sha256,
                args.g2_variant: args.g2_gate_sha256,
            },
            output_path=args.output_json,
        )
    else:
        binding = validate_alignment_artifact(
            path=args.artifact,
            digest=args.artifact_sha256,
            expected_commit=args.expected_commit,
            expected_variant=args.variant,
        )
        print(json.dumps(binding, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
