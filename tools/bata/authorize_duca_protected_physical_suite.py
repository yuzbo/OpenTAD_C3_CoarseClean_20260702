from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    sha256_file,
)


SCHEMA = "duca_protected_physical_authorization_v1"
PROTOCOL_SCHEMA = "duca_protected_physical_protocol_manifest_v1"
FULL_MODEL_GATE_SCHEMA = "duca_protected_physical_full_model_gate_v1"
P3_AGGREGATE_SCHEMA = "duca_protected_physical_p3_aggregate_v1"
MAIN_ARM = "protected_e2e"
BRIDGE025_ARM = "protected_e2e_bridge025"
HOMOTOPY_ARM = "protected_e2e_homotopy025"
UNI_COMPANION_ARM = "protected_e2e_uni_companion"
RHO_ARM = "protected_e2e_rho001"


class ProtectedPhysicalAuthorizationFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtectedPhysicalAuthorizationFailure(
            f"protected physical suite authorization failed: {message}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"{label} must be an exact SHA256",
    )
    return digest


def _require_git_object(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", digest) is not None,
        f"{label} must be an exact 40-character Git object",
    )
    return digest


def _require_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    schema: str,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    expected = _require_sha256(expected_sha256, f"{label} expected hash")
    actual = sha256_file(resolved)
    _require(actual == expected, f"{label} SHA256 mismatch")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtectedPhysicalAuthorizationFailure(
            f"{label} is not valid UTF-8 JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    _require(payload.get("schema") == schema, f"{label} schema mismatch")
    _require(payload.get("ok") is True, f"{label} did not pass")
    return payload, resolved, actual


def _validate_protocol(
    protocol: Mapping[str, Any],
) -> tuple[str, str, str, Mapping[str, Any], str, Mapping[str, Any]]:
    _require(
        protocol.get("paper_claim_allowed") is False,
        "P0 weakens the paper-claim contract",
    )
    commit = _require_git_object(protocol.get("git_commit"), "P0 git_commit")
    tree = _require_git_object(protocol.get("git_tree"), "P0 git_tree")
    content_hash = _require_sha256(
        protocol.get("manifest_content_sha256"),
        "P0 manifest_content_sha256",
    )
    unsigned = dict(protocol)
    unsigned.pop("manifest_content_sha256", None)
    _require(
        canonical_sha256(unsigned) == content_hash,
        "P0 internal content hash mismatch",
    )
    configs = _require_mapping(protocol.get("configs"), "P0 configs")
    arms = _require_mapping(configs.get("arms"), "P0 configs.arms")
    _require(
        all(
            arm in arms
            for arm in (
                MAIN_ARM,
                BRIDGE025_ARM,
                HOMOTOPY_ARM,
                UNI_COMPANION_ARM,
                RHO_ARM,
            )
        ),
        "P0 lacks a required protected/optimization config",
    )
    expected_updates = _require_int(
        protocol.get("expected_successful_optimizer_updates_per_arm"),
        "P0 expected successful optimizer updates",
    )
    homotopy_arm = _require_mapping(
        arms.get(HOMOTOPY_ARM),
        f"P0 {HOMOTOPY_ARM} config",
    )
    _require(
        _require_int(
            homotopy_arm.get("homotopy_total_steps"),
            "P0 homotopy total steps",
        )
        == expected_updates,
        "P0 homotopy schedule differs from optimizer exposure",
    )
    frozen_method = _require_mapping(
        protocol.get("frozen_method"),
        "P0 frozen method",
    )
    homotopy_contract = _require_mapping(
        frozen_method.get("homotopy"),
        "P0 frozen homotopy contract",
    )
    _require(
        homotopy_contract.get("arm") == HOMOTOPY_ARM
        and float(homotopy_contract.get("warmup_fraction", -1.0)) == 0.05
        and float(homotopy_contract.get("transition_fraction", -1.0)) == 0.30
        and homotopy_contract.get("transition_shape") == "cosine"
        and homotopy_contract.get("alpha_zero_contract")
        == "hard_forward_exact_uniform"
        and _require_int(
            homotopy_contract.get("total_successful_updates"),
            "P0 frozen homotopy total updates",
        )
        == expected_updates
        and float(homotopy_contract.get("inference_alpha", -1.0)) == 1.0,
        "P0 frozen homotopy schedule semantics drift",
    )
    pretrain = _require_mapping(
        protocol.get("videomae_pretrain"),
        "P0 videomae_pretrain",
    )
    pretrain_hash = _require_sha256(
        pretrain.get("sha256"),
        "P0 VideoMAE pretrain hash",
    )
    p3_population = _require_mapping(
        protocol.get("p3_population"),
        "P0 p3_population",
    )
    _require(
        _require_int(p3_population.get("window_count"), "P0 P3 window_count") == 48,
        "P0 P3 window count drift",
    )
    _require(
        _require_int(
            p3_population.get("swaps_per_window"),
            "P0 P3 swaps_per_window",
        )
        == 12,
        "P0 P3 swaps-per-window drift",
    )
    _require(
        _require_int(
            p3_population.get("preregistered_swap_count"),
            "P0 P3 preregistered_swap_count",
        )
        == 576,
        "P0 P3 swap count drift",
    )
    return commit, tree, content_hash, arms, pretrain_hash, homotopy_contract


def _validate_full_model_gate(
    gate: Mapping[str, Any],
    *,
    label: str,
    expected_arm: str,
    expected_commit: str,
    expected_tree: str,
    protocol_manifest_sha256: str,
    protocol_content_sha256: str,
    expected_config_sha256: str,
    expected_pretrain_sha256: str,
    expected_homotopy_contract: Mapping[str, Any] | None = None,
) -> None:
    _require(
        gate.get("status") == "p1_p2_full_model_gate_passed",
        f"{label} status mismatch",
    )
    _require(
        gate.get("paper_claim_allowed") is False,
        f"{label} weakens the paper-claim contract",
    )
    runtime = _require_mapping(gate.get("runtime"), f"{label}.runtime")
    _require(
        _require_git_object(runtime.get("git_commit"), f"{label} commit")
        == expected_commit,
        f"{label} commit differs from P0",
    )
    _require(
        _require_git_object(runtime.get("git_tree"), f"{label} tree") == expected_tree,
        f"{label} tree differs from P0",
    )
    protocol_binding = _require_mapping(
        gate.get("protocol_manifest"),
        f"{label}.protocol_manifest",
    )
    _require(
        _require_sha256(
            protocol_binding.get("sha256"),
            f"{label} P0 file hash",
        )
        == protocol_manifest_sha256,
        f"{label} is bound to another P0 manifest",
    )
    _require(
        _require_sha256(
            protocol_binding.get("content_sha256"),
            f"{label} P0 content hash",
        )
        == protocol_content_sha256,
        f"{label} P0 content binding mismatch",
    )
    config = _require_mapping(gate.get("config"), f"{label}.config")
    _require(
        config.get("arm") == expected_arm,
        f"{label} arm must be {expected_arm}",
    )
    _require(
        _require_sha256(config.get("sha256"), f"{label} config hash")
        == expected_config_sha256,
        f"{label} config differs from P0",
    )
    pretrain = _require_mapping(
        gate.get("adatad_pretrain"),
        f"{label}.adatad_pretrain",
    )
    _require(
        _require_sha256(pretrain.get("sha256"), f"{label} pretrain hash")
        == expected_pretrain_sha256,
        f"{label} pretrain differs from P0",
    )
    _require(
        gate.get("hard_forward_equals_real_backbone_input") is True,
        f"{label} lacks hard-forward equality",
    )
    _require(
        gate.get("optimizer_exact_coverage") is True,
        f"{label} lacks exact optimizer coverage",
    )
    parity = _require_mapping(
        gate.get("exact_uniform_physical_legacy_parity"),
        f"{label}.exact_uniform_physical_legacy_parity",
    )
    _require(
        parity.get("target_and_decode_parity") is True
        and parity.get("target_assignment_parity") is True
        and parity.get("decode_parity") is True,
        f"{label} lacks exact-uniform target/decode parity",
    )
    for window_key in ("full_window", "short_padded_window"):
        window = _require_mapping(
            parity.get(window_key),
            f"{label}.exact_uniform_physical_legacy_parity.{window_key}",
        )
        assignment = _require_mapping(
            window.get("target_assignment"),
            f"{label}.{window_key}.target_assignment",
        )
        _require(
            window.get("target_assignment_parity") is True
            and window.get("decode_parity") is True
            and assignment.get("classification_targets_equal") is True
            and assignment.get("positive_masks_equal") is True
            and assignment.get("physical_regression_targets_equal") is True,
            f"{label} lacks explicit {window_key} assignment/decode parity",
        )
    padded = _require_mapping(
        gate.get("padded_real_window_audit"),
        f"{label}.padded_real_window_audit",
    )
    padded_valid_len = _require_int(
        padded.get("valid_len"),
        f"{label} padded valid_len",
    )
    _require(
        padded.get("hard_forward_equal") is True
        and padded.get("tail_padding_mode") == "replicate_last_selected"
        and padded.get("tail_padding_reference_equal") is True
        and padded_valid_len < 384
        and _require_int(
            padded.get("effective_k"),
            f"{label} padded effective_k",
        )
        == min(384, padded_valid_len),
        f"{label} lacks padded-window K_eff coverage",
    )
    update = _require_mapping(
        gate.get("real_optimizer_step_audit"),
        f"{label}.real_optimizer_step_audit",
    )
    _require(
        _require_int(
            update.get("successful_optimizer_updates"),
            f"{label} successful optimizer updates",
        )
        == 3
        and update.get("scheduler_and_ema_updated") is True
        and update.get("full_batch_update") is True
        and update.get("padded_batch_update") is True
        and update.get("short_padded_batch_update") is True
        and update.get("successful_batch_updates")
        == ["full", "padded", "short_padded"],
        f"{label} lacks real optimizer/scheduler/EMA updates",
    )
    companion = _require_mapping(
        gate.get("training_companion_audit"),
        f"{label}.training_companion_audit",
    )
    expected_bridge_scale = {
        MAIN_ARM: 1.0,
        BRIDGE025_ARM: 0.25,
        HOMOTOPY_ARM: 0.25,
        UNI_COMPANION_ARM: 0.25,
        RHO_ARM: 1.0,
    }[expected_arm]
    _require(
        _require_int(
            companion.get("detector_forward_count"),
            f"{label} detector forward count",
        )
        == 1
        and float(companion.get("detector_bridge_gradient_scale", -1.0))
        == expected_bridge_scale,
        f"{label} bridge scale or detector-forward contract drift",
    )
    if expected_arm == UNI_COMPANION_ARM:
        _require(
            companion.get("training_only") is True
            and _require_int(
                companion.get("uniform_companion_count"),
                f"{label} uniform companion count",
            )
            == 1
            and _require_int(
                companion.get("learned_detector_count"),
                f"{label} learned detector count",
            )
            == 1,
            f"{label} lacks one-pass learned/uniform companion routing",
        )
    else:
        _require(
            companion.get("training_only") is False
            and _require_int(
                companion.get("uniform_companion_count"),
                f"{label} uniform companion count",
            )
            == 0,
            f"{label} unexpectedly enables the Uni companion route",
        )
    homotopy = _require_mapping(
        gate.get("policy_homotopy_audit"),
        f"{label}.policy_homotopy_audit",
    )
    if expected_arm == HOMOTOPY_ARM:
        _require(
            isinstance(expected_homotopy_contract, Mapping),
            f"{label} lacks its P0 homotopy binding",
        )
        total_steps = _require_int(
            expected_homotopy_contract.get("total_successful_updates"),
            f"{label} P0 homotopy total steps",
        )
        expected_warmup = round(
            total_steps * float(expected_homotopy_contract["warmup_fraction"])
        )
        expected_transition = round(
            total_steps * float(expected_homotopy_contract["transition_fraction"])
        )
        _require(
            homotopy.get("enabled") is True
            and homotopy.get("alpha_zero_contract")
            == "hard_forward_exact_uniform"
            and homotopy.get("alpha_one_equals_direct_learned_potential") is True
            and homotopy.get("inference_forces_alpha_one") is True
            and _require_int(homotopy.get("total_steps"), f"{label} total steps")
            == total_steps
            and _require_int(homotopy.get("warmup_steps"), f"{label} warmup steps")
            == expected_warmup
            and _require_int(
                homotopy.get("transition_steps"), f"{label} transition steps"
            )
            == expected_transition
            and float(homotopy.get("gradient_audit_alpha", -1.0)) == 0.5,
            f"{label} lacks the homotopy endpoint contract",
        )
        rows = homotopy.get("alpha_zero_exact_uniform_rows")
        _require(
            isinstance(rows, list)
            and len(rows) >= 2
            and all(row.get("exact_uniform_equal") is True for row in rows),
            f"{label} lacks full/short alpha-zero uniform parity",
        )
        _require(
            update.get("selector_schedule_enabled") is True
            and _require_int(
                update.get("selector_schedule_step"),
                f"{label} selector schedule step",
            )
            == _require_int(
                update.get("initial_selector_schedule_step"),
                f"{label} initial selector schedule step",
            )
            + 3
            and _require_int(
                update.get("ema_selector_schedule_step"),
                f"{label} EMA selector schedule step",
            )
            == _require_int(
                update.get("selector_schedule_step"),
                f"{label} final selector schedule step",
            ),
            f"{label} lacks successful-update/EMA schedule parity",
        )
    else:
        _require(
            homotopy.get("enabled") is False
            and update.get("selector_schedule_enabled") is False
            and _require_int(
                update.get("selector_schedule_step"),
                f"{label} selector schedule step",
            )
            == 0,
            f"{label} unexpectedly enables a selector schedule",
        )


def _validate_p3_aggregate(
    p3: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_tree: str,
    protocol_manifest_sha256: str,
    protocol_content_sha256: str,
    expected_config_sha256: str,
    expected_pretrain_sha256: str,
) -> None:
    _require(
        p3.get("status") == "p3_aggregate_passed",
        "P3 aggregate status mismatch",
    )
    _require(
        p3.get("paper_claim_allowed") is False,
        "P3 aggregate weakens the paper-claim contract",
    )
    _require(
        _require_git_object(p3.get("git_commit"), "P3 git_commit") == expected_commit,
        "P3 commit differs from P0",
    )
    _require(
        _require_git_object(p3.get("git_tree"), "P3 git_tree") == expected_tree,
        "P3 tree differs from P0",
    )
    _require(
        _require_sha256(
            p3.get("protocol_manifest_sha256"),
            "P3 protocol_manifest_sha256",
        )
        == protocol_manifest_sha256,
        "P3 is bound to another P0 manifest",
    )
    _require(
        _require_sha256(
            p3.get("protocol_manifest_content_sha256"),
            "P3 protocol_manifest_content_sha256",
        )
        == protocol_content_sha256,
        "P3 P0 content binding mismatch",
    )
    _require(
        _require_sha256(p3.get("config_sha256"), "P3 config hash")
        == expected_config_sha256,
        "P3 config differs from P0",
    )
    _require(
        _require_sha256(p3.get("pretrain_sha256"), "P3 pretrain hash")
        == expected_pretrain_sha256,
        "P3 pretrain differs from P0",
    )
    _require(
        _require_int(p3.get("window_count"), "P3 window_count") == 48,
        "P3 window count drift",
    )
    _require(
        _require_int(p3.get("swap_count"), "P3 swap_count") == 576,
        "P3 swap count drift",
    )
    _require(
        p3.get("strata") == ["short", "medium", "long"],
        "P3 strata drift",
    )
    input_hashes = _require_mapping(p3.get("input_hashes"), "P3 input_hashes")
    _require(
        set(input_hashes) == {"short", "medium", "long"},
        "P3 input hash set is incomplete",
    )
    for stratum, digest in input_hashes.items():
        _require_sha256(digest, f"P3 {stratum} input hash")
    aggregate = _require_mapping(p3.get("aggregate"), "P3 aggregate result")
    _require(
        _require_sha256(
            p3.get("aggregate_content_sha256"),
            "P3 aggregate content hash",
        )
        == canonical_sha256(dict(aggregate)),
        "P3 nested aggregate content hash mismatch",
    )
    _require(
        aggregate.get("schema") == P3_AGGREGATE_SCHEMA and aggregate.get("ok") is True,
        "nested aggregate_p3_rows result did not pass",
    )
    bootstrap = _require_mapping(
        aggregate.get("bootstrap"),
        "P3 aggregate bootstrap",
    )
    _require(
        _require_int(bootstrap.get("replicates"), "P3 bootstrap replicates") == 2000
        and _require_int(bootstrap.get("seed"), "P3 bootstrap seed") == 20260720,
        "P3 bootstrap contract drift",
    )
    _require(
        _require_int(
            aggregate.get("preregistered_count"),
            "P3 aggregate preregistered_count",
        )
        == 576,
        "P3 nested preregistered count drift",
    )
    checks = _require_mapping(aggregate.get("checks"), "P3 aggregate checks")
    _require(
        bool(checks) and all(value is True for value in checks.values()),
        "P3 aggregate contains a failed preregistered check",
    )


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("x", encoding="utf-8") as handle:
            created = True
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created and target.exists():
            target.unlink()
        raise
    return target


def authorize_suite(
    *,
    protocol_manifest: str | Path,
    protocol_manifest_sha256: str,
    main_gate: str | Path,
    main_gate_sha256: str,
    bridge025_gate: str | Path,
    bridge025_gate_sha256: str,
    homotopy_gate: str | Path,
    homotopy_gate_sha256: str,
    uni_companion_gate: str | Path,
    uni_companion_gate_sha256: str,
    rho_gate: str | Path,
    rho_gate_sha256: str,
    p3_aggregate: str | Path,
    p3_aggregate_sha256: str,
    output_json: str | Path,
) -> dict[str, Any]:
    protocol, protocol_path, protocol_file_hash = _load_bound_json(
        protocol_manifest,
        protocol_manifest_sha256,
        label="P0 protocol manifest",
        schema=PROTOCOL_SCHEMA,
    )
    (
        commit,
        tree,
        content_hash,
        arms,
        pretrain_hash,
        homotopy_contract,
    ) = _validate_protocol(protocol)
    main, main_path, main_hash = _load_bound_json(
        main_gate,
        main_gate_sha256,
        label="main full-model gate",
        schema=FULL_MODEL_GATE_SCHEMA,
    )
    bridge025, bridge025_path, bridge025_hash = _load_bound_json(
        bridge025_gate,
        bridge025_gate_sha256,
        label="bridge025 full-model gate",
        schema=FULL_MODEL_GATE_SCHEMA,
    )
    homotopy, homotopy_path, homotopy_hash = _load_bound_json(
        homotopy_gate,
        homotopy_gate_sha256,
        label="homotopy full-model gate",
        schema=FULL_MODEL_GATE_SCHEMA,
    )
    companion, companion_path, companion_hash = _load_bound_json(
        uni_companion_gate,
        uni_companion_gate_sha256,
        label="Uni companion full-model gate",
        schema=FULL_MODEL_GATE_SCHEMA,
    )
    rho, rho_path, rho_hash = _load_bound_json(
        rho_gate,
        rho_gate_sha256,
        label="rho full-model gate",
        schema=FULL_MODEL_GATE_SCHEMA,
    )
    p3, p3_path, p3_hash = _load_bound_json(
        p3_aggregate,
        p3_aggregate_sha256,
        label="P3 aggregate",
        schema=P3_AGGREGATE_SCHEMA,
    )

    main_config = _require_mapping(arms.get(MAIN_ARM), f"P0 {MAIN_ARM} config")
    bridge025_config = _require_mapping(
        arms.get(BRIDGE025_ARM),
        f"P0 {BRIDGE025_ARM} config",
    )
    homotopy_config = _require_mapping(
        arms.get(HOMOTOPY_ARM),
        f"P0 {HOMOTOPY_ARM} config",
    )
    companion_config = _require_mapping(
        arms.get(UNI_COMPANION_ARM),
        f"P0 {UNI_COMPANION_ARM} config",
    )
    rho_config = _require_mapping(arms.get(RHO_ARM), f"P0 {RHO_ARM} config")
    main_config_hash = _require_sha256(
        main_config.get("source_sha256"),
        f"P0 {MAIN_ARM} config hash",
    )
    bridge025_config_hash = _require_sha256(
        bridge025_config.get("source_sha256"),
        f"P0 {BRIDGE025_ARM} config hash",
    )
    homotopy_config_hash = _require_sha256(
        homotopy_config.get("source_sha256"),
        f"P0 {HOMOTOPY_ARM} config hash",
    )
    companion_config_hash = _require_sha256(
        companion_config.get("source_sha256"),
        f"P0 {UNI_COMPANION_ARM} config hash",
    )
    rho_config_hash = _require_sha256(
        rho_config.get("source_sha256"),
        f"P0 {RHO_ARM} config hash",
    )
    p3_population = _require_mapping(
        protocol.get("p3_population"),
        "P0 p3_population",
    )
    p3_config_hash = _require_sha256(
        p3_population.get("config_sha256"),
        "P0 P3 config hash",
    )

    _validate_full_model_gate(
        main,
        label="main full-model gate",
        expected_arm=MAIN_ARM,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=main_config_hash,
        expected_pretrain_sha256=pretrain_hash,
    )
    _validate_full_model_gate(
        bridge025,
        label="bridge025 full-model gate",
        expected_arm=BRIDGE025_ARM,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=bridge025_config_hash,
        expected_pretrain_sha256=pretrain_hash,
    )
    _validate_full_model_gate(
        homotopy,
        label="homotopy full-model gate",
        expected_arm=HOMOTOPY_ARM,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=homotopy_config_hash,
        expected_pretrain_sha256=pretrain_hash,
        expected_homotopy_contract=homotopy_contract,
    )
    _validate_full_model_gate(
        companion,
        label="Uni companion full-model gate",
        expected_arm=UNI_COMPANION_ARM,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=companion_config_hash,
        expected_pretrain_sha256=pretrain_hash,
    )
    _validate_full_model_gate(
        rho,
        label="rho full-model gate",
        expected_arm=RHO_ARM,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=rho_config_hash,
        expected_pretrain_sha256=pretrain_hash,
    )
    _validate_p3_aggregate(
        p3,
        expected_commit=commit,
        expected_tree=tree,
        protocol_manifest_sha256=protocol_file_hash,
        protocol_content_sha256=content_hash,
        expected_config_sha256=p3_config_hash,
        expected_pretrain_sha256=pretrain_hash,
    )

    result = {
        "schema": SCHEMA,
        "ok": True,
        "status": "p0_p1_p2_p3_authorized_for_official60_training",
        "git_commit": commit,
        "git_tree": tree,
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_sha256": protocol_file_hash,
        "protocol_manifest_content_sha256": content_hash,
        "pretrain_sha256": pretrain_hash,
        "config_hashes": {
            MAIN_ARM: main_config_hash,
            BRIDGE025_ARM: bridge025_config_hash,
            HOMOTOPY_ARM: homotopy_config_hash,
            UNI_COMPANION_ARM: companion_config_hash,
            RHO_ARM: rho_config_hash,
            "p3": p3_config_hash,
        },
        "input_paths": {
            "protocol_manifest": str(protocol_path),
            "main_full_model_gate": str(main_path),
            "bridge025_full_model_gate": str(bridge025_path),
            "homotopy_full_model_gate": str(homotopy_path),
            "uni_companion_full_model_gate": str(companion_path),
            "rho_full_model_gate": str(rho_path),
            "p3_aggregate": str(p3_path),
        },
        "input_hashes": {
            "protocol_manifest": protocol_file_hash,
            "main_full_model_gate": main_hash,
            "bridge025_full_model_gate": bridge025_hash,
            "homotopy_full_model_gate": homotopy_hash,
            "uni_companion_full_model_gate": companion_hash,
            "rho_full_model_gate": rho_hash,
            "p3_aggregate": p3_hash,
        },
        "authorized_scope": {
            "official60_four_arm_training": True,
            "official60_uni_companion_training": True,
            "official60_homotopy_training": True,
            "paper_claim": False,
        },
        "paper_claim_allowed": False,
    }
    _write_json_exclusive(output_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorize the frozen DUCA protected-physical training suite."
    )
    parser.add_argument(
        "--protocol-manifest",
        "--p0-manifest",
        dest="protocol_manifest",
        required=True,
    )
    parser.add_argument(
        "--protocol-manifest-sha256",
        "--p0-manifest-sha256",
        dest="protocol_manifest_sha256",
        required=True,
    )
    parser.add_argument(
        "--main-gate",
        "--main-full-model-gate",
        dest="main_gate",
        required=True,
    )
    parser.add_argument(
        "--main-gate-sha256",
        "--main-full-model-gate-sha256",
        dest="main_gate_sha256",
        required=True,
    )
    parser.add_argument(
        "--bridge025-gate",
        "--bridge025-full-model-gate",
        dest="bridge025_gate",
        required=True,
    )
    parser.add_argument(
        "--bridge025-gate-sha256",
        "--bridge025-full-model-gate-sha256",
        dest="bridge025_gate_sha256",
        required=True,
    )
    parser.add_argument(
        "--homotopy-gate",
        "--homotopy-full-model-gate",
        dest="homotopy_gate",
        required=True,
    )
    parser.add_argument(
        "--homotopy-gate-sha256",
        "--homotopy-full-model-gate-sha256",
        dest="homotopy_gate_sha256",
        required=True,
    )
    parser.add_argument(
        "--uni-companion-gate",
        "--uni-companion-full-model-gate",
        dest="uni_companion_gate",
        required=True,
    )
    parser.add_argument(
        "--uni-companion-gate-sha256",
        "--uni-companion-full-model-gate-sha256",
        dest="uni_companion_gate_sha256",
        required=True,
    )
    parser.add_argument(
        "--rho-gate",
        "--rho-full-model-gate",
        dest="rho_gate",
        required=True,
    )
    parser.add_argument(
        "--rho-gate-sha256",
        "--rho-full-model-gate-sha256",
        dest="rho_gate_sha256",
        required=True,
    )
    parser.add_argument(
        "--p3-aggregate",
        "--p3-aggregate-json",
        dest="p3_aggregate",
        required=True,
    )
    parser.add_argument("--p3-aggregate-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        result = authorize_suite(
            protocol_manifest=args.protocol_manifest,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
            main_gate=args.main_gate,
            main_gate_sha256=args.main_gate_sha256,
            bridge025_gate=args.bridge025_gate,
            bridge025_gate_sha256=args.bridge025_gate_sha256,
            homotopy_gate=args.homotopy_gate,
            homotopy_gate_sha256=args.homotopy_gate_sha256,
            uni_companion_gate=args.uni_companion_gate,
            uni_companion_gate_sha256=args.uni_companion_gate_sha256,
            rho_gate=args.rho_gate,
            rho_gate_sha256=args.rho_gate_sha256,
            p3_aggregate=args.p3_aggregate,
            p3_aggregate_sha256=args.p3_aggregate_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "status": "p0_p3_authorization_failed",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "paper_claim_allowed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
