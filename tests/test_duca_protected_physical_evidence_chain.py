from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata import aggregate_duca_protected_physical_p3 as p3_aggregator
from tools.bata.authorize_duca_protected_physical_suite import (
    authorize_suite,
)


COMMIT = "a" * 40
GIT_TREE = "b" * 40
PRETRAIN_SHA256 = "d" * 64
P3_CONFIG_SHA256 = "e" * 64
ARM_CONFIG_SHA256 = {
    "protected_e2e": "1" * 64,
    "protected_e2e_bridge025": "2" * 64,
    "protected_e2e_homotopy025": "5" * 64,
    "protected_e2e_uni_companion": "3" * 64,
    "protected_e2e_rho001": "4" * 64,
}
STRATA = ("short", "medium", "long")


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _p3_report(ok: bool = True):
    return {
        "schema": "duca_protected_physical_p3_aggregate_v1",
        "ok": ok,
        "preregistered_count": 576,
        "bootstrap": {"replicates": 2000, "seed": 20260720},
        "checks": {"focused_test_gate": ok},
    }


@pytest.fixture(autouse=True)
def _stub_expensive_p3_statistics(monkeypatch):
    monkeypatch.setattr(
        p3_aggregator,
        "aggregate_p3_rows",
        lambda rows, **kwargs: _p3_report(True),
    )


def _window_rows(stratum: str, stratum_index: int):
    windows = []
    rows = []
    for local_index in range(16):
        global_index = stratum_index * 16 + local_index
        video_id = f"{stratum}_video_{local_index:02d}"
        window_start = global_index * 10
        windows.append(
            {
                "video_id": video_id,
                "window_start": window_start,
                "duration_stratum": stratum,
            }
        )
        for swap_index in range(12):
            predicted = float(swap_index - 5.5)
            removed = swap_index
            incoming = 100 + swap_index
            rows.append(
                {
                    "video_id": video_id,
                    "window_start": window_start,
                    "duration_stratum": stratum,
                    "window_kind": ("padded" if local_index < 4 else "full"),
                    "boundary_source": "original_uncropped_annotation",
                    "boundary_distance_stratum": ("near", "mid", "far")[swap_index % 3],
                    "removed": removed,
                    "incoming": incoming,
                    "sampling_sha256": hashlib.sha256(
                        (
                            f"{video_id}|{window_start}|{removed}|{incoming}|"
                            "20260720"
                        ).encode("utf-8")
                    ).hexdigest(),
                    "quartile": swap_index // 3,
                    "predicted_delta": predicted,
                    "actual_delta": predicted * 0.1,
                    "predicted_best_quartile": swap_index < 3,
                    "boundary_distance_gain_seconds": (0.5 if swap_index < 3 else -0.1),
                    "excluded_reason": None,
                    "physical_violation_count": 0,
                    "restoration_mismatch": False,
                    "repeated_base_loss_abs_error": 0.0,
                    "hard_forward_equal": True,
                }
            )
    return windows, rows


def _make_bundle(tmp_path: Path):
    shard_content = {}
    all_windows = []
    for stratum_index, stratum in enumerate(STRATA):
        windows, rows = _window_rows(stratum, stratum_index)
        shard_content[stratum] = {"windows": windows, "rows": rows}
        all_windows.extend(windows)

    protocol = {
        "schema": "duca_protected_physical_protocol_manifest_v1",
        "ok": True,
        "git_commit": COMMIT,
        "git_tree": GIT_TREE,
        "paper_claim_allowed": False,
        "configs": {
            "arms": {
                arm: {
                    "path": f"configs/{arm}.py",
                    "source_sha256": source_sha256,
                    "resolved_sha256": f"{index + 3}" * 64,
                    "homotopy_total_steps": (
                        6000 if arm == "protected_e2e_homotopy025" else 0
                    ),
                }
                for index, (arm, source_sha256) in enumerate(ARM_CONFIG_SHA256.items())
            }
        },
        "expected_successful_optimizer_updates_per_arm": 6000,
        "frozen_method": {
            "homotopy": {
                "arm": "protected_e2e_homotopy025",
                "warmup_fraction": 0.05,
                "transition_fraction": 0.30,
                "transition_shape": "cosine",
                "total_successful_updates": 6000,
                "alpha_zero_contract": "hard_forward_exact_uniform",
                "inference_alpha": 1.0,
            }
        },
        "p3_population": {
            "config_path": "configs/p3.py",
            "config_sha256": P3_CONFIG_SHA256,
            "windows": all_windows,
            "windows_sha256": _canonical_sha256(all_windows),
            "window_count": 48,
            "swaps_per_window": 12,
            "preregistered_swap_count": 576,
        },
        "videomae_pretrain": {"sha256": PRETRAIN_SHA256},
    }
    protocol["manifest_content_sha256"] = _canonical_sha256(protocol)
    protocol_path = _write_json(tmp_path / "p0.json", protocol)
    protocol_sha256 = _file_sha256(protocol_path)

    gate_paths = {}
    for arm, config_sha256 in ARM_CONFIG_SHA256.items():
        gate_paths[arm] = _write_json(
            tmp_path / f"{arm}_gate.json",
            {
                "schema": "duca_protected_physical_full_model_gate_v1",
                "ok": True,
                "status": "p1_p2_full_model_gate_passed",
                "runtime": {
                    "git_commit": COMMIT,
                    "git_tree": GIT_TREE,
                },
                "protocol_manifest": {
                    "sha256": protocol_sha256,
                    "content_sha256": protocol["manifest_content_sha256"],
                },
                "config": {
                    "arm": arm,
                    "sha256": config_sha256,
                },
                "adatad_pretrain": {"sha256": PRETRAIN_SHA256},
                "hard_forward_equals_real_backbone_input": True,
                "optimizer_exact_coverage": True,
                "exact_uniform_physical_legacy_parity": {
                    "full_window": {
                        "target_assignment": {
                            "classification_targets_equal": True,
                            "positive_masks_equal": True,
                            "physical_regression_targets_equal": True,
                        },
                        "target_assignment_parity": True,
                        "decode_parity": True,
                        "target_and_decode_parity": True,
                    },
                    "short_padded_window": {
                        "target_assignment": {
                            "classification_targets_equal": True,
                            "positive_masks_equal": True,
                            "physical_regression_targets_equal": True,
                        },
                        "target_assignment_parity": True,
                        "decode_parity": True,
                        "target_and_decode_parity": True,
                    },
                    "target_assignment_parity": True,
                    "decode_parity": True,
                    "target_and_decode_parity": True,
                },
                "padded_real_window_audit": {
                    "valid_len": 320,
                    "effective_k": 320,
                    "hard_forward_equal": True,
                    "tail_padding_mode": "replicate_last_selected",
                    "tail_padding_reference_equal": True,
                },
                "real_optimizer_step_audit": {
                    "successful_optimizer_updates": 3,
                    "successful_batch_updates": [
                        "full",
                        "padded",
                        "short_padded",
                    ],
                    "full_batch_update": True,
                    "padded_batch_update": True,
                    "short_padded_batch_update": True,
                    "scheduler_and_ema_updated": True,
                    "selector_schedule_enabled": (
                        arm == "protected_e2e_homotopy025"
                    ),
                    "initial_selector_schedule_step": (
                        1199 if arm == "protected_e2e_homotopy025" else 0
                    ),
                    "selector_schedule_step": (
                        1202 if arm == "protected_e2e_homotopy025" else 0
                    ),
                    "ema_selector_schedule_step": (
                        1202 if arm == "protected_e2e_homotopy025" else 0
                    ),
                },
                "training_companion_audit": {
                    "training_only": arm == "protected_e2e_uni_companion",
                    "detector_forward_count": 1,
                    "uniform_companion_count": (
                        1 if arm == "protected_e2e_uni_companion" else 0
                    ),
                    "learned_detector_count": (
                        1 if arm == "protected_e2e_uni_companion" else 0
                    ),
                    "detector_bridge_gradient_scale": (
                        0.25
                        if arm
                        in {
                            "protected_e2e_bridge025",
                            "protected_e2e_homotopy025",
                            "protected_e2e_uni_companion",
                        }
                        else 1.0
                    ),
                },
                "policy_homotopy_audit": (
                    {
                        "enabled": True,
                        "alpha_zero_contract": "hard_forward_exact_uniform",
                        "alpha_zero_exact_uniform_rows": [
                            {"window": "full", "exact_uniform_equal": True},
                            {
                                "window": "short_padded",
                                "exact_uniform_equal": True,
                            },
                        ],
                        "alpha_one_equals_direct_learned_potential": True,
                        "inference_forces_alpha_one": True,
                        "total_steps": 6000,
                        "warmup_steps": 300,
                        "transition_steps": 1800,
                        "gradient_audit_alpha": 0.5,
                    }
                    if arm == "protected_e2e_homotopy025"
                    else {"enabled": False}
                ),
                "paper_claim_allowed": False,
            },
        )

    shard_paths = {}
    for stratum in STRATA:
        content = shard_content[stratum]
        shard_paths[stratum] = _write_json(
            tmp_path / f"p3_{stratum}.json",
            {
                "schema": "duca_protected_physical_p3_shard_v1",
                "ok": True,
                "runtime": {
                    "git_commit": COMMIT,
                    "git_tree": GIT_TREE,
                },
                "stratum": stratum,
                "config_sha256": P3_CONFIG_SHA256,
                "protocol_manifest_path": str(protocol_path.resolve()),
                "protocol_manifest_sha256": protocol_sha256,
                "pretrain_sha256": PRETRAIN_SHA256,
                "optimizer_step": 0,
                "seed": 3407,
                "train_split_only": True,
                "test_loader_built": False,
                "checkpoint_written": False,
                "windows": content["windows"],
                "rows": content["rows"],
                "row_sha256": _canonical_sha256(content["rows"]),
                "paper_claim_allowed": False,
            },
        )

    return {
        "protocol": protocol_path,
        "protocol_sha256": protocol_sha256,
        "gates": gate_paths,
        "shards": shard_paths,
    }


def _aggregate(bundle, output: Path):
    return p3_aggregator.aggregate_p3_shards(
        short_json=bundle["shards"]["short"],
        short_sha256=_file_sha256(bundle["shards"]["short"]),
        medium_json=bundle["shards"]["medium"],
        medium_sha256=_file_sha256(bundle["shards"]["medium"]),
        long_json=bundle["shards"]["long"],
        long_sha256=_file_sha256(bundle["shards"]["long"]),
        output_json=output,
    )


def _authorize(bundle, aggregate: Path, output: Path):
    return authorize_suite(
        protocol_manifest=bundle["protocol"],
        protocol_manifest_sha256=bundle["protocol_sha256"],
        main_gate=bundle["gates"]["protected_e2e"],
        main_gate_sha256=_file_sha256(bundle["gates"]["protected_e2e"]),
        bridge025_gate=bundle["gates"]["protected_e2e_bridge025"],
        bridge025_gate_sha256=_file_sha256(bundle["gates"]["protected_e2e_bridge025"]),
        homotopy_gate=bundle["gates"]["protected_e2e_homotopy025"],
        homotopy_gate_sha256=_file_sha256(
            bundle["gates"]["protected_e2e_homotopy025"]
        ),
        uni_companion_gate=bundle["gates"]["protected_e2e_uni_companion"],
        uni_companion_gate_sha256=_file_sha256(
            bundle["gates"]["protected_e2e_uni_companion"]
        ),
        rho_gate=bundle["gates"]["protected_e2e_rho001"],
        rho_gate_sha256=_file_sha256(bundle["gates"]["protected_e2e_rho001"]),
        p3_aggregate=aggregate,
        p3_aggregate_sha256=_file_sha256(aggregate),
        output_json=output,
    )


def _rewrite(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def test_evidence_chain_happy_path_authorizes_official60(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    aggregate_path = tmp_path / "aggregate.json"
    authorization_path = tmp_path / "authorization.json"

    aggregate = _aggregate(bundle, aggregate_path)

    assert aggregate["schema"] == "duca_protected_physical_p3_aggregate_v1"
    assert aggregate["ok"] is True
    assert aggregate["git_commit"] == COMMIT
    assert aggregate["protocol_manifest_sha256"] == bundle["protocol_sha256"]
    assert aggregate["strata"] == list(STRATA)
    assert aggregate["window_count"] == 48
    assert aggregate["swap_count"] == 576
    assert aggregate["aggregate"]["ok"] is True

    receipt = _authorize(bundle, aggregate_path, authorization_path)

    assert receipt["schema"] == "duca_protected_physical_authorization_v1"
    assert receipt["ok"] is True
    assert receipt["git_commit"] == COMMIT
    assert receipt["protocol_manifest_sha256"] == bundle["protocol_sha256"]
    assert receipt["input_hashes"]["p3_aggregate"] == _file_sha256(aggregate_path)
    assert receipt["authorized_scope"] == {
        "official60_four_arm_training": True,
        "official60_uni_companion_training": True,
        "official60_homotopy_training": True,
        "paper_claim": False,
    }
    assert receipt["paper_claim_allowed"] is False


def test_aggregate_rejects_duplicate_stratum(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    bundle["shards"]["long"] = bundle["shards"]["medium"]

    with pytest.raises(RuntimeError, match="stratum"):
        _aggregate(bundle, tmp_path / "aggregate.json")


def test_aggregate_rejects_duplicate_window(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)

    def duplicate_window(payload):
        duplicate_key = {
            "video_id": payload["windows"][0]["video_id"],
            "window_start": payload["windows"][0]["window_start"],
        }
        payload["windows"][1].update(duplicate_key)
        for row in payload["rows"][12:24]:
            row.update(duplicate_key)
        payload["row_sha256"] = _canonical_sha256(payload["rows"])

    _rewrite(bundle["shards"]["medium"], duplicate_window)

    with pytest.raises(RuntimeError, match="window"):
        _aggregate(bundle, tmp_path / "aggregate.json")


def test_aggregate_rejects_duplicate_swap(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)

    def duplicate_swap(payload):
        for key in ("removed", "incoming", "sampling_sha256"):
            payload["rows"][1][key] = payload["rows"][0][key]
        payload["row_sha256"] = _canonical_sha256(payload["rows"])

    _rewrite(bundle["shards"]["short"], duplicate_swap)

    with pytest.raises(RuntimeError, match="swap"):
        _aggregate(bundle, tmp_path / "aggregate.json")


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("commit", "commit"),
        ("p0_hash", "P0"),
        ("config", "config"),
        ("pretrain", "pretrain"),
    ),
)
def test_aggregate_rejects_bound_evidence_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    bundle = _make_bundle(tmp_path)

    def mutate(payload):
        if drift == "commit":
            payload["runtime"]["git_commit"] = "9" * 40
        elif drift == "p0_hash":
            payload["protocol_manifest_sha256"] = "9" * 64
        elif drift == "config":
            payload["config_sha256"] = "9" * 64
        elif drift == "pretrain":
            payload["pretrain_sha256"] = "9" * 64

    _rewrite(bundle["shards"]["short"], mutate)

    with pytest.raises(RuntimeError, match=message):
        _aggregate(bundle, tmp_path / "aggregate.json")


def test_aggregate_rejects_failed_full_model_gate(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    aggregate_path = tmp_path / "aggregate.json"
    _aggregate(bundle, aggregate_path)
    _rewrite(
        bundle["gates"]["protected_e2e"],
        lambda payload: payload.update(ok=False),
    )

    with pytest.raises(RuntimeError, match="gate"):
        _authorize(bundle, aggregate_path, tmp_path / "authorization.json")


def test_authorizer_rejects_aggregate_loss_as_target_parity(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    aggregate_path = tmp_path / "aggregate.json"
    _aggregate(bundle, aggregate_path)

    def remove_explicit_targets(payload):
        parity = payload["exact_uniform_physical_legacy_parity"]
        parity["full_window"]["target_assignment"]["positive_masks_equal"] = False

    _rewrite(bundle["gates"]["protected_e2e"], remove_explicit_targets)
    with pytest.raises(RuntimeError, match="assignment/decode parity"):
        _authorize(bundle, aggregate_path, tmp_path / "authorization.json")


def test_authorizer_requires_partial_padded_real_optimizer_update(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    aggregate_path = tmp_path / "aggregate.json"
    _aggregate(bundle, aggregate_path)

    def remove_partial_update(payload):
        update = payload["real_optimizer_step_audit"]
        update["padded_batch_update"] = False
        update["successful_batch_updates"] = [
            "full",
            "short_padded",
            "short_padded",
        ]

    _rewrite(bundle["gates"]["protected_e2e"], remove_partial_update)
    with pytest.raises(RuntimeError, match="optimizer/scheduler/EMA"):
        _authorize(bundle, aggregate_path, tmp_path / "authorization.json")


def test_authorizer_rejects_failed_p3_aggregate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = _make_bundle(tmp_path)
    monkeypatch.setattr(
        p3_aggregator,
        "aggregate_p3_rows",
        lambda rows, **kwargs: _p3_report(False),
    )

    aggregate_path = tmp_path / "aggregate.json"
    aggregate = _aggregate(bundle, aggregate_path)
    assert aggregate["ok"] is False
    assert aggregate["aggregate"]["ok"] is False

    with pytest.raises(RuntimeError, match="[Pp]3"):
        _authorize(
            bundle,
            aggregate_path,
            tmp_path / "authorization.json",
        )
