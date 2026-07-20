from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from opentad.datasets import build_dataloader, build_dataset
from opentad.models.duca.acquisition import C3CoarseProbeActionnessSource
from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    derive_train_loader_contract,
    sha256_file,
)
from tools.bata.duca_protected_physical_p3 import stratified_window_manifest


SCHEMA = "duca_protected_physical_protocol_manifest_v1"
CONFIGS = {
    "exact_uniform": "configs/adatad/thumos/duca_protected_physical_exact_uniform_fixed384_official60.py",
    "transition_no_bridge": "configs/adatad/thumos/duca_protected_physical_transition_no_bridge_fixed384_official60.py",
    "protected_e2e": "configs/adatad/thumos/duca_protected_physical_e2e_fixed384_official60.py",
    "protected_e2e_rho001": "configs/adatad/thumos/duca_protected_physical_e2e_rho001_fixed384_official60.py",
}


class ProtocolFreezeFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolFreezeFailure(f"P0 protocol freeze failed: {message}")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _bind_git(expected_commit: str) -> dict[str, Any]:
    expected = str(expected_commit).lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected) is not None,
        "expected commit must be a full SHA",
    )
    _require(_git_output("rev-parse", "HEAD") == expected, "commit drift")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=normal"),
        "clean exact-commit tree required",
    )
    return {
        "git_commit": expected,
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
    }


def _normalized_config(cfg: Config) -> dict[str, Any]:
    payload = cfg.to_dict()
    payload.pop("work_dir", None)
    payload.pop("duca_variant_contract", None)
    selector = payload["model"]["frame_selector"]
    selector.pop("arm")
    source = selector.get("actionness_source_cfg")
    if isinstance(source, dict):
        source.pop("policy_hidden_gradient_scope", None)
    selector["actionness_source_cfg"] = "ARM_CONTROLLED_COARSE_SOURCE"
    return payload


def _config_evidence() -> tuple[dict[str, Config], dict[str, Any]]:
    configs = {
        arm: Config.fromfile(str(ROOT / relative_path))
        for arm, relative_path in CONFIGS.items()
    }
    normalized = {
        arm: _normalized_config(cfg) for arm, cfg in configs.items()
    }
    reference = normalized["protected_e2e"]
    _require(
        all(payload == reference for payload in normalized.values()),
        "four resolved configs differ outside the arm whitelist",
    )
    evidence = {}
    for arm, cfg in configs.items():
        relative_path = CONFIGS[arm]
        _require(cfg.model.frame_selector.arm == arm, f"{arm} arm drift")
        _require(cfg.dataset.val is None, f"{arm} exposes validation data")
        _require(
            bool(cfg.workflow.seal_eval_dataloaders_during_training),
            f"{arm} does not seal evaluation loaders",
        )
        evidence[arm] = {
            "path": relative_path,
            "source_sha256": sha256_file(ROOT / relative_path),
            "resolved_sha256": canonical_sha256(cfg.to_dict()),
        }
    return configs, {
        "arms": evidence,
        "normalized_shared_sha256": canonical_sha256(reference),
        "whitelisted_differences": [
            "model.frame_selector.arm",
            "model.frame_selector.actionness_source_cfg",
            "duca_variant_contract",
            "work_dir",
        ],
    }


def _official_asformer_evidence(cfg: Config) -> dict[str, Any]:
    source_cfg = copy.deepcopy(
        cfg.model.frame_selector.actionness_source_cfg.to_dict()
    )
    source_cfg.pop("type")
    source = C3CoarseProbeActionnessSource(**source_cfg)
    probe = source.probe
    official = dict(probe.official_source)
    source_file = Path(official["source_file"]).resolve()
    _require(source_file.is_file(), "official ASFormer source file is missing")
    return {
        "backend": source.probe_model,
        "official_backend": official["backend"],
        "source_file": str(source_file),
        "source_sha256": sha256_file(source_file),
        "source_normalized_lf_sha256": hashlib.sha256(
            source_file.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "compatibility_shim": official.get("compatibility_shim"),
    }


def freeze_protocol(
    *,
    expected_commit: str,
    adatad_pretrain: str,
    adatad_pretrain_sha256: str,
    output_json: str,
) -> dict[str, Any]:
    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite P0 evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ProtocolFreezeFailure("P0 evidence must be outside the worktree")
    git = _bind_git(expected_commit)
    configs, config_evidence = _config_evidence()
    cfg = configs["protected_e2e"]

    pretrain = Path(adatad_pretrain).expanduser().resolve()
    _require(pretrain.is_file(), "VideoMAE-S pretrain is missing")
    actual_pretrain_sha = sha256_file(pretrain)
    _require(
        actual_pretrain_sha == str(adatad_pretrain_sha256).lower(),
        "VideoMAE-S pretrain SHA256 mismatch",
    )

    train_dataset = build_dataset(
        copy.deepcopy(cfg.dataset.train),
        default_args={"logger": None},
    )
    train_loader = build_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        **copy.deepcopy(cfg.solver.train),
    )
    loader_contract = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=train_dataset,
        train_loader=train_loader,
        world_size=1,
    )
    expected_updates = int(loader_contract["loader_length"]) * 60
    p3_config_path = (
        ROOT
        / "configs/adatad/thumos/duca_protected_physical_p3_train_windows.py"
    )
    p3_cfg = Config.fromfile(str(p3_config_path))
    p3_dataset = build_dataset(
        copy.deepcopy(p3_cfg.dataset.train),
        default_args={"logger": None},
    )
    p3_windows = stratified_window_manifest(p3_dataset)
    annotation = Path(cfg.dataset.train.ann_file).expanduser().resolve()
    class_map = Path(cfg.dataset.train.class_map).expanduser().resolve()
    _require(annotation.is_file(), "THUMOS annotation file is missing")
    _require(class_map.is_file(), "THUMOS class map is missing")

    payload = {
        "schema": SCHEMA,
        "ok": True,
        **git,
        "task": "offline_temporal_action_detection",
        "dataset": "THUMOS14",
        "seed": 3407,
        "epochs": 60,
        "train_split": "training",
        "validation_split": None,
        "test_split": "validation_sealed_until_all_terminal_ema_exist",
        "train_loader_contract": loader_contract,
        "expected_successful_optimizer_updates_per_arm": expected_updates,
        "p3_population": {
            "config_path": str(p3_config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(p3_config_path),
            "dataset_length": int(len(p3_dataset)),
            "windows": p3_windows,
            "windows_sha256": canonical_sha256(p3_windows),
            "window_count": len(p3_windows),
            "swaps_per_window": 12,
            "preregistered_swap_count": len(p3_windows) * 12,
        },
        "checkpoint_interval": 5,
        "primary_checkpoint": "epoch_59.pth:state_dict_ema",
        "intermediate_checkpoint_selection": False,
        "configs": config_evidence,
        "data_files": {
            "annotation_path": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "class_map_path": str(class_map),
            "class_map_sha256": sha256_file(class_map),
        },
        "videomae_pretrain": {
            "path": str(pretrain),
            "sha256": actual_pretrain_sha,
        },
        "official_asformer": _official_asformer_evidence(cfg),
        "frozen_method": {
            "dense_window_size": 768,
            "budget": 384,
            "gap_unit": "seconds",
            "gap_cap": "per_sample_exact_uniform_reference",
            "coverage_floor_weight": 0.10,
            "score_temperature": 0.70,
            "path_temperature": 1.00,
            "rho_main": 0.00,
            "rho_ablation": 0.01,
            "action_loss_weight": 1.00,
            "transition_loss_weight": 0.50,
            "transition_boundary_loss_weight": 0.25,
            "detector_bridge_coefficient": 1.00,
        },
        "paper_claim_allowed": False,
    }
    payload["manifest_content_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--adatad-pretrain", required=True)
    parser.add_argument("--adatad-pretrain-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        freeze_protocol(
            expected_commit=args.expected_commit,
            adatad_pretrain=args.adatad_pretrain,
            adatad_pretrain_sha256=args.adatad_pretrain_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
