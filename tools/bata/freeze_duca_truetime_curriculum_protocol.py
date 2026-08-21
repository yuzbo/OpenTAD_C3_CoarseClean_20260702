from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from opentad.datasets import build_dataloader, build_dataset
from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    derive_train_loader_contract,
    sha256_file,
)
from tools.bata.freeze_duca_protected_physical_protocol import (
    _official_asformer_evidence,
)


SCHEMA = "duca_protected_physical_protocol_manifest_v1"
SELECTOR_ARM = "protected_e2e_homotopy025"
CONFIGS = {
    "RANKPACK_K384": "configs/adatad/thumos/duca_rankpack_k384_curriculum.py",
    "TRUETIME_K384": "configs/adatad/thumos/duca_truetime_k384_curriculum.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"TrueTime protocol freeze failed: {message}")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def freeze(
    *,
    route_arm: str,
    expected_commit: str,
    adatad_pretrain: str,
    adatad_pretrain_sha256: str,
    output_json: str,
) -> dict:
    _require(route_arm in CONFIGS, f"unknown route arm {route_arm!r}")
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "bad commit")
    _require(_git("rev-parse", "HEAD") == expected_commit, "commit drift")
    _require(not _git("status", "--porcelain", "--untracked-files=normal"), "dirty tree")

    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite protocol evidence")
    try:
        output.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("protocol evidence must live outside the worktree")

    relative_config = CONFIGS[route_arm]
    cfg = Config.fromfile(str(ROOT / relative_config))
    _require(str(cfg.arm) == route_arm, "route-arm declaration drift")
    _require(str(cfg.model.frame_selector.arm) == SELECTOR_ARM, "selector arm drift")
    _require(int(cfg.model.frame_selector.homotopy_total_steps) == 6000, "total steps drift")
    _require(int(cfg.model.frame_selector.homotopy_warmup_steps) == 2000, "warmup drift")
    _require(int(cfg.model.frame_selector.homotopy_transition_steps) == 2000, "transition drift")
    _require(cfg.dataset.val is None, "validation must stay sealed during training")

    pretrain = Path(adatad_pretrain).expanduser().resolve()
    _require(pretrain.is_file(), "VideoMAE-S pretrain is missing")
    actual_pretrain_sha256 = sha256_file(pretrain)
    _require(actual_pretrain_sha256 == adatad_pretrain_sha256.lower(), "pretrain hash drift")

    train_dataset = build_dataset(copy.deepcopy(cfg.dataset.train), default_args={"logger": None})
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
    _require(expected_updates == 6000, "real loader exposure is not 100 batches/epoch")

    annotation = Path(cfg.dataset.train.ann_file).expanduser().resolve()
    class_map = Path(cfg.dataset.train.class_map).expanduser().resolve()
    _require(annotation.is_file(), "THUMOS14 annotation is missing")
    _require(class_map.is_file(), "THUMOS14 class map is missing")
    source_sha = sha256_file(ROOT / relative_config)
    resolved_sha = canonical_sha256(cfg.to_dict())
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "git_commit": expected_commit,
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "task": "offline_temporal_action_detection",
        "route": "DUCA_TRUE_TIME_INDIRECT_CURRICULUM",
        "route_arm": route_arm,
        "seed": 3407,
        "epochs": 60,
        "train_split": "training",
        "validation_split": None,
        "test_split": "validation_sealed_until_terminal_ema",
        "train_loader_contract": loader_contract,
        "expected_successful_optimizer_updates_per_arm": expected_updates,
        "checkpoint_interval": 5,
        "primary_checkpoint": "epoch_59.pth:state_dict_ema",
        "intermediate_checkpoint_selection": False,
        "configs": {
            "arms": {
                route_arm: {
                    "path": relative_config,
                    "selector_arm": SELECTOR_ARM,
                    "route_arm": route_arm,
                    "source_sha256": source_sha,
                    "resolved_sha256": resolved_sha,
                    "homotopy_total_steps": 6000,
                    "homotopy_warmup_steps": 2000,
                    "homotopy_transition_steps": 2000,
                }
            }
        },
        "data_files": {
            "annotation_path": str(annotation),
            "annotation_sha256": sha256_file(annotation),
            "class_map_path": str(class_map),
            "class_map_sha256": sha256_file(class_map),
        },
        "videomae_pretrain": {
            "path": str(pretrain),
            "sha256": actual_pretrain_sha256,
        },
        "official_asformer": _official_asformer_evidence(cfg),
        "frozen_method": {
            "selected_budget": 384,
            "dense_window_size": 768,
            "selector": "train_only_ASFormer_actionness_boundary_indirect",
            "curriculum_successful_updates": [2000, 2000, 2000],
            "transition_shape": "cosine",
            "rankpack_or_truetime_only_difference": route_arm,
        },
        "paper_claim_allowed": False,
    }
    payload["manifest_content_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-arm", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--adatad-pretrain", required=True)
    parser.add_argument("--adatad-pretrain-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        freeze(
            route_arm=args.route_arm,
            expected_commit=args.expected_commit,
            adatad_pretrain=args.adatad_pretrain,
            adatad_pretrain_sha256=args.adatad_pretrain_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
