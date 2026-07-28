from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.duca_rime_training import derive_train_loader_contract


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_clean_commit(repo_root: str | Path, expected_commit: str) -> str:
    root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != str(expected_commit) or status:
        raise RuntimeError("RIME exposure requires the exact clean Git commit")
    return commit


def create_training_exposure(
    *,
    repo_root: str | Path,
    expected_commit: str,
    config: str | Path,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    output: str | Path,
    research_phase: int,
    seed: int,
    detector_backend: str,
    target_mean_cost: float,
) -> dict[str, Any]:
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset

    if int(research_phase) not in {2, 3, 4}:
        raise ValueError("RIME training exposure phase must be 2, 3, or 4")
    if detector_backend not in {"ActionFormer", "TriDet"}:
        raise ValueError("unsupported RIME detector backend")
    if int(research_phase) == 2 and (
        int(seed) != 3407
        or detector_backend != "ActionFormer"
        or float(target_mean_cost) != 384.0
    ):
        raise ValueError(
            "Phase-2 mixed-K exposure is frozen to seed 3407/"
            "ActionFormer/mean K=384"
        )
    if int(research_phase) == 3 and (
        int(seed) != 3407
        or detector_backend != "ActionFormer"
        or float(target_mean_cost) != 384.0
    ):
        raise ValueError("Phase-3 exposure is frozen to seed 3407/ActionFormer/mean K=384")
    if int(research_phase) == 4 and float(target_mean_cost) not in {192.0, 384.0}:
        raise ValueError("Phase-4 exposure requires the registered 192/384 budget panel")

    commit = _exact_clean_commit(repo_root, expected_commit)
    split_validation = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    cfg = Config.fromfile(str(Path(config).resolve()))
    dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )
    contract = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=loader,
        world_size=1,
    )
    expected_videos = set(
        str(value)
        for value in split["train_roles"]["detector_selector_train"]["videos"]
    )
    if set(contract["ordered_video_ids"]) != expected_videos:
        raise RuntimeError("runtime train loader differs from detector_selector_train split")

    mixed_k_schedule = None
    if int(research_phase) == 2:
        from opentad.models.duca.rime import build_cost_matched_mixed_k_cycle

        variant = cfg.duca_rime_variant
        budgets = tuple(int(value) for value in variant.candidate_budgets)
        counts = tuple(int(value) for value in variant.training_schedule_counts)
        schedule_seed = int(variant.training_schedule_seed)
        schedule_source = str(variant.training_schedule_source)
        if (
            budgets != (192, 256, 384, 512)
            or counts != (8, 12, 16, 24)
            or schedule_seed != 3407
            or schedule_source != "stateless_epoch_plus_sample_index"
            or variant.exact_per_video_histogram is not True
            or len(contract["ordered_video_ids"]) != 100
        ):
            raise RuntimeError("Phase-2 mixed-K schedule contract drift")
        cycle = build_cost_matched_mixed_k_cycle(
            budgets,
            counts,
            target_mean_cost=float(target_mean_cost),
            schedule_seed=schedule_seed,
        )
        per_video_histograms = {}
        for sample_index, video_id in enumerate(contract["ordered_video_ids"]):
            values = [
                int(cycle[(epoch + sample_index) % len(cycle)])
                for epoch in range(60)
            ]
            histogram = tuple(values.count(value) for value in budgets)
            if histogram != counts or sum(values) / len(values) != float(
                target_mean_cost
            ):
                raise RuntimeError("Phase-2 per-video mixed-K exposure drift")
            per_video_histograms[str(video_id)] = {
                str(budget): count
                for budget, count in zip(budgets, histogram)
            }
        mixed_k_schedule = {
            "candidate_budgets": list(budgets),
            "per_video_counts": list(counts),
            "target_mean_cost": float(target_mean_cost),
            "schedule_seed": schedule_seed,
            "schedule_source": schedule_source,
            "cycle": cycle,
            "per_video_histograms": per_video_histograms,
        }

    payload = {
        "schema_version": (
            "duca_rime_phase2_mixed_k_training_exposure_v1"
            if int(research_phase) == 2
            else (
                "duca_rime_phase3_training_exposure_v1"
                if int(research_phase) == 3
                else "duca_rime_phase4_training_exposure_v1"
            )
        ),
        "research_phase": int(research_phase),
        "git_commit": commit,
        "seed": int(seed),
        "detector_backend": detector_backend,
        "target_mean_cost": float(target_mean_cost),
        "successful_detector_updates": 6000,
        "split_manifest_path": str(Path(split_manifest).resolve()),
        "split_manifest_sha256": split_validation["manifest_sha256"],
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "train_role": "detector_selector_train",
        "official_final_subset_consumed": False,
        "train_loader_contract": contract,
    }
    if mixed_k_schedule is not None:
        payload["mixed_k_schedule"] = mixed_k_schedule
    target = Path(output).resolve()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different exposure: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "payload": payload,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal the exact DUCA-RIME video/update exposure schedule."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--research-phase",
        type=int,
        choices=(2, 3, 4),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--detector-backend",
        choices=("ActionFormer", "TriDet"),
        required=True,
    )
    parser.add_argument("--target-mean-cost", type=float, required=True)
    args = parser.parse_args(argv)
    result = create_training_exposure(
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        config=args.config,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        output=args.output,
        research_phase=args.research_phase,
        seed=args.seed,
        detector_backend=args.detector_backend,
        target_mean_cost=args.target_mean_cost,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
