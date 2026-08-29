from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def _selector_state_dict(state_dict):
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    selected = {
        key[len("frame_selector.") :]: value
        for key, value in normalized.items()
        if key.startswith("frame_selector.")
    }
    if not selected:
        raise ValueError("checkpoint has no frame_selector.* parameters")
    return selected


def build_table(args):
    import torch
    from mmengine.config import Config

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models.builder import build_selector
    from opentad.models.duca.tubelet_coreset import (
        aggregate_frame_signals_to_tubelets,
        assign_dynamic_native_tubelet_clip_budgets,
        task_state_tubelet_scores,
    )

    cfg = Config.fromfile(str(Path(args.config).resolve()))
    selector_cfg = Config.fromfile(str(Path(args.selector_config or args.config).resolve()))
    dataset = build_dataset(cfg.dataset[args.split], default_args=dict(logger=None))
    loader_cfg = dict(cfg.solver[args.split])
    loader_cfg["batch_size"] = int(args.batch_size)
    loader_cfg["num_workers"] = int(args.num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    selector = build_selector(selector_cfg.model.frame_selector)
    checkpoint = torch.load(str(Path(args.checkpoint).resolve()), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-1 checkpoint must be a mapping")
    state = checkpoint.get("state_dict_ema")
    if not isinstance(state, Mapping) or int(checkpoint.get("epoch", -1)) != 29:
        raise ValueError("dynamic budget tables require Stage-1 epoch-29 state_dict_ema")
    incompatible = selector.load_state_dict(_selector_state_dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-1 selector state does not match the frozen selector")
    device = torch.device(args.device)
    selector = selector.to(device).eval()
    selector.raw_actionness_source.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            masks = batch["masks"].to(device=device, dtype=torch.bool, non_blocking=True)
            metas = [dict(meta) for meta in batch["metas"]]
            scout = selector.raw_actionness_source(inputs, valid_mask=masks)
            hidden = scout.get("coarse_hidden_features", scout.get("hidden_features"))
            if hidden is None:
                raise ValueError("frozen H65 scout did not return hidden features")
            evidence = aggregate_frame_signals_to_tubelets(
                scout["p_action"].detach(),
                scout["transition_score"].detach(),
                hidden.detach(),
                masks,
            )
            scored = task_state_tubelet_scores(
                evidence["actionness"],
                evidence["boundary"],
                evidence["hidden"],
                evidence["valid_mask"],
            )
            for index, meta in enumerate(metas):
                valid_len = int(evidence["valid_mask"][index].long().sum().item())
                if valid_len <= 0:
                    raise ValueError("window has no valid native tubelet")
                rows.append(
                    {
                        "video_name": str(meta["video_name"]),
                        "window_start_frame": int(meta["window_start_frame"]),
                        "mean_actionness": float(
                            evidence["actionness"][index, :valid_len].double().mean().item()
                        ),
                        "p90_boundary": float(
                            torch.quantile(
                                evidence["boundary"][index, :valid_len].double(),
                                0.9,
                                interpolation="linear",
                            ).item()
                        ),
                        "p90_novelty": float(
                            torch.quantile(
                                scored["novelty"][index, :valid_len].double(),
                                0.9,
                                interpolation="linear",
                            ).item()
                        ),
                        "valid_native_tubelets": valid_len,
                    }
                )
    assigned = assign_dynamic_native_tubelet_clip_budgets(rows)
    for row in assigned:
        required_tubelets = int(row["clip_budget"]) * 8
        if int(row["valid_native_tubelets"]) < required_tubelets:
            raise ValueError(
                "assigned dynamic clip budget exceeds a window's valid physical tubelet grid: "
                f"video={row['video_name']} start={row['window_start_frame']} "
                f"valid={row['valid_native_tubelets']} required={required_tubelets}"
            )
    output = {
        "schema_version": "duca_dynamic_native_tubelet_budget_v1",
        "split": str(args.split),
        "checkpoint_epoch": 29,
        "checkpoint_state_key": "state_dict_ema",
        "decision_inputs": ["mean_actionness", "p90_boundary", "p90_novelty"],
        "uses_gt": False,
        "uses_teacher": False,
        "uses_detector_predictions": False,
        "rows": assigned,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selector-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    build_table(args)


if __name__ == "__main__":
    main()
