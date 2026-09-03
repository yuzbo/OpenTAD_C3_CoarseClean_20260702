from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    EXPECTED_EVALUATION_VIDEOS,
    EXPECTED_EVALUATION_WINDOWS,
    SEEDS,
    atomic_publish_json,
    canonical_sha256,
    require_clean_commit,
    sha256_file,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_eval import (
    build_prediction_bundle_payload,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_train import (
    REQUIRED_IDENTITY_HASHES,
    validate_full_data_manifest,
)
from tools.bata.zoomtoken_full200_matrix_spec import (
    get_matrix_spec,
    validate_matrix_cell,
)


MATRIX_SPEC = get_matrix_spec()
ARMS = MATRIX_SPEC.arms
PROTOCOL_ID = MATRIX_SPEC.protocol_id


CHECKPOINT_SEAL_SCHEMA = "s2_v3_full200_checkpoint_seal_v1"


def _load_identity_hashes(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != REQUIRED_IDENTITY_HASHES or any(
        not isinstance(value, str) or len(value) != 64 for value in payload.values()
    ):
        raise ValueError("formal identity hash file is incomplete")
    return {str(key): str(value) for key, value in payload.items()}


def build_checkpoint_seal(
    *,
    matrix_path: str | Path,
    population_manifest_sha256: str,
    expected_commit: str,
    output_path: str | Path,
) -> dict[str, Any]:
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    rows = matrix.get("cells")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("checkpoint matrix must contain exactly 9 cells")
    keyed: dict[tuple[str, int], Mapping[str, Any]] = {}
    sealed = []
    for row in rows:
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        if key in keyed or key[0] not in ARMS or key[1] not in SEEDS:
            raise ValueError("checkpoint matrix contains a duplicate or unknown cell")
        keyed[key] = row
    if set(keyed) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("checkpoint matrix is not the frozen 3x3 grid")
    for arm in ARMS:
        for seed in SEEDS:
            row = keyed[(arm, seed)]
            checkpoint = Path(row["checkpoint_path"]).resolve()
            config = Path(row["config_path"]).resolve()
            terminal = Path(row["training_terminal_receipt_path"]).resolve()
            if not checkpoint.is_file() or not config.is_file() or not terminal.is_file():
                raise ValueError("checkpoint cell is missing a terminal artifact")
            terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
            checked_terminal = dict(terminal_payload)
            terminal_digest = checked_terminal.pop("receipt_sha256", None)
            if (
                not isinstance(terminal_digest, str)
                or canonical_sha256(checked_terminal) != terminal_digest
                or terminal_payload.get("complete") is not True
                or terminal_payload.get("protocol_id") != PROTOCOL_ID
                or terminal_payload.get("arm") != arm
                or int(terminal_payload.get("seed", -1)) != seed
                or terminal_payload.get("checkpoint_sha256") != sha256_file(checkpoint)
                or terminal_payload.get("checkpoint_state")
                != "epoch_59_state_dict_ema_update_6000"
            ):
                raise ValueError("checkpoint cell has no valid full-training terminal receipt")
            sealed.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_path": checkpoint.as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_state": "state_dict_ema",
                    "config_path": config.as_posix(),
                    "config_sha256": sha256_file(config),
                    "training_terminal_receipt_path": terminal.as_posix(),
                    "training_terminal_receipt_sha256": sha256_file(terminal),
                }
            )
    seal: dict[str, Any] = {
        "schema_version": CHECKPOINT_SEAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "candidate_commit": expected_commit,
        "population_manifest_sha256": population_manifest_sha256,
        "checkpoint_state": "epoch_59_state_dict_ema_update_6000",
        "rows": sealed,
        "row_count": len(sealed),
    }
    seal["seal_sha256"] = canonical_sha256(seal)
    atomic_publish_json(output_path, seal)
    return seal


def load_checkpoint_seal(
    path: str | Path,
    *,
    expected_commit: str,
    expected_population_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = dict(payload)
    digest = checked.pop("seal_sha256", None)
    if not isinstance(digest, str) or canonical_sha256(checked) != digest:
        raise ValueError("checkpoint seal self-hash mismatch")
    if (
        payload.get("schema_version") != CHECKPOINT_SEAL_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("candidate_commit") != expected_commit
        or payload.get("population_manifest_sha256")
        != expected_population_manifest_sha256
        or int(payload.get("row_count", -1)) != 9
        or len(payload.get("rows", ())) != 9
    ):
        raise ValueError("checkpoint seal identity differs from the frozen matrix")
    return payload


def _cell_from_seal(seal: Mapping[str, Any], *, arm: str, seed: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in seal["rows"]
        if row.get("arm") == arm and int(row.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint seal does not contain exactly one requested cell")
    row = matches[0]
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("config_path", "config_sha256"),
        ("training_terminal_receipt_path", "training_terminal_receipt_sha256"),
    ):
        if sha256_file(row[path_key]) != row[hash_key]:
            raise ValueError(f"sealed cell artifact changed: {path_key}")
    return row


def _load_ema_into_plain_model(
    model: Any,
    checkpoint_path: str | Path,
    *,
    expected_identity_hashes: Mapping[str, str],
) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("identity_hashes") != dict(expected_identity_hashes):
        raise ValueError("sealed checkpoint identity hashes differ from inference")
    state = checkpoint.get("state_dict_ema")
    if not isinstance(state, Mapping):
        raise ValueError("sealed checkpoint has no state_dict_ema")
    keys = tuple(map(str, state))
    if keys and all(key.startswith("module.") for key in keys):
        state = {str(key)[7:]: value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"EMA state does not exactly match the inference model: missing={missing} unexpected={unexpected}"
        )


def _validate_dataset_windows(dataset: Any, manifest: Mapping[str, Any]) -> None:
    expected = manifest["evaluation"]["ordered_windows"]
    if len(dataset.data_list) != EXPECTED_EVALUATION_WINDOWS or len(expected) != len(
        dataset.data_list
    ):
        raise ValueError("runtime inference loader is not the complete 792-window population")
    seen_videos = set()
    for ordinal, (dataset_row, manifest_row) in enumerate(zip(dataset.data_list, expected)):
        video_id = str(dataset_row[0])
        centers = dataset_row[3]
        if (
            int(manifest_row["ordinal"]) != ordinal
            or manifest_row["video_id"] != video_id
            or int(manifest_row["snippet_count"]) != len(centers)
            or int(manifest_row["window_start_frame"]) != int(centers[0])
            or int(manifest_row["window_end_frame"]) != int(centers[-1])
        ):
            raise ValueError("runtime inference window order differs from the sealed manifest")
        seen_videos.add(video_id)
    if seen_videos != set(manifest["evaluation"]["video_order"]):
        raise ValueError("runtime inference does not cover all 211 videos")


def post_nms_with_prediction_uids(
    raw_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    nms_config: Mapping[str, Any],
    video_order: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Run unchanged batched_nms, then deterministically associate retained UIDs."""

    import torch
    from opentad.models.utils.post_processing import batched_nms

    outputs: dict[str, list[dict[str, Any]]] = {}
    for video_id in video_order:
        rows = list(raw_results.get(video_id, ()))
        if not rows:
            outputs[video_id] = []
            continue
        class_labels: list[str] = []
        class_indices = []
        for row in rows:
            label = str(row["label"])
            if label not in class_labels:
                class_labels.append(label)
            class_indices.append(class_labels.index(label))
        segments = torch.tensor([row["segment"] for row in rows], dtype=torch.float32)
        scores = torch.tensor([row["score"] for row in rows], dtype=torch.float32)
        labels = torch.tensor(class_indices)
        kept_segments, kept_scores, kept_labels = batched_nms(
            segments, scores, labels, **dict(nms_config)
        )
        used: set[int] = set()
        retained = []
        for segment, score, label_index in zip(
            kept_segments, kept_scores, kept_labels
        ):
            label = class_labels[int(label_index.item())]
            candidates = [
                index
                for index, row in enumerate(rows)
                if index not in used
                and str(row["label"]) == label
                and torch.equal(segments[index], segment.to(dtype=torch.float32))
            ]
            if not candidates:
                raise RuntimeError("Soft-NMS output cannot be associated with a raw proposal UID")
            selected = min(candidates, key=lambda index: tuple(rows[index]["prediction_uid"]))
            used.add(selected)
            retained.append(
                {
                    "segment": [round(value.item(), 2) for value in segment],
                    "label": label,
                    "score": round(score.item(), 4),
                    "prediction_uid": list(rows[selected]["prediction_uid"]),
                }
            )
        outputs[video_id] = retained
    if set(outputs) != set(map(str, video_order)):
        raise RuntimeError("post-NMS output omits an evaluation video")
    return outputs


def run_label_free_inference(args: argparse.Namespace) -> dict[str, Any]:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("formal label-free inference requires a Slurm allocation")
    import torch
    from mmengine.config import Config

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.utils.post_processing import build_classifier
    from opentad.utils import set_seed, setup_logger

    if torch.cuda.device_count() < 1:
        raise RuntimeError("formal label-free inference requires a Slurm-allocated GPU")
    manifest = validate_full_data_manifest(args.manifest)
    manifest_sha = manifest["manifest_sha256"]
    seal = load_checkpoint_seal(
        args.checkpoint_seal,
        expected_commit=args.expected_commit,
        expected_population_manifest_sha256=manifest_sha,
    )
    cell = _cell_from_seal(seal, arm=args.arm, seed=args.seed)
    identity_hashes = _load_identity_hashes(args.identity_hashes)
    if identity_hashes["config_sha256"] != sha256_file(cell["config_path"]):
        raise ValueError("identity hash file does not bind the inference config")
    require_clean_commit(args.expected_commit, Path(__file__).resolve().parents[2])
    validate_matrix_cell(
        cell["config_path"], arm=args.arm, seed=args.seed, spec=MATRIX_SPEC
    )
    cfg = Config.fromfile(cell["config_path"])
    cfg.dataset.test.ann_file = manifest["evaluation"]["heldout_inference_annotation"]
    cfg.dataset.test.class_map = manifest["class_map"]["path"]
    cfg.dataset.test.data_path = manifest["media"]["root"]
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.inference.save_raw_prediction = False
    cfg.inference.load_from_raw_predictions = False
    set_seed(args.seed, False, deterministic_warn_only=True)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(
        f"{MATRIX_SPEC.key}LabelFreeInfer",
        save_dir=cfg.work_dir,
        distributed_rank=0,
    )
    dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    _validate_dataset_windows(dataset, manifest)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )
    model = build_detector(cfg.model).cuda().eval()
    backbone_module = getattr(model, "backbone", None)
    if hasattr(backbone_module, "fusion"):
        fusion_params = sum(p.numel() for p in backbone_module.fusion.parameters())
        if fusion_params != 0:
            raise RuntimeError(f"Inference audit failure: fusion has {fusion_params} parameters, expected 0")
        if getattr(backbone_module, "fusion_mode", None) != "fixed_mean":
            raise RuntimeError(f"Inference audit failure: fusion_mode is {backbone_module.fusion_mode}, expected fixed_mean")
    _load_ema_into_plain_model(
        model,
        cell["checkpoint_path"],
        expected_identity_hashes=identity_hashes,
    )
    post_config = copy.deepcopy(cfg.post_processing)
    post_config.sliding_window = True
    external_classifier = (
        build_classifier(post_config.external_cls)
        if post_config.get("external_cls", None) is not None
        else dataset.class_map
    )
    inference_config = copy.deepcopy(cfg.inference)
    inference_config.folder = str(args.work_dir / "internal_outputs_disabled")
    raw: dict[str, list[dict[str, Any]]] = {
        video_id: [] for video_id in manifest["evaluation"]["video_order"]
    }
    for dataset_index, data in enumerate(loader):
        window_ordinal = int(
            manifest["evaluation"]["ordered_windows"][dataset_index]["ordinal"]
        )
        with torch.inference_mode(), torch.cuda.amp.autocast(
            dtype=torch.float16, enabled=bool(cfg.solver.amp)
        ):
            window_results = model(
                **data,
                return_loss=False,
                infer_cfg=inference_config,
                post_cfg=post_config,
                ext_cls=external_classifier,
            )
        for video_id, rows in window_results.items():
            if video_id not in raw:
                raise ValueError("model emitted an out-of-population video")
            for proposal_ordinal, row in enumerate(rows):
                label = str(row["label"])
                if label not in manifest["class_map"]["classes"]:
                    raise ValueError("model emitted a label outside the frozen class map")
                class_index = manifest["class_map"]["classes"].index(label)
                raw[video_id].append(
                    {
                        **dict(row),
                        "prediction_uid": [
                            video_id,
                            window_ordinal,
                            proposal_ordinal,
                            class_index,
                        ],
                    }
                )
    results = post_nms_with_prediction_uids(
        raw,
        nms_config=post_config.nms,
        video_order=manifest["evaluation"]["video_order"],
    )
    payload = build_prediction_bundle_payload(
        arm=args.arm,
        seed=args.seed,
        population_manifest_sha256=manifest_sha,
        video_order=manifest["evaluation"]["video_order"],
        results=results,
        class_map=manifest["class_map"]["classes"],
    )
    atomic_publish_json(args.output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal checkpoints or run label-free inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-checkpoints")
    seal.add_argument("--matrix", type=Path, required=True)
    seal.add_argument("--population-manifest-sha256", required=True)
    seal.add_argument("--expected-commit", required=True)
    seal.add_argument("--output", type=Path, required=True)
    infer = subparsers.add_parser("infer-cell")
    infer.add_argument("--arm", choices=ARMS, required=True)
    infer.add_argument("--seed", choices=SEEDS, type=int, required=True)
    infer.add_argument("--expected-commit", required=True)
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--checkpoint-seal", type=Path, required=True)
    infer.add_argument("--identity-hashes", type=Path, required=True)
    infer.add_argument("--work-dir", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "seal-checkpoints":
        payload = build_checkpoint_seal(
            matrix_path=args.matrix,
            population_manifest_sha256=args.population_manifest_sha256,
            expected_commit=args.expected_commit,
            output_path=args.output,
        )
    else:
        payload = run_label_free_inference(args)
    print(
        json.dumps(
            {"status": "PASS", "command": args.command, "artifact_sha256": canonical_sha256(payload)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SEAL_SCHEMA",
    "build_checkpoint_seal",
    "load_checkpoint_seal",
    "post_nms_with_prediction_uids",
    "run_label_free_inference",
]
