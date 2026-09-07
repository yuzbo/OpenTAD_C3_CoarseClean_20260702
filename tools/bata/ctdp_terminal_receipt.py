"""Independent terminal evaluation for the frozen CT-DP geometry matrix."""

import hashlib
import json
import subprocess
from pathlib import Path

from tools.bata.ctdp_training import validate_ctdp_progress


TRAINING_COMMIT = "78cde1be1cb8b3acc7d750afc92ea740f2a03d06"
CONFIG_NAMES = {f"duca_ctdp_geometry_g{i}.py" for i in range(4)}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_payload(payload):
    payload = dict(payload)
    payload.pop("receipt_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


def source_identity(root, config_path):
    root = Path(root).resolve()
    config = Path(config_path).resolve()
    if config.name not in CONFIG_NAMES or config.parent != root / "configs/adatad/thumos":
        raise ValueError("CT-DP evaluation requires a frozen G0-G3 source config")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("CT-DP evaluator checkout must be clean")
    if git("diff", "--name-only", TRAINING_COMMIT, "HEAD", "--", "opentad", "configs"):
        raise ValueError("CT-DP model/data source differs from the training commit")
    return dict(training_git_commit=TRAINING_COMMIT,
                evaluator_git_commit=git("rev-parse", "HEAD"), clean_tree=True)


def validate_request(cfg, *, seed, world_size, not_eval, max_batches, precheck, cfg_options):
    if not cfg.get("ctdp_training") or seed != 3407 or world_size != 1:
        raise ValueError("CT-DP evaluation requires seed3407 and one GPU")
    if not cfg.solver.ema or not cfg.solver.amp or cfg.solver.test.batch_size != 2:
        raise ValueError("CT-DP evaluation must preserve EMA, AMP and batch2")
    allowed = {"work_dir", "model.backbone.custom.pretrain"}
    if set(cfg_options or {}) - allowed:
        raise ValueError("CT-DP evaluation cannot override the scientific config")
    if cfg.inference.load_from_raw_predictions or cfg.evaluation.type != "mAP":
        raise ValueError("CT-DP evaluation requires fresh inference and official mAP")
    if precheck:
        if not not_eval or max_batches != 1:
            raise ValueError("CT-DP evaluation precheck is one batch without metrics")
    elif not_eval or max_batches is not None:
        raise ValueError("CT-DP terminal receipt requires complete official evaluation")


def checkpoint_counts(checkpoint, config_path, *, precheck=False):
    epoch, updates, batches = (1, 6, 3) if precheck else (59, 6000, 100)
    meta = checkpoint.get("experiment_metadata", {})
    if (checkpoint.get("epoch") != epoch or not checkpoint.get("state_dict_ema")
            or meta.get("epoch") != epoch or meta.get("checkpoint_state_key") != "state_dict_ema"):
        raise ValueError("CT-DP checkpoint epoch/EMA identity mismatch")
    if (meta.get("route") != "CT-DP" or meta.get("git_commit") != TRAINING_COMMIT
            or meta.get("clean_tree") is not True or meta.get("seed") != 3407
            or Path(meta.get("config_path", "")).name != Path(config_path).name):
        raise ValueError("CT-DP checkpoint source/arm/seed identity mismatch")
    contract = dict(formal=not precheck, epochs=epoch + 1, batches_per_epoch=batches,
                    expected_successful_optimizer_updates=updates)
    if meta.get("contract") != contract:
        raise ValueError("CT-DP checkpoint has the wrong training contract")
    audit = meta["update_audit"]
    validate_ctdp_progress(contract, epoch, audit["successful_optimizer_updates"],
                           audit, checkpoint["scheduler"]["last_epoch"])
    steps = [float(state["step"]) for state in checkpoint.get("optimizer", {}).get("state", {}).values()
             if "step" in state]
    if not steps or set(steps) != {float(updates)}:
        raise ValueError("CT-DP actual optimizer steps disagree with the update budget")
    return dict(checkpoint_epoch=epoch, checkpoint_state_key="state_dict_ema",
                successful_optimizer_updates=updates, scheduler_updates=updates,
                ema_updates=updates, optimizer_state_count=len(steps))


def write_receipt(output, *, cfg, config_path, checkpoint_path, checkpoint, identity, video_ids):
    counts = checkpoint_counts(checkpoint, config_path)
    # The unchanged production evaluator writes this file rather than returning metrics.
    metrics_path = Path(cfg.work_dir) / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("evaluation_epoch") != 59:
        raise ValueError("CT-DP official metrics are not from the terminal evaluation")
    keys = ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")
    values = {key: float(metrics[key]) for key in keys}
    if not all(0 <= value <= 1 for value in values.values()):
        raise ValueError("CT-DP official evaluator returned invalid mAP")
    root = Path(__file__).resolve().parents[2]
    prediction_path = Path(cfg.work_dir) / "result_detection.json"
    payload = dict(
        schema_version="CTDP-TERMINAL-EVAL-v001", **identity, **counts,
        arm=Path(config_path).stem.rsplit("_", 1)[-1].upper(), seed=3407, world_size=1,
        global_batch=2, training_metadata=checkpoint["experiment_metadata"],
        config_path=str(Path(config_path).resolve()), config_sha256=file_hash(config_path),
        resolved_config=cfg.to_dict(), checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=file_hash(checkpoint_path),
        prediction_path=str(prediction_path), prediction_sha256=file_hash(prediction_path),
        annotation_sha256=file_hash(cfg.evaluation.ground_truth_filename),
        class_map_sha256=file_hash(cfg.dataset.test.class_map),
        evaluator_path=str(root / "opentad/evaluations/mAP.py"),
        evaluator_sha256=file_hash(root / "opentad/evaluations/mAP.py"),
        evaluated_video_ids=sorted(set(video_ids)), metrics=values,
    )
    sealed = seal_payload(payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return sealed
