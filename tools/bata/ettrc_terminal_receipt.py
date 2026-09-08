"""Terminal receipts for the frozen ET-TRC OFF/ON pair."""

import hashlib
import json
import subprocess
from pathlib import Path


TRAINING_COMMIT = "9a346f0d71e8870ad499bb0a58b9f1824b8904c0"
TRAINING_ROOT = Path("/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal")
PROTOCOL = "TEST_GUIDED_EXPLORATORY_EVAL5"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def terminal_counts(checkpoint):
    if checkpoint.get("epoch") != 59 or not checkpoint.get("state_dict_ema"):
        raise ValueError("ET-TRC requires an epoch-59 EMA checkpoint")
    steps = [float(state["step"]) for state in checkpoint.get("optimizer", {}).get("state", {}).values()
             if "step" in state]
    if not steps or set(steps) != {6000.0}:
        raise ValueError("ET-TRC optimizer states must all have 6000 successful updates")
    if checkpoint.get("scheduler", {}).get("last_epoch") != 6000:
        raise ValueError("ET-TRC scheduler must have 6000 updates")
    return dict(checkpoint_epoch=59, checkpoint_state_key="state_dict_ema",
                successful_optimizer_updates=6000, scheduler_updates=6000,
                optimizer_state_count=len(steps))


def validate_request(cfg, *, seed, world_size, not_eval, max_batches):
    if cfg.model.backbone.backbone.type != "ETTRCVisionTransformerAdapter":
        raise ValueError("ET-TRC receipt requires the ET-TRC backbone")
    if seed != 4407 or world_size != 2 or cfg.solver.test.batch_size != 2:
        raise ValueError("ET-TRC receipt requires seed4407/world2/global-batch2")
    if not_eval or max_batches is not None or cfg.inference.load_from_raw_predictions:
        raise ValueError("ET-TRC official receipt requires complete fresh inference")
    if not cfg.solver.ema or cfg.solver.amp:
        raise ValueError("ET-TRC frozen pair uses EMA and FP32")
    if cfg.evaluation.type != "mAP" or not cfg.post_processing.save_dict:
        raise ValueError("ET-TRC receipt requires official mAP and saved predictions")
    if not cfg.get("test_guided_exploratory", False):
        raise ValueError("ET-TRC terminal evaluation requires the disclosed eval5 config")


def training_binding(cfg, checkpoint_path, *, seed):
    arm = "on" if cfg.model.backbone.backbone.enable_taylor else "off"
    expected = TRAINING_ROOT / arm / "gpu2_id0/checkpoint/epoch_59.pth"
    if Path(checkpoint_path).resolve() != expected.resolve():
        raise ValueError("ET-TRC checkpoint must match the frozen training arm and run")
    protocol_path = expected.parents[1] / "test_guided/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = protocol.pop("receipt_sha256")
    if digest != canonical_hash(protocol):
        raise ValueError("ET-TRC training protocol self-hash mismatch")
    if protocol.get("source_sha") != TRAINING_COMMIT or protocol.get("seed") != seed:
        raise ValueError("ET-TRC training protocol source/seed mismatch")
    if (protocol.get("protocol") != PROTOCOL
            or protocol.get("model_selection_uses_test") is not True
            or protocol.get("unseen_test_claim_allowed") is not False
            or protocol.get("evaluation_epochs") != list(range(5, 61, 5))):
        raise ValueError("ET-TRC training protocol must disclose five-epoch test selection")
    video_ids = sorted(protocol["test_video_ids"])
    if len(set(video_ids)) != 211 or protocol["test_window_count"] != 792:
        raise ValueError("ET-TRC training protocol must cover the full test population")
    return dict(training_protocol_path=str(protocol_path), training_protocol_sha256=digest,
                evaluated_video_ids=video_ids, test_window_count=protocol["test_window_count"])


def validate_test_population(dataset, binding):
    actual_ids = sorted({str(row[0]) for row in dataset.data_list})
    if (not dataset.test_mode or actual_ids != binding["evaluated_video_ids"]
            or len(dataset) != binding["test_window_count"]):
        raise ValueError("ET-TRC evaluation must match the complete label-free training test population")


def source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("ET-TRC evaluator checkout must be clean")
    if git("diff", "--name-only", TRAINING_COMMIT, "HEAD", "--", "opentad", "configs"):
        raise ValueError("ET-TRC model/data configuration differs from the frozen training commit")
    return dict(training_git_commit=TRAINING_COMMIT, evaluator_git_commit=git("rev-parse", "HEAD"), clean_tree=True)


def seal_payload(payload):
    payload = dict(payload)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = canonical_hash(payload)
    return payload


def write_receipt(output, *, cfg, config_path, checkpoint_path, checkpoint,
                  metrics, identity, seed, world_size):
    counts = terminal_counts(checkpoint)
    required = ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")
    values = {key: float(metrics[key]) for key in required}
    if not all(0 <= value <= 1 for value in values.values()):
        raise ValueError("Official evaluator returned invalid mAP")
    prediction_path = Path(cfg.work_dir) / "result_detection.json"
    evaluator_path = Path(__file__).resolve().parents[2] / "opentad/evaluations/mAP.py"
    payload = dict(
        schema_version="ETTRC-TERMINAL-EVAL-v001", **identity, **counts,
        protocol=PROTOCOL, evaluation_role="INDEPENDENT_TERMINAL_EMA",
        model_selection_uses_test=True, unseen_test_claim_allowed=False,
        ema_update_count_stored=False,
        ema_count_evidence="FP32 successful-update training control flow; no standalone EMA counter",
        arm="ON" if cfg.model.backbone.backbone.enable_taylor else "OFF",
        seed=seed, world_size=world_size, global_batch=cfg.solver.test.batch_size,
        config_path=str(Path(config_path).resolve()), config_sha256=file_hash(config_path),
        resolved_config=cfg.to_dict(),
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=file_hash(checkpoint_path),
        prediction_path=str(prediction_path), prediction_sha256=file_hash(prediction_path),
        evaluation_config=cfg.evaluation.to_dict(),
        annotation_sha256=file_hash(cfg.evaluation.ground_truth_filename),
        class_map_sha256=file_hash(cfg.dataset.test.class_map),
        evaluator_path=str(evaluator_path), evaluator_sha256=file_hash(evaluator_path),
        metrics=values,
    )
    sealed = seal_payload(payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, allow_nan=False)
        stream.write("\n")
