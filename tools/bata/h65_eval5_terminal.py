"""Independent terminal binding for the frozen H65 eval5 matched controls."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from tools.bata.duca_selected_axis_training import canonical_sha256

TRAINING_COMMIT = "629162cdddbe490d896899a067e9803e237df467"
TRAINING_SOURCE = Path("/data/run01/sczc063/yuzibo/projects/h65_tia_eval5_629162cd")
TRAINING_ROOT = Path("/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal")
PROTOCOL = "TEST_GUIDED_EXPLORATORY_EVAL5"
ARMS = {"TEST-UNIFORM": "uniform", "TEST-PHASEOFF": "phaseoff", "TEST-PHASEON": "phaseon"}


def is_eval5(cfg):
    return cfg.get("test_guided_exploratory", False) and cfg.get("h65_pro_experiment_id") in ARMS


def training_config_sha256(value):
    # Frozen tools/train.py records protocol dataclasses via their string representation.
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def evaluation_binding(cfg, *, config_path, checkpoint_path, evaluator_root, seed, world_size, variant):
    if not is_eval5(cfg) or seed != 3407 or world_size != 1:
        raise ValueError("H65 eval5 requires a registered arm, seed3407 and one GPU")
    arm = ARMS[cfg.h65_pro_experiment_id]
    if variant != f"h65_pro_eval5_{arm}":
        raise ValueError("H65 eval5 variant does not match the config")
    config = TRAINING_SOURCE / f"configs/adatad/thumos/h65_pro/h65_pro_eval5_{arm}.py"
    checkpoint = TRAINING_ROOT / f"{cfg.h65_pro_experiment_id}_seed3407/gpu1_id0/checkpoint/epoch_59.pth"
    if Path(config_path).resolve() != config.resolve() or Path(checkpoint_path).resolve() != checkpoint.resolve():
        raise ValueError("H65 eval5 requires the frozen training config and matching terminal checkpoint")
    if (not cfg.solver.ema or not cfg.solver.amp or cfg.solver.test.batch_size != 2
            or cfg.inference.load_from_raw_predictions or not cfg.post_processing.save_dict):
        raise ValueError("H65 eval5 requires EMA/AMP/global-test-batch2 and fresh saved predictions")
    if (_git(TRAINING_SOURCE, "rev-parse", "HEAD") != TRAINING_COMMIT
            or _git(TRAINING_SOURCE, "status", "--porcelain")
            or _git(evaluator_root, "status", "--porcelain")):
        raise ValueError("H65 training and evaluator checkouts must be exact and clean")
    if _git(evaluator_root, "diff", "--name-only", TRAINING_COMMIT, "HEAD", "--", "opentad", "configs"):
        raise ValueError("H65 evaluator must not change the frozen model or configuration")
    protocol_path = checkpoint.parents[1] / "test_guided/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = protocol.pop("receipt_sha256")
    if digest != canonical_sha256(protocol):
        raise ValueError("H65 training protocol self-hash mismatch")
    if (protocol.get("source_sha") != TRAINING_COMMIT or protocol.get("seed") != 3407
            or protocol.get("protocol") != PROTOCOL
            or protocol.get("model_selection_uses_test") is not True
            or protocol.get("unseen_test_claim_allowed") is not False
            or protocol.get("evaluation_epochs") != list(range(5, 61, 5))):
        raise ValueError("H65 training protocol identity or test-reuse disclosure mismatch")
    ids = sorted(protocol["test_video_ids"])
    if len(ids) != 211 or len(set(ids)) != 211 or protocol["test_window_count"] != 792:
        raise ValueError("H65 eval5 requires the complete 211-video/792-window test set")
    return dict(
        training_git_commit=TRAINING_COMMIT, evaluator_git_commit=_git(evaluator_root, "rev-parse", "HEAD"),
        clean_tree=True, training_source_clean=True, protocol=PROTOCOL,
        evaluation_role="INDEPENDENT_TERMINAL_EMA", model_selection_uses_test=True,
        unseen_test_claim_allowed=False, runtime_gt_input_to_selector=False,
        training_protocol_path=str(protocol_path), training_protocol_sha256=digest,
        evaluated_video_ids=ids, test_window_count=792,
    )


def validate_test_population(dataset, binding):
    ids = sorted({str(row[0]) for row in dataset.data_list})
    if not dataset.test_mode or len(dataset) != 792 or ids != binding["evaluated_video_ids"]:
        raise ValueError("H65 evaluation must use the full label-free test population")


def terminal_counts(checkpoint):
    if checkpoint.get("epoch") != 59 or not checkpoint.get("state_dict_ema"):
        raise ValueError("H65 eval5 requires epoch59 EMA")
    if checkpoint.get("successful_optimizer_updates") != 6000 or checkpoint["scheduler"]["last_epoch"] != 6000:
        raise ValueError("H65 actual global optimizer and scheduler updates must be 6000")
    steps = [int(s["step"]) for s in checkpoint["optimizer"]["state"].values() if "step" in s]
    if not steps or max(steps) != 6000 or min(steps) <= 0:
        raise ValueError("H65 optimizer parameter steps are inconsistent with the terminal budget")
    return dict(
        successful_optimizer_updates=6000, scheduler_updates=6000,
        optimizer_state_count=len(steps), optimizer_steps_distribution=dict(Counter(steps)),
        parameter_participation_note="Conditional parameter steps are separate from the stored global successful-update counter.",
    )
