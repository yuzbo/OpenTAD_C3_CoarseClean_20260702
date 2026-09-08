"""User-authorized test-guided exploration, separate from sealed-test results."""

import copy
import hashlib
import json
import math
import os
import random
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import _atomic_torch_save


PROTOCOL = "TEST_GUIDED_EXPLORATORY_EVAL5"


def evaluation_due(epoch):
    return (int(epoch) + 1) % 5 == 0


def improves(metrics, best):
    score = float(metrics["average_mAP"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("test average_mAP must be a finite fraction")
    return best is None or score > float(best["metrics"]["average_mAP"])


def write_record(path, payload):
    payload = dict(payload)
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def preserve_training_state(model):
    # Evaluation must not change the next training augmentation/dropout stream.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    flags = [(module, module.training) for module in model.modules()]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in flags:
            module.training = training


class TestGuidedEvaluation:
    def __init__(self, cfg, args, loader):
        self.cfg, self.args, self.loader = cfg, args, loader
        workflow = cfg.workflow
        if args.not_eval or not cfg.solver.ema:
            raise ValueError("test-guided selection requires evaluated EMA weights")
        if workflow.val_eval_interval != 5 or workflow.val_start_epoch != 4:
            raise ValueError("test-guided full evaluation must start at epoch 5, every 5 epochs")
        if workflow.get("val_loss_interval", -1) > 0:
            raise ValueError("test-guided selection is by Avg-mAP, not validation loss")
        if cfg.inference.get("load_from_raw_predictions", False):
            raise ValueError("each periodic evaluation requires fresh inference")
        expected = json.loads(Path(cfg.evaluation.ground_truth_filename).read_text())["database"]
        subset = cfg.evaluation.subset
        self.video_ids = sorted(key for key, value in expected.items() if value["subset"] == subset)
        actual = sorted({str(row[0]) for row in loader.dataset.data_list})
        if actual != self.video_ids or loader.drop_last:
            raise ValueError("test-guided loader must cover the complete official test population")
        if not loader.dataset.test_mode:
            raise ValueError("periodic test requires label-free test_mode inference")
        self.root = Path(cfg.work_dir) / "test_guided"
        repo = Path(__file__).resolve().parents[2]
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
            raise ValueError("test-guided training requires a clean source checkout")
        self.best = None
        best_path = self.root / "best_test_metrics.json"
        if best_path.exists():
            self.best = json.loads(best_path.read_text(encoding="utf-8"))
        if args.rank == 0:
            write_record(self.root / "protocol.json", {
                "protocol": PROTOCOL, "source_sha": self.commit, "seed": args.seed,
                "selection_metric": "average_mAP", "tie_break": "earliest_epoch",
                "model_selection_uses_test": True,
                "test_guided_hyperparameter_tuning_authorized": True,
                "unseen_test_claim_allowed": False, "evaluation_epochs": list(range(5, 61, 5)),
                "test_video_ids": self.video_ids, "test_window_count": len(loader.dataset),
                "terminal_checkpoint": "checkpoint/epoch_59.pth",
            })

    def run(self, epoch, updates, model, model_ema, logger, use_amp, evaluate):
        if not evaluation_due(epoch):
            raise ValueError("periodic full-test evaluation called outside its five-epoch schedule")
        if model_ema is None:
            raise ValueError("missing EMA for test-guided model selection")
        cfg = copy.deepcopy(self.cfg)
        cfg.work_dir = str(self.root / f"epoch_{epoch:02d}")
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        cfg.post_processing.save_dict = True
        cfg.inference.save_raw_prediction = False
        logger.info(f"{PROTOCOL}: full test after epoch {epoch + 1}; test-selected, not unseen-test evidence")
        with preserve_training_state(model):
            metrics = evaluate(
                self.loader, model, cfg, logger, self.args.rank,
                model_ema=model_ema, use_amp=use_amp, world_size=self.args.world_size,
                not_eval=False, max_batches=None, epoch=epoch,
            )
        if self.args.rank == 0:
            if metrics is None:
                raise RuntimeError("full test returned no official metrics")
            better = improves(metrics, self.best)
            record = {
                "protocol": PROTOCOL, "source_sha": self.commit, "seed": self.args.seed,
                "epoch": epoch, "completed_epochs": epoch + 1,
                "successful_updates": int(updates), "checkpoint_state_key": "state_dict_ema",
                "metrics": {key: float(value) for key, value in metrics.items()},
                "test_video_count": len(self.video_ids), "test_window_count": len(self.loader.dataset),
                "model_selection_uses_test": True, "unseen_test_claim_allowed": False,
                "prediction_path": str(Path(cfg.work_dir) / "result_detection.json"),
            }
            write_record(Path(cfg.work_dir) / "metrics.json", record)
            if better:
                best_path = self.root / "best_test.pth"
                _atomic_torch_save({
                    "epoch": epoch, "state_dict_ema": model_ema.module.state_dict(),
                    "experiment_metadata": record,
                }, str(best_path))
                self.best = dict(record, checkpoint_path=str(best_path))
                write_record(self.root / "best_test_metrics.json", self.best)
                logger.info(f"New test-selected best EMA: epoch {epoch + 1}, Avg-mAP={100 * metrics['average_mAP']:.4f}%")
        if dist.is_initialized():
            dist.barrier()
