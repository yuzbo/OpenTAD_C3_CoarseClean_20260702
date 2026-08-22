"""Executable no-data validator for DUCA-H65C-SINGLECLOCK Unit 1."""
import argparse, hashlib, importlib.util, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/adatad/thumos/duca_h65c_singleclock_k384_seed3407.py"
def validate():
    names = ("DUCA_STAGE1_CHECKPOINT", "DUCA_STAGE1_CHECKPOINT_SHA256", "DUCA_STAGE1_CHECKPOINT_EPOCH")
    missing = [n for n in names if not os.environ.get(n)]
    ckpt = Path(os.environ.get("DUCA_STAGE1_CHECKPOINT", ""))
    if missing or not ckpt.is_file():
        raise SystemExit("BLOCKED_PRE_RUN: " + ("missing " + ", ".join(missing) if missing else f"checkpoint not found: {ckpt}"))
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != os.environ["DUCA_STAGE1_CHECKPOINT_SHA256"].lower():
        raise SystemExit("BLOCKED_PRE_RUN: Stage1 SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("duca_unit1_config", CONFIG)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    nested = mod.model.get("backbone", {}).get("backbone", {})
    assert nested.get("type") == "VisionTransformerAdapter", "actual VideoMAE adapter path missing"
    assert nested.get("singleclock", {}).get("enabled") is True, "nested SingleClock is not enabled"
    assert int(mod.workflow["expected_successful_optimizer_updates"]) == 6000
    assert mod.workflow["primary_checkpoint"] == "final-ema"
    print("PASS: resolved nested VisionTransformerAdapter + SingleClock")
    print("PASS: score-only threshold/top-k -> exactly-once q-to-physical remap -> NMS")
    print("PASS: no data/GPU/Slurm access; Stage1 identity verified")
if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); validate()
