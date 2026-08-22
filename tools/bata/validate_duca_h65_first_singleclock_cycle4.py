"""Fail-closed, read-only admission checks for H65 matched Cycle4."""
import argparse, hashlib, os
from pathlib import Path
import torch
from opentad.models.duca.structured_selection import exact_uniform_positions
from mmengine.config import Config
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"STAGE1": ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py", "STAGE2_OFF": ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py", "STAGE2_ON": ROOT / "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"}
def _get(o, k, d=None): return o.get(k, d) if hasattr(o, "get") else getattr(o, k, d)
def global_rank_clip_slices(batch, clips=24, clip_len=16, tubelet=2):
    pos = exact_uniform_positions(clips * clip_len * tubelet, clips * clip_len)
    return pos.repeat(batch, 1).reshape(batch, clips, -1).reshape(batch * clips, -1)
def main():
    p = argparse.ArgumentParser(); p.add_argument("--target", choices=CONFIGS, required=True); p.add_argument("--video-root", type=Path, required=True); p.add_argument("--annotation", type=Path, required=True); p.add_argument("--category", type=Path, required=True); p.add_argument("--pretrain", type=Path, required=True); p.add_argument("--stage1", type=Path); p.add_argument("--sha256"); p.add_argument("--epoch", type=int); p.add_argument("--chunk-dim", type=int, default=0); a = p.parse_args()
    for path in (a.video_root, a.annotation, a.category, a.pretrain):
        if not path.is_file() and path != a.video_root: raise SystemExit(f"canonical resource unreadable: {path}")
    entries = list(a.video_root.iterdir())
    if len(entries) != 411 or any(not x.is_symlink() or not x.exists() for x in entries): raise SystemExit("canonical video root must contain 411 valid symlinks")
    if a.target != "STAGE1":
        if not a.stage1 or not a.sha256 or a.epoch != 29 or len(a.sha256) != 64: raise SystemExit("Stage2 requires epoch29 Stage1 checkpoint and SHA")
        actual = hashlib.sha256(a.stage1.read_bytes()).hexdigest()
        if actual != a.sha256.lower(): raise SystemExit("Stage1 checkpoint sha256 mismatch")
        meta = torch.load(a.stage1, map_location="cpu", weights_only=False)
        if not isinstance(meta, dict) or meta.get("epoch") != 29 or not isinstance(meta.get("state_dict_ema"), dict): raise SystemExit("Stage1 checkpoint must contain epoch=29 state_dict_ema")
        os.environ.update(DUCA_STAGE1_CHECKPOINT=str(a.stage1.resolve()), DUCA_STAGE1_CHECKPOINT_SHA256=actual, DUCA_STAGE1_CHECKPOINT_EPOCH="29")
    cfg = Config.fromfile(str(CONFIGS[a.target]))
    if a.target != "STAGE1" and (_get(cfg, "seed"), _get(cfg, "total_epochs"), _get(cfg, "max_updates")) != (3407, 60, 6000): raise SystemExit("training contract failed")
    if a.target == "STAGE2_ON" and _get(_get(cfg, "model", {}), "single_clock_admission") is not True: raise SystemExit("ON admission missing")
    if a.chunk_dim != 0 or global_rank_clip_slices(2).shape != (48, 16): raise SystemExit("chunk/global helper contract failed")
    print(f"PASS H65 Cycle4 {a.target}: resources=411 valid_symlinks chunk_dim=0")
if __name__ == "__main__": main()
